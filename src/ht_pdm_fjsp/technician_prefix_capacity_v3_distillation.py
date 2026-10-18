"""Frozen N4 PPO teachers and shared, training-only distillation phase."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_balanced_algorithms as learner
from ht_pdm_fjsp.maintenance_opportunity_cost import Journal,write_json
from ht_pdm_fjsp.technician_prefix_capacity_v3_env import VERSION,ResourceEnv,scaled_config

SOURCE_PROTOCOL='maintenance_balanced_algorithm_benchmark_v1'
DEFAULT_TEACHERS=Path('/home/bachng/ht-pdm-fjsp/artifacts/maintenance_balanced_algorithm_benchmark_v1_full_cpu_20261012T115140Z')
GRAPH_KEYS=('machines','technicians','edges','global_features','pair_mask','state','joint_mask')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_teachers(source,out,cfg):
    source=Path(source);out=Path(out)/'teachers';out.mkdir()
    manifest=json.loads((source/'manifest.json').read_text());audit=json.loads((source/'audit.json').read_text())
    if manifest['status']!='COMPLETED' or audit['status']!='PASS':raise RuntimeError('unverified teacher source')
    choices=json.loads((source/'checkpoint_choices.json').read_text());result={}
    bindings=manifest.get('output_bindings',{})
    for name in ('manifest.json','audit.json','checkpoint_choices.json','resolved_config.json'):
        if name in bindings and sha(source/name)!=bindings[name]['sha256']:raise RuntimeError('source binding mismatch')
        shutil.copy2(source/name,out/name)
    original_cfg=json.loads((out/'resolved_config.json').read_text())
    if original_cfg['env_steps']!=491520 or len(original_cfg['train_seeds'])!=5:raise RuntimeError('foundation budget mismatch')
    for index,t in enumerate(cfg['train_seeds']):
        old=46000000+index
        for a in ('ppo','mappo'):
            c=choices[f'{a}:{old}'];p=source/c['path']
            if sha(p)!=c['sha256']:raise RuntimeError('source teacher checkpoint changed')
            target=out/f'{a}_{old}.pt';shutil.copy2(p,target)
            meta=dict(c,path=str(target.relative_to(out.parent)),original_train_seed=old,bundle_train_seed=t,
                algorithm=a,foundation_training_steps=491520)
            load_reference(out.parent,meta)
            result[f'{a}:{t}']=meta
    report=dict(source=str(source),source_commit=manifest['git']['commit'],references=result,
        source_hashes={name:sha(out/name) for name in ('manifest.json','audit.json','checkpoint_choices.json','resolved_config.json')},
        budget_disclosure='each graph arm: shared N4 bank122880 plus RL491520 per scale; foundation PPO491520 already trained; no sample-efficiency claim')
    write_json(out/'references.json',report);return report


def load_reference(out,meta):
    path=Path(out)/meta['path']
    if sha(path)!=meta['sha256']:raise RuntimeError('frozen teacher hash')
    p=torch.load(path,map_location='cpu',weights_only=True)
    expected=dict(protocol=SOURCE_PROTOCOL,algorithm=meta['algorithm'],train_seed=meta['original_train_seed'],steps=meta['steps'],hidden=64,mixer_hidden=32)
    if any(p.get(k)!=v for k,v in expected.items()) or p['config']!=asdict(scaled_config(4)):raise RuntimeError('teacher metadata/physics mismatch')
    model=learner.Model(meta['algorithm'],64);model.load_state_dict(p['state_dict']);model.eval()
    if learner.parameter_digest(model)!=p['parameter_sha256'] or p['parameter_sha256']!=meta['parameter_sha256']:raise RuntimeError('teacher parameter digest')
    return model


def matching_tokens(indices,actions):
    result=np.zeros((len(indices),2),dtype=np.int64)
    for row,index in enumerate(indices):
        for m,j in actions[int(index)]:result[row,j]=m+1
    return result


def build_bank(cfg,seed,out,teacher,timeout):
    out=Path(out);workers=cfg['vector_envs'];total=cfg['teacher_bank_steps']
    if total%(48*workers):raise ValueError('teacher bank must contain complete episodes')
    envs=[ResourceEnv(seed,4,'ppo',1000+w,True) for w in range(workers)]
    observations=[env.reset() for env in envs];data={key:[] for key in GRAPH_KEYS};data.update(labels=[],behavior=[],reward=[],done=[])
    rng=torch.Generator().manual_seed(learner.role_seed(seed,VERSION+':teacher_actions'))
    journal=Journal(out/'bank_episodes.csv');rewards=np.zeros(workers);episodes=0
    try:
        for vector_tick in tqdm(range(total//workers),desc=f'Teacher bank {seed}',unit='vector-step'):
            timeout();batch=learner.tensors(observations)
            with torch.no_grad():
                logits=teacher.policy(batch);label=logits.argmax(-1).numpy()
                behavior=torch.multinomial(torch.softmax(logits/cfg['teacher_temperature'],-1),1,generator=rng).squeeze(1).numpy()
            for key in GRAPH_KEYS:data[key].append(np.stack([obs[key] for obs in observations]))
            data['labels'].append(matching_tokens(label,envs[0].actions));data['behavior'].append(behavior)
            following=[];tick_rewards=[];dones=[]
            for w,env in enumerate(envs):
                obs,r,done,info=env.step(int(behavior[w]));rewards[w]+=r;tick_rewards.append(r);dones.append(done)
                if done:
                    metrics=info['episode_metrics']
                    if abs(-20*rewards[w]-metrics['extended_cost'])>1e-5:raise RuntimeError('teacher bank reward reconciliation')
                    journal.add(dict(train_seed=seed,worker=1000+w,episode_reward=float(rewards[w]),**metrics));episodes+=1;rewards[w]=0.
                    # Avoid constructing an unused episode at the final tick.
                    if vector_tick+1<total//workers:obs=env.reset()
                following.append(obs)
            observations=following;data['reward'].append(np.asarray(tick_rewards,dtype=np.float32));data['done'].append(np.asarray(dones,bool))
    finally:journal.close()
    returns=learner.mc_returns(torch.from_numpy(np.stack(data['reward'])),torch.from_numpy(np.stack(data['done']))).numpy()
    arrays={key:np.concatenate(value,axis=0) for key,value in data.items()};arrays['returns']=returns.reshape(-1)
    np.savez_compressed(out/'bank.npz',**arrays)
    audit=audit_bank(arrays,teacher,envs[0].actions,cfg)
    audit.update(train_seed=seed,bank_sha256=sha(out/'bank.npz'),episodes=episodes,
        physical_ticks=episodes*60,behavior_temperature=cfg['teacher_temperature'])
    write_json(out/'bank_audit.json',audit);return arrays,audit


def audit_bank(data,teacher,actions,cfg):
    total=cfg['teacher_bank_steps'];workers=cfg['vector_envs']
    if any(len(v)!=total for v in data.values()) or data['labels'].shape!=(total,2):raise RuntimeError('bank shape/sample budget')
    shaped={k:data[k].reshape(total//workers,workers,*data[k].shape[1:]) for k in ('reward','done')}
    if not np.all(shaped['done'][47::48]) or np.count_nonzero(data['done'])!=total//48:raise RuntimeError('teacher bank terminal grid')
    expected=learner.mc_returns(torch.from_numpy(shaped['reward']),torch.from_numpy(shaped['done'])).numpy().reshape(-1)
    if not np.array_equal(expected,data['returns']):raise RuntimeError('teacher bank return audit')
    for start in range(0,total,512):
        batch=learner.tensors({k:data[k][start:start+512] for k in ('state','joint_mask')})
        with torch.no_grad():labels=teacher.policy(batch).argmax(-1).numpy()
        if not np.array_equal(matching_tokens(labels,actions),data['labels'][start:start+512]):raise RuntimeError('teacher labels changed')
    for j in range(2):
        token=data['labels'][:,j];live=token>0
        if not data['pair_mask'][np.arange(total)[live],token[live]-1,j].all():raise RuntimeError('illegal teacher label')
    if np.any((data['labels'][:,0]>0)&(data['labels'][:,0]==data['labels'][:,1])):raise RuntimeError('duplicate teacher target')
    return dict(status='PASS',state_exposures=total,labels_replayed=total)


def rename_presentation(batch,labels,generator):
    """Bijective N4 machine/technician relabeling; no new teacher queries."""
    b,n,k,_=batch['edges'].shape
    if (n,k)!=(4,2) or labels.shape!=(b,k):raise ValueError('N4 teacher presentation required')
    machines=torch.rand((b,n),generator=generator).argsort(1)
    technicians=torch.rand((b,k),generator=generator).argsort(1)
    result=dict(batch)
    result['machines']=batch['machines'].gather(1,machines[...,None].expand(-1,-1,6))
    result['technicians']=batch['technicians'].gather(1,technicians[...,None].expand(-1,-1,3))
    edge=batch['edges'].gather(1,machines[:,:,None,None].expand(-1,-1,k,4))
    result['edges']=edge.gather(2,technicians[:,None,:,None].expand(-1,n,-1,4))
    mask=batch['pair_mask'].gather(1,machines[:,:,None].expand(-1,-1,k))
    result['pair_mask']=mask.gather(2,technicians[:,None,:].expand(-1,n,-1))
    result['state']=torch.cat([result['machines'].flatten(1),result['technicians'].flatten(1),
        result['edges'].flatten(1),result['global_features']],1)
    from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue
    result['joint_mask']=torch.stack([torch.stack([result['pair_mask'][:,m,j] for m,j in action],1).all(1)
        if action else torch.ones(b,dtype=torch.bool) for action in catalogue(4,2)],1)
    old_tokens=labels.gather(1,technicians)
    inverse=machines.argsort(1)
    following=torch.where(old_tokens>0,inverse.gather(1,(old_tokens-1).clamp_min(0))+1,0)
    return result,following,machines,technicians


def prepare_warm_job(job):
    """Generate one shared bank and three fixed-epoch warm starts for a seed."""
    cfg,seed,out_name,global_start=job
    from ht_pdm_fjsp.technician_prefix_capacity_v3_experiment import make_model
    out=Path(out_name);torch.set_num_threads(1)
    def timeout():
        if (out/'abort.json').exists():raise RuntimeError('sibling job failed; technical abort')
        if time.monotonic()-global_start>cfg['max_seconds']:raise TimeoutError('registered global deadline')
    timeout();directory=out/'warm'/str(seed);directory.mkdir(parents=True)
    references=json.loads((out/'teachers/references.json').read_text())['references'];teacher=load_reference(out,references[f'ppo:{seed}'])
    arrays,bank=build_bank(cfg,seed,directory,teacher,timeout);results={};initials=[]
    for algorithm in ('reservation_ppo','mask_only_ppo','independent_ppo'):
        timeout();model=make_model(algorithm,4,cfg,seed);initial=learner.parameter_digest(model);initials.append(initial)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        rng=torch.Generator().manual_seed(learner.role_seed(seed,VERSION+':teacher_minibatches'))
        augmentation=torch.Generator().manual_seed(learner.role_seed(seed,VERSION+':teacher_augmentation'))
        transformations=hashlib.sha256()
        journal=Journal(directory/f'{algorithm}_pretraining.csv');updates=presentations=0
        try:
            for epoch in tqdm(range(cfg['teacher_epochs']),desc=f'Distill {algorithm}/{seed}',unit='epoch'):
                permutation=torch.randperm(cfg['teacher_bank_steps'],generator=rng).numpy();losses=[]
                for start in range(0,len(permutation),cfg['teacher_minibatch']):
                    timeout();ids=permutation[start:start+cfg['teacher_minibatch']]
                    batch=learner.tensors({key:arrays[key][ids] for key in GRAPH_KEYS})
                    labels=torch.from_numpy(arrays['labels'][ids]);returns=torch.from_numpy(arrays['returns'][ids])
                    batch,labels,machines,technicians=rename_presentation(batch,labels,augmentation)
                    transformations.update(machines.numpy().tobytes());transformations.update(technicians.numpy().tobytes())
                    log,_,value=model.evaluate(batch,labels)
                    nll=-log.mean()/2;vf=torch.nn.functional.smooth_l1_loss(value,returns);loss=nll+.1*vf
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite distillation loss')
                    optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
                    if not torch.isfinite(norm):raise RuntimeError('nonfinite distillation gradients')
                    optimizer.step();updates+=1;presentations+=len(ids);losses.append(float(nll.detach()))
                journal.add(dict(algorithm=algorithm,train_seed=seed,epoch=epoch+1,optimizer_steps=updates,
                    state_presentations=presentations,mean_nll=float(np.mean(losses)),bank_sha256=bank['bank_sha256'],augmentation_sha256=transformations.hexdigest()))
        finally:journal.close()
        digest=learner.parameter_digest(model);path=directory/f'{algorithm}_warm.pt'
        payload=dict(protocol=VERSION,algorithm=algorithm,train_seed=seed,hidden=cfg['hidden'],n=4,
            initial_parameter_sha256=initial,parameter_sha256=digest,bank_sha256=bank['bank_sha256'],state_dict=model.state_dict())
        torch.save(payload,path)
        # Audit deterministic teacher matching agreement for diagnostics only.
        correct=0
        with torch.no_grad():
            for start in range(0,len(arrays['labels']),512):
                batch=learner.tensors({key:arrays[key][start:start+512] for key in GRAPH_KEYS})
                predicted=model.sample(batch,deterministic=True)[0].numpy();correct+=np.all(predicted==arrays['labels'][start:start+512],1).sum()
        result=dict(path=str(path.relative_to(out)),sha256=sha(path),initial_parameter_sha256=initial,
            parameter_sha256=digest,bank_sha256=bank['bank_sha256'],optimizer_steps=updates,
            state_presentations=presentations,epochs=cfg['teacher_epochs'],state_exposures=cfg['teacher_bank_steps'],
            teacher_matching_agreement=float(correct/len(arrays['labels'])),teacher_sha256=references[f'ppo:{seed}']['sha256'],augmentation_sha256=transformations.hexdigest())
        if updates!=cfg['teacher_epochs']*((cfg['teacher_bank_steps']+cfg['teacher_minibatch']-1)//cfg['teacher_minibatch']) or presentations!=cfg['teacher_epochs']*cfg['teacher_bank_steps'] or initial==digest:raise RuntimeError('warm training budget')
        results[algorithm]=result
    if len(set(initials))!=1:raise RuntimeError('warm initial pairing')
    if len({r['augmentation_sha256'] for r in results.values()})!=1:raise RuntimeError('teacher transformation pairing')
    write_json(directory/'warm_audit.json',dict(status='PASS',bank=bank,arms=results));return dict(train_seed=seed,bank=bank,arms=results)


def load_warm(model,out,seed):
    metadata=json.loads((Path(out)/'warm_choices.json').read_text())[str(seed)]['arms'][model.algorithm]
    path=Path(out)/metadata['path']
    if sha(path)!=metadata['sha256']:raise RuntimeError('warm checkpoint hash')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['protocol']!=VERSION or payload['algorithm']!=model.algorithm or payload['train_seed']!=seed:raise RuntimeError('warm checkpoint metadata')
    if learner.parameter_digest(model)!=metadata['initial_parameter_sha256']:raise RuntimeError('warm untrained initialization mismatch')
    model.load_state_dict(payload['state_dict'])
    if learner.parameter_digest(model)!=metadata['parameter_sha256']:raise RuntimeError('warm parameter digest')
    return metadata
