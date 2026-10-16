"""Frozen stochastic-policy entropy alignment; exact dense N4 reference only."""
import argparse,csv,hashlib,importlib.util,itertools,json,shutil,subprocess,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_broker_marginal_experiment as x
from ht_pdm_fjsp import technician_broker_learning_efficiency as efficiency
from ht_pdm_fjsp.technician_broker_marginal_policy import physical_logprob,physical_tokens,priorities

ROOT=Path(__file__).resolve().parents[2];VERSION='technician_broker_entropy_alignment_diagnostic_v1'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md';REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
STEPS=efficiency.STEPS;EARLY=STEPS[1:6];TRAINS=tuple(range(310000000,310000003));ARM=x.CORE_ARMS[0]
COMMIT='8a3c1f9e77449ccb3a51c91effe6005a90c28095'


def check(ok,label):
    if not ok:raise AssertionError(label)
def read(p):return json.loads(p.read_text())
def helpers():
    spec=importlib.util.spec_from_file_location('entropy_physics',ROOT/'scripts/verify_broker_marginal_development.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);return h


def seed_audit():
    r=read(REGISTRY)['diagnostic'];physical=list(range(r['physical_start'],r['physical_start']+r['physical_count']));actions=r['action_seeds'];raw=set(physical+actions);check(len(physical)==64 and len(actions)==3 and len(raw)==67,'fixed67raw panel')
    histories={};history=set();own=x.sha(REGISTRY)
    for p in [*(ROOT/'configs').glob('*seed_registry.json'),*Path('/home/bachng/ht-pdm-fjsp-worktrees').glob('*/configs/*seed_registry.json'),*Path('/home/bachng/ht-pdm-fjsp/configs').glob('*seed_registry.json')]:
        digest=x.sha(p)
        if digest==own or digest in histories:continue
        histories[digest]=str(p);history|=x.registered_integers(read(p))
    derived=[x.role_seed(s,f'{x.VERSION}:N4:physical') for s in physical]+[x.learner.role_seed(s,f'{VERSION}:replica:{i}:actions') for i,s in enumerate(actions)];known=set()
    for module in (x,x.reference,x.old_rsc,x.clearing_reference,x.raw_reference,x.budget_reference):
        for profile in ('development','confirmation','engineering'):known.update(x.derived_streams(module.settings(profile),module.VERSION))
    for profile in ('development','confirmation','engineering'):known.update(x.derived_streams(efficiency.settings(profile),x.VERSION))
    known.update(x.role_seed(s,f'{x.VERSION}:N4:physical') for s in history)
    check(not raw&history and len(derived)==67 and len(set(derived))==67 and not set(derived)&(raw|history|known),'fresh raw/actualderived diagnostic roles')
    return dict(status='PASS',raw_unique=67,derived_unique=67,physical_seeds=physical,action_seeds=actions,derived_seeds=derived,historical_registries=histories,known_historical_derived_checked=len(known),test_states_constructed=False)


def preimage_matrix(h):
    proposals=np.array(list(itertools.product(range(5),repeat=2)),int);matrix=np.zeros((25,21));o=h.observation(dict(ages=[3]*4,failed=[False]*4,pending_wait=[0]*4,remaining=[0,0],assigned=[-1,-1]),0);catalogue=h.baseline_catalogue()
    for i,u in enumerate(proposals):matrix[i,catalogue.index(h.independent_decode(u,np.ones((4,2),bool),ARM,o['edges'])[0])]=1
    return proposals,matrix


def entropies(logp,proposals,matrix):
    check(logp.ndim==3 and logp.shape[1:]==(2,5),'N4K2 reference entropy shape');p=logp.exp();u=torch.as_tensor(proposals,device=logp.device);mapping=torch.as_tensor(matrix,dtype=logp.dtype,device=logp.device)
    mass=(p[:,0,u[:,0]]*p[:,1,u[:,1]])@mapping
    hp=-(p*logp).sum(-1).mean(-1);hq=-(mass*mass.clamp_min(torch.finfo(mass.dtype).tiny).log()).sum(-1)/2
    return hp,hq,mass


def alignment(g_reward,g_proposal,g_physical):
    r,p,q=[np.asarray(g,np.float64) for g in (g_reward,g_proposal,g_physical)];rn,pn,qn=[float(np.dot(g,g)) for g in (r,p,q)];delta=q-p
    return dict(reward_squared_norm=rn,proposal_entropy_squared_norm=pn,physical_entropy_squared_norm=qn,entropy_gradients_nondegenerate=pn>1e-20 and qn>1e-20,
        entropy_gradient_cosine=float(np.dot(p,q)/np.sqrt(pn*qn)) if pn*qn>1e-40 else 0.,replacement_norm_ratio=float(.01*np.linalg.norm(delta)/max(np.sqrt(rn),1e-20)),projected_replacement=.01*float(np.dot(r,delta))/max(rn,1e-20))


def gradients(model,batch,actions,reward,value,h):
    count=reward.shape[1];check(reward.shape==value.shape==(48,count) and count%4==0,'complete equal episode blocks');returns=np.cumsum(np.asarray(reward,np.float64)[::-1],axis=0)[::-1].copy();done=torch.zeros((48,count),dtype=torch.bool);done[-1]=True;source_returns=x.learner.mc_returns(torch.tensor(reward,dtype=torch.float32),done);a=(source_returns-torch.tensor(value,dtype=torch.float32)).reshape(-1);a=(a-a.mean())/(a.std(unbiased=False)+1e-8)
    logp,_=model.proposal_log_probs(batch);priority=priorities(batch,ARM);physical=physical_tokens(actions,batch['pair_mask'],priority);logq=physical_logprob(logp,physical,batch['pair_mask'],priority);u,matrix=preimage_matrix(h);hp,hq,mass=entropies(logp,u,matrix)
    check(float((mass.sum(-1)-1).abs().max().detach())<5e-6 and float((hp-hq).min().detach())>=-2e-6,'normalized matching entropy chain rule')
    parameters=list(model.actor.parameters());expressions=dict(reward=(logq*a).mean(),proposal=hp.mean(),physical=hq.mean());vectors={}
    for name,expr in expressions.items():vectors[name]=torch.cat([g.reshape(-1) for g in torch.autograd.grad(expr,parameters,retain_graph=True)]).detach().numpy().astype(np.float64)
    blocks=[]
    for block in range(4):
        ids=(torch.arange(48)[:,None]*count+torch.arange(block*count//4,(block+1)*count//4)[None]).reshape(-1)
        g=torch.autograd.grad((logq[ids]*a[ids]).mean(),parameters,retain_graph=block<3);blocks.append(torch.cat([v.reshape(-1) for v in g]).detach().numpy().astype(np.float64))
    vectors['reward_blocks']=np.stack(blocks);check(all(np.isfinite(v).all() for v in vectors.values()),'finite actor gradients');block_error=float(np.abs(vectors['reward_blocks'].mean(0)-vectors['reward']).max());check(np.allclose(vectors['reward_blocks'].mean(0),vectors['reward'],atol=2e-6,rtol=2e-5),'block/global reward gradient identity')
    stats=dict(mean_proposal_entropy=float(hp.mean().detach()),mean_physical_entropy=float(hq.mean().detach()),mean_entropy_gap=float((hp-hq).mean().detach()),relative_alias_gap=float((hp.mean()-hq.mean()).detach())/max(float(hp.mean().detach()),1e-20),block_gradient_max_identity_error=block_error,**alignment(vectors['reward'],vectors['proposal'],vectors['physical']))
    return stats,vectors,dict(reward=np.asarray(reward,np.float64),value=np.asarray(value,np.float64),returns=returns,advantage=a.numpy().reshape(48,count),hp=hp.detach().numpy(),hq=hq.detach().numpy(),matching_mass=mass.detach().numpy(),logq=logq.detach().numpy())


def collect(model,seeds,rng,directory,h,guard=lambda:None):
    physical=x.Journal(directory/'physical.csv');policy=x.Journal(directory/'policy.csv');envs=[x.ResourceEnv(s,4,ARM,journal=x.Tags(physical,dict(episode=i))) for i,s in enumerate(seeds)];observations=[e.reset() for e in envs];batches=[];actions=[];logs=[];values=[];rewards=[];episodes=[]
    try:
        for tick in tqdm(range(48),desc=f'{len(seeds)} stochastic episodes,48ticks',leave=False,mininterval=1):
            guard();batch=x.learner.tensors(observations)
            with torch.no_grad():action,log,value=model.sample(batch,rng,epsilon=0.)
            batches.append(batch);actions.append(action);logs.append(log);values.append(value);next_obs=[];reward=[]
            for i,env in enumerate(envs):
                submitted=action[i].numpy().copy();pairs,diag=env.resolve(submitted);obs,r,done,info=env.step(submitted);check(done==(tick==47) and diag==info['coordination'],'clock/arbitration replay');next_obs.append(obs);reward.append(r)
                policy.add(dict(episode=i,seed=seeds[i],tick=tick,physical_seed=env.physical_seed,action=json.dumps(submitted.tolist()),pairs=json.dumps(pairs),diagnostics=json.dumps(diag),observation_sha256=hashlib.sha256(observations[i]['state'].tobytes()).hexdigest(),reward=r,log_probability=float(log[i]),value=float(value[i])))
                if done:episodes.append(dict(episode=i,seed=seeds[i],**info['episode_metrics']))
            rewards.append(reward);observations=next_obs
    finally:physical.close();policy.close()
    reward=np.asarray(rewards,np.float64);value=torch.stack(values).numpy()
    for ep in episodes:check(abs(-20*reward[:,ep['episode']].sum()-ep['extended_cost'])<1e-7 and ep['physical_ticks']==60 and ep['invalid_assignments']==0,'episode reward/cost/clock')
    # Independent pure dictionary replay of all repository scalar transitions.
    replay={}
    with (directory/'physical.csv').open() as stream:
        for record in csv.DictReader(stream):
            i=int(record['episode']);tick=int(record['tick']);s=json.loads(record['state']);pairs=tuple(map(tuple,json.loads(record['action'])))
            if i not in replay:
                initial,events,pop=h.initial_events(int(record['physical_seed']));replay[i]=dict(state=initial,events=events,pop=pop,tick=0,total=0.)
            ep=replay[i];check(ep['tick']==tick and ep['state']==s and ep['events'][tick]==json.loads(record['events']),'seeded physical state/event chain');following,parts=h.primitive_step(s,pairs,ep['events'][tick]);check(following==json.loads(record['following']) and abs(parts['objective']-float(record['cost']))<1e-8,'independent primitive physical transition');ep['state']=following;ep['tick']+=1;ep['total']+=parts['objective']
    check(len(episodes)==len(seeds) and len(replay)==len(seeds),'complete episode grid')
    for row in episodes:
        ep=replay[row['episode']];check(ep['tick']==60 and ep['pop']==row['population_sha256'] and ep['total']==row['extended_cost'],'physical aggregate/population')
    flat={k:torch.cat([b[k] for b in batches]) for k in batches[0]}
    return flat,torch.cat(actions),torch.cat(logs),value,reward,episodes


def screen(records):
    checks={};means={}
    for t in TRAINS:
        rows=[r for r in records if r['train_seed']==t and r['checkpoint_steps'] in EARLY];check(len(rows)==5 and {r['checkpoint_steps'] for r in rows}==set(EARLY),'fixed five early checkpoints')
        values=dict(relative_alias_gap=float(np.mean([r['relative_alias_gap'] for r in rows])),opposed_entropy_checkpoints=sum(r['entropy_gradients_nondegenerate'] and r['entropy_gradient_cosine']<=0 for r in rows),material_replacement_checkpoints=sum(r['reward_squared_norm']>1e-20 and r['replacement_norm_ratio']>=.10 for r in rows),mean_projected_replacement=float(np.mean([r['projected_replacement'] for r in rows])),all_reward_gradients_nondegenerate=all(r['reward_squared_norm']>1e-20 for r in rows));means[str(t)]=values
        checks[str(t)]=dict(material_alias_gap=values['relative_alias_gap']>=.10,majority_entropy_opposition=values['opposed_entropy_checkpoints']>=3,majority_material_replacement=values['material_replacement_checkpoints']>=3,favorable_mean_projection=values['mean_projected_replacement']>=.01 and values['all_reward_gradients_nondegenerate'])
    return dict(classification='FROZEN_ENTROPY_ALIGNMENT_DEV_ONLY',primary_replica_metrics=means,checks=checks,entropy_remedy_further_design_motivated=all(all(v.values()) for v in checks.values()),training_authorized=False,goal_pass=False,test_opened=False,updates=0,limitation='Frozen finite-batch actor-only directions at behavior parameters, not Adam/global-clip/critic-coupled steps or causal H60/AUC gains. Dense N4 entropy is diagnostic only; no coefficient/window sweep or test opening.')


def run(source,verification,output):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(1);h=helpers()
    manifest=dict(protocol=VERSION,status='RUNNING',source=str(source),verification=str(verification),source_manifest_sha256=x.sha(source/'manifest.json'),verification_manifest_sha256=x.sha(verification/'manifest.json'),git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())),goal_pass=False,test_opened=False,updates=0,teacher_assistance=False)
    x.write_json(out/'manifest.json',manifest)
    def guard():check(time.monotonic()-start<900,'15minute hardstop')
    def bindings():
        for folder,key,binding in ((source,'source_manifest_sha256','output_bindings'),(verification,'verification_manifest_sha256','outputs')):
            check(x.sha(folder/'manifest.json')==manifest[key],'input manifest unchanged')
            for name,meta in read(folder/'manifest.json')[binding].items():guard();check(x.sha(folder/name)==meta['sha256'] and (folder/name).stat().st_size==meta['bytes'],'input binding '+name)
        for name,digest in read(source/'manifest.json')['source_seal'].items():
            if name.startswith('source_snapshot/'):check(x.sha(Path(x.__file__).parent/Path(name).name)==digest,'frozen source code')
    try:
        m=read(source/'manifest.json');v=read(verification/'verification.json');check(not manifest['git']['dirty'] and m['git']['commit']==COMMIT and not m['git']['dirty'] and m['status']=='COMPLETED' and m['audit_status']=='PASS' and not m['development_screen_pass'] and not m['test_opened'],'frozen rejected source');check(read(verification/'manifest.json')['status']=='COMPLETED' and v['status']=='PASS' and v['source']==str(source) and v['source_manifest_sha256']==manifest['source_manifest_sha256'] and not v['original_development_pass'],'source independent verification');check(not any('from multiprocessing.spawn import spawn_main' in line for line in subprocess.check_output(['ps','-eo','args'],text=True).splitlines()),'no active training workers');bindings();audit=seed_audit();x.write_json(out/'seed_audit.json',audit)
        for p,name in ((PLAN,'protocol.md'),(REGISTRY,'seed_registry.json'),(Path(__file__),'diagnostic.py'),(ROOT/'scripts/verify_broker_marginal_development.py','reference_helpers.py')):shutil.copy2(p,out/name)
        seals={name:x.sha(out/name) for name in ('protocol.md','seed_registry.json','diagnostic.py','reference_helpers.py')};c=read(source/'resolved_config.json');records=[];all_episodes=[];inputs=[];population={}
        for t,step in tqdm([(t,s) for t in TRAINS for s in STEPS],desc='24frozen entropy alignment models'):
            guard();path=Path('N4')/ARM/str(t)/'checkpoints'/f'step_{step:09d}.pt';model=x.load_model(source/path,ARM,c,t,step);before=x.learner.parameter_digest(model);directory=out/str(t)/str(step);directory.mkdir(parents=True);index=TRAINS.index(t);action_seed=x.learner.role_seed(audit['action_seeds'][index],f'{VERSION}:replica:{index}:actions');rng=torch.Generator().manual_seed(action_seed)
            batch,actions,old_log,value,reward,episodes=collect(model,audit['physical_seeds'],rng,directory,h,guard)
            with torch.no_grad():
                log,_,current_value=model.evaluate(batch,actions);replay=x.learner.behavior_replay_audit(log,old_log,current_value,torch.tensor(value.reshape(-1)),dict(behavior_log_atol=2e-5,behavior_log_rtol=0.,behavior_value_atol=2e-5*x.value_unit(4),behavior_value_rtol=1e-6));done=torch.zeros((48,64),dtype=torch.bool);done[-1]=True;source_returns=x.learner.mc_returns(torch.tensor(reward,dtype=torch.float32),done).numpy()
            stats,vectors,arrays=gradients(model,batch,actions,reward,value,h);error=float(np.max(np.abs(arrays['returns']-source_returns)));check(error<=5e-4,'float64/source MC return identity');check(x.learner.parameter_digest(model)==before and all(p.grad is None for p in model.parameters()),'frozen parameters/grad fields');np.savez_compressed(directory/'arrays.npz',**arrays,actions=actions.numpy(),old_log=old_log.numpy());np.savez_compressed(directory/'observations.npz',**{k:v.numpy() for k,v in batch.items()});np.savez_compressed(directory/'gradients.npz',**vectors)
            record=dict(train_seed=t,checkpoint_steps=step,action_seed=action_seed,parameter_sha256=before,checkpoint_path=str(path),checkpoint_sha256=x.sha(source/path),**stats,behavior_replay=replay,mc_return_max_error=error,mean_stochastic_cost_per_machine=float(np.mean([r['extended_cost']/4 for r in episodes])))
            x.write_json(directory/'statistics.json',record);records.append(record);inputs.append(dict(train_seed=t,checkpoint_steps=step,path=str(path),sha256=record['checkpoint_sha256'],parameter_sha256=before))
            for ep in episodes:check(population.setdefault(ep['seed'],ep['population_sha256'])==ep['population_sha256'],'all24models paired tapes');all_episodes.append(dict(train_seed=t,checkpoint_steps=step,**ep))
            x.write_json(out/'records.partial.json',records)
        check(len(records)==24 and len(all_episodes)==1536,'complete24model/1536episode grid');x.write_json(out/'checkpoint_bindings.json',inputs);x.write_csv(out/'episodes.csv',all_episodes);summary=dict(**screen(records),models=24,episodes=1536,policy_decisions=73728,physical_ticks=92160,records=records);x.write_json(out/'summary.json',summary);bindings()
        check(all(x.sha(out/name)==digest for name,digest in seals.items()) and x.sha(Path(__file__))==seals['diagnostic.py'] and x.sha(PLAN)==seals['protocol.md'] and x.sha(REGISTRY)==seals['seed_registry.json'] and x.sha(ROOT/'scripts/verify_broker_marginal_development.py')==seals['reference_helpers.py'],'diagnostic code/protocol unchanged');manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,entropy_remedy_further_design_motivated=summary['entropy_remedy_further_design_motivated'],output_bindings={str(p.relative_to(out)):dict(sha256=x.sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'});x.write_json(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),status='PASS',checks=summary['checks'])));return summary
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);x.write_json(out/'manifest.json',manifest);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.source,args.verification,args.output)
