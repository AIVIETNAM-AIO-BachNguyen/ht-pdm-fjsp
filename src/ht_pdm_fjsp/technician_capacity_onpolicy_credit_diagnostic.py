"""Frozen stochastic-policy credit diagnostics; no optimizer or teacher targets."""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_capacity_factorized_experiment as factored
from ht_pdm_fjsp import technician_residual_capacity_experiment as rsc

ROOT=Path(__file__).resolve().parents[2]
VERSION='technician_capacity_onpolicy_credit_diagnostic_v1'
REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
ARMS={'rsc':('residual_capacity_ppo','no_capacity_ppo'),
      'factored':('residual_capacity_ppo','no_capacity_ppo','flat_capacity_ppo','ppo','normalized_ppo')}
SOURCES={'rsc':('b44f114a8bc44ca2fefdfcd4eadf18d1e0f5322d',519,rsc),
         'factored':('a10e049e3f8b4e242efbc879b2d3dc3c13925bf4',548,factored)}


def seed_audit():
    registry=json.loads(REGISTRY.read_text())['diagnostic']
    physical=list(range(registry['physical_start'],registry['physical_start']+registry['physical_count']))
    actions=registry['action_seeds'];own=set(physical+actions)
    if len(physical)!=64 or len(actions)!=3 or len(own)!=67:raise ValueError('fixed seed grid')
    history=set();hashes={};own_hash=factored.sha(REGISTRY)
    paths=list((ROOT/'configs').glob('*seed_registry.json'))+list(Path('/home/bachng/ht-pdm-fjsp-worktrees').glob('*/configs/*seed_registry.json'))+list(Path('/home/bachng/ht-pdm-fjsp/configs').glob('*seed_registry.json'))
    for p in paths:
        h=factored.sha(p)
        if h==own_hash or h in hashes:continue
        hashes[h]=str(p);history|=factored.registered_integers(json.loads(p.read_text()))
    derived=[factored.physical_seed(s,f'{factored.VERSION}:N4:physical') for s in physical]
    derived += [factored.learner.role_seed(s,f'{VERSION}:{role}:actions') for s in actions for role in ('graph','ppo')]
    historical_derived=set()
    for _,_,module in SOURCES.values():
        for profile in module.PROFILES:
            c=module.settings(profile)
            historical_derived.add(module.learner.role_seed(c['analysis_seed'],module.VERSION+':bootstrap'))
            for n in c['scales']:
                for s in (*c['development_seeds'],*c['test_seeds']):historical_derived.add(module.physical_seed(s,f'{module.VERSION}:N{n}:physical'))
                for t in c['train_seeds']:
                    for role in ('graph','ppo','mappo','full_mappo'):
                        historical_derived.add(module.learner.role_seed(t,f'{module.VERSION}:initialization:{role}'))
                        for stream in ('actions','minibatches','replay'):
                            historical_derived.add(module.learner.role_seed(t,f'{module.VERSION}:N{n}:{role}:{stream}'))
                    for w in range(c['vector_envs']):
                        for e in range(c['env_steps']//(48*c['vector_envs'])):
                            historical_derived.add(module.physical_seed(t,f'{module.VERSION}:N{n}:training_physical:worker:{w}:episode:{e}'))
    if own&history or len(derived)!=len(set(derived)) or set(derived)&(own|history|historical_derived):raise ValueError('seed collision')
    return dict(status='PASS',raw_unique=len(own),derived_unique=len(derived),historical_registries=hashes,
        source_family_historical_derived_checked=len(historical_derived),physical_seeds=physical,action_seeds=actions,
        limitation='Historical raw registries across local worktrees and derived streams for the two source families; unrelated historical algorithms have no unified derived-stream registry.')


def credit_arrays(reward,value,lam):
    reward,value=np.asarray(reward,np.float64),np.asarray(value,np.float64)
    if reward.shape!=value.shape or reward.ndim!=2 or reward.shape[0]!=48 or not 0<=lam<=1:
        raise ValueError('whole-episode credit shape/lambda')
    returns=np.cumsum(reward[::-1],axis=0)[::-1].copy()
    following=np.concatenate([value[1:],np.zeros_like(value[:1])])
    delta=reward+following-value;advantage=np.empty_like(delta);future=np.zeros(reward.shape[1])
    for tick in reversed(range(48)):
        future=delta[tick]+lam*future;advantage[tick]=future
    return returns,advantage


def cosine(a,b):
    a,b=np.asarray(a,np.float64).ravel(),np.asarray(b,np.float64).ravel()
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denominator) if denominator>1e-20 else 0.


def credit_statistics(reward,value):
    returns,mc=credit_arrays(reward,value,1.);_,gae=credit_arrays(reward,value,.95)
    error=np.asarray(value)-returns;variance=float(np.var(returns));sd=variance**.5
    if not np.isfinite(returns).all() or variance<=0 or np.max(np.abs(mc-(returns-value)))>1e-10:
        raise RuntimeError('MC/GAE identity or finite variance')
    reference=np.empty_like(returns)
    reference[:,::2]=returns[:,1::2].mean(1,keepdims=True)
    reference[:,1::2]=returns[:,::2].mean(1,keepdims=True)
    stats=dict(return_mean=float(returns.mean()),return_sd=sd,value_mean=float(np.mean(value)),
        prediction_rmse=float(np.sqrt(np.mean(error**2))),prediction_mean_error=float(error.mean()),
        absolute_mean_error_in_return_sd=abs(float(error.mean()))/sd,
        critic_explained_variance=1-float(np.var(error))/variance,
        critic_mse_skill=1-float(np.mean(error**2))/variance,
        crossfit_time_residual_skill=1-float(np.sum(error**2))/max(float(np.sum((returns-reference)**2)),1e-20),
        raw_advantage_variance_ratio=float(np.var(gae)/max(np.var(mc),1e-20)),
        mean_advantage_shift_in_mc_sd=float((gae.mean()-mc.mean())/max(mc.std(),1e-20)),
        centered_advantage_cosine=cosine(mc-mc.mean(),gae-gae.mean()),lambda1_max_identity_error=float(np.max(np.abs(mc-(returns-value)))))
    return stats,dict(reward=np.asarray(reward,np.float64),value=np.asarray(value,np.float64),returns=returns,mc_advantage=mc,gae_advantage=gae,time_reference=reference)


def block_dispersion(gradients):
    g=np.asarray(gradients,np.float64);mean=g.mean(0);norm=float(np.dot(mean,mean))
    dispersion=float(np.mean(np.sum((g-mean)**2,axis=1))/max(norm,1e-20))
    return dict(dispersion=dispersion,mean_gradient_squared_norm=norm,block_gradient_norms=np.linalg.norm(g,axis=1).tolist())


def gradient_diagnostic(model,batch,actions,arrays):
    count=arrays['reward'].shape[1]
    if count!=64:raise ValueError('four fixed blocks of16 episodes')
    parameters=list(model.gradient_groups()['actor'] if hasattr(model,'gradient_groups') else model.actor.parameters())
    results={};vectors={}
    for label,key in (('mc','mc_advantage'),('gae','gae_advantage')):
        a=torch.tensor(arrays[key].reshape(-1),dtype=torch.float32)
        a=(a-a.mean())/(a.std(unbiased=False)+1e-8);grads=[]
        for block in range(4):
            indices=(torch.arange(48)[:,None]*64+torch.arange(block*16,(block+1)*16)[None]).reshape(-1)
            mini={k:v[indices] for k,v in batch.items()}
            log,_,_=model.evaluate(mini,actions[indices]);loss=-(log*a[indices]).mean()
            tensors=torch.autograd.grad(loss,parameters,allow_unused=True)
            flat=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(parameters,tensors)])
            if not torch.isfinite(flat).all():raise RuntimeError('nonfinite actor score gradient')
            grads.append(flat.detach().numpy().astype(np.float64))
        vectors[label]=np.stack(grads);results[label]=block_dispersion(vectors[label])
    results['dispersion_ratio']=results['gae']['dispersion']/max(results['mc']['dispersion'],1e-20)
    results['mean_gradient_cosine']=cosine(vectors['mc'].mean(0),vectors['gae'].mean(0))
    return results,vectors


def eligibility(records):
    families={}
    for family in ARMS:
        candidate=[r for r in records if r['family']==family and r['method']=='residual_capacity_ppo']
        if len(candidate)!=3 or len({r['train_seed'] for r in candidate})!=3:raise ValueError('candidate replica grid')
        checks={str(r['train_seed']):dict(critic_explained_variance=r['critic_explained_variance']>=.5,
            prediction_mean_error=r['absolute_mean_error_in_return_sd']<=.1,
            gradient_dispersion=r['gradient']['dispersion_ratio']<=.75,
            gradient_direction=r['gradient']['mean_gradient_cosine']>=.8,
            nonzero_mc_gradient=r['gradient']['mc']['mean_gradient_squared_norm']>1e-20) for r in candidate}
        families[family]=dict(checks=checks,eligible=all(all(c.values()) for c in checks.values()))
    preferred='rsc' if families['rsc']['eligible'] else 'factored' if families['factored']['eligible'] else None
    return dict(families=families,preferred_followup=preferred,training_authorized=False,
        limitation='Fixed engineering signal only; any learning trial requires a new registered protocol, random initialization and matched estimators/budgets across controls.')


def verify_source(path,family,guard):
    expected,count,module=SOURCES[family];manifest=json.loads((path/'manifest.json').read_text())
    c=json.loads((path/'resolved_config.json').read_text())
    if manifest['status']!='COMPLETED' or manifest['audit_status']!='PASS' or manifest['test_opened'] or manifest['git']['commit']!=expected or len(manifest['output_bindings'])!=count or c!=module.settings('development'):
        raise RuntimeError('source contract '+family)
    for name,meta in tqdm(manifest['output_bindings'].items(),desc=f'Verify {family} inputs',leave=False):
        guard();p=path/name
        if factored.sha(p)!=meta['sha256'] or p.stat().st_size!=meta['bytes']:raise RuntimeError('changed source '+name)
    for name,h in manifest['source_seal'].items():
        if factored.sha(path/name)!=h:raise RuntimeError('source seal')
        if name.startswith('source_snapshot/') and factored.sha(Path(__file__).parent/Path(name).name)!=h:raise RuntimeError('local learning source changed '+name)
    return c,json.loads((path/'checkpoint_choices.json').read_text())


def collect(model,method,seeds,rng,guard):
    envs=[factored.ResourceEnv(s,4,method) for s in seeds];observations=[e.reset() for e in envs]
    batches=[];action_rows=[];logs=[];values=[];rewards=[];episodes=[];decisions=[]
    for tick in range(48):
        guard();batch=factored.learner.tensors(observations)
        with torch.no_grad():action,log,value=model.sample(batch,rng,epsilon=0.)
        batches.append(batch);action_rows.append(action);logs.append(log);values.append(value);reward=[];next_obs=[]
        for i,env in enumerate(envs):
            submitted=(model.environment_action(action[i]) if hasattr(model,'environment_action')
                       else int(action[i]) if model.algorithm=='ppo' else action[i].numpy().copy())
            obs,r,done,info=env.step(submitted)
            if done!=(tick==47):raise RuntimeError('terminal timing')
            reward.append(r);next_obs.append(obs)
            decisions.append(dict(episode=i,seed=seeds[i],tick=tick,physical_seed=env.physical_seed,
                action=json.dumps(int(submitted) if np.isscalar(submitted) else submitted.tolist()),
                observation_sha256=hashlib.sha256(observations[i]['state'].tobytes()).hexdigest(),
                reward=r,value=float(value[i]),log_probability=float(log[i])))
            if done:episodes.append(dict(episode=i,seed=seeds[i],**info['episode_metrics']))
        rewards.append(reward);observations=next_obs
    batch={k:torch.cat([b[k] for b in batches]) for k in batches[0]}
    reward=np.asarray(rewards,np.float64);values=torch.stack(values).numpy()
    for ep in episodes:
        if abs(-20*reward[:,ep['episode']].sum()-ep['extended_cost'])>1e-7 or ep['physical_ticks']!=60 or ep['invalid_assignments']!=0:
            raise RuntimeError('episode reward/cost/health reconciliation')
    return batch,torch.cat(action_rows),torch.cat(logs),values,reward,episodes,decisions


def run(source_paths,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);start=time.monotonic()
    def guard():
        if time.monotonic()-start>900:raise TimeoutError('15min hardstop')
    sources={k:Path(v).resolve() for k,v in source_paths.items()}
    manifest=dict(protocol=VERSION,status='RUNNING',classification='POSTHOC_ONPOLICY_DEVELOPMENT_DIAGNOSTIC',goal_pass=False,test_opened=False,training_steps=0,
        git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())),
        sources={k:dict(path=str(p),manifest_sha256=factored.sha(p/'manifest.json')) for k,p in sources.items()})
    factored.write_json(out/'manifest.json',manifest)
    try:
        if manifest['git']['dirty']:raise RuntimeError('dirty code')
        audited={family:verify_source(path,family,guard) for family,path in sources.items()};audit=seed_audit()
        factored.write_json(out/'seed_audit.json',audit)
        for src,name in ((PLAN,'protocol.md'),(REGISTRY,'seed_registry.json'),(Path(__file__),'diagnostic.py')):shutil.copy2(src,out/name)
        records=[];inputs=[];population={};all_episodes=[]
        jobs=[(family,method,i,t) for family,(c,_) in audited.items() for i,t in enumerate(c['train_seeds']) for method in ARMS[family]]
        for family,method,index,t in tqdm(jobs,desc='21 frozen on-policy models'):
            guard();c,choices=audited[family];module=SOURCES[family][2];choice=choices[f'4:{method}:{t}']
            model=module.load_model(sources[family]/choice['path'],4,c,t,method,choice['steps']);before=module.learner.parameter_digest(model)
            directory=out/family/method/str(t);directory.mkdir(parents=True)
            role='ppo' if method in ('ppo','normalized_ppo') else 'graph'
            action_seed=module.learner.role_seed(audit['action_seeds'][index],f'{VERSION}:{role}:actions')
            rng=torch.Generator().manual_seed(action_seed)
            batch,actions,old_log,value,reward,episodes,decisions=collect(model,method,audit['physical_seeds'],rng,guard)
            with torch.no_grad():
                log,_,current_value=model.evaluate(batch,actions)
                unit=1 if method=='ppo' else module.value_unit(4)
                replay=module.learner.behavior_replay_audit(log,old_log,current_value,torch.tensor(value.reshape(-1)),
                    dict(behavior_log_atol=2e-5,behavior_log_rtol=0.,behavior_value_atol=2e-5*unit,behavior_value_rtol=1e-6))
                done=torch.zeros((48,64),dtype=torch.bool);done[-1]=True
                reference=module.learner.mc_returns(torch.tensor(reward,dtype=torch.float32),done).numpy()
            stats,arrays=credit_statistics(reward,value)
            error=float(np.max(np.abs(reference-arrays['returns'])))
            if error>5e-4:raise RuntimeError('independent return disagreement')
            gradient,vectors=gradient_diagnostic(model,batch,actions,arrays)
            if before!=module.learner.parameter_digest(model) or any(p.grad is not None for p in model.parameters()):raise RuntimeError('model mutated')
            np.savez_compressed(directory/'credit_arrays.npz',**arrays)
            np.savez_compressed(directory/'block_gradients.npz',**vectors)
            np.savez_compressed(directory/'observations.npz',**{k:v.numpy() for k,v in batch.items()},actions=actions.numpy())
            factored.write_csv(directory/'decisions.csv',decisions);factored.write_csv(directory/'episodes.csv',episodes)
            for ep in episodes:
                previous=population.setdefault(ep['seed'],ep['population_sha256'])
                if previous!=ep['population_sha256']:raise RuntimeError('unpaired physical tape')
                all_episodes.append(dict(family=family,method=method,train_seed=t,**ep))
            record=dict(family=family,method=method,train_seed=t,checkpoint_steps=choice['steps'],action_seed=action_seed,
                **stats,gradient=gradient,behavior_replay=replay,independent_return_max_error=error,
                mean_cost_per_machine=float(np.mean([ep['extended_cost']/4 for ep in episodes])))
            factored.write_json(directory/'statistics.json',record);records.append(record)
            inputs.append(dict(family=family,method=method,train_seed=t,choice=choice,parameter_sha256=before))
        if len(records)!=21 or len(all_episodes)!=1344:raise RuntimeError('fixed full grid')
        summary=dict(classification=manifest['classification'],goal_pass=False,test_opened=False,models=21,episodes=1344,
            policy_decisions=64512,physical_ticks=80640,training_steps=0,eligibility=eligibility(records),records=records)
        factored.write_json(out/'inputs.json',inputs);factored.write_json(out/'summary.json',summary);factored.write_csv(out/'episodes.csv',all_episodes)
        (out/'report.md').write_text('# Frozen on-policy credit diagnostic\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
        for family,path in sources.items():
            verify_source(path,family,guard)
            if factored.sha(path/'manifest.json')!=manifest['sources'][family]['manifest_sha256']:raise RuntimeError('source manifest changed')
        manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,
            output_bindings={str(p.relative_to(out)):dict(sha256=factored.sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'})
        factored.write_json(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),eligibility=summary['eligibility'])));return manifest
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);factored.write_json(out/'manifest.json',manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rsc-source',required=True);p.add_argument('--factored-source',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    run(dict(rsc=a.rsc_source,factored=a.factored_source),a.output_dir)

if __name__=='__main__':main()
