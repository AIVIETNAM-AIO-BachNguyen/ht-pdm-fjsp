"""Frozen policy mean versus single-sample execution on fresh diagnostic tapes."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,shutil,subprocess,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_joint_clearing_experiment as x

ROOT=Path(__file__).resolve().parents[2];VERSION='technician_joint_execution_diagnostic_v1'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md';REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'


def primitive_helpers():
    spec=importlib.util.spec_from_file_location('joint_execution_primitive',ROOT/'scripts/verify_joint_clearing_development.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def seed_audit():
    raw=json.loads(REGISTRY.read_text())['diagnostic'];physical=list(range(raw['physical_start'],raw['physical_start']+raw['physical_count']));actions=list(range(raw['action_start'],raw['action_start']+raw['action_count']))
    if len(physical)!=64 or len(actions)!=12:raise ValueError('fixed panel')
    own=set(physical+actions);history=set();hashes={};own_hash=x.sha(REGISTRY)
    paths=list((ROOT/'configs').glob('*seed_registry.json'))+list(Path('/home/bachng/ht-pdm-fjsp-worktrees').glob('*/configs/*seed_registry.json'))
    for p in paths:
        h=x.sha(p)
        if h==own_hash or h in hashes:continue
        hashes[h]=str(p);history|=x.registered_integers(json.loads(p.read_text()))
    derived=[x.role_seed(s,f'{x.VERSION}:N4:physical') for s in physical]
    derived += [x.learner.role_seed(s,f'{VERSION}:{role}:actions') for s in actions for role in ('core','ppo','mappo')]
    old=set()
    for module in (x,x.reference,x.old_rsc):
        for profile in ('development','confirmation','engineering'):old.update(x.derived_streams(module.settings(profile),module.VERSION))
    if len(own)!=76 or own&history or len(derived)!=len(set(derived)) or set(derived)&(history|own|old):raise ValueError('seed collision')
    return dict(status='PASS',raw_unique=len(own),derived_unique=len(derived),physical_seeds=physical,action_seeds=actions,historical_registries=hashes,
        known_historical_derived_checked=len(old),limitation='Historical raw registries across worktrees; derived streams for three source learning families explicitly checked.')


def collect(model,method,seeds,mode,rng,t,repeat,physical_journal,policy_journal,guard):
    if mode not in ('mean','sample'):raise ValueError('mode')
    tags=[dict(method=method,train_seed=t,mode=mode,repeat=repeat,episode=i) for i in range(len(seeds))]
    envs=[x.ResourceEnv(s,4,method,journal=x.Tags(physical_journal,tags[i])) for i,s in enumerate(seeds)]
    observations=[e.reset() for e in envs];rewards=[];episodes=[];batches=[];actions=[];logs=[];values=[]
    for tick in range(48):
        guard()
        if mode=='sample':
            batch=x.learner.tensors(observations)
            with torch.no_grad():a,log,value=model.sample(batch,rng,epsilon=0.)
            batches.append(batch);actions.append(a);logs.append(log);values.append(value)
        reward=[];next_obs=[]
        for i,env in enumerate(envs):
            if mode=='mean':submitted=model.select(observations[i])
            else:submitted=model.environment_action(a[i]) if hasattr(model,'environment_action') else int(a[i]) if method=='ppo' else a[i].numpy().copy()
            before_hash=hashlib.sha256(observations[i]['state'].tobytes()).hexdigest();obs,r,done,info=env.step(submitted)
            if done!=(tick==47):raise RuntimeError('episode timing')
            policy_journal.add(dict(tags[i],seed=seeds[i],tick=tick,submitted=json.dumps(int(submitted) if np.isscalar(submitted) else submitted.tolist()),
                observation_sha256=before_hash,reward=r,log_probability=json.dumps(log[i].tolist()) if mode=='sample' else '',value=float(value[i]) if mode=='sample' else '',diagnostics=json.dumps(info['coordination'],sort_keys=True)))
            reward.append(r);next_obs.append(obs)
            if done:episodes.append(dict(tags[i],seed=seeds[i],**info['episode_metrics']))
        rewards.append(reward);observations=next_obs
    reward=np.asarray(rewards,np.float64)
    for ep in episodes:
        if abs(-20*reward[:,ep['episode']].sum()-ep['extended_cost'])>1e-7 or ep['physical_ticks']!=60 or ep['invalid_assignments']!=0:raise RuntimeError('reward/health reconciliation')
    replay=None
    if mode=='sample':
        flat={k:torch.cat([b[k] for b in batches]) for k in batches[0]}
        with torch.no_grad():current_log,_,current_value=model.evaluate(flat,torch.cat(actions))
        replay=x.learner.behavior_replay_audit(current_log,torch.cat(logs),current_value,torch.cat(values),
            dict(behavior_log_atol=2e-5,behavior_log_rtol=0.,behavior_value_atol=2e-5*(1 if method=='ppo' else x.value_unit(4)),behavior_value_rtol=1e-6))
    return episodes,replay


def audit_physics(directory,episodes,helpers,guard):
    lookup={(r['mode'],r['repeat'],r['episode']):r for r in episodes};policies={};states={};count=0
    with (directory/'policy.csv').open() as f:
        for r in csv.DictReader(f):
            key=r['mode'],int(r['repeat']),int(r['episode']),int(r['tick'])
            if key in policies:raise RuntimeError('duplicate policy record')
            policies[key]=r
    with (directory/'physical.csv').open() as f:
        for r in csv.DictReader(f):
            if count%1000==0:guard()
            key=r['mode'],int(r['repeat']),int(r['episode']);tick=int(r['tick']);state=json.loads(r['state']);pairs=tuple(map(tuple,json.loads(r['action'])))
            if key not in lookup:raise RuntimeError('unexpected physical condition')
            if key not in states:
                initial,events,population=helpers.initial_events(int(r['physical_seed']));states[key]=dict(state=initial,events=events,tick=0,population=population,metrics=defaultdict(float))
            ep=states[key]
            if state!=ep['state'] or tick!=ep['tick'] or json.loads(r['events'])!=ep['events'][tick]:raise RuntimeError('independent state/event chain')
            if tick<48:
                policy=policies[key+(tick,)];obs=helpers.observation(state,tick)
                if hashlib.sha256(obs['state'].tobytes()).hexdigest()!=policy['observation_sha256']:raise RuntimeError('independent observation')
                submitted=json.loads(policy['submitted']);method=r['method'];diag=json.loads(policy['diagnostics'])
                if method in x.CORE_ARMS:helpers.verify_decoder(submitted,obs['pair_mask'],method,pairs,diag)
                else:helpers.verify_baseline(submitted,obs['pair_mask'],method,state,pairs)
            else:
                score=np.array([[helpers.urgency(state,m)+helpers.R[m,j]/helpers.D[m,j] for j in range(2)] for m in range(4)])
                if abs(sum(score[m,j] for m,j in pairs)-helpers.top_two_optimum(score,helpers.support(state)))>1e-8:raise RuntimeError('common tail')
            following,components=helpers.primitive_step(state,pairs,ep['events'][tick])
            if following!=json.loads(r['following']) or abs(components['objective']-float(r['cost']))>1e-8:raise RuntimeError('independent primitive transition')
            for k,v in components.items():ep['metrics'][k]+=v
            ep['state']=following;ep['tick']+=1;count+=1
            if tick==47:ep['cutoff']=dict(ep['metrics']);ep['cutoff_state']=following
    for key,ep in states.items():
        r=lookup[key]
        if ep['tick']!=60 or ep['population']!=r['population_sha256'] or any(abs(v-r[k])>1e-8 for k,v in ep['metrics'].items()):raise RuntimeError('independent aggregate')
        pending=sum(ep['cutoff_state']['failed'][m] and m not in ep['cutoff_state']['assigned'] for m in range(4))
        if abs(r['availability']-(1-ep['cutoff']['unavailable_ticks']/192))>1e-10 or r['policy_failures']!=ep['cutoff']['failures'] or r['terminal_pending']!=pending:raise RuntimeError('independent cutoff health')
    if set(states)!=set(lookup) or count!=60*len(episodes) or len(policies)!=48*len(episodes):raise RuntimeError('complete physical grid')
    return dict(status='PASS',episodes=len(episodes),physical_ticks=count,policy_decisions=len(policies))


def summarize(episodes,helpers):
    fields=('normalized_cost','availability','normalized_pending','normalized_failures')
    def metrics(group):return {k:float(np.mean([r['extended_cost']/4 if k=='normalized_cost' else r['terminal_pending']/4 if k=='normalized_pending' else r['policy_failures']/4 if k=='normalized_failures' else r[k] for r in group])) for k in fields}
    means={mode:{a:metrics([r for r in episodes if r['mode']==mode and r['method']==a]) for a in x.ARMS} for mode in ('mean','sample')}
    per={mode:{a:{str(t):metrics([r for r in episodes if r['mode']==mode and r['method']==a and r['train_seed']==t]) for t in x.settings()['train_seeds']} for a in x.ARMS} for mode in means}
    mixed={a:means['sample'][a] if a in x.CORE_ARMS else means['mean'][a] for a in x.ARMS}
    points,gates=helpers.endpoint_gates(mixed)
    gain=1-means['sample'][x.CORE_ARMS[0]]['normalized_cost']/means['mean'][x.CORE_ARMS[0]]['normalized_cost']
    replica_gain={str(t):1-per['sample'][x.CORE_ARMS[0]][str(t)]['normalized_cost']/per['mean'][x.CORE_ARMS[0]][str(t)]['normalized_cost'] for t in x.settings()['train_seeds']}
    checks=dict(own_mean_gain_ge_one_percent=gain>=.01,every_replica_positive=all(v>0 for v in replica_gain.values()),all13_quality_health_gates=all(gates.values()),zero_invalid=all(r['invalid_assignments']==0 for r in episodes))
    repeat_means={a:{str(t):{str(rep):metrics([r for r in episodes if r['mode']=='sample' and r['method']==a and r['train_seed']==t and r['repeat']==rep]) for rep in range(4)} for t in x.settings()['train_seeds']} for a in x.ARMS}
    return dict(classification='POSTHOC_FROZEN_EXECUTION_DIAGNOSTIC',means=means,per_replica=per,action_repeat_means=repeat_means,
        joint_sample_vs_mean_gain=gain,per_replica_gain=replica_gain,mixed_reference_points=points,mixed_reference_gates=gates,engineering_checks=checks,
        stochastic_deployment_trial_motivated=all(checks.values()),goal_pass=False,test_opened=False,training_steps=0,
        limitation='Four action repeats share64physical tapes and3training replicas; no CI or checkpoint reselection. Strong references remain PPO/norm/MAPPO mean policies.')


def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);start=time.monotonic()
    def guard():
        if time.monotonic()-start>900:raise TimeoutError('15min hardstop')
    manifest=dict(status='RUNNING',protocol=VERSION,goal_pass=False,test_opened=False,training_steps=0,source=str(source),source_manifest_sha256=x.sha(source/'manifest.json'),
        git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())))
    x.write_json(out/'manifest.json',manifest)
    try:
        original=json.loads((source/'manifest.json').read_text());c=json.loads((source/'resolved_config.json').read_text());choices=json.loads((source/'checkpoint_choices.json').read_text())
        if manifest['git']['dirty'] or original['status']!='COMPLETED' or original['audit_status']!='PASS' or original['test_opened'] or original['git']['commit']!='690be2d8241b9e5a2625c537e0c684f95674b4d0' or len(original['output_bindings'])!=451 or c!=x.settings():raise RuntimeError('frozen source contract')
        def verify():
            for name,meta in tqdm(original['output_bindings'].items(),desc='451 frozen source bindings',leave=False):
                guard();p=source/name
                if x.sha(p)!=meta['sha256'] or p.stat().st_size!=meta['bytes']:raise RuntimeError('changed source '+name)
            for name,h in original['source_seal'].items():
                if x.sha(source/name)!=h:raise RuntimeError('changed source seal')
                if name.startswith('source_snapshot/') and x.sha(Path(__file__).parent/Path(name).name)!=h:raise RuntimeError('changed local learning source')
            if x.sha(source/'manifest.json')!=manifest['source_manifest_sha256']:raise RuntimeError('changed manifest')
        verify();audit=seed_audit();x.write_json(out/'seed_audit.json',audit);helpers=primitive_helpers()
        for src,name in ((PLAN,'protocol.md'),(REGISTRY,'seed_registry.json'),(Path(__file__),'diagnostic.py'),(Path(helpers.__file__),'primitive_helpers.py')):shutil.copy2(src,out/name)
        all_episodes=[];replay_audits=[];inputs=[];populations={};all_physics=[]
        for a,t in tqdm([(a,t) for a in x.ARMS for t in c['train_seeds']],desc='24 frozen models, five execution batches'):
            choice=choices[f'{a}:{t}'];model=x.load_model(source/choice['path'],a,c,t,choice['steps']);before=x.learner.parameter_digest(model)
            directory=out/a/str(t);directory.mkdir(parents=True);physical=x.Journal(directory/'physical.csv');policy=x.Journal(directory/'policy.csv');episodes=[]
            try:
                for mode,repeat in tqdm([('mean',-1),*(('sample',r) for r in range(4))],desc=f'{a}/{t}',leave=False):
                    role='core' if a in x.CORE_ARMS else 'ppo' if a in ('ppo','normalized_ppo') else 'mappo'
                    raw_seed=audit['action_seeds'][c['train_seeds'].index(t)*4+repeat] if repeat>=0 else None
                    action_seed=x.learner.role_seed(raw_seed,f'{VERSION}:{role}:actions') if raw_seed is not None else None
                    generator=torch.Generator().manual_seed(action_seed) if action_seed is not None else None
                    collected,replay=collect(model,a,audit['physical_seeds'],mode,generator,t,repeat,physical,policy,guard)
                    episodes.extend(collected);replay_audits.append(dict(method=a,train_seed=t,mode=mode,repeat=repeat,action_seed=action_seed,behavior_replay=replay))
            finally:physical.close();policy.close()
            if before!=x.learner.parameter_digest(model) or any(p.grad is not None for p in model.parameters()):raise RuntimeError('frozen weights mutated')
            if len(episodes)!=320 or len({(r['mode'],r['repeat'],r['seed']) for r in episodes})!=320:raise RuntimeError('mode/repeat seed grid')
            for r in episodes:
                if populations.setdefault(r['seed'],r['population_sha256'])!=r['population_sha256']:raise RuntimeError('unpaired physical tape')
            x.write_csv(directory/'episodes.csv',episodes);all_physics.append(dict(method=a,train_seed=t,**audit_physics(directory,episodes,helpers,guard)))
            inputs.append(dict(method=a,train_seed=t,choice=choice,parameter_sha256=before));all_episodes.extend(episodes)
        if len(all_episodes)!=7680 or len(populations)!=64:raise RuntimeError('full fixed budget')
        summary=summarize(all_episodes,helpers);x.write_csv(out/'episodes.csv',all_episodes);x.write_json(out/'inputs.json',inputs);x.write_json(out/'behavior_replay.json',replay_audits);x.write_json(out/'physical_audit.json',all_physics);x.write_json(out/'summary.json',summary)
        (out/'report.md').write_text('# Frozen execution modes, diagnostic only\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
        verify();manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,episodes=7680,policy_decisions=368640,physical_ticks=460800,
            output_bindings={str(p.relative_to(out)):dict(sha256=x.sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'})
        x.write_json(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),stochastic_deployment_trial_motivated=summary['stochastic_deployment_trial_motivated'],joint_sample_vs_mean_gain=summary['joint_sample_vs_mean_gain'])));return manifest
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);x.write_json(out/'manifest.json',manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.source,a.output_dir)

if __name__=='__main__':main()
