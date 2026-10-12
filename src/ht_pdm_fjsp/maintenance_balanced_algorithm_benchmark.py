"""Locked lab comparison of five controllers and four learned algorithms.

Mac smoke is engineering evidence only. Full execution is handed to the user.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_cross_skill_mac as engine
from ht_pdm_fjsp import maintenance_skill_contention_screen as screen
from ht_pdm_fjsp.maintenance_balanced_confirmation_mac import registered_integers
from ht_pdm_fjsp import maintenance_balanced_algorithms as learners
from ht_pdm_fjsp.maintenance_balanced_benchmark_env import BenchmarkEnv, VERSION, fixed_config, observation, arbitrate
from ht_pdm_fjsp import maintenance_balanced_benchmark_statistics as statistics
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_opportunity_cost import Journal, write_csv, write_json

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
ALGORITHMS=learners.ALGORITHMS
BASELINES=statistics.BASELINES


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def settings(profile):
    if profile not in ('smoke','full'): raise ValueError('unknown profile')
    raw=json.loads(REGISTRY.read_text())['panels'][profile]; full=profile=='full'
    return dict(profile=profile,train_seeds=raw['train_seeds'],
        **{f'{p}_seeds':list(range(raw[f'{p}_start'],raw[f'{p}_start']+raw[f'{p}_count'])) for p in ('development','test')},
        analysis_seed=raw['analysis_seed'],forecast_seed=raw['forecast_seed'],
        env_steps=491520 if full else 384, vector_envs=16 if full else 2,
        checkpoints=[122880,245760,368640,491520] if full else [192,384],
        hidden=64,mixer_hidden=32,gamma=1.,learning_rate=.0003,rollout=3072 if full else 192,
        epochs=4 if full else 2,minibatch=256 if full else 48,clip=.2,
        entropy_weight=.01,value_weight=.5,gradient_clip=.5,reward_scale=20.,
        replay_capacity=65536 if full else 512,learning_starts=3072 if full else 96,
        batch_size=128 if full else 32,train_frequency=16 if full else 2,
        target_interval=3072 if full else 96,epsilon_start=1.,epsilon_end=.05,epsilon_fraction=.5,
        scenarios=128 if full else 8,bootstrap_samples=40000 if full else 200,
        confidence=1-.05/28,max_seconds=86400 if full else 1200,torch_threads=1,audit_seeds=2)


def seed_audit(cfg):
    if cfg!=settings(cfg['profile']): raise ValueError('locked protocol configuration')
    historical=set(); hashes={}; groups=[]; derived=[]
    for path in sorted((ROOT/'configs').glob('*seed_registry.json')):
        if path!=REGISTRY:
            historical|=registered_integers(json.loads(path.read_text())); hashes[path.name]=sha(path)
    for p in ('smoke','full'):
        c=settings(p)
        groups.extend(set(c[k]) for k in ('train_seeds','development_seeds','test_seeds'))
        groups.extend(({c['analysis_seed']},{c['forecast_seed']}))
        derived.extend(role_seed(s,f'{VERSION}:physical') for k in ('development_seeds','test_seeds') for s in c[k])
        episodes=c['env_steps']//(48*c['vector_envs'])
        derived.extend(role_seed(s,f'{VERSION}:training_physical:worker:{w}:episode:{e}')
            for s in c['train_seeds'] for w in range(c['vector_envs']) for e in range(episodes))
        derived.extend(engine.bank_seed(c,'confirmation',s,t) for s in c['test_seeds'] for t in range(48))
        derived.extend(learners.role_seed(s,f'initialization:{a}') for s in c['train_seeds'] for a in ALGORITHMS)
        derived.extend(learners.role_seed(s,r) for s in c['train_seeds'] for r in ('actions','minibatches','replay'))
        derived.append(learners.role_seed(c['analysis_seed'],'crossed_bootstrap'))
    own=set.union(*groups)
    if any(a&b for a,b in itertools.combinations(groups,2)) or own&historical:
        raise ValueError('raw seed roles overlap history or one another')
    if len(derived)!=len(set(derived)) or set(derived)&(own|historical): raise ValueError('derived seed overlap')
    return dict(status='PASS',raw_unique=len(own),derived_unique=len(derived),historical_registry_hashes=hashes,
        intentional_pairing='same physical train worker/episode tape across algorithms; same evaluation tape across all controllers')


def make_model(algorithm,cfg,seed):
    with torch.random.fork_rng():
        torch.manual_seed(learners.role_seed(seed,f'initialization:{algorithm}'))
        return learners.make_model(algorithm,cfg['hidden'],cfg['mixer_hidden'])


def checkpoint(model,path,cfg,seed,steps,probe):
    if path.exists(): raise FileExistsError('checkpoint overwrite')
    torch.save(dict(protocol=VERSION,algorithm=model.algorithm,train_seed=seed,steps=steps,
        hidden=cfg['hidden'],mixer_hidden=cfg['mixer_hidden'],config=asdict(fixed_config()),
        parameter_sha256=learners.parameter_digest(model),state_dict=model.state_dict()),path)
    loaded=load_model(path,cfg,model.algorithm,seed,steps)
    if not np.array_equal(model.select(probe),loaded.select(probe)):
        raise RuntimeError('checkpoint prediction roundtrip failed')
    return dict(path=str(path),sha256=sha(path),parameter_sha256=learners.parameter_digest(model))


def load_model(path,cfg,algorithm,seed,steps):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    expected=dict(protocol=VERSION,algorithm=algorithm,train_seed=seed,steps=steps,
        hidden=cfg['hidden'],mixer_hidden=cfg['mixer_hidden'],config=asdict(fixed_config()))
    if any(payload.get(k)!=v for k,v in expected.items()): raise RuntimeError('checkpoint contract mismatch')
    model=make_model(algorithm,cfg,seed); model.load_state_dict(payload['state_dict'],strict=True); model.eval()
    if learners.parameter_digest(model)!=payload['parameter_sha256']: raise RuntimeError('checkpoint parameter hash mismatch')
    return model


class TaggedJournal:
    def __init__(self,journal,tags): self.journal=journal; self.tags=tags
    def add(self,row): self.journal.add(dict(self.tags,**row))


def evaluate_episode(method,train_seed,seed,cfg,model=None,journal=None,coordination=None,timeout=None):
    tags=dict(method=method,train_seed=train_seed,seed=seed)
    env=BenchmarkEnv(seed,journal=TaggedJournal(journal,tags) if journal else None)
    obs=env.reset(); wall=0.; forecasts=0; conflicts=0; rejections=0; unused=0; reward_sum=0.
    for tick in range(48):
        if timeout: timeout()
        started=time.perf_counter()
        if model is not None:
            submitted=model.select(obs); diagnostic=dict(planned=False,forecast_transitions=0)
        else:
            action,diagnostic=engine.select(env.inner.config,env.inner.env.state,tick,method,
                'isolated_h6',cfg,seed,'confirmation',timeout)
            submitted=env.inner.actions.index(action)
        wall+=time.perf_counter()-started; forecasts+=diagnostic['forecast_transitions']
        obs_hash=hashlib.sha256(obs['state'].tobytes()).hexdigest()
        obs,reward,done,info=env.step(submitted); reward_sum+=reward
        diag=info['coordination']; conflicts+=diag['conflict_workers']; rejections+=diag['rejected_proposals']; unused+=diag['unused_feasible_capacity']
        if coordination:
            coordination.add(dict(tags,tick=tick,observation_sha256=obs_hash,
                submitted=json.dumps(int(submitted) if np.isscalar(submitted) else np.asarray(submitted).tolist()),
                coordination=json.dumps(diag,sort_keys=True),planner=json.dumps(diagnostic,sort_keys=True)))
    if not done or abs(-20*reward_sum-env.inner.episode_metrics['extended_cost'])>1e-6:
        raise RuntimeError('complete H+L reward reconciliation failed')
    metrics=env.inner.episode_metrics
    if metrics['invalid_assignments']!=0: raise RuntimeError('invalid assignment')
    return dict(tags,**metrics,decision_seconds=wall,forecast_transitions=forecasts,
        conflict_workers=conflicts,rejected_proposals=rejections,unused_feasible_capacity=unused)


def audit_training(path,cfg,audits):
    expected_steps=cfg['env_steps']; expected_episodes=expected_steps//48
    if len(audits)!=4*len(cfg['train_seeds']): raise RuntimeError('training audit population')
    for a in audits:
        if a['learned_steps']!=expected_steps or a['physical_ticks']!=expected_episodes*60 or a['training_episodes']!=expected_episodes or not a['changed'] or a['partial_episode_steps']:
            raise RuntimeError('training counts or update audit')
    paired={}; seen=set(); count=0
    for r in engine.records(path):
        algorithm=r['algorithm']; seed=int(r['train_seed']); worker=int(r['worker']); episode=int(r['episode_index'])
        key=seed,worker,episode; identity=key+(algorithm,)
        if identity in seen: raise RuntimeError('duplicate training episode')
        seen.add(identity)
        physical=role_seed(seed,f'{VERSION}:training_physical:worker:{worker}:episode:{episode}')
        tape=(physical,r['population_sha256'])
        if int(r['physical_seed'])!=physical or paired.setdefault(key,tape)!=tape:
            raise RuntimeError('training physical population pairing')
        if abs(float(r['episode_reward'])*20+float(r['extended_cost']))>1e-5:
            raise RuntimeError('training reward tail reconciliation')
        count+=1
    expected={(s,w,e,a) for s in cfg['train_seeds'] for w in range(cfg['vector_envs'])
        for e in range(expected_steps//(48*cfg['vector_envs'])) for a in ALGORITHMS}
    if seen!=expected or count!=4*len(cfg['train_seeds'])*expected_episodes: raise RuntimeError('training episode grid')
    return dict(status='PASS',episodes=count,paired_populations=len(paired),physical_ticks=count*60)


def replay_test(out,cfg,rows,choices,timeout):
    lookup={(r['method'],r['train_seed'],r['seed']):r for r in rows}
    statistic_arrays=statistics.arrays(rows,cfg['train_seeds'],cfg['test_seeds'])
    del statistic_arrays
    coord=iter(engine.records(out/'coordination.csv')); models={}; seen=set(); ticks=0; forecasts=0
    active=None; costs=defaultdict(float); coord_sums=defaultdict(int)
    for r in tqdm(engine.records(out/'testdecisions.csv'),desc='Audit physical test trajectories',unit='tick'):
        timeout(); key=r['method'],int(r['train_seed']),int(r['seed']); tick=int(r['tick'])
        if tick==0:
            if active is not None or key in seen or key not in lookup: raise RuntimeError('trajectory start grid')
            seen.add(key); env=BenchmarkEnv(key[2]); env.reset(); active=key; costs=defaultdict(float); coord_sums=defaultdict(int)
        if active!=key or env.inner.env.time!=tick: raise RuntimeError('trajectory order')
        physical=role_seed(key[2],f'{VERSION}:physical')
        state=env.inner.env.state; action=tuple(map(tuple,json.loads(r['action'])))
        if int(r['physical_seed'])!=physical or json.loads(r['state'])!=json.loads(json.dumps(asdict(state))) or tuple(json.loads(r['events']))!=env.inner.env.events[tick]:
            raise RuntimeError('physical seed/state/shock replay')
        validate_matching(env.inner.config,state,action)
        if action not in engine.f.old.screen.action_space(env.inner.config,state)[2]: raise RuntimeError('illegal executed root')
        bounded=key[2] in cfg['test_seeds'][:cfg['audit_seeds']]
        if tick<48:
            cr=next(coord)
            if (cr['method'],int(cr['train_seed']),int(cr['seed']),int(cr['tick']))!=(*key,tick): raise RuntimeError('coordination grid')
            obs=observation(env.inner); submitted=json.loads(cr['submitted']); diagnostic=json.loads(cr['planner'])
            if cr['observation_sha256']!=hashlib.sha256(obs['state'].tobytes()).hexdigest(): raise RuntimeError('observation reconstruction')
            if key[0] in ('mappo','iql','qmix'):
                if not isinstance(submitted,list): raise RuntimeError('local proposal action contract')
            elif type(submitted) is not int or submitted not in range(21) or env.inner.actions[submitted]!=action:
                raise RuntimeError('central/baseline action contract')
            if isinstance(submitted,list):
                proposed_action,expected_coord=arbitrate(env.inner,np.asarray(submitted))
                if proposed_action!=action: raise RuntimeError('proposal arbitration replay')
            else:
                capacity=max(map(len,(a for a,ok in zip(env.inner.actions,env.inner.action_masks()) if ok)))
                expected_coord=dict(proposals=None,conflict_workers=0,rejected_proposals=0,
                    proposed_starts=len(action),executed_starts=len(action),unused_feasible_capacity=capacity-len(action))
            if expected_coord!=json.loads(cr['coordination']): raise RuntimeError('coordination counters replay')
            if key[0] in ALGORITHMS:
                if diagnostic!=dict(planned=False,forecast_transitions=0): raise RuntimeError('learned forecast contract')
                if bounded:
                    model_key=key[:2]
                    if model_key not in models:
                        choice=choices[f'{key[0]}:{key[1]}']
                        models[model_key]=load_model(out/choice['path'],cfg,key[0],key[1],choice['steps'])
                    if not np.array_equal(models[model_key].select(obs),submitted): raise RuntimeError('deterministic checkpoint decision replay')
            else:
                if key[0] in engine.CHEAP or bounded:
                    expected,expected_diag=engine.select(env.inner.config,state,tick,key[0],'isolated_h6',cfg,key[2],'confirmation',timeout)
                    if expected!=action or expected_diag!=diagnostic: raise RuntimeError('baseline/forecast replay')
                    forecasts+=expected_diag['forecast_transitions']
                else:
                    pool=engine.root_pool(env.inner.config,state,tick,key[0],'isolated_h6')
                    if action not in pool: raise RuntimeError('planner root pool')
                    if len(pool)==1:
                        if diagnostic!=dict(planned=False,forecast_transitions=0): raise RuntimeError('singleton forecast contract')
                    else:
                        index=diagnostic['selected_root']
                        if (type(index) is not int or index not in range(len(pool)) or not diagnostic['planned']
                            or diagnostic['selection_seed']!=engine.bank_seed(cfg,'confirmation',key[2],tick)
                            or diagnostic['candidate_count']!=len(pool) or diagnostic['forecast_transitions']!=len(pool)*cfg['scenarios']*(60-tick)
                            or pool[index]!=action or not np.isfinite([diagnostic['selected_mean'],diagnostic['baseline_mean']]).all()
                            or diagnostic['selected_mean']>diagnostic['baseline_mean']+1e-9): raise RuntimeError('planner bank contract')
            cs=json.loads(cr['coordination'])
            for field in ('conflict_workers','rejected_proposals','unused_feasible_capacity'): coord_sums[field]+=cs[field]
        elif action!=screen.choose(env.inner.config,state,'deadline_matching'):
            raise RuntimeError('common tail policy replay')
        _,_,_,info=env.inner.env.step(action)
        if json.loads(r['following'])!=json.loads(json.dumps(asdict(env.inner.env.state))) or json.loads(r['info'])!=info or abs(float(r['cost'])-info['objective'])>1e-8:
            raise RuntimeError('scalar physical transition/cost replay')
        if tick<48:
            for field,value in info.items(): costs[field]+=value
        if tick==47:
            cutoff_pending=screen.pending(env.inner.config,env.inner.env.state)
            cutoff_failed=sum(env.inner.env.state.failed)
            if cutoff_pending!=lookup[key]['terminal_pending']: raise RuntimeError('H backlog replay')
        if tick==59:
            row=lookup[key]
            checked=dict(env.inner.env.metrics,extended_cost=env.inner.env.metrics['objective'],policy_cost=costs['objective'],
                tail_cost=env.inner.env.metrics['objective']-costs['objective'],availability=1-costs['unavailable_ticks']/192,
                technician_utilization=costs['service_ticks']/96,terminal_pending=cutoff_pending,terminal_failed=cutoff_failed,
                policy_failures=costs['failures'],policy_overdue_ticks=costs['overdue_waiting_ticks'],
                tail_pending=screen.pending(env.inner.config,env.inner.env.state),tail_failed=sum(env.inner.env.state.failed),
                physical_ticks=60,physical_seed=physical,raw_seed=key[2],episode_index=0,worker=0,**coord_sums)
            if row['stream_seed'] is not None: raise RuntimeError('evaluation training-stream contamination')
            if row['population_sha256']!=env.initial_population or any(abs(float(row[k])-v)>1e-8 for k,v in checked.items()): raise RuntimeError('episode aggregate replay')
            active=None
        ticks+=1; env.inner.policy_time=min(48,tick+1)
    if active or seen!=set(lookup) or ticks!=len(rows)*60 or next(coord,None) is not None: raise RuntimeError('completed physical/coordination grids')
    return dict(status='PASS',test_episodes=len(rows),physical_ticks=ticks,policy_records=len(rows)*48,
        invalid_assignments=0,bounded_forecast_transitions_rebuilt=forecasts,
        checkpoint_replayed_seeds=cfg['test_seeds'][:cfg['audit_seeds']])


def run(profile,output_dir,device='cpu'):
    if device!='cpu': raise ValueError('CPU profile only')
    cfg=settings(profile); seeds=seed_audit(cfg); git=engine.f.old.screen.git_metadata()
    out=Path(output_dir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())): raise FileExistsError('refusing prior artifact overwrite')
    if profile=='full' and git['dirty']: raise RuntimeError('full requires clean committed code')
    out.mkdir(parents=True,exist_ok=True); started=time.monotonic(); torch.set_num_threads(1)
    def timeout():
        if time.monotonic()-started>cfg['max_seconds']: raise TimeoutError('registered hard limit; incomplete run, not algorithm performance')
    manifest=dict(protocol=VERSION,status='RUNNING',profile=profile,git=git,started_utc=datetime.now(timezone.utc).isoformat(),
        runtime=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device=device),
        test_outcomes_opened=False,sealed_full_test_opened=False,stage='initialization')
    write_json(out/'manifest.json',manifest)
    opened=[]
    try:
        write_json(out/'benchmark_config.json',asdict(fixed_config())); write_json(out/'resolved_config.json',cfg)
        write_json(out/'seed_audit.json',seeds); shutil.copy2(PLAN,out/'protocol.md'); shutil.copy2(REGISTRY,out/'registry.json')
        snap=out/'source_snapshot'; snap.mkdir()
        for path in Path(__file__).parent.glob('*.py'): shutil.copy2(path,snap/path.name)
        history=out/'registry_snapshot'; history.mkdir()
        for name in seeds['historical_registry_hashes']: shutil.copy2(ROOT/'configs'/name,history/name)
        immutable=[out/'protocol.md',out/'registry.json',out/'benchmark_config.json',out/'resolved_config.json',*snap.iterdir(),*history.iterdir()]
        seals={str(p.relative_to(out)):sha(p) for p in immutable}; manifest['source_seal']=seals
        write_json(out/'manifest.json',manifest)
        journals={name:Journal(out/f'{name}.csv') for name in ('training_progress','training_episodes','development_episodes','episodes.partial','testdecisions','coordination')}
        opened.extend(journals.values()); audits=[]; devrows=[]; choices={}; parameters={}
        probe=BenchmarkEnv(cfg['development_seeds'][0]); probe_obs=probe.reset()
        for algorithm,train_seed in itertools.product(ALGORITHMS,cfg['train_seeds']):
            timeout(); manifest.update(stage=f'train:{algorithm}:{train_seed}'); write_json(out/'manifest.json',manifest)
            model=make_model(algorithm,cfg,train_seed); parameters[algorithm]=sum(p.numel() for p in model.parameters())
            directory=out/algorithm/str(train_seed)/'checkpoints'; directory.mkdir(parents=True)
            candidates=[]; checkpoint_eval_seconds=0.
            def assess(current,steps,audit=None):
                nonlocal checkpoint_eval_seconds
                before=time.monotonic(); path=directory/f'step_{steps:09d}.pt'
                metadata=checkpoint(current,path,cfg,train_seed,steps,probe_obs)
                metadata['path']=str(path.relative_to(out))
                rows=[]; current.eval()
                for seed in tqdm(cfg['development_seeds'],desc=f'Development {algorithm}/{train_seed} step {steps}',unit='episode'):
                    row=evaluate_episode(algorithm,train_seed,seed,cfg,current,timeout=timeout)
                    row['checkpoint_steps']=steps; journals['development_episodes'].add(row); devrows.append(row); rows.append(row)
                current.train()
                if steps: candidates.append(dict(metadata,steps=steps,development_cost=float(np.mean([r['extended_cost'] for r in rows]))))
                checkpoint_eval_seconds+=time.monotonic()-before
            assess(model,0)
            initial_eval_seconds=checkpoint_eval_seconds
            before=time.monotonic()
            audit=learners.fit(model,cfg,train_seed,
                lambda seed,worker: BenchmarkEnv(seed,worker,training=True),journals,assess,timeout)
            audit['training_seconds_including_callbacks']=time.monotonic()-before
            # Initial diagnostic is outside fit wall time.
            audit['development_callback_seconds_including_initial']=checkpoint_eval_seconds
            audit['training_seconds_excluding_callbacks']=audit['training_seconds_including_callbacks']-(checkpoint_eval_seconds-initial_eval_seconds)
            audits.append(audit); write_json(out/'training_audit.json',audits)
            choice=min(candidates,key=lambda c:(c['development_cost'],c['steps']))
            choices[f'{algorithm}:{train_seed}']=choice
            write_json(out/'checkpoint_choices.partial.json',choices)
        journals['training_progress'].close(); opened.remove(journals['training_progress'])
        journals['training_episodes'].close(); opened.remove(journals['training_episodes'])
        training_gate=audit_training(out/'training_episodes.csv',cfg,audits)
        expected_dev={(a,t,step,s) for a in ALGORITHMS for t in cfg['train_seeds'] for step in [0,*cfg['checkpoints']] for s in cfg['development_seeds']}
        if len(devrows)!=len(expected_dev) or {(r['method'],r['train_seed'],r['checkpoint_steps'],r['seed']) for r in devrows}!=expected_dev:
            raise RuntimeError('development checkpoint grid')
        write_json(out/'checkpoint_choices.json',choices); write_json(out/'model_parameters.json',parameters)
        choices_hash=sha(out/'checkpoint_choices.json')
        for choice in choices.values():
            if sha(out/choice['path'])!=choice['sha256']: raise RuntimeError('sealed checkpoint mismatch')
        manifest.update(stage='test',test_outcomes_opened=True,sealed_full_test_opened=profile=='full',checkpoint_choices_sha256=choices_hash,
            choices_sealed_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/'manifest.json',manifest); rows=[]
        controllers=[(a,t) for a in ALGORITHMS for t in cfg['train_seeds']]+[(b,-1) for b in BASELINES]
        for method,train_seed in tqdm(controllers,desc='Test controllers',unit='controller'):
            choice=choices.get(f'{method}:{train_seed}')
            model=load_model(out/choice['path'],cfg,method,train_seed,choice['steps']) if choice else None
            for seed in tqdm(cfg['test_seeds'],desc=f'Test {method}/{train_seed}',unit='episode'):
                row=evaluate_episode(method,train_seed,seed,cfg,model,journals['testdecisions'],journals['coordination'],timeout)
                rows.append(row); journals['episodes.partial'].add(row)
        for journal in opened: journal.close()
        opened.clear(); write_csv(out/'episodes.csv',rows)
        manifest['stage']='audits'; write_json(out/'manifest.json',manifest)
        audit=replay_test(out,cfg,rows,choices,timeout); audit['training']=training_gate; audit['development_episodes']=len(devrows)
        if any(sha(out/key)!=value for key,value in seals.items()) or sha(out/'checkpoint_choices.json')!=choices_hash:
            raise RuntimeError('source/choice seals changed')
        if any(sha(out/c['path'])!=c['sha256'] for c in choices.values()): raise RuntimeError('checkpoint seal changed')
        if sha(out/'episodes.csv')!=sha(out/'episodes.partial.csv'): raise RuntimeError('partial/final episode identity')
        if sha(PLAN)!=seals['protocol.md'] or sha(REGISTRY)!=seals['registry.json'] or any(sha(Path(__file__).parent/p.name)!=seals[f'source_snapshot/{p.name}'] for p in snap.iterdir()):
            raise RuntimeError('working source changed during run')
        if any(sha(ROOT/'configs'/name)!=value for name,value in seeds['historical_registry_hashes'].items()):
            raise RuntimeError('historical registry changed during run')
        write_json(out/'audit.json',audit); timeout()
        summary=statistics.summarize(rows,cfg); write_json(out/'summary.json',summary); timeout()
        manifest.update(status='COMPLETED',stage='complete',elapsed_seconds=time.monotonic()-started,
            completed_utc=datetime.now(timezone.utc).isoformat(),audit_status='PASS',training_models=len(audits),
            training_learned_steps=sum(a['learned_steps'] for a in audits),training_physical_ticks=sum(a['physical_ticks'] for a in audits),
            development_episodes=len(devrows),test_episodes=len(rows),output_files=sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()))
        manifest['output_bindings']={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size)
            for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        write_json(out/'manifest.json',manifest)
        print(json.dumps(dict(status='COMPLETED',profile=profile,output_dir=str(out),elapsed_seconds=manifest['elapsed_seconds'],
            training_models=len(audits),test_episodes=len(rows),audit='PASS',sealed_full_test_opened=manifest['sealed_full_test_opened'])))
        return manifest
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-started)
        write_json(out/'manifest.json',manifest); raise
    finally:
        for journal in opened: journal.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('smoke','full'),required=True)
    parser.add_argument('--device',choices=('cpu',),default='cpu')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(); run(args.profile,args.output_dir,args.device)


if __name__=='__main__': main()
