"""Fixed-budget masked PPO probe; physics selection remains planning evidence."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime,timezone
import itertools
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
import random
import shutil
import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_joint_frontier_mac as frontier
from ht_pdm_fjsp.maintenance_joint_learning_env import JointLearningEnv
from ht_pdm_fjsp.maintenance_opportunity_cost import Journal,write_csv,write_json

VERSION='maintenance_joint_learning_mac_v1'
ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
SOURCES=tuple(dict.fromkeys(('maintenance_joint_learning_mac.py','maintenance_joint_learning_env.py',*frontier.SOURCES)))
BASELINES=(*frontier.CHEAP,'joint_rollout')


def settings(profile):
    if profile not in ('smoke','full'):raise ValueError('unknown profile')
    raw=json.loads(REGISTRY.read_text())['panels'][profile];full=profile=='full'
    return dict(profile=profile,horizon=48,tail_ticks=12,train_seeds=raw['train_seeds'],
        development_seeds=list(range(raw['development_start'],raw['development_start']+raw['development_count'])),
        test_seeds=list(range(raw['test_start'],raw['test_start']+raw['test_count'])),analysis_seed=raw['analysis_seed'],forecast_seed=raw['forecast_seed'],
        steps=65536 if full else 512,envs=8 if full else 2,n_steps=256 if full else 32,batch_size=128 if full else 32,epochs=4 if full else 2,
        checkpoints=[16384,32768,49152,65536] if full else [128,256,384,512],scenarios=128 if full else 8,
        bootstrap_samples=20000 if full else 200,max_seconds=7200 if full else 600,audit_seeds=2,
        candidate=frontier.old.CANDIDATE,baseline_continuation='deadline_matching')


def raw_audit(cfg):
    if cfg!=settings(cfg['profile']):raise ValueError('locked settings changed')
    history=set();hashes={}
    for p in sorted((ROOT/'configs').glob('*seed_registry.json')):
        if p!=REGISTRY:history|=frontier.old.integers(json.loads(p.read_text()));hashes[p.name]=frontier.old.screen.sha(p)
    groups=[]
    for profile in ('smoke','full'):
        c=settings(profile);groups += [set(c['train_seeds']),set(c['development_seeds']),set(c['test_seeds']),{c['analysis_seed']},{c['forecast_seed']}]
    own=set.union(*groups)
    if any(a&b for a,b in itertools.combinations(groups,2)) or own&history:raise ValueError('raw panels overlap')
    evaluation=[frontier.role_seed(seed,f'{VERSION}:physical') for profile in ('smoke','full') for phase in ('development_seeds','test_seeds') for seed in settings(profile)[phase]]
    if len(evaluation)!=len(set(evaluation)) or set(evaluation)&(own|history):raise ValueError('derived evaluation seed overlap')
    return dict(status='PASS',historical_registry_hashes=hashes,raw_unique=len(own),derived_evaluation_unique=len(evaluation))


def c_for(cfg):return frontier.make_config(cfg)


@contextmanager
def preserve_rng():
    python_state,numpy_state,torch_state=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    try:yield
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state);torch.set_rng_state(torch_state)


def save_model(path,model,cfg,seed,steps):
    with preserve_rng():
        path=Path(path)
        if path.exists():raise FileExistsError('checkpoint overwrite refused')
        model.save(path)
        restored=MaskablePPO.load(path,device='cpu')
        for name,value in model.policy.state_dict().items():
            if not torch.equal(value,restored.policy.state_dict()[name]):raise RuntimeError('checkpoint parameter mismatch')
        env=JointLearningEnv(c_for(cfg),VERSION);obs,_=env.reset(seed=cfg['development_seeds'][0]);mask=env.action_masks()
        if not np.array_equal(model.predict(obs,action_masks=mask,deterministic=True)[0],restored.predict(obs,action_masks=mask,deterministic=True)[0]):raise RuntimeError('checkpoint masked prediction mismatch')
        write_json(path.with_suffix('.json'),dict(protocol=VERSION,train_seed=seed,learned_steps=steps,settings=cfg,checkpoint_sha256=frontier.old.screen.sha(path)))
        return restored

class Sink:
    def __init__(self,journal,identity):self.journal=journal;self.identity=identity
    def add(self,row):self.journal.add({**self.identity,**row})


def evaluation(cfg,controller,seeds,model=None,train_seed=-1,checkpoint_steps=-1,journal=None,timeout=None):
    c=c_for(cfg);rows=[]
    for seed in tqdm(seeds,desc=f'Evaluate {controller} seed{train_seed} steps{checkpoint_steps}',unit='episode',leave=False):
        env=JointLearningEnv(c,VERSION,journal=Sink(journal,dict(controller=controller,train_seed=train_seed,checkpoint_steps=checkpoint_steps,seed=seed)) if journal else None)
        obs,_=env.reset(seed=seed);spent=0;latency=0.
        for tick in range(48):
            if timeout:timeout()
            before=time.perf_counter()
            if model is not None:
                index=int(model.predict(obs,action_masks=env.action_masks(),deterministic=True)[0])
            else:
                action,diag=frontier.select(c,env.env.state,tick,controller,'deadline_matching',cfg,seed,timeout)
                index=env.actions.index(action);spent+=diag['forecast_transitions']
            latency+=time.perf_counter()-before
            obs,reward,done,truncated,info=env.step(index)
            if done!=(tick==47) or truncated:raise RuntimeError('evaluation episode boundary')
        rows.append(dict(controller=controller,train_seed=train_seed,checkpoint_steps=checkpoint_steps,seed=seed,
            **env.episode_metrics,forecast_transitions=spent,planning_seconds=latency))
    return rows


def checkpoint_choice(records):
    if not records or any(r['steps']<=0 for r in records):raise ValueError('trained checkpoints only')
    return min(records,key=lambda r:(r['development_cost'],r['steps']))


class Monitor(BaseCallback):
    def __init__(self,cfg,seed,out,journals,timeout):
        super().__init__();self.cfg,self.seed,self.out,self.journals,self.timeout=cfg,seed,out,journals,timeout
        self.records=[];self.progress=tqdm(total=cfg['steps'],desc=f'Train masked PPO seed{seed}',unit='step');self.last=0;self.costs=[];self.completed=0;self.started=time.perf_counter()
    def checkpoint(self):
        step=self.num_timesteps
        if step not in self.cfg['checkpoints'] or any(r['steps']==step for r in self.records):return
        self.timeout();path=self.out/f'checkpoint_{step:06d}.zip';restored=save_model(path,self.model,self.cfg,self.seed,step)
        rows=evaluation(self.cfg,'ppo',self.cfg['development_seeds'],restored,self.seed,step,timeout=self.timeout)
        for row in rows:self.journals['development_episodes'].add(row)
        self.records.append(dict(steps=step,path=str(path),development_cost=float(np.mean([r['extended_cost'] for r in rows])),completed_optimization_epochs=self.model._n_updates,completed_rollout_updates=step//(self.cfg['envs']*self.cfg['n_steps'])))
    def _on_rollout_start(self):
        self.checkpoint()
        if self.num_timesteps:
            stats={k:float(v) for k,v in self.model.logger.name_to_value.items() if k.startswith('train/') and isinstance(v,(int,float,np.floating))}
            if not all(np.isfinite(v) for v in stats.values()):raise RuntimeError('nonfinite PPO optimization')
            self.journals['training_progress'].add(dict(train_seed=self.seed,learned_steps=self.num_timesteps,completed_episodes=self.completed,
                recent_episode_cost=float(np.mean(self.costs[-64:])) if self.costs else 0,elapsed_seconds=time.perf_counter()-self.started,statistics=frontier.old.compact(stats)))
    def _on_step(self):
        self.timeout();self.progress.update(self.num_timesteps-self.last);self.last=self.num_timesteps
        for info in self.locals['infos']:
            if 'episode_metrics' in info:
                metrics=info['episode_metrics'];self.costs.append(metrics['extended_cost']);self.completed+=1
                self.journals['training_episodes'].add(dict(train_seed=self.seed,learned_steps=self.num_timesteps,**metrics))
                if metrics['invalid_assignments'] or metrics['physical_ticks']!=60:raise RuntimeError('training feasibility/episode count')
        return True
    def _on_training_end(self):
        self.checkpoint();self.progress.close()
        self.journals['training_progress'].add(dict(train_seed=self.seed,learned_steps=self.num_timesteps,completed_episodes=self.completed,
            recent_episode_cost=float(np.mean(self.costs[-64:])),elapsed_seconds=time.perf_counter()-self.started,
            statistics=frontier.old.compact({k:float(v) for k,v in self.model.logger.name_to_value.items() if k.startswith('train/') and isinstance(v,(int,float,np.floating))})))


def summarize(cfg,rows):
    names=[('ppo',s) for s in cfg['train_seeds']]+[('initial',s) for s in cfg['train_seeds']]+[(m,-1) for m in BASELINES]
    lookup={(r['controller'],r['train_seed'],r['seed']):r for r in rows};expected={(m,t,s) for m,t in names for s in cfg['test_seeds']}
    if len(rows)!=len(expected) or set(lookup)!=expected:raise RuntimeError('test episode grid')
    fields=('extended_cost','availability','terminal_pending')
    matrix=lambda name:np.asarray([[[lookup[name,s,seed][k] for k in fields] for seed in cfg['test_seeds']] for s in cfg['train_seeds']])
    learned,initial=matrix('ppo'),matrix('initial');base=np.asarray([[lookup['deadline_matching',-1,seed][k] for k in fields] for seed in cfg['test_seeds']])
    rng=np.random.default_rng(frontier.role_seed(cfg['analysis_seed'],f'{VERSION}:bootstrap'));nr,ne=learned.shape[:2]
    replica=rng.integers(nr,size=(cfg['bootstrap_samples'],nr));episode=rng.integers(ne,size=(cfg['bootstrap_samples'],ne))
    arrays_=[]
    for r,e in zip(replica,episode,strict=True):
        arrays_.append((learned[r[:,None],e].mean((0,1)),initial[r[:,None],e].mean((0,1)),base[e].mean(0)))
    a,b,c=np.asarray(arrays_).transpose(1,0,2);point=learned.mean((0,1));initial_point=initial.mean((0,1));base_point=base.mean(0)
    dist=dict(gain_vs_initial=1-a[:,0]/b[:,0],gain_vs_deadline=1-a[:,0]/c[:,0],availability_difference=a[:,1]-c[:,1],pending_difference=a[:,2]-c[:,2])
    estimates=dict(gain_vs_initial=1-point[0]/initial_point[0],gain_vs_deadline=1-point[0]/base_point[0],availability_difference=point[1]-base_point[1],pending_difference=point[2]-base_point[2])
    contrasts={k:dict(estimate=float(estimates[k]),ci95=np.quantile(v,[.025,.975]).tolist(),scope='exploratory crossed train-replica/test-seed bootstrap; four uncorrected comparisons') for k,v in dist.items()}
    means=[]
    for name,seed in names:
        group=[r for r in rows if r['controller']==name and r['train_seed']==seed]
        means.append(dict(controller=name,train_seed=seed,**{k:float(np.mean([r[k] for r in group])) for k in ('extended_cost','policy_cost','tail_cost','availability','terminal_pending','tail_pending','failures','policy_failures','waiting_cost','policy_overdue_ticks','forecast_transitions','planning_seconds')}))
    return dict(status='EXPLORATORY_LEARNING_PROBE' if cfg['profile']=='full' else 'ENGINEERING_ONLY',candidate=cfg['candidate'],contrasts=contrasts,means=means,
        setting_status='JOINT_SETTING_CONFIRMED in prior independent planning experiment; this pilot does not alter selection',training_replicas=nr,
        inference_scope='fixed profile/shock test panel; centralized PPO, not MARL benefit or RL impossibility')


def replay(out,cfg,rows,models,timeout):
    c=c_for(cfg);lookup={(r['controller'],r['train_seed'],r['seed']):r for r in rows};envs={};count=0;forecasts=0;paired={};cutoffs={};spent=defaultdict(int)
    selections=json.loads((out/'selections.json').read_text())
    for row in tqdm(frontier.read_csv(out/'decisions.csv'),desc='Replay learned/reference physical episodes',unit='tick'):
        timeout();key=row['controller'],int(row['train_seed']),int(row['seed']);tick=int(row['tick']);seed=key[-1]
        if tick==0:
            if key in envs:raise RuntimeError('duplicate replay episode')
            e=JointLearningEnv(c,VERSION);e.reset(seed=seed);envs[key]=e.env
            if paired.setdefault(seed,(e.env.state,e.env.events))!=(e.env.state,e.env.events):raise RuntimeError('paired tapes')
        e=envs[key]
        expected_step=selections[str(key[1])]['steps'] if key[0]=='ppo' else 0 if key[0]=='initial' else -1
        if int(row['checkpoint_steps'])!=expected_step or lookup[key]['checkpoint_steps']!=expected_step:raise RuntimeError('checkpoint selection identity')
        if row['phase']!=('policy' if tick<48 else 'tail') or int(row['policy_time'])!=(tick if tick<48 else 48) or int(row['raw_seed'])!=seed or int(row['worker'])!=0 or int(row['episode_index'])!=0:raise RuntimeError('evaluation trajectory identity')
        if int(row['physical_seed'])!=frontier.role_seed(seed,f'{VERSION}:physical') or e.time!=tick or e.state!=frontier.old.screen.base.unpack_state(json.loads(row['state'])) or tuple(json.loads(row['events']))!=e.events[tick]:raise RuntimeError('replay state/seed/events')
        action=tuple(map(tuple,json.loads(row['action'])))
        if tick>=48:expected=frontier.old.myopic(c,e.state)
        elif key[0] in ('ppo','initial'):
            probe=JointLearningEnv(c,VERSION);probe.env.state=e.state;probe.policy_time=tick
            index=int(models[key[:2]].predict(probe._observation(),action_masks=probe.action_masks(),deterministic=True)[0]);expected=probe.actions[index]
        elif key[0]!='joint_rollout' or seed in cfg['test_seeds'][:cfg['audit_seeds']]:
            expected,diag=frontier.select(c,e.state,tick,key[0],'deadline_matching',cfg,seed,timeout);forecasts+=diag['forecast_transitions']
        else:expected=action
        if action!=expected or action not in frontier.old.screen.action_space(c,e.state)[2]:raise RuntimeError('model/reference masked action replay')
        if tick<48 and key[0]=='joint_rollout':
            pool=frontier.roots(c,e.state,tick,'deadline_matching')
            spent[key]+=len(pool)*cfg['scenarios']*(60-tick) if len(pool)>1 else 0
        _,_,_,info=e.step(action);frontier.old.close_enough(info,json.loads(row['info']));frontier.old.close_enough(asdict(e.state),json.loads(row['following']))
        if abs(float(row['cost'])-info['objective'])>1e-8:raise RuntimeError('physical cost replay')
        if tick==47:
            cutoffs[key]=(e.metrics.copy(),frontier.old.screen.base.pending(c,e.state),sum(e.state.failed))
            checked=dict(policy_cost=e.metrics['objective'],availability=1-e.metrics['unavailable_ticks']/192,terminal_pending=frontier.old.screen.base.pending(c,e.state))
            for k,v in checked.items():frontier.old.close_enough(lookup[key][k],v)
        if tick==59:
            for k,v in e.metrics.items():frontier.old.close_enough(lookup[key][k],v)
            h,pending,failed=cutoffs[key]
            checked=dict(extended_cost=e.metrics['objective'],tail_cost=e.metrics['objective']-h['objective'],availability=1-h['unavailable_ticks']/192,
                technician_utilization=h['service_ticks']/96,terminal_pending=pending,terminal_failed=failed,tail_pending=frontier.old.screen.base.pending(c,e.state),
                tail_failed=sum(e.state.failed),policy_failures=h['failures'],policy_overdue_ticks=h['overdue_waiting_ticks'],physical_ticks=60,
                physical_seed=frontier.role_seed(seed,f'{VERSION}:physical'),raw_seed=seed,episode_index=0,worker=0,forecast_transitions=spent[key])
            for k,v in checked.items():frontier.old.close_enough(lookup[key][k],v)
        count+=1
    if set(envs)!=set(lookup) or any(e.time!=60 for e in envs.values()):raise RuntimeError('replay episode grid')
    return dict(status='PASS',episodes=len(envs),physical_transitions=count,model_predictions_rebuilt=True,forecast_audit_seeds=cfg['test_seeds'][:cfg['audit_seeds']],replayed_forecast_transitions=forecasts)


def training_audit(cfg,completed,partial):
    result={}
    def integer(row,key):return int(row[key])
    for seed in cfg['train_seeds']:
        rows=[r for r in completed if integer(r,'train_seed')==seed];tails=[r for r in partial if integer(r,'train_seed')==seed]
        if len(tails)!=cfg['envs']:raise RuntimeError('training partial worker grid')
        total=48*len(rows)+sum(integer(r,'learned_ticks') for r in tails)
        if total!=cfg['steps']:raise RuntimeError('training trajectory step accounting')
        for worker in range(cfg['envs']):
            done=[r for r in rows if integer(r,'worker')==worker];left=[r for r in tails if integer(r,'worker')==worker]
            if len(left)!=1 or sorted(integer(r,'episode_index') for r in done)!=list(range(len(done))) or integer(left[0],'episode_index')!=len(done):raise RuntimeError('contiguous training episode provenance')
            for r in done+left:
                if integer(r,'stream_seed')!=seed+worker:raise RuntimeError('training stream identity')
                derived=frontier.role_seed(integer(r,'stream_seed'),f"{VERSION}:training_physical:worker:{worker}:episode:{integer(r,'episode_index')}")
                if integer(r,'physical_seed')!=derived:raise RuntimeError('training physical seed provenance')
            if any(integer(r,'physical_ticks')!=60 for r in done) or integer(left[0],'physical_ticks')!=integer(left[0],'learned_ticks') or left[0]['completed']:raise RuntimeError('training partial physical accounting')
        result[str(seed)]=dict(learned_steps=total,completed_episodes=len(rows),partial_workers=len(tails),
            physical_transitions=60*len(rows)+sum(integer(r,'physical_ticks') for r in tails))
    if len(completed)!=sum(v['completed_episodes'] for v in result.values()) or len(partial)!=len(cfg['train_seeds'])*cfg['envs']:raise RuntimeError('extra training replica rows')
    return result


def run(profile,out_dir):
    cfg=settings(profile);seeds=raw_audit(cfg);git=frontier.old.screen.git_metadata();out=Path(out_dir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):raise FileExistsError('overwrite refused')
    if profile=='full' and git['dirty']:raise RuntimeError('full requires clean committed source')
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic();torch.set_num_threads(1)
    def timeout():
        if time.monotonic()-started>cfg['max_seconds']:raise TimeoutError('registered hard timeout')
    manifest=dict(protocol=VERSION,status='RUNNING',git=git,profile=profile,test_outcomes_opened=False,started_utc=datetime.now(timezone.utc).isoformat(),
        runtime=dict(platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',
            stable_baselines3=package_version('stable-baselines3'),sb3_contrib=package_version('sb3-contrib'),gymnasium=package_version('gymnasium')))
    write_json(out/'manifest.json',manifest)
    try:
        shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'registry.json');snap=out/'source_snapshot';snap.mkdir()
        for name in SOURCES:shutil.copy2(Path(__file__).parent/name,snap/name)
        seals={str(p.relative_to(out)):frontier.old.screen.sha(p) for p in (out/'protocol.md',out/'registry.json',*snap.iterdir())}
        write_json(out/'resolved_config.json',dict(settings=cfg,config=asdict(c_for(cfg))))
        manifest['source_seal']=seals;write_json(out/'manifest.json',manifest)
        journals={name:Journal(out/f'{name}.csv') for name in ('training_episodes','training_progress','development_episodes')}
        selections={};initial_paths={};partial=[];training_steps=0
        try:
            for seed in tqdm(cfg['train_seeds'],desc='Independent PPO training replicas',unit='replica'):
                timeout();vec=DummyVecEnv([lambda worker=worker:JointLearningEnv(c_for(cfg),VERSION,worker,True) for worker in range(cfg['envs'])])
                model=MaskablePPO('MlpPolicy',vec,seed=seed,device='cpu',n_steps=cfg['n_steps'],batch_size=cfg['batch_size'],n_epochs=cfg['epochs'],
                    gamma=1.,gae_lambda=1.,learning_rate=3e-4,clip_range=.2,ent_coef=.01,vf_coef=.5,max_grad_norm=.5,policy_kwargs=dict(net_arch=[64,64]),verbose=0)
                model_dir=out/f'seed_{seed}';model_dir.mkdir();initial_paths[seed]=model_dir/'initial.zip';save_model(initial_paths[seed],model,cfg,seed,0)
                monitor=Monitor(cfg,seed,model_dir,journals,timeout)
                try:model.learn(cfg['steps'],callback=monitor)
                finally:monitor.progress.close()
                if model.num_timesteps!=cfg['steps'] or {r['steps'] for r in monitor.records}!=set(cfg['checkpoints']):raise RuntimeError('training/checkpoint budget')
                selections[str(seed)]=checkpoint_choice(monitor.records);training_steps+=model.num_timesteps
                for env in vec.envs:
                    partial.append(dict(train_seed=seed,stream_seed=env.stream_seed,worker=env.worker,episode_index=env.episode_index,physical_seed=env.physical_seed,
                        learned_ticks=env.policy_time,physical_ticks=env.env.time,observed_cost=env.env.metrics['objective'],completed=not env._active))
                vec.close()
        finally:
            for journal in journals.values():journal.close()
        write_json(out/'training_partial.json',partial);write_json(out/'selections.json',selections);selection_sha=frontier.old.screen.sha(out/'selections.json')
        manifest.update(selection_sha256=selection_sha,test_outcomes_opened=profile=='full');write_json(out/'manifest.json',manifest)
        models={};rows=[];decision=Journal(out/'decisions.csv');partial_journal=Journal(out/'episodes.partial.csv')
        try:
            for seed in cfg['train_seeds']:
                models['ppo',seed]=MaskablePPO.load(selections[str(seed)]['path'],device='cpu');models['initial',seed]=MaskablePPO.load(initial_paths[seed],device='cpu')
                for name in ('ppo','initial'):
                    steps=selections[str(seed)]['steps'] if name=='ppo' else 0
                    evaluated=evaluation(cfg,name,cfg['test_seeds'],models[name,seed],seed,steps,decision,timeout);rows.extend(evaluated)
                    for row in evaluated:partial_journal.add(row)
            for name in BASELINES:
                evaluated=evaluation(cfg,name,cfg['test_seeds'],journal=decision,timeout=timeout);rows.extend(evaluated)
                for row in evaluated:partial_journal.add(row)
        finally:decision.close();partial_journal.close()
        write_csv(out/'episodes.csv',rows);audit=replay(out,cfg,rows,models,timeout)
        training=frontier.read_csv(out/'training_episodes.csv');training_counts=training_audit(cfg,training,partial);observed=[int(r['physical_seed']) for r in training]+[r['physical_seed'] for r in partial if r['learned_ticks']>0]
        eval_roles={frontier.role_seed(s,f'{VERSION}:physical') for s in cfg['development_seeds']+cfg['test_seeds']}
        if len(observed)!=len(set(observed)) or set(observed)&eval_roles:raise RuntimeError('training/evaluation physical seed overlap')
        if training_steps!=len(cfg['train_seeds'])*cfg['steps'] or any(float(r['invalid_assignments']) for r in training):raise RuntimeError('training budget/feasibility')
        development=frontier.read_csv(out/'development_episodes.csv')
        dev_grid={(seed,step,raw) for seed,step,raw in itertools.product(cfg['train_seeds'],cfg['checkpoints'],cfg['development_seeds'])}
        if len(development)!=len(dev_grid) or {(int(r['train_seed']),int(r['checkpoint_steps']),int(r['seed'])) for r in development}!=dev_grid:raise RuntimeError('development checkpoint episode grid')
        for seed in cfg['train_seeds']:
            records=[dict(steps=step,development_cost=float(np.mean([float(r['extended_cost']) for r in development if int(r['train_seed'])==seed and int(r['checkpoint_steps'])==step]))) for step in cfg['checkpoints']]
            if checkpoint_choice(records)['steps']!=selections[str(seed)]['steps']:raise RuntimeError('development selection reconstruction')
        expected=len(cfg['test_seeds'])*(2*len(cfg['train_seeds'])+len(BASELINES))
        if len(rows)!=expected or audit['physical_transitions']!=expected*60:raise RuntimeError('full counts')
        if frontier.old.screen.sha(out/'selections.json')!=selection_sha:raise RuntimeError('frozen selections changed')
        for path in out.glob('seed_*/*.json'):
            binding=json.loads(path.read_text())
            if binding['protocol']!=VERSION or frontier.old.screen.sha(path.with_suffix('.zip'))!=binding['checkpoint_sha256']:raise RuntimeError('checkpoint source binding')
        for seed in cfg['train_seeds']:
            if all(torch.equal(v,models['initial',seed].policy.state_dict()[k]) for k,v in models['ppo',seed].policy.state_dict().items()):raise RuntimeError('no learned parameter change')
        for name,digest in seals.items():
            if frontier.old.screen.sha(out/name)!=digest:raise RuntimeError('source seal')
        for name in SOURCES:
            if frontier.old.screen.sha(Path(__file__).parent/name)!=seals[f'source_snapshot/{name}']:raise RuntimeError('live source changed')
        if frontier.old.screen.sha(PLAN)!=seals['protocol.md'] or frontier.old.screen.sha(REGISTRY)!=seals['registry.json']:raise RuntimeError('live protocol changed')
        summary=summarize(cfg,rows);write_json(out/'summary.json',summary)
        audit.update(development_episodes=len(development),development_selection_rebuilt=True,seed_audit=seeds,training_trajectory_audit=training_counts,training_completed_episodes=len(training),training_partial_trajectories=partial,training_steps=training_steps,
            checkpoints=len(cfg['train_seeds'])*5,expected_test_episodes=expected,source_seal_unchanged=True,selection_frozen=True,training_unique_physical_seeds=len(observed));write_json(out/'audit.json',audit)
        files={str(p.relative_to(out)):dict(sha256=frontier.old.screen.sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        timeout();manifest.update(status='COMPLETED',audit='PASS',elapsed_seconds=time.monotonic()-started,training_steps=training_steps,test_episodes=len(rows),scientific_status=summary['status'],files=files,artifact_bytes=sum(v['bytes'] for v in files.values()));write_json(out/'manifest.json',manifest)
        return manifest
    except Exception as error:
        manifest.update(status='FAILED',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started);write_json(out/'manifest.json',manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=('smoke','full'),required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',choices=('cpu',),default='cpu');a=p.parse_args();m=run(a.profile,a.output_dir)
    print(json.dumps({k:m[k] for k in ('status','audit','scientific_status','elapsed_seconds','training_steps','test_episodes')},indent=2))

if __name__=='__main__':main()
