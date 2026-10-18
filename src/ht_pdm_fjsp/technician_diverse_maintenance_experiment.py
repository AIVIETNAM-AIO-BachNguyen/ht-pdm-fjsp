"""Prospective C1 smoke and development training on diverse maintenance."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_balanced_algorithms as learner
from ht_pdm_fjsp.maintenance_balanced_confirmation_mac import registered_integers
from ht_pdm_fjsp.maintenance_opportunity_cost import Journal, write_csv, write_json
from ht_pdm_fjsp.technician_diverse_maintenance_env import (
    DiverseEnv, FAMILIES, SCALES, VERSION, family_config,
)
from ht_pdm_fjsp.technician_relational_prefix_distillation import (
    DEFAULT_TEACHERS, freeze_teachers, load_warm, prepare_warm_job,
)
from ht_pdm_fjsp.technician_relational_prefix_experiment import make_model


ROOT=Path(__file__).resolve().parents[2]
REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
ARMS=('reservation_ppo','mask_only_ppo','independent_ppo','scratch_reservation_ppo')
CONFIGS=tuple((family,n) for family in FAMILIES for n in SCALES)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(profile):
    if profile not in ('smoke','development'):
        raise ValueError('confirmation is sealed until DEV gate and verifier exist')
    raw=json.loads(REGISTRY.read_text())['panels'][profile]
    smoke=profile=='smoke'
    return dict(profile=profile,train_seeds=raw['train_seeds'],
        development_seeds=list(range(raw['development_start'],raw['development_start']+raw['development_count'])),
        test_seeds=list(range(raw['test_start'],raw['test_start']+raw['test_count'])) if smoke else [],
        analysis_seed=raw['analysis_seed'],
        configs=[list(pair) for pair in CONFIGS if not smoke or pair[1]==8],
        env_steps=384 if smoke else 491520,vector_envs=2 if smoke else 16,
        training_processes=4,rollout=192 if smoke else 3072,
        checkpoints=[192,384] if smoke else [122880,245760,368640,491520],
        hidden=64,gamma=1.,learning_rate=.0003,epochs=2 if smoke else 4,
        minibatch=48 if smoke else 256,clip=.2,entropy_weight=.01,
        value_weight=.5,gradient_clip=.5,
        teacher_bank_steps=192 if smoke else 122880,
        teacher_epochs=2 if smoke else 32,
        teacher_minibatch=48 if smoke else 512,teacher_temperature=1.5,
        behavior_log_atol=2e-5,behavior_log_rtol=0.,
        behavior_value_atol=2e-5,behavior_value_rtol=1e-6,
        torch_threads=1,max_seconds=300 if smoke else 259200)


def seed_audit():
    history=set();historical={}
    for path in sorted((ROOT/'configs').glob('*seed_registry.json')):
        if path!=REGISTRY:
            history|=registered_integers(json.loads(path.read_text()))
            historical[path.name]=sha(path)
    groups=[]
    for profile in ('smoke','development','confirmation'):
        raw=json.loads(REGISTRY.read_text())['panels'][profile]
        groups.extend((set(raw['train_seeds']),
            set(range(raw['development_start'],raw['development_start']+raw['development_count'])),
            set(range(raw.get('test_start',0),raw.get('test_start',0)+raw.get('test_count',0))),
            {raw['analysis_seed']}))
    own=set().union(*groups)
    if any(a&b for a,b in itertools.combinations(groups,2)) or own&history:
        raise RuntimeError('raw C1 seed overlap')
    return dict(status='PASS',raw_unique=len(own),historical_registry_hashes=historical)


def internal_algorithm(arm):
    return 'reservation_ppo' if arm=='scratch_reservation_ppo' else arm


class Tags:
    def __init__(self,journal,tags):
        self.journal=journal;self.tags=tags
    def add(self,row):
        self.journal.add(dict(row,**self.tags))


def evaluate(family,n,arm,train_seed,seed,model=None,coordination=None,decisions=None,latencies=None):
    tags=dict(family=family,n=n,method=arm,train_seed=train_seed,seed=seed)
    env=DiverseEnv(seed,n,family,internal_algorithm(arm) if arm in ARMS else arm,
        journal=Tags(decisions,tags) if decisions is not None else None)
    obs=env.reset();reward_sum=0.;select_times=[];end_to_end=[];conflicts=rejected=0
    for tick in range(48):
        before=time.perf_counter();action=model.select(obs) if model is not None else None
        select_ms=(time.perf_counter()-before)*1000
        obs_sha=hashlib.sha256(obs['state'].tobytes()).hexdigest()
        before=time.perf_counter();obs,r,done,info=env.step(action)
        env_ms=(time.perf_counter()-before)*1000
        select_times.append(select_ms);end_to_end.append(select_ms+env_ms)
        reward_sum+=r
        conflicts+=info['coordination']['conflicts']
        rejected+=info['coordination']['rejected_proposals']
        if coordination is not None:
            coordination.add(dict(tags,tick=tick,observation_sha256=obs_sha,
                submitted=json.dumps(np.asarray(action).tolist()) if action is not None else 'null',
                decision_ms=select_ms,environment_ms=env_ms,**info['coordination']))
    metrics=info['episode_metrics']
    if not done or abs(-20*reward_sum-metrics['extended_cost'])>1e-5 or metrics['invalid_assignments']:
        raise RuntimeError('C1 episode accounting or feasibility')
    if latencies is not None:latencies.extend(select_times)
    return dict(tags,**metrics,conflicts=conflicts,rejected_proposals=rejected,
        normalized_cost=metrics['extended_cost']/n,
        normalized_pending=metrics['terminal_pending']/n,
        normalized_failures=metrics['policy_failures']/n,
        decision_mean_ms=float(np.mean(select_times)),
        decision_p95_ms=float(np.quantile(select_times,.95)),
        end_to_end_mean_ms=float(np.mean(end_to_end)),
        end_to_end_p95_ms=float(np.quantile(end_to_end,.95)))


def save_model(model,path,family,n,arm,seed,steps,probe):
    if path.exists():raise FileExistsError('checkpoint overwrite')
    payload=dict(protocol=VERSION,family=family,n=n,arm=arm,
        algorithm=model.algorithm,train_seed=seed,steps=steps,hidden=64,
        parameter_sha256=learner.parameter_digest(model),state_dict=model.state_dict())
    torch.save(payload,path)
    check=load_model(path,family,n,arm,seed,steps)
    if not np.array_equal(model.select(probe),check.select(probe)):
        raise RuntimeError('C1 checkpoint roundtrip')
    return dict(path=str(path),sha256=sha(path),parameter_sha256=payload['parameter_sha256'])


def load_model(path,family,n,arm,seed,steps):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    for key,value in dict(protocol=VERSION,family=family,n=n,arm=arm,
                          algorithm=internal_algorithm(arm),train_seed=seed,
                          steps=steps,hidden=64).items():
        if payload.get(key)!=value:raise RuntimeError('C1 checkpoint metadata')
    model=make_model(internal_algorithm(arm),n,dict(hidden=64),seed)
    model.load_state_dict(payload['state_dict']);model.eval()
    if learner.parameter_digest(model)!=payload['parameter_sha256']:
        raise RuntimeError('C1 checkpoint digest')
    return model


def fit_job(job):
    cfg,family,n,arm,seed,out_name,start=job
    out=Path(out_name);torch.set_num_threads(1)
    def guard():
        if (out/'abort.json').exists():raise RuntimeError('sibling technical abort')
        if time.monotonic()-start>cfg['max_seconds']:raise TimeoutError('C1 hardstop')
    guard();algorithm=internal_algorithm(arm)
    model=make_model(algorithm,n,cfg,seed)
    initial=learner.parameter_digest(model)
    warm=None if arm=='scratch_reservation_ppo' else load_warm(model,out,seed)
    path=out/f'{family}_N{n}'/arm/str(seed);path.mkdir(parents=True)
    checkpoints=path/'checkpoints';checkpoints.mkdir()
    journals={name:Journal(path/f'{name}.csv') for name in
        ('training_progress','training_episodes','development_episodes')}
    probe=DiverseEnv(cfg['development_seeds'][0],n,family,algorithm).reset()
    rows=[];candidates=[];begin=time.monotonic()
    def assess(current,steps,audit=None):
        guard();file=checkpoints/f'step_{steps:09d}.pt'
        meta=save_model(current,file,family,n,arm,seed,steps,probe)
        meta['path']=str(file.relative_to(out))
        panel=[];current.eval()
        for physical in tqdm(cfg['development_seeds'],
                desc=f'C1 DEV {family}/N{n}/{arm}/{seed}/{steps}',leave=False):
            guard();row=evaluate(family,n,arm,seed,physical,current)
            row['checkpoint_steps']=steps
            journals['development_episodes'].add(row);rows.append(row);panel.append(row)
        current.train()
        candidates.append(dict(meta,steps=steps,
            development_cost_per_machine=float(np.mean([r['normalized_cost'] for r in panel]))))
    try:
        assess(model,0)
        audit=learner.fit(model,dict(cfg,rng_namespace=f'{VERSION}:{family}:N{n}:'),seed,
            lambda s,w:DiverseEnv(s,n,family,algorithm,w,True),
            {name:Tags(journals[name],dict(family=family,n=n,arm=arm))
             for name in ('training_progress','training_episodes')},
            assess,guard)
        return dict(family=family,n=n,arm=arm,train_seed=seed,
            initial=initial,parameters=sum(p.numel() for p in model.parameters()),
            development_rows=rows,
            audit=dict(audit,family=family,n=n,arm=arm,
                warm_state_exposures=warm['state_exposures'] if warm else 0,
                warm_optimizer_steps=warm['optimizer_steps'] if warm else 0,
                bank_sha256=warm['bank_sha256'] if warm else None,
                seconds_including_development=time.monotonic()-begin),
            choice=min(candidates,key=lambda v:(v['development_cost_per_machine'],v['steps'])))
    finally:
        for journal in journals.values():journal.close()


def development_summary(rows,reference,cfg):
    configs=[tuple(pair) for pair in cfg['configs']]
    by_family={family:[n for f,n in configs if f==family] for family in FAMILIES}
    selected={};means={};health={};auc={};final={}
    for family,n in configs:
        label=f'{family}:N{n}'
        means[label]={};health[label]={};auc[label]={};final[label]={}
        for arm in ARMS:
            chosen_cost=[];selected_rows=[];auc_values=[];final_values=[]
            for seed in cfg['train_seeds']:
                panel=[r for r in rows if (r['family'],r['n'],r['method'],r['train_seed'])==(family,n,arm,seed)]
                costs={step:float(np.mean([r['normalized_cost'] for r in panel
                    if r['checkpoint_steps']==step])) for step in [0,*cfg['checkpoints']]}
                step=min(costs,key=lambda s:(costs[s],s))
                selected[f'{label}:{arm}:{seed}']=step
                chosen_cost.append(costs[step])
                selected_rows.extend(r for r in panel if r['checkpoint_steps']==step)
                final_values.append(costs[cfg['env_steps']])
                ordered=sorted(costs)
                auc_values.append(float(np.trapezoid([costs[s] for s in ordered],ordered)
                    /(ordered[-1]-ordered[0])))
            means[label][arm]=float(np.mean(chosen_cost))
            health[label][arm]={name:float(np.mean([r[name] for r in selected_rows]))
                for name in ('availability','normalized_pending','normalized_failures','invalid_assignments')}
            auc[label][arm]=dict(mean=float(np.mean(auc_values)),per_replica=auc_values)
            final[label][arm]=dict(mean=float(np.mean(final_values)),per_replica=final_values)
    heuristic_means={arm:float(np.mean([r['normalized_cost'] for r in reference
        if r['method']==arm])) for arm in ('fcfs','risk')}
    selected_rule=min(('fcfs','risk'),key=lambda a:(heuristic_means[a],('fcfs','risk').index(a)))
    full=np.mean([means[f'{f}:N{n}']['reservation_ppo'] for f,n in configs])
    mask=np.mean([means[f'{f}:N{n}']['mask_only_ppo'] for f,n in configs])
    heuristic=heuristic_means[selected_rule]
    pooled=1-full/mask;rule_gain=1-full/heuristic
    family_gains={family:1-sum(means[f'{family}:N{n}']['reservation_ppo'] for n in by_family[family])/
        sum(means[f'{family}:N{n}']['mask_only_ppo'] for n in by_family[family]) for family in FAMILIES}
    health_guards={family:dict(
        availability=bool(np.mean([health[f'{family}:N{n}']['reservation_ppo']['availability']-
            health[f'{family}:N{n}']['mask_only_ppo']['availability'] for n in by_family[family]])>=-.01),
        pending=bool(np.mean([health[f'{family}:N{n}']['reservation_ppo']['normalized_pending']-
            health[f'{family}:N{n}']['mask_only_ppo']['normalized_pending'] for n in by_family[family]])<=.025),
        failures=bool(np.mean([health[f'{family}:N{n}']['reservation_ppo']['normalized_failures']-
            health[f'{family}:N{n}']['mask_only_ppo']['normalized_failures'] for n in by_family[family]])<=.25),
        invalid=all(health[f'{family}:N{n}'][a]['invalid_assignments']==0
            for n in by_family[family] for a in ARMS)) for family in FAMILIES}
    passed=(pooled>=.005 and all(v>=-.01 for v in family_gains.values()) and
        rule_gain>=-.01 and all(all(v.values()) for v in health_guards.values()))
    return dict(classification='DEVELOPMENT_ONLY',test_opened=False,
        development_screen_pass=bool(passed),selected_rule=selected_rule,
        heuristic_means=heuristic_means,pooled_gain_full_vs_mask=float(pooled),
        pooled_gain_full_vs_heuristic=float(rule_gain),family_gains=family_gains,
        health_guards=health_guards,means=means,health=health,
        selected_steps=selected,learning_auc=auc,fixed_final_cost_per_machine=final)


def run(profile,output_dir,teacher_dir=None):
    cfg=settings(profile);out=Path(output_dir).resolve()
    configs=[tuple(pair) for pair in cfg['configs']]
    if out.exists():raise FileExistsError('fresh C1 artifact required')
    out.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(1)
    git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()))
    if profile!='smoke' and git['dirty']:
        raise RuntimeError('commit and push verified code before C1 development FULL')
    manifest=dict(protocol=VERSION,status='RUNNING',profile=profile,git=git,test_opened=False,
        invocation=dict(executable=sys.executable,argv=sys.argv,cwd=str(Path.cwd())),
        started_utc=datetime.now(timezone.utc).isoformat(),
        runtime=dict(platform=platform.platform(),python=platform.python_version(),
            numpy=np.__version__,torch=torch.__version__,device='cpu'),phase_wall_seconds={})
    write_json(out/'manifest.json',manifest)
    def guard():
        if time.monotonic()-start>cfg['max_seconds']:raise TimeoutError('C1 registered hardstop')
    try:
        seed_report=seed_audit()
        teacher_report=freeze_teachers(teacher_dir or DEFAULT_TEACHERS,out,cfg)
        write_json(out/'resolved_config.json',cfg)
        write_json(out/'seed_audit.json',seed_report)
        write_json(out/'physics.json',{f'{f}:N{n}':family_config(f,n).__dict__ for f,n in configs})
        write_json(out/'execution_contract.json',dict(
            request_timing='threshold fixed for all arms; persistent until assigned',
            feature_adapter='machine feature 2 is pending request age/12, else zero',
            actor='B2 candidate-conditioned selected-edge attention, identical parameter count across warm graph arms',
            teacher='frozen N4 centralized PPO bank shared across C1 configurations',
            physical='audited WaitingEnv shared technician transition/cost, common FCFS tail and terminal liability'))
        snap=out/'source_snapshot';snap.mkdir()
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
        history=out/'registry_snapshot';history.mkdir()
        for name in seed_report['historical_registry_hashes']:
            shutil.copy2(ROOT/'configs'/name,history/name)
        shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'seed_registry.json')
        seal={str(p.relative_to(out)):sha(p) for p in
            [*snap.iterdir(),*history.iterdir(),out/'protocol.md',out/'seed_registry.json',
             out/'physics.json',out/'execution_contract.json',out/'resolved_config.json',
             *(out/'teachers').iterdir()]}
        manifest['source_seal']=seal;write_json(out/'manifest.json',manifest)
        begin=time.monotonic();manifest['stage']='teacher_bank_and_distillation';write_json(out/'manifest.json',manifest)
        warm={}
        with ProcessPoolExecutor(max_workers=cfg['training_processes'],mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(prepare_warm_job,(cfg,seed,str(out),start)) for seed in cfg['train_seeds']]
            for future in tqdm(as_completed(futures),total=len(futures),desc='C1 teacher banks',unit='bundle'):
                result=future.result();warm[str(result['train_seed'])]=result;guard()
        write_json(out/'warm_choices.json',warm)
        manifest['phase_wall_seconds']['teacher_bank_and_distillation']=time.monotonic()-begin
        begin=time.monotonic();manifest['stage']='training_pool';write_json(out/'manifest.json',manifest)
        jobs=[(cfg,f,n,a,t,str(out),start) for f,n in configs for a in ARMS for t in cfg['train_seeds']]
        results={}
        with ProcessPoolExecutor(max_workers=cfg['training_processes'],mp_context=mp.get_context('spawn')) as pool:
            futures={pool.submit(fit_job,job):job for job in jobs}
            for future in tqdm(as_completed(futures),total=len(futures),desc='C1 training models',unit='model'):
                result=future.result();guard()
                key=f"{result['family']}:N{result['n']}:{result['arm']}:{result['train_seed']}"
                results[key]=result
                manifest['completed_models']=len(results);write_json(out/'manifest.json',manifest)
        manifest['phase_wall_seconds']['training_including_development']=time.monotonic()-begin
        journals={name:Journal(out/f'{name}.csv') for name in
            ('training_progress','training_episodes','development_episodes')}
        choices={};audits=[];rows=[];initial={};parameters={}
        try:
            for _,f,n,a,t,_,_ in jobs:
                key=f'{f}:N{n}:{a}:{t}';result=results[key]
                choices[key]=result['choice'];audits.append(result['audit'])
                rows.extend(result['development_rows']);initial[key]=result['initial']
                parameters[key]=result['parameters']
                directory=out/f'{f}_N{n}'/a/str(t)
                for name in journals:
                    with (directory/f'{name}.csv').open() as stream:
                        for row in csv.DictReader(stream):journals[name].add(row)
        finally:
            for journal in journals.values():journal.close()
        for f,n in configs:
            for t in cfg['train_seeds']:
                if len({initial[f'{f}:N{n}:{a}:{t}'] for a in ARMS})!=1:
                    raise RuntimeError('C1 initial parameter mismatch')
        write_json(out/'checkpoint_choices.json',choices)
        write_json(out/'training_audit.json',audits)
        write_json(out/'initial_parameter_hashes.json',initial)
        write_json(out/'parameter_counts.json',parameters)
        references=[]
        for f,n in configs:
            for method in ('fcfs','risk'):
                for seed in tqdm(cfg['development_seeds'],desc=f'C1 DEV {f}/N{n}/{method}',unit='episode'):
                    guard();references.append(evaluate(f,n,method,-1,seed))
        write_csv(out/'development_reference.csv',references)
        summary=development_summary(rows,references,cfg)
        write_json(out/'reference_selection.json',dict(selected_rule=summary['selected_rule'],
            heuristic_means=summary['heuristic_means'],test_opened=False))
        write_json(out/'summary.json',summary)
        engineering_test=[]
        if profile=='smoke':
            manifest['stage']='engineering_test'
            manifest['test_opened']=True
            manifest['choices_sha256']=sha(out/'checkpoint_choices.json')
            manifest['reference_selection_sha256']=sha(out/'reference_selection.json')
            write_json(out/'manifest.json',manifest)
            for f,n in configs:
                for arm in (*ARMS,'fcfs','risk'):
                    for train_seed in (cfg['train_seeds'] if arm in ARMS else [-1]):
                        model=None
                        if arm in ARMS:
                            choice=choices[f'{f}:N{n}:{arm}:{train_seed}']
                            model=load_model(out/choice['path'],f,n,arm,train_seed,choice['steps'])
                        for physical in tqdm(cfg['test_seeds'],
                                desc=f'C1 engineering TEST {f}/N{n}/{arm}',unit='episode'):
                            guard();engineering_test.append(evaluate(f,n,arm,train_seed,physical,model))
            write_csv(out/'engineering_test_episodes.csv',engineering_test)
            populations={}
            for row in engineering_test:
                key=row['family'],row['n'],row['seed']
                tape=row['physical_seed'],row['population_sha256']
                if key in populations and populations[key]!=tape:
                    raise RuntimeError('C1 engineering TEST pairing')
                populations[key]=tape
        # Independent artifact/grid checks before sealing this DEV verdict.
        expected={(f,n,a,t,s,d) for f,n in configs for a in ARMS
            for t in cfg['train_seeds'] for s in [0,*cfg['checkpoints']]
            for d in cfg['development_seeds']}
        actual={(r['family'],r['n'],r['method'],r['train_seed'],r['checkpoint_steps'],r['seed']) for r in rows}
        if len(rows)!=len(expected) or actual!=expected:raise RuntimeError('C1 development grid')
        if len(audits)!=len(configs)*len(ARMS)*len(cfg['train_seeds']):raise RuntimeError('C1 training grid')
        if any(a['learned_steps']!=cfg['env_steps'] or a['training_episodes']!=cfg['env_steps']//48
               or a['optimizer_steps']<=0 for a in audits):raise RuntimeError('C1 PPO budget')
        write_json(out/'audit.json',dict(status='PASS',development_episodes=len(rows),
            training_models=len(audits),teacher_bank_exposures=cfg['teacher_bank_steps']*len(cfg['train_seeds']),
            engineering_test_episodes=len(engineering_test),
            scientific_test_opened=False,invalid_assignments=0))
        if any(sha(out/path)!=digest for path,digest in seal.items()):
            raise RuntimeError('C1 source seal changed')
        manifest.update(status='COMPLETED',stage='complete',audit_status='PASS',
            elapsed_seconds=time.monotonic()-start,completed_utc=datetime.now(timezone.utc).isoformat(),
            development_screen_pass=summary['development_screen_pass'],test_opened=profile=='smoke')
        manifest['output_bindings']={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size)
            for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        write_json(out/'manifest.json',manifest)
        return manifest
    except Exception as exc:
        write_json(out/'abort.json',dict(reason='TECHNICAL_FAILURE',error=f'{type(exc).__name__}: {exc}'))
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',
            elapsed_seconds=time.monotonic()-start)
        write_json(out/'manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('smoke','development'),required=True)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--teacher-dir',type=Path)
    args=parser.parse_args()
    result=run(args.profile,args.output_dir,args.teacher_dir)
    print(json.dumps(dict(status=result['status'],output_dir=str(args.output_dir),
        development_screen_pass=result['development_screen_pass'])))


if __name__=='__main__':main()
