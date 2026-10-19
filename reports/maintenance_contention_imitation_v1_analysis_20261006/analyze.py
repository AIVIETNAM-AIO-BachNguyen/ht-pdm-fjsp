"""Read-only integrity audit and fixed-protocol analysis of the transferred run."""
import csv
from collections import defaultdict
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
from tqdm.auto import tqdm

CHECKOUT=Path('/private/tmp/marl-oracle-representation-20261003')
sys.path.insert(0,str(CHECKOUT/'src'))
from ht_pdm_fjsp import maintenance_contention_imitation as runner
from ht_pdm_fjsp.maintenance_contention_learning_policy import feature_batch
from ht_pdm_fjsp.maintenance_contention_learning_training import episode_spec
from ht_pdm_fjsp.maintenance_contention_imitation_training import accepted_targets,diagnostics
from ht_pdm_fjsp.maintenance_contention_model import actions,cohort_config,compact,physical_action,plan,reference,subsets
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

torch.set_num_threads(1)
ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl')
RAW=ROOT/'lab_results/maintenance_contention_imitation_v1_full_cpu_20261006T063713Z'
OUT=Path(__file__).resolve().parent
START=perf_counter()

def load(name):return json.loads((RAW/name).read_text())
def rows(name):
    with (RAW/name).open() as stream:return list(csv.DictReader(stream))
def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
def hash_file(path):
    d=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):d.update(block)
    return d.hexdigest()
def key(row):return (int(row['train_seed']),int(row['cohort']),row['capacity'],int(row['horizon']),int(row['seed']),row['controller'])
def numeric(row):
    result={k:v if k in ('capacity','controller') else float(v) for k,v in row.items()}
    for k in ('train_seed','checkpoint_step','cohort','horizon','seed'):result[k]=int(result[k])
    return result

def reconcile(row):
    assert abs(float(row['objective'])-float(row['base_cost'])-float(row['waiting_cost']))<1e-8
    assert abs(float(row['base_cost'])-sum(float(row[k]) for k in ('maintenance_cost','failure_cost','unavailability_cost')))<1e-8
    assert float(row['invalid_assignments'])==0

manifest=load('manifest.json');cfg=manifest['config'];expected=runner.counts(cfg)
assert manifest['status']=='COMPLETED' and not manifest['dirty']
assert manifest['commit']=='de85e82e5aa2b5e9a1def742b325b12316101cda'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()==manifest['commit']
assert cfg==runner.settings('full') and manifest['actual_counts']==expected==manifest['expected_counts']
for name,meta in tqdm(manifest['artifacts'].items(),desc='artifact integrity'):
    p=RAW/name;assert p.stat().st_size==meta['bytes'] and hash_file(p)==meta['sha256'],name
for p in (CHECKOUT/'src/ht_pdm_fjsp').glob('maintenance_contention*.py'):
    assert hash_file(p)==hash_file(RAW/'source_snapshot'/p.name)
for p in [CHECKOUT/f'docs/{runner.VERSION}_plan.md',CHECKOUT/f'configs/{runner.VERSION}_seed_registry.json',CHECKOUT/'uv.lock']:
    assert hash_file(p)==hash_file(RAW/'source_snapshot'/p.name)
audit,forbidden=runner.seed_audit(cfg);assert audit==load('seed_audit.json')
assert hash_file(RAW/'episodes.csv')==hash_file(RAW/'episodes.partial.csv')
episodes=[numeric(r) for r in rows('episodes.csv')]
development=[numeric(r) for r in rows('development_episodes.csv')]
assert len(episodes)==105000 and len(development)==62400
for r in episodes:reconcile(r)
selection=load('reference_selection.json');selected_refs=selection['selected']
refdev=[r for r in development if r['train_seed']==-1];assert len(refdev)==2400
for cap in cfg['capacities']:
    means={n:runner.mean([r for r in refdev if r['capacity']==cap and r['controller']==n],'objective') for n in cfg['controls'][:-1]}
    assert means==selection['development_means'][cap]
    assert min(means,key=lambda n:(means[n],n))==selected_refs[cap]
selected_dev=[r for r in refdev if r['controller']==selected_refs[r['capacity']]]
checkpoints=load('checkpoint_selection.json')['selections'];assert len(checkpoints)==30
models={};curves=[];initials={};changed={}
for arm,seed in tqdm(list(itertools.product(runner.ARMS,cfg['train_seeds'])),desc='checkpoint provenance'):
    meta=checkpoints[f'{arm}:{seed}'];records=[]
    for candidate in meta['candidates']:
        model,payload=runner.load_checkpoint(RAW/candidate['path'])
        assert payload['physical_steps']==candidate['steps'] and payload['algorithm']==arm and payload['train_seed']==seed
        if candidate['steps']==0:initials[arm,seed]={k:v.clone() for k,v in model.state_dict().items()}
        else:assert any(not torch.equal(v,initials[arm,seed][k]) for k,v in model.state_dict().items())
        panel=[r for r in development if r['controller']==arm and r['train_seed']==seed and r['checkpoint_step']==candidate['steps']]
        assert len(panel)==400
        assert {(r['cohort'],r['capacity'],r['seed']) for r in panel}==set(itertools.product(cfg['development_cohorts'],('middle','low'),cfg['development_seeds']))
        records.append({**candidate,'episodes':panel})
        for cap in ('middle','low'):
            own=[r for r in panel if r['capacity']==cap]
            curves.append(dict(arm=arm,train_seed=seed,step=candidate['steps'],update=candidate['update'],capacity=cap,
                               **{m:runner.mean(own,m) for m in ('objective','base_cost','waiting_violation','failures','utilization')}))
    chosen,status=runner.select_checkpoint(records,selected_dev)
    assert status==meta['status'] and chosen['steps']==meta['selected_steps'] and chosen['development_primary_cost']==meta['development_cost']
    assert hash_file(RAW/meta['checkpoint'])==meta['checkpoint_sha256']==hash_file(RAW/chosen['path'])
    models[arm,seed],payload=runner.load_checkpoint(RAW/meta['checkpoint'])
    if arm==runner.IMITATION:
        assert all(torch.equal(v,initials[arm,seed][k]) for k,v in models[arm,seed].state_dict().items() if k.startswith('critic.'))
for seed in cfg['train_seeds']:
    assert all(torch.equal(v,initials[runner.IMITATION,seed][k]) for k,v in initials[runner.DIRECT,seed].items())
    assert all(torch.equal(v,initials[runner.HYBRID,seed][k]) for k,v in models[runner.IMITATION,seed].state_dict().items())
    assert checkpoints[f'{runner.HYBRID}:{seed}']['parent_checkpoint_sha256']==checkpoints[f'{runner.IMITATION}:{seed}']['checkpoint_sha256']
paired,cohorts,results=runner.summarize(episodes,selected_refs,cfg)
saved=load('summary.json')
assert json.loads(json.dumps(results['primary']))==saved['primary'] and results['arms']==saved['arms']
metrics=[k for k in episodes[0] if k not in ('train_seed','checkpoint_step','cohort','capacity','horizon','seed','controller')]
groups={}
for cap,h,name in itertools.product(cfg['capacities'],cfg['horizons'],[*cfg['controls'],*runner.ARMS]):
    panel=[r for r in episodes if r['capacity']==cap and r['horizon']==h and r['controller']==name]
    groups[f'{cap}_H{h}_{name}']={m:runner.mean(panel,m) for m in metrics}
dump('statistics.json',dict(results=results,groups=groups,paired_seeds=paired,cohorts=cohorts,
                            development_curves=curves,selected={k:{a:b for a,b in v.items() if a!='candidates'} for k,v in checkpoints.items()}))
print('Integrity, source, selection and exact result recomputation PASS',flush=True)
print(json.dumps(results['primary']),flush=True)
# Each physical training episode's seed/configuration/cost counters are audited.
training_count=defaultdict(int);mix=defaultdict(lambda:defaultdict(int));profiles=defaultdict(set)
with (RAW/'training_episodes.csv').open() as stream:
    for row in tqdm(csv.DictReader(stream),total=819200,desc='PPO training provenance'):
        seed=int(row['train_seed']);arm=row['arm'];index=int(row['episode']);k=(arm,seed)
        training_count[k]+=1;assert index==training_count[k]
        c,cap,cs,es=episode_spec(seed,index,12,forbidden)
        assert cs==int(row['config_seed']) and es==int(row['environment_seed']) and cap==row['capacity']
        digest=hashlib.sha256(json.dumps(asdict(c),sort_keys=True).encode()).hexdigest();assert digest==row['config_sha256']
        profiles[k].add(digest);mix[k][cap]+=1
        assert int(row['rollout_update'])==math.ceil(index/128) and int(row['physical_steps'])==math.ceil(index/32)*32*12
        reconcile(row)
for r in load('training_coverage.json'):
    k=(r['arm'],r['train_seed'])
    if r['arm']==runner.IMITATION:
        assert r['env_steps']==0 and r['optimizer_steps']==5120 and r['teacher_label_rows']==12288
        assert r['critic_head_unchanged'] and r['validation_training_rows']==0
    else:
        assert training_count[k]==40960 and dict(mix[k])==r['mixture_counts']==dict(low=8192,middle=32768)
        assert r['env_steps']==491520 and r['optimizer_steps']==10240 and r['teacher_label_rows']==0
progress=rows('training_progress.csv');assert len(progress)==6400
for arm,seed in itertools.product((runner.DIRECT,runner.HYBRID),cfg['train_seeds']):
    panel=[r for r in progress if r['arm']==arm and int(r['train_seed'])==seed]
    assert [int(r['rollout_update']) for r in panel]==list(range(1,321))
    for r in panel:
        u=int(r['rollout_update']);assert int(r['physical_steps'])==u*1536 and int(r['episodes'])==u*128 and int(r['optimizer_steps'])==u*32
        assert all(math.isfinite(float(v)) for k,v in r.items() if k!='arm')
bc_progress=rows('bc_progress.csv');assert len(bc_progress)==800
for seed in cfg['train_seeds']:
    panel=[r for r in bc_progress if int(r['train_seed'])==seed]
    assert [int(r['epoch']) for r in panel]==list(range(1,81))
    for r in panel:assert int(r['optimizer_steps'])==int(r['epoch'])*64 and all(math.isfinite(float(v)) for v in r.values())
# Every learned decision is replayed in batched inference; every physical state
# transition is replayed with the recorded actual shock seed. Planner forecasts
# are fully recomputed for all60 episodes with the first test shock seed.
counts=defaultdict(int);totals={};state_key=None;env=None;neural=[]
neural_count=0;action_mismatches=[];max_lp_error=max_value_error=0.;forecast_replay=0

def flush_neural():
    global neural_count,max_lp_error,max_value_error
    if not neural:return
    model=models[neural[0]['arm'],neural[0]['train_seed']]
    batch=feature_batch([r['config'] for r in neural],[r['compact'] for r in neural],[r['remaining'] for r in neural])
    predicted,lp,value=model.sample(batch,deterministic=True)
    for i,r in enumerate(neural):
        if int(predicted[i])!=r['diagnostics']['action_index']:
            action_mismatches.append(dict(key=r['key'],time=r['time'],expected=r['diagnostics']['action_index'],actual=int(predicted[i])))
        max_lp_error=max(max_lp_error,abs(float(lp[i])-r['diagnostics']['log_probability']))
        max_value_error=max(max_value_error,abs(float(value[i])-r['diagnostics']['value']))
        assert np.isclose(float(lp[i]),r['diagnostics']['log_probability'],atol=5e-5,rtol=1e-5)
        assert np.isclose(float(value[i]),r['diagnostics']['value'],atol=5e-5,rtol=1e-5)
    neural_count+=len(neural);neural.clear()
with (RAW/'decisions.csv').open() as stream:
    for row in tqdm(csv.DictReader(stream),total=1890000,desc='physical and policy replay'):
        k=key(row);t=int(row['time'])
        if k!=state_key:
            env=WaitingEnv(cohort_config(k[1],k[2],k[3]));env.reset(k[4]);state_key=k
        assert t==counts[k]
        assert asdict(env.state)=={n:tuple(v) for n,v in json.loads(row['state']).items()}
        diag=json.loads(row['controller_diagnostics']);pairs=tuple(tuple(p) for p in json.loads(row['action']));state=compact(env.state)
        if k[-1] in runner.ARMS:
            if neural and ((neural[0]['arm'],neural[0]['train_seed'])!=(k[-1],k[0]) or len(neural)>=512):flush_neural()
            assert int(row['checkpoint_step'])==checkpoints[f'{k[-1]}:{k[0]}']['selected_steps']
            assert physical_action(env.state,subsets(4)[diag['action_index']])==pairs
            neural.append(dict(arm=k[-1],train_seed=k[0],config=env.config,compact=state,remaining=k[3]-t,diagnostics=diag,key=k,time=t))
        elif k[-1]=='joint_rollout':
            assert physical_action(env.state,diag['candidates'][int(np.argmin(diag['means']))])==pairs
            assert diag['scenario_seed']==role_seed(k[4],f'contention_planner:{t}')
            if k[4]==cfg['test_seeds'][0]:
                chosen,replay=plan(env.config,state,k[3]-t,selected_refs[k[2]],diag['scenario_seed'],32)
                assert json.loads(json.dumps(replay))==diag;forecast_replay+=1
        else:assert physical_action(env.state,reference(env.config,state,k[3]-t,k[-1]))==pairs
        _,_,_,info=env.step(pairs)
        assert info==json.loads(row['physical_metrics'])
        assert asdict(env.state)=={n:tuple(v) for n,v in json.loads(row['following']).items()}
        counts[k]+=1;totals[k]=dict(env.metrics)
flush_neural()
assert sum(counts.values())==1890000 and len(counts)==105000 and neural_count==1620000
for row in episodes:
    k=key(row);assert counts[k]==k[3]
    for metric,value in totals[k].items():assert abs(value-row[metric])<1e-8
# Machine-level cost and service duration diagnostics.
machine_panels=defaultdict(list);duration_panels=defaultdict(list)
with (RAW/'machine_metrics.csv').open() as stream:
    for row in csv.DictReader(stream):
        k=key(row);machine_panels[k].append(row)
        if k[3]==12:
            c=cohort_config(k[1],k[2],k[3]);d=c.service_time[int(row['machine'])][0]
            duration_panels[k[2],k[-1],d].append(row)
assert sum(map(len,machine_panels.values()))==420000
for row in episodes:
    panel=machine_panels[key(row)];assert sorted(int(r['machine']) for r in panel)==list(range(4))
    for metric in ('unavailable_ticks','failed_waiting_ticks','overdue_waiting_ticks','terminal_pending','terminal_failed'):
        assert sum(float(r[metric]) for r in panel)==row[metric]
    assert sum(float(r['service_starts']) for r in panel)==row['jobs']
by_duration={f'{cap}_{name}_duration{d}':{m:float(np.mean([float(r[m]) for r in panel])) for m in ('service_starts','unavailable_ticks','failed_waiting_ticks','overdue_waiting_ticks','max_wait','terminal_pending','violation')} for (cap,name,d),panel in duration_panels.items()}
seen_profiles={f'{arm}:{seed}':sum(hashlib.sha256(json.dumps(asdict(cohort_config(cs,'middle',12)),sort_keys=True).encode()).hexdigest() in profiles[arm,seed] for cs in cfg['test_cohorts']) for arm,seed in profiles}
for name,meta in manifest['artifacts'].items():assert hash_file(RAW/name)==meta['sha256']
dump('analysis.json',dict(raw=str(RAW),commit=manifest['commit'],started=manifest['started'],finished=manifest['finished'],
                         results=results,groups=groups,paired_seeds=paired,cohorts=cohorts,development_curves=curves,
                         selected_steps={k:r['selected_steps'] for k,r in checkpoints.items()},
                         seen_primary_profiles=seen_profiles,machine_by_duration=by_duration,
                         audit=dict(status='PASS' if not action_mismatches else 'NUMERICAL_ACTION_MISMATCH',raw_files=len(manifest['artifacts']),
                                    checkpoint_files=180,models=30,training_episode_provenance=819200,optimizer_progress_rows=7200,
                                    development_episodes=62400,test_episodes=105000,physical_intervals=1890000,
                                    planner_forecast_replay_intervals=forecast_replay,learned_action_replay_intervals=neural_count,
                                    machine_rows=420000,neural_action_mismatches=action_mismatches,
                                    max_log_probability_error=max_lp_error,max_value_error=max_value_error,
                                    raw_hashes_unchanged=True,elapsed_seconds=perf_counter()-START)))
print(json.dumps(dict(status='PASS' if not action_mismatches else 'NUMERICAL_ACTION_MISMATCH',mismatches=len(action_mismatches),elapsed_seconds=perf_counter()-START)),flush=True)
