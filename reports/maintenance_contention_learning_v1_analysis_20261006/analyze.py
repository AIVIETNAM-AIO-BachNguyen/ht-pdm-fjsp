"""Read-only raw audit, batched checkpoint replay and fixed-protocol analysis."""
import csv
from collections import defaultdict
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from tqdm.auto import tqdm

CHECKOUT=Path('/private/tmp/marl-oracle-representation-20261003')
sys.path.insert(0,str(CHECKOUT/'src'))
from ht_pdm_fjsp import maintenance_contention_learning as runner
from ht_pdm_fjsp.maintenance_contention_learning_policy import feature_batch
from ht_pdm_fjsp.maintenance_contention_learning_training import episode_spec
from ht_pdm_fjsp.maintenance_contention_model import cohort_config,compact,physical_action,plan,reference,subsets
from ht_pdm_fjsp.maintenance_dispatch import DispatchState,role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

torch.set_num_threads(1)
ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl')
RAW=ROOT/'lab_results/maintenance_contention_learning_v1_full_cpu_20261006T023329Z'
OUT=Path(__file__).resolve().parent
START=perf_counter()

def load(name):return json.loads((RAW/name).read_text())
def rows(name):
    with (RAW/name).open() as stream:return list(csv.DictReader(stream))
def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
def hash_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def key(row):return (int(row['train_seed']),int(row['cohort']),row['capacity'],int(row['horizon']),int(row['seed']),row['controller'])
def numeric(row):
    nonnumeric={'capacity','controller'}
    result={k:v if k in nonnumeric else float(v) for k,v in row.items()}
    for k in ('train_seed','checkpoint_step','cohort','horizon','seed'):result[k]=int(result[k])
    return result

manifest=load('manifest.json');cfg=manifest['config']
assert manifest['status']=='COMPLETED' and not manifest['dirty']
assert manifest['commit']=='82707dbfc81aeeb40c6cd0ad296c4942ab9a1ce6'
assert cfg==runner.settings('full') and manifest['actual_counts']==runner.counts(cfg)
for name,meta in tqdm(manifest['artifacts'].items(),desc='raw artifact hashes'):
    p=RAW/name;assert p.stat().st_size==meta['bytes'] and hash_file(p)==meta['sha256'],name
for p in (CHECKOUT/'src/ht_pdm_fjsp').glob('maintenance_contention*.py'):
    assert hash_file(p)==hash_file(RAW/'source_snapshot'/p.name)
audit,forbidden=runner.seed_audit(cfg)
assert audit==load('seed_audit.json')
budget=load('budget.json')
assert budget['teacher_labels']==0
assert hash_file(RAW/'episodes.csv')==hash_file(RAW/'episodes.partial.csv')
episodes=[numeric(r) for r in rows('episodes.csv')]
development=[numeric(r) for r in rows('development_episodes.csv')]
assert len(episodes)==45000 and len(development)==22400
selection=load('reference_selection.json');selected_refs=selection['selected']
reference_dev=[r for r in development if r['train_seed']==-1]
assert len(reference_dev)==2400
for cap in cfg['capacities']:
    means={name:runner.mean([r for r in reference_dev if r['capacity']==cap and r['controller']==name],'objective') for name in cfg['controls'][:-1]}
    assert means==selection['development_means'][cap]
    assert min(means,key=lambda n:(means[n],n))==selected_refs[cap]
selected_dev=[r for r in reference_dev if r['controller']==selected_refs[r['capacity']]]
checkpoints=load('checkpoint_selection.json')['selections']
models={};curves=[];parameter_changes={}
for seed in cfg['train_seeds']:
    meta=checkpoints[str(seed)]
    records=[];initial_state=None;changes=[]
    for candidate in meta['candidates']:
        model,payload=runner.load_checkpoint(RAW/candidate['path'])
        assert payload['physical_steps']==candidate['steps'] and payload['train_seed']==seed
        if candidate['steps']==0:initial_state={k:v.clone() for k,v in model.state_dict().items()}
        else:
            changes.append(float(sum((v-initial_state[k]).square().sum() for k,v in model.state_dict().items()).sqrt()))
        panel=[r for r in development if r['train_seed']==seed and r['checkpoint_step']==candidate['steps']]
        assert len(panel)==400
        expected_dev=set(itertools.product(cfg['development_cohorts'],('middle','low'),cfg['development_seeds']))
        assert {(r['cohort'],r['capacity'],r['seed']) for r in panel}==expected_dev
        records.append({**candidate,'episodes':panel})
        for cap in ('middle','low'):
            own=[r for r in panel if r['capacity']==cap]
            curves.append(dict(train_seed=seed,step=candidate['steps'],capacity=cap,**{metric:runner.mean(own,metric) for metric in ('objective','base_cost','waiting_violation','failures','utilization')}))
    chosen,status=runner.select_checkpoint(records,selected_dev)
    assert status==meta['status'] and chosen['steps']==meta['selected_steps']
    assert abs(chosen['development_primary_cost']-meta['development_cost'])<1e-8
    assert hash_file(RAW/meta['checkpoint'])==meta['checkpoint_sha256']==hash_file(RAW/chosen['path'])
    models[seed],payload=runner.load_checkpoint(RAW/meta['checkpoint'])
    assert payload['physical_steps']==meta['selected_steps']
    assert all(x>0 for x in changes);parameter_changes[str(seed)]=changes

# Independently recompute training configuration and counter-derived seed roles,
# exact mixtures, complete episode costs, progress counters and finite statistics.
training_count=defaultdict(int);mix=defaultdict(lambda:defaultdict(int));profile_sets=defaultdict(set)
with (RAW/'training_episodes.csv').open() as stream:
    for row in tqdm(csv.DictReader(stream),total=409600,desc='training provenance'):
        seed=int(row['train_seed']);index=int(row['episode'])
        training_count[seed]+=1;assert index==training_count[seed]
        c,cap,cs,es=episode_spec(seed,index,12,forbidden)
        assert cs==int(row['config_seed']) and es==int(row['environment_seed']) and cap==row['capacity']
        digest=hashlib.sha256(json.dumps(asdict(c),sort_keys=True).encode()).hexdigest()
        assert digest==row['config_sha256']
        profile_sets[seed].add(digest);mix[seed][cap]+=1
        assert int(row['rollout_update'])==math.ceil(index/128)
        assert int(row['physical_steps'])==math.ceil(index/32)*32*12
        assert abs(float(row['objective'])-float(row['base_cost'])-float(row['waiting_cost']))<1e-8
        assert abs(float(row['base_cost'])-sum(float(row[k]) for k in ('maintenance_cost','failure_cost','unavailability_cost')))<1e-8
for record in load('training_coverage.json'):
    seed=record['train_seed']
    assert training_count[seed]==40960 and dict(mix[seed])==record['mixture_counts']==dict(low=8192,middle=32768)
    assert record['env_steps']==491520 and record['optimizer_steps']==10240 and record['teacher_label_rows']==0
progress=rows('training_progress.csv');assert len(progress)==3200
for seed in cfg['train_seeds']:
    panel=[r for r in progress if int(r['train_seed'])==seed]
    assert [int(r['rollout_update']) for r in panel]==list(range(1,321))
    for r in panel:
        update=int(r['rollout_update'])
        assert int(r['physical_steps'])==update*1536 and int(r['episodes'])==update*128 and int(r['optimizer_steps'])==update*32
        assert all(math.isfinite(float(v)) for v in r.values())
seen_profiles={str(seed):sum(hashlib.sha256(json.dumps(asdict(cohort_config(cs,'middle',12)),sort_keys=True).encode()).hexdigest() in profile_sets[seed] for cs in cfg['test_cohorts']) for seed in cfg['train_seeds']}

paired,cohorts,contrasts=runner.summarize(episodes,selected_refs,cfg)
saved=load('summary.json')
assert contrasts==saved['contrasts']
for row in episodes:
    assert row['invalid_assignments']==0 and abs(row['objective']-row['base_cost']-row['waiting_cost'])<1e-8
    assert abs(row['base_cost']-sum(row[k] for k in ('maintenance_cost','unavailability_cost','failure_cost')))<1e-8
    assert abs(row['waiting_cost']-12*row['overdue_waiting_ticks'])<1e-8

groups={}
metrics=[k for k in episodes[0] if k not in ('train_seed','checkpoint_step','cohort','capacity','horizon','seed','controller')]
for cap,h,name in itertools.product(cfg['capacities'],cfg['horizons'],[*cfg['controls'],runner.LEARNER]):
    panel=[r for r in episodes if r['capacity']==cap and r['horizon']==h and r['controller']==name]
    groups[f'{cap}_H{h}_{name}']={m:runner.mean(panel,m) for m in metrics}

# Scalar physical replay of every recorded transition. Neural inference is
# batched separately over the same logged pre-states for all 540000 learned
# actions, respecting deterministic near-ties and categorical log probability.
counts=defaultdict(int);totals={};state_key=None;env=None
neural=[];neural_count=0;action_mismatches=[];max_lp_error=max_value_error=0.0
forecast_replay=0

def flush_neural():
    global neural_count,max_lp_error,max_value_error
    if not neural:return
    seed=neural[0]['train_seed'];model=models[seed]
    configs=[r['config'] for r in neural]
    batch=feature_batch(configs,[r['compact'] for r in neural],[r['remaining'] for r in neural])
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
    for row in tqdm(csv.DictReader(stream),total=810000,desc='physical + checkpoint replay'):
        k=key(row);t=int(row['time'])
        if k!=state_key:
            env=WaitingEnv(cohort_config(k[1],k[2],k[3]));env.reset(k[4]);state_key=k
        assert t==counts[k]
        assert asdict(env.state)=={name:tuple(value) for name,value in json.loads(row['state']).items()}
        diag=json.loads(row['controller_diagnostics']);pairs=tuple(tuple(p) for p in json.loads(row['action']))
        state=compact(env.state)
        if k[-1]==runner.LEARNER:
            if neural and (neural[0]['train_seed']!=k[0] or len(neural)>=512):flush_neural()
            assert int(row['checkpoint_step'])==checkpoints[str(k[0])]['selected_steps']
            assert physical_action(env.state,subsets(4)[diag['action_index']])==pairs
            neural.append(dict(train_seed=k[0],config=env.config,compact=state,remaining=k[3]-t,diagnostics=diag,key=k,time=t))
        elif k[-1]=='joint_rollout':
            chosen=diag['candidates'][int(np.argmin(diag['means']))]
            assert physical_action(env.state,chosen)==pairs
            assert diag['scenario_seed']==role_seed(k[4],f'contention_planner:{t}')
            if k[4]==cfg['test_seeds'][0]:
                selected,replay=plan(env.config,state,k[3]-t,selected_refs[k[2]],diag['scenario_seed'],cfg['scenarios'])
                assert json.loads(json.dumps(replay))==diag;forecast_replay+=1
        else:
            assert physical_action(env.state,reference(env.config,state,k[3]-t,k[-1]))==pairs
        _,_,_,info=env.step(pairs)
        assert info==json.loads(row['physical_metrics'])
        assert asdict(env.state)=={name:tuple(value) for name,value in json.loads(row['following']).items()}
        counts[k]+=1;totals[k]=dict(env.metrics)
flush_neural()
assert sum(counts.values())==810000 and len(counts)==45000 and neural_count==540000
for row in episodes:
    k=key(row);assert counts[k]==k[3]
    for metric,val in totals[k].items():assert abs(val-row[metric])<1e-8

machine_panels=defaultdict(list);machine_by_duration=defaultdict(list)
with (RAW/'machine_metrics.csv').open() as stream:
    for row in csv.DictReader(stream):
        k=key(row);machine_panels[k].append(row)
        if k[3]==12:
            c=cohort_config(k[1],k[2],k[3]);duration=c.service_time[int(row['machine'])][0]
            machine_by_duration[(k[2],k[-1],duration)].append(row)
assert sum(map(len,machine_panels.values()))==180000
for row in episodes:
    panel=machine_panels[key(row)];assert sorted(int(r['machine']) for r in panel)==list(range(4))
    for metric in ('unavailable_ticks','failed_waiting_ticks','overdue_waiting_ticks','terminal_pending','terminal_failed'):
        assert sum(float(r[metric]) for r in panel)==row[metric]
    assert sum(float(r['service_starts']) for r in panel)==row['jobs']
by_duration={f'{cap}_{name}_duration{d}':{m:float(np.mean([float(r[m]) for r in panel])) for m in ('service_starts','unavailable_ticks','failed_waiting_ticks','overdue_waiting_ticks','max_wait','terminal_pending','violation')} for (cap,name,d),panel in machine_by_duration.items()}

for name,meta in manifest['artifacts'].items():assert hash_file(RAW/name)==meta['sha256']
dump('analysis.json',dict(raw=str(RAW),commit=manifest['commit'],started=manifest['started'],finished=manifest['finished'],contrasts=contrasts,groups=groups,paired_seeds=paired,cohorts=cohorts,development_curves=curves,selected_steps={s:r['selected_steps'] for s,r in checkpoints.items()},training_progress=progress,seen_primary_profiles_by_training_seed=seen_profiles,machine_by_duration=by_duration,audit=dict(status='PASS' if not action_mismatches else 'NUMERICAL_ACTION_MISMATCH',raw_files=len(manifest['artifacts']),models=10,training_episode_provenance=409600,optimizer_progress_rows=3200,checkpoint_files=60,development_episodes=22400,test_episodes=45000,physical_intervals=810000,reference_action_intervals=216000,planner_candidate_action_intervals=54000,planner_forecast_replay_intervals=forecast_replay,learned_action_replay_intervals=neural_count,machine_rows=180000,neural_action_mismatches=action_mismatches,max_log_probability_error=max_lp_error,max_value_error=max_value_error,raw_hashes_unchanged=True,elapsed_seconds=perf_counter()-START)))
print(json.dumps(dict(status='PASS' if not action_mismatches else 'NUMERICAL_ACTION_MISMATCH',neural_action_mismatches=len(action_mismatches),max_log_probability_error=max_lp_error,max_value_error=max_value_error,elapsed_seconds=perf_counter()-START)))
