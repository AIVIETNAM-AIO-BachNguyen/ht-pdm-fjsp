"""Read-only raw-result audit and descriptive analysis; no model selection."""
import csv
from collections import defaultdict
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from tqdm.auto import tqdm

CHECKOUT = Path('/private/tmp/marl-oracle-representation-20261003')
sys.path.insert(0, str(CHECKOUT / 'src'))
from ht_pdm_fjsp.maintenance_contention_model import cohort_config, compact, physical_action, reference, plan
from ht_pdm_fjsp.maintenance_contention_headroom import settings, seed_audit
from ht_pdm_fjsp.maintenance_contention_oracle import run_exact
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

ROOT = Path('/Users/bachng/Coding/Reinforcement Learning/marl')
RAW = ROOT / 'lab_results/maintenance_contention_headroom_v1_full_cpu_20261005T183911Z'
OUT = Path(__file__).resolve().parent

def load(name):
    return json.loads((RAW / name).read_text())

def csv_rows(name):
    with (RAW / name).open() as stream:
        return list(csv.DictReader(stream))

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item()) + '\n')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

manifest = load('manifest.json')
assert manifest['status'] == 'COMPLETED'
assert manifest['commit'] == '89ccb60152648f0201e11c8df5dbd7fc02d6bc55'
assert not manifest['dirty']
for name, meta in tqdm(manifest['artifacts'].items(), desc='artifact hashes'):
    path = RAW / name
    assert path.stat().st_size == meta['bytes'] and digest(path) == meta['sha256'], name
for path in (CHECKOUT / 'src/ht_pdm_fjsp').glob('maintenance_contention*.py'):
    assert digest(path) == digest(RAW / 'source_snapshot' / path.name)
assert seed_audit(settings('full'))['status'] == 'PASS'
cfg = manifest['config']
assert cfg == settings('full')
selection = load('reference_selection.json')
selected = selection['selected']
episodes, development, exact_jobs, exact_rows = (csv_rows(name) for name in ('episodes.csv', 'development_episodes.csv', 'exact_jobs.csv', 'exact_metrics.csv'))
assert len(episodes) == 15000 and len(development) == 2400 and len(exact_jobs) == 30 and len(exact_rows) == 120
assert digest(RAW/'episodes.csv') == digest(RAW/'episodes.partial.csv')
expected = set(itertools.product(cfg['cohorts'], cfg['capacities'], cfg['horizons'], cfg['test_seeds'], cfg['controllers']))
key = lambda r: (int(r['cohort']), r['capacity'], int(r['horizon']), int(r['seed']), r['controller'])
assert len({key(r) for r in episodes}) == len(episodes) and {key(r) for r in episodes} == expected
for cap in cfg['capacities']:
    means = {name: np.mean([float(r['objective']) for r in development if r['capacity']==cap and r['controller']==name]) for name in cfg['controllers'][:-1]}
    assert min(means, key=lambda n: (means[n], n)) == selected[cap]
    for n, val in means.items():
        assert abs(val-selection['development_means'][cap][n]) < 1e-8
for row in episodes:
    assert float(row['invalid_assignments']) == 0
    assert abs(float(row['objective'])-float(row['base_cost'])-float(row['waiting_cost'])) < 1e-8
    assert abs(float(row['base_cost'])-sum(float(row[k]) for k in ('maintenance_cost','unavailability_cost','failure_cost'))) < 1e-8
    assert abs(float(row['waiting_cost'])-12*float(row['overdue_waiting_ticks'])) < 1e-8

metrics = [k for k in episodes[0] if k not in ('cohort','capacity','horizon','seed','controller')]
groups = {}
for cap, h, name in itertools.product(cfg['capacities'], cfg['horizons'], cfg['controllers']):
    panel = [r for r in episodes if r['capacity']==cap and int(r['horizon'])==h and r['controller']==name]
    groups[f'{cap}_H{h}_{name}'] = {k: float(np.mean([float(r[k]) for r in panel])) for k in metrics}

hypotheses = {}
cohort_rows = []
for cap, h in itertools.product(cfg['capacities'], cfg['horizons']):
    delta, bc, pc, bb, pb = [], [], [], [], []
    for cohort in cfg['cohorts']:
        panels = [[r for r in episodes if int(r['cohort'])==cohort and r['capacity']==cap and int(r['horizon'])==h and r['controller']==name] for name in (selected[cap], 'joint_rollout')]
        baseline, planner = [np.mean([float(r['objective']) for r in p]) for p in panels]
        cb, cp = [np.mean([float(r['base_cost']) for r in p]) for p in panels]
        d = planner-baseline
        delta.append(d); bc.append(baseline); pc.append(planner); bb.append(cb); pb.append(cp)
        cohort_rows.append(dict(capacity=cap,horizon=h,cohort=cohort,baseline=float(baseline),planner=float(planner),reduction=float(1-planner/baseline),difference=float(d)))
    mean, se = float(np.mean(delta)), float(np.std(delta,ddof=1)/np.sqrt(10))
    ci = [mean-3.2498355415921254*se,mean+3.2498355415921254*se]
    result = dict(reference=selected[cap],reference_priced_cost=float(np.mean(bc)),planner_priced_cost=float(np.mean(pc)),paired_difference=mean,ci99=ci,reduction=float(1-np.mean(pc)/np.mean(bc)),cohort_wins=sum(d < -1e-9 for d in delta),base_cost_ratio=float(np.mean(pb)/np.mean(bb)),cohort_differences=delta)
    summary = load('summary.json')['hypotheses'][f'{cap}_H{h}']
    for k in ('paired_difference','reduction','base_cost_ratio','reference_priced_cost','planner_priced_cost'):
        assert abs(summary[k]-result[k]) < 1e-8
    assert np.allclose(summary['ci99'], ci) and summary['cohort_wins']==result['cohort_wins']
    hypotheses[f'{cap}_H{h}'] = result

# Replay ALL recorded physical trajectories and reference actions. Independent
# environment RNG is reconstructed from the sealed seed, never from the trace.
counts = defaultdict(int)
totals = {}
env = None
current_key = None
planner_replay = 0
start = perf_counter()
with (RAW/'decisions.csv').open() as stream:
    for row in tqdm(csv.DictReader(stream), total=270000, desc='scalar/action replay'):
        k = key(row)
        if k != current_key:
            env = WaitingEnv(cohort_config(k[0],k[1],k[2]))
            env.reset(k[3])
            current_key=k
        t = int(row['time'])
        assert t == counts[k]
        assert asdict(env.state) == {field: tuple(value) for field,value in json.loads(row['state']).items()}
        diag=json.loads(row['planner'])
        if k[-1] != 'joint_rollout':
            action = physical_action(env.state, reference(env.config,compact(env.state),k[2]-t,k[-1]))
            assert [list(x) for x in action] == json.loads(row['action'])
        else:
            candidates = diag['candidates']
            means = diag['means']
            assert len(candidates)==len(means)==len(diag['mc_se'])
            chosen=candidates[int(np.argmin(means))]
            assert [list(x) for x in physical_action(env.state,chosen)] == json.loads(row['action'])
            assert diag['scenario_seed']==role_seed(k[3],f'contention_planner:{t}')
            if k[3] == cfg['test_seeds'][0]:
                chosen, replay = plan(env.config,compact(env.state),k[2]-t,selected[k[1]],diag['scenario_seed'],cfg['scenarios'])
                assert json.loads(json.dumps(replay))==diag
                planner_replay += 1
        _,_,_,info=env.step(tuple(tuple(pair) for pair in json.loads(row['action'])))
        assert info==json.loads(row['physical_metrics'])
        assert asdict(env.state)=={field: tuple(value) for field,value in json.loads(row['following']).items()}
        counts[k]+=1
        totals[k]=dict(env.metrics)
assert len(counts)==15000 and sum(counts.values())==270000
for row in episodes:
    k=key(row)
    assert counts[k]==k[2]
    for metric,value in totals[k].items():
        assert abs(value-float(row[metric]))<1e-8

machines = defaultdict(list)
with (RAW/'machine_metrics.csv').open() as stream:
    for row in csv.DictReader(stream):
        machines[key(row)].append(row)
assert sum(map(len,machines.values()))==60000
for row in episodes:
    panel=machines[key(row)]
    assert sorted(int(r['machine']) for r in panel)==list(range(4))
    for metric in ('unavailable_ticks','failed_waiting_ticks','overdue_waiting_ticks','terminal_pending','terminal_failed'):
        assert sum(float(r[metric]) for r in panel)==float(row[metric])
    assert sum(float(r['service_starts']) for r in panel)==float(row['jobs'])
    assert max(float(r['max_wait']) for r in panel)==float(row['max_wait'])

# Check every exported state has a solved value, min-Q consistency and the
# exact panel's 27 initial states; reproduce one H6 job at each capacity.
for job in tqdm(exact_jobs, desc='exact graph integrity'):
    assert job['status']=='COMPLETED' and int(job['states'])==int(job['solved_states']) and float(job['bellman_residual'])<1e-7
    path=RAW/'oracle_tables'/f"{job['cohort']}_{job['capacity']}.jsonl.gz"
    initials=[]; n=0
    with gzip.open(path,'rt') as stream:
        for line in stream:
            node=json.loads(line); n+=1
            assert node['status']=='SOLVED' and abs(node['value']-min(node['q']))<1e-8
            if node['horizon_remaining']==6 and all(r[1:]==[0,0,0] and r[0]<3 for r in node['state']):
                initials.append(node['value'])
    assert n==int(job['states']) and len(initials)==27 and abs(np.mean(initials)-float(job['joint_optimum']))<1e-8
for cap in cfg['capacities']:
    job, rows = run_exact(cohort_config(134000,cap,6,3),OUT/f'reproduced_exact_134000_{cap}.jsonl.gz',progress=False)
    saved=[r for r in exact_rows if r['cohort']=='134000' and r['capacity']==cap]
    for result in rows:
        old=next(r for r in saved if r['controller']==result['controller'])
        for metric in ('priced_cost','joint_optimum','relaxed_lower_bound','unavoidable_scarcity_cost','avoidable_decision_cost','early_regret','late_regret'):
            assert abs(result[metric]-float(old[metric]))<1e-8

exact_groups={}
for cap, name in itertools.product(cfg['capacities'],cfg['controllers'][:-1]):
    panel=[r for r in exact_rows if r['capacity']==cap and r['controller']==name]
    exact_groups[f'{cap}_{name}']={metric:float(np.mean([float(r[metric]) for r in panel])) for metric in ('priced_cost','joint_optimum','relaxed_lower_bound','unavoidable_scarcity_cost','avoidable_decision_cost','early_regret','late_regret')}

# No writes to raw files. Final hash pass binds the report to unchanged input.
for name, meta in manifest['artifacts'].items():
    assert digest(RAW/name)==meta['sha256']
write('analysis.json',dict(raw=str(RAW),commit=manifest['commit'],runtime_started=manifest['started'],runtime_finished=manifest['finished'],groups=groups,hypotheses=hypotheses,cohorts=cohort_rows,exact_groups=exact_groups,audit=dict(status='PASS',raw_files=len(manifest['artifacts']),episodes=15000,development=2400,decision_intervals=270000,reference_action_replay_intervals=216000,planner_diagnostic_action_checks=54000,planner_forecast_replay_intervals=planner_replay,machine_rows=60000,exact_jobs=30,exact_policy_rows=120,exact_states=sum(int(j['states']) for j in exact_jobs),exact_jobs_independently_reproduced=3,zero_mismatches=True,raw_hashes_unchanged=True,audit_seconds=perf_counter()-start)))
print(json.dumps(dict(status='PASS',output=str(OUT),planner_replay=planner_replay,audit_seconds=perf_counter()-start)))
