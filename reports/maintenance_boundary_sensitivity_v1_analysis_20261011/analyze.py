"""Read-only audit and independent analysis of downloaded boundary results."""
import csv
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import types

sys.dont_write_bytecode = True
import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'lab_results/maintenance_boundary_sensitivity_v1_full_cpu_20261010T101822Z'
OUT = Path(__file__).resolve().parent


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csvrows(name):
    with (RUN/name).open() as f:
        yield from csv.DictReader(f)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    if a is None or b is None:
        assert a is b, (a,b)
    else:
        assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-8), (a,b)


def key(r):
    return r['condition'],int(r['horizon']),int(r['cohort']),int(r['shock']),r['controller']


def numeric(r):
    return {k:v if k in ('condition','controller') else None if v=='' else float(v) for k,v in r.items()}


def ints(v):
    if isinstance(v,dict):
        return set().union(set(),*(ints(x) for x in v.values()))
    if isinstance(v,list):
        return set().union(set(),*(ints(x) for x in v))
    return {v} if isinstance(v,int) and not isinstance(v,bool) else set()


m=json.loads((RUN/'manifest.json').read_text())
assert m['status']=='COMPLETED' and m['audit']=='PASS' and not m['dirty']
assert m['commit']=='54d17f9ce9c195c59ec6745868ae9b3450c0b1c5'
assert m['device']=='cpu' and all(v=='1' for v in m['thread_environment'].values())
hashes={}
for name,meta in m['artifacts'].items():
    assert (RUN/name).stat().st_size==meta['bytes'] and digest(RUN/name)==meta['sha256'], name
    hashes[name]=meta['sha256']
assert set(hashes)=={str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file() and p.name!='manifest.json'}
source={}
for p in (RUN/'source_snapshot').iterdir():
    path='src/ht_pdm_fjsp/'+p.name if p.suffix=='.py' else 'configs/'+p.name if p.suffix=='.json' else p.name
    committed=subprocess.check_output(['git','show',f"{m['commit']}:{path}"],cwd=ROOT)
    assert committed==p.read_bytes(), path
    source[path]=hashlib.sha256(committed).hexdigest()
assert subprocess.check_output(['git','show',f"{m['commit']}:docs/maintenance_boundary_sensitivity_v1_plan.md"],cwd=ROOT)==(RUN/'plan.md').read_bytes()
package=types.ModuleType('ht_pdm_fjsp');package.__path__=[str(RUN/'source_snapshot')]
sys.modules['ht_pdm_fjsp']=package
model=importlib.import_module('ht_pdm_fjsp.maintenance_boundary_sensitivity')
physical=importlib.import_module('ht_pdm_fjsp.maintenance_dispatch')
waiting=importlib.import_module('ht_pdm_fjsp.maintenance_waiting')
cfg=m['settings']; assert cfg==model.settings('full')
assert m['actual_counts']==m['expected_counts']==model.counts(cfg)
seed_panels=[set(cfg[k]) for k in ('development_cohorts','development_shocks','test_cohorts','test_shocks')]+[{cfg['analysis_seed']}]
assert not any(a&b for a,b in itertools.combinations(seed_panels,2))
historical=set()
for name,sha in m['seed_audit']['historical_registry_hashes'].items():
    b=subprocess.check_output(['git','show',f"{m['commit']}:configs/{name}"],cwd=ROOT)
    assert hashlib.sha256(b).hexdigest()==sha
    historical|=ints(json.loads(b))
exclusion_path=RUN/'source_snapshot/maintenance_boundary_sensitivity_v1_profile_exclusions.json'
exclusion=json.loads(exclusion_path.read_text())
historical|=ints(exclusion['previous_seed_panels'])
assert not set.union(*seed_panels)&historical
assert digest(exclusion_path)==m['seed_audit']['prior_physical_exclusion_sha256']
own_path=RUN/'source_snapshot/maintenance_boundary_sensitivity_v1_seed_registry.json'
own=json.loads(own_path.read_text())['panels']
assert digest(own_path)==m['seed_audit']['raw_seed_registry_sha256']
assert all(cfg[k]==v for k,v in own['full'].items()) and not ints(own['full'])&ints(own['smoke'])
resolved=json.loads((RUN/'resolved_config.json').read_text());assert resolved['settings']==cfg
configs={};signatures=defaultdict(set)
for entry in resolved['physical_profiles']:
    data=entry['config'].copy()
    for field in ('service_time','restoration'):
        data[field]=tuple(tuple(row) for row in data[field])
    c=waiting.WaitingConfig(**data);c.validate()
    assert c==model.prior.make_config(entry['cohort'],entry['condition'],48,entry['generator_attempt'])
    sig=model.signature(c)
    assert sig not in {model.frozen(s) for s in exclusion['previous_physical_signatures'][entry['condition']]}
    assert sig not in signatures[entry['condition'],entry['split']]
    signatures[entry['condition'],entry['split']].add(sig)
    configs[entry['split'],entry['cohort'],entry['condition']]=c
assert len(configs)==240
for condition in cfg['conditions']:
    assert len(signatures[condition,'development'])==16 and len(signatures[condition,'test'])==64
    assert not signatures[condition,'development']&signatures[condition,'test']
episodes={}
for r in csvrows('episodes.csv'):
    assert key(r) not in episodes;episodes[key(r)]=numeric(r)
expected=set(itertools.product(cfg['conditions'],cfg['horizons'],cfg['test_cohorts'],cfg['test_shocks'],cfg['controllers']))
assert set(episodes)==expected and len(episodes)==23040
assert hashes['episodes.csv']==hashes['episodes.partial.csv']
development=list(csvrows('development_episodes.csv'))
assert len(development)==4032
assert {key(r) for r in development}==set(itertools.product(cfg['conditions'],cfg['horizons'],cfg['development_cohorts'],cfg['development_shocks'],cfg['rules']))
chosen=json.loads((RUN/'development_selection.json').read_text());assert chosen['locked_before_test']
for condition,h in itertools.product(cfg['conditions'],cfg['horizons']):
    cell=f'{condition}:{h}'; metric='interior_cost_per_machine_tick' if h==48 else 'extended_objective'
    scores={name:float(np.mean([float(r[metric]) for r in development if r['condition']==condition and int(r['horizon'])==h and r['controller']==name])) for name in cfg['rules']}
    assert chosen['selected'][cell]['best_rule']==min(cfg['rules'],key=scores.get)
    assert chosen['selected'][cell]['metric']==metric
    for name,v in scores.items():close(v,chosen['selected'][cell]['scores'][name])
coordination=list(csvrows('coordination.csv'))
assert len(coordination)==23040 and {key(r) for r in coordination}==expected
assert all(r['invalid_actions']=='0' and r['scalar_batch_replay']=='PASS' and int(r['base_decisions'])==int(r['horizon']) and int(r['tail_decisions'])==12 for r in coordination)
wall=(datetime.fromisoformat(m['finished'])-datetime.fromisoformat(m['started'])).total_seconds()
write('integrity_audit.json',dict(status='PASS',artifact_files=len(hashes),hashes=hashes,source_files=source,
    commit=m['commit'],counts=m['actual_counts'],wall_seconds=wall,development_selection_reproduced=True,
    old_new_and_development_test_profiles_disjoint=True,raw_seeds_disjoint=True,
    full_test_opened_on_lab=True,raw_artifacts_modified=False))
print('Integrity/source/split/counts/development selection PASS; wall hours:',wall/3600,flush=True)

# Independently rebuild all 27 planned contrasts, keeping profiles as the bootstrap unit.
original=json.loads((RUN/'summary.json').read_text());means={};contrasts={};profile_diffs={}
for condition,h in itertools.product(cfg['conditions'],cfg['horizons']):
    cell=f'{condition}:{h}'
    means[cell]={name:{metric:float(np.mean([episodes[condition,h,c,s,name][metric] for c in cfg['test_cohorts'] for s in cfg['test_shocks']])) for metric in original['means'][cell][name]} for name in cfg['controllers']}
    for name in cfg['controllers']:
        for metric,v in means[cell][name].items():close(v,original['means'][cell][name][metric])
    defs=(('pending_terminal','fixed_zero','fixed_tail','cutoff_pending',h==12),
          ('interior_headroom','best_rule','fixed_tail','interior_cost_per_machine_tick',h==48),
          ('interior_assignment','fixed_tail','flexible_tail','interior_cost_per_machine_tick',h==48))
    for label,left,right,metric,target in defs:
        primary=condition=='skill_mask' and target; confidence=1-.05/3 if primary else .95
        a=np.array([np.mean([episodes[condition,h,c,s,left][metric] for s in cfg['test_shocks']]) for c in cfg['test_cohorts']])
        b=np.array([np.mean([episodes[condition,h,c,s,right][metric] for s in cfg['test_shocks']]) for c in cfg['test_cohorts']])
        rng=np.random.default_rng(physical.role_seed(cfg['analysis_seed'],f'{cell}:{label}'))
        draws=rng.integers(64,size=(20000,64));diff=(a-b)[draws].mean(1)
        tails=[(1-confidence)/2,(1+confidence)/2];ci=np.quantile(diff,tails).tolist()
        relative=float((a-b).mean()/a.mean());relative_ci=np.quantile(diff/a[draws].mean(1),tails).tolist()
        screen=ci if label=='pending_terminal' else relative_ci;margin=.1 if label=='pending_terminal' else .01
        status='PASS_RESEARCH_MARGIN' if screen[0]>margin else 'BELOW_RESEARCH_MARGIN' if screen[1]<margin else 'INCONCLUSIVE'
        v=dict(primary=primary,confidence=confidence,mean_difference=float((a-b).mean()),confidence_interval=ci,
            relative_improvement=relative,relative_confidence_interval=relative_ci,profile_wins=int(np.sum(a>b)),
            profile_ties=int(np.sum(a==b)),profile_losses=int(np.sum(a<b)),status=status)
        name=f'{cell}:{label}';o=original['contrasts'][name]
        for field in ('primary','confidence','profiles','status'):
            assert o[field]==(64 if field=='profiles' else v[field])
        for field in ('mean_difference','relative_improvement','profile_wins','profile_ties'):close(v[field],o[field])
        np.testing.assert_allclose(v['confidence_interval'],o['confidence_interval'],rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(v['relative_confidence_interval'],o['relative_confidence_interval'],rtol=1e-10,atol=1e-8)
        contrasts[name]=v
        for c,x,y in zip(cfg['test_cohorts'],a,b):profile_diffs[condition,h,label,c]=(x,y,x-y)
for r in csvrows('profile_differences.csv'):
    for x,field in zip(profile_diffs[r['condition'],int(r['horizon']),r['contrast'],int(r['cohort'])],('left_mean','right_mean','difference')):close(x,r[field])
x,y=means['skill_mask:48']['fixed_tail'],means['skill_mask:48']['best_rule']
guards=dict(pending=x['interior_pending_mean']<=y['interior_pending_mean']+.1,
    total_failed=x['interior_failed_mean']<=y['interior_failed_mean']+.1,
    overdue=x['interior_overdue_per_tick']<=y['interior_overdue_per_tick']+.25/12)
screen='PASS' if contrasts['skill_mask:48:interior_headroom']['status']=='PASS_RESEARCH_MARGIN' and all(guards.values()) else 'NOT_ESTABLISHED'
assert guards==original['service_guard'] and screen==original['timing_learning_screen']
write('independent_statistics.json',dict(verification_status='ANALYZED',means=means,contrasts=contrasts,service_guard=guards,
    timing_learning_screen=screen,bootstrap_scope='64 paired profiles conditional on fixed 8-shock panel; 20,000 draws',
    multiplicity='3 primary contrasts with 98.333333% intervals; other 24 exploratory at 95%',
    physical_replay_pending=True))
print('All 27 CIs and 3 primary screens independently reproduced',flush=True)
for name,v in contrasts.items():
    if v['primary']:print(name,json.dumps(v),flush=True)
print('Service guard:',guards,'screen:',screen,flush=True)

# Independent request tracking, cost accumulation, states and time profiles.
def snap(state):
    busy={m for m in state.assigned if m>=0}
    return dict(pending=sum(f and i not in busy for i,f in enumerate(state.failed)),failed=sum(state.failed),
                servicing=len(busy),cm_in_service=sum(state.failed[i] for i in busy))


def request_stats(rows):
    served=[r['wait'] for r in rows if r['served']]
    return dict(requests=len(rows),served_requests=len(served),censored_requests=sum(r['censored'] for r in rows),
                overdue_requests=sum(r['overdue'] for r in rows),observed_wait=sum(r['wait'] for r in rows),
                served_wait_mean=float(np.mean(served)) if served else None,
                served_wait_p95=float(np.quantile(served,.95,method='inverted_cdf')) if served else None)


def finish_requests(served,pending,end):
    return [*served,*[dict(machine=i,request=n,arrival=t,end=end,wait=end-t,served=False,censored=True,
                         overdue=end-t>4) for i,(t,n) in pending.items()]]


def check_requests(records,k,scope):
    global request_count
    for record in records:
        raw=next(request_stream)
        assert key(raw)==k and raw['scope']==scope
        for name,value in record.items():
            assert (raw[name]=='True' if isinstance(value,bool) else int(raw[name]))==value,(k,name,raw,record)
        request_count+=1


request_stream=iter(csvrows('requests.csv'));carry_stream=iter(csvrows('carryover.csv'))
request_count=carry_count=rule_reproduced=decision_count=0
seen=set();current=None;time_totals=defaultdict(lambda:defaultdict(float))
for row in tqdm(csvrows('decisions.csv'),total=921600,desc='Replay all physical decisions'):
    k=key(row)
    if k!=current:
        if current is not None:assert env.time==h+12
        assert k not in seen;seen.add(k);current=k
        condition,h,cohort,shock,controller=k
        c=replace(configs['test',cohort,condition],horizon=h+12)
        env=waiting.WaitingEnv(c);env_seed=physical.role_seed(shock,f'{model.VERSION}:physical:{cohort}');env.reset(env_seed)
        pending={};sequence=[0]*c.machines;served=[];interior=defaultdict(float)
        if any(env.state.failed):raise AssertionError('unexpected initially failed machine')
    t=int(row['time']);phase='base' if t<h else 'tail'
    assert t==env.time and row['phase']==phase and int(row['env_seed'])==env_seed
    assert int(row['forecast_seed'])==physical.role_seed(shock,f'{model.VERSION}:forecast:{cohort}:{t}')
    state=env.state
    assert asdict(state)=={f:tuple(v) for f,v in json.loads(row['state']).items()}
    action=tuple(tuple(p) for p in json.loads(row['action']))
    physical.validate_matching(c,state,action)
    if phase=='base' and controller.startswith('fixed'):assert action in model.fixed_candidates(c,state)
    if phase=='tail' or controller=='best_rule':
        rule=model.TAIL_RULE if phase=='tail' else chosen['selected'][f'{condition}:{h}']['best_rule']
        assert action==model.prior.rule_action(c,state,c.horizon-t if phase=='tail' else h-t,rule)
        rule_reproduced+=1
    for i,_ in action:
        if state.failed[i]:
            arrival,n=pending.pop(i);wait=t-arrival
            served.append(dict(machine=i,request=n,arrival=arrival,end=t,wait=wait,served=True,censored=False,overdue=wait>4))
    _,_,done,info=env.step(action)
    assert info==json.loads(row['physical_metrics'])
    assert asdict(env.state)=={f:tuple(v) for f,v in json.loads(row['following']).items()}
    servicing=set(state.assigned)|{i for i,_ in action}
    for i,failed in enumerate(env.state.failed):
        if failed and not state.failed[i] and i not in servicing:
            assert i not in pending;sequence[i]+=1;pending[i]=(t+1,sequence[i])
    totals=time_totals[condition,h,controller,t,phase];totals['episodes']+=1
    for metric,value in info.items():totals[metric]+=value
    for prefix,s in (('before',state),('after',env.state)):
        for metric,value in snap(s).items():totals[f'{prefix}_{metric}']+=value
    if h//4<=t<h-h//4:
        for metric in ('objective','maintenance_cost','failure_cost','unavailability_cost','waiting_cost','overdue_waiting_ticks'):
            interior[metric]+=info[metric]
        for metric,value in snap(state).items():interior[metric]+=value
    ep=episodes[k]
    if t+1==h:
        base_metrics=env.metrics.copy();cut=finish_requests(served,pending,h)
        for metric,value in snap(env.state).items():close(value,ep[f'cutoff_{metric}'])
        check_requests(cut,k,'cutoff')
    if done:
        extended=finish_requests(served,pending,h+12);check_requests(extended,k,'extended')
        idx={(r['machine'],r['request']):r for r in extended};carry=[]
        for r in cut:
            if r['censored']:
                later=idx[r['machine'],r['request']]
                value=dict(machine=r['machine'],request=r['request'],arrival=r['arrival'],wait_at_cutoff=r['wait'],
                    started_in_tail=later['served'],wait_at_start_or_end=later['wait'],censored_after_tail=later['censored'])
                raw=next(carry_stream);assert key(raw)==k
                for field,v in value.items():assert (raw[field]=='True' if isinstance(v,bool) else int(raw[field]))==v
                carry.append(value);carry_count+=1
        for metric,value in env.metrics.items():
            close(value,ep[f'extended_{metric}']);close(base_metrics[metric],ep[f'base_{metric}'])
            close(value-base_metrics[metric],ep[f'tail_{metric}'])
        for metric,value in snap(env.state).items():close(value,ep[f'post_tail_{metric}'])
        for scope,records in (('cutoff',cut),('extended',extended)):
            for metric,value in request_stats(records).items():close(value,ep[f'{scope}_{metric}'])
        window=h-h//4-h//4
        for metric in ('objective','maintenance_cost','failure_cost','unavailability_cost','waiting_cost'):close(interior[metric],ep[f'interior_{metric}'])
        close(h//4,ep['interior_begin']);close(h-h//4,ep['interior_end'])
        close(interior['objective']/(4*window),ep['interior_cost_per_machine_tick'])
        close(interior['pending']/window,ep['interior_pending_mean']);close(interior['failed']/window,ep['interior_failed_mean'])
        close(interior['overdue_waiting_ticks']/window,ep['interior_overdue_per_tick'])
        close(base_metrics['objective']/(4*h),ep['base_cost_per_machine_tick'])
        close(env.metrics['objective']/(4*(h+12)),ep['extended_cost_per_machine_tick'])
        close(len(carry),ep['carryover_requests']);close(sum(r['started_in_tail'] for r in carry),ep['carryover_started_in_tail'])
        close(sum(r['censored_after_tail'] for r in carry),ep['carryover_censored_after_tail'])
    decision_count+=1
assert decision_count==921600 and seen==expected and env.time==h+12
assert next(request_stream,None) is None and next(carry_stream,None) is None
profile_seen=set()
for row in csvrows('time_profiles.csv'):
    k=row['condition'],int(row['horizon']),row['controller'],int(row['time']),row['phase']
    assert k not in profile_seen;profile_seen.add(k);totals=time_totals[k]
    assert int(row['episodes'])==512==totals['episodes']
    for metric,value in totals.items():
        if metric!='episodes':close(value/512,row[metric])
assert profile_seen==set(time_totals)
write('physical_replay_audit.json',dict(status='PASS',episodes=len(seen),decisions=decision_count,requests=request_count,
    carryover_requests=carry_count,rule_actions_reproduced=rule_reproduced,time_profile_rows=len(profile_seen),
    planner_actions='feasibility/fixed restrictions checked; full planner inference not rerun',
    cutoff_and_tail_dynamics_preserved=True,interior_and_all_cost_components_reconstructed=True))
print('Full physical/request/carryover/time-profile replay PASS',flush=True)

# Reconcile latency records, including per-episode quantiles and actual forecast work.
latency=defaultdict(list);work=defaultdict(lambda:defaultdict(float));current=None;lat_count=0
def finish_latency(k,base,tail):
    ep=episodes[k];close(np.mean(base),ep['latency_mean_ms'])
    for q,label in ((.5,'p50'),(.95,'p95'),(.99,'p99')):close(np.quantile(base,q),ep[f'latency_{label}_ms'])
    close(np.mean(tail),ep['tail_rule_latency_mean_ms'])


for r in tqdm(csvrows('latency.csv'),total=921600,desc='Audit latency and forecast work'):
    k=key(r)
    if k!=current:
        if current is not None:finish_latency(current,base_lat,tail_lat)
        current=k;base_lat=[];tail_lat=[];times=0
    condition,h,_,_,name=k;t=int(r['time']);assert times==t;times+=1
    phase='base' if t<h else 'tail';assert r['phase']==phase
    a=int(r['candidate_count']);s=int(r['forecast_scenarios']);f=int(r['forecast_ticks']);tail=int(r['terminal_ticks'])
    planner=phase=='base' and name!='best_rule'
    assert s==(128 if planner else 0)
    assert tail==(12 if planner and name.endswith('tail') else 0)
    assert f==(h-t+tail if planner else 0)
    assert (1<=a<=21) if planner else a==0
    assert int(r['forecast_transition_samples'])==a*s*f
    v=float(r['latency_ms']);assert math.isfinite(v) and v>=0
    (base_lat if phase=='base' else tail_lat).append(v)
    latency[condition,h,name,phase].append(v)
    totals=work[condition,h,name,phase];totals['decisions']+=1;totals['latency_ms']+=v
    totals['forecast_transition_samples']+=a*s*f;totals['forced_decisions']+=int(planner and a==1)
    lat_count+=1
finish_latency(current,base_lat,tail_lat);assert lat_count==921600
latency_summary={f'{co}:{h}:{name}:{phase}':dict(decisions=len(values),mean_ms=float(np.mean(values)),
    p95_ms=float(np.quantile(values,.95)),p99_ms=float(np.quantile(values,.99)),max_ms=max(values),
    **{k:v for k,v in work[co,h,name,phase].items() if k!='decisions'}) for (co,h,name,phase),values in latency.items()}
write('latency_analysis.json',dict(status='PASS',cells=latency_summary,forecast_samples=sum(v['forecast_transition_samples'] for v in work.values()),
    forecast_forced_decisions=sum(v['forced_decisions'] for v in work.values()),
    measured_action_selection_seconds=sum(v['latency_ms'] for v in work.values())/1000,wall_seconds=wall))
statistics=json.loads((OUT/'independent_statistics.json').read_text());statistics['physical_replay_pending']=False
statistics['physical_replay']='PASS';write('independent_statistics.json',statistics)
print('Latency audit PASS; actual forecast samples:',sum(v['forecast_transition_samples'] for v in work.values()),flush=True)
