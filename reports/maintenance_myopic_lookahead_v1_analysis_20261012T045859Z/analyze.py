"""Read-only reconstruction and independent statistics of the downloaded run."""
import csv
from collections import defaultdict, Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from tqdm.auto import tqdm
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
RUN=PROJECT/'lab_results/maintenance_myopic_lookahead_v1_full_cpu_20261012T045644Z'
CODE=Path('/private/tmp/marl-maintenance-assignment-confidence-v1')
COMMIT='ac9fc61316d26a75fc3a5869e084d4476bf828db'
sys.path.insert(0,str(CODE/'src'))
from ht_pdm_fjsp import maintenance_myopic_lookahead as s
from ht_pdm_fjsp.maintenance_dispatch import role_seed

def same(a,b):
    s.close_enough(a,b)
def dump(name,data):
    (HERE/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def read(name,strings,ints=()):
    with (RUN/name).open() as f: rows=list(csv.DictReader(f))
    return [{k:v if k in strings else int(v) if k in ints else float(v) for k,v in r.items()} for r in rows]

manifest=json.loads((RUN/'manifest.json').read_text());audit=json.loads((RUN/'audit.json').read_text())
assert manifest['status']=='COMPLETED' and manifest['audit']=='PASS'
assert manifest['git']['commit']==COMMIT and not manifest['git']['dirty']
assert manifest['full_evaluation_outcomes_opened']
for name, record in manifest['files'].items():
    data=(RUN/name).read_bytes()
    assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
bindings={}
for path in (RUN/'source_snapshot').iterdir():
    original=('configs/' if path.suffix=='.json' else 'src/ht_pdm_fjsp/')+path.name
    committed=subprocess.check_output(['git','show',COMMIT+':'+original],cwd=CODE)
    assert committed==path.read_bytes()==(CODE/original).read_bytes()
    bindings[path.name]=hashlib.sha256(committed).hexdigest()
for name,original in [('protocol.md',f'docs/{s.VERSION}_plan.md'),('seed_registry.json',f'configs/{s.VERSION}_seed_registry.json')]:
    assert (RUN/name).read_bytes()==subprocess.check_output(['git','show',COMMIT+':'+original],cwd=CODE)
for name,digest in manifest['source_seal'].items():
    assert manifest['files'][name]['sha256']==digest
resolved=json.loads((RUN/'resolved_config.json').read_text());cfg=resolved['settings']
same(cfg,s.settings('full'));same(resolved['config'],json.loads(s.compact(s.asdict(s.make_config(cfg)))))
same(s.seed_audit(cfg),audit['seed_audit'])
episodes=read('episodes.csv',{'controller'},{'seed','physical_seed'})
landmarks=read('landmarks.csv',{'state','joint_action','assignment_action'},
    {'seed','tick','roots','selection_seed','holdout_seed','forecast_transitions','joint_root','assignment_root'})
with (RUN/'landmark_roots.csv').open() as f:root_rows=list(csv.DictReader(f))
assert (RUN/'episodes.csv').read_bytes()==(RUN/'episodes.partial.csv').read_bytes()
expected=dict(episodes=384,physical_transitions=23040,landmarks=896)
assert manifest['actual_counts']==manifest['expected_counts']==audit['actual_counts']==audit['expected_counts']==expected
assert len(episodes)==384 and len(landmarks)==896 and len(root_rows)==2366
def oracle(c, state, selected):
    demand = [m for m in range(c.machines) if m not in state.assigned and (state.failed[m] or state.ages[m] >= 3)]
    free = [j for j in range(c.technicians) if state.remaining[j] == 0]
    valid = {m: [j for j in free if c.service_time[m][j] > 0] for m in demand}
    prefs = {m: {j for j in js if not any(
        c.service_time[m][k] <= c.service_time[m][j] and c.restoration[m][k] >= c.restoration[m][j]
        and (c.service_time[m][k] < c.service_time[m][j] or c.restoration[m][k] > c.restoration[m][j])
        for k in js)} for m, js in valid.items()}
    actions = []
    def visit(i, chosen):
        if i == len(free):
            actions.append(tuple(sorted(chosen))); return
        j = free[i]; visit(i + 1, chosen)
        for m in demand:
            if j in valid[m] and all(m != old for old, _ in chosen): visit(i + 1, chosen + [(m, j)])
    visit(0, [])
    best = [a for a in actions if all(j in prefs[m] for m, j in a)]
    f, p = max(map(len, actions)), max(map(len, best))
    grouped = defaultdict(list)
    physical = defaultdict(set)
    for a in actions:
        subset = tuple(m for m, _ in a)
        grouped[subset].append(c.unavailable_cost * sum(c.service_time[m][j] for m, j in a))
        physical[subset].add(tuple((m, c.service_time[m][j], c.restoration[m][j]) for m,j in a))
    preferred_subsets = {tuple(m for m, _ in a) for a in best if len(a) == f}
    joint = p < f and any(len(subset) == f and len(values) > 1 and subset not in preferred_subsets
        for subset, values in physical.items())
    spans = {subset: max(v) - min(v) for subset,v in grouped.items()}
    possible = max(v for subset,v in spans.items() if len(subset) == f)
    subset = tuple(m for m, _ in selected)
    selected_span = spans[subset]
    threshold = max(c.unavailable_cost, 1e-12)
    metrics = dict(skill_contention_rate=int(p < f), contested_choice_rate=int(joint),
        feasible_matching_cardinality=f, preferred_matching_cardinality=p, preferred_deficit=f-p,
        cost_sensitive_contention_rate=int(p < f and possible >= threshold),
        selected_cost_sensitive_contention_rate=int(p < f and len(subset) == f and selected_span >= threshold),
        committed_cost_span_max=possible, selected_committed_cost_span=selected_span)
    return metrics, grouped


# Reconstruct all logged decisions/forecasts without evaluating any new policy.
# Only audit coverage expands to all 128 seeds; algorithm parameters stay locked.
original_metrics=s.screen.state_metrics
oracle_ticks=0
def checked_metrics(c,state,selected,space=None):
    global oracle_ticks
    actual=original_metrics(c,state,selected,space)
    independent,_=oracle(c,state,selected)
    same(independent,{k:actual[k] for k in independent});oracle_ticks+=1
    return actual
s.screen.state_metrics=checked_metrics
progress=tqdm(total=23040,desc='Verify all recorded physical decisions and forecasts',unit='decision')
original_select=s.select
def checked_select(*args,**kwargs):
    result=original_select(*args,**kwargs);progress.update(1);return result
s.select=checked_select
verified=s.replay_audit(RUN/'decisions.csv',{**cfg,'audit_seeds':128},episodes,landmarks,root_rows)
progress.close();s.select=original_select;s.screen.state_metrics=original_metrics
assert oracle_ticks==18432
assert verified['replay_forecast_transitions']==manifest['forecast_transitions']==42316800
assert verified['forecast_audited_seeds']==128 and verified['audited_landmarks']==896

# Independent count-weight bootstrap, not the runner's indexed 3D resampling.
lookup={(r['controller'],r['seed']):r for r in episodes}
n=len(cfg['evaluation_seeds']);rng=np.random.default_rng(role_seed(cfg['analysis_seed'],f'{s.VERSION}:bootstrap'))
indices=rng.integers(n,size=(cfg['bootstrap_samples'],n))
weights=np.zeros((len(indices),n))
np.add.at(weights,(np.repeat(np.arange(len(indices)),n),indices.ravel()),1/n)
fields=['extended_cost','availability','terminal_pending']
values=np.asarray([[[lookup[p,seed][k] for k in fields] for p in s.CONTROLLERS] for seed in cfg['evaluation_seeds']])
boot=np.einsum('bn,npm->bpm',weights,values,optimize=True);point=values.mean(0);alpha=(1-cfg['confidence'])/2
comparisons={}
for key,improved,base in [('joint_vs_myopic',1,0),('assignment_vs_myopic',2,0),('joint_vs_assignment',1,2)]:
    comparisons[key]=dict(estimate=float(1-point[improved,0]/point[base,0]),
        ci=np.quantile(1-boot[:,improved,0]/boot[:,base,0],[alpha,1-alpha]).tolist(),confidence=cfg['confidence'])
for key,column in [('availability_joint_minus_myopic',1),('pending_joint_minus_myopic',2)]:
    comparisons[key]=dict(estimate=float(point[1,column]-point[0,column]),
        ci=np.quantile(boot[:,1,column]-boot[:,0,column],[alpha,1-alpha]).tolist(),confidence=cfg['confidence'])
summary=json.loads((RUN/'summary.json').read_text());same(comparisons,summary['comparisons'])
gates=dict(practical_cost=comparisons['joint_vs_myopic']['ci'][0]>=cfg['practical_margin'],
    availability=comparisons['availability_joint_minus_myopic']['ci'][0]>=-cfg['availability_margin'],
    pending=comparisons['pending_joint_minus_myopic']['ci'][1]<=cfg['pending_margin'])
same(gates,summary['gates']);assert not all(gates.values()) and summary['scientific_status']==manifest['scientific_status']=='NOT_CONFIRMED'
means=[]
for p in s.CONTROLLERS:
    group=[r for r in episodes if r['controller']==p]
    means.append(dict(controller=p,**{k:float(np.mean([r[k] for r in group])) for k in group[0] if k not in ('controller','seed','physical_seed')}))
same(means,summary['means'])
local={(r['seed'],r['tick']):r for r in landmarks};mechanisms={}
for name in ['joint','assignment']:
    seed_values=np.asarray([np.mean([local[seed,t][f'{name}_heldout_gain'] for t in cfg['landmarks']]) for seed in cfg['evaluation_seeds']])
    mechanisms[name]=dict(mean_holdout_12_tick_gain_per_landmark=float(seed_values.mean()),
        exploratory_ci95=np.quantile(weights@seed_values,[.025,.975]).tolist(),
        changed_roots=sum(r[f'{name}_disagreement_rate'] for r in landmarks),total_landmarks=len(landmarks),
        scope='independent forecast bank, myopic states, fixed landmarks; not episode savings')
same(mechanisms,summary['mechanism_diagnostics'])

# Posthoc diagnostics from existing logs only: exact action/state equality,
# return distribution, change types and a recurrent short-horizon pattern.
with (RUN/'decisions.csv').open() as f:raw_decisions=list(csv.DictReader(f))
traces=defaultdict(list)
counts=defaultdict(Counter);patterns=defaultdict(Counter);selected_improvements=defaultdict(list)
for r in raw_decisions:
    p,seed,tick=r['controller'],int(r['seed']),int(r['tick'])
    traces[p,seed].append(r)
    if tick<48:
        state=s.screen.base.unpack_state(json.loads(r['state']));chosen=tuple(map(tuple,json.loads(r['action'])))
        delta=s.change_metrics(s.myopic(s.make_config(cfg),state),chosen)
        counts[p].update(delta)
        if delta['disagreement_rate']:
            signature=(state.ages,state.failed,state.pending_wait,state.remaining,state.assigned)
            patterns[p][str(signature)]+=1
            diag=json.loads(r['diagnostic']);selected_improvements[p].append(diag['myopic_mean']-diag['selected_mean'])
contrasts={}
for p in s.CONTROLLERS[1:]:
    gains=np.asarray([lookup['myopic_matching',seed]['extended_cost']-lookup[p,seed]['extended_cost'] for seed in cfg['evaluation_seeds']])
    exact_paths=sum(all(a['state']==b['state'] and a['action']==b['action'] and a['following']==b['following'] for a,b in zip(traces['myopic_matching',seed],traces[p,seed])) for seed in cfg['evaluation_seeds'])
    contrasts[p]=dict(mean_absolute_saving=float(gains.mean()),wins=int((gains>0).sum()),ties=int((gains==0).sum()),losses=int((gains<0).sum()),
        min_saving=float(gains.min()),max_saving=float(gains.max()),exactly_identical_full_paths=exact_paths,
        changes=dict(counts[p]),most_common_change_states=patterns[p].most_common(5),
        predicted_gain_at_changes=dict(mean=float(np.mean(selected_improvements[p])),min=float(min(selected_improvements[p])),max=float(max(selected_improvements[p]))))
seed=36000083;example={}
for p in s.CONTROLLERS:
    g=traces[p,seed]
    cumulative={str(h):sum(json.loads(r['info'])['objective'] for r in g[:h]) for h in [12,24,36,48,60]}
    deviations=[]
    for r in g[:48]:
        state=s.screen.base.unpack_state(json.loads(r['state']));action=tuple(map(tuple,json.loads(r['action'])))
        if action!=s.myopic(s.make_config(cfg),state):
            deviations.append(dict(tick=int(r['tick']),state=json.loads(r['state']),action=json.loads(r['action']),diagnostic=json.loads(r['diagnostic'])))
    if p!='myopic_matching':
        assert [r['tick'] for r in deviations]==[0,12,24,36]
        assert all(r['state']==deviations[0]['state'] for r in deviations)
    example[p]=dict(cumulative_cost=cumulative,deviations=deviations)
example_landmark=local[seed,0]
assert example_landmark['assignment_heldout_gain']==1 and example_landmark['assignment_heldout_gain_se']==0

result=dict(run=str(RUN),commit=COMMIT,validation=dict(status='PASS',hashed_files=len(manifest['files']),source_bindings=bindings,
    exact_counts=expected,seed_audit='PASS',all_physical_replay=verified,independent_metric_oracle_ticks=oracle_ticks,
    independent_five_CIs_match=True,independent_means_and_mechanism_CIs_match=True,all_original_files_immutable=True),
    summary=summary,paired_return_and_path_diagnostics=contrasts,
    posthoc_recurrent_horizon_example=dict(seed=seed,landmark=example_landmark,traces=example),
    cost_decomposition={r['controller']:{k:r[k]-means[0][k] for k in ('policy_cost','tail_cost','extended_maintenance_cost','extended_unavailability_cost','extended_failure_cost','extended_waiting_cost','extended_cost')} for r in means[1:]})
# Raw hashes are checked again after reconstruction.
for name,record in manifest['files'].items():
    assert hashlib.sha256((RUN/name).read_bytes()).hexdigest()==record['sha256']
dump('analysis.json',result)
print(json.dumps(result['validation'],indent=2));print(json.dumps(contrasts,indent=2))
