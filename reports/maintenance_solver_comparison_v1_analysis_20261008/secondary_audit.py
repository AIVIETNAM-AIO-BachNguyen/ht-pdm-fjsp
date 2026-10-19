"""Reconcile machine totals; summarize development curves and profile consistency."""
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

OUT=Path(__file__).resolve().parent
RUN=OUT.parents[1]/'lab_results/maintenance_solver_comparison_v1_full_cpu_20261007T151955Z'
def rows(name):
    with (RUN/name).open(newline='') as f: yield from csv.DictReader(f)
def key(r): return tuple(r[k] for k in ('controller','train_seed','cohort','condition','horizon','seed'))
metrics=['objective','base_cost','waiting_cost','maintenance_cost','failure_cost','unavailability_cost',
         'unavailable_ticks','failed_waiting_ticks','overdue_waiting_ticks','terminal_failed','terminal_pending']
expected={key(r):{m:float(r[m]) for m in metrics} for r in rows('episodes.csv')}
previous=None;total=Counter();machines=set();count=0;episodes=0
def check():
    assert machines=={0,1,2,3}
    assert all(abs(total[m]-expected[previous][m])<1e-8 for m in metrics)
for r in rows('machine_metrics.csv'):
    k=key(r)
    if k!=previous:
        if previous is not None:check();episodes+=1
        previous=k;total=Counter();machines=set()
    m=int(r['machine']);assert m not in machines;machines.add(m)
    total.update({m:float(r[m]) for m in metrics});count+=1
check();episodes+=1
assert count==1146880 and episodes==286720
dev=defaultdict(list);devcount=0
for r in rows('development_episodes.csv'):
    devcount+=1
    if r['condition']=='specialized' and r['horizon']=='12' and int(r['train_seed'])>=0:
        dev[(r['controller'],int(r['checkpoint_step']))].append(float(r['objective']))
assert devcount==82560
cohorts=defaultdict(list)
for r in rows('episodes.csv'):
    if r['condition']=='specialized' and r['horizon']=='12':
        cohorts[(r['controller'],int(r['cohort']))].append(float(r['objective']))
paircounts={}
for a,b in [('cooperative_local_ppo','independent_local_ppo'),('cooperative_full_ppo','central_matching_ppo'),
            ('central_fixed_allocation_ppo','adaptive_isolated_matching'),('central_matching_ppo','joint_rollout')]:
    differences=[st.mean(cohorts[(a,c)])-st.mean(cohorts[(b,c)]) for c in range(163000,163032)]
    paircounts[f'{a}_vs_{b}']={'descriptive_profile_wins':sum(x<0 for x in differences),
        'profiles':32,'minimum_delta':min(differences),'maximum_delta':max(differences)}
result={'machine_rows':count,'reconciled_episodes':episodes,'development_rows':devcount,
        'development_specialized_H12_mean_by_checkpoint':{f'{a}:{s}':st.mean(v) for (a,s),v in dev.items()},
        'profile_consistency':paircounts}
(OUT/'secondary_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
