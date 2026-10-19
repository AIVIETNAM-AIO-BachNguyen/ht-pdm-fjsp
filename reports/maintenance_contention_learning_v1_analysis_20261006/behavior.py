import csv,json
from collections import defaultdict
from pathlib import Path
import sys
from functools import lru_cache
sys.path.insert(0,'/private/tmp/marl-oracle-representation-20261003/src')
from ht_pdm_fjsp.maintenance_contention_model import cohort_config
OUT=Path(__file__).resolve().parent
RAW=Path('/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_contention_learning_v1_full_cpu_20261006T023329Z')
@lru_cache(maxsize=60)
def config(cohort,cap,h):return cohort_config(cohort,cap,h)
counts=defaultdict(lambda:defaultdict(int))
with (RAW/'decisions.csv').open() as stream:
 for row in csv.DictReader(stream):
  cohort=int(row['cohort']);h=int(row['horizon']);cap=row['capacity'];name=row['controller']
  c=config(cohort,cap,h);state=json.loads(row['state']);pairs=json.loads(row['action'])
  key=f'{cap}_H{h}_{name}';r=counts[key];r['ticks']+=1
  busy={m for m in state['assigned'] if m>=0};free=sum(x==0 for x in state['remaining'])
  selected={m for m,j in pairs}
  failed_eligible={m for m,b in enumerate(state['failed']) if b and m not in busy}
  if free>0 and failed_eligible:
   r['idle_with_failed_demand']+=1
   if pairs and not selected.intersection(failed_eligible):r['corrective_deferred_for_preventive']+=1
   if not pairs:r['corrective_deferred_for_stop']+=1
  if not pairs and free>0:r['stop_with_idle']+=1
  for m,j in pairs:
   r['starts']+=1;r[f'starts_duration{c.service_time[m][0]}']+=1
   if state['failed'][m]:
    r['corrective_starts']+=1;r['corrective_start_wait_total']+=state['pending_wait'][m]
   else:
    r['preventive_starts']+=1;r[f'preventive_age{state["ages"][m]}']+=1
    if state['ages'][m]<2:r['early_preventive_starts']+=1
result={k:dict(v,early_pm_fraction=v.get('early_preventive_starts',0)/v['preventive_starts'] if v['preventive_starts'] else None,
 corrective_deferred_for_pm_fraction=v.get('corrective_deferred_for_preventive',0)/v['idle_with_failed_demand'] if v['idle_with_failed_demand'] else None,
 mean_corrective_start_wait=v['corrective_start_wait_total']/v['corrective_starts'] if v['corrective_starts'] else None) for k,v in counts.items()}
(OUT/'behavior.json').write_text(json.dumps(result,indent=2)+'\n')
for cap,h in [('middle',12),('high',12),('low',24)]:
 for n in ['independent_dp_matching' if cap!='low' else 'deadline_matching','joint_rollout','central_subset_ppo']:
  r=result[f'{cap}_H{h}_{n}'];print(cap,h,n,{k:r[k] for k in ['early_pm_fraction','corrective_deferred_for_pm_fraction','mean_corrective_start_wait']})
