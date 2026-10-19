"""Descriptive root-action restriction value on the SAME logged forecast states."""
import csv,json,sys,types
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl');RAW=ROOT/'lab_results/maintenance_heterogeneous_marl_v1_full_cpu_20261006T150059Z'
OUT=Path(__file__).resolve().parent
package=types.ModuleType('ht_pdm_fjsp');package.__path__=[str(RAW/'source_snapshot')];sys.modules['ht_pdm_fjsp']=package
from ht_pdm_fjsp import maintenance_heterogeneous_model as world
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
stats=defaultdict(lambda:defaultdict(float));cache={}
for row in csv.DictReader((RAW/'decisions.csv').open()):
 if int(row['train_seed'])!=-1:break
 if row['controller']!='joint_rollout':continue
 cs=int(row['cohort']);cond=row['condition'];h=int(row['horizon']);ckey=(cs,cond,h)
 if ckey not in cache:cache[ckey]=world.cohort_config(*ckey)
 c=cache[ckey];state=DispatchState(**{k:tuple(v) for k,v in json.loads(row['state']).items()})
 diag=json.loads(row['controller_diagnostics']);pairs=tuple(tuple(p) for p in json.loads(row['action']))
 candidates=[tuple(tuple(p) for p in candidate) for candidate in diag['candidates']]
 fixed=set(world.fixed_candidates(c,state));assert fixed<=set(candidates)
 restricted=min(cost for p,cost in zip(candidates,diag['means'],strict=True) if p in fixed)
 gain=restricted-diag['selected_score'];assert gain>=-1e-9
 s=stats[f'{cond}_H{h}'];s['intervals']+=1
 s['allocation_options_exist']+=len(candidates)>len(fixed)
 s['selected_outside_fixed']+=pairs not in fixed
 s['positive_forecast_allocation_gain']+=gain>1e-9
 s['total_forecast_gain']+=gain
 s['max_forecast_gain']=max(s['max_forecast_gain'],gain)
for s in stats.values():
 s.update(outside_fixed_frequency=s['selected_outside_fixed']/s['intervals'],
  positive_forecast_gain_frequency=s['positive_forecast_allocation_gain']/s['intervals'],
  mean_forecast_gain_per_interval=s['total_forecast_gain']/s['intervals'])
(OUT/'planner_root_allocation_diagnostic.json').write_text(json.dumps(dict(status='DESCRIPTIVE',panels=stats,
 explanation='Min forecast cost with root fixed allocation vs all root matchings, on the same logged full-planner state/scenarios. Continuation stays identical. This is a rollout-model diagnostic, not an optimality bound or a new confirmatory test.'),indent=2)+'\n')
print(json.dumps(stats,indent=2))
