"""Inspect numerical ties and replay only affected complete evaluation episodes."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from collections import defaultdict
import torch
sys.path.insert(0,'/private/tmp/marl-oracle-representation-20261003/src')
from ht_pdm_fjsp import maintenance_contention_imitation as runner
from ht_pdm_fjsp.maintenance_contention_learning_policy import feature_batch
from ht_pdm_fjsp.maintenance_contention_model import cohort_config,compact,physical_action,subsets
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl')
RAW=ROOT/'lab_results/maintenance_contention_imitation_v1_full_cpu_20261006T063713Z'
OUT=Path(__file__).resolve().parent
a=json.loads((OUT/'analysis.json').read_text());mismatches=a['audit']['neural_action_mismatches']
wanted={(tuple(r['key']),r['time']):r for r in mismatches}
selection=json.loads((RAW/'checkpoint_selection.json').read_text())['selections']
def key(r):return (int(r['train_seed']),int(r['cohort']),r['capacity'],int(r['horizon']),int(r['seed']),r['controller'])
torch.set_num_threads(1);models={};details=[]
with (RAW/'decisions.csv').open() as s:
 for r in csv.DictReader(s):
  k=key(r);t=int(r['time'])
  if (k,t) not in wanted:continue
  c=cohort_config(k[1],k[2],k[3]);state=json.loads(r['state'])
  reduced_remaining=[0]*4
  for assigned,remaining in zip(state['assigned'],state['remaining']):
   if assigned>=0:reduced_remaining[assigned]=remaining
  compact_state=tuple(zip(state['ages'],map(int,state['failed']),state['pending_wait'],reduced_remaining))
  m=wanted[k,t];old=subsets(4)[m['expected']];new=subsets(4)[m['actual']]
  removed=set(old)-set(new);added=set(new)-set(old)
  signatures=lambda machines:sorted((compact_state[i],c.service_time[i][0],c.restoration[i][0]) for i in machines)
  symmetric=signatures(removed)==signatures(added)
  mk=(k[-1],k[0])
  if mk not in models:models[mk]=runner.load_checkpoint(RAW/selection[f'{k[-1]}:{k[0]}']['checkpoint'])[0]
  features=feature_batch([c],[compact_state],[k[3]-t])
  with torch.no_grad():dist,_=models[mk].distribution(features)
  scalar,diag=models[mk].select(c,compact_state,k[3]-t)
  details.append(dict(**m,old_subset=old,batched_subset=new,compact_state=compact_state,
                      symmetric_current_profiles=symmetric,scalar_subset=scalar,
                      scalar_logits_gap=float(dist.logits[0,m['actual']]-dist.logits[0,m['expected']])))
assert len(details)==27
recorded={}
with (RAW/'episodes.csv').open() as s:
 for r in csv.DictReader(s):
  k=key(r)
  if any(tuple(m['key'])==k for m in mismatches):recorded[k]=r
sensitivity=[];deltas=defaultdict(float)
for k,r in recorded.items():
 c=cohort_config(k[1],k[2],k[3]);env=WaitingEnv(c);env.reset(k[4]);model=models[k[-1],k[0]]
 for t in range(c.horizon):
  choice,_=model.select(c,compact(env.state),c.horizon-t)
  env.step(physical_action(env.state,choice))
 delta=env.metrics['objective']-float(r['objective'])
 sensitivity.append(dict(key=k,recorded_cost=float(r['objective']),mac_replay_cost=env.metrics['objective'],difference=delta))
 deltas[k[-1],k[2],k[3]]+=delta
result=dict(mismatches=len(details),all_symmetric_current_profiles=all(d['symmetric_current_profiles'] for d in details),
            primary_mismatches=sum(d['key'][2]=='middle' and d['key'][3]==12 for d in details),
            affected_episodes=len(recorded),details=details,episode_sensitivity=sensitivity,
            aggregate_mean_changes={f'{arm}_{cap}_H{h}':v/5000 for (arm,cap,h),v in deltas.items()})
(OUT/'numerical_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('details','episode_sensitivity')},indent=2))
