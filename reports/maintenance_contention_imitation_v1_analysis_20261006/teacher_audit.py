"""Audit teacher data/provenance, physical trajectories and diagnostic checkpoints."""
import csv
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
from time import perf_counter
import numpy as np
import torch
from tqdm.auto import tqdm

CHECKOUT=Path('/private/tmp/marl-oracle-representation-20261003')
sys.path.insert(0,str(CHECKOUT/'src'))
from ht_pdm_fjsp import maintenance_contention_imitation as runner
from ht_pdm_fjsp.maintenance_contention_imitation_training import diagnostics,accepted_targets,checked_seed
from ht_pdm_fjsp.maintenance_contention_learning_policy import feature_batch
from ht_pdm_fjsp.maintenance_contention_model import actions,cohort_config,compact,physical_action,plan,subsets
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl')
RAW=ROOT/'lab_results/maintenance_contention_imitation_v1_full_cpu_20261006T063713Z'
OUT=Path(__file__).resolve().parent
cfg=json.loads((RAW/'manifest.json').read_text())['config']
selection=json.loads((RAW/'checkpoint_selection.json').read_text())['selections']
refs=json.loads((RAW/'reference_selection.json').read_text())['selected']
_,forbidden=runner.seed_audit(cfg)
torch.set_num_threads(1)
START=perf_counter()
def dump(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def read(name):
 with (RAW/name).open() as s:return list(csv.DictReader(s))

logged=read('bc_diagnostics.csv');assert len(logged)==100
recomputed=[];data_shapes=[]
for seed in tqdm(cfg['train_seeds'],desc='teacher tensors and BC diagnostics'):
 data=torch.load(RAW/'teacher'/f'train_seed_{seed}'/'teacher_data.pt',map_location='cpu',weights_only=True)
 for split,n in [('train',12288),('validation',1536)]:
  d=data[split];assert len(d['targets'])==len(d['metadata'])==n
  cs=[cohort_config(m['config_seed'],m['capacity'],12) for m in d['metadata']]
  for c,m in zip(cs,d['metadata'],strict=True):
   assert json.loads(json.dumps(asdict(c)))==json.loads(json.dumps(m['config']))
   assert m['config_seed'] not in forbidden and m['environment_seed'] not in forbidden
  feat=feature_batch(cs,[m['state'] for m in d['metadata']],[m['remaining_horizon'] for m in d['metadata']])
  assert all(torch.equal(v,d['features'][k]) for k,v in feat.items())
  assert not (d['targets'] & ~feat['mask']).any() and d['targets'].any(-1).all()
  assert torch.equal(torch.isfinite(d['scores']),feat['mask'])
  data_shapes.append(dict(train_seed=seed,split=split,rows=n))
 for candidate in selection[f'{runner.IMITATION}:{seed}']['candidates']:
  model,payload=runner.load_checkpoint(RAW/candidate['path'])
  for split in ('train','validation'):
   diag=diagnostics(model,data[split],192)
   saved=next(r for r in logged if int(r['train_seed'])==seed and int(r['epoch'])==candidate['update'] and r['split']==split)
   for k,v in diag.items():assert np.isclose(v,float(saved[k]),rtol=1e-5,atol=1e-6),(seed,candidate['update'],split,k,v,saved[k])
   recomputed.append(dict(train_seed=seed,epoch=candidate['update'],split=split,**diag))
 del data
# Verify every CSV trajectory, independently derived roles, teacher target and
# random exploratory behavior; reforecast a predetermined subset of60 episodes.
count=0;forecasts=0;keys=defaultdict(int);last=None;env=None;rng=None
split_episodes=defaultdict(set);choice_states=defaultdict(int)
with (RAW/'teacher_rows.csv').open() as stream:
 for row in tqdm(csv.DictReader(stream),total=138240,desc='teacher physical and label audit'):
  seed=int(row['train_seed']);split=row['split'];index=int(row['episode']);t=int(row['time']);key=(seed,split,index)
  cap='low' if index%5==0 else 'middle'
  cs=checked_seed(seed,f'imitation_{split}_configuration:{index}',forbidden)
  es=checked_seed(seed,f'imitation_{split}_environment:{index}',forbidden)
  bs=checked_seed(seed,f'imitation_{split}_behavior:{index}',forbidden)
  ps=checked_seed(seed,f'imitation_{split}_planner:{index}:{t}',forbidden)
  assert (cs,es,bs,ps)==tuple(int(row[k]) for k in ('config_seed','environment_seed','behavior_seed','planner_seed'))
  assert cap==row['capacity'] and int(row['remaining_horizon'])==12-t and row['continuation']==refs[cap]
  if key!=last:
   c=cohort_config(cs,cap,12);env=WaitingEnv(c);env.reset(es);rng=random.Random(bs);last=key
   assert asdict(c)=={k:tuple(tuple(vv) for vv in v) if k in ('service_time','restoration') else v for k,v in json.loads(row['config']).items()}
   assert hashlib.sha256(json.dumps(asdict(c),sort_keys=True).encode()).hexdigest()==row['config_sha256']
   split_episodes[seed,split].add(index)
  assert t==keys[key]
  state=compact(env.state);assert json.loads(json.dumps(state))==json.loads(row['state'])
  diag=json.loads(row['teacher_diagnostics']);feasible=actions(c,state)
  assert feasible==tuple(tuple(x) for x in diag['candidates']) and diag['scenario_seed']==ps
  target=accepted_targets(feasible,diag['means'],4)
  assert torch.nonzero(target).flatten().tolist()==json.loads(row['accepted_indices'])
  chosen=feasible[int(np.argmin(diag['means']))]
  executed=feasible[rng.randrange(len(feasible))] if rng.random()<.2 else chosen
  assert executed==tuple(json.loads(row['executed_subset']))
  if index in ({1,5,257} if split=='train' else {1,5,128}):
   result,check=plan(c,state,12-t,refs[cap],ps,32)
   assert result==chosen and np.allclose(check['means'],diag['means'],atol=1e-12,rtol=0)
   assert np.allclose(check['mc_se'],diag['mc_se'],atol=1e-12,rtol=0)
   forecasts+=1
  _,_,done,info=env.step(physical_action(env.state,executed))
  assert done==(t==11) and info==json.loads(row['physical_metrics'])
  assert json.loads(json.dumps(compact(env.state)))==json.loads(row['following'])
  keys[key]+=1;count+=1;choice_states[split]+=len(feasible)>1
assert count==138240 and len(keys)==11520 and all(x==12 for x in keys.values())
for seed in cfg['train_seeds']:
 assert split_episodes[seed,'train']==set(range(1,1025)) and split_episodes[seed,'validation']==set(range(1,129))
# Independent high-scenario teacher probes: exact root means, valid selection,
# and original root regret. These are finite MC diagnostics, not exact labels.
noise=read('teacher_noise.csv');assert len(noise)==320
for seed in tqdm(cfg['train_seeds'],desc='independent teacher noise replay'):
 data=torch.load(RAW/'teacher'/f'train_seed_{seed}'/'teacher_data.pt',map_location='cpu',weights_only=True)['validation']
 panel=[r for r in noise if int(r['train_seed'])==seed]
 assert [int(r['validation_row']) for r in panel]==list(range(32))
 for row in panel:
  m=data['metadata'][int(row['validation_row'])];c=cohort_config(m['config_seed'],m['capacity'],12)
  ns=checked_seed(seed,f"imitation_noise:{m['episode']}:{m['time']}",forbidden)
  assert int(row['probe_planner_seed'])==ns and int(row['original_scenarios'])==32 and int(row['probe_scenarios'])==128
  chosen,diag=plan(c,m['state'],m['remaining_horizon'],m['continuation'],ns,128)
  assert chosen==tuple(json.loads(row['probe_subset']))
  saved=json.loads(row['probe_diagnostics']);assert np.allclose(diag['means'],saved['means'],atol=1e-12,rtol=0)
  old=tuple(json.loads(row['original_subset']));old_index=diag['candidates'].index(old)
  regret=diag['means'][old_index]-min(diag['means']);assert regret==float(row['probe_regret_of_original'])
  assert int(row['changed'])==int(old!=chosen)
  row['choices']=len(diag['candidates']);row['regret']=regret
 del data
probe_summary=dict(probes=320,changed_rate=float(np.mean([int(r['changed']) for r in noise])),
                   mean_regret=float(np.mean([r['regret'] for r in noise])),
                   p95_regret=float(np.percentile([r['regret'] for r in noise],95)),
                   max_regret=max(r['regret'] for r in noise))
choices=[r for r in noise if r['choices']>1]
probe_summary.update(choice_probes=len(choices),choice_changed_rate=float(np.mean([int(r['changed']) for r in choices])),
                     choice_mean_regret=float(np.mean([r['regret'] for r in choices])))
dump('teacher_analysis.json',dict(audit=dict(status='PASS',csv_rows=count,trajectories=len(keys),
                                           tensor_rows=138240,bc_checkpoint_diagnostics=100,
                                           sampled_teacher_forecast_replays=forecasts,independent_noise_replays=320,
                                           elapsed_seconds=perf_counter()-START),
                                  bc_diagnostics=recomputed,teacher_noise=probe_summary,
                                  teacher_shapes=data_shapes,choice_states=dict(choice_states)))
print(json.dumps(probe_summary),flush=True)
