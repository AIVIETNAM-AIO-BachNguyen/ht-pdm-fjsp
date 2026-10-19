"""Audit saved tensors and replay common DEVELOPMENT states only."""
from pathlib import Path
from dataclasses import asdict
from collections import Counter,defaultdict
import csv,hashlib,json,sys
import torch
WORK=Path('/Users/bachng/.codex/worktrees/maintenance-solver-comparison/marl')
sys.path.insert(0,str(WORK/'src'))
from ht_pdm_fjsp import maintenance_training_coverage as r
from ht_pdm_fjsp.maintenance_dispatch import DispatchState,validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig,WaitingEnv
RUN=Path('/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_training_coverage_v1_full_cpu_20261009T044236Z')
OUT=Path(__file__).resolve().parent
torch.set_num_threads(1)
coverage=json.loads((RUN/'training_coverage.json').read_text());cfg=json.loads((RUN/'benchmark_config.json').read_text())
selected=json.loads((RUN/'checkpoint_selection.json').read_text());refs=json.loads((RUN/'reference_selection.json').read_text())['selected']
for record in coverage:
 reg,arm,seed=record['training_regime'],record['algorithm'],record['train_seed']
 directory=RUN/reg/arm/f'train_seed_{seed}'
 initial=None;last=None
 for steps in [0,122880,245760,368640,491520]:
  m,p=r.load_checkpoint(directory/'checkpoints'/f'step_{steps:08d}.pt')
  assert p['physical_steps']==steps and p['rollout_update']==steps//1536 and p['train_seed']==seed and p['training_regime']==reg and p['algorithm']==arm
  assert sum(v.numel() for v in m.parameters())==record['parameter_count']
  assert all(torch.isfinite(v).all() for v in m.state_dict().values())
  if steps==0:
   initial={k:v.clone() for k,v in m.state_dict().items()}
   digest=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
   assert digest==record['initial_parameters_sha256']
   regenerated=r.base.model_for(cfg,seed,arm)
   assert all(torch.equal(v,regenerated.state_dict()[k]) for k,v in initial.items())
  if steps==491520:last={k:v.clone() for k,v in m.state_dict().items()}
 assert any(not torch.equal(v,last[k]) for k,v in initial.items())
 final,p=r.load_checkpoint(directory/'model.pt')
 assert all(torch.equal(v,final.state_dict()[k]) for k,v in last.items())
print('All360checkpoints validated; initial regenerated; final exactcopy; weights finite/changed',flush=True)
states=json.loads((RUN/'latency_states.json').read_text());assert len(states)==48
# Regenerate rule state trajectories on the SAME development panel, never sealed test.
for j,cond in enumerate(r.CONDITIONS):
 c=r.cohort_config(cfg['development_cohorts'][0],cond,12,profile='full')
 e=WaitingEnv(c);e.reset(cfg['development_seeds'][0])
 for t in range(12):
  logged=states[j*12+t]
  assert logged['condition']==cond and logged['remaining']==12-t
  assert json.loads(json.dumps(asdict(c)))==logged['config']
  assert json.loads(json.dumps(asdict(e.state)))==logged['state']
  e.step(r.reference(c,e.state,12-t,refs[cond]))
logged_actions={}
for row in csv.DictReader((RUN/'latency.csv').open()):
 key=(row['training_regime'],row['controller'],int(row['train_seed']),int(row['state_index']))
 action=tuple(tuple(v) for v in json.loads(row['action']))
 if key in logged_actions:assert action==logged_actions[key]
 logged_actions[key]=action
replays=0;mismatches=[]
for choice in selected.values():
 m,_=r.load_checkpoint(RUN/choice['checkpoint'])
 for i,logged in enumerate(states):
  c=WaitingConfig(**{k:tuple(tuple(x) for x in v) if k in ('service_time','restoration') else v for k,v in logged['config'].items()})
  s=DispatchState(**{k:tuple(v) for k,v in logged['state'].items()})
  action,diag=m.select(c,s,logged['remaining']);validate_matching(c,s,action)
  key=(choice['training_regime'],choice['algorithm'],choice['train_seed'],i)
  if action!=logged_actions[key]:mismatches.append(dict(key=key,original=logged_actions[key],replay=action))
  replays+=1
for i,logged in enumerate(states):
 c=WaitingConfig(**{k:tuple(tuple(x) for x in v) if k in ('service_time','restoration') else v for k,v in logged['config'].items()})
 s=DispatchState(**{k:tuple(v) for k,v in logged['state'].items()})
 for name in cfg['controls']:validate_matching(c,s,logged_actions[('reference',name,-1,i)])
 action=r.reference(c,s,logged['remaining'],refs[logged['condition']])
 assert action==logged_actions[('reference','selected_rule',-1,i)]
output=dict(checkpoint_files=360,initial_tensors_regenerated=True,final_weights_equal_final_snapshot=True,finite_parameters=True,changed_weights=True,development_rule_states_regenerated=48,model_state_replays=replays,greedy_action_mismatches=mismatches,scope='common development state actions only; no full test trajectory or training rerun')
(OUT/'checkpoint_replay.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output),flush=True)
