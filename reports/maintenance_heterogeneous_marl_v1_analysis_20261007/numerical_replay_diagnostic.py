import csv,json,sys,types
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm.auto import tqdm
ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl');RAW=ROOT/'lab_results/maintenance_heterogeneous_marl_v1_full_cpu_20261006T150059Z';OUT=Path(__file__).resolve().parent
a=json.loads((OUT/'physical_policy_audit.json').read_text())
package=types.ModuleType('ht_pdm_fjsp');package.__path__=[str(RAW/'source_snapshot')];sys.modules['ht_pdm_fjsp']=package
from ht_pdm_fjsp import maintenance_heterogeneous_marl as r
from ht_pdm_fjsp import maintenance_heterogeneous_model as world
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

torch.set_num_threads(1)
selection=json.loads((RAW/'checkpoint_selection.json').read_text())['selections'];models={}
expected={}
for row in csv.DictReader((RAW/'episodes.csv').open()):
 key=(int(row['train_seed']),int(row['cohort']),row['condition'],int(row['horizon']),int(row['seed']),row['controller'])
 expected[key]=float(row['objective'])
keys=sorted({tuple(m['key']) for m in a['mismatches']});counterfactual=[]
for key in tqdm(keys,desc='cross-platform scalar diagnostic'):
 seed,cs,cond,h,es,arm=key;mk=(arm,seed)
 if mk not in models:models[mk],_=r.load_checkpoint(RAW/selection[f'{arm}:{seed}']['checkpoint'])
 net=models[mk];c=world.cohort_config(cs,cond,h);env=WaitingEnv(c);env.reset(es)
 for t in range(h):
  pairs,_=net.select(c,env.state,h-t);env.step(pairs)
 difference=env.metrics['objective']-expected[key]
 counterfactual.append(dict(key=key,recorded_cost=expected[key],local_scalar_cost=env.metrics['objective'],difference=difference))
result=dict(status='POSTHOC_NUMERICAL_SENSITIVITY',batch_action_mismatches=len(a['mismatches']),
 scalar_matches_on_logged_state=sum(m['scalar_physical']==m['logged_physical'] for m in a['mismatches']),
 changed_episode_costs=sum(abs(r['difference'])>1e-9 for r in counterfactual),
 max_episode_cost_change=max(abs(r['difference']) for r in counterfactual),
 total_difference_on_affected_episodes=sum(r['difference'] for r in counterfactual),
 affected_episodes=len(keys),rows=counterfactual,
 scope='Only episodes with cross-platform/batch action mismatch were re-evaluated on local Mac CPU. Not a rerun of training or a replacement of original test estimates. No primary specialized or skill-mask actions mismatched.')
(OUT/'numerical_replay_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
