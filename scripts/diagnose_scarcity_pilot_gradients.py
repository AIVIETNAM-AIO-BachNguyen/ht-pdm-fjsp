"""Exploratory current-policy gradients on the closed pilot development panel."""
import argparse,json,hashlib,csv
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from ht_pdm_fjsp import technician_scarcity_experiment as x
from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors,mc_returns
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output-root',type=Path,required=True);args=p.parse_args()
source=args.source;manifest=json.loads((source/'manifest.json').read_text());assert manifest['status']=='COMPLETED' and manifest['audit_status']=='PASS' and not manifest['test_opened']
c=json.loads((source/'resolved_config.json').read_text());torch.set_num_threads(1);rows=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
consumed={'resolved_config.json':sha(source/'resolved_config.json')}
assert consumed['resolved_config.json']==manifest['output_bindings']['resolved_config.json']['sha256']
for name,digest in manifest['source_seal'].items():
 if name.startswith('source_snapshot/'):
  assert sha(x.ROOT/'src/ht_pdm_fjsp'/Path(name).name)==digest,name
def norm(params):return float(torch.sqrt(sum((p.grad.square().sum() for p in params if p.grad is not None),torch.tensor(0.))))
for n in c['scales']:
 for seed in c['train_seeds']:
  for step in [0,*c['checkpoints']]:
   path=source/f'N{n}/scarcity_ppo/{seed}/checkpoints/step_{step:09d}.pt';model=x.load_model(path,n,c,seed,'scarcity_ppo',step)
   relative=str(path.relative_to(source));consumed[relative]=sha(path);assert consumed[relative]==manifest['output_bindings'][relative]['sha256']
   data=[];rewards=[];done=[]
   # Reuses only one already opened DEV seed; no new/test panel, no update.
   env=x.ResourceEnv(c['development_seeds'][0],n,'scarcity_ppo');obs=env.reset()
   for tick in range(48):
    batch=tensors([obs]);action=model.select(obs);data.append((batch,torch.tensor(action)[None]));obs,r,d,info=env.step(action);rewards.append([r]);done.append([d])
   batch={key:torch.cat([r[0][key] for r in data]) for key in data[0][0]};actions=torch.cat([r[1] for r in data])
   log,entropy,value=model.evaluate(batch,actions);returns=mc_returns(torch.tensor(rewards),torch.tensor(done)).reshape(-1)
   advantage=returns-value.detach();adv=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
   actor=-torch.mean(log*adv)-c['entropy_weight']*entropy.mean()
   critic=.5*torch.nn.functional.mse_loss(value/(n/4),returns/(n/4))
   (actor+critic).backward();critic_params=[p for name,p in model.named_parameters() if name.startswith(('value_encoder.','critic.'))];actor_params=[p for name,p in model.named_parameters() if not name.startswith(('value_encoder.','critic.'))]
   an,cn=norm(actor_params),norm(critic_params);whole=(an*an+cn*cn)**.5;factor=min(1.,c['gradient_clip']/max(whole,1e-12))
   rows.append(dict(n=n,train_seed=seed,step=step,value_loss=float(critic.detach()),actor_grad=an,critic_grad=cn,global_clip_factor=factor,actor_grad_after_global_clip=an*factor,actor_grad_after_own_clip=min(an,c['gradient_clip']),mean_value_error=float(advantage.abs().mean()),episode_cost=info['episode_metrics']['extended_cost']))
out=args.output_root/('technician_scarcity_ppo_v1_gradient_diagnostic_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
with (out/'gradients.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(out/'diagnostic_manifest.json').write_text(json.dumps(dict(classification='EXPLORATORY_DEV_ONLY_NO_TRAINING',source=str(source),source_manifest_sha256=sha(source/'manifest.json'),input_hashes=consumed,script_sha256=sha(Path(__file__)),development_seed=c['development_seeds'][0],model_updates=0,test_opened=False,limitation='Greedy current-policy dev gradients, not logged training updates. Adam normalization prevents inferring a proportional parameter-update reduction from clipping alone.'),indent=2)+'\n')
(out/'diagnostic_source.py').write_text(Path(__file__).read_text());print(out);print(json.dumps(rows))
