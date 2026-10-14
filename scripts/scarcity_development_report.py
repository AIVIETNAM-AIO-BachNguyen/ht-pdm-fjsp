"""Read-only, timestamped development/pilot report with no scientific claims."""
import argparse,csv,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output-root',type=Path,required=True);args=p.parse_args();source=args.source
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((source/'manifest.json').read_text());assert m['status']=='COMPLETED' and m['audit_status']=='PASS' and not m['test_opened']
inputs=['summary.json','resolved_config.json','development_episodes.csv','audit.json','training_audit.json']
for name in inputs:assert sha(source/name)==m['output_bindings'][name]['sha256'],name
s=json.loads((source/'summary.json').read_text());c=json.loads((source/'resolved_config.json').read_text());assert s['classification']=='DEVELOPMENT_ONLY' and not s['teacher_assistance']
with (source/'development_episodes.csv').open() as f:rows=list(csv.DictReader(f))
colors={'scarcity_ppo':'#2166ac','reservation_ppo':'#d6604d','independent_ppo':'#8c6bb1','global_clip_ppo':'#5c8655','ppo':'#333333','mappo':'#c29b24','full_mappo':'#888888'}
out=args.output_root/(m['protocol']+'_development_report_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
fig,axs=plt.subplots(1,len(c['scales']),figsize=(6*len(c['scales']),4),squeeze=False)
for ax,n in zip(axs[0],c['scales']):
 for arm,color in colors.items():
  data=[r for r in rows if int(r['n'])==n and r['method']==arm]
  if not data:continue
  steps=sorted({int(r['checkpoint_steps']) for r in data});seeds=sorted({int(r['train_seed']) for r in data})
  values=np.array([[np.mean([float(r['normalized_cost']) for r in data if int(r['checkpoint_steps'])==step and int(r['train_seed'])==seed]) for step in steps] for seed in seeds])
  x=np.array(steps)/1000;ax.plot(x,values.mean(0),'-o',label=arm,color=color);ax.fill_between(x,values.min(0),values.max(0),color=color,alpha=.1)
 ax.axhline(s['means'][str(n)]['deadline']['normalized_cost'],color='#555555',linestyle='--',label='deadline');ax.set_title(f'N{n}, K{n//2}');ax.set_xlabel('Joint decisions / replica (thousands)');ax.set_ylabel('Development H+L cost / machine');ax.legend(fontsize=7);ax.grid(alpha=.15)
fig.suptitle(f'{m["protocol"]}: {c["profile"]}, {len(c["train_seeds"])} replicas, no scientific test',fontsize=11);fig.text(.5,.005,'Shading: all-replica min–max, not CI. Full budgets spent; checkpoint selection uses development only.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.05,1,.92));fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.pdf');plt.close(fig)
text=['# Teacher-free development findings','','Classification: DEVELOPMENT_ONLY. No scientific test opened.',f'Source: `{source}`; frozen commit `{m["git"]["commit"]}`.',f'Audit PASS. Full pilot budget {c["env_steps"]} decisions per fit; {len(c["train_seeds"])} replicas. Screen pass: {s["development_screen_pass"]}.','','| N | Method | Selected cost/machine | Availability | Pending/machine | Failures/machine |','|---|---|---:|---:|---:|---:|']
for n,panel in s['means'].items():
 for arm,v in panel.items():text.append(f'| {n} | {arm} | {v["normalized_cost"]:.4f} | {v["availability"]:.6f} | {v["normalized_pending"]:.6f} | {v["normalized_failures"]:.6f} |')
text+=['','## Descriptive intervention activity','','| N | Step | Price mean magnitude | Active slot fraction | Same-weight unpriced matching changes/episode |','|---|---:|---:|---:|---:|']
for n in c['scales']:
 for step in [0,*c['checkpoints']]:
  panel=[r for r in rows if int(r['n'])==n and r['method']=='scarcity_ppo' and int(r['checkpoint_steps'])==step];v={k:np.mean([float(r[k]) for r in panel]) for k in ('price_mean_abs','price_active_fraction','same_weight_unpriced_matching_changes')}
  text.append(f'| {n} | {step} | {v["price_mean_abs"]:.8f} | {v["price_active_fraction"]:.6f} | {v["same_weight_unpriced_matching_changes"]:.4f} |')
text+=['','## Learning time','','| N | Method | Seed | Seconds including dev | Decisions | Optimizer steps |','|---|---|---:|---:|---:|---:|']
for r in json.loads((source/'training_audit.json').read_text()):text.append(f'| {r["n"]} | {r["algorithm"]} | {r["train_seed"]} | {r["seconds_including_development"]:.2f} | {r["learned_steps"]} | {r["optimizer_steps"]} |')
text+=['','These are exploratory development results with short pilot budgets. Point gains','over equally short baseline fits are not evidence against full-budget PPO and','not scientific superiority claims. No confidence interval or alpha is spent.','Preserve negative screens and test the next design on a new registered panel.']
(out/'findings.md').write_text('\n'.join(text)+'\n');shutil.copyfile(__file__,out/'report_source.py')
(out/'report_manifest.json').write_text(json.dumps(dict(classification='DEVELOPMENT_ONLY',source=str(source),input_hashes={name:sha(source/name) for name in ['manifest.json',*inputs]},script_sha256=sha(Path(__file__)),matplotlib=matplotlib.__version__,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out)
