"""Read-only supplementary plots; refuses incomplete runs and never edits inputs."""
import argparse,csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output-root',required=True);a=p.parse_args()
source=Path(a.source).resolve();manifest=json.loads((source/'manifest.json').read_text())
if manifest['status']!='COMPLETED' or manifest.get('audit_status')!='PASS':raise SystemExit('Refusing analysis of incomplete/unverified run')
bindings=manifest.get('output_bindings',{})
for name in ('summary.json','resolved_config.json','development_episodes.csv','audit.json'):
    if name in bindings and hashlib.sha256((source/name).read_bytes()).hexdigest()!=bindings[name]['sha256']:raise SystemExit('Consumed input differs from sealed artifact')
summary=json.loads((source/'summary.json').read_text());cfg=json.loads((source/'resolved_config.json').read_text())
out=Path(a.output_root)/f'{manifest["protocol"]}_analysis_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}';out.mkdir(parents=True)
colors={'reservation_ppo':'#1764ab','independent_ppo':'#d46a1c','ppo':'#333d46','mappo':'#8150ad','deadline':'#4c836c'}
labels={'reservation_ppo':'Reservation PPO','independent_ppo':'No reservation','ppo':'Central PPO','mappo':'Local MAPPO','deadline':'Deadline matching'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
with (source/'development_episodes.csv').open() as f:rows=list(csv.DictReader(f))
teacher_assisted='teacher_bank_steps' in cfg
if teacher_assisted:
    labels.update(reservation_ppo='Teacher-guided reservation PPO',independent_ppo='Teacher-guided independent PPO',ppo='Frozen PPO reference',mappo='Frozen MAPPO reference')
scales=cfg['scales'];fig,axs=plt.subplots(1,len(scales),figsize=(5.4*len(scales),4.2),squeeze=False)
for ax,n in zip(axs[0],scales):
    for arm in colors:
        data=[r for r in rows if int(r['n'])==n and r['method']==arm]
        if not data:continue
        steps=sorted({int(r['checkpoint_steps']) for r in data});replicas=sorted({int(r['train_seed']) for r in data})
        values=np.array([[np.mean([float(r['extended_cost'])/n for r in data if int(r['checkpoint_steps'])==step and int(r['train_seed'])==t]) for step in steps] for t in replicas])
        if teacher_assisted and arm in ('ppo','mappo'):
            ax.axhline(values.mean(),color=colors[arm],linestyle='--',label=labels[arm])
            continue
        x=np.array(steps)/1000;ax.plot(x,values.mean(0),'-o',color=colors[arm],label=labels[arm],markersize=4)
        ax.fill_between(x,values.min(0),values.max(0),color=colors[arm],alpha=.10)
    ax.set_title(f'{n} machines / {n//2} technicians');ax.set_xlabel('PPO fine-tuning decisions after warm start (thousands)' if teacher_assisted else 'Learned decisions per replica (thousands)')
    ax.set_ylabel('H+L cost per machine');ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.suptitle(f'{manifest["protocol"]} — development learning curves',fontsize=12)
fig.text(.5,.005,('Teacher bank and foundation-training budgets are additional. ' if teacher_assisted else '')+'Shading is the min–max across training replicas, not a confidence interval. Development results guide design; no test claim.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(out/'development_learning_curves.png',dpi=180);fig.savefig(out/'development_learning_curves.pdf');plt.close(fig)
if summary.get('classification')=='DEVELOPMENT_ONLY':
    table=[]
    for n,panel in summary['means'].items():
        for arm,cost in panel.items():table.append(dict(n=int(n),method=arm,cost=cost,cost_per_machine=cost/int(n)))
    with (out/'selected_development_means.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
else:
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.1))
    for arm in colors:
        pairs=[(n,summary['means'][str(n)][arm]['normalized_cost']) for n in scales if arm in summary['means'][str(n)]]
        if pairs:axs[0].plot([n for n,_ in pairs],[v for _,v in pairs],'-o',label=labels[arm],color=colors[arm])
    axs[0].set_xlabel('Machines');axs[0].set_ylabel('Test H+L cost per machine');axs[0].legend(fontsize=8)
    keys=['N4_gain_vs_ppo',*[f'N{n}_gain_vs_independent_ppo' for n in scales if n!=4]]
    intervals=[summary['intervals'][k] for k in keys];est=np.array([i['estimate']*100 for i in intervals]);low=np.array([i['lower']*100 for i in intervals]);high=np.array([i['upper']*100 for i in intervals])
    axs[1].errorbar(range(len(keys)),est,yerr=np.stack([est-low,high-est]),fmt='o',capsize=4,color=colors['reservation_ppo'])
    axs[1].axhline(0,color='#888888',linewidth=1);axs[1].set_xticks(range(len(keys)),['N4 vs PPO',*[f'N{n} vs no reservation' for n in scales if n!=4]],rotation=25,ha='right')
    axs[1].set_ylabel('Cost reduction (%)');axs[1].set_title('Predeclared simultaneous intervals')
    lat=[summary['latency'][f'{n}:reservation_ppo']['p95_ms'] for n in scales]
    axs[2].plot(scales,lat,'-o',color=colors['reservation_ppo']);axs[2].axhline(50,color='#b9483b',linestyle='--',label='Locked 50 ms limit')
    axs[2].set_xlabel('Machines');axs[2].set_ylabel('P95 single-state decision latency (ms)');axs[2].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.15)
    test_label='engineering-only smoke' if cfg['profile']=='smoke' else 'sealed scientific tests'
    fig.suptitle(f'{manifest["protocol"]} — {test_label}; goal pass = {summary["goal_pass"]}',fontsize=12)
    fig.tight_layout();fig.savefig(out/'test_quality_mechanism_latency.png',dpi=180);fig.savefig(out/'test_quality_mechanism_latency.pdf');plt.close(fig)
ns=[4,8,16,32];counts=[sum(math.comb(n,r)*math.comb(n//2,r)*math.factorial(r) for r in range(n//2+1)) for n in ns]
fig,ax=plt.subplots(figsize=(6.8,3.8));ax.plot(ns,counts,'-o',label='Complete matching catalogue',color='#333d46');ax.plot(ns,[n*n//2 for n in ns],'-o',label='Resource graph edges',color='#1764ab')
ax.set_yscale('log');ax.set_xticks(ns);ax.set_xlabel('Machines (technicians = machines / 2)');ax.set_ylabel('Count, logarithmic scale');ax.grid(alpha=.15);ax.legend();ax.set_title('Static representation sizes, not a runtime measurement');fig.tight_layout();fig.savefig(out/'representation_sizes.png',dpi=180);fig.savefig(out/'representation_sizes.pdf');plt.close(fig)
if teacher_assisted:
    warm=json.loads((source/'warm_choices.json').read_text())
    warm=dict(sorted(warm.items(),key=lambda item:int(item[0])))
    fig,axs=plt.subplots(1,2,figsize=(10.4,4.4))
    for arm in ('reservation_ppo','independent_ppo'):
        curves=[]
        for seed,panel in warm.items():
            with (source/'warm'/seed/f'{arm}_pretraining.csv').open() as stream:curve=list(csv.DictReader(stream))
            curves.append([float(row['mean_nll']) for row in curve])
        values=np.array(curves);epochs=np.arange(1,values.shape[1]+1)
        axs[0].plot(epochs,values.mean(0),color=colors[arm],label=labels[arm]);axs[0].fill_between(epochs,values.min(0),values.max(0),color=colors[arm],alpha=.1)
        agreement=[panel['arms'][arm]['teacher_matching_agreement']*100 for panel in warm.values()]
        xpos=np.arange(len(agreement))+(0 if arm=='reservation_ppo' else .35)
        axs[1].bar(xpos,agreement,width=.35,label=labels[arm],color=colors[arm])
    axs[0].set_xlabel('Fixed pretraining epoch');axs[0].set_ylabel('Teacher-label NLL / technician');axs[0].legend(fontsize=7)
    axs[1].set_xticks(np.arange(len(warm))+.175,list(warm));axs[1].set_xlabel('Student seed (mapped teacher bundle)');axs[1].set_ylabel('Exact teacher matching agreement (%)');axs[1].set_ylim(0,100);axs[1].legend(fontsize=7,loc='lower right',bbox_to_anchor=(1,1.02))
    for ax in axs:ax.grid(alpha=.15)
    fig.suptitle('Training-bank diagnostics; fixed epochs, no scientific test',fontsize=12);fig.tight_layout(rect=(0,0,1,.88));fig.savefig(out/'teacher_transfer.png',dpi=180);fig.savefig(out/'teacher_transfer.pdf');plt.close(fig)
inputs=['manifest.json','summary.json','resolved_config.json','development_episodes.csv','audit.json']
if teacher_assisted:inputs+=['warm_choices.json','teachers/references.json']
report=dict(classification='READ_ONLY_SUPPLEMENT',source=str(source),protocol=manifest['protocol'],git=manifest['git'],
    report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in inputs},
    matplotlib=matplotlib.__version__,numpy=np.__version__,figures=sorted(x.name for x in out.glob('*.png')))
(out/'analysis_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(out)
