"""Standalone descriptive DEV figures after independent verification; no learning."""
import argparse,csv,hashlib,json,shutil,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CORE=('gibbs_joint_entropy_ppo','myopic_joint_entropy_ppo','reverse_joint_entropy_ppo','gibbs_machine_entropy_ppo','myopic_machine_entropy_ppo','reverse_machine_entropy_ppo')
ARMS=(*CORE,'ppo','normalized_ppo','mappo','full_mappo')
LABELS=('Gibbs H','Myopic H','Reverse H','Gibbs H/4','Myopic H/4','Reverse H/4','PPO','Normalized PPO','Local MAPPO','Full-state MAPPO')

def read(p):return json.loads(p.read_text())
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(source,verification,output):
    source,verification,out=map(lambda p:Path(p).resolve(),(source,verification,output));out.mkdir(parents=True,exist_ok=False);start=time.monotonic();audit=read(verification/'verification.json');vm=read(verification/'manifest.json');sm=read(source/'manifest.json');summary=read(source/'summary.json')
    assert vm['status']=='COMPLETED' and audit['status']=='PASS' and audit['source']==str(source) and audit['source_manifest_sha256']==sha(source/'manifest.json') and sm['status']=='COMPLETED' and audit['original_development_pass']==summary['development_screen_pass'] and not summary['test_opened']
    for name,meta in vm['outputs'].items():assert sha(verification/name)==meta['sha256'] and (verification/name).stat().st_size==meta['bytes']
    means=summary['means'];fig=plt.figure(figsize=(15,9),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1.65,1]);ax=fig.add_subplot(grid[0,:]);x=np.arange(10);colors=['#006c80','#d47a00','#9374aa','#74b5c0','#e2ba7c','#c9b6d6','#596575','#8b98a5','#a4aaae','#b8bfc5'];values=[means[a]['normalized_cost'] for a in ARMS]
    ax.bar(x,values,color=colors);ax.set_xticks(x,LABELS,rotation=25,ha='right');ax.set_ylabel('Extended cost per machine (lower is better)');ax.set_ylim(min(values)*.96,max(values)*1.035);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for i,v in enumerate(values):ax.text(i,v+max(values)*.001,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
    ax.set_title('N4/K2 — fixed final checkpoint; 3 training replicas × 32 paired DEV seeds',fontsize=13)
    for column,(metric,title,bound) in enumerate((('availability','Availability difference',-.01),('normalized_pending','Pending jobs / machine difference',.025),('normalized_failures','Failures / machine difference',.25))):
        a=fig.add_subplot(grid[1,column]);d=[means[CORE[0]][metric]-means[r][metric] for r in ('ppo','normalized_ppo')];a.bar([0,1],d,color=['#596575','#8b98a5']);a.set_xticks([0,1],['vs PPO','vs normalized PPO']);a.axhline(bound,color='#b92323',ls='--',label=f'Locked limit {bound:+g}');a.axhline(0,color='black',lw=.6);a.set_title(title,fontsize=11);a.legend(fontsize=9);a.grid(axis='y',alpha=.2)
    n=sum(summary['checks'].values());fig.suptitle(f'DEV point gates: {n}/15 — overall screen {"PASS" if summary["development_screen_pass"] else "FAIL"}\nDescriptive points only; no held-out TEST or confidence claim',fontsize=15);fig.savefig(out/'dev_quality_health.png',dpi=160);plt.close(fig)
    dev=rows(source/'development_episodes.csv');probe=rows(source/'entropy_probe.csv');steps=(0,122880,245760,368640,491520);trains=read(source/'resolved_config.json')['train_seeds'];fig,axes=plt.subplots(3,2,figsize=(14,11),layout='constrained')
    for i,name in enumerate(('Gibbs','Myopic','Reverse-myopic')):
        for j,(suffix,color) in enumerate((('H','#006c80'),('H/4','#d47a00'))):
            arm=CORE[i+3*j];costs=np.array([[np.mean([float(r['normalized_cost']) for r in dev if r['method']==arm and int(r['train_seed'])==t and int(r['checkpoint_steps'])==s]) for s in steps] for t in trains]);entropy=np.array([[float(next(r['mean_normalized_entropy'] for r in probe if r['method']==arm and int(r['train_seed'])==t and int(r['checkpoint_steps'])==s)) for s in steps] for t in trains])
            for k in range(3):axes[i,0].plot(steps,costs[k],color=color,alpha=.25,lw=1);axes[i,1].plot(steps,entropy[k],color=color,alpha=.25,lw=1)
            axes[i,0].plot(steps,costs.mean(0),'-o',color=color,label=suffix,lw=2);axes[i,1].plot(steps,entropy.mean(0),'-o',color=color,label=suffix,lw=2)
        axes[i,0].set_title(f'{name}: paired DEV cost');axes[i,0].set_ylabel('Cost / machine');axes[i,1].set_title(f'{name}: same 48 diagnostic states');axes[i,1].set_ylabel('Physical entropy / log(support size)');axes[i,1].set_ylim(-.03,1.03)
        for a in axes[i]:a.grid(alpha=.2);a.legend(title='Entropy units');a.ticklabel_format(axis='x',style='sci',scilimits=(3,3));a.set_xlabel('Training decisions')
    fig.suptitle('All six core arms — thin lines: each replica; thick lines: mean\nEntropy panel is one fixed deadline episode; never a model-selection score',fontsize=14);fig.savefig(out/'dev_learning_entropy.png',dpi=160);plt.close(fig)
    report=dict(classification='DESCRIPTIVE_DEV_REPORT_ONLY',source=str(source),verification=str(verification),source_manifest_sha256=sha(source/'manifest.json'),verification_manifest_sha256=sha(verification/'manifest.json'),original_development_pass=summary['development_screen_pass'],checks=summary['checks'],extra_checks=summary['extra_checks'],factorial_descriptors=audit['factorial_descriptors'],secondary_entropy_readout=audit['common_probe']['secondary'],goal_pass=False,test_opened=False)
    (out/'report.json').write_text(json.dumps(report,indent=2));shutil.copy2(__file__,out/'reporter.py');(out/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',elapsed_seconds=time.monotonic()-start,outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()}),indent=2));print(json.dumps(dict(output=str(out),status='PASS',development_pass=report['original_development_pass'])))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.verification,a.output)
