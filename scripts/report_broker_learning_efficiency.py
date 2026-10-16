"""Descriptive figures for the independently verified fixed-budget efficiency run."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def table(p):
    with p.open() as f:return list(csv.DictReader(f))
def check(ok,label):
    if not ok:raise AssertionError(label)


def run(source,verification,output):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);sm=read(source/'manifest.json');vm=read(verification/'manifest.json');v=read(verification/'verification.json')
    check(sm['status']=='COMPLETED' and sm['audit_status']=='PASS' and vm['status']=='COMPLETED' and v['status']=='PASS' and v['source']==str(source) and v['source_manifest_sha256']==sha(source/'manifest.json'),'independently verified study')
    summary=read(source/'summary.json');dev=table(source/'development_episodes.csv');episodes=table(source/'selected_episodes.csv');arms=list(summary['auc_means']);trains=sorted({int(r['train_seed']) for r in dev});steps=sorted({int(r['checkpoint_steps']) for r in dev});names=['Physical-q','Proposal-p','ID physical-q','ID proposal-p','PPO','PPO norm','MAPPO local','MAPPO full'];curves=[]
    for a in arms:
        for t in trains:
            for step in steps:
                rr=[r for r in dev if r['method']==a and int(r['train_seed'])==t and int(r['checkpoint_steps'])==step];curves.append(dict(method=a,train_seed=t,steps=step,normalized_cost=float(np.mean([float(r['normalized_cost']) for r in rr]))))
    components={a:{k:float(np.mean([float(r[k])/4 for r in episodes if r['method']==a])) for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost')} for a in arms}
    candidate=arms[0];differences={a:{k:components[candidate][k]-components[a][k] for k in components[a]} for a in (arms[1],'ppo','normalized_ppo')}
    data=dict(classification='DESCRIPTIVE_DEV_ONLY',source=str(source),verification=str(verification),summary=summary,curves=curves,final_components_per_machine=components,candidate_minus_control_components=differences,fixed_choices=read(source/'checkpoint_choices.json'),ancillary_choices=read(source/'ancillary_checkpoint_choices.json'),limitation='No CI; shading/bars describe three replica means. Fixed final245760 endpoint and prespecified true-step trapezoid only; prior long-budget failure unchanged.')
    (out/'analysis.json').write_text(json.dumps(data,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained');colors=plt.cm.tab10(np.arange(8));xx=np.array(steps)/1000
    for a,name,color in zip(arms,names,colors):
        values=np.array([[r['normalized_cost'] for r in curves if r['method']==a and r['steps']==step] for step in steps]);axes[0].plot(xx,values.mean(1),label=name,color=color,lw=1.3);axes[0].fill_between(xx,values.min(1),values.max(1),color=color,alpha=.07)
    axes[0].set_xlabel('Training decisions (thousands)');axes[0].set_ylabel('Mean H60 cost / machine');axes[0].set_title('Full prespecified learning interval');axes[0].legend(fontsize=7,ncol=2)
    per=summary['per_replica_auc'];gains=[100*per[str(t)]['gain'] for t in trains];axes[1].bar(range(3),gains,color='#1976D2');axes[1].axhline(.5,color='#C62828',ls='--',label='Locked practical AUC gain: 0.5%');axes[1].axhline(0,color='black',lw=.6);axes[1].set_xticks(range(3),['Replica1','Replica2','Replica3']);axes[1].set_ylabel('Candidate AUC gain vs proposal-p (%)');axes[1].set_title(f"Pooled AUC gain: {100*summary['endpoints']['auc_gain_vs_broker_proposal_ppo']:.3f}%");axes[1].legend(fontsize=8)
    for i,value in enumerate(gains):axes[1].annotate(f'{value:.3f}%',(i,value),xytext=(0,4 if value>=0 else -12),textcoords='offset points',ha='center',fontsize=9)
    fig.suptitle('Learning efficiency DEV — fixed interval; descriptive means/min-max, no CI',fontsize=12);fig.savefig(out/'learning_efficiency.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained');y=np.arange(8);means=summary['final_means'];ppo=means['ppo']['normalized_cost'];cost=[100*(means[a]['normalized_cost']/ppo-1) for a in arms];avail=[means[a]['availability'] for a in arms];failure=[means[a]['normalized_failures'] for a in arms]
    for axis,values,title in zip(axes,(cost,avail,failure),('H60 cost vs PPO (%)','H48 availability','H48 failures / machine')):
        axis.barh(y,values,color=['#1976D2']+['#90A4AE']*7);axis.set_yticks(y,names);axis.invert_yaxis();axis.set_xlabel(title)
    axes[0].axvline(0,color='black',lw=.6);axes[1].axvline(means['ppo']['availability']-.01,color='#C62828',ls='--',lw=1);axes[2].axvline(means['ppo']['normalized_failures']+.25,color='#C62828',ls='--',lw=1);fig.suptitle('Fixed final checkpoint245760 — original health bounds retained, no CI',fontsize=12);fig.savefig(out/'fixed_final_quality_health.png',dpi=170);plt.close(fig)
    shutil.copy2(__file__,out/'reporter.py');manifest=dict(status='COMPLETED',source_manifest_sha256=sha(source/'manifest.json'),verification_manifest_sha256=sha(verification/'manifest.json'),goal_pass=False,test_opened=False,outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(str(out))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.source,args.verification,args.output)
