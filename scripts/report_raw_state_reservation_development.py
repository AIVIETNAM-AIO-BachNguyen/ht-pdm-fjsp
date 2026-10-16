"""Read-only DEV plots and physical accounting; no new scientific endpoints."""
import argparse,csv,hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np

CORE=('raw_reservation_ppo','mask_only_ppo','reverse_reservation_ppo','independent_raw_ppo')
ARMS=(*CORE,'ppo','normalized_ppo','mappo','full_mappo')
LABELS=dict(zip(ARMS,('Reservation','Mask only','Reverse','Independent','PPO','PPO normalized','MAPPO local','MAPPO full info')))
LABELS.update({f'frozen_{a}':f'Reservation weights / {LABELS[a].lower()}' for a in CORE[1:]})
COLORS=dict(zip(ARMS,('#2166ac','#4393c3','#67a9cf','#d6604d','#444444','#888888','#998ec3','#7b3294')))
ROOT=Path(__file__).resolve().parents[1]
PLOTPY='/home/bachng/.cache/uv/archive-v0/ClsAIx6OkIupFN6T/bin/python'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def csvrows(p):
    with p.open() as f:return list(csv.DictReader(f))


def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    data=read(out/'analysis.json');v=data['verification'];per=v['selected']['per_replica'];means=v['selected']['means'];y=np.arange(8)
    fig,axs=plt.subplots(2,2,figsize=(12.4,8.5),constrained_layout=True)
    specifications=[('normalized_cost','Cost / machine (48 decisions + 12-tick tail)',min(means[a]['normalized_cost'] for a in ('ppo','normalized_ppo'))*1.01,'maximum'),
        ('availability','Availability during 48 policy ticks',max(means[a]['availability'] for a in ('ppo','normalized_ppo'))-.01,'minimum'),
        ('normalized_failures','Failures / machine during policy',min(means[a]['normalized_failures'] for a in ('ppo','normalized_ppo'))+.25,'maximum'),
        ('normalized_pending','Pending / machine at tick 48',min(means[a]['normalized_pending'] for a in ('ppo','normalized_ppo'))+.025,'maximum')]
    for ax,(metric,label,threshold,bound) in zip(axs.flat,specifications):
        for i,a in enumerate(ARMS):
            values=[r[metric] for r in per[a].values()];color=COLORS[a]
            ax.plot([min(values),max(values)],[i,i],color=color,lw=2)
            ax.scatter(values,np.full(3,i)+np.array([-.12,0,.12]),s=18,color=color,alpha=.65)
            ax.scatter([means[a][metric]],[i],marker='D',s=45,color=color,zorder=4)
        ax.axvline(threshold,color='#b2182b',ls='--',lw=1.3,label=f'PPO health/quality {bound}')
        ax.set_yticks(y,labels=[LABELS[a] for a in ARMS]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_xlabel(label);ax.legend(fontsize=8,loc='best')
    fig.suptitle('Raw-state reservation PPO: N4/K2 full development\nDiamonds: means; dots/ranges: three training replicas (not confidence intervals)',fontsize=13)
    fig.savefig(out/'quality_health.png');plt.close(fig)
    groups=[*CORE,*[f'frozen_{a}' for a in CORE[1:]]]
    fig,axs=plt.subplots(1,2,figsize=(12.5,5.8),constrained_layout=True)
    for ax,metric,label in zip(axs,('normalized_cost','normalized_failures'),('Cost / machine (including common tail)','Failures / machine during policy')):
        for i,a in enumerate(groups):
            values=[r[metric] for r in per[a].values()];color=COLORS.get(a,'#1b9e77')
            ax.plot([min(values),max(values)],[i,i],color=color,lw=2);ax.scatter(values,np.full(3,i)+[-.12,0,.12],color=color,s=20,alpha=.65)
            ax.scatter([means[a][metric]],[i],color=color,marker='D',s=45)
        ax.set_yticks(np.arange(7),labels=[LABELS[a] for a in groups]);ax.invert_yaxis();ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Trained coordination controls and frozen candidate interventions\nSame current information; frozen controls add no learning (DEV, no CI)',fontsize=12)
    fig.savefig(out/'mechanism.png');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14.5,4.8),constrained_layout=True)
    for ax,metric,title,methods in [(axs[0],'normalized_cost','Development cost / machine',ARMS[:6]),(axs[1],'normalized_failures','Development failures / machine',ARMS[:6]),(axs[2],'availability','Development availability',ARMS[:6])]:
        for a in methods:
            allvalues=np.array([[r[metric] for r in data['curves'] if r['method']==a and r['steps']==step] for step in (122880,245760,368640,491520)])
            xs=np.array([122880,245760,368640,491520])/1000;ax.plot(xs,allvalues.mean(1),marker='o',ms=3,color=COLORS[a],label=LABELS[a])
            ax.fill_between(xs,allvalues.min(1),allvalues.max(1),color=COLORS[a],alpha=.08)
        ax.set_xlabel('Training decisions (thousands)');ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('All positive-step checkpoints; shading is replica min/max, not CI\nInitial step 0 retained in analysis.json and source CSV; no checkpoint/budget changes',fontsize=11)
    fig.savefig(out/'learning_curves.png');plt.close(fig)


def run(source,verification,output):
    import torch
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    v=read(verification/'verification.json');vm=read(verification/'manifest.json');m=read(source/'manifest.json')
    assert v['status']=='PASS' and vm['status']=='COMPLETED' and v['source']==str(source) and v['source_manifest_sha256']==sha(source/'manifest.json')
    for name,meta in vm['outputs'].items():assert sha(verification/name)==meta['sha256']
    for name,meta in m['output_bindings'].items():assert sha(source/name)==meta['sha256']
    episodes=csvrows(source/'selected_episodes.csv');dev=csvrows(source/'development_episodes.csv');curves=[]
    for a in ARMS:
        for t in range(270000000,270000003):
            for step in (0,122880,245760,368640,491520):
                group=[r for r in dev if r['method']==a and int(r['train_seed'])==t and int(r['checkpoint_steps'])==step]
                assert len(group)==32
                row=dict(method=a,train_seed=t,steps=step,**{k:float(np.mean([float(r[k]) for r in group])) for k in ('normalized_cost','normalized_failures','availability','normalized_pending')})
                curves.append(row)
    decomposition={}
    for a in (*ARMS,*[f'frozen_{a}' for a in CORE[1:]],'deadline'):
        group=[r for r in episodes if r['method']==a]
        for r in group:assert abs(sum(float(r[k]) for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost'))-float(r['extended_cost']))<1e-8
        decomposition[a]={k:float(np.mean([float(r[k])/4 for r in group])) for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost','extended_cost','service_ticks','completed_jobs')}
    differences={a:{k:decomposition[CORE[0]][k]-decomposition[a][k] for k in decomposition[a]} for a in ('ppo','normalized_ppo',CORE[3])}
    data=dict(classification='POSTHOC_DEVELOPMENT_REPORT',verification=v,curves=curves,physical_decomposition_per_machine=decomposition,candidate_minus_control=differences,
        limitation='Cost components and service/completion counts cover all60ticks; availability/failure gate uses policy48ticks. No CI or new endpoint/trajectory.')
    (out/'analysis.json').write_text(json.dumps(data,indent=2)+'\n');shutil.copy2(__file__,out/'reporter.py')
    subprocess.run([PLOTPY,str(Path(__file__).resolve()),'--plot-only','--output',str(out)],check=True)
    assert sha(source/'manifest.json')==v['source_manifest_sha256']
    manifest=dict(status='COMPLETED',classification=data['classification'],source=str(source),source_manifest_sha256=sha(source/'manifest.json'),
        verification=str(verification),verification_manifest_sha256=sha(verification/'manifest.json'),goal_pass=False,test_opened=False,
        output_bindings={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(output=str(out),differences=differences)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--verification');p.add_argument('--output',required=True);p.add_argument('--plot-only',action='store_true');a=p.parse_args()
    if a.plot_only:plots(Path(a.output))
    else:run(a.source,a.verification,a.output)

if __name__=='__main__':main()
