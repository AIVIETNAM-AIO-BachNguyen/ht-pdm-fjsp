"""Read-only descriptive figures for the independently verified frozen diagnostic."""
import argparse,hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(source,verification,output):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);sm=read(source/'manifest.json');vm=read(verification/'manifest.json');v=read(verification/'verification.json')
    assert sm['status']=='COMPLETED' and sm['audit_status']=='PASS' and vm['status']=='COMPLETED' and v['status']=='PASS' and v['source']==str(source) and v['source_manifest_sha256']==sha(source/'manifest.json')
    summary=read(source/'summary.json');early=(15360,30720,61440,92160,122880);trains=sorted({r['train_seed'] for r in summary['records']});aggregate={}
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained');specs=[('relative_alias_gap',100.,10.,'Relative proposal-minus-matching entropy (%)'),('entropy_gradient_cosine',1.,0.,'Cosine: proposal vs matching entropy gradients'),('replacement_norm_ratio',100.,10.,'Scaled replacement norm / reward-gradient norm (%)'),('projected_replacement',100.,1.,'Local replacement projection / reward-gradient norm² (%)')]
    for t in trains:
        rows=sorted([r for r in summary['records'] if r['train_seed']==t and r['checkpoint_steps'] in early],key=lambda r:r['checkpoint_steps']);aggregate[str(t)]={k:dict(mean=float(np.mean([r[k] for r in rows])),minimum=float(min(r[k] for r in rows)),maximum=float(max(r[k] for r in rows))) for k,_,_,_ in specs}
        for axis,(key,factor,threshold,label) in zip(axes.flat,specs):axis.plot(np.array(early)/1000,[factor*r[key] for r in rows],marker='o',ms=4,label=f'Replica {trains.index(t)+1}')
    for axis,(key,factor,threshold,label) in zip(axes.flat,specs):axis.axhline(threshold,color='#C62828',ls='--',lw=1,label='Locked threshold');axis.set_xlabel('Frozen checkpoint steps (thousands)');axis.set_ylabel(label,fontsize=9);axis.grid(alpha=.15)
    axes[0,1].set_ylim(-.05,1.05);axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle('Frozen entropy diagnostic: primary five early checkpoints only; no CI / optimizer step',fontsize=12);fig.savefig(out/'entropy_alignment.png',dpi=170);plt.close(fig)
    data=dict(classification='FROZEN_DIAGNOSTIC_DESCRIPTIVE_ONLY',source=str(source),verification=str(verification),summary=summary,primary_ranges=aggregate,verification_maximum_errors=v['maximum_errors'],limitation='Thresholds are an engineering screen. Projection values are not percentages of cost improvement; no training or scalable entropy estimator.')
    (out/'analysis.json').write_text(json.dumps(data,indent=2));shutil.copy2(__file__,out/'reporter.py');manifest=dict(status='COMPLETED',source_manifest_sha256=sha(source/'manifest.json'),verification_manifest_sha256=sha(verification/'manifest.json'),goal_pass=False,test_opened=False,outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(str(out))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.source,args.verification,args.output)
