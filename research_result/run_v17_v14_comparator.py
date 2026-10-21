"""Frozen V14 structured actors on the already opened V17 final DEV panel."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_framework_v1 import CASES,configs,evaluate_episode
from ht_pdm_fjsp.maintenance_framework_v14_campaign import V14Controller,load_checkpoint

ROOT=Path(__file__).resolve().parent.parent
PLAN=ROOT/'research_result/v17_v14_comparator_plan_vi.md'
V14=ROOT/'research_result/framework_v14_dev_20261020T185303Z'
V14_AUDIT=ROOT/'research_result/framework_v14_audit_20261020T200552Z'
V17_DEV=ROOT/'research_result/framework_v17_dev_20261021T022840Z'
V17_DEV_AUDIT=ROOT/'research_result/framework_v17_audit_20261021T031626Z'
V17_SMOKE=ROOT/'research_result/framework_v17_smoke_20261021T022743Z'
V17_SMOKE_AUDIT=ROOT/'research_result/framework_v17_audit_20261021T022759Z'
ARMS=('competition','count_calibrated')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def cost_tensor(controller,train_seeds,source,cases,physical):
    lookup={(r['controller'],str(r['train_seed']),r['case'],int(r['seed'])):float(r['objective'])
            for r in source}
    return np.asarray([[[lookup[(controller,str(t),c,seed)] for c in cases]
                        for seed in physical] for t in train_seeds])

def bootstrap(a,b):
    rng=np.random.default_rng(73040005);results=np.empty(20000)
    for left in tqdm(range(0,20000,1000),desc='V14/V17 hierarchical bootstrap',unit='chunk'):
        ia=rng.integers(0,5,size=(1000,5));ib=rng.integers(0,5,size=(1000,5))
        physical=rng.integers(0,16,size=(1000,5,16))
        results[left:left+1000]=(np.take_along_axis(a[ia],physical[:,:,:,None],axis=2).mean((1,2,3))-
                                 np.take_along_axis(b[ib],physical[:,:,:,None],axis=2).mean((1,2,3)))
    return np.quantile(results,[.025,.975]).tolist()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=('smoke','dev'),required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();started=time.monotonic();deadline=started+1200
    out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    source_manifest=json.loads((V14/'manifest.json').read_text())
    audit=json.loads((V14_AUDIT/'independent_audit.json').read_text())
    assert source_manifest['status']=='COMPLETED' and audit['status']=='PASS'
    assert Path(audit['run']).resolve()==V14
    selection=json.loads((V14/'selection.json').read_text())
    assert sha(V14/'selection.json')==source_manifest['selection_sha256']
    v17=V17_DEV if args.profile=='dev' else V17_SMOKE
    v17_audit=V17_DEV_AUDIT if args.profile=='dev' else V17_SMOKE_AUDIT
    v17_manifest=json.loads((v17/'manifest.json').read_text())
    assert v17_manifest['status']=='COMPLETED'
    v17_report=json.loads((v17_audit/'independent_audit.json').read_text())
    assert v17_report['status']=='PASS' and Path(v17_report['run']).resolve()==v17
    seeds=tuple(range(72600000,72600005)) if args.profile=='dev' else (72600000,)
    cases=CASES if args.profile=='dev' else (CASES[0],)
    physical=tuple(range(73011000,73011016)) if args.profile=='dev' else (73020000,73020001)
    models={};checkpoints={}
    for arm in ARMS:
        for train in seeds:
            item=selection[arm][str(train)];cp=V14/item['checkpoint']
            assert sha(cp)==item['sha256']
            checkpoints[f'{arm}/{train}']=item['sha256']
            models[(arm,train)]=load_checkpoint(cp)
    grid=[(arm,train,case,seed) for arm in ARMS for train in seeds for case in cases for seed in physical]
    rows=[]
    for arm,train,case,seed in tqdm(grid,desc='Frozen V14 on V17 final DEV',unit='episode'):
        if time.monotonic()>deadline:raise TimeoutError('V14 comparator deadline')
        row=evaluate_episode(configs()[case],case,seed,V14Controller(models[(arm,train)],arm))
        row.update(train_seed=train,checkpoint_sha256=checkpoints[f'{arm}/{train}'])
        rows.append(row)
    write(out/'episodes.csv',rows)
    summary=dict(status='COMPLETED',profile=args.profile,episodes=len(rows),
        v14_pooled={arm:float(np.mean([float(r['objective']) for r in rows if r['controller']==arm])) for arm in ARMS},
        v14_by_case={case:{arm:float(np.mean([float(r['objective']) for r in rows if r['case']==case and r['controller']==arm]))
                           for arm in ARMS} for case in cases},
        v14_by_train_seed={arm:{str(train):float(np.mean([float(r['objective']) for r in rows
                                                  if r['controller']==arm and int(r['train_seed'])==train]))
                                for train in seeds} for arm in ARMS},
        v14_selection_seconds={arm:float(np.mean([float(r['selection_seconds']) for r in rows if r['controller']==arm]))
                               for arm in ARMS})
    if args.profile=='dev':
        existing=read(v17/'episodes.csv')
        candidate=cost_tensor('rank_projection',range(73000000,73000005),existing,CASES,physical)
        summary['v17_rank_pooled']=float(candidate.mean())
        summary['v17_rank_minus_v14']={arm:dict(mean=float(candidate.mean()-cost_tensor(arm,seeds,rows,CASES,physical).mean()),
            ci95=bootstrap(candidate,cost_tensor(arm,seeds,rows,CASES,physical))) for arm in ARMS}
    else:summary['v17_rank_pooled']=None;summary['v17_rank_minus_v14']=None
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest=dict(status='COMPLETED',profile=args.profile,v14_run=str(V14),v17_run=str(v17),
        v14_manifest_sha256=sha(V14/'manifest.json'),v14_audit_sha256=sha(V14_AUDIT/'independent_audit.json'),
        v14_selection_sha256=sha(V14/'selection.json'),
        v17_manifest_sha256=sha(v17/'manifest.json'),v17_audit_sha256=sha(v17_audit/'independent_audit.json'),
        v17_episodes_sha256=sha(v17/'episodes.csv'),
        plan_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),checkpoint_hashes=checkpoints,
        wall_seconds=time.monotonic()-started,
        artifact_hashes={n:sha(out/n) for n in ('episodes.csv','summary.json')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
