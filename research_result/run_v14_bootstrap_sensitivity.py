"""Prelocked crossed-versus-nested V14 DEV uncertainty sensitivity."""
from __future__ import annotations

import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parent.parent
PROTOCOL=ROOT/'research_result/v14_bootstrap_sensitivity_prelock_vi.md'
CASES=('reciprocal_scale2','interchangeable_scale1','reciprocal_scale3','specialist_generalist_scale2')
SEEDS=tuple(range(72600000,72600005))
PHYSICAL=tuple(range(72610000,72610016))
DRAW_SEED=72640001
DRAWS=20000

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cube_from_rows(path):
    with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
    lookup={(r['case'],r['controller'],r['train_seed'],int(r['seed'])):float(r['objective'])
            for r in rows if r['controller'] in ('competition','count_calibrated')}
    expected={(c,a,str(s),p) for c in CASES for a in ('competition','count_calibrated')
              for s in SEEDS for p in PHYSICAL}
    assert set(lookup)==expected and len(lookup)==len(expected)
    cube=np.asarray([[[lookup[(c,'competition',str(s),p)]-
                       lookup[(c,'count_calibrated',str(s),p)] for c in CASES]
                       for p in PHYSICAL] for s in SEEDS],dtype=float)
    assert cube.shape==(5,16,4)
    return cube

def nested(cube):
    rng=np.random.default_rng(DRAW_SEED)
    values=np.empty((DRAWS,4))
    for left in tqdm(range(0,DRAWS,1000),desc='V14 nested bootstrap',unit='chunk',leave=False):
        n=min(1000,DRAWS-left)
        train=rng.integers(0,5,size=(n,5))
        physical=rng.integers(0,16,size=(n,5,16))
        sampled=np.take_along_axis(cube[train],physical[:,:,:,None],axis=2)
        values[left:left+n]=sampled.mean(axis=(1,2))
    return values

def run(source,audit_dir,out):
    started=time.monotonic();deadline=started+300
    manifest=json.loads((source/'manifest.json').read_text())
    audit=json.loads((audit_dir/'independent_audit.json').read_text())
    assert manifest['status']=='COMPLETED' and manifest['profile']=='dev'
    assert audit['status']=='PASS' and Path(audit['run']).resolve()==source
    assert manifest['registry']['bootstrap_seed']==72640000
    for name,digest in manifest['artifact_hashes'].items():assert sha(source/name)==digest
    cube=cube_from_rows(source/'episodes.csv')
    raw=json.loads((source/'summary.json').read_text())
    assert abs(float(cube.mean())-raw['ablation']['mean'])<1e-10
    values=nested(cube)
    assert time.monotonic()<deadline
    main=np.quantile(values.mean(axis=1),[.025,.975])
    case_ci=np.quantile(values,[.025,.975],axis=0)
    summary=dict(status='COMPLETED',mean=float(cube.mean()),
        original_crossed_ci95=raw['ablation']['hierarchical_ci95'],
        nested_ci95=main.tolist(),nested_upper_below_zero=bool(main[1]<0),
        by_case={case:dict(mean=float(cube[:,:,i].mean()),ci95=case_ci[:,i].tolist())
                 for i,case in enumerate(CASES)},
        by_train_seed={str(seed):float(cube[i].mean()) for i,seed in enumerate(SEEDS)},
        draws=DRAWS,seed=DRAW_SEED)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=dict(status='COMPLETED',source_run=str(source),main_audit=str(audit_dir),
        source_manifest_sha256=sha(source/'manifest.json'),
        source_episodes_sha256=sha(source/'episodes.csv'),
        source_summary_sha256=sha(source/'summary.json'),
        main_audit_sha256=sha(audit_dir/'independent_audit.json'),
        protocol_sha256=sha(PROTOCOL),script_sha256=sha(Path(__file__)),
        summary_sha256=sha(out/'summary.json'),wall_seconds=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--main-audit-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    run(args.run_dir.resolve(),args.main_audit_dir.resolve(),out)

if __name__=='__main__':main()
