"""Independent V14 nested-bootstrap sensitivity audit from immutable episodes."""
from __future__ import annotations

import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

CASES=('reciprocal_scale2','interchangeable_scale1','reciprocal_scale3','specialist_generalist_scale2')
SEEDS=tuple(range(72600000,72600005))
PHYSICAL=tuple(range(72610000,72610016))
ROOT=Path(__file__).resolve().parent.parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b):assert np.isclose(float(a),float(b),atol=1e-10,rtol=0),(a,b)

def audit(source,diagnostic):
    manifest=json.loads((diagnostic/'manifest.json').read_text())
    assert manifest['status']=='COMPLETED' and Path(manifest['source_run']).resolve()==source
    assert sha(source/'manifest.json')==manifest['source_manifest_sha256']
    assert sha(source/'episodes.csv')==manifest['source_episodes_sha256']
    assert sha(source/'summary.json')==manifest['source_summary_sha256']
    assert sha(ROOT/'research_result/v14_bootstrap_sensitivity_prelock_vi.md')==manifest['protocol_sha256']
    assert sha(ROOT/'research_result/run_v14_bootstrap_sensitivity.py')==manifest['script_sha256']
    assert sha(diagnostic/'summary.json')==manifest['summary_sha256']
    main=Path(manifest['main_audit'])
    assert sha(main/'independent_audit.json')==manifest['main_audit_sha256']
    report=json.loads((main/'independent_audit.json').read_text())
    assert report['status']=='PASS' and Path(report['run']).resolve()==source
    with (source/'episodes.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    left={(int(r['train_seed']),int(r['seed']),r['case']):float(r['objective']) for r in rows
          if r['controller']=='competition'}
    right={(int(r['train_seed']),int(r['seed']),r['case']):float(r['objective']) for r in rows
           if r['controller']=='count_calibrated'}
    expected={(s,p,c) for s in SEEDS for p in PHYSICAL for c in CASES}
    assert set(left)==set(right)==expected
    matrix=np.array([[[left[s,p,c]-right[s,p,c] for c in CASES] for p in PHYSICAL]
                     for s in SEEDS],dtype=float)
    summary=json.loads((diagnostic/'summary.json').read_text())
    assert summary['status']=='COMPLETED' and summary['draws']==20000 and summary['seed']==72640001
    close(summary['mean'],matrix.mean())
    original=json.loads((source/'summary.json').read_text())
    for x,y in zip(summary['original_crossed_ci95'],original['ablation']['hierarchical_ci95']):close(x,y)
    rng=np.random.default_rng(72640001)
    boot=np.empty((20000,4))
    for k in range(20):
        idx=rng.integers(0,5,size=(1000,5))
        physical=rng.integers(0,16,size=(1000,5,16))
        boot[k*1000:(k+1)*1000]=matrix[idx[:, :, None],physical].mean(axis=(1,2))
    ci=np.quantile(boot.mean(axis=1),[.025,.975])
    for x,y in zip(summary['nested_ci95'],ci):close(x,y)
    assert summary['nested_upper_below_zero']==bool(ci[1]<0)
    for i,case in enumerate(CASES):
        close(summary['by_case'][case]['mean'],matrix[:,:,i].mean())
        for x,y in zip(summary['by_case'][case]['ci95'],np.quantile(boot[:,i],[.025,.975])):close(x,y)
    for i,seed in enumerate(SEEDS):close(summary['by_train_seed'][str(seed)],matrix[i].mean())
    return dict(status='PASS',source_run=str(source),diagnostic=str(diagnostic),
                rows=len(expected)*2,draws=20000,checks=['hashes','complete paired grid','nested and crossed uncertainty'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--diagnostic-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    result=audit(args.run_dir.resolve(),args.diagnostic_dir.resolve())
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
