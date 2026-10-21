"""Independent source-row, deterministic-action and CI audit for V16 comparison."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parent.parent
CASES=('reciprocal_scale2','interchangeable_scale1','reciprocal_scale3','specialist_generalist_scale2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def key(r):return tuple(r[k] for k in ('train_seed','iteration','case','physical_seed','tick','rank'))
def close(a,b):assert np.isclose(float(a),float(b),rtol=0,atol=1e-8),(a,b)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison-dir',type=Path,required=True)
    p.add_argument('--v15-audit-dir',type=Path,required=True)
    p.add_argument('--v16-audit-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();begin=time.monotonic();deadline=begin+600
    comp=args.comparison_dir.resolve();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest=json.loads((comp/'manifest.json').read_text())
    v15=Path(manifest['v15_dir']);v16=Path(manifest['v16_dir'])
    assert manifest['status']=='COMPLETED'
    assert manifest['v15_manifest_sha256']==sha(v15/'manifest.json')
    assert manifest['v16_manifest_sha256']==sha(v16/'manifest.json')
    assert manifest['plan_sha256']==sha(ROOT/'research_result/v16_deterministic_actor_comparison_plan_vi.md')
    assert manifest['script_sha256']==sha(ROOT/'research_result/run_v16_deterministic_actor_comparison.py')
    for path in (v15,v16):
        source=json.loads((path/'manifest.json').read_text())
        assert source['status']=='COMPLETED' and source['profile']==manifest['profile']
        for name,digest in source['artifact_hashes'].items():assert sha(path/name)==digest
    for name,digest in manifest['artifact_hashes'].items():assert sha(comp/name)==digest
    for audit,run in ((args.v15_audit_dir,v15),(args.v16_audit_dir,v16)):
        j=json.loads((audit/'independent_audit.json').read_text())
        assert j['status']=='PASS' and Path(j.get('probe',j.get('diagnostic',''))).resolve()==run
    old={key(r):r for r in read(v15/'rows.csv')}
    new={(key(r),r['init_seed']):r for r in read(v16/'heldout_rows.csv')}
    rows=read(comp/'joined_rows.csv')
    assert len(rows)==len(new)==(23040 if manifest['profile']=='dev' else 4)
    assert len({(key(r),r['init_seed']) for r in rows})==len(rows)
    for row in tqdm(rows,desc='Independent deterministic Q join',unit='row'):
        if time.monotonic()>deadline:raise TimeoutError('comparison audit')
        base=old[key(row)];head=new[(key(row),row['init_seed'])]
        logits={int(a):float(v) for a,v in json.loads(base['logits']).items()}
        deterministic=min(logits,key=lambda a:(-logits[a],a))
        assert deterministic==int(row['actor_action'])==json.loads(base['z_actions'])[0]
        q={int(a):float(v) for a,v in json.loads(base['q']).items()}
        assert q=={int(a):float(v) for a,v in json.loads(head['q']).items()}
        close(row['actor_q'],q[deterministic])
        for arm in ('post','pre'):
            assert int(row[f'{arm}_action'])==int(head[f'{arm}_action'])
            close(row[f'{arm}_q'],head[f'{arm}_q'])
            close(row[f'{arm}_minus_actor'],float(head[f'{arm}_q'])-q[deterministic])
        close(row['best_q'],min(q.values()))
    summary=json.loads((comp/'summary.json').read_text())
    assert summary['rows']==len(rows)
    for arm in ('post','pre'):
        close(summary['mean'][arm],np.mean([float(r[f'{arm}_minus_actor']) for r in rows]))
    if manifest['profile']=='dev':
        for arm in ('post','pre'):
            cube=np.empty((5,6,4,64))
            for si,seed in enumerate(range(72600000,72600005)):
                for update in range(1,7):
                    for ci,case in enumerate(CASES):
                        group=[r for r in rows if int(r['train_seed'])==seed and int(r['iteration'])==update
                               and r['case']==case]
                        assert len(group)==192
                        cube[si,update-1,ci]=[np.mean([float(r[f'{arm}_minus_actor']) for r in group
                            if r['rank']==rank]) for rank in sorted({r['rank'] for r in group})]
            rng=np.random.default_rng(72950004);boot=np.empty(20000)
            for left in tqdm(range(0,20000,1000),desc=f'Independent {arm} CI',unit='chunk'):
                seeds=rng.integers(0,5,size=(1000,5))
                ranks=rng.integers(0,64,size=(1000,5,6,4,64))
                boot[left:left+1000]=np.take_along_axis(cube[seeds],ranks,axis=-1).mean((1,2,3,4))
            for got,want in zip(summary['ci95'][arm],np.quantile(boot,[.025,.975])):close(got,want)
            close(summary['mean'][arm],cube.mean())
    else:assert summary['ci95'] is None
    report=dict(status='PASS',comparison=str(comp),profile=manifest['profile'],checked_rows=len(rows),
                mean=summary['mean'],wall_seconds=time.monotonic()-begin)
    (out/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
