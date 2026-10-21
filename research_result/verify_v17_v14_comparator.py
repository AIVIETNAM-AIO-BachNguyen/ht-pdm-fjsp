"""Independent all-episode replay and bootstrap audit of frozen V14 on V17 DEV."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from verify_framework_v14 import actor_at,episode_replay

ROOT=Path(__file__).resolve().parent.parent
CASES=('reciprocal_scale2','interchangeable_scale1','reciprocal_scale3','specialist_generalist_scale2')
ARMS=('competition','count_calibrated')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert np.isclose(float(a),float(b),rtol=0,atol=1e-8),(a,b)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();begin=time.monotonic();deadline=begin+1200
    run=args.run_dir.resolve();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    m=json.loads((run/'manifest.json').read_text());profile=m['profile']
    assert m['status']=='COMPLETED' and profile in ('smoke','dev')
    v14=Path(m['v14_run']);v17=Path(m['v17_run'])
    v14m=json.loads((v14/'manifest.json').read_text());v17m=json.loads((v17/'manifest.json').read_text())
    assert v14m['status']==v17m['status']=='COMPLETED'
    assert sha(v14/'manifest.json')==m['v14_manifest_sha256']
    assert sha(v14/'selection.json')==m['v14_selection_sha256']
    assert sha(v17/'manifest.json')==m['v17_manifest_sha256']
    assert sha(v17/'episodes.csv')==m['v17_episodes_sha256']
    assert sha(ROOT/'research_result/v17_v14_comparator_plan_vi.md')==m['plan_sha256']
    assert sha(ROOT/'research_result/run_v17_v14_comparator.py')==m['script_sha256']
    for name,digest in m['artifact_hashes'].items():assert sha(run/name)==digest
    for name,digest in v14m['artifact_hashes'].items():assert sha(v14/name)==digest
    for name,digest in v17m['artifact_hashes'].items():assert sha(v17/name)==digest
    v14a=ROOT/'research_result/framework_v14_audit_20261020T200552Z/independent_audit.json'
    v17a=(ROOT/'research_result/framework_v17_audit_20261021T031626Z/independent_audit.json'
          if profile=='dev' else ROOT/'research_result/framework_v17_audit_20261021T022759Z/independent_audit.json')
    assert sha(v14a)==m['v14_audit_sha256'] and sha(v17a)==m['v17_audit_sha256']
    assert json.loads(v14a.read_text())['status']==json.loads(v17a.read_text())['status']=='PASS'
    selection=json.loads((v14/'selection.json').read_text())
    seeds=tuple(range(72600000,72600005)) if profile=='dev' else (72600000,)
    cases=CASES if profile=='dev' else (CASES[0],)
    physical=tuple(range(73011000,73011016)) if profile=='dev' else (73020000,73020001)
    checkpoints={};models={}
    for arm in ARMS:
        for train in seeds:
            item=selection[arm][str(train)];cp=v14/item['checkpoint']
            assert sha(cp)==item['sha256']==m['checkpoint_hashes'][f'{arm}/{train}']
            models[(arm,train)]=actor_at(cp)
    rows=read(run/'episodes.csv')
    assert len(rows)==len(ARMS)*len(seeds)*len(cases)*len(physical)
    expected={(arm,str(train),case,str(seed)) for arm in ARMS for train in seeds
              for case in cases for seed in physical}
    assert {(r['controller'],r['train_seed'],r['case'],r['seed']) for r in rows}==expected
    for r in tqdm(rows,desc='Independent V14 comparator episode replay',unit='episode'):
        if time.monotonic()>deadline:raise TimeoutError('V14 comparator audit')
        arm=r['controller'];train=int(r['train_seed'])
        assert r['checkpoint_sha256']==m['checkpoint_hashes'][f'{arm}/{train}']
        episode_replay(r,models[(arm,train)])
        assert np.isfinite(float(r['selection_seconds'])) and float(r['selection_seconds'])>=0
    summary=json.loads((run/'summary.json').read_text())
    assert summary['status']=='COMPLETED' and summary['profile']==profile and summary['episodes']==len(rows)
    for arm in ARMS:
        close(summary['v14_pooled'][arm],np.mean([float(r['objective']) for r in rows if r['controller']==arm]))
        for case in cases:
            close(summary['v14_by_case'][case][arm],np.mean([float(r['objective']) for r in rows
                if r['controller']==arm and r['case']==case]))
    if profile=='dev':
        existing=read(v17/'episodes.csv')
        v17_lookup={(r['controller'],r['train_seed'],r['case'],r['seed']):float(r['objective']) for r in existing}
        old_lookup={(r['controller'],r['train_seed'],r['case'],r['seed']):float(r['objective']) for r in rows}
        a=np.asarray([[[v17_lookup[('rank_projection',str(train),case,str(physical_seed))] for case in CASES]
                       for physical_seed in physical] for train in range(73000000,73000005)])
        close(summary['v17_rank_pooled'],a.mean())
        for arm in ARMS:
            b=np.asarray([[[old_lookup[(arm,str(train),case,str(physical_seed))] for case in CASES]
                           for physical_seed in physical] for train in seeds])
            close(summary['v17_rank_minus_v14'][arm]['mean'],a.mean()-b.mean())
            rng=np.random.default_rng(73040005);values=np.empty(20000)
            for left in tqdm(range(0,20000,1000),desc=f'Independent V14 {arm} CI',unit='chunk'):
                ia=rng.integers(0,5,size=(1000,5));ib=rng.integers(0,5,size=(1000,5))
                ix=rng.integers(0,16,size=(1000,5,16))
                values[left:left+1000]=(np.take_along_axis(a[ia],ix[:,:,:,None],axis=2).mean((1,2,3))-
                                        np.take_along_axis(b[ib],ix[:,:,:,None],axis=2).mean((1,2,3)))
            for got,want in zip(summary['v17_rank_minus_v14'][arm]['ci95'],np.quantile(values,[.025,.975])):
                close(got,want)
    else:assert summary['v17_rank_minus_v14'] is None
    report=dict(status='PASS',run=str(run),profile=profile,episodes_replayed=len(rows),
                checkpoint_hashes=len(checkpoints) if checkpoints else len(models),
                wall_seconds=time.monotonic()-begin)
    (out/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
