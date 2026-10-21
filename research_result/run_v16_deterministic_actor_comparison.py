"""Compare held-out V16 head Q with audited deterministic V14 actor Q."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parent.parent
PLAN=ROOT/'research_result/v16_deterministic_actor_comparison_plan_vi.md'
SEEDS=tuple(range(72600000,72600005))
CASES=('reciprocal_scale2','interchangeable_scale1','reciprocal_scale3','specialist_generalist_scale2')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def key(r):return (int(r['train_seed']),int(r['iteration']),r['case'],int(r['physical_seed']),int(r['tick']),r['rank'])
def write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def interval(cube):
    rng=np.random.default_rng(72950004);values=np.empty(20000)
    for left in tqdm(range(0,20000,1000),desc='V16 deterministic comparison bootstrap',unit='chunk'):
        selected=rng.integers(0,5,size=(1000,5))
        ranks=rng.integers(0,64,size=(1000,5,6,4,64))
        values[left:left+1000]=np.take_along_axis(cube[selected],ranks,axis=-1).mean((1,2,3,4))
    return [float(x) for x in np.quantile(values,[.025,.975])]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v15-dir',type=Path,required=True);p.add_argument('--v16-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--profile',choices=('smoke','dev'),required=True)
    args=p.parse_args();start=time.monotonic();deadline=start+600
    v15=args.v15_dir.resolve();v16=args.v16_dir.resolve();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    for run in (v15,v16):
        manifest=json.loads((run/'manifest.json').read_text())
        assert manifest['status']=='COMPLETED' and manifest['profile']==args.profile
        for name,digest in manifest['artifact_hashes'].items():assert sha(run/name)==digest
    a=read(v15/'rows.csv');b=read(v16/'heldout_rows.csv')
    old={key(r):r for r in a}
    assert len(old)==len(a)==(7680 if args.profile=='dev' else 12)
    assert len(b)==(23040 if args.profile=='dev' else 4)
    rows=[]
    for new in tqdm(b,desc='Join deterministic V14 Q',unit='row'):
        if time.monotonic()>deadline:raise TimeoutError('deterministic comparison')
        prior=old[key(new)]
        assert json.loads(prior['legal'])==json.loads(new['legal'])
        q1={int(k):float(v) for k,v in json.loads(prior['q']).items()}
        q2={int(k):float(v) for k,v in json.loads(new['q']).items()}
        assert q1==q2
        act_z=int(json.loads(prior['z_actions'])[0]);act_b=int(json.loads(prior['b_actions'])[0])
        assert act_z==act_b and act_z in q1
        value=float(json.loads(prior['z_q'])[0])
        assert value==float(json.loads(prior['b_q'])[0])==q1[act_z]
        rows.append(dict(**{k:new[k] for k in ('train_seed','iteration','case','physical_seed','tick','rank','init_seed',
            'post_action','pre_action','post_q','pre_q','best_q','source_action','source_q')},
            actor_action=act_z,actor_q=value,post_minus_actor=float(new['post_q'])-value,
            pre_minus_actor=float(new['pre_q'])-value))
    write(out/'joined_rows.csv',rows)
    means={arm:float(np.mean([float(r[f'{arm}_minus_actor']) for r in rows])) for arm in ('post','pre')}
    summary=dict(status='COMPLETED',profile=args.profile,rows=len(rows),mean=means,
        by_case={case:{arm:float(np.mean([float(r[f'{arm}_minus_actor']) for r in rows if r['case']==case]))
                       for arm in ('post','pre')} for case in CASES if any(r['case']==case for r in rows)},
        by_seed={str(seed):{arm:float(np.mean([float(r[f'{arm}_minus_actor']) for r in rows
                                       if int(r['train_seed'])==seed])) for arm in ('post','pre')}
                 for seed in sorted({int(r['train_seed']) for r in rows})},
        by_start={str(seed):{arm:float(np.mean([float(r[f'{arm}_minus_actor']) for r in rows
                                        if int(r['init_seed'])==seed])) for arm in ('post','pre')}
                  for seed in sorted({int(r['init_seed']) for r in rows})},
        actor_q=float(np.mean([float(r['actor_q']) for r in rows])),
        actor_regret=float(np.mean([float(r['actor_q'])-float(r['best_q']) for r in rows])),
        changed={arm:sum(int(r[f'{arm}_action'])!=int(r['actor_action']) for r in rows)
                 for arm in ('post','pre')})
    if args.profile=='dev':
        cubes={arm:np.empty((5,6,4,64)) for arm in ('post','pre')}
        for si,seed in enumerate(SEEDS):
            for update in range(1,7):
                for ci,case in enumerate(CASES):
                    cell=[r for r in rows if int(r['train_seed'])==seed and int(r['iteration'])==update
                          and r['case']==case]
                    assert len(cell)==192
                    ranks=sorted({r['rank'] for r in cell});assert len(ranks)==64
                    for arm in cubes:
                        cubes[arm][si,update-1,ci]=[np.mean([float(r[f'{arm}_minus_actor']) for r in cell
                            if r['rank']==rank]) for rank in ranks]
        summary['ci95']={arm:interval(cube) for arm,cube in cubes.items()}
    else:summary['ci95']=None
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest=dict(status='COMPLETED',profile=args.profile,v15_dir=str(v15),v16_dir=str(v16),
        v15_manifest_sha256=sha(v15/'manifest.json'),v16_manifest_sha256=sha(v16/'manifest.json'),
        plan_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),wall_seconds=time.monotonic()-start,
        artifact_hashes={n:sha(out/n) for n in ('joined_rows.csv','summary.json')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
