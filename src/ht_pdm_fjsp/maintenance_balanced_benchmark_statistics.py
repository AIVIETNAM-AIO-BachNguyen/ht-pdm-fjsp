"""Crossed training-replica and physical-episode bootstrap, paired methods."""
import numpy as np
from ht_pdm_fjsp.maintenance_balanced_algorithms import ALGORITHMS, role_seed

BASELINES=('deadline_matching','isolated_h6','isolated_endpoint','assignment_rollout','joint_rollout')
METRICS=('extended_cost','availability','terminal_pending')


def arrays(rows,train_seeds,test_seeds):
    lookup={(r['method'],r['train_seed'],r['seed']):r for r in rows}
    expected={(a,t,s) for a in ALGORITHMS for t in train_seeds for s in test_seeds}
    expected|={(b,-1,s) for b in BASELINES for s in test_seeds}
    if len(rows)!=len(expected) or set(lookup)!=expected:
        raise RuntimeError('exact test replica/episode grid required')
    learned=np.array([[[[lookup[a,t,s][m] for m in METRICS] for s in test_seeds]
        for t in train_seeds] for a in ALGORITHMS],dtype=float)
    baseline=np.array([[[lookup[b,-1,s][m] for m in METRICS] for s in test_seeds]
        for b in BASELINES],dtype=float)
    if not np.isfinite(learned).all() or not np.isfinite(baseline).all() or (learned[...,0]<=0).any() or (baseline[...,0]<=0).any():
        raise RuntimeError('positive finite cost population required')
    return learned,baseline


def estimates(learned,baseline):
    result={}
    for i,a in enumerate(ALGORITHMS):
        for j,b in enumerate(BASELINES):
            result[f'{a}_cost_vs_{b}']=1-learned[...,i,0]/baseline[...,j,0]
        h6=BASELINES.index('isolated_h6')
        result[f'{a}_availability_difference']=learned[...,i,1]-baseline[...,h6,1]
        result[f'{a}_pending_difference']=learned[...,i,2]-baseline[...,h6,2]
    return result


def summarize(rows,cfg):
    learned,baseline=arrays(rows,cfg['train_seeds'],cfg['test_seeds'])
    point=estimates(learned.mean((1,2)),baseline.mean(1))
    ntrain,ntest=learned.shape[1:3]
    rng=np.random.default_rng(role_seed(cfg['analysis_seed'],'crossed_bootstrap'))
    draws={key:[] for key in point}
    # Batch counts avoid an enormous bootstrap×seed×feature tensor.
    for start in range(0,cfg['bootstrap_samples'],256):
        count=min(256,cfg['bootstrap_samples']-start)
        ti=rng.integers(ntrain,size=(count,ntrain)); si=rng.integers(ntest,size=(count,ntest))
        tc=np.eye(ntrain)[ti].mean(1); sc=np.eye(ntest)[si].mean(1)
        lm=np.einsum('bt,bs,atsk->bak',tc,sc,learned,optimize=True)
        bm=np.einsum('bs,ask->bak',sc,baseline,optimize=True)
        for key,value in estimates(lm,bm).items(): draws[key].extend(value.tolist())
    tail=(1-cfg['confidence'])/2
    intervals={key:dict(estimate=float(value),lower=float(np.quantile(draws[key],tail)),
        upper=float(np.quantile(draws[key],1-tail))) for key,value in point.items()}
    guards={a:dict(health_pass=intervals[f'{a}_availability_difference']['lower']>=-.01
        and intervals[f'{a}_pending_difference']['upper']<=.1,
        statistically_better_than=[b for b in BASELINES if intervals[f'{a}_cost_vs_{b}']['lower']>0],
        practically_better_than=[b for b in BASELINES if intervals[f'{a}_cost_vs_{b}']['lower']>=.01]) for a in ALGORITHMS}
    return dict(intervals=intervals,guards=guards,family_size=len(intervals),confidence=cfg['confidence'],
        bootstrap_samples=cfg['bootstrap_samples'],bootstrap_unit='crossed training replica and paired physical episode',
        interpretation='fixed simulator; differing observation/action architectures; five full training replicas; approximate CIs',
        means={**{a:dict(zip(METRICS,learned[i].mean((0,1)).tolist())) for i,a in enumerate(ALGORITHMS)},
            **{b:dict(zip(METRICS,baseline[i].mean(0).tolist())) for i,b in enumerate(BASELINES)}})
