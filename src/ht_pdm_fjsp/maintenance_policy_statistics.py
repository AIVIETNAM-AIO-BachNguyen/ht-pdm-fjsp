"""Conditional complete-return contrasts with crossed replica/profile uncertainty."""
import itertools
import numpy as np
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_timing_statistics import crossed_ratio


def summarize(rows,cfg):
    keys=('condition','cohort','shock','model_seed','origin','tick')
    index={tuple(r[k] for k in keys):r for r in rows}
    grid=set(itertools.product(cfg['conditions'],cfg['cohorts'],cfg['physical_shocks'],cfg['model_seeds'],cfg['origins'],cfg['landmarks']))
    if len(index)!=len(rows) or set(index)!=grid:raise RuntimeError('incomplete/duplicate state grid')
    for row in rows:
        for contrast in ('pm','now'):
            value,exposure=row[f'{contrast}_delta'],row[f'{contrast}_exposure']
            if exposure not in (0,1) or not np.isfinite(value) or (not exposure and value!=0):
                raise ValueError('invalid conditional numerator/exposure')
    n,p=len(cfg['model_seeds']),len(cfg['cohorts'])
    rng=np.random.default_rng(role_seed(cfg['analysis_seed'],'maintenance_policy_continuation_v1:crossed_bootstrap'))
    rd=rng.integers(n,size=(cfg['bootstrap_samples'],n));pd=rng.integers(p,size=(cfg['bootstrap_samples'],p))
    contrasts={};cells=[]
    for condition,contrast in itertools.product(cfg['conditions'],('pm','now')):
        num=np.zeros((n,p));den=np.zeros((n,p))
        for i,seed in enumerate(cfg['model_seeds']):
            for j,cohort in enumerate(cfg['cohorts']):
                group=[index[condition,cohort,shock,seed,origin,tick] for shock,origin,tick in itertools.product(cfg['physical_shocks'],cfg['origins'],cfg['landmarks'])]
                num[i,j]=sum(r[f'{contrast}_delta'] for r in group);den[i,j]=sum(r[f'{contrast}_exposure'] for r in group)
                cells.append(dict(condition=condition,contrast=contrast,model_seed=seed,cohort=cohort,numerator=float(num[i,j]),exposure=int(den[i,j])))
        draws=crossed_ratio(num,den,rd,pd);finite=draws[np.isfinite(draws)]
        empty=float(1-len(finite)/len(draws));mean=float(num.sum()/den.sum()) if den.sum() else None
        ci=np.quantile(finite,[.00625,.99375]).tolist() if mean is not None and empty<=.05 else None
        verdict='INSUFFICIENT' if ci is None else 'HIGHER_ROOT_COST' if ci[0]>0 else 'LOWER_ROOT_COST' if ci[1]<0 else 'UNRESOLVED'
        group=[r for r in rows if r['condition']==condition and r[f'{contrast}_exposure']]
        contrasts[f'{condition}:{contrast}']=dict(estimate=mean,confidence=.9875,confidence_interval=ci,
            exposure=int(den.sum()),empty_bootstrap_fraction=empty,status='ENGINEERING_ONLY' if cfg['profile']=='smoke' else verdict,
            units='total simulator cost from sampled tick through59',
            sign='positive favors removing current PM starts' if contrast=='pm' else 'positive favors WAIT instead of all current starts',
            continuation='frozen deterministic RL through47; deadline tail48..59',
            state_mean_signs={label:sum(test(r[f'{contrast}_delta']) for r in group) for label,test in
                [('negative',lambda v:v<0),('zero',lambda v:v==0),('positive',lambda v:v>0)]})
    secondary=[]
    for condition,origin,tick in itertools.product(cfg['conditions'],cfg['origins'],cfg['landmarks']):
        group=[r for r in rows if (r['condition'],r['origin'],r['tick'])==(condition,origin,tick)]
        row=dict(condition=condition,origin=origin,tick=tick,states=len(group),remaining_physical_ticks=60-tick)
        for contrast in ('pm','now'):
            eligible=[r for r in group if r[f'{contrast}_exposure']]
            row[f'{contrast}_exposure']=len(eligible)
            for arm in ('rl','deadline','interaction'):
                key=f'{contrast}_delta' if arm=='rl' else f'{contrast}_{arm}_delta'
                row[f'{contrast}_{arm}_mean']=float(np.mean([r[key] for r in eligible])) if eligible else None
        secondary.append(row)
    behavior=[]
    for condition,busy in itertools.product(cfg['conditions'],range(3)):
        group=[r for r in rows if r['condition']==condition and r['busy_workers']==busy]
        if not group:continue
        item=dict(condition=condition,busy_workers=busy,states=len(group),
            rl_wait_states=sum(r['rl_wait'] for r in group),reference_wait_states=sum(r['reference_wait'] for r in group))
        for risk in ('at_risk','young'):
            item[f'{risk}_opportunities']=sum(r[f'{risk}_opportunities'] for r in group)
            item[f'{risk}_pm_starts']=sum(r[f'{risk}_pm_starts'] for r in group)
        choices=[r for r in group if r['structural_choice']]
        item['structural_choices']=len(choices)
        for arm in ('rl','deadline'):
            item[f'{arm}_reference_mean']=float(np.mean([r[f'{arm}_reference_delta'] for r in choices])) if choices else None
        for name in ('pm','now'):
            eligible=[r for r in group if r[f'{name}_exposure']]
            item[f'{name}_exposure']=len(eligible)
            item[f'{name}_mean']=float(np.mean([r[f'{name}_delta'] for r in eligible])) if eligible else None
        behavior.append(item)
    return dict(primary_contrasts=contrasts,secondary_strata=secondary,behavior_strata=behavior,
        uncertainty='crossed model/profile bootstrap conditional on fixed forecast tapes; repeated shocks/origins/ticks paired',
        multiplicity='four primary exploratory contrasts, approximate98.75% percentile coverage',
        input_scope='historical states intentionally reused; no independent physical generalization test'),cells,secondary
