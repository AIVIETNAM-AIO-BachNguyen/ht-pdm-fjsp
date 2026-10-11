"""Exposure-aware crossed bootstrap for conditional one-root diagnostics."""
import itertools
import numpy as np
from ht_pdm_fjsp.maintenance_dispatch import role_seed


def crossed_ratio(numerators, exposures, replica_draws, profile_draws):
    values = np.full(len(replica_draws), np.nan)
    for start in range(0, len(values), 1000):
        ids = (replica_draws[start:start+1000, :, None], profile_draws[start:start+1000, None, :])
        num = numerators[ids].sum((1, 2))
        den = exposures[ids].sum((1, 2))
        values[start:start+1000] = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den>0)
    return values


def summarize(rows, cfg):
    for r in rows:
        for value_key, exposure_key in [('pm_delta','pm_exposure'),('rule_delta','choice_exposure')]:
            if r[exposure_key] not in (0,1) or not np.isfinite(r[value_key]) or (not r[exposure_key] and r[value_key] != 0):
                raise ValueError('invalid conditional numerator/exposure')
    keys = ('condition', 'cohort', 'shock', 'model_seed', 'origin', 'tick')
    index = {tuple(r[k] for k in keys): r for r in rows}
    expected = set(itertools.product(cfg['conditions'], cfg['cohorts'], cfg['physical_shocks'],
        cfg['model_seeds'], cfg['origins'], cfg['landmarks']))
    if len(index) != len(rows) or set(index) != expected:
        raise RuntimeError('incomplete/duplicate state grid')
    n, p = len(cfg['model_seeds']), len(cfg['cohorts'])
    rng = np.random.default_rng(role_seed(cfg['analysis_seed'], 'maintenance_timing_diagnostic_v1:crossed_bootstrap'))
    rd = rng.integers(n, size=(cfg['bootstrap_samples'], n))
    pd = rng.integers(p, size=(cfg['bootstrap_samples'], p))
    contrasts, cells = {}, []
    for condition, contrast in itertools.product(cfg['conditions'], ['pm_deferral', 'rule_root']):
        num = np.zeros((n, p));den = np.zeros((n, p))
        value_key, exposure_key = ('pm_delta', 'pm_exposure') if contrast == 'pm_deferral' else ('rule_delta', 'choice_exposure')
        for i, seed in enumerate(cfg['model_seeds']):
            for j, cohort in enumerate(cfg['cohorts']):
                group = [index[condition, cohort, shock, seed, origin, tick] for shock, origin, tick in
                    itertools.product(cfg['physical_shocks'], cfg['origins'], cfg['landmarks'])]
                num[i, j] = sum(r[value_key] for r in group)
                den[i, j] = sum(r[exposure_key] for r in group)
                cells.append(dict(condition=condition, contrast=contrast, model_seed=seed, cohort=cohort,
                    numerator=float(num[i,j]), exposure=int(den[i,j])))
        draws = crossed_ratio(num, den, rd, pd)
        finite = draws[np.isfinite(draws)]
        empty = float(1-len(finite)/len(draws))
        estimate = float(num.sum()/den.sum()) if den.sum() else None
        interval = None if estimate is None or empty>.05 else np.quantile(finite, [.00625,.99375]).tolist()
        status = 'INSUFFICIENT' if interval is None else 'HIGHER_ROOT_COST' if interval[0]>0 else 'LOWER_ROOT_COST' if interval[1]<0 else 'UNRESOLVED'
        contrasts[f'{condition}:{contrast}'] = dict(estimate=estimate, confidence=.9875, confidence_interval=interval,
            exposure=int(den.sum()), total_states=sum(r['condition']==condition for r in rows), empty_bootstrap_fraction=empty,
            status='ENGINEERING_ONLY' if cfg['profile']=='smoke' else status,
            sign='positive favors omitting current PM starts / using the rule root',
            units='cost per machine-forecast-interval',
            interpretation='conditional root-action value under shared deadline continuation, not learned-policy Q')
        eligible = [r for r in rows if r['condition']==condition and r[exposure_key]]
        contrasts[f'{condition}:{contrast}']['state_delta_signs'] = {label:sum(test(r[value_key]) for r in eligible)
            for label,test in [('negative',lambda x:x<0),('zero',lambda x:x==0),('positive',lambda x:x>0)]}
    behavior = {}
    for condition, origin in itertools.product(cfg['conditions'], cfg['origins']):
        group = [r for r in rows if r['condition']==condition and r['origin']==origin]
        behavior[f'{condition}:{origin}'] = dict(states=len(group),choice_states=sum(r['choice_exposure'] for r in group),
            pm_states=sum(r['pm_exposure'] for r in group),root_agreement=sum(r['actions_agree'] for r in group),
            rl_wait_with_choice=sum(r['rl_kind']=='WAIT' and r['choice_exposure'] for r in group),
            rule_wait_with_choice=sum(r['rule_kind']=='WAIT' and r['choice_exposure'] for r in group),
            rl_pm_rule_wait=sum(r['pm_exposure'] and r['rule_kind']=='WAIT' for r in group))
    return dict(primary_contrasts=contrasts,behavior=behavior,
        uncertainty='paired crossed model-seed/profile bootstrap; conditional on fixed shocks, origins, ticks and MC tapes',
        multiplicity='four exploratory contrasts, two-sided98.75% percentile intervals; approximate coverage',
        reference='sample minimum root value is noisy/optimistic and is not an oracle'), cells
