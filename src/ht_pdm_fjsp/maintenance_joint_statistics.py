"""Paired crossed bootstrap over training replicas and physical profiles."""
import itertools
import numpy as np
from ht_pdm_fjsp.maintenance_dispatch import role_seed


def bootstrap_samples(a,b,replicas,profiles,relative=False):
    samples=[]
    for start in range(0,len(replicas),1000):
        r=replicas[start:start+1000,:,None];p=profiles[start:start+1000,None,:]
        left=a[r,p].mean((1,2));delta=(a-b)[r,p].mean((1,2))
        if relative and np.any(left<=0):raise RuntimeError("positive baseline cost required")
        samples.append(delta/left if relative else delta)
    return np.concatenate(samples)


def summarize(rows,cfg):
    conditions=cfg["conditions"];arms=cfg["arms"];seeds=cfg["train_seeds"]
    key=lambda r:(r["condition"],r["cohort"],r["shock"],r["controller"],r["train_seed"])
    index={key(r):r for r in rows}
    expected={(cond,c,s,name,seed) for cond,c,s in itertools.product(conditions,cfg["test_cohorts"],cfg["test_shocks"])
              for name,seed in [("best_rule",-1),*itertools.product(arms,seeds)]}
    if any(r["horizon"]!=cfg["horizon"] for r in rows):raise RuntimeError("test horizon mismatch")
    if len(index)!=len(rows) or set(index)!=expected:raise RuntimeError("incomplete/duplicate test grid")
    rng=np.random.default_rng(role_seed(cfg["analysis_seed"],"maintenance_joint_rl_v1:crossed_bootstrap"))
    replicas=rng.integers(len(seeds),size=(cfg["bootstrap_samples"],len(seeds)))
    profiles=rng.integers(len(cfg["test_cohorts"]),size=(cfg["bootstrap_samples"],len(cfg["test_cohorts"])))
    matrix=lambda cond,name,metric:np.array([[np.mean([index[cond,c,s,name,seed if name!='best_rule' else -1][metric]
            for s in cfg["test_shocks"]]) for c in cfg["test_cohorts"]] for seed in seeds])
    q=[(1-cfg["primary_confidence"])/2,(1+cfg["primary_confidence"])/2]
    primary,service,differences={},{},[]
    means={}
    metrics=("extended_cost_per_machine_tick","extended_objective","base_objective","tail_objective",
             "interior_cost_per_machine_tick","interior_pending_mean","interior_failed_mean","interior_overdue_per_tick",
             "cutoff_pending","cutoff_failed","post_tail_pending","post_tail_failed","nonfixed_assignments",
             "wait_with_choice","preventive_starts","corrective_starts","extended_censored_requests")
    for cond in conditions:
        means[cond]={name:{metric:float(np.mean([r[metric] for r in rows if r['condition']==cond and r['controller']==name]))
                    for metric in metrics} for name in ['best_rule',*arms]}
        for left in ('best_rule','fixed_ppo'):
            a,b=matrix(cond,left,'extended_cost_per_machine_tick'),matrix(cond,'joint_ppo','extended_cost_per_machine_tick')
            samples=bootstrap_samples(a,b,replicas,profiles,True)
            ci=np.quantile(samples,q).tolist()
            status='MATERIAL_GAIN' if ci[0]>cfg['cost_margin'] else 'BELOW_MARGIN' if ci[1]<cfg['cost_margin'] else 'INCONCLUSIVE'
            primary[f'{cond}:joint_vs_{left}']=dict(left=left,right='joint_ppo',metric='extended_cost_per_machine_tick',
                estimate=float((a-b).mean()/a.mean()),absolute_difference=float((a-b).mean()),left_mean=float(a.mean()),
                right_mean=float(b.mean()),confidence=cfg['primary_confidence'],confidence_interval=ci,
                margin=cfg['cost_margin'],training_replicas=len(seeds),profiles=len(cfg['test_cohorts']),
                status='ENGINEERING_ONLY' if cfg['profile']=='smoke' else status)
            for i,seed in enumerate(seeds):
                for j,cohort in enumerate(cfg['test_cohorts']):
                    differences.append(dict(condition=cond,contrast=f'joint_vs_{left}',train_seed=seed,cohort=cohort,
                        left_mean=float(a[i,j]),right_mean=float(b[i,j]),difference=float(a[i,j]-b[i,j])))
        for metric,margin in (('interior_pending_mean',.1),('interior_failed_mean',.1),('interior_overdue_per_tick',.25/12)):
            a,b=matrix(cond,'joint_ppo',metric),matrix(cond,'best_rule',metric)
            samples=bootstrap_samples(a,b,replicas,profiles)
            service[f'{cond}:{metric}']=dict(estimate=float((a-b).mean()),confidence=.95,
                confidence_interval=np.quantile(samples,[.025,.975]).tolist(),margin=margin,exploratory=True)
    return dict(means=means,primary_contrasts=primary,exploratory_service=service,
        uncertainty='paired crossed training-replica/profile bootstrap; fixed shock panel conditional',
        multiplicity='four primary two-sided98.75% percentile intervals, approximate Bonferroni coverage',
        interpretation='learning both decisions, assignment value and superiority over rules are separate questions; no RL impossibility or deployment claim'),differences
