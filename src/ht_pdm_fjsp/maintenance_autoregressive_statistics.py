"""Predeclared paired pilot contrasts; uncertainty limited by three replicas."""
import itertools
import numpy as np
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_joint_statistics import bootstrap_samples
from ht_pdm_fjsp.maintenance_autoregressive_policy import CATALOGUE_ARM,CONTEXT,FROZEN


def summarize(rows,cfg):
    key=lambda r:(r["condition"],r["cohort"],r["shock"],r["controller"],r["train_seed"])
    index={key(r):r for r in rows};seeds=cfg["train_seeds"]
    expected={(cond,c,s,name,seed) for cond,c,s in itertools.product(cfg["conditions"],cfg["test_cohorts"],cfg["test_shocks"])
        for name,seed in [("best_rule",-1),*itertools.product(cfg["arms"],seeds)]}
    if len(index)!=len(rows) or set(index)!=expected or any(r["horizon"]!=cfg["horizon"] for r in rows):
        raise RuntimeError("incomplete/duplicate test grid or horizon")
    rng=np.random.default_rng(role_seed(cfg["analysis_seed"],"maintenance_autoregressive_pilot_v1:crossed_bootstrap"))
    replicas=rng.integers(len(seeds),size=(cfg["bootstrap_samples"],len(seeds)))
    profiles=rng.integers(len(cfg["test_cohorts"]),size=(cfg["bootstrap_samples"],len(cfg["test_cohorts"])))
    def matrix(cond,name):
        return np.array([[np.mean([index[cond,c,s,name,seed if name!="best_rule" else -1]["extended_objective"]
            for s in cfg["test_shocks"]]) for c in cfg["test_cohorts"]] for seed in seeds])
    primary={};reference={};differences=[];means={}
    metrics=("extended_objective","base_objective","tail_objective","extended_cost_per_machine_tick",
        "extended_failure_cost","extended_waiting_cost","extended_maintenance_cost","extended_unavailability_cost",
        "preventive_starts","corrective_starts","wait_with_choice","pm_feasible_ticks","wait_with_pm_feasible",
        "pm_two_worker_opportunity_ticks","stop_with_pm_extension_feasible","pm_assignment_alternative_ticks",
        "pm_starts_with_two_workers","cm_only_assignment_alternative_ticks")
    for cond in cfg["conditions"]:
        means[cond]={name:{metric:float(np.mean([r[metric] for r in rows if r['condition']==cond and r['controller']==name]))
            for metric in metrics} for name in ["best_rule",*cfg["arms"]]}
        for baseline in (CATALOGUE_ARM,FROZEN,"best_rule"):
            a,b=matrix(cond,baseline),matrix(cond,CONTEXT)
            confidence=.95 if baseline=="best_rule" else cfg["primary_confidence"]
            samples=bootstrap_samples(a,b,replicas,profiles,relative=True)
            ci=np.quantile(samples,[(1-confidence)/2,(1+confidence)/2]).tolist()
            gain=float((a-b).mean()/a.mean())
            status="PILOT_GAIN_SIGNAL" if ci[0]>cfg["cost_margin"] else "BELOW_MARGIN" if ci[1]<cfg["cost_margin"] else "INCONCLUSIVE"
            result=dict(baseline=baseline,method=CONTEXT,metric="extended_objective",relative_gain=gain,
                absolute_gain=float((a-b).mean()),baseline_mean=float(a.mean()),method_mean=float(b.mean()),
                confidence=confidence,confidence_interval=ci,material_margin=cfg["cost_margin"],
                replica_gains={str(seed):float((a[i]-b[i]).mean()/a[i].mean()) for i,seed in enumerate(seeds)},
                leave_one_replica_out={str(seed):float((np.delete(a,i,0)-np.delete(b,i,0)).mean()/np.delete(a,i,0).mean())
                    for i,seed in enumerate(seeds)} if len(seeds)>1 else {},
                training_replicas=len(seeds),profiles=len(cfg["test_cohorts"]),exploratory=baseline=="best_rule",
                status="ENGINEERING_ONLY" if cfg["profile"]=="smoke" else status)
            (reference if baseline=="best_rule" else primary)[f"{cond}:context_vs_{baseline}"]=result
            for i,seed in enumerate(seeds):
                for j,cohort in enumerate(cfg["test_cohorts"]):
                    differences.append(dict(condition=cond,contrast=f"context_vs_{baseline}",train_seed=seed,cohort=cohort,
                        baseline_mean=float(a[i,j]),method_mean=float(b[i,j]),gain=float(a[i,j]-b[i,j])))
    return dict(means=means,primary_contrasts=primary,exploratory_rule_contrasts=reference,
        all_four_primary_signal=all(r["status"]=="PILOT_GAIN_SIGNAL" for r in primary.values()),
        uncertainty="paired crossed training-replica/profile bootstrap; fixed shock panel; only three replicas in pilot",
        multiplicity="four two-sided98.75% primary percentile intervals; approximate Bonferroni",
        interpretation="pilot; explicit context effect isolated only by frozen-context ablation; behavior does not establish optimal timing or technician; no novelty or scale-transfer claim"),differences
