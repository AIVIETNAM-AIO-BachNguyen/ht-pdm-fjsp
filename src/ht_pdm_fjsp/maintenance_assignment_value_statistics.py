"""Four primary episode contrasts and separately clustered exploratory diagnostics."""
import itertools
import numpy as np
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_joint_statistics import bootstrap_samples
from ht_pdm_fjsp.maintenance_timing_statistics import crossed_ratio
from ht_pdm_fjsp.maintenance_assignment_value_benchmark import VERSION
NAMES=("frozen_ppo","true_value","shuffled_value")


def draws(cfg,n,p):
    rng=np.random.default_rng(role_seed(cfg["analysis_seed"],f"{VERSION}:crossed_bootstrap"))
    return rng.integers(n,size=(cfg["bootstrap_samples"],n)),rng.integers(p,size=(cfg["bootstrap_samples"],p))


def summarize(rows,cfg):
    index={(r["condition"],r["cohort"],r["shock"],r["controller"],r["train_seed"]):r for r in rows}
    expected=set(itertools.product(cfg["conditions"],cfg["test_cohorts"],cfg["test_shocks"],NAMES,cfg["model_seeds"]))
    if len(index)!=len(rows) or set(index)!=expected or any(r["horizon"]!=48 for r in rows):raise RuntimeError("test grid mismatch")
    rd,pd=draws(cfg,len(cfg["model_seeds"]),len(cfg["test_cohorts"]));primary={};cells=[];means={}
    metrics=("extended_objective","base_objective","tail_objective","extended_failure_cost","extended_waiting_cost","extended_maintenance_cost","extended_unavailability_cost","preventive_starts","corrective_starts","wait_with_pm_feasible","stop_with_pm_extension_feasible","pm_assignment_alternative_ticks")
    for cond in cfg["conditions"]:
        means[cond]={name:{m:float(np.mean([r[m] for r in rows if r["condition"]==cond and r["controller"]==name])) for m in metrics} for name in NAMES}
        matrix=lambda name:np.array([[np.mean([index[cond,c,s,name,seed]["extended_objective"] for s in cfg["test_shocks"]]) for c in cfg["test_cohorts"]] for seed in cfg["model_seeds"]])
        for left in ("frozen_ppo","shuffled_value"):
            a,b=matrix(left),matrix("true_value");confidence=cfg["primary_confidence"]
            samples=bootstrap_samples(a,b,rd,pd,True);ci=np.quantile(samples,[(1-confidence)/2,(1+confidence)/2]).tolist()
            key=f"{cond}:true_value_vs_{left}"
            primary[key]=dict(baseline=left,method="true_value",baseline_mean=float(a.mean()),method_mean=float(b.mean()),
                absolute_gain=float((a-b).mean()),relative_gain=float((a-b).mean()/a.mean()),confidence=confidence,confidence_interval=ci,material_margin=cfg["cost_margin"],
                replica_gains={str(seed):float((a[i]-b[i]).mean()/a[i].mean()) for i,seed in enumerate(cfg["model_seeds"])},
                leave_one_replica_out={str(seed):float((np.delete(a,i,0)-np.delete(b,i,0)).mean()/np.delete(a,i,0).mean()) for i,seed in enumerate(cfg["model_seeds"])} if len(a)>1 else {},
                status="ENGINEERING_ONLY" if cfg["profile"]=="smoke" else "PILOT_GAIN_SIGNAL" if ci[0]>cfg["cost_margin"] else "BELOW_MARGIN" if ci[1]<cfg["cost_margin"] else "UNRESOLVED")
            for i,seed in enumerate(cfg["model_seeds"]):
                for j,c in enumerate(cfg["test_cohorts"]):cells.append(dict(condition=cond,contrast=key,model_seed=seed,cohort=c,baseline_mean=float(a[i,j]),method_mean=float(b[i,j]),gain=float(a[i,j]-b[i,j])))
    return dict(means=means,primary_contrasts=primary,multiplicity="four primary98.75% crossed paired replica/profile intervals; approximate Bonferroni",
        limitations="one policy improvement, three frozen source replicas; fixed shocks; Delta Q^pi labels need not equal improved-policy value; no end-to-end timing/assignment claim"),cells


def conditional(rows,cfg,condition,metric,profiles,controller=None,exposure_key="exposure"):
    n,p=len(cfg["model_seeds"]),len(profiles);num=np.zeros((n,p));den=np.zeros((n,p));cells=[]
    for i,seed in enumerate(cfg["model_seeds"]):
        for j,cohort in enumerate(profiles):
            group=[r for r in rows if r["condition"]==condition and r["model_seed"]==seed and r["cohort"]==cohort and (controller is None or r["controller"]==controller)]
            num[i,j]=sum(r[metric] for r in group if r.get(exposure_key,1));den[i,j]=sum(r.get(exposure_key,1) for r in group)
            cells.append(dict(condition=condition,controller=controller,metric=metric,model_seed=seed,cohort=cohort,numerator=float(num[i,j]),exposure=int(den[i,j])))
    rd,pd=draws(cfg,n,p);samples=crossed_ratio(num,den,rd,pd);finite=samples[np.isfinite(samples)];empty=1-len(finite)/len(samples)
    result=dict(estimate=float(num.sum()/den.sum()) if den.sum() else None,exposures=int(den.sum()),confidence=.95,
        confidence_interval=np.quantile(finite,[.025,.975]).tolist() if den.sum() and empty<=.05 else None,empty_bootstrap_fraction=empty,
        replica_estimates={str(seed):float(num[i].sum()/den[i].sum()) if den[i].sum() else None for i,seed in enumerate(cfg["model_seeds"])},exploratory=True)
    return result,cells


def diagnostic_summary(ranking,pairs,timing,cfg):
    results={};cells=[]
    for cond in cfg["conditions"]:
        for name in (*NAMES,"simulation_reference"):
            result,c=conditional(ranking,cfg,cond,f"{name}_gain",cfg["test_cohorts"]);results[f"assignment:{cond}:{name}"]=result;cells+=c
        for name in ("true_value","shuffled_value"):
            result,c=conditional(pairs,cfg,cond,f"{name}_squared_error",cfg["test_cohorts"],exposure_key="pair_count")
            result["rmse"]=float(np.sqrt(result["estimate"])) if result["estimate"] is not None else None
            results[f"ranking_mse:{cond}:{name}"]=result;cells+=c
        for name in ("frozen_ppo","true_value"):
            for metric in ("correction_gain","no_pm_gain"):
                result,c=conditional(timing,cfg,cond,metric,cfg["test_cohorts"][:cfg["timing_profiles"]],controller=name)
                results[f"timing:{cond}:{name}:{metric}"]=result;cells+=c
    return dict(exploratory_contrasts=results,ranking_states=len(ranking),ranking_pairs=len(pairs),timing_records=len(timing),
        uncertainty="crossed replica/profile conditional ratio; classes/futures are not independent profiles; no interval when >5% empty bootstrap cells",
        ranking_continuation="frozen PPO",timing_continuation="same evaluated controller at t+1..47; common deadline tail",simulation_reference="chosen on32 independent-selection futures, measured on64 evaluation futures; noisy, not oracle"),cells
