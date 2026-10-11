"""Predeclared scorer contrasts; decoder/timing mechanism checks kept separate."""
import itertools
import numpy as np
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_joint_statistics import bootstrap_samples
from ht_pdm_fjsp.maintenance_assignment_policy import ARMS, DECODERS, controller


def summarize(rows, cfg):
    index = {(r["condition"], r["cohort"], r["shock"], r["controller"], r["train_seed"]): r for r in rows}
    names = [controller(a, d) for a, d in itertools.product(ARMS, DECODERS)]
    seeds, profiles = cfg["train_seeds"], cfg["test_cohorts"]
    expected = {(cond,c,s,name,seed) for cond,c,s in itertools.product(cfg["conditions"],profiles,cfg["test_shocks"])
                for name,seed in [("best_rule",-1),*itertools.product(names,seeds)]}
    if len(index) != len(rows) or set(index) != expected or any(r["horizon"] != cfg["horizon"] for r in rows):
        raise RuntimeError("incomplete/duplicate test grid or horizon")
    rng = np.random.default_rng(role_seed(cfg["analysis_seed"], "maintenance_assignment_scoring_v1:crossed_bootstrap"))
    rd = rng.integers(len(seeds), size=(cfg["bootstrap_samples"],len(seeds)))
    pd = rng.integers(len(profiles), size=(cfg["bootstrap_samples"],len(profiles)))
    def matrix(cond, name):
        return np.array([[np.mean([index[cond,c,s,name,seed if name != "best_rule" else -1]["extended_objective"]
            for s in cfg["test_shocks"]]) for c in profiles] for seed in seeds])
    primary, secondary, cells, means = {}, {}, [], {}
    metrics = ("extended_objective","base_objective","tail_objective", "extended_failure_cost",
        "extended_waiting_cost","extended_maintenance_cost","extended_unavailability_cost",
        "preventive_starts","corrective_starts","pm_feasible_ticks","wait_with_pm_feasible",
        "stop_with_pm_extension_feasible","pm_assignment_alternative_ticks")
    for cond in cfg["conditions"]:
        means[cond] = {name:{m:float(np.mean([r[m] for r in rows if r["condition"] == cond and r["controller"] == name]))
            for m in metrics} for name in ["best_rule",*names]}
        comparisons = [(controller(ARMS[0],d),controller(ARMS[1],d),True) for d in DECODERS]
        comparisons += [(controller(a,"greedy"),controller(a,"assignment_mass"),False) for a in ARMS]
        comparisons += [("best_rule",controller(ARMS[1],d),False) for d in DECODERS]
        for left,right,is_primary in comparisons:
            a,b=matrix(cond,left),matrix(cond,right)
            confidence=cfg["primary_confidence"] if is_primary else .95
            draws=bootstrap_samples(a,b,rd,pd,relative=True)
            ci=np.quantile(draws,[(1-confidence)/2,(1+confidence)/2]).tolist()
            status="PILOT_GAIN_SIGNAL" if ci[0]>cfg["cost_margin"] else "BELOW_MARGIN" if ci[1]<cfg["cost_margin"] else "UNRESOLVED"
            result=dict(baseline=left,method=right,baseline_mean=float(a.mean()),method_mean=float(b.mean()),
                relative_gain=float((a-b).mean()/a.mean()),absolute_gain=float((a-b).mean()),
                confidence=confidence,confidence_interval=ci,material_margin=cfg["cost_margin"],
                replica_gains={str(seed):float((a[i]-b[i]).mean()/a[i].mean()) for i,seed in enumerate(seeds)},
                leave_one_replica_out={str(seed):float((np.delete(a,i,0)-np.delete(b,i,0)).mean()/np.delete(a,i,0).mean())
                    for i,seed in enumerate(seeds)} if len(seeds)>1 else {},
                status="ENGINEERING_ONLY" if cfg["profile"]=="smoke" else status,exploratory=not is_primary)
            name=f"{cond}:{right}_vs_{left}"
            (primary if is_primary else secondary)[name]=result
            for i,seed in enumerate(seeds):
                for j,cohort in enumerate(profiles):
                    cells.append(dict(condition=cond,contrast=name,train_seed=seed,cohort=cohort,
                        baseline_mean=float(a[i,j]),method_mean=float(b[i,j]),gain=float(a[i,j]-b[i,j])))
    return dict(means=means,primary_contrasts=primary,secondary_contrasts=secondary,
        uncertainty="crossed paired training-replica/profile bootstrap; fixed shocks; three replicas limit calibration",
        multiplicity="four primary two-sided98.75% intervals; approximate Bonferroni; secondary95% exploratory",
        interpretation="scorer and decoding mechanisms separated; timing diagnostics separate; no optimality/novelty/transfer claim"),cells
