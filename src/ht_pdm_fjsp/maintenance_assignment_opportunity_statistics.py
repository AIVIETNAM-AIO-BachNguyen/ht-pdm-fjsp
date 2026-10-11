"""Prospective episode contrasts with source/profile crossed uncertainty."""
import itertools
import numpy as np
from ht_pdm_fjsp.maintenance_assignment_opportunity_benchmark import VERSION, NAMES
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_joint_statistics import bootstrap_samples
from ht_pdm_fjsp.maintenance_timing_statistics import crossed_ratio


def draws(cfg):
    rng = np.random.default_rng(role_seed(cfg["analysis_seed"], f"{VERSION}:crossed_bootstrap"))
    n, p = len(cfg["model_seeds"]), len(cfg["test_cohorts"])
    return rng.integers(n, size=(cfg["bootstrap_samples"], n)), rng.integers(p, size=(cfg["bootstrap_samples"], p))


def summarize(rows, cfg):
    index = {(r["condition"], r["cohort"], r["shock"], r["controller"], r["train_seed"]): r for r in rows}
    expected = set(itertools.product(cfg["conditions"], cfg["test_cohorts"], cfg["test_shocks"], NAMES, cfg["model_seeds"]))
    if len(index) != len(rows) or set(index) != expected or any(r["horizon"] != 48 for r in rows):
        raise RuntimeError("test grid mismatch")
    rd, pd = draws(cfg); primary = {}; secondary = {}; cells = []; means = {}
    metrics = ("extended_objective", "base_objective", "tail_objective", "extended_failure_cost",
        "extended_waiting_cost", "extended_maintenance_cost", "extended_unavailability_cost",
        "preventive_starts", "corrective_starts", "wait_with_pm_feasible", "pm_assignment_alternative_ticks",
        "assignment_override_ticks", "guard_rejected_ticks", "batch_amortized_policy_ms_mean")
    for cond in cfg["conditions"]:
        means[cond] = {name: {m: float(np.mean([r[m] for r in rows if r["condition"] == cond and r["controller"] == name])) for m in metrics} for name in NAMES}
        matrix = lambda name: np.array([[np.mean([index[cond, c, s, name, seed]["extended_objective"] for s in cfg["test_shocks"]])
            for c in cfg["test_cohorts"]] for seed in cfg["model_seeds"]])
        contrasts = [("opportunity_high_gated", "frozen_ppo", True),
            ("opportunity_high_gated", "opportunity_high_ungated", True)]
        contrasts += [(f"{method}_{suffix}", f"{baseline}_{suffix}", False)
            for method, baseline in (("opportunity_low", "landmark_low"), ("opportunity_high", "opportunity_low"))
            for suffix in ("ungated", "gated")]
        contrasts += [(f"{arm}_gated", f"{arm}_ungated", False) for arm in ("landmark_low", "opportunity_low", "opportunity_high")]
        for method, baseline, is_primary in contrasts:
            a, b = matrix(baseline), matrix(method)
            confidence = cfg["primary_confidence"] if is_primary else .95
            samples = bootstrap_samples(a, b, rd, pd, True)
            ci = np.quantile(samples, [(1 - confidence) / 2, (1 + confidence) / 2]).tolist()
            key = f"{cond}:{method}_vs_{baseline}"
            row = dict(method=method, baseline=baseline, baseline_mean=float(a.mean()), method_mean=float(b.mean()),
                absolute_gain=float((a - b).mean()), relative_gain=float((a - b).mean() / a.mean()),
                confidence=confidence, confidence_interval=ci, material_margin=cfg["cost_margin"],
                replica_gains={str(seed): float((a[i] - b[i]).mean() / a[i].mean()) for i, seed in enumerate(cfg["model_seeds"])},
                leave_one_replica_out={str(seed): float((np.delete(a, i, 0) - np.delete(b, i, 0)).mean() / np.delete(a, i, 0).mean())
                    for i, seed in enumerate(cfg["model_seeds"])} if len(a) > 1 else {},
                status="ENGINEERING_ONLY" if cfg["profile"] == "smoke" else "PILOT_GAIN_SIGNAL" if ci[0] > cfg["cost_margin"] else
                    "BELOW_MARGIN" if ci[1] < cfg["cost_margin"] else "UNRESOLVED", exploratory=not is_primary)
            (primary if is_primary else secondary)[key] = row
            for i, seed in enumerate(cfg["model_seeds"]):
                for j, cohort in enumerate(cfg["test_cohorts"]):
                    cells.append(dict(condition=cond, contrast=key, primary=is_primary, model_seed=seed, cohort=cohort,
                        baseline_mean=float(a[i, j]), method_mean=float(b[i, j]), gain=float(a[i, j] - b[i, j])))
    return dict(means=means, primary_contrasts=primary, secondary_contrasts=secondary,
        multiplicity="four primary98.75% intervals, approximate Bonferroni; secondary95% exploratory",
        limitations="three sources, natural opportunity selection, nested label futures, empirical uncertainty guard, Q^pi continuation mismatch; no timing/scale/novelty claim"), cells


def diagnostic_summary(rows, cfg):
    profiles = cfg["test_cohorts"][:cfg["diagnostic_profiles"]]
    rng = np.random.default_rng(role_seed(cfg["analysis_seed"], f"{VERSION}:crossed_bootstrap"))
    n, p = len(cfg["model_seeds"]), len(profiles)
    rd = rng.integers(n, size=(cfg["bootstrap_samples"], n)); pd = rng.integers(p, size=(cfg["bootstrap_samples"], p))
    results = {}
    for cond, origin, name in itertools.product(cfg["conditions"], ("natural",), NAMES[1:]):
        selected = [r for r in rows if r["condition"] == cond and r["origin"] == origin]
        num, den = np.zeros((n, p)), np.zeros((n, p))
        for i, seed in enumerate(cfg["model_seeds"]):
            for j, cohort in enumerate(profiles):
                group = [r for r in selected if r["model_seed"] == seed and r["cohort"] == cohort and r["exposure"]]
                num[i, j] = sum(r[f"{name}_gain"] for r in group); den[i, j] = len(group)
        samples = crossed_ratio(num, den, rd, pd); finite = samples[np.isfinite(samples)]
        empty = 1 - len(finite) / len(samples)
        override = [r for r in selected if r[f"{name}_changed"]]
        results[f"{cond}:{origin}:{name}"] = dict(states=len(selected), exposures=int(den.sum()), overrides=len(override),
            harmful_overrides=sum(r[f"{name}_gain"] < 0 for r in override),
            largest_harm=max([-r[f"{name}_gain"] for r in override if r[f"{name}_gain"] < 0], default=0.),
            mean_gain=float(num.sum() / den.sum()) if den.sum() else None,
            confidence_interval=np.quantile(finite, [.025, .975]).tolist() if den.sum() and empty <= .05 else None,
            empty_bootstrap_fraction=empty, exploratory=True)
    return dict(root_contrasts=results, records=len(rows), continuation="frozen PPO at original clock; common tail",
        reference="independent selection/evaluation; not oracle", uncertainty="profile/source clustering, fixed MC tapes; natural source opportunities only")
