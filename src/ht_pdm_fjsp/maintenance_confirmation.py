"""Fresh-panel confirmation of cost headroom and service noninferiority.

Only the macOS smoke belongs to the assistant; the full run is handed to the lab.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_boundary_sensitivity as boundary
from ht_pdm_fjsp import maintenance_opportunity_cost as prior
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, ENV_VERSION
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays, batch_tick, fixed_candidates
from ht_pdm_fjsp.maintenance_solver_diagnostics import Requests

VERSION = "maintenance_confirmation_v1"
ROOT = Path(__file__).resolve().parents[2]
CONDITION = "skill_mask"
CONTROLLERS = ("best_rule", "fixed_tail")
BASELINE = TAIL_RULE = "deadline_matching"
snapshot = boundary.snapshot
request_metrics = boundary.request_metrics


def settings(execution_profile):
    if execution_profile not in ("smoke", "full"):
        raise ValueError("unknown profile")
    full = execution_profile == "full"
    return dict(profile=execution_profile, horizon=48 if full else 12,
        tail_ticks=12 if full else 3, scenarios=128 if full else 8,
        test_cohorts=list(range(11100000, 11100256)) if full else [11400000, 11400001],
        test_shocks=list(range(11200000, 11200016)) if full else [11500000, 11500001],
        analysis_seed=11300000 if full else 11300001,
        bootstrap_samples=20000 if full else 200,
        condition=CONDITION, controllers=list(CONTROLLERS), baseline=BASELINE,
        continuation="adaptive_isolated_matching", tail_rule=TAIL_RULE, discount=1.,
        cost_margin=.01, service_guard_pending=.1, service_guard_failed=.1,
        service_guard_overdue_per_tick=.25/12,
        primary_confidence=1-.05/4, family_size=4,
        training="NOT_APPLICABLE", train_seeds=[], development_seeds=[],
        singleton_fast_path=True, thread_limit=1)


def counts(cfg):
    episodes = len(cfg["test_cohorts"])*len(cfg["test_shocks"])*len(CONTROLLERS)
    return dict(development_episodes=0, test_episodes=episodes,
        base_decisions=episodes*cfg["horizon"],
        tail_decisions=episodes*cfg["tail_ticks"],
        test_decisions=episodes*(cfg["horizon"]+cfg["tail_ticks"]))


def exclusion_registry():
    return json.loads((ROOT/f"configs/{VERSION}_profile_exclusions.json").read_text())


def build_panels(cfg):
    # Physical signatures ignore H and arbitrary machine/worker labels.
    seen = {boundary.frozen(s) for s in exclusion_registry()["previous_physical_signatures"][CONDITION]}
    if cfg["profile"] == "smoke":
        full, _ = build_panels(settings("full"))  # metadata only, no outcomes
        seen.update(boundary.signature(c) for c in full.values())
    panels, rows = {}, []
    for cohort in cfg["test_cohorts"]:
        for attempt in range(10000):
            c = prior.make_config(cohort, CONDITION, cfg["horizon"], attempt)
            if boundary.signature(c) not in seen:
                break
        else:
            raise RuntimeError("physical profile pool exhausted")
        seen.add(boundary.signature(c))
        panels[cohort] = c
        rows.append(dict(split="test" if cfg["profile"]=="full" else "smoke",
                         cohort=cohort, condition=CONDITION,
                         generator_attempt=attempt, config=asdict(c)))
    return panels, rows


def seed_audit(cfg):
    def integers(value):
        if isinstance(value, dict):
            return set().union(set(), *(integers(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(set(), *(integers(v) for v in value))
        return {value} if isinstance(value, int) and not isinstance(value, bool) else set()
    own_path = ROOT/f"configs/{VERSION}_seed_registry.json"
    own = json.loads(own_path.read_text())["panels"]
    if any(cfg[k] != v for k,v in own[cfg["profile"]].items()):
        raise ValueError("settings differ from committed seed registry")
    if integers(own["full"]) & integers(own["smoke"]):
        raise ValueError("smoke/full seed overlap")
    historical = integers(exclusion_registry()["previous_seed_panels"])
    hashes = {}
    for p in sorted((ROOT/"configs").glob("*seed_registry.json")):
        if p == own_path:
            continue
        historical |= integers(json.loads(p.read_text()))
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for panel in own.values():
        groups = [set(panel["test_cohorts"]), set(panel["test_shocks"]), {panel["analysis_seed"]}]
        if any(a & b for a,b in itertools.combinations(groups,2)) or set.union(*groups) & historical:
            raise ValueError("seed overlap")
    return dict(status="PASS", historical_registry_hashes=hashes,
                raw_seed_registry_sha256=hashlib.sha256(own_path.read_bytes()).hexdigest(),
                physical_exclusion_sha256=hashlib.sha256((ROOT/f"configs/{VERSION}_profile_exclusions.json").read_bytes()).hexdigest())


def forecast_plan(c, state, h, tail, forecast_seed, scenarios, fixed=True, fast_path=True):
    if not fixed:
        raise ValueError("confirmation uses fixed allocation only")
    if h < 1 or tail < 0 or scenarios < 2:
        raise ValueError("positive horizon, nonnegative tail and >=2 scenarios required")
    candidates = fixed_candidates(c, state)
    if fast_path and len(candidates) == 1:
        # No comparison is possible. Scores deliberately remain uncomputed.
        return candidates[0], dict(candidates=candidates, forecast_ticks=0,
            terminal_ticks=0, selected_score=None, singleton_skipped=True)
    action, diag = boundary.forecast_plan(c, state, h, tail, forecast_seed, scenarios, fixed=True)
    diag["singleton_skipped"] = False
    return action, diag


def decide(contrasts, engineering=False):
    if engineering:
        return "ENGINEERING_ONLY"
    if all(c["status"] == "PASS" for c in contrasts.values()):
        return "GO"
    if any(c["status"] == "FAIL" for c in contrasts.values()):
        return "NO_GO"
    return "INCONCLUSIVE"


def summary(rows, cfg):
    key = lambda r:(r["condition"],r["horizon"],r["cohort"],r["shock"],r["controller"])
    indexed = {key(r):r for r in rows}
    expected = set(itertools.product([CONDITION],[cfg["horizon"]],cfg["test_cohorts"],cfg["test_shocks"],CONTROLLERS))
    if len(rows) != len(indexed) or set(indexed) != expected:
        raise RuntimeError("incomplete or duplicated test grid")
    definitions = (
        ("cost_headroom", "interior_cost_per_machine_tick", "best_rule", "fixed_tail", cfg["cost_margin"], True),
        ("pending_guard", "interior_pending_mean", "fixed_tail", "best_rule", cfg["service_guard_pending"], False),
        ("failed_guard", "interior_failed_mean", "fixed_tail", "best_rule", cfg["service_guard_failed"], False),
        ("overdue_guard", "interior_overdue_per_tick", "fixed_tail", "best_rule", cfg["service_guard_overdue_per_tick"], False),
    )
    rng = np.random.default_rng(role_seed(cfg["analysis_seed"],f"{VERSION}:profile_bootstrap"))
    draws = rng.integers(len(cfg["test_cohorts"]),size=(cfg["bootstrap_samples"],len(cfg["test_cohorts"])))
    q = [(1-cfg["primary_confidence"])/2,(1+cfg["primary_confidence"])/2]
    contrasts, differences = {}, []
    for label, metric, left, right, margin, relative in definitions:
        a,b = [np.array([np.mean([indexed[CONDITION,cfg["horizon"],cohort,shock,name][metric]
                      for shock in cfg["test_shocks"]]) for cohort in cfg["test_cohorts"]]) for name in (left,right)]
        delta = (a-b)[draws].mean(1)
        statistic = float((a-b).mean())
        if relative:
            denominators = a[draws].mean(1)
            if a.mean()<=0 or np.any(denominators<=0):
                raise RuntimeError("positive baseline cost required")
            samples = delta/denominators
            statistic /= float(a.mean())
        else:
            samples = delta
        ci = np.quantile(samples,q).tolist()
        status = ("PASS" if (ci[0]>margin if relative else ci[1]<margin) else
                  "FAIL" if (ci[1]<margin if relative else ci[0]>margin) else "INCONCLUSIVE")
        contrasts[label] = dict(metric=metric,left=left,right=right,profiles=len(a),
            shocks_per_profile=len(cfg["test_shocks"]),left_mean=float(a.mean()),right_mean=float(b.mean()),
            mean_difference=float((a-b).mean()), estimate=statistic,
            units="fraction of baseline cost" if relative else "absolute difference",
            confidence=cfg["primary_confidence"],confidence_interval=ci,
            exploratory_95_interval=np.quantile(samples,[.025,.975]).tolist(),
            margin=margin,pass_direction="lower > margin" if relative else "upper < margin",
            profile_wins=int(np.sum(a>b) if relative else np.sum(a<b)),
            profile_losses=int(np.sum(a<b) if relative else np.sum(a>b)),profile_ties=int(np.sum(a==b)),
            status="ENGINEERING_ONLY" if cfg["profile"]=="smoke" else status)
        for cohort,x,y in zip(cfg["test_cohorts"],a,b):
            differences.append(dict(condition=CONDITION,horizon=cfg["horizon"],contrast=label,
                                    cohort=cohort,left_mean=float(x),right_mean=float(y),difference=float(x-y)))
    def observed_mean(name, metric):
        values = [r[metric] for r in rows if r["controller"]==name and r[metric] is not None]
        return float(np.mean(values)) if values else None
    means = {name:{metric:observed_mean(name,metric)
                  for metric in rows[0] if metric not in ("condition","cohort","shock","horizon","controller")}
             for name in CONTROLLERS}
    result = decide(contrasts,cfg["profile"]=="smoke")
    return dict(means=means,contrasts=contrasts,decision=result,
        advance_to_training=result=="GO",
        resource_policy="advance only on GO; inconclusive is not evidence of no effect",
        uncertainty_scope="paired profile bootstrap conditional on frozen shock panel",
        multiplicity="four two-sided Bonferroni-adjusted 98.75% percentile intervals; approximate bootstrap coverage",
        terminal_value_scope="finite shared deadline continuation; not an infinite-horizon optimum"),differences


def episode(config, cohort, shock, condition, horizon, controller, best_rule, cfg,
            journals=None, time_totals=None):
    tail = cfg["tail_ticks"]
    c = replace(config, horizon=horizon+tail)
    env = WaitingEnv(c)
    env_seed = role_seed(shock, f"{VERSION}:physical:{cohort}")
    env.reset(env_seed)
    base_requests, all_requests = Requests(env.state), Requests(env.state)
    identity = dict(condition=condition, cohort=cohort, shock=shock, horizon=horizon, controller=controller)
    base_metrics = None
    interior_begin, interior_end = horizon//4, horizon-horizon//4
    interior = defaultdict(float)
    online_latency = []
    tail_latency = []
    for t in range(c.horizon):
        state = env.state
        phase = "base" if t < horizon else "tail"
        forecast_seed = role_seed(shock, f"{VERSION}:forecast:{cohort}:{t}")
        start = time.perf_counter_ns()
        if phase == "tail":
            action = prior.rule_action(c, state, c.horizon-t, TAIL_RULE)
            diag = dict(candidates=(), forecast_ticks=0, terminal_ticks=0, selected_score=None)
        elif controller in CONTROLLERS[1:]:
            action, diag = forecast_plan(c, state, horizon-t,
                tail if controller.endswith("tail") else 0, forecast_seed,
                cfg["scenarios"], fixed=True, fast_path=cfg["singleton_fast_path"])
        else:
            action = prior.rule_action(c, state, horizon-t, best_rule if controller == "best_rule" else controller)
            diag = dict(candidates=(), forecast_ticks=0, terminal_ticks=0, selected_score=None)
        validate_matching(c, state, action)
        if phase == "base" and controller == "fixed_tail" and action not in fixed_candidates(c,state):
            raise RuntimeError("fixed allocation audit mismatch")
        elapsed = (time.perf_counter_ns()-start)/1e6
        (online_latency if phase == "base" else tail_latency).append(elapsed)
        _, _, _, info = env.step(action)
        selected = np.zeros((1, c.machines, c.technicians), dtype=bool)
        for m, j in action:
            selected[0, m, j] = True
        following, costs = batch_tick(c, arrays(state), selected, np.asarray([env.events[t]]))
        if not all(np.array_equal(following[k][0], getattr(env.state, k)) for k in following):
            raise RuntimeError("scalar/batch state mismatch")
        if not np.isclose(costs[0], info["objective"], atol=1e-8, rtol=0):
            raise RuntimeError("scalar/batch cost mismatch")
        all_requests.step(t, state, action, env.state)
        if phase == "base":
            base_requests.step(t, state, action, env.state)
        if interior_begin <= t < interior_end:
            for metric in ("objective", "maintenance_cost", "failure_cost", "unavailability_cost", "waiting_cost", "overdue_waiting_ticks"):
                interior[metric] += info[metric]
            for metric, value in snapshot(state).items():
                interior[metric] += value
        if time_totals is not None:
            totals = time_totals[condition, horizon, controller, t, phase]
            totals["episodes"] += 1
            for metric, value in info.items():
                totals[metric] += value
            for metric, value in snapshot(state).items():
                totals[f"before_{metric}"] += value
            for metric, value in snapshot(env.state).items():
                totals[f"after_{metric}"] += value
        if journals:
            journals["decisions"].add({**identity, "time": t, "phase": phase,
                "env_seed": env_seed, "forecast_seed": forecast_seed,
                "state": prior.compact(asdict(state)), "action": prior.compact(action),
                "following": prior.compact(asdict(env.state)), "physical_metrics": prior.compact(info)})
            journals["latency"].add({**identity, "time": t, "phase": phase,
                "latency_ms": elapsed, "candidate_count": len(diag["candidates"]),
                "singleton_skipped": diag.get("singleton_skipped",False),
                "configured_forecast_ticks": horizon-t+tail if phase=="base" and controller=="fixed_tail" else 0,
                "configured_forecast_scenarios": cfg["scenarios"] if phase=="base" and controller=="fixed_tail" else 0,
                "forecast_scenarios": cfg["scenarios"] if diag["forecast_ticks"] else 0,
                "forecast_ticks": diag["forecast_ticks"], "terminal_ticks": diag["terminal_ticks"],
                "forecast_transition_samples": len(diag["candidates"])*cfg["scenarios"]*diag["forecast_ticks"],
                "predicted_action_cost": diag["selected_score"]})
        if t+1 == horizon:
            base_metrics = env.metrics.copy()
            cutoff = snapshot(env.state)
            cut_requests = [{**r, "overdue": r["wait"] > c.waiting_limit} for r in base_requests.finish(horizon)]
    extended_requests = [{**r, "overdue": r["wait"] > c.waiting_limit} for r in all_requests.finish(c.horizon)]
    extended_index = {(r["machine"],r["request"]): r for r in extended_requests}
    carryover = []
    for r in cut_requests:
        if r["censored"]:
            later = extended_index[r["machine"],r["request"]]
            carryover.append(dict(machine=r["machine"], request=r["request"], arrival=r["arrival"],
                                  wait_at_cutoff=r["wait"], started_in_tail=later["served"],
                                  wait_at_start_or_end=later["wait"], censored_after_tail=later["censored"]))
    if journals:
        for scope, rows in (("cutoff",cut_requests),("extended",extended_requests)):
            for r in rows:
                journals["requests"].add({**identity, "scope": scope, **r})
        for r in carryover:
            journals["carryover"].add({**identity, **r})
        journals["coordination"].add({**identity, "base_decisions": horizon,
            "tail_decisions": tail, "invalid_actions": 0, "fixed_allocation_audit": "PASS", "scalar_batch_replay": "PASS"})
    window = interior_end-interior_begin
    result = {**identity,
        **{f"base_{k}": v for k,v in base_metrics.items()},
        **{f"tail_{k}": v-base_metrics[k] for k,v in env.metrics.items()},
        **{f"extended_{k}": v for k,v in env.metrics.items()},
        **{f"cutoff_{k}": v for k,v in cutoff.items()},
        **{f"post_tail_{k}": v for k,v in snapshot(env.state).items()},
        **{f"cutoff_{k}": v for k,v in request_metrics(cut_requests).items()},
        **{f"extended_{k}": v for k,v in request_metrics(extended_requests).items()},
        "base_cost_per_machine_tick": base_metrics["objective"]/(c.machines*horizon),
        "extended_cost_per_machine_tick": env.metrics["objective"]/(c.machines*c.horizon),
        "interior_begin": interior_begin, "interior_end": interior_end,
        "interior_cost_per_machine_tick": interior["objective"]/(c.machines*window),
        "interior_pending_mean": interior["pending"]/window,
        "interior_failed_mean": interior["failed"]/window,
        "interior_overdue_per_tick": interior["overdue_waiting_ticks"]/window,
        **{f"interior_{k}": interior[k] for k in
           ("objective","maintenance_cost","failure_cost","unavailability_cost","waiting_cost")},
        "carryover_requests": len(carryover),
        "carryover_started_in_tail": sum(r["started_in_tail"] for r in carryover),
        "carryover_censored_after_tail": sum(r["censored_after_tail"] for r in carryover),
        "latency_mean_ms": float(np.mean(online_latency)),
        "latency_p50_ms": float(np.quantile(online_latency,.5)),
        "latency_p95_ms": float(np.quantile(online_latency,.95)),
        "latency_p99_ms": float(np.quantile(online_latency,.99)),
        "tail_rule_latency_mean_ms": float(np.mean(tail_latency)),
    }
    if not np.isclose(result["extended_objective"],result["base_objective"]+result["tail_objective"],atol=1e-8,rtol=0):
        raise RuntimeError("base/tail reconciliation")
    return result



def run(output, execution_profile="smoke", device="cpu"):
    if device!="cpu":
        raise ValueError("CPU only")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("new/empty output directory required")
    output.mkdir(parents=True,exist_ok=True)
    cfg = settings(execution_profile)
    now = lambda:datetime.now(timezone.utc).isoformat()
    git = lambda *args:subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
    manifest = dict(protocol=VERSION,status="RUNNING",started=now(),commit=git("rev-parse","HEAD"),
        dirty=bool(git("status","--porcelain")),settings=cfg,expected_counts=counts(cfg),
        environment_version=ENV_VERSION,device=device,python=platform.python_version(),
        platform=platform.platform(),machine=platform.machine(),logical_cpus=os.cpu_count(),
        numpy=np.__version__,training="NOT_APPLICABLE",full_test_outcomes_opened=False,
        thread_environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})
    prior.write_json(output/"manifest.json",manifest)
    journals = {}
    try:
        if execution_profile=="full" and manifest["dirty"]:
            raise ValueError("full experiment requires a clean committed worktree")
        if execution_profile=="full" and any(v!="1" for v in manifest["thread_environment"].values()):
            raise ValueError("full experiment requires OMP/OPENBLAS/MKL_NUM_THREADS=1")
        manifest["seed_audit"] = seed_audit(cfg)
        panels,registry = build_panels(cfg)
        prior.write_json(output/"resolved_config.json",dict(settings=cfg,physical_profiles=registry))
        prior.write_json(output/"seed_registry.json",dict(raw_panels={k:cfg[k] for k in ("test_cohorts","test_shocks","analysis_seed")},
                                                       exclusion_registry=exclusion_registry()))
        frozen_rule = json.loads((ROOT/f"configs/{VERSION}_frozen_baseline.json").read_text())
        if frozen_rule["baseline"]!=BASELINE or frozen_rule["source_cell"]!="skill_mask:48":
            raise ValueError("frozen baseline mismatch")
        prior.write_json(output/"frozen_baseline.json",frozen_rule)
        shutil.copyfile(ROOT/f"docs/{VERSION}_plan.md",output/"plan.md")
        snapshot_dir = output/"source_snapshot"; snapshot_dir.mkdir()
        for filename in ("maintenance_confirmation.py","maintenance_boundary_sensitivity.py","maintenance_opportunity_cost.py",
                         "maintenance_dispatch.py","maintenance_waiting.py","maintenance_heterogeneous_model.py","maintenance_solver_diagnostics.py"):
            shutil.copyfile(ROOT/f"src/ht_pdm_fjsp/{filename}",snapshot_dir/filename)
        for filename in ("pyproject.toml","uv.lock"):
            shutil.copyfile(ROOT/filename,snapshot_dir/filename)
        for p in sorted((ROOT/"configs").glob("*seed_registry.json")):
            shutil.copyfile(p,snapshot_dir/p.name)
        for suffix in ("profile_exclusions","frozen_baseline"):
            shutil.copyfile(ROOT/f"configs/{VERSION}_{suffix}.json",snapshot_dir/f"{VERSION}_{suffix}.json")
        for name in ("episodes.partial","decisions","requests","carryover","latency","coordination"):
            journals[name] = prior.Journal(output/f"{name}.csv")
        manifest["full_test_outcomes_opened"] = execution_profile=="full"
        prior.write_json(output/"manifest.json",manifest)
        rows = []
        time_totals = defaultdict(lambda:defaultdict(float))
        grid = list(itertools.product(cfg["test_cohorts"],cfg["test_shocks"],CONTROLLERS))
        for cohort,shock,controller in tqdm(grid,desc="Paired cost/service confirmation"):
            row = episode(panels[cohort],cohort,shock,CONDITION,cfg["horizon"],controller,BASELINE,cfg,journals,time_totals)
            rows.append(row); journals["episodes.partial"].add(row)
        result,differences = summary(rows,cfg)
        prior.write_json(output/"summary.json",result)
        prior.write_csv(output/"episodes.csv",rows)
        prior.write_csv(output/"profile_differences.csv",differences)
        time_rows = []
        for (cond,horizon,controller,t,phase),values in sorted(time_totals.items()):
            time_rows.append(dict(condition=cond,horizon=horizon,controller=controller,time=t,phase=phase,
                                 episodes=int(values["episodes"]),
                                 **{k:v/values["episodes"] for k,v in values.items() if k!="episodes"}))
        prior.write_csv(output/"time_profiles.csv",time_rows)
        actual = dict(development_episodes=0,test_episodes=len(rows),base_decisions=sum(r["horizon"] for r in rows),
                      tail_decisions=len(rows)*cfg["tail_ticks"],test_decisions=sum(r["horizon"]+cfg["tail_ticks"] for r in rows))
        if actual!=counts(cfg):
            raise RuntimeError("count audit mismatch")
        manifest.update(status="COMPLETED",audit="PASS",actual_counts=actual)
    except BaseException as error:
        manifest.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for journal in journals.values():
            journal.close()
        manifest["finished"] = now()
        manifest["artifacts"] = {str(p.relative_to(output)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
                                 for p in sorted(output.rglob("*")) if p.is_file() and p.name!="manifest.json"}
        prior.write_json(output/"manifest.json",manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=("smoke","full"),default="smoke")
    parser.add_argument("--device",choices=("cpu",),default="cpu")
    parser.add_argument("--output-dir",required=True,type=Path)
    args = parser.parse_args()
    run(args.output_dir,args.profile,args.device)


if __name__=="__main__":
    main()
