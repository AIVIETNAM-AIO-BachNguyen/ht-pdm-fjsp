"""Read-only audit/reanalysis of the downloaded full run; outputs here only."""
import csv
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import types

sys.dont_write_bytecode = True

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "lab_results/maintenance_opportunity_cost_v1_full_cpu_20261010T074101Z"
OUT = Path(__file__).resolve().parent


def read_csv(name):
    with (RUN / name).open() as stream:
        yield from csv.DictReader(stream)


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def key(row):
    return row["condition"], int(row["cohort"]), int(row["shock"]), row["controller"]


def close(a, b):
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-8), (a, b)


manifest = json.loads((RUN / "manifest.json").read_text())
assert manifest["status"] == "COMPLETED"
assert manifest["commit"] == "f25876eab7d5ba138a1cfe05d5dc8eb7cb96f736"
assert not manifest["dirty"]
hashes = {}
for name, meta in manifest["artifacts"].items():
    path = RUN / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == meta["sha256"] and path.stat().st_size == meta["bytes"], name
    hashes[name] = digest
source_check = {}
for path in (RUN / "source_snapshot").iterdir():
    relative = f"src/ht_pdm_fjsp/{path.name}" if path.suffix == ".py" else path.name
    committed = subprocess.check_output(["git", "show", f"{manifest['commit']}:{relative}"], cwd=ROOT)
    assert committed == path.read_bytes(), path.name
    source_check[relative] = hashlib.sha256(committed).hexdigest()
assert subprocess.check_output(["git", "show", f"{manifest['commit']}:docs/maintenance_opportunity_cost_v1_plan.md"], cwd=ROOT) == (RUN / "plan.md").read_bytes()
# Only import source bytes verified against the specified committed source.
package = types.ModuleType("ht_pdm_fjsp")
package.__path__ = [str(RUN / "source_snapshot")]
sys.modules["ht_pdm_fjsp"] = package
model = importlib.import_module("ht_pdm_fjsp.maintenance_opportunity_cost")
physical = importlib.import_module("ht_pdm_fjsp.maintenance_dispatch")
waiting = importlib.import_module("ht_pdm_fjsp.maintenance_waiting")
cfg = manifest["settings"]
assert cfg == model.settings("full")
assert manifest["actual_counts"] == dict(development_episodes=5376, test_episodes=30720,
                                         test_decisions=368640, exact_states=48)
seed_sets = [set(cfg[k]) for k in ("development_cohorts", "development_shocks", "test_cohorts", "test_shocks")]
assert not any(a & b for a, b in itertools.combinations(seed_sets, 2))
def integer_values(value):
    if isinstance(value, dict):
        return set().union(set(), *(integer_values(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(set(), *(integer_values(v) for v in value))
    return {value} if isinstance(value, int) and not isinstance(value, bool) else set()
historical = set()
for name, digest in manifest["seed_audit"]["historical_registry_hashes"].items():
    data = subprocess.check_output(["git", "show", f"{manifest['commit']}:configs/{name}"], cwd=ROOT)
    assert hashlib.sha256(data).hexdigest() == digest
    historical |= integer_values(json.loads(data))
assert not set.union(*seed_sets) & historical
resolved = json.loads((RUN / "resolved_config.json").read_text())
assert resolved["settings"] == cfg
configs = {}
signatures = defaultdict(set)
for entry in resolved["physical_profiles"]:
    data = entry["config"].copy()
    for field in ("service_time", "restoration"):
        data[field] = tuple(tuple(row) for row in data[field])
    c = waiting.WaitingConfig(**data)
    c.validate()
    assert c == model.make_config(entry["cohort"], entry["condition"], 12, entry["generator_attempt"])
    # Ineligible-edge restoration cannot distinguish physical equivalence.
    equivalent_rows = [tuple(sorted(tuple((c.service_time[m][j], c.restoration[m][j] if c.service_time[m][j] else 0)
                               for j in order) for m in range(c.machines)))
                       for order in itertools.permutations(range(c.technicians))]
    signature = (c.failure_age, c.failure_probability, min(equivalent_rows))
    signatures[entry["condition"], entry["split"]].add(signature)
    configs[entry["split"], entry["cohort"], entry["condition"]] = c
for condition in cfg["conditions"]:
    assert len(signatures[condition, "development"]) == 32
    assert len(signatures[condition, "test"]) == 64
    assert not signatures[condition, "development"] & signatures[condition, "test"]
assert len(configs) == 288
write_json("integrity_audit.json", dict(status="PASS", artifact_count=len(hashes), source_files=source_check,
           hashes=hashes, seed_panels_disjoint=True, historical_raw_seeds_disjoint=True,
           relabeling_equivalent_profiles_disjoint=True, commit=manifest["commit"]))
print("Integrity/source/split checks PASS", flush=True)

# Decode episodes independently; every declared controller/cohort/shock is present.
episodes = {}
for row in read_csv("episodes.csv"):
    k = key(row)
    assert k not in episodes
    episodes[k] = {f: v if f in ("condition", "controller") else None if v == "" else float(v)
                   for f, v in row.items()}
expected = set(itertools.product(cfg["conditions"], cfg["test_cohorts"], cfg["test_shocks"], cfg["controllers"]))
assert set(episodes) == expected
development = list(read_csv("development_episodes.csv"))
assert len(development) == 5376
assert len({key(row) for row in development}) == len(development)
selected = json.loads((RUN / "development_selection.json").read_text())
for condition in cfg["conditions"]:
    scores = {rule: np.mean([float(r["objective"]) for r in development
                            if r["condition"] == condition and r["controller"] == rule])
              for rule in cfg["rules"]}
    assert selected["selected"][condition]["best_rule"] == min(cfg["rules"], key=scores.get)
    assert selected["selected"][condition]["scarcity"] == min(cfg["rules"][3:], key=scores.get)
    for rule, score in scores.items():
        close(score, selected["selected"][condition]["scores"][rule])
assert selected["locked_before_test"]

# Full physical replay and independent reconstruction of all request histories.
request_rows = defaultdict(list)
trajectories = {}
time_totals = defaultdict(lambda: defaultdict(float))
current = None
seen_episodes = set()
decision_count = 0
for row in tqdm(read_csv("decisions.csv"), total=368640, desc="Replay all physical decisions"):
    k = key(row)
    if k != current:
        if current is not None:
            assert env.time == 12
        assert k not in seen_episodes
        seen_episodes.add(k)
        current = k
        condition, cohort, shock, controller = k
        c = configs["test", cohort, condition]
        env = waiting.WaitingEnv(c)
        env_seed = physical.role_seed(shock, f"{model.VERSION}:physical:{cohort}")
        env.reset(env_seed)
        pending = {}
        sequence = [0]*c.machines
        if controller in ("fixed_rollout", "flexible_rollout"):
            trajectories[k] = []
    t = int(row["time"])
    assert t == env.time and int(row["env_seed"]) == env_seed
    assert int(row["forecast_seed"]) == physical.role_seed(shock, f"{model.VERSION}:forecast:{cohort}:{t}")
    before = env.state
    assert asdict(before) == {f: tuple(v) for f, v in json.loads(row["state"]).items()}
    action = tuple(tuple(p) for p in json.loads(row["action"]))
    physical.validate_matching(c, before, action)
    if controller == "fixed_rollout":
        assert action in model.fixed_candidates(c, before)
    if controller in ("fixed_rollout", "flexible_rollout"):
        trajectories[k].append((before, action))
    # Reproduce rules exactly; approximate planner decisions are NOT rerun.
    if controller in ("isolated", "scarcity", "best_rule"):
        name = "adaptive_isolated_matching" if controller == "isolated" else selected["selected"][condition][controller]
        assert action == model.rule_action(c, before, 12-t, name)
    for m, _ in action:
        if before.failed[m]:
            arrival, number = pending.pop(m)
            request_rows[k].append(dict(machine=m, request=number, arrival=arrival, end=t,
                                       wait=t-arrival, served=True, censored=False, overdue=t-arrival>c.waiting_limit))
    _, _, terminated, info = env.step(action)
    assert info == json.loads(row["physical_metrics"])
    assert asdict(env.state) == {f: tuple(v) for f, v in json.loads(row["following"]).items()}
    servicing = set(before.assigned) | {m for m, _ in action}
    for m in range(c.machines):
        if env.state.failed[m] and not before.failed[m] and m not in servicing:
            assert m not in pending
            sequence[m] += 1
            pending[m] = t+1, sequence[m]
    totals = time_totals[condition, controller, t]
    for metric in ("jobs", "corrective_jobs", "preventive_jobs", "failures", "objective",
                   "overdue_waiting_ticks", "failed_waiting_ticks"):
        totals[metric] += info[metric]
    totals["pending_before"] += sum(before.failed[m] and m not in before.assigned for m in range(c.machines))
    totals["pending_after"] += sum(env.state.failed[m] and m not in env.state.assigned for m in range(c.machines))
    if terminated:
        for m, (arrival, number) in pending.items():
            request_rows[k].append(dict(machine=m, request=number, arrival=arrival, end=12,
                                       wait=12-arrival, served=False, censored=True, overdue=12-arrival>c.waiting_limit))
        for metric, value in env.metrics.items():
            close(value, episodes[k][metric])
        close(episodes[k]["terminal_pending"], sum(env.state.failed[m] and m not in env.state.assigned for m in range(c.machines)))
        close(episodes[k]["terminal_failed"], sum(env.state.failed))
        close(episodes[k]["terminal_servicing"], sum(r > 0 for r in env.state.remaining))
    decision_count += 1
assert len(seen_episodes) == 30720 and decision_count == 368640 and env.time == 12
raw_requests = defaultdict(list)
for row in read_csv("requests.csv"):
    raw_requests[key(row)].append({f: row[f] == "True" if f in ("served", "censored", "overdue") else int(row[f])
                                  for f in ("machine", "request", "arrival", "end", "wait", "served", "censored", "overdue")})
for k in expected:
    sort_key = lambda r: (r["machine"], r["request"])
    assert sorted(raw_requests[k], key=sort_key) == sorted(request_rows[k], key=sort_key)
    rs = request_rows[k]
    served = [r["wait"] for r in rs if r["served"]]
    ep = episodes[k]
    close(ep["requests"], len(rs)); close(ep["served_requests"], len(served))
    close(ep["censored_requests"], sum(r["censored"] for r in rs))
    close(ep["overdue_requests"], sum(r["overdue"] for r in rs))
    close(ep["observed_wait_all_requests"], sum(r["wait"] for r in rs))
    close(ep["served_wait_mean"], np.mean(served) if served else None)
    close(ep["served_wait_p95"], np.quantile(served, .95, method="inverted_cdf") if served else None)
coordination = list(read_csv("coordination.csv"))
assert len(coordination) == 30720 and {key(r) for r in coordination} == expected
assert all(r["invalid_actions"] == "0" and r["scalar_batch_replay"] == "PASS" and r["decisions"] == "12" for r in coordination)
write_json("physical_replay_audit.json", dict(status="PASS", decisions=decision_count,
           episodes=len(seen_episodes), requests=sum(map(len, request_rows.values())),
           rule_actions_reproduced=3*3*2048*12, planner_actions="feasibility and fixed restriction checked; inference not rerun"))
print("Full physical replay/request audit PASS", flush=True)

# Recompute all planned contrasts independently with the frozen random seed.
original = json.loads((RUN / "summary.json").read_text())
means = {}
contrasts = {}
metrics = ("objective", "base_cost", "maintenance_cost", "failure_cost", "unavailability_cost", "waiting_cost",
           "failures", "overdue_waiting_ticks", "terminal_pending", "terminal_failed", "terminal_servicing",
           "censored_requests", "served_requests", "observed_wait_all_requests", "latency_mean_ms")
for condition in cfg["conditions"]:
    means[condition] = {}
    for name in cfg["controllers"]:
        group = [episodes[condition, cohort, shock, name] for cohort in cfg["test_cohorts"] for shock in cfg["test_shocks"]]
        means[condition][name] = {metric: float(np.mean([r[metric] for r in group])) for metric in metrics}
        for metric in metrics:
            close(means[condition][name][metric], original["means"][condition][name][metric])
    for reference in ("fixed_rollout", "best_rule", "isolated", "scarcity"):
        left = np.array([np.mean([episodes[condition, cohort, shock, reference]["objective"] for shock in cfg["test_shocks"]]) for cohort in cfg["test_cohorts"]])
        right = np.array([np.mean([episodes[condition, cohort, shock, "flexible_rollout"]["objective"] for shock in cfg["test_shocks"]]) for cohort in cfg["test_cohorts"]])
        primary = condition == "skill_mask" and reference in ("fixed_rollout", "best_rule")
        confidence = .975 if primary else .95
        rng = np.random.default_rng(physical.role_seed(cfg["analysis_seed"], f"{condition}:{reference}"))
        draws = rng.integers(64, size=(20000, 64))
        diff = (left-right)[draws].mean(axis=1)
        relative = diff/left[draws].mean(axis=1)
        tails = [(1-confidence)/2, (1+confidence)/2]
        name = f"{condition}:{reference}_minus_flexible"
        values = dict(primary=primary, confidence=confidence, mean_difference=float((left-right).mean()),
                      confidence_interval=np.quantile(diff, tails).tolist(),
                      relative_improvement=float((left-right).mean()/left.mean()),
                      relative_confidence_interval=np.quantile(relative, tails).tolist(),
                      profile_wins=int(np.sum(left>right)), profile_ties=int(np.sum(left==right)),
                      profile_losses=int(np.sum(left<right)))
        for metric in ("mean_difference", "relative_improvement"):
            close(values[metric], original["contrasts"][name][metric])
        for metric in ("confidence_interval", "relative_confidence_interval"):
            for a, b in zip(values[metric], original["contrasts"][name][metric]):
                close(a, b)
        contrasts[name] = values

# Exact deterministic value reproducibility; no full rollout/training rerun.
exact = []
solvers = {}
for row in tqdm(list(read_csv("action_values.csv")), desc="Verify exact Q values"):
    p, h = float(row["probability"]), int(row["horizon"])
    if (p, h) not in solvers:
        c = waiting.WaitingConfig(2, 2, h, 4, p, ((2, 3), (3, 0)), ((1., 1.), (1., 1.)), waiting_price=12.)
        solvers[p, h] = model.Exact(c)
    solver = solvers[p, h]
    state = physical.DispatchState(**{f: tuple(v) for f, v in json.loads(row["state"]).items()})
    scores = {a: solver.q(state, a, h) for a in model.feasible(solver.c, state)}
    optimum = min(scores.values())
    fixed = min(scores[a] for a in model.fixed_candidates(solver.c, state))
    for field, value in dict(q_fast=scores[((0,0),)], q_reserve=scores[((0,1),)],
                             optimum=optimum, fixed_root_gap=fixed-optimum,
                             fast_minus_reserve=scores[((0,0),)]-scores[((0,1),)]).items():
        close(row[field], value)
    exact.append(dict(probability=p, horizon=h, a_age=int(row["a_age"]), b_age=int(row["b_age"]),
                      fast_minus_reserve=scores[((0,0),)]-scores[((0,1),)],
                      fixed_root_gap=fixed-optimum,
                      globally_optimal_actions=[a for a, q in scores.items() if abs(q-optimum)<1e-8]))
assert len(exact) == 48

# First divergence only: both planner actions originate from the SAME state.
divergence = {}
for condition in cfg["conditions"]:
    counts = dict(identical_trajectory=0, first_divergence_assignment_only=0,
                  first_divergence_subset_changed=0, flexible_action_outside_fixed_candidates=0)
    for cohort, shock in itertools.product(cfg["test_cohorts"], cfg["test_shocks"]):
        f = trajectories[condition, cohort, shock, "fixed_rollout"]
        g = trajectories[condition, cohort, shock, "flexible_rollout"]
        for (state_f, action_f), (state_g, action_g) in zip(f, g):
            assert state_f == state_g
            if action_f != action_g:
                label = "first_divergence_assignment_only" if {m for m, j in action_f} == {m for m, j in action_g} else "first_divergence_subset_changed"
                counts[label] += 1
                counts["flexible_action_outside_fixed_candidates"] += action_g not in model.fixed_candidates(configs["test", cohort, condition], state_g)
                break
        else:
            counts["identical_trajectory"] += 1
    divergence[condition] = counts

latencies = defaultdict(list)
latency_by_ep = defaultdict(list)
for row in read_csv("latency.csv"):
    assert int(row["forecast_scenarios"]) == (128 if row["controller"].endswith("rollout") else 0)
    latency_by_ep[key(row)].append(float(row["latency_ms"]))
    latencies[row["condition"], row["controller"]].append(float(row["latency_ms"]))
assert len(latency_by_ep) == 30720
for k, samples in latency_by_ep.items():
    assert len(samples) == 12
    for field, q in (("latency_p50_ms", .5), ("latency_p95_ms", .95), ("latency_p99_ms", .99)):
        close(episodes[k][field], np.quantile(samples, q))
    close(episodes[k]["latency_mean_ms"], np.mean(samples))
pooled = {}
for (condition, name), samples in latencies.items():
    demand = [r for k, rows in request_rows.items() if k[0] == condition and k[3] == name for r in rows]
    served = [r["wait"] for r in demand if r["served"]]
    censored = [r["wait"] for r in demand if r["censored"]]
    pooled[f"{condition}:{name}"] = dict(requests=len(demand), served=len(served), censored=len(censored),
        censored_fraction=len(censored)/len(demand), censored_observed_mean=float(np.mean(censored)),
        censored_age_zero=sum(w==0 for w in censored), overdue_requests=sum(r["overdue"] for r in demand),
        observed_wait_all_requests=sum(r["wait"] for r in demand),
        served_wait_mean=float(np.mean(served)), served_wait_p95=float(np.quantile(served,.95,method="inverted_cdf")),
        latency_mean_ms=float(np.mean(samples)), latency_p50_p95_p99_ms=np.quantile(samples,[.5,.95,.99]).tolist())
    saved = original["service_and_latency"][f"{condition}:{name}"]
    close(saved["request_count"], len(demand)); close(saved["censored_count"], len(censored))
    for a, b in zip(saved["latency_p50_p95_p99_ms"], pooled[f"{condition}:{name}"]["latency_p50_p95_p99_ms"]):
        close(a, b)

time_rows = [{"condition": k[0], "controller": k[1], "time": k[2],
              **{m: v/2048 for m, v in values.items()}} for k, values in sorted(time_totals.items())]
with (OUT / "time_profiles.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, list(time_rows[0])); writer.writeheader(); writer.writerows(time_rows)
results = dict(means=means, contrasts=contrasts, exact=exact, first_divergence=divergence,
               service_latency=pooled, development_selection=selected,
               completed_duration_seconds=(datetime.fromisoformat(manifest["finished"])-datetime.fromisoformat(manifest["started"])).total_seconds(),
               exact_all_fixed_root_gaps_zero=all(abs(r["fixed_root_gap"])<1e-8 for r in exact),
               counts=manifest["actual_counts"], frozen_analysis_reproduced=True,
               reproducibility_scope="full scalar physics, rule decisions, request histories, exact Q and aggregate statistics; planner inference not rerun")
write_json("independent_analysis.json", results)
print(json.dumps(dict(status="PASS", exact_all_fixed_root_gaps_zero=results["exact_all_fixed_root_gaps_zero"],
                     first_divergence=divergence, skill_mask_primary={k:v for k,v in contrasts.items() if v["primary"]}),indent=2), flush=True)
