"""Persistent-request, shared-technician maintenance factorial benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_coordination_references import marginal_gain, maximum_weight_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, observation

VERSION = "maintenance_shared_technician_factorial_v1"
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / f"{VERSION}_protocol.md"
REGISTRY = ROOT / "configs" / f"{VERSION}_seed_registry.json"
TIMINGS = ("reactive", "threshold", "capacity_aware")
DISPATCHERS = ("fcfs", "risk", "myopic", "lookahead")
CONDITIONS = ("scarce_overlap", "partitioned", "ample")
H, TAIL = 48, 12


def config(condition):
    if condition == "scarce_overlap":
        durations = ((1, 0), (1, 3), (1, 2), (1, 5))
    elif condition == "partitioned":
        durations = ((1, 0), (3, 0), (0, 1), (0, 3))
    elif condition == "ample":
        durations = ((1, 2, 3, 2), (2, 1, 2, 3), (3, 2, 1, 2), (2, 3, 2, 1))
    else:
        raise ValueError(condition)
    c = WaitingConfig(4, len(durations[0]), H + TAIL, 4, .45,
        durations, tuple(tuple(.75 for _ in row) for row in durations),
        waiting_limit=4, waiting_price=12.)
    c.validate()
    return c


def panel(profile):
    r = json.loads(REGISTRY.read_text())
    if profile == "smoke":
        return {"smoke": r["smoke"]}
    if profile == "full":
        return {name: list(range(r[name]["start"], r[name]["start"] + r[name]["count"]))
                for name in ("development", "sealed_test")}
    raise ValueError(profile)


def seed_audit():
    ours = json.loads(REGISTRY.read_text())
    groups = [set(ours["smoke"]),
              set(range(ours["development"]["start"], ours["development"]["start"] + ours["development"]["count"])),
              set(range(ours["sealed_test"]["start"], ours["sealed_test"]["start"] + ours["sealed_test"]["count"])),
              {ours["analysis"]}]
    if sum(map(len, groups)) != len(set().union(*groups)):
        raise RuntimeError("seed split overlap")
    historical = set()
    def ints(x):
        if isinstance(x, bool):
            return
        if isinstance(x, int):
            historical.add(x)
        elif isinstance(x, list):
            for item in x: ints(item)
        elif isinstance(x, dict):
            for item in x.values(): ints(item)
    for path in (ROOT / "configs").glob("*seed_registry.json"):
        if path != REGISTRY:
            ints(json.loads(path.read_text()))
    if set().union(*groups) & historical:
        raise RuntimeError("historical seed overlap")
    return {"status": "PASS", "panel_counts": list(map(len, groups)),
            "historical_raw_seeds": len(historical)}


def proposals(c, state, timing):
    if timing not in TIMINGS:
        raise ValueError(timing)
    new = []
    occupied = set(state.assigned)
    for m in range(c.machines):
        if m in occupied:
            continue
        if state.failed[m]:
            new.append(m)
        elif timing != "reactive":
            threshold = c.failure_age - 1
            if timing == "capacity_aware":
                compatible = [j for j, d in enumerate(c.service_time[m]) if d]
                if min(state.remaining[j] for j in compatible) >= 1:
                    threshold -= 1
            if state.ages[m] >= threshold:
                new.append(m)
    return new


def priority(c, state, m):
    return (100 if state.failed[m] else 0) + 12 * state.pending_wait[m] + state.ages[m] + (
        10 * c.failure_probability if state.ages[m] + 1 >= c.failure_age else 0)


def dispatch(c, state, pending, since, method, tick):
    if method not in DISPATCHERS:
        raise ValueError(method)
    idle = {j for j, rem in enumerate(state.remaining) if rem == 0}
    eligible = lambda m, j: j in idle and c.service_time[m][j] > 0
    if method in ("fcfs", "risk"):
        order = sorted(pending, key=(lambda m: (since[m], m)) if method == "fcfs"
                       else (lambda m: (-priority(c, state, m), since[m], m)))
        pairs = []
        for m in order:
            choices = [j for j in idle if eligible(m, j)]
            if choices:
                j = min(choices, key=lambda j: (c.service_time[m][j], j))
                pairs.append((m, j))
                idle.remove(j)
        return tuple(sorted(pairs))
    weights = [[-math.inf] * c.technicians for _ in range(c.machines)]
    obs = observation(c, state, tick) if method == "lookahead" else None
    for m in pending:
        for j in idle:
            if not eligible(m, j):
                continue
            d = c.service_time[m][j]
            if method == "lookahead":
                weights[m][j] = marginal_gain(obs, m, j)
            elif state.failed[m]:
                weights[m][j] = c.unavailable_cost + (
                    c.waiting_price if state.pending_wait[m] + 1 > c.waiting_limit else 0
                ) - c.corrective_cost - .25 * d
            else:
                imminent = state.ages[m] + 1 >= c.failure_age
                weights[m][j] = (c.failure_probability * (c.failure_cost + c.unavailable_cost)
                                  if imminent else 0) - c.preventive_cost - c.unavailable_cost * d
    return maximum_weight_matching(weights)


def terminal_liability(c, state, pending):
    cost = 0.
    for m in pending:
        d = min(x for x in c.service_time[m] if x)
        cost += (c.corrective_cost if state.failed[m] else c.preventive_cost) + c.unavailable_cost * d
        if state.failed[m]:
            cost += c.waiting_price * d
    cost += c.unavailable_cost * sum(state.remaining)
    return cost


def episode(condition, timing, method, seed, trace=None):
    c = config(condition)
    env = WaitingEnv(c)
    physical_seed = role_seed(seed, f"{VERSION}:{condition}:physical")
    env.reset(physical_seed)
    pending, active = set(), set()
    since = {}
    requests = starts = completions = 0
    waits = []
    policy_cost = policy_unavailable = policy_service = policy_failures = 0.
    policy_failed_wait = policy_overdue = 0.
    h_pending = h_failed = 0
    for tick in range(H + TAIL):
        before = env.state
        proposed = proposals(c, before, timing) if tick < H else [m for m in range(c.machines)
                                                                   if before.failed[m] and m not in before.assigned]
        for m in proposed:
            if m not in pending and m not in active:
                pending.add(m)
                since[m] = tick
                requests += 1
        pending_before = sorted(pending)
        method_now = method if tick < H else "fcfs"
        pairs = dispatch(c, before, pending, since, method_now, tick)
        validate_matching(c, before, pairs)
        if any(m not in pending for m, _ in pairs):
            raise RuntimeError("dispatch outside pending pool")
        for m, _ in pairs:
            pending.remove(m)
            active.add(m)
            waits.append(tick - since.pop(m))
            starts += 1
        _, _, _, info = env.step(pairs)
        after = env.state
        for m in tuple(active):
            if m not in after.assigned:
                active.remove(m)
                completions += 1
        # Failure converts an existing preventive request to corrective demand;
        # it never deletes or creates a duplicate request.
        for m in range(c.machines):
            if after.failed[m] and m not in pending and m not in active:
                pending.add(m)
                since[m] = tick + 1
                requests += 1
        if requests != len(pending) + len(active) + completions:
            raise RuntimeError("request conservation")
        if tick < H:
            policy_cost += info["objective"]
            policy_unavailable += info["unavailable_ticks"]
            policy_service += info["service_ticks"]
            policy_failures += info["failures"]
            policy_failed_wait += info["failed_waiting_ticks"]
            policy_overdue += info["overdue_waiting_ticks"]
        if tick == H - 1:
            h_pending = len(pending)
            h_failed = sum(after.failed)
        if trace is not None:
            trace.writerow(dict(condition=condition, timing=timing, dispatcher=method,
                seed=seed, physical_seed=physical_seed, tick=tick, phase="policy" if tick < H else "tail",
                state=json.dumps(asdict(before), separators=(',', ':')),
                proposals=json.dumps(proposed, separators=(',', ':')),
                pending_before=json.dumps(pending_before, separators=(',', ':')),
                assignment=json.dumps(pairs, separators=(',', ':')),
                following=json.dumps(asdict(after), separators=(',', ':')),
                cost=info["objective"], pending_after=len(pending), active_after=len(active)))
    if starts != completions + len(active):
        raise RuntimeError("start/completion reconciliation")
    liability = terminal_liability(c, env.state, pending)
    components = sum(env.metrics[k] for k in ("maintenance_cost", "unavailability_cost", "failure_cost", "waiting_cost"))
    if not math.isclose(components, env.metrics["objective"], abs_tol=1e-8):
        raise RuntimeError("component cost reconciliation")
    total = env.metrics["objective"] + liability
    if not math.isclose(total, policy_cost + (env.metrics["objective"] - policy_cost) + liability, abs_tol=1e-8):
        raise RuntimeError("horizon cost reconciliation")
    return dict(condition=condition, timing=timing, dispatcher=method, seed=seed,
        physical_seed=physical_seed, ticks=env.time, cost=total, policy_cost=policy_cost,
        tail_cost=env.metrics["objective"] - policy_cost, terminal_liability=liability,
        maintenance_cost=env.metrics["maintenance_cost"], unavailability_cost=env.metrics["unavailability_cost"],
        failure_cost=env.metrics["failure_cost"], waiting_cost=env.metrics["waiting_cost"],
        availability=1 - policy_unavailable / (c.machines * H),
        failures=env.metrics["failures"], policy_failures=policy_failures,
        failed_waiting_ticks=policy_failed_wait, overdue_ticks=policy_overdue,
        pending_h=h_pending, failed_h=h_failed, pending_end=len(pending),
        active_end=len(active), requests=requests, starts=starts, completions=completions,
        request_wait_mean=statistics.fmean(waits) if waits else 0.,
        request_wait_p95=float(np.quantile(waits, .95)) if waits else 0.,
        utilization=policy_service / (c.technicians * H),
        invalid_assignments=env.metrics["invalid_assignments"])


def audit_rows(rows, splits):
    expected = {(split, s, condition, timing, dispatcher)
                for split, seeds in splits.items() for s in seeds for condition in CONDITIONS
                for timing in TIMINGS for dispatcher in DISPATCHERS}
    actual = [(r["split"], r["seed"], r["condition"], r["timing"], r["dispatcher"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError("incomplete or duplicate panel")
    physical = {}
    for r in rows:
        key = (r["split"], r["seed"], r["condition"])
        expected_seed = role_seed(r["seed"], f"{VERSION}:{r['condition']}:physical")
        if r["physical_seed"] != expected_seed or (key in physical and physical[key] != r["physical_seed"]):
            raise RuntimeError("physical pairing")
        physical[key] = r["physical_seed"]
        if r["ticks"] != H + TAIL or r["invalid_assignments"] != 0:
            raise RuntimeError("physical completion or feasibility")
        if r["requests"] != r["completions"] + r["pending_end"] + r["active_end"]:
            raise RuntimeError("request conservation artifact")
        if not math.isclose(r["cost"], sum(r[k] for k in
                            ("maintenance_cost", "unavailability_cost", "failure_cost", "waiting_cost", "terminal_liability")), abs_tol=1e-8):
            raise RuntimeError("cost artifact")
    return {"status": "PASS", "episodes": len(rows), "unique_keys": len(expected),
            "seed_audit": seed_audit()}


def audit_decisions(path, expected_episodes):
    """Replay saved decisions from the independent physical shock stream."""
    episodes = ticks = 0
    previous = None
    pending = set()
    env = None
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["condition"], row["timing"], row["dispatcher"], int(row["seed"]))
            tick = int(row["tick"])
            if key != previous:
                if previous is not None and env.time != H + TAIL:
                    raise RuntimeError("incomplete decision trace")
                if tick != 0:
                    raise RuntimeError("trace does not begin at tick zero")
                env = WaitingEnv(config(key[0]))
                expected_seed = role_seed(key[3], f"{VERSION}:{key[0]}:physical")
                env.reset(expected_seed)
                pending = set()
                episodes += 1
                previous = key
            if tick != env.time or int(row["physical_seed"]) != expected_seed:
                raise RuntimeError("trace time or physical seed")
            if json.loads(row["state"]) != json.loads(json.dumps(asdict(env.state))):
                raise RuntimeError("trace state mismatch")
            proposed = set(json.loads(row["proposals"]))
            before = set(json.loads(row["pending_before"]))
            if before != pending | proposed:
                raise RuntimeError(f"erased or invented pending demand {key} tick={tick} before={before} expected={pending | proposed}")
            pairs = tuple(tuple(pair) for pair in json.loads(row["assignment"]))
            validate_matching(env.config, env.state, pairs)
            if not set(m for m, _ in pairs) <= before:
                raise RuntimeError("assignment without request")
            _, _, _, info = env.step(pairs)
            if json.loads(row["following"]) != json.loads(json.dumps(asdict(env.state))) or not math.isclose(
                    float(row["cost"]), info["objective"], abs_tol=1e-8):
                raise RuntimeError("physical transition mismatch")
            pending = before - {m for m, _ in pairs}
            pending |= {m for m, failed in enumerate(env.state.failed)
                        if failed and m not in env.state.assigned and m not in pending}
            if int(row["pending_after"]) != len(pending):
                raise RuntimeError("pending count mismatch")
            ticks += 1
    if episodes != expected_episodes or (env is not None and env.time != H + TAIL) or ticks != episodes * (H + TAIL):
        raise RuntimeError("incomplete trace panel")
    return {"status": "PASS", "episodes": episodes, "ticks": ticks}


def analysis(rows, split):
    selected = [r for r in rows if r["split"] == split]
    by = {(r["seed"], r["condition"], r["timing"], r["dispatcher"]): r for r in selected}
    seeds = sorted({r["seed"] for r in selected})
    result = {"split": split, "seed_count": len(seeds), "conditions": {}}
    rng = np.random.default_rng(role_seed(91000400, f"{VERSION}:bootstrap"))
    for condition in CONDITIONS:
        means = {f"{t}/{d}": statistics.fmean(by[s, condition, t, d]["cost"] for s in seeds)
                 for t in TIMINGS for d in DISPATCHERS}
        metrics = {f"{t}/{d}": {key: statistics.fmean(by[s, condition, t, d][key] for s in seeds)
                  for key in ("availability", "failures", "failed_waiting_ticks", "overdue_ticks",
                              "pending_h", "pending_end", "utilization", "request_wait_mean", "terminal_liability")}
                  for t in TIMINGS for d in DISPATCHERS}
        contrasts = {}
        definitions = {
            "allocation": (("capacity_aware", "lookahead"), ("capacity_aware", "fcfs")),
            "request": (("capacity_aware", "lookahead"), ("threshold", "lookahead")),
            "joint_vs_separate": (("capacity_aware", "lookahead"), ("threshold", "fcfs")),
            "timing_under_fcfs": (("capacity_aware", "fcfs"), ("threshold", "fcfs")),
            "allocation_under_threshold": (("threshold", "lookahead"), ("threshold", "fcfs")),
        }
        draws = rng.integers(0, len(seeds), size=(10000, len(seeds))) if split == "sealed_test" else None
        for name, (a, b) in definitions.items():
            diff = np.asarray([by[s, condition, *a]["cost"] - by[s, condition, *b]["cost"] for s in seeds])
            base = means[f"{b[0]}/{b[1]}"]
            interval = np.quantile(diff[draws].mean(1), [.00625, .99375]).tolist() if draws is not None else None
            contrasts[name] = {"mean_difference": float(diff.mean()), "relative_reduction": float(-diff.mean()/base),
                "ci_9875": interval, "paired_standardized_effect": float(diff.mean()/diff.std(ddof=1)) if len(diff)>1 and diff.std(ddof=1)>0 else None,
                "win_fraction": float(np.mean(diff < 0))}
        interaction = np.asarray([by[s, condition, "capacity_aware", "lookahead"]["cost"]
            - by[s, condition, "capacity_aware", "fcfs"]["cost"]
            - by[s, condition, "threshold", "lookahead"]["cost"]
            + by[s, condition, "threshold", "fcfs"]["cost"] for s in seeds])
        contrasts["interaction"] = {"mean_difference": float(interaction.mean()),
            "fraction_of_reference": float(interaction.mean()/means["threshold/fcfs"]),
            "ci_9875": np.quantile(interaction[draws].mean(1), [.00625,.99375]).tolist() if draws is not None else None}
        result["conditions"][condition] = {"mean_cost": means, "metrics": metrics, "contrasts": contrasts}
    if split == "sealed_test":
        c = result["conditions"]["scarce_overlap"]["contrasts"]
        h1 = all(c[k]["relative_reduction"] >= .05 and c[k]["ci_9875"][1] < 0 for k in ("allocation", "request"))
        h2 = c["interaction"]["fraction_of_reference"] <= -.02 and c["interaction"]["ci_9875"][1] < 0
        result["hypotheses"] = {"H1": "SUPPORTED" if h1 else "NOT_SUPPORTED",
                                "H2": "SUPPORTED" if h2 else "NOT_SUPPORTED",
                                "H3": "DESCRIPTIVE"}
    return result


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def run(profile, output_dir):
    if profile == "full" and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise RuntimeError("full requires clean git checkout")
    splits = panel(profile)
    audit = seed_audit()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_snapshot").mkdir()
    for source in (PROTOCOL, REGISTRY, Path(__file__)):
        shutil.copy2(source, output / "source_snapshot" / source.name)
    started = time.monotonic()
    manifest = {"version": VERSION, "profile": profile, "status": "RUNNING", "git_commit": git_revision(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "seeds": splits,
        "training_seeds": [], "expected_episodes": sum(map(len, splits.values())) * len(CONDITIONS)*len(TIMINGS)*len(DISPATCHERS),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROTOCOL, REGISTRY, Path(__file__))},
        "seed_audit": audit, "device": "cpu", "max_seconds": 7200, "completed_episodes": 0}
    write_json(output / "manifest.json", manifest)
    write_json(output / "resolved_config.json", {name: asdict(config(name)) for name in CONDITIONS})
    rows = []
    partial = output / "episodes.partial.csv"
    trace_path = output / "decisions.csv"
    try:
        with partial.open("w", newline="") as ep_file, trace_path.open("w", newline="") as tr_file:
            ep_writer = None
            trace_fields = ("condition", "timing", "dispatcher", "seed", "physical_seed", "tick", "phase",
                            "state", "proposals", "pending_before", "assignment", "following", "cost", "pending_after", "active_after")
            tr_writer = csv.DictWriter(tr_file, fieldnames=trace_fields)
            tr_writer.writeheader()
            jobs = [(split, seed, condition, timing, dispatcher)
                    for split, seeds in splits.items() for seed in seeds for condition in CONDITIONS
                    for timing in TIMINGS for dispatcher in DISPATCHERS]
            for split, seed, condition, timing, dispatcher in tqdm(jobs, desc=f"{profile} episodes", unit="episode"):
                if time.monotonic() - started > 7200:
                    raise TimeoutError("locked 7200-second stopping rule")
                row = {"split": split, **episode(condition, timing, dispatcher, seed,
                    tr_writer if split in ("sealed_test", "smoke") else None)}
                if ep_writer is None:
                    ep_writer = csv.DictWriter(ep_file, fieldnames=list(row))
                    ep_writer.writeheader()
                ep_writer.writerow(row)
                rows.append(row)
                if len(rows) % 12 == 0:
                    ep_file.flush(); tr_file.flush()
                    manifest["completed_episodes"] = len(rows)
                    write_json(output / "manifest.json", manifest)
        integrity = audit_rows(rows, splits)
        integrity["decision_replay"] = audit_decisions(
            trace_path, sum(len(v) for k, v in splits.items() if k in ("sealed_test", "smoke"))
            * len(CONDITIONS) * len(TIMINGS) * len(DISPATCHERS))
        write_json(output / "integrity.json", integrity)
        shutil.copy2(partial, output / "episodes.csv")
        summaries = {split: analysis(rows, split) for split in splits}
        write_json(output / "summary.json", {"status": "ENGINEERING_ONLY" if profile == "smoke" else "COMPLETED",
                    "panels": summaries})
        manifest.update(status="COMPLETED", completed_episodes=len(rows),
                        duration_seconds=time.monotonic()-started,
                        completed_utc=datetime.now(timezone.utc).isoformat(), integrity="PASS",
                        files=[p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()])
        write_json(output / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="FAILED", completed_episodes=len(rows), error=repr(exc),
                        duration_seconds=time.monotonic()-started)
        write_json(output / "manifest.json", manifest)
        raise
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(run(args.profile, args.output_dir))


if __name__ == "__main__":
    main()
