"""Bounded joint-resource forecast against frozen simple maintenance dispatchers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_shared_technician_factorial as old
from ht_pdm_fjsp.maintenance_dispatch import matching_actions, role_seed, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, transition

VERSION = "maintenance_coupled_dispatch_followup_v1"
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / f"{VERSION}_protocol.md"
REGISTRY = ROOT / "configs" / f"{VERSION}_seed_registry.json"
H, TAIL = old.H, old.TAIL
TIMINGS = ("threshold", "capacity_aware")
SIMPLE = ("fcfs", "risk", "myopic")
DISPATCHERS = (*SIMPLE, "isolated", "coupled_rollout")
MAX_CANDIDATES = 10
FULL_SCENARIOS = 4
SMOKE_SCENARIOS = 2
DIAGNOSTIC_SCENARIOS = 16
MAX_SECONDS = 7200
TRACE_FIELDS = ("condition", "timing", "dispatcher", "seed", "physical_seed", "tick",
    "phase", "state", "proposals", "pending_before", "assignment", "following",
    "cost", "pending_after", "active_after")


def utc():
    return datetime.now(timezone.utc).isoformat()


def registry():
    return json.loads(REGISTRY.read_text())


def matrix_config(config_id):
    r = registry()
    matrices = {**r["development_configs"], **r["sealed_test_configs"]}
    if config_id not in matrices:
        raise ValueError(config_id)
    durations = tuple(tuple(int(x) for x in row) for row in matrices[config_id])
    c = WaitingConfig(4, len(durations[0]), H + TAIL, 4, .45, durations,
        tuple(tuple(.75 for _ in row) for row in durations), waiting_limit=4,
        waiting_price=12.)
    c.validate()
    return c


def panels(profile):
    r = registry()
    if profile == "smoke":
        return {"smoke": {name: list(r["smoke"]) for name in ("hub_dev", "balanced_dev")}}
    if profile == "full":
        return {
            "development": {name: list(range(r["development"]["start"],
                r["development"]["start"] + r["development"]["count"]))
                for name in r["development_configs"]},
            "sealed_test": {name: list(range(r["sealed_test"]["start"],
                r["sealed_test"]["start"] + r["sealed_test"]["count"]))
                for name in r["sealed_test_configs"]},
        }
    raise ValueError(profile)


def seed_audit():
    r = registry()
    own = [set(r["smoke"]),
        set(range(r["development"]["start"], r["development"]["start"] + r["development"]["count"])),
        set(range(r["sealed_test"]["start"], r["sealed_test"]["start"] + r["sealed_test"]["count"])),
        {r["analysis"]}]
    if len(set().union(*own)) != sum(map(len, own)):
        raise RuntimeError("own seed overlap")
    historical = set()
    def gather(x):
        if isinstance(x, bool):
            return
        if isinstance(x, int):
            historical.add(x)
        elif isinstance(x, dict):
            for v in x.values(): gather(v)
        elif isinstance(x, list):
            for v in x: gather(v)
    for path in (ROOT / "configs").glob("*seed_registry.json"):
        if path != REGISTRY:
            gather(json.loads(path.read_text()))
    if set().union(*own) & historical:
        raise RuntimeError("historical seed overlap")
    if set(r["development_configs"]) & set(r["sealed_test_configs"]):
        raise RuntimeError("configuration ID overlap")
    if len({json.dumps(x) for x in (*r["development_configs"].values(),
                                 *r["sealed_test_configs"].values())}) != 12:
        raise RuntimeError("duplicate physical matrix")
    return dict(status="PASS", raw_panel_counts=list(map(len, own)),
        historical_raw_count=len(historical), development_configs=list(r["development_configs"]),
        test_configs=list(r["sealed_test_configs"]))


def physical_seed(config_id, raw_seed):
    return role_seed(raw_seed, f"{VERSION}:{config_id}:physical")


def scenario_seed(config_id, raw_seed, tick, scenario_index, bank):
    if bank not in ("decision", "diagnostic"):
        raise ValueError(bank)
    return role_seed(raw_seed, f"{VERSION}:{config_id}:{bank}:{tick}:{scenario_index}")


def scenario_tapes(c, config_id, raw_seed, tick, bank, count):
    tapes = []
    for i in range(count):
        rng = random.Random(scenario_seed(config_id, raw_seed, tick, i, bank))
        tapes.append(tuple(tuple(rng.random() < c.failure_probability
            for _ in range(c.machines)) for _ in range(tick, H + TAIL)))
    return tuple(tapes)


def add_requests(c, state, pending, since, timing, tick):
    proposed = old.proposals(c, state, timing) if tick < H else [m for m in range(c.machines)
        if state.failed[m] and m not in state.assigned]
    for m in proposed:
        if m not in pending and m not in state.assigned:
            pending.add(m)
            since[m] = tick
    return proposed


def forecast_step(c, state, pending, since, action, events, tick):
    """One exact physical/queue update; caller already added this tick's requests."""
    validate_matching(c, state, action)
    if any(m not in pending for m, _ in action):
        raise ValueError("forecast assignment without pending request")
    following, cost, info = transition(c, state, action, events)
    next_pending = set(pending) - {m for m, _ in action}
    next_since = {m: since[m] for m in next_pending}
    for m, failed in enumerate(following.failed):
        if failed and m not in next_pending and m not in following.assigned:
            next_pending.add(m)
            next_since[m] = tick + 1
    return following, next_pending, next_since, cost, info


def forecast_total(c, state, pending, since, timing, continuation, tick, action, tape):
    """Same H+tail+liability objective, with future shocks from a separate bank."""
    if len(tape) != H + TAIL - tick:
        raise ValueError("forecast tape length")
    state, pending, since, cost, _ = forecast_step(c, state, pending, since, action,
        tape[0], tick)
    for future_tick in range(tick + 1, H + TAIL):
        add_requests(c, state, pending, since, timing, future_tick)
        rule = continuation if future_tick < H else "fcfs"
        next_action = old.dispatch(c, state, pending, since, rule, future_tick)
        state, pending, since, amount, _ = forecast_step(c, state, pending, since,
            next_action, tape[future_tick - tick], future_tick)
        cost += amount
    return cost + old.terminal_liability(c, state, pending)


def candidate_actions(c, state, pending, since, tick, continuation):
    baseline = old.dispatch(c, state, pending, since, continuation, tick)
    actions = [baseline, ()]
    for rule in (*SIMPLE, "lookahead"):
        actions.append(old.dispatch(c, state, pending, since, rule, tick))
    legal = [action for action in matching_actions(c, state)
             if all(m in pending for m, _ in action)]
    def score(action):
        return sum(old.priority(c, state, m) + 2 * c.restoration[m][j] / c.service_time[m][j]
                   for m, j in action)
    legal.sort(key=lambda action: (-score(action), action))
    actions.extend(legal)
    unique = []
    for action in actions:
        if action not in unique:
            unique.append(action)
        if len(unique) == MAX_CANDIDATES:
            break
    for action in unique:
        validate_matching(c, state, action)
        if any(m not in pending for m, _ in action):
            raise RuntimeError("candidate outside demand")
    return baseline, tuple(unique)


def choose_coupled(c, config_id, raw_seed, state, pending, since, timing, tick,
                   continuation, scenario_count):
    baseline, candidates = candidate_actions(c, state, pending, since, tick, continuation)
    if not pending or len(candidates) == 1:
        return baseline, dict(candidate_count=len(candidates), trajectories=0,
                              selected_estimate=None, baseline_estimate=None)
    tapes = scenario_tapes(c, config_id, raw_seed, tick, "decision", scenario_count)
    scores = {action: statistics.fmean(forecast_total(c, state, pending, since,
        timing, continuation, tick, action, tape) for tape in tapes)
        for action in candidates}
    chosen = min(candidates, key=lambda action: (round(scores[action], 9),
        action != baseline, action))
    return chosen, dict(candidate_count=len(candidates),
        trajectories=len(candidates) * len(tapes), selected_estimate=scores[chosen],
        baseline_estimate=scores[baseline])


def run_episode(config_id, timing, dispatcher, raw_seed, split, continuation,
                scenario_count, trace_writer=None, diagnostic_writer=None):
    if timing not in TIMINGS or dispatcher not in DISPATCHERS:
        raise ValueError("unregistered arm")
    c = matrix_config(config_id)
    env = WaitingEnv(c)
    seed = physical_seed(config_id, raw_seed)
    env.reset(seed)
    pending, active, since = set(), set(), {}
    requests = starts = completions = 0
    waits = []
    h_cost = h_unavailable = h_service = h_failures = h_failed_wait = h_overdue = 0.
    h_pending = h_failed = 0
    dispatch_seconds = 0.
    forecast_calls = trajectory_count = candidate_total = changed_from_base = 0
    diagnostic_done = False
    for tick in range(H + TAIL):
        before = env.state
        previous_pending = len(pending)
        proposed = add_requests(c, before, pending, since, timing, tick)
        requests += len(pending) - previous_pending
        for m in proposed:
            # The request pool contains all new proposals after add_requests;
            # `since` entries are created once and persist until assignment.
            if m not in pending and m not in active and m not in before.assigned:
                raise RuntimeError("request vanished before dispatch")
        pending_before = tuple(sorted(pending))
        if tick < H:
            start = time.perf_counter()
            if dispatcher == "coupled_rollout":
                action, forecast = choose_coupled(c, config_id, raw_seed, before,
                    pending, since, timing, tick, continuation, scenario_count)
                forecast_calls += forecast["trajectories"] > 0
                trajectory_count += forecast["trajectories"]
                candidate_total += forecast["candidate_count"]
                baseline = old.dispatch(c, before, pending, since, continuation, tick)
                changed_from_base += action != baseline
            else:
                action = old.dispatch(c, before, pending, since,
                    "lookahead" if dispatcher == "isolated" else dispatcher, tick)
                forecast = None
            dispatch_seconds += time.perf_counter() - start
            if (dispatcher == "coupled_rollout" and split == "sealed_test"
                    and diagnostic_writer is not None and not diagnostic_done
                    and forecast["trajectories"] > 0):
                diagnostic_done = True
                bank = scenario_tapes(c, config_id, raw_seed, tick, "diagnostic",
                                      DIAGNOSTIC_SCENARIOS)
                chosen_values = [forecast_total(c, before, pending, since, timing,
                    continuation, tick, action, tape) for tape in bank]
                baseline_values = [forecast_total(c, before, pending, since, timing,
                    continuation, tick, baseline, tape) for tape in bank]
                diff = np.asarray(chosen_values) - np.asarray(baseline_values)
                diagnostic_writer.writerow(dict(condition=config_id, timing=timing,
                    seed=raw_seed, tick=tick, pending=len(pending),
                    chosen=json.dumps(action), baseline=json.dumps(baseline),
                    decision_estimate_delta=forecast["selected_estimate"]-forecast["baseline_estimate"],
                    independent_mean_delta=float(diff.mean()),
                    independent_sd_delta=float(diff.std(ddof=1)),
                    scenarios=DIAGNOSTIC_SCENARIOS))
        else:
            action = old.dispatch(c, before, pending, since, "fcfs", tick)
        validate_matching(c, before, action)
        if any(m not in pending for m, _ in action):
            raise RuntimeError("assignment without pending request")
        for m, _ in action:
            pending.remove(m)
            active.add(m)
            waits.append(tick - since.pop(m))
            starts += 1
        _, _, _, info = env.step(action)
        for m in tuple(active):
            if m not in env.state.assigned:
                active.remove(m)
                completions += 1
        for m, failed in enumerate(env.state.failed):
            if failed and m not in pending and m not in active:
                pending.add(m)
                since[m] = tick + 1
                requests += 1
        if requests != completions + len(pending) + len(active):
            raise RuntimeError("request conservation")
        if tick < H:
            h_cost += info["objective"]
            h_unavailable += info["unavailable_ticks"]
            h_service += info["service_ticks"]
            h_failures += info["failures"]
            h_failed_wait += info["failed_waiting_ticks"]
            h_overdue += info["overdue_waiting_ticks"]
        if tick == H-1:
            h_pending, h_failed = len(pending), sum(env.state.failed)
        if trace_writer is not None:
            trace_writer.writerow(dict(condition=config_id, timing=timing,
                dispatcher=dispatcher, seed=raw_seed, physical_seed=seed,
                tick=tick, phase="policy" if tick < H else "tail",
                state=json.dumps(asdict(before), separators=(',', ':')),
                proposals=json.dumps(proposed, separators=(',', ':')),
                pending_before=json.dumps(pending_before, separators=(',', ':')),
                assignment=json.dumps(action, separators=(',', ':')),
                following=json.dumps(asdict(env.state), separators=(',', ':')),
                cost=info["objective"], pending_after=len(pending), active_after=len(active)))
    if starts != completions + len(active):
        raise RuntimeError("completion count")
    liability = old.terminal_liability(c, env.state, pending)
    physical_cost = env.metrics["objective"]
    total = physical_cost + liability
    if not math.isclose(total, h_cost + (physical_cost-h_cost) + liability, abs_tol=1e-8):
        raise RuntimeError("phase cost")
    if not math.isclose(physical_cost, sum(env.metrics[k] for k in
            ("maintenance_cost", "unavailability_cost", "failure_cost", "waiting_cost")), abs_tol=1e-8):
        raise RuntimeError("component cost")
    return dict(split=split, condition=config_id, timing=timing, dispatcher=dispatcher,
        seed=raw_seed, physical_seed=seed, ticks=env.time, cost=total,
        policy_cost=h_cost, tail_cost=physical_cost-h_cost, terminal_liability=liability,
        maintenance_cost=env.metrics["maintenance_cost"],
        unavailability_cost=env.metrics["unavailability_cost"],
        failure_cost=env.metrics["failure_cost"], waiting_cost=env.metrics["waiting_cost"],
        availability=1-h_unavailable/(c.machines*H), failures=env.metrics["failures"],
        policy_failures=h_failures, failed_waiting_ticks=h_failed_wait,
        overdue_ticks=h_overdue, pending_h=h_pending, failed_h=h_failed,
        pending_end=len(pending), active_end=len(active), requests=requests,
        starts=starts, completions=completions,
        request_wait_mean=statistics.fmean(waits) if waits else 0.,
        request_wait_p95=float(np.quantile(waits,.95)) if waits else 0.,
        utilization=h_service/(c.technicians*H),
        invalid_assignments=env.metrics["invalid_assignments"],
        dispatch_seconds=dispatch_seconds, forecast_calls=forecast_calls,
        scenario_trajectories=trajectory_count, candidate_total=candidate_total,
        changed_from_base=changed_from_base)


def select_baseline(rows, config_ids, seeds):
    expected = {(c, s, t, d) for c in config_ids for s in seeds for t in TIMINGS
                for d in SIMPLE}
    lookup = {(r["condition"], r["seed"], r["timing"], r["dispatcher"]): r
              for r in rows if r["dispatcher"] in SIMPLE}
    if set(lookup) != expected:
        raise RuntimeError("incomplete baseline selection panel")
    means = {d: statistics.fmean(lookup[c,s,t,d]["cost"] for c in config_ids
              for s in seeds for t in TIMINGS) for d in SIMPLE}
    chosen = min(SIMPLE, key=lambda d: (means[d], SIMPLE.index(d)))
    return dict(selected=chosen, means=means, rule="development equal-weight config/shock/timing mean; FCFS-risk-myopic tie order",
                configurations=list(config_ids), shock_seeds=list(seeds),
                test_opened=False, selected_utc=utc())


def development_activation(trace_path, rows, config_ids, seeds, selected):
    indexed = {}
    with Path(trace_path).open(newline="") as stream:
        for r in csv.DictReader(stream):
            if int(r["tick"]) < H and r["dispatcher"] == selected:
                indexed[r["condition"], int(r["seed"]), r["timing"], int(r["tick"])] = r
    by_config = {}
    for c in config_ids:
        proposal_diff = assignment_diff = contention = 0
        for seed in seeds:
            for tick in range(H):
                a = indexed[c,seed,"threshold",tick]
                b = indexed[c,seed,"capacity_aware",tick]
                proposal_diff += a["proposals"] != b["proposals"]
                assignment_diff += a["assignment"] != b["assignment"]
                state = json.loads(a["state"])
                pending = json.loads(a["pending_before"])
                free = sum(r == 0 for r in state["remaining"])
                contention += len(pending) > free
        coupled_changes = sum(r["changed_from_base"] for r in rows
            if r["condition"] == c and r["dispatcher"] == "coupled_rollout")
        by_config[c] = dict(policy_ticks=len(seeds)*H, proposal_diff_ticks=proposal_diff,
            assignment_diff_ticks=assignment_diff, contention_ticks=contention,
            coupled_vs_selected_action_changes=coupled_changes)
    return dict(by_config=by_config,
        configs_with_proposal_difference=sum(x["proposal_diff_ticks"]>0 for x in by_config.values()),
        configs_with_contention=sum(x["contention_ticks"]>0 for x in by_config.values()),
        configs_with_coupled_change=sum(x["coupled_vs_selected_action_changes"]>0 for x in by_config.values()),
        no_configuration_filtered=True)


def audit_panel(rows, panel_map, selected, trace_results):
    expected = {(split,c,s,t,d) for split,configs in panel_map.items()
                for c,seeds in configs.items() for s in seeds for t in TIMINGS for d in DISPATCHERS}
    keys = [(r["split"],r["condition"],r["seed"],r["timing"],r["dispatcher"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise RuntimeError("incomplete/duplicate full panel")
    for r in rows:
        if r["physical_seed"] != physical_seed(r["condition"],r["seed"]):
            raise RuntimeError("physical seed mapping")
        if r["ticks"] != H+TAIL or r["invalid_assignments"] != 0:
            raise RuntimeError("physical completion/feasibility")
        if r["requests"] != r["completions"]+r["pending_end"]+r["active_end"]:
            raise RuntimeError("request conservation artifact")
        if not math.isclose(r["cost"],sum(r[k] for k in
            ("maintenance_cost","unavailability_cost","failure_cost","waiting_cost","terminal_liability")),abs_tol=1e-8):
            raise RuntimeError("component cost artifact")
        if not math.isclose(r["cost"],r["policy_cost"]+r["tail_cost"]+r["terminal_liability"],abs_tol=1e-8):
            raise RuntimeError("phase cost artifact")
    if selected not in SIMPLE:
        raise RuntimeError("unselected baseline")
    if any(result["status"] != "PASS" for result in trace_results.values()):
        raise RuntimeError("trace replay")
    return dict(status="PASS", unique_episodes=len(expected), trace_replay=trace_results,
        seed_audit=seed_audit(), selected_baseline=selected)


def summarize(rows, test_configs, test_seeds, selected, analysis_seed):
    data = {(r["condition"],r["seed"],r["timing"],r["dispatcher"]):r for r in rows
            if r["split"] == "sealed_test"}
    expected = {(c,s,t,d) for c in test_configs for s in test_seeds for t in TIMINGS
                for d in DISPATCHERS}
    if set(data) != expected:
        raise RuntimeError("test panel missing")
    metrics = ("cost","policy_cost","tail_cost","terminal_liability",
        "maintenance_cost","unavailability_cost","failure_cost","waiting_cost",
        "availability","failures","failed_waiting_ticks","overdue_ticks",
        "pending_h","pending_end","utilization","requests","starts","completions",
        "dispatch_seconds","forecast_calls","scenario_trajectories","changed_from_base")
    by_config = {}
    for c in test_configs:
        by_config[c] = {f"{t}/{d}": {m:statistics.fmean(data[c,s,t,d][m] for s in test_seeds)
            for m in metrics} for t in TIMINGS for d in DISPATCHERS}
    rng = np.random.default_rng(role_seed(analysis_seed,f"{VERSION}:hierarchical_bootstrap"))
    config_draws = rng.integers(0,len(test_configs),(10000,len(test_configs)))
    shock_draws = rng.integers(0,len(test_seeds),(10000,len(test_configs),len(test_seeds)))
    fixed_config_draws = np.broadcast_to(np.arange(len(test_configs)),config_draws.shape)
    definitions = {
        "A_allocation_capacity_aware": (("capacity_aware","coupled_rollout"),
                                         ("capacity_aware",selected), .03),
        "B_timing_selected_simple": (("capacity_aware",selected),
                                     ("threshold",selected), .03),
        "C_interaction": (None,("threshold",selected), .02),
    }
    contrasts = {}
    for name,(arm,reference,margin) in definitions.items():
        matrix = np.asarray([[ (
            data[c,s,"capacity_aware","coupled_rollout"]["cost"]
            - data[c,s,"capacity_aware",selected]["cost"]
            - data[c,s,"threshold","coupled_rollout"]["cost"]
            + data[c,s,"threshold",selected]["cost"])
            if arm is None else data[c,s,*arm]["cost"]-data[c,s,*reference]["cost"]
            for s in test_seeds] for c in test_configs],dtype=float)
        base = statistics.fmean(data[c,s,*reference]["cost"] for c in test_configs for s in test_seeds)
        mean = float(matrix.mean())
        boot = matrix[config_draws[:,:,None],shock_draws].mean((1,2))
        shock_only = matrix[fixed_config_draws[:,:,None],shock_draws].mean((1,2))
        ci = np.quantile(boot,[1/120,119/120]).tolist()  # 98.33% two-sided
        per_config = {c:float(matrix[i].mean()) for i,c in enumerate(test_configs)}
        config_wins = sum(x<0 for x in per_config.values())
        practical = margin * base
        supported = mean <= -practical and ci[1] < 0 and config_wins >= 4
        status = "SUPPORTED" if supported else (
            "INCONCLUSIVE" if ci[0] <= -practical and ci[1] >= 0 else "NOT_SUPPORTED")
        sd = float(matrix.mean(1).std(ddof=1))
        contrasts[name] = dict(status=status, mean_difference=mean,
            relative_reduction=-mean/base, baseline_mean=base, practical_margin=practical,
            ci_9833_hierarchical=ci,
            ci_95_shock_only=np.quantile(shock_only,[.025,.975]).tolist(),
            paired_config_standardized_effect=mean/sd if sd>0 else None,
            config_wins=config_wins, per_config_difference=per_config)
    runtime = {d:dict(total_dispatch_seconds=sum(r["dispatch_seconds"] for r in data.values()
            if r["dispatcher"]==d),
            mean_dispatch_seconds_per_episode=statistics.fmean(r["dispatch_seconds"] for r in data.values()
            if r["dispatcher"]==d)) for d in DISPATCHERS}
    return dict(status="COMPLETED", selected_simple=selected, test_configs=list(test_configs),
        test_shock_seeds=list(test_seeds), all_arms_by_config=by_config,
        primary_contrasts=contrasts, runtime=runtime,
        inference_unit="configuration; shocks nested within configuration")


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")


def git_revision():
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()


def run(profile, output_dir):
    if profile == "full" and subprocess.check_output(["git","status","--porcelain"],cwd=ROOT):
        raise RuntimeError("full requires clean Git checkout")
    panel_map = panels(profile)
    audit = seed_audit()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    source_paths = (PROTOCOL,REGISTRY,Path(__file__),Path(old.__file__),
        ROOT/"src/ht_pdm_fjsp/maintenance_waiting.py",
        ROOT/"src/ht_pdm_fjsp/maintenance_dispatch.py",
        ROOT/"src/ht_pdm_fjsp/maintenance_coordination_references.py")
    (output/"source_snapshot").mkdir()
    for source in source_paths:
        shutil.copy2(source,output/"source_snapshot"/source.name)
    expected = sum(len(seeds) for configs in panel_map.values() for seeds in configs.values())*len(TIMINGS)*len(DISPATCHERS)
    scenario_count = FULL_SCENARIOS if profile=="full" else SMOKE_SCENARIOS
    manifest = dict(version=VERSION,profile=profile,status="RUNNING",started_utc=utc(),
        git_commit=git_revision(),expected_episodes=expected,completed_episodes=0,
        panels=panel_map,training_seeds=[],scenario_count=scenario_count,
        diagnostic_scenarios=DIAGNOSTIC_SCENARIOS,device="cpu",max_seconds=MAX_SECONDS,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        seed_audit=audit,selected_baseline=None,test_opened_utc=None)
    write_json(output/"manifest.json",manifest)
    all_configs={name:asdict(matrix_config(name)) for name in
        (*registry()["development_configs"],*registry()["sealed_test_configs"])}
    write_json(output/"resolved_config.json",all_configs)
    rows=[]
    started=time.monotonic()
    try:
        with (output/"episodes.partial.csv").open("w",newline="") as ep_file, \
             (output/"development_decisions.csv").open("w",newline="") as dev_trace_file, \
             (output/"decisions.csv").open("w",newline="") as test_trace_file, \
             (output/"forecast_diagnostics.csv").open("w",newline="") as diag_file:
            ep_writer=None
            dev_writer=csv.DictWriter(dev_trace_file,fieldnames=TRACE_FIELDS);dev_writer.writeheader()
            test_writer=csv.DictWriter(test_trace_file,fieldnames=TRACE_FIELDS);test_writer.writeheader()
            diag_fields=("condition","timing","seed","tick","pending","chosen","baseline",
                "decision_estimate_delta","independent_mean_delta","independent_sd_delta","scenarios")
            diag_writer=csv.DictWriter(diag_file,fieldnames=diag_fields);diag_writer.writeheader()
            def append(row):
                nonlocal ep_writer
                if time.monotonic()-started>MAX_SECONDS:
                    raise TimeoutError("locked 7200-second stopping rule")
                if ep_writer is None:
                    ep_writer=csv.DictWriter(ep_file,fieldnames=tuple(row));ep_writer.writeheader()
                ep_writer.writerow(row)
                rows.append(row)
                if len(rows)%10==0:
                    ep_file.flush();dev_trace_file.flush();test_trace_file.flush();diag_file.flush()
                    manifest["completed_episodes"]=len(rows)
                    write_json(output/"manifest.json",manifest)
            with tqdm(total=expected,desc=f"{profile} episodes",unit="episode") as bar:
                stage="development" if profile=="full" else "smoke"
                configs=panel_map[stage]
                for config_id,seeds in configs.items():
                    for raw_seed in seeds:
                        for timing in TIMINGS:
                            for dispatcher in (*SIMPLE,"isolated"):
                                row=run_episode(config_id,timing,dispatcher,raw_seed,stage,"fcfs",
                                    scenario_count,dev_writer if profile=="full" else test_writer)
                                append(row);bar.update(1)
                selection=select_baseline(rows,tuple(configs),tuple(next(iter(configs.values()))))
                write_json(output/"selection.json",selection)
                manifest["selected_baseline"]=selection["selected"]
                manifest["selection_utc"]=selection["selected_utc"]
                write_json(output/"manifest.json",manifest)
                for config_id,seeds in configs.items():
                    for raw_seed in seeds:
                        for timing in TIMINGS:
                            row=run_episode(config_id,timing,"coupled_rollout",raw_seed,stage,
                                selection["selected"],scenario_count,
                                dev_writer if profile=="full" else test_writer)
                            append(row);bar.update(1)
                dev_trace_file.flush();test_trace_file.flush()
                activation=development_activation(
                    output/("development_decisions.csv" if profile=="full" else "decisions.csv"),
                    rows,tuple(configs),tuple(next(iter(configs.values()))),selection["selected"])
                write_json(output/"development_activation.json",activation)
                if profile=="full":
                    manifest["test_opened_utc"]=utc()
                    write_json(output/"manifest.json",manifest)
                    for config_id,seeds in panel_map["sealed_test"].items():
                        for raw_seed in seeds:
                            for timing in TIMINGS:
                                for dispatcher in DISPATCHERS:
                                    row=run_episode(config_id,timing,dispatcher,raw_seed,"sealed_test",
                                        selection["selected"],scenario_count,test_writer,diag_writer)
                                    append(row);bar.update(1)
        trace_results={
            "development":old.audit_decisions(output/"development_decisions.csv",
                len(panel_map.get("development",{}))*8*len(TIMINGS)*len(DISPATCHERS) if profile=="full" else 0,
                config_fn=matrix_config,version=VERSION),
            "test_or_smoke":old.audit_decisions(output/"decisions.csv",
                (len(panel_map.get("sealed_test",{}))*16 if profile=="full" else
                 len(panel_map["smoke"]))*len(TIMINGS)*len(DISPATCHERS),
                config_fn=matrix_config,version=VERSION)}
        integrity=audit_panel(rows,panel_map,selection["selected"],trace_results)
        if profile=="full" and not (manifest["selection_utc"]<manifest["test_opened_utc"]):
            raise RuntimeError("test opened before baseline selection")
        write_json(output/"integrity.json",integrity)
        shutil.copy2(output/"episodes.partial.csv",output/"episodes.csv")
        with (output/"development_episodes.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=tuple(rows[0]));writer.writeheader()
            writer.writerows(r for r in rows if r["split"]!="sealed_test")
        summary=(summarize(rows,tuple(panel_map["sealed_test"]),
            tuple(next(iter(panel_map["sealed_test"].values()))),
            selection["selected"],registry()["analysis"]) if profile=="full" else
            dict(status="ENGINEERING_ONLY",episodes=len(rows),selected_simple=selection["selected"]))
        write_json(output/"summary.json",summary)
        manifest.update(status="COMPLETED",completed_episodes=len(rows),
            completed_utc=utc(),duration_seconds=time.monotonic()-started,
            integrity="PASS",files=sorted(p.relative_to(output).as_posix()
                for p in output.rglob("*") if p.is_file()))
        write_json(output/"manifest.json",manifest)
    except BaseException as exc:
        manifest.update(status="FAILED",completed_episodes=len(rows),
            error=repr(exc),duration_seconds=time.monotonic()-started)
        write_json(output/"manifest.json",manifest)
        raise
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--profile",choices=("smoke","full"),required=True)
    parser.add_argument("--device",choices=("cpu",),default="cpu")
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    print(run(args.profile,args.output_dir))


if __name__=="__main__":
    main()
