"""Read-only provenance, request-ledger and lookahead audit for v1/v2 runs."""

import argparse
from collections import defaultdict
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

from ht_pdm_fjsp import maintenance_shared_technician_factorial as core
from ht_pdm_fjsp import maintenance_shared_technician_exploratory_v2 as v2
from ht_pdm_fjsp.maintenance_coordination_references import marginal_gain
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, observation

ARMS = {(t, d) for t in core.TIMINGS for d in core.DISPATCHERS}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_from_snapshot(row):
    data = dict(row)
    for key in ("service_time", "restoration"):
        data[key] = tuple(tuple(x) for x in data[key])
    return core.WaitingConfig(**data)


def audit_run(directory, version, config_fn, conditions, source_paths):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    integrity = json.loads((directory / "integrity.json").read_text())
    if manifest["status"] != "COMPLETED" or integrity["status"] != "PASS":
        raise RuntimeError("incomplete prior run")
    if manifest["version"] != version:
        raise RuntimeError("wrong experiment version")
    if manifest["completed_episodes"] != manifest["expected_episodes"] != integrity["episodes"]:
        raise RuntimeError("wrong episode count")
    provenance = {}
    for filename, git_path in source_paths.items():
        path = directory / "source_snapshot" / filename
        blob = subprocess.check_output(["git", "show", f"{manifest['git_commit']}:{git_path}"])
        if path.read_bytes() != blob or sha(path) != manifest["source_sha256"][filename]:
            raise RuntimeError("source/Git/hash mismatch")
        provenance[filename] = sha(path)
    snapshot = json.loads((directory / "resolved_config.json").read_text())
    if set(snapshot) != set(conditions):
        raise RuntimeError("configuration panel mismatch")
    for condition in conditions:
        if asdict(config_fn(condition)) != asdict(config_from_snapshot(snapshot[condition])):
            raise RuntimeError("current/snapshot config mismatch")
    splits = {name: set(seeds) for name, seeds in manifest["seeds"].items()}
    if len(set().union(*splits.values())) != sum(map(len, splits.values())):
        raise RuntimeError("within-run seed overlap")
    rows = list(csv.DictReader((directory / "episodes.csv").open(newline="")))
    expected = {(split, int(seed), condition, timing, dispatch)
                for split, seeds in splits.items() for seed in seeds
                for condition in conditions for timing, dispatch in ARMS}
    keys = [(r["split"], int(r["seed"]), r["condition"], r["timing"], r["dispatcher"])
            for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise RuntimeError("episode key mismatch")
    numeric = ("cost", "policy_cost", "tail_cost", "terminal_liability", "maintenance_cost",
               "unavailability_cost", "failure_cost", "waiting_cost", "availability", "failures",
               "failed_waiting_ticks", "overdue_ticks", "pending_h", "pending_end", "active_end",
               "requests", "starts", "completions", "utilization")
    for r in rows:
        if not math.isclose(float(r["cost"]), sum(float(r[k]) for k in
            ("maintenance_cost", "unavailability_cost", "failure_cost", "waiting_cost", "terminal_liability")), abs_tol=1e-8):
            raise RuntimeError("component cost mismatch")
        if not math.isclose(float(r["cost"]), sum(float(r[k]) for k in
            ("policy_cost", "tail_cost", "terminal_liability")), abs_tol=1e-8):
            raise RuntimeError("phase cost mismatch")
        if int(float(r["requests"])) != sum(int(float(r[k])) for k in
            ("completions", "pending_end", "active_end")):
            raise RuntimeError("request conservation in episode")
        if float(r["invalid_assignments"]) != 0 or int(r["ticks"]) != core.H + core.TAIL:
            raise RuntimeError("physical episode invalid")
    arm_rows = []
    for condition in conditions:
        for timing in core.TIMINGS:
            for dispatcher in core.DISPATCHERS:
                panel = [r for r in rows if r["split"] == "sealed_test" and
                         r["condition"] == condition and r["timing"] == timing and r["dispatcher"] == dispatcher]
                arm_rows.append(dict(study=version, condition=condition, timing=timing,
                    dispatcher=dispatcher, seeds=len(panel),
                    **{k:statistics.fmean(float(r[k]) for r in panel) for k in numeric}))
    trace_count = 0
    episodes = 0
    current = None
    env = None
    pending, active, since = set(), set(), {}
    generated = completed = 0
    ledger = defaultdict(lambda: defaultdict(int))
    examples = []
    expected_trace_episodes = len(splits["sealed_test"]) * len(conditions) * len(ARMS)
    with (directory / "decisions.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["condition"], row["timing"], row["dispatcher"], int(row["seed"]))
            tick = int(row["tick"])
            if key != current:
                if current is not None and env.time != core.H + core.TAIL:
                    raise RuntimeError("trace truncated")
                if tick != 0:
                    raise RuntimeError("trace starts late")
                env = WaitingEnv(config_fn(key[0]))
                physical = role_seed(key[3], f"{version}:{key[0]}:physical")
                env.reset(physical)
                current = key
                pending, active, since = set(), set(), {}
                generated = completed = 0
                episodes += 1
            if tick != env.time or int(row["physical_seed"]) != physical:
                raise RuntimeError("trace time or seed")
            if json.loads(row["state"]) != json.loads(json.dumps(asdict(env.state))):
                raise RuntimeError("trace state")
            c = env.config
            proposed = core.proposals(c, env.state, key[1]) if tick < core.H else [
                m for m in range(c.machines) if env.state.failed[m] and m not in env.state.assigned]
            if json.loads(row["proposals"]) != proposed:
                raise RuntimeError("request rule / cleanup proposal")
            if tick >= core.H and any(not env.state.failed[m] for m in proposed):
                raise RuntimeError("preventive proposal in cleanup")
            for m in proposed:
                if m not in pending and m not in active:
                    pending.add(m)
                    since[m] = tick
                    generated += 1
            if set(json.loads(row["pending_before"])) != pending:
                raise RuntimeError("pending demand lost or invented")
            name = key[2] if tick < core.H else "fcfs"
            action = core.dispatch(c, env.state, pending, since, name, tick)
            if [list(pair) for pair in action] != json.loads(row["assignment"]):
                raise RuntimeError("dispatcher / cleanup action mismatch")
            scope = ledger[key[0], key[1], key[2]]
            if tick < core.H:
                scope["policy_ticks"] += 1
                scope["pending_machine_ticks"] += len(pending)
                scope["both_busy_ticks"] += all(x > 0 for x in env.state.remaining)
                if key[2] == "lookahead" and pending:
                    idle = [j for j, r in enumerate(env.state.remaining) if r == 0]
                    obs = observation(c, env.state, tick)
                    gains = [(m, j, marginal_gain(obs, m, j)) for m in pending for j in idle
                             if c.service_time[m][j] > 0]
                    if gains and not action:
                        scope["wait_with_eligible_pending"] += 1
                        if any(env.state.failed[m] for m in pending):
                            scope["wait_with_failed_pending"] += 1
                        if len(examples) < 20 and any(env.state.failed[m] for m in pending):
                            examples.append(dict(study=version, condition=key[0], timing=key[1],
                                seed=key[3], tick=tick, state=asdict(env.state),
                                pending=sorted(pending), gains=[(m,j,round(g,6)) for m,j,g in gains],
                                action=action))
            else:
                scope["tail_ticks"] += 1
            for m, _ in action:
                pending.remove(m)
                active.add(m)
                del since[m]
            scope["assignments"] += len(action)
            _, _, _, info = env.step(action)
            if json.loads(row["following"]) != json.loads(json.dumps(asdict(env.state))) or not math.isclose(
                float(row["cost"]), info["objective"], abs_tol=1e-8):
                raise RuntimeError("physical transition mismatch")
            for m in tuple(active):
                if m not in env.state.assigned:
                    active.remove(m)
                    completed += 1
                    scope["completions"] += 1
            for m, failed in enumerate(env.state.failed):
                if failed and m not in pending and m not in active:
                    pending.add(m)
                    since[m] = tick + 1
                    generated += 1
            if generated != completed + len(active) + len(pending):
                raise RuntimeError("request ledger mismatch")
            if int(row["pending_after"]) != len(pending) or int(row["active_after"]) != len(active):
                raise RuntimeError("trace demand/active count mismatch")
            trace_count += 1
    if episodes != expected_trace_episodes or trace_count != episodes * (core.H + core.TAIL):
        raise RuntimeError("trace panel incomplete")
    return dict(directory=str(directory), commit=manifest["git_commit"],
        manifest_sha256=sha(directory / "manifest.json"),
        episodes_sha256=sha(directory / "episodes.csv"),
        decisions_sha256=sha(directory / "decisions.csv"),
        source_sha256=provenance, seeds={k:sorted(v) for k,v in splits.items()},
        rows=arm_rows, ledger={"/".join(k):v for k,v in ledger.items()},
        lookahead_examples=examples, trace_episodes=episodes, trace_ticks=trace_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    runs = [audit_run(args.v1, core.VERSION, core.config, core.CONDITIONS,
        {"maintenance_shared_technician_factorial.py":"src/ht_pdm_fjsp/maintenance_shared_technician_factorial.py",
         "maintenance_shared_technician_factorial_v1_protocol.md":"docs/maintenance_shared_technician_factorial_v1_protocol.md",
         "maintenance_shared_technician_factorial_v1_seed_registry.json":"configs/maintenance_shared_technician_factorial_v1_seed_registry.json"}),
        audit_run(args.v2, v2.VERSION, v2.config, v2.CONDITIONS,
        {"maintenance_shared_technician_factorial.py":"src/ht_pdm_fjsp/maintenance_shared_technician_factorial.py",
         "maintenance_shared_technician_exploratory_v2.py":"src/ht_pdm_fjsp/maintenance_shared_technician_exploratory_v2.py",
         "maintenance_shared_technician_exploratory_v2_protocol.md":"docs/maintenance_shared_technician_exploratory_v2_protocol.md",
         "maintenance_shared_technician_exploratory_v2_seed_registry.json":"configs/maintenance_shared_technician_exploratory_v2_seed_registry.json"})]
    if set(runs[0]["seeds"]["sealed_test"]) & set(runs[1]["seeds"]["sealed_test"]):
        raise RuntimeError("cross-study seed overlap")
    output.mkdir(parents=True)
    (output / "source_snapshot").mkdir()
    shutil.copy2(Path(__file__), output / "source_snapshot" / Path(__file__).name)
    arm_fields = tuple(runs[0]["rows"][0])
    with (output / "all_arms.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, arm_fields)
        writer.writeheader()
        for run in runs:
            writer.writerows(run.pop("rows"))
    (output / "audit.json").write_text(json.dumps(dict(status="PASS",
        created_utc=datetime.now(timezone.utc).isoformat(),
        audit_source_sha256=sha(Path(__file__)), runs={"v1":runs[0],"v2":runs[1]}),
        indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
