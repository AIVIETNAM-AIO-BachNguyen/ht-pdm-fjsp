"""Independent artifact and inference audit for the v5 post-decision capacity campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

VERSION = "maintenance_framework_v5_postdecision_capacity"
CASES = ("reciprocal_scale2", "interchangeable_scale1", "reciprocal_scale3",
         "specialist_generalist_scale2")
LEARNED = ("flat", "edge_only", "additive_queue", "postdecision")
CONTROLLERS = ("fcfs", "risk_matching", "rollout", *LEARNED)
PAIRS = (
    ("postdecision", "additive_queue"), ("additive_queue", "edge_only"),
    ("postdecision", "edge_only"), ("postdecision", "flat"),
    ("postdecision", "risk_matching"), ("postdecision", "rollout"), ("postdecision", "fcfs"),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def role_seed(seed, role):
    return int.from_bytes(hashlib.sha256(f"{seed}:{role}".encode()).digest()[:8], "big")


def panel(registry, name):
    spec = registry[name]
    return list(range(spec["start"], spec["start"] + spec["count"]))


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-8, rel_tol=0):
        raise AssertionError(f"{label}: {actual} != {expected}")


def bootstrap(values, seed, key):
    value = np.asarray(values, dtype=float)
    rng = np.random.default_rng(role_seed(seed, f"{VERSION}:{key}"))
    draws = rng.integers(0, len(value), size=(20000, len(value)))
    return np.quantile(value[draws].mean(axis=1), [.025, .975])


def verify(run):
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((run / "manifest.json").read_text())
    registry = json.loads((run / "seed_registry.json").read_text())
    resolved = json.loads((run / "resolved_config.json").read_text())
    training_audit = json.loads((run / "training_audit.json").read_text())
    tapes = json.loads((run / "training_tapes.json").read_text())
    selection = json.loads((run / "selection.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    development = load_csv(run / "development.csv")
    episodes = load_csv(run / "episodes.csv")
    assert manifest["status"] == "COMPLETED"
    assert manifest["protocol"] == registry["campaign_version"] == VERSION
    assert manifest["seed_registry"] == registry
    assert set(resolved["cases"]) == set(CASES)
    assert resolved["train_case_distribution"] == "uniform over four fixed cases"
    assert resolved["arms"] == list(LEARNED)
    groups = [panel(registry, key) for key in ("training", "development", "sealed_test", "smoke")]
    groups += [[registry["forecast_seed"]], [registry["bootstrap_seed"]]]
    all_seeds = [seed for group in groups for seed in group]
    assert len(all_seeds) == len(set(all_seeds)), "overlapping seed splits"
    for path in (root / "configs").glob("*seed_registry.json"):
        if path == root / "configs/maintenance_framework_v5_seed_registry.json":
            continue
        data = json.loads(path.read_text())
        historical = []
        for key, value in data.items():
            if isinstance(value, dict) and "start" in value and "count" in value:
                historical.extend(range(value["start"], value["start"] + value["count"]))
            elif key.endswith("_seed") and isinstance(value, int):
                historical.append(value)
        assert not set(all_seeds) & set(historical), f"historical split overlap: {path}"
    full = manifest["profile"] == "full"
    train = groups[0] if full else groups[3][:1]
    dev = groups[1] if full else groups[3]
    test = groups[2] if full else groups[3]
    steps = [50176, 100352, 149504, 199680] if full else [1024]
    assert resolved["steps"] == steps
    for relative, expected in manifest["source_hashes"].items():
        assert sha(root / relative) == expected, f"source changed: {relative}"
    for name, expected in manifest["artifact_hashes"].items():
        assert sha(run / name) == expected, f"artifact changed: {name}"
    assert (run / "protocol.md").read_bytes() == (
        root / "research_result/framework_v5_protocol_vi.md").read_bytes()
    assert (run / "method_proposal.md").read_bytes() == (
        root / "research_result/method_v5_proposal_vi.md").read_bytes()
    assert set(tapes) == set(training_audit) == {str(seed) for seed in train}
    checkpoints = []
    for training_seed in train:
        recorded = tapes[str(training_seed)]
        assert set(recorded) == set(LEARNED)
        reference = recorded[LEARNED[0]]
        assert reference, "empty training tape"
        for arm in LEARNED:
            assert recorded[arm] == reference, "unpaired training stream"
        expected_cases = np.random.default_rng(
            role_seed(training_seed, VERSION + ":case")
        ).integers(0, len(CASES), size=len(reference))
        expected_physical = np.random.default_rng(
            role_seed(training_seed, VERSION + ":physical_sequence")
        ).integers(0, 2**31, size=len(reference))
        assert reference == [[CASES[int(i)], int(raw)]
                             for i, raw in zip(expected_cases, expected_physical)]
        info = training_audit[str(training_seed)]
        assert set(info) == set(LEARNED)
        structured = {info[arm]["initial_policy_sha256"] for arm in LEARNED[1:]}
        assert len(structured) == 1, "structured initialization mismatch"
        digest = hashlib.sha256(json.dumps(reference, separators=(",", ":")).encode()).hexdigest()
        for arm in LEARNED:
            item = info[arm]
            assert item["training_episode_count"] == len(reference)
            assert item["training_tape_sha256"] == digest
            assert item["wall_seconds"] > 0
    for arm in LEARNED:
        progress = load_csv(run / "arms" / arm / "training_progress.csv")
        expected_progress = {(seed, step) for seed in train for step in steps}
        actual_progress = {(int(row["train_seed"]), int(row["steps"])) for row in progress}
        assert len(progress) == len(expected_progress) and actual_progress == expected_progress
        for row in progress:
            path = run / "arms" / arm / row["checkpoint"]
            assert path.is_file()
            checkpoints.append(path)
    expected_dev = {(arm, seed, step, case, physical)
                    for arm in LEARNED for seed in train for step in steps
                    for case in CASES for physical in dev}
    actual_dev = {(row["controller"], int(row["train_seed"]), int(row["checkpoint_step"]),
                   row["case"], int(row["seed"])) for row in development}
    assert len(development) == len(expected_dev) and actual_dev == expected_dev
    assert set(selection) == set(LEARNED)
    for arm in LEARNED:
        values = {}
        for row in development:
            if row["controller"] == arm:
                key = (int(row["train_seed"]), int(row["checkpoint_step"]))
                values.setdefault(key, []).append(float(row["objective"]))
        assert all(len(cell) == len(CASES) * len(dev) for cell in values.values())
        selected = min((mean(cell), seed, step) for (seed, step), cell in values.items())
        item = selection[arm]
        close(item["development_mean_cost"], selected[0], arm + " DEV score")
        assert (item["train_seed"], item["steps"]) == selected[1:]
        assert len(item["candidate_scores"]) == len(train) * len(steps)
        assert sha(run / item["checkpoint"]) == item["checkpoint_sha256"]
    assert sha(run / "selection.json") == manifest["selection_sha256"]
    expected_grid = {(case, controller, physical) for case in CASES
                     for controller in CONTROLLERS for physical in test}
    actual_grid = {(row["case"], row["controller"], int(row["seed"])) for row in episodes}
    assert len(episodes) == len(expected_grid) and actual_grid == expected_grid
    for row in episodes:
        label = f'{row["case"]}/{row["controller"]}/{row["seed"]}'
        assert int(row["physical_seed"]) == role_seed(int(row["seed"]), registry["version"] + ":physical")
        close(row["objective"], -float(row["episode_return"]), label + " reward")
        close(row["objective"], float(row["H_cost"]) + float(row["tail_cost"]), label + " tail")
        close(row["objective"], sum(float(row[key]) for key in
              ("maintenance_cost", "unavailability_cost", "failure_cost")), label + " components")
        close(float(row["requests_created"]) - float(row["requests_completed"]),
              float(row["end_pending"]) + float(row["end_in_service"]), label + " requests")
        close(float(row["service_starts"]) - float(row["requests_completed"]),
              float(row["end_in_service"]), label + " service")
        close(row["request_wait_ticks"], float(row["queued_pm_ticks"]) + float(row["queued_cm_ticks"]),
              label + " waiting")
        assert float(row["invalid_assignments"]) == 0
        assert 0 <= float(row["selection_seconds"]) <= float(row["episode_wall_seconds"])
    panels = {}
    for case in CASES:
        panels[case] = {}
        for controller in CONTROLLERS:
            rows = sorted((row for row in episodes if row["case"] == case
                           and row["controller"] == controller), key=lambda row: int(row["seed"]))
            panels[case][controller] = rows
            item = summary[f"{case}/{controller}"]
            assert item["count"] == len(test)
            for metric, reported in item["means"].items():
                close(mean(float(row[metric]) for row in rows), reported,
                      f"{case}/{controller}/{metric}")
        for subject, comparator in PAIRS:
            lhs, rhs = panels[case][subject], panels[case][comparator]
            assert [int(row["seed"]) for row in lhs] == [int(row["seed"]) for row in rhs]
            delta = np.asarray([float(a["objective"]) - float(b["objective"]) for a, b in zip(lhs, rhs)])
            key = f"{case}/{subject}_minus_{comparator}"
            item = summary[key]
            close(delta.mean(), item["paired_mean"], key + " mean")
            for actual, reported in zip(bootstrap(delta, registry["bootstrap_seed"], key),
                                        item["ci95_descriptive"]):
                close(actual, reported, key + " CI")
            assert (item["wins"], item["ties"], item["losses"]) == (
                int((delta < 0).sum()), int((delta == 0).sum()), int((delta > 0).sum()))
    seeds = [int(row["seed"]) for row in panels[CASES[0]]["postdecision"]]
    for subject, comparator in PAIRS:
        values = []
        for index, seed in enumerate(seeds):
            terms = []
            for case in CASES:
                lhs, rhs = panels[case][subject][index], panels[case][comparator][index]
                assert int(lhs["seed"]) == int(rhs["seed"]) == seed
                terms.append(float(lhs["objective"]) - float(rhs["objective"]))
            values.append(mean(terms))
        delta = np.asarray(values)
        key = f"pooled_{subject}_minus_{comparator}"
        item = summary[key]
        close(delta.mean(), item["paired_mean"], key + " mean")
        interval = bootstrap(delta, registry["bootstrap_seed"], key)
        for actual, reported in zip(interval, item["cluster_bootstrap_ci95"]):
            close(actual, reported, key + " CI")
        assert item["seed_count"] == len(test) and item["cases_per_seed"] == len(CASES)
        assert (item["wins"], item["ties"], item["losses"]) == (
            int((delta < 0).sum()), int((delta == 0).sum()), int((delta > 0).sum()))
        primary = subject == "postdecision" and comparator == "additive_queue"
        assert item["confirmed"] == (bool(delta.mean() < 0 and interval[1] < 0) if primary else None)
    assert manifest["counts"] == {
        "ppo_training_steps": len(train) * len(LEARNED) * steps[-1],
        "ppo_checkpoints": len(checkpoints),
        "development": len(development), "evaluation": len(episodes),
    }
    assert set(manifest["policy_parameters"]) == set(LEARNED)
    assert all(value > 0 for value in manifest["policy_parameters"].values())
    return {
        "status": "PASS", "run": str(run), "commit": manifest["git"]["commit"],
        "profile": manifest["profile"], "ppo_steps": manifest["counts"]["ppo_training_steps"],
        "development": len(development), "evaluation": len(episodes),
        "primary": summary["pooled_postdecision_minus_additive_queue"],
        "checks": ["source/artifact/checkpoint hashes", "fresh splits/full grids",
                   "independently regenerated paired training tapes",
                   "DEV selection", "episode economics/request conservation",
                   "all paired bootstrap intervals"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = verify(args.run_dir.resolve())
    (out / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
