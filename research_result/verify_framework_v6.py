"""Independent artifact, grid, selection and hierarchical-CI audit for v6."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

VERSION = "maintenance_framework_v6_edge_sharing"
CASES = ("reciprocal_scale2", "interchangeable_scale1", "reciprocal_scale3",
         "specialist_generalist_scale2")
LEARNED = ("flat", "unshared_edge", "shared_edge")
RULES = ("fcfs", "risk_matching", "rollout")
CONTROLLERS = (*RULES, *LEARNED)
PAIRS = (("shared_edge", "unshared_edge"), ("shared_edge", "flat"),
         ("shared_edge", "risk_matching"), ("shared_edge", "rollout"),
         ("shared_edge", "fcfs"), ("unshared_edge", "flat"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def role_seed(seed, role):
    return int.from_bytes(hashlib.sha256(f"{seed}:{role}".encode()).digest()[:8], "big")


def panel(registry, name):
    item = registry[name]
    return list(range(item["start"], item["start"] + item["count"]))


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-8, rel_tol=0):
        raise AssertionError(f"{label}: {actual} != {expected}")


def hierarchical(values, bootstrap_seed, key):
    matrix = np.asarray(values, dtype=float)
    assert matrix.ndim == 2
    rng = np.random.default_rng(role_seed(bootstrap_seed, VERSION + ":" + key))
    training = rng.integers(0, matrix.shape[0], size=(20000, matrix.shape[0]))
    physical = rng.integers(0, matrix.shape[1], size=(20000, matrix.shape[1]))
    sample_means = matrix[training[:, :, None], physical[:, None, :]].mean(axis=(1, 2))
    return np.quantile(sample_means, [.025, .975]).tolist()


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
    assert len(all_seeds) == len(set(all_seeds)), "overlapping splits"
    for path in (root / "configs").glob("*seed_registry.json"):
        if path == root / "configs/maintenance_framework_v6_seed_registry.json":
            continue
        data = json.loads(path.read_text())
        historical = []
        for key, value in data.items():
            if isinstance(value, dict) and "start" in value and "count" in value:
                historical.extend(range(value["start"], value["start"] + value["count"]))
            elif key.endswith("_seed") and isinstance(value, int):
                historical.append(value)
        assert not set(all_seeds) & set(historical), f"historical seed overlap: {path}"
    full = manifest["profile"] == "full"
    train = groups[0] if full else groups[3][:1]
    dev = groups[1] if full else groups[3]
    test = groups[2] if full else groups[3]
    steps = [50176, 100352] if full else [1024]
    assert resolved["steps"] == steps
    for relative, expected in manifest["source_hashes"].items():
        assert sha(root / relative) == expected, f"source changed: {relative}"
    for name, expected in manifest["artifact_hashes"].items():
        assert sha(run / name) == expected, f"artifact changed: {name}"
    assert (run / "protocol.md").read_bytes() == (root / "research_result/framework_v6_protocol_vi.md").read_bytes()
    assert (run / "method_proposal.md").read_bytes() == (root / "research_result/method_v6_proposal_vi.md").read_bytes()
    assert set(tapes) == set(training_audit) == {str(seed) for seed in train}
    checkpoints = []
    for training_seed in train:
        recorded = tapes[str(training_seed)]
        assert set(recorded) == set(LEARNED)
        reference = recorded[LEARNED[0]]
        assert reference and all(recorded[arm] == reference for arm in LEARNED)
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
        assert info["shared_edge"]["common_start_sha256"] == info["unshared_edge"]["common_start_sha256"]
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
        assert set(selection[arm]) == {str(seed) for seed in train}
        for training_seed in train:
            candidates = []
            for step in steps:
                cell = [float(row["objective"]) for row in development
                        if row["controller"] == arm and int(row["train_seed"]) == training_seed
                        and int(row["checkpoint_step"]) == step]
                assert len(cell) == len(CASES) * len(dev)
                candidates.append((mean(cell), step))
            expected_score, expected_step = min(candidates)
            item = selection[arm][str(training_seed)]
            close(item["development_mean_cost"], expected_score, f"{arm}/{training_seed} DEV")
            assert item["steps"] == expected_step
            assert len(item["candidate_scores"]) == len(steps)
            assert sha(run / item["checkpoint"]) == item["checkpoint_sha256"]
    assert sha(run / "selection.json") == manifest["selection_sha256"]
    expected_grid = {(case, controller, training_seed if controller in LEARNED else None, physical)
                     for case in CASES for controller in CONTROLLERS
                     for training_seed in (train if controller in LEARNED else [None]) for physical in test}
    actual_grid = {(row["case"], row["controller"],
                    int(row["train_seed"]) if row["train_seed"] else None, int(row["seed"]))
                   for row in episodes}
    assert len(episodes) == len(expected_grid) and actual_grid == expected_grid
    for row in episodes:
        label = f'{row["case"]}/{row["controller"]}/{row["train_seed"]}/{row["seed"]}'
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
        if row["controller"] in LEARNED:
            item = selection[row["controller"]][row["train_seed"]]
            assert row["checkpoint"] == item["checkpoint"]
            assert int(row["checkpoint_step"]) == item["steps"]
        else:
            assert row["train_seed"] == row["checkpoint"] == row["checkpoint_step"] == ""
    cells = {(row["case"], row["controller"],
              int(row["train_seed"]) if row["train_seed"] else None,
              int(row["seed"])): row for row in episodes}
    for case in CASES:
        for controller in CONTROLLERS:
            keys = ([(case, controller, seed, physical) for seed in train for physical in test]
                    if controller in LEARNED else
                    [(case, controller, None, physical) for physical in test])
            rows = [cells[key] for key in keys]
            item = summary[f"{case}/{controller}"]
            assert item["count"] == len(rows)
            for metric, reported in item["means"].items():
                close(mean(float(row[metric]) for row in rows), reported,
                      f"{case}/{controller}/{metric}")
            if controller in LEARNED:
                for training_seed in train:
                    reported = item["train_seed_mean_cost"][str(training_seed)]
                    actual = mean(float(cells[(case, controller, training_seed, physical)]["objective"])
                                  for physical in test)
                    close(actual, reported, f"{case}/{controller}/{training_seed}")
        for subject, comparator in PAIRS:
            matrix = np.asarray([[float(cells[(case, subject, train_seed, physical)]["objective"]) -
                                  float(cells[(case, comparator,
                                               train_seed if comparator in LEARNED else None,
                                               physical)]["objective"])
                                  for physical in test] for train_seed in train])
            key = f"{case}/{subject}_minus_{comparator}"
            item = summary[key]
            close(matrix.mean(), item["paired_mean"], key + " mean")
            for actual, reported in zip(hierarchical(matrix, registry["bootstrap_seed"], key),
                                        item["ci95_descriptive"]):
                close(actual, reported, key + " CI")
            for i, train_seed in enumerate(train):
                close(matrix[i].mean(), item["train_seed_means"][str(train_seed)], key + " train")
    for subject, comparator in PAIRS:
        matrix = np.asarray([[
            mean(float(cells[(case, subject, train_seed, physical)]["objective"]) -
                 float(cells[(case, comparator,
                              train_seed if comparator in LEARNED else None,
                              physical)]["objective"])
                 for case in CASES)
            for physical in test] for train_seed in train])
        key = f"pooled_{subject}_minus_{comparator}"
        item = summary[key]
        interval = hierarchical(matrix, registry["bootstrap_seed"], key)
        close(matrix.mean(), item["paired_mean"], key + " mean")
        for actual, reported in zip(interval, item["hierarchical_ci95"]):
            close(actual, reported, key + " CI")
        assert item["training_seed_count"] == len(train)
        assert item["physical_seed_count"] == len(test)
        assert item["cases_per_cell"] == len(CASES)
        assert (item["wins"], item["ties"], item["losses"]) == (
            int((matrix < 0).sum()), int((matrix == 0).sum()), int((matrix > 0).sum()))
        for i, train_seed in enumerate(train):
            close(matrix[i].mean(), item["train_seed_means"][str(train_seed)], key + " train")
        primary = subject == "shared_edge" and comparator == "unshared_edge"
        assert item["confirmed"] == (bool(matrix.mean() < 0 and interval[1] < 0) if primary else None)
    assert manifest["counts"] == {
        "ppo_training_steps": len(train) * len(LEARNED) * steps[-1],
        "ppo_checkpoints": len(checkpoints),
        "development": len(development), "evaluation": len(episodes),
    }
    assert set(manifest["policy_parameters"]) == set(LEARNED)
    assert abs(manifest["policy_parameters"]["shared_edge"] -
               manifest["policy_parameters"]["unshared_edge"]) <= 10
    return {
        "status": "PASS", "run": str(run), "commit": manifest["git"]["commit"],
        "profile": manifest["profile"], "ppo_steps": manifest["counts"]["ppo_training_steps"],
        "development": len(development), "evaluation": len(episodes),
        "primary": summary["pooled_shared_edge_minus_unshared_edge"],
        "checks": ["source/artifact/checkpoint hashes", "fresh splits/full grids",
                   "independently regenerated paired training tapes",
                   "within-training-seed DEV selection", "economics/request conservation",
                   "all hierarchical paired bootstrap intervals"],
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
