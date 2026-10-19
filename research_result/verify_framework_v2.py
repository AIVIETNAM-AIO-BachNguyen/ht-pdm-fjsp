"""Independent exact-grid/accounting/bootstrap verifier for mixture campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

CASES = ("reciprocal_scale2", "interchangeable_scale1", "reciprocal_scale3",
         "specialist_generalist_scale2")
CONTROLLERS = ("fcfs", "risk_matching", "rollout", "anchor_ppo", "mixture_ppo")
COMPARATORS = ("anchor_ppo", "fcfs", "risk_matching", "rollout")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def panel(registry, name):
    spec = registry[name]
    return list(range(spec["start"], spec["start"] + spec["count"]))


def close(a, b, label):
    if not math.isclose(float(a), float(b), abs_tol=1e-8, rel_tol=0):
        raise AssertionError(f"{label}: {a} != {b}")


def verify(run_dir):
    repo = Path(__file__).resolve().parent.parent
    manifest = json.loads((run_dir / "manifest.json").read_text())
    registry = json.loads((run_dir / "seed_registry.json").read_text())
    config = json.loads((run_dir / "resolved_config.json").read_text())
    selection = json.loads((run_dir / "selection.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    counts = json.loads((run_dir / "training_case_counts.json").read_text())
    training = rows(run_dir / "training_progress.csv")
    development = rows(run_dir / "development.csv")
    episodes = rows(run_dir / "episodes.csv")
    assert manifest["status"] == "COMPLETED" and manifest["profile"] == "full"
    assert manifest["protocol"] == registry["campaign_version"]
    assert manifest["seed_registry"] == registry
    assert set(config["cases"]) == set(CASES)
    groups = [panel(registry, name) for name in ("training", "development", "sealed_test", "smoke")]
    groups += [[registry["forecast_seed"]], [registry["bootstrap_seed"]]]
    flat = [seed for group in groups for seed in group]
    assert len(flat) == len(set(flat)), "seed panel overlap"
    for relative, expected in manifest["source_hashes"].items():
        assert sha(repo / relative) == expected, f"source hash: {relative}"
    for name, expected in manifest["artifact_hashes"].items():
        assert sha(run_dir / name) == expected, f"artifact hash: {name}"
    anchor = repo / manifest["anchor_checkpoint"]
    assert sha(anchor) == manifest["anchor_sha256"]
    train, dev, test = groups[:3]
    steps = config["steps"]
    expected_checkpoints = {(seed, step) for seed in train for step in steps}
    assert len(training) == len(expected_checkpoints)
    assert {(int(row["train_seed"]), int(row["steps"])) for row in training} == expected_checkpoints
    for row in training:
        assert (run_dir / row["checkpoint"]).is_file()
    assert set(counts) == {str(seed) for seed in train}
    for case_counts in counts.values():
        assert set(case_counts) == set(CASES) and all(value > 0 for value in case_counts.values())
        assert sum(case_counts.values()) >= steps[-1] // 48
    expected_dev = {(seed, step, case, physical) for seed in train for step in steps
                    for case in CASES for physical in dev}
    actual_dev = {(int(row["train_seed"]), int(row["checkpoint_step"]), row["case"], int(row["seed"]))
                  for row in development}
    assert len(development) == len(expected_dev) and actual_dev == expected_dev
    candidates = {}
    for row in development:
        candidates.setdefault((int(row["train_seed"]), int(row["checkpoint_step"])), []).append(float(row["objective"]))
    winner = min((mean(values), seed, step) for (seed, step), values in candidates.items())
    assert (selection["train_seed"], selection["steps"]) == winner[1:]
    close(selection["development_case_balanced_mean_cost"], winner[0], "checkpoint selection")
    assert (run_dir / selection["checkpoint"]).is_file()
    assert manifest["selected_checkpoint"] == selection["checkpoint"]
    expected_grid = {(case, controller, seed) for case in CASES for controller in CONTROLLERS for seed in test}
    actual_grid = {(row["case"], row["controller"], int(row["seed"])) for row in episodes}
    assert len(episodes) == len(expected_grid) and actual_grid == expected_grid
    for row in episodes:
        label = f'{row["case"]}/{row["controller"]}/{row["seed"]}'
        raw = int(row["seed"])
        derived = int.from_bytes(hashlib.sha256(f'{raw}:{registry["version"]}:physical'.encode()).digest()[:8], "big")
        assert int(row["physical_seed"]) == derived, label
        close(row["objective"], sum(float(row[key]) for key in
             ("maintenance_cost", "unavailability_cost", "failure_cost")), label + " components")
        close(row["objective"], -float(row["episode_return"]), label + " reward")
        close(row["objective"], float(row["H_cost"]) + float(row["tail_cost"]), label + " tail")
        assert float(row["invalid_assignments"]) == 0, label
        assert 0 <= float(row["selection_seconds"]) <= float(row["episode_wall_seconds"]), label
        assert float(row["requests_created"]) - float(row["requests_completed"]) == (
            float(row["end_pending"]) + float(row["end_in_service"])), label
        assert float(row["service_starts"]) - float(row["requests_completed"]) == float(row["end_in_service"]), label
        close(row["request_wait_ticks"], float(row["queued_pm_ticks"]) + float(row["queued_cm_ticks"]), label + " wait")
    by_case = {}
    rng = np.random.default_rng(registry["bootstrap_seed"])
    for case in CASES:
        by_controller = {name: sorted((row for row in episodes if row["case"] == case and row["controller"] == name),
                                      key=lambda row: int(row["seed"])) for name in CONTROLLERS}
        by_case[case] = by_controller
        for name, panel_rows in by_controller.items():
            entry = summary[f"{case}/{name}"]
            assert len(panel_rows) == entry["count"] == len(test)
            for metric, reported in entry["means"].items():
                close(mean(float(row[metric]) for row in panel_rows), reported, f"{case}/{name}/{metric}")
        mix = by_controller["mixture_ppo"]
        for comparator in COMPARATORS:
            other = by_controller[comparator]
            assert [int(row["seed"]) for row in mix] == [int(row["seed"]) for row in other]
            differences = np.asarray([float(a["objective"]) - float(b["objective"])
                                      for a, b in zip(mix, other)])
            entry = summary[f"{case}/mixture_minus_{comparator}"]
            close(differences.mean(), entry["paired_mean"], f"{case}/{comparator} paired")
            assert (entry["wins"], entry["ties"], entry["losses"]) == (
                int((differences < 0).sum()), int((differences == 0).sum()), int((differences > 0).sum()))
            draw_indices = rng.integers(0, len(test), size=(20000, len(test)))
            ci = np.quantile(differences[draw_indices].mean(axis=1), [.025, .975])
            for actual, reported in zip(ci, entry["bootstrap_ci95_descriptive"]):
                close(actual, reported, f"{case}/{comparator} interval")
    pooled = []
    for index, seed in enumerate(test):
        delta = []
        for case in CASES:
            mix = by_case[case]["mixture_ppo"][index]
            anchor = by_case[case]["anchor_ppo"][index]
            assert int(mix["seed"]) == int(anchor["seed"]) == seed
            delta.append(float(mix["objective"]) - float(anchor["objective"]))
        pooled.append(mean(delta))
    pooled = np.asarray(pooled)
    entry = summary["primary_case_balanced_mixture_minus_anchor"]
    close(pooled.mean(), entry["paired_mean"], "primary paired mean")
    indices = rng.integers(0, len(test), size=(20000, len(test)))
    ci = np.quantile(pooled[indices].mean(axis=1), [.025, .975])
    for actual, reported in zip(ci, entry["cluster_bootstrap_ci95"]):
        close(actual, reported, "primary cluster interval")
    assert entry["positive_improvement_confirmed"] == bool(pooled.mean() < 0 and ci[1] < 0)
    assert (entry["seed_count"], entry["cases_per_seed"]) == (len(test), len(CASES))
    assert manifest["counts"] == {"episodes": len(episodes), "development": len(development),
        "training_steps": len(train) * steps[-1], "checkpoints": len(training)}
    return {"status": "PASS", "run": str(run_dir), "commit": manifest["git"]["commit"],
            "episodes": len(episodes), "development_episodes": len(development),
            "checkpoints": len(training), "primary_paired_mean": float(pooled.mean()),
            "primary_ci95": ci.tolist(), "primary_confirmed": entry["positive_improvement_confirmed"],
            "checks": ["source/artifact/comparator hashes", "disjoint seed panels", "checkpoint selection",
                       "complete paired grid", "physical seed pairing", "cost/reward/tail conservation",
                       "request conservation", "all means and paired bootstrap intervals"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = verify(args.run_dir.resolve())
    (output / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
