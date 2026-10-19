"""Independent aggregate/provenance audit for framework v1 campaign artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def panel(registry: dict, name: str) -> list[int]:
    spec = registry[name]
    return list(range(spec["start"], spec["start"] + spec["count"]))


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8):
        raise AssertionError(f"{label}: {actual} != {expected}")


def verify(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    registry = json.loads((run_dir / "seed_registry.json").read_text())
    config = json.loads((run_dir / "resolved_config.json").read_text())
    selection = json.loads((run_dir / "selection.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    episodes = load_csv(run_dir / "episodes.csv")
    development = load_csv(run_dir / "development.csv")
    training = load_csv(run_dir / "training_progress.csv")
    assert manifest["status"] == "COMPLETED" and manifest["profile"] == "full"
    assert registry == manifest["seed_registry"]
    groups = [panel(registry, name) for name in ("training", "development", "sealed_test", "smoke")]
    groups += [[registry["forecast_seed"]], [registry["bootstrap_seed"]]]
    flat = [seed for group in groups for seed in group]
    assert len(flat) == len(set(flat)), "seed split overlap"
    root = Path(__file__).resolve().parent.parent
    for name, saved in manifest["source_hashes"].items():
        assert digest(root / name) == saved, f"source mismatch: {name}"
    for name, saved in manifest["artifact_hashes"].items():
        assert digest(run_dir / name) == saved, f"artifact mismatch: {name}"
    train, dev, test = groups[:3]
    steps = config["steps"]
    expected_checkpoints = {(seed, step) for seed in train for step in steps}
    assert {(int(row["train_seed"]), int(row["steps"])) for row in training} == expected_checkpoints
    assert len(training) == len(expected_checkpoints)
    for row in training:
        assert (run_dir / row["checkpoint"]).is_file()
    assert (run_dir / selection["checkpoint"]).is_file()
    candidates = {}
    for row in development:
        assert row["case"] == "reciprocal_scale2" and int(row["seed"]) in dev
        key = (int(row["train_seed"]), int(row["checkpoint_step"]))
        candidates.setdefault(key, {})[int(row["seed"])] = float(row["objective"])
    assert set(candidates) == expected_checkpoints
    assert all(set(seed_costs) == set(dev) for seed_costs in candidates.values())
    winner = min((mean(seed_costs.values()), seed, step) for (seed, step), seed_costs in candidates.items())
    close(winner[0], selection["development_mean_cost"], "selection score")
    assert (winner[1], winner[2]) == (selection["train_seed"], selection["steps"])
    assert manifest["selected_checkpoint"] == selection["checkpoint"]
    cases = set(config["cases"])
    controllers = {"fcfs", "risk_matching", "rollout", "learner"}
    expected_grid = {(case, controller, seed) for case in cases for controller in controllers for seed in test}
    actual_grid = {(r["case"], r["controller"], int(r["seed"])) for r in episodes}
    assert actual_grid == expected_grid and len(episodes) == len(expected_grid)
    for row in episodes:
        label = f'{row["case"]}/{row["controller"]}/{row["seed"]}'
        raw_seed = int(row["seed"])
        physical_seed = int.from_bytes(hashlib.sha256(
            f'{raw_seed}:{registry["version"]}:physical'.encode()).digest()[:8], "big")
        assert int(row["physical_seed"]) == physical_seed, label
        close(float(row["objective"]), sum(float(row[key]) for key in
              ("maintenance_cost", "unavailability_cost", "failure_cost")), label + " components")
        close(float(row["objective"]), -float(row["episode_return"]), label + " reward")
        close(float(row["objective"]), float(row["H_cost"]) + float(row["tail_cost"]), label + " tail")
        assert int(float(row["invalid_assignments"])) == 0, label
        assert 0 <= float(row["H_availability"]) <= 1, label
        assert 0 <= float(row["selection_seconds"]) <= float(row["episode_wall_seconds"]), label
        assert int(float(row["requests_created"])) - int(float(row["requests_completed"])) == (
            int(float(row["end_pending"])) + int(float(row["end_in_service"]))), label
        assert int(float(row["service_starts"])) - int(float(row["requests_completed"])) == int(float(row["end_in_service"])), label
        assert int(float(row["request_wait_ticks"])) == (int(float(row["queued_pm_ticks"])) + int(float(row["queued_cm_ticks"]))), label
    for case in cases:
        by_controller = {name: sorted((r for r in episodes if r["case"] == case and r["controller"] == name),
                                      key=lambda r: int(r["seed"])) for name in controllers}
        for name, rows in by_controller.items():
            entry = summary[f"{case}/{name}"]
            assert entry["count"] == len(test)
            for metric, reported in entry["means"].items():
                close(mean(float(row[metric]) for row in rows), reported, f"{case}/{name}/{metric}")
        learned = by_controller["learner"]
        for baseline in controllers - {"learner"}:
            others = by_controller[baseline]
            differences = [float(a["objective"]) - float(b["objective"]) for a, b in zip(learned, others)]
            entry = summary[f"{case}/learner_minus_{baseline}"]
            close(mean(differences), entry["paired_mean"], f"{case}/{baseline} paired")
            assert [entry[key] for key in ("wins", "ties", "losses")] == [
                sum(d < 0 for d in differences), sum(d == 0 for d in differences),
                sum(d > 0 for d in differences)]
            assert len(entry["bootstrap_ci95"]) == 2 and entry["bootstrap_ci95"][0] <= entry["bootstrap_ci95"][1]
    assert manifest["counts"] == {"episodes": len(episodes), "development": len(development),
                                  "training_steps": len(train) * steps[-1]}
    return {"status": "PASS", "run": str(run_dir), "commit": manifest["git"]["commit"],
            "episodes": len(episodes), "development_episodes": len(development),
            "checkpoints": len(training), "checks": ["source and artifact hashes", "seed splits",
            "checkpoint selection", "complete paired grid", "physical seed pairing",
            "economic accounting", "request conservation", "summary means and paired differences"]}


def main() -> None:
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
