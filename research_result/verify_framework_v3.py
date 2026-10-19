"""Independent raw-artifact audit for rollout-teacher maintenance PPO."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from ht_pdm_fjsp.maintenance_framework_v1 import FrameworkEnv, configs

VERSION = "maintenance_framework_v3_teacher_ppo"
CASES = ("reciprocal_scale2", "interchangeable_scale1", "reciprocal_scale3",
         "specialist_generalist_scale2")
CONTROLLERS = ("fcfs", "risk_matching", "rollout", "scratch_ppo", "warm_bc", "teacher_ppo")
LEARNED = ("scratch_ppo", "warm_bc", "teacher_ppo")
COMPARATORS = ("scratch_ppo", "warm_bc", "fcfs", "risk_matching", "rollout")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def panel(registry, name):
    spec = registry[name]
    return list(range(spec["start"], spec["start"] + spec["count"]))


def role_seed(seed, role):
    return int.from_bytes(hashlib.sha256(f"{seed}:{role}".encode()).digest()[:8], "big")


def close(a, b, label):
    if not math.isclose(float(a), float(b), abs_tol=1e-8, rel_tol=0):
        raise AssertionError(f"{label}: {a} != {b}")


def verify(run_dir):
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((run_dir / "manifest.json").read_text())
    registry = json.loads((run_dir / "seed_registry.json").read_text())
    config = json.loads((run_dir / "resolved_config.json").read_text())
    bank_meta = json.loads((run_dir / "teacher_bank_meta.json").read_text())
    training_audit = json.loads((run_dir / "training_audit.json").read_text())
    sequences = json.loads((run_dir / "training_seed_sequences.json").read_text())
    selections = json.loads((run_dir / "selection.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    development = load_csv(run_dir / "development.csv")
    episodes = load_csv(run_dir / "episodes.csv")
    bc = load_csv(run_dir / "bc_progress.csv")
    assert manifest["status"] == "COMPLETED"
    assert manifest["protocol"] == VERSION == registry["campaign_version"]
    assert manifest["seed_registry"] == registry
    assert set(config["cases"]) == set(CASES) and config["train_case"] == "reciprocal_scale2"
    groups = [panel(registry, name) for name in
              ("training", "development", "sealed_test", "smoke", "teacher_bank")]
    groups += [[registry["forecast_seed"]], [registry["bootstrap_seed"]]]
    flat = [seed for group in groups for seed in group]
    assert len(flat) == len(set(flat)), "seed overlap"
    full = manifest["profile"] == "full"
    train = groups[0] if full else groups[3][:1]
    dev = groups[1] if full else groups[3]
    test = groups[2] if full else groups[3]
    steps = [50176, 100352, 149504, 199680] if full else [1024]
    bank_count = 32 if full else 2
    epochs = 10 if full else 2
    assert config["steps"] == steps
    assert config["teacher_bank_episodes_per_replica"] == bank_count
    assert config["bc_epochs"] == epochs and config["bc_batch_size"] == 256
    for relative, expected in manifest["source_hashes"].items():
        assert sha(root / relative) == expected, f"source: {relative}"
    for name, expected in manifest["artifact_hashes"].items():
        assert sha(run_dir / name) == expected, f"artifact: {name}"
    assert (run_dir / "protocol.md").read_bytes() == (root / "research_result/framework_v3_protocol_vi.md").read_bytes()
    assert set(bank_meta) == set(training_audit) == set(sequences) == {str(seed) for seed in train}
    total_transitions = 0
    for index, seed in enumerate(train):
        item = bank_meta[str(seed)]
        assigned = groups[4][index * 32:index * 32 + bank_count]
        assert item["physical_seeds"] == assigned
        assert item["episodes"] == bank_count and item["labels"] == bank_count * 48
        bank_path = run_dir / "teacher_bank" / f"{seed}.npz"
        assert sha(bank_path) == item["sha256"]
        with np.load(bank_path) as bank:
            obs, masks, actions = bank["observations"], bank["masks"], bank["actions"]
            assert obs.shape == (bank_count * 48, 56) and masks.shape == (bank_count * 48, 21)
            assert actions.shape == (bank_count * 48,)
            assert bool(masks[np.arange(len(actions)), actions].all()), "illegal teacher label"
            assert bank["episode_seeds"].tolist() == [raw for raw in assigned for _ in range(48)]
            assert bank["ticks"].tolist() == list(range(48)) * bank_count
            assert not np.isnan(obs).any()
            replay = FrameworkEnv(configs()["reciprocal_scale2"])
            replay_costs = []
            for episode_index, raw in enumerate(assigned):
                replay.reset(seed=raw)
                for tick in range(48):
                    row_index = episode_index * 48 + tick
                    np.testing.assert_array_equal(replay._observation(), obs[row_index])
                    np.testing.assert_array_equal(replay.action_masks(), masks[row_index])
                    replay.step(int(actions[row_index]))
                replay_costs.append(replay.inner.metrics["objective"])
            close(mean(replay_costs), item["mean_teacher_cost"], "teacher bank replay cost")
        total_transitions += item["forecast_transitions"]
        pair = sequences[str(seed)]
        assert pair["scratch_ppo"] == pair["teacher_ppo"]
        expected_sequence = [seed] + np.random.default_rng(seed).integers(
            0, 2**31, size=len(pair["scratch_ppo"]) - 1).tolist()
        assert pair["scratch_ppo"] == expected_sequence
        digest = hashlib.sha256(json.dumps(pair["scratch_ppo"], separators=(",", ":")).encode()).hexdigest()
        audit = training_audit[str(seed)]
        assert audit["physical_sequence_sha256"] == digest
        assert audit["training_episode_seed_count"] == len(pair["scratch_ppo"])
        assert audit["scratch_initial_policy_sha256"] == audit["teacher_initial_policy_sha256"]
        assert audit["paired_sequence_equal"] is True
        assert audit["bc"]["epochs"] == epochs and audit["bc"]["labels"] == bank_count * 48
        expected_updates = epochs * math.ceil(bank_count * 48 / 256)
        assert audit["bc"]["optimizer_updates"] == expected_updates
    assert len(bc) == len(train) * epochs
    for seed in train:
        subset = [row for row in bc if int(row["train_seed"]) == seed]
        assert [int(row["epoch"]) for row in subset] == list(range(1, epochs + 1))
        assert all(math.isfinite(float(row["mean_nll"])) for row in subset)
    checkpoints = []
    for arm in ("scratch_ppo", "teacher_ppo"):
        progress = load_csv(run_dir / "arms" / arm / "training_progress.csv")
        assert len(progress) == len(train) * len(steps)
        assert {(int(row["train_seed"]), int(row["steps"])) for row in progress} == (
            {(seed, step) for seed in train for step in steps})
        for row in progress:
            path = run_dir / "arms" / arm / row["checkpoint"]
            assert path.is_file()
            checkpoints.append(path)
    for seed in train:
        assert (run_dir / "arms" / "warm_bc" / f"{seed}.zip").is_file()
    expected_dev = {(arm, seed, step, physical) for arm in LEARNED for seed in train
                    for step in ((0,) if arm == "warm_bc" else steps) for physical in dev}
    actual_dev = {(row["controller"], int(row["train_seed"]), int(row["checkpoint_step"]), int(row["seed"]))
                  for row in development}
    assert len(development) == len(expected_dev) and actual_dev == expected_dev
    assert set(selections) == set(LEARNED)
    for arm in LEARNED:
        scores = {}
        for row in development:
            if row["controller"] == arm:
                scores.setdefault((int(row["train_seed"]), int(row["checkpoint_step"])), []).append(float(row["objective"]))
        selected = min((mean(values), seed, step) for (seed, step), values in scores.items())
        item = selections[arm]
        close(selected[0], item["development_mean_cost"], arm + " selection")
        assert (item["train_seed"], item["steps"]) == selected[1:]
        assert sha(run_dir / item["checkpoint"]) == item["checkpoint_sha256"]
    assert sha(run_dir / "selection.json") == manifest["selection_sha256"]
    expected_grid = {(case, controller, physical) for case in CASES
                     for controller in CONTROLLERS for physical in test}
    actual_grid = {(row["case"], row["controller"], int(row["seed"])) for row in episodes}
    assert len(episodes) == len(expected_grid) and actual_grid == expected_grid
    for row in episodes:
        label = f'{row["case"]}/{row["controller"]}/{row["seed"]}'
        derived = role_seed(int(row["seed"]), registry["version"] + ":physical")
        assert int(row["physical_seed"]) == derived, label
        close(row["objective"], sum(float(row[key]) for key in
              ("maintenance_cost", "unavailability_cost", "failure_cost")), label + " components")
        close(row["objective"], -float(row["episode_return"]), label + " reward")
        close(row["objective"], float(row["H_cost"]) + float(row["tail_cost"]), label + " tail")
        assert float(row["invalid_assignments"]) == 0, label
        assert 0 <= float(row["selection_seconds"]) <= float(row["episode_wall_seconds"]), label
        close(float(row["requests_created"]) - float(row["requests_completed"]),
              float(row["end_pending"]) + float(row["end_in_service"]), label + " requests")
        close(float(row["service_starts"]) - float(row["requests_completed"]),
              float(row["end_in_service"]), label + " service")
        close(row["request_wait_ticks"], float(row["queued_pm_ticks"]) + float(row["queued_cm_ticks"]),
              label + " waiting")
    for case in CASES:
        by_controller = {name: sorted((row for row in episodes if row["case"] == case
                                       and row["controller"] == name), key=lambda row: int(row["seed"]))
                         for name in CONTROLLERS}
        for name, panel_rows in by_controller.items():
            item = summary[f"{case}/{name}"]
            assert item["count"] == len(test)
            for metric, reported in item["means"].items():
                close(mean(float(row[metric]) for row in panel_rows), reported, f"{case}/{name}/{metric}")
        teacher = by_controller["teacher_ppo"]
        for comparator in COMPARATORS:
            other = by_controller[comparator]
            assert [int(row["seed"]) for row in teacher] == [int(row["seed"]) for row in other]
            delta = np.asarray([float(a["objective"]) - float(b["objective"])
                                for a, b in zip(teacher, other)])
            key = f"{case}/teacher_minus_{comparator}"
            item = summary[key]
            close(delta.mean(), item["paired_mean"], key + " mean")
            assert (item["wins"], item["ties"], item["losses"]) == (
                int((delta < 0).sum()), int((delta == 0).sum()), int((delta > 0).sum()))
            primary = case == "reciprocal_scale2" and comparator in ("scratch_ppo", "warm_bc")
            rng = np.random.default_rng(role_seed(registry["bootstrap_seed"], f"{VERSION}:{key}"))
            indices = rng.integers(0, len(delta), size=(20000, len(delta)))
            interval = np.quantile(delta[indices].mean(axis=1), [.0125, .9875] if primary else [.025, .975])
            for actual, reported in zip(interval, item["ci"]):
                close(actual, reported, key + " CI")
            assert item["ci_level"] == (.975 if primary else .95)
            assert item["confirmed"] == (bool(delta.mean() < 0 and interval[1] < 0) if primary else None)
    assert summary["co_primary_both_confirmed"] == bool(
        summary["reciprocal_scale2/teacher_minus_scratch_ppo"]["confirmed"] and
        summary["reciprocal_scale2/teacher_minus_warm_bc"]["confirmed"])
    counts = manifest["counts"]
    assert counts == {"teacher_bank_episodes": len(train) * bank_count,
        "teacher_labels": len(train) * bank_count * 48,
        "bc_optimizer_updates": sum(item["bc"]["optimizer_updates"] for item in training_audit.values()),
        "ppo_training_steps": len(train) * 2 * steps[-1],
        "development": len(development), "evaluation": len(episodes),
        "ppo_checkpoints": len(checkpoints), "warm_checkpoints": len(train)}
    return {"status": "PASS", "run": str(run_dir), "commit": manifest["git"]["commit"],
        "profile": manifest["profile"], "teacher_bank_episodes": counts["teacher_bank_episodes"],
        "teacher_labels": counts["teacher_labels"], "teacher_forecast_transitions": total_transitions,
        "ppo_steps": counts["ppo_training_steps"], "development": len(development),
        "evaluation": len(episodes), "co_primary_both_confirmed": summary["co_primary_both_confirmed"],
        "primary_teacher_minus_scratch": summary["reciprocal_scale2/teacher_minus_scratch_ppo"],
        "primary_teacher_minus_warm": summary["reciprocal_scale2/teacher_minus_warm_bc"],
        "checks": ["source/artifact/bank/checkpoint hashes", "fresh split/grid",
            "teacher-bank physical replay and label legality", "paired training seed sequences",
            "initial policy hashes", "BC/update counts", "DEV-only selections",
            "episode economics/request conservation", "all paired bootstrap intervals"]}


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
