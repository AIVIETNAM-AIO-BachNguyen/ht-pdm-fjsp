"""Read-only selected shared-policy matching size and duration replay for v6."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, FrameworkEnv, configs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(campaign, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((campaign / "manifest.json").read_text())
    if manifest["status"] != "COMPLETED" or manifest["protocol"] != "maintenance_framework_v6_edge_sharing":
        raise RuntimeError("requires completed v6 campaign")
    registry = json.loads((campaign / "seed_registry.json").read_text())
    selection = json.loads((campaign / "selection.json").read_text())["shared_edge"]
    train_spec = registry["training"] if manifest["profile"] == "full" else registry["smoke"]
    train = [train_spec["start"] + n for n in range(train_spec["count"] if manifest["profile"] == "full" else 1)]
    test_spec = registry["sealed_test"] if manifest["profile"] == "full" else registry["smoke"]
    physical = [test_spec["start"] + n for n in range(test_spec["count"])]
    expected = {}
    with (campaign / "episodes.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["controller"] == "shared_edge":
                expected[(int(row["train_seed"]), row["case"], int(row["seed"]))] = float(row["objective"])
    grid = [(training_seed, case, seed) for training_seed in train for case in CASES for seed in physical]
    if len(expected) != len(grid):
        raise RuntimeError("incomplete selected shared-policy TEST grid")
    fields = ("train_seed", "case", "physical_seed", "service_starts", "underfilled_extendable_ticks",
              "two_service_legal_ticks", "two_service_chosen_ticks", "same_set_duration_excess_ticks",
              "same_set_duration_excess_sum", "objective")
    summary = {}
    checkpoints = {}
    current_train = None
    model = None
    with (output / "episodes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for training_seed, case, seed in tqdm(grid, desc="v6 matching replay", unit="episode"):
            if training_seed != current_train:
                item = selection[str(training_seed)]
                checkpoint = campaign / item["checkpoint"]
                if sha(checkpoint) != item["checkpoint_sha256"]:
                    raise RuntimeError("checkpoint hash mismatch")
                checkpoints[str(training_seed)] = dict(path=str(checkpoint), sha256=sha(checkpoint))
                model = MaskablePPO.load(checkpoint, device="cpu")
                current_train = training_seed
            env = FrameworkEnv(configs()[case])
            observation, _ = env.reset(seed=seed)
            counters = dict(service_starts=0, underfilled_extendable_ticks=0,
                            two_service_legal_ticks=0, two_service_chosen_ticks=0,
                            same_set_duration_excess_ticks=0, same_set_duration_excess_sum=0.0)
            for _ in range(48):
                mask = env.action_masks()
                legal = [env.actions[i] for i, allowed in enumerate(mask) if allowed]
                action = int(model.predict(observation, action_masks=mask, deterministic=True)[0])
                if not mask[action]:
                    raise RuntimeError("masked action")
                matching = env.actions[action]
                selected = set(matching)
                max_size = max(map(len, legal))
                counters["service_starts"] += len(matching)
                counters["underfilled_extendable_ticks"] += any(selected < set(other) for other in legal)
                counters["two_service_legal_ticks"] += max_size == 2
                counters["two_service_chosen_ticks"] += len(matching) == 2
                machines = {machine for machine, _ in matching}
                duration = sum(env.config.service_time[machine][worker] for machine, worker in matching)
                same_set = [other for other in legal if {machine for machine, _ in other} == machines]
                best_duration = min(sum(env.config.service_time[machine][worker]
                                        for machine, worker in other) for other in same_set)
                excess = duration - best_duration
                counters["same_set_duration_excess_ticks"] += excess > 0
                counters["same_set_duration_excess_sum"] += excess
                observation, _, done, _, _ = env.step(action)
            cost = float(env.episode_metrics()["objective"])
            if not done or abs(cost - expected[(training_seed, case, seed)]) > 1e-9:
                raise RuntimeError("episode replay cost mismatch")
            writer.writerow(dict(train_seed=training_seed, case=case, physical_seed=seed,
                                 **counters, objective=cost))
            cell = summary.setdefault(str(training_seed), {}).setdefault(case,
                dict(episodes=0, **{name: 0 for name in counters}))
            cell["episodes"] += 1
            for name, value in counters.items():
                cell[name] += value
    write_json(output / "summary.json", dict(label="post hoc matching diagnostic; no cost claim",
                                                  by_train_case=summary))
    write_json(output / "manifest.json", dict(status="COMPLETED", source_run=str(campaign),
        source_commit=manifest["git"]["commit"], checkpoints=checkpoints,
        episodes=len(grid), decisions=len(grid)*48,
        episodes_sha256=sha(output / "episodes.csv"), summary_sha256=sha(output / "summary.json")))
    print(json.dumps({"status": "COMPLETED", "output_dir": str(output),
                      "episodes": len(grid)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
