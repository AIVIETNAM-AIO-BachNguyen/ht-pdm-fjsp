"""Read-only v6 replay of voluntary WAIT decisions by selected checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, FrameworkEnv, configs

ARMS = ("flat", "unshared_edge", "shared_edge")


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
    selection = json.loads((campaign / "selection.json").read_text())
    training = registry["training"] if manifest["profile"] == "full" else registry["smoke"]
    train_seeds = [training["start"] + offset for offset in range(training["count"] if manifest["profile"] == "full" else 1)]
    physical = registry["sealed_test"] if manifest["profile"] == "full" else registry["smoke"]
    physical_seeds = [physical["start"] + offset for offset in range(physical["count"])]
    expected = {}
    with (campaign / "episodes.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["controller"] in ARMS:
                expected[(row["controller"], int(row["train_seed"]), row["case"], int(row["seed"]))] = float(row["objective"])
    grid = [(arm, train, case, seed) for arm in ARMS for train in train_seeds
            for case in CASES for seed in physical_seeds]
    if len(expected) != len(grid):
        raise RuntimeError("incomplete source learned-policy TEST grid")
    fields = ("arm", "train_seed", "case", "physical_seed", "legal_service_ticks",
              "voluntary_wait_ticks", "wait_ticks", "service_starts", "objective")
    aggregate = {}
    checkpoints = {}
    with (output / "episodes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        current_key = None
        model = None
        for arm, train, case, seed in tqdm(grid, desc="v6 WAIT replay", unit="episode"):
            key = (arm, train)
            if key != current_key:
                item = selection[arm][str(train)]
                path = campaign / item["checkpoint"]
                if sha(path) != item["checkpoint_sha256"]:
                    raise RuntimeError("checkpoint hash mismatch")
                checkpoints[f"{arm}/{train}"] = dict(path=str(path), sha256=sha(path))
                model = MaskablePPO.load(path, device="cpu")
                current_key = key
            env = FrameworkEnv(configs()[case])
            observation, _ = env.reset(seed=seed)
            legal_service_ticks = voluntary_wait_ticks = wait_ticks = 0
            for _ in range(48):
                mask = env.action_masks()
                legal_service = bool(mask[1:].any())
                action = int(model.predict(observation, action_masks=mask, deterministic=True)[0])
                if not mask[action]:
                    raise RuntimeError("selected masked action")
                legal_service_ticks += legal_service
                wait_ticks += action == 0
                voluntary_wait_ticks += action == 0 and legal_service
                observation, _, done, _, _ = env.step(action)
            cost = float(env.episode_metrics()["objective"])
            if not done or abs(cost - expected[(arm, train, case, seed)]) > 1e-9:
                raise RuntimeError("replay objective differs from campaign")
            row = dict(arm=arm, train_seed=train, case=case, physical_seed=seed,
                       legal_service_ticks=legal_service_ticks,
                       voluntary_wait_ticks=voluntary_wait_ticks,
                       wait_ticks=wait_ticks,
                       service_starts=env.episode_metrics()["service_starts"],
                       objective=cost)
            writer.writerow(row)
            cell = aggregate.setdefault(arm, {}).setdefault(str(train),
                dict(episodes=0, legal_service_ticks=0, voluntary_wait_ticks=0,
                     wait_ticks=0, service_starts=0))
            cell["episodes"] += 1
            for field in ("legal_service_ticks", "voluntary_wait_ticks", "wait_ticks", "service_starts"):
                cell[field] += row[field]
    for arm in ARMS:
        for item in aggregate[arm].values():
            item["voluntary_wait_per_legal_service_tick"] = (
                item["voluntary_wait_ticks"] / item["legal_service_ticks"]
                if item["legal_service_ticks"] else 0)
            item["service_starts_per_episode"] = item["service_starts"] / item["episodes"]
    write_json(output / "summary.json", dict(
        label="post hoc selected-policy replay, not a new cost endpoint", by_arm=aggregate))
    write_json(output / "manifest.json", dict(
        status="COMPLETED", source_run=str(campaign), source_commit=manifest["git"]["commit"],
        checkpoints=checkpoints, episodes=len(grid), decisions=len(grid)*48,
        episodes_sha256=sha(output / "episodes.csv"), summary_sha256=sha(output / "summary.json")))
    print(json.dumps({"status": "COMPLETED", "output_dir": str(output), "summary": aggregate}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
