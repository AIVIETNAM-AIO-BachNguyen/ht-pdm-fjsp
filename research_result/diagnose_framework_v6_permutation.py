"""Read-only true-state label permutation audit for selected v6 actors."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, FrameworkEnv, configs
from ht_pdm_fjsp.maintenance_framework_v6_policy import ACTIONS

MACHINE_ORDER = (2, 0, 3, 1)  # new index -> old index
WORKER_ORDER = (1, 0)
ACTION_MAP = tuple(ACTIONS.index(tuple(sorted(
    (MACHINE_ORDER.index(machine), WORKER_ORDER.index(worker))
    for machine, worker in matching))) for matching in ACTIONS)
ARMS = ("shared_edge", "unshared_edge")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, item):
    path.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n")


def permute_state(observation, mask):
    result = observation.copy()
    result[2:34] = observation[2:34].reshape(4, 8)[list(MACHINE_ORDER)].reshape(-1)
    workers = observation[34:40].reshape(2, 3)[list(WORKER_ORDER)].copy()
    for worker in workers:
        old_machine = round(float(worker[2]) * 5) - 1
        worker[2] = 0 if old_machine < 0 else (MACHINE_ORDER.index(old_machine) + 1) / 5
    result[34:40] = workers.reshape(-1)
    result[40:56] = observation[40:56].reshape(4, 2, 2)[list(MACHINE_ORDER)][:, list(WORKER_ORDER)].reshape(-1)
    new_mask = np.zeros_like(mask)
    new_mask[list(ACTION_MAP)] = mask
    return result, new_mask


def run(campaign, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((campaign / "manifest.json").read_text())
    if manifest["status"] != "COMPLETED" or manifest["protocol"] != "maintenance_framework_v6_edge_sharing":
        raise RuntimeError("requires completed v6 campaign")
    registry = json.loads((campaign / "seed_registry.json").read_text())
    selection = json.loads((campaign / "selection.json").read_text())
    train_spec = registry["training"] if manifest["profile"] == "full" else registry["smoke"]
    train_seeds = [train_spec["start"] + offset for offset in range(train_spec["count"] if manifest["profile"] == "full" else 1)]
    physical_spec = registry["sealed_test"] if manifest["profile"] == "full" else registry["smoke"]
    physical_seeds = [physical_spec["start"] + offset for offset in range(physical_spec["count"])]
    expected = {}
    with (campaign / "episodes.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["controller"] in ARMS:
                expected[(row["controller"], int(row["train_seed"]), row["case"], int(row["seed"]))] = float(row["objective"])
    grid = [(arm, train, case, physical)
            for arm in ARMS for train in train_seeds for case in CASES for physical in physical_seeds]
    if len(expected) != len(grid):
        raise RuntimeError("incomplete selected policy episode grid")
    torch.set_num_threads(1)
    fields = ("arm", "train_seed", "case", "physical_seed", "tick",
              "normal_action", "permuted_mapped_action", "action_changed",
              "max_abs_logit_error", "legal_score_gap")
    summary = {arm: dict(states=0, action_changed=0, unambiguous_changes=0,
                         max_abs_logit_error=0.0, mean_abs_logit_error_sum=0.0)
               for arm in ARMS}
    checkpoints = {}
    with (output / "trace.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        current_key = None
        model = None
        for arm, train, case, physical in tqdm(grid, desc="v6 TEST permutation diagnostic", unit="episode"):
            key = (arm, train)
            if key != current_key:
                item = selection[arm][str(train)]
                path = campaign / item["checkpoint"]
                if sha(path) != item["checkpoint_sha256"]:
                    raise RuntimeError("checkpoint hash mismatch")
                checkpoints[f"{arm}/{train}"] = dict(path=str(path), sha256=sha(path))
                model = MaskablePPO.load(path, device="cpu")
                if model.policy.action_net.variant != arm:
                    raise RuntimeError("checkpoint actor variant mismatch")
                current_key = key
            actor = model.policy.action_net
            env = FrameworkEnv(configs()[case])
            observation, _ = env.reset(seed=physical)
            for tick in range(48):
                mask = env.action_masks()
                relabeled, relabeled_mask = permute_state(observation, mask)
                tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
                tensor_relabel = torch.as_tensor(relabeled, dtype=torch.float32).reshape(1, -1)
                with torch.no_grad():
                    logits = actor(tensor).numpy()[0]
                    relabeled_logits = actor(tensor_relabel).numpy()[0]
                errors = np.abs(logits - relabeled_logits[list(ACTION_MAP)])
                max_error = float(errors.max())
                normal_action = int(model.predict(observation, action_masks=mask,
                                                  deterministic=True)[0])
                relabeled_action = int(model.predict(relabeled, action_masks=relabeled_mask,
                                                     deterministic=True)[0])
                if not mask[normal_action] or not relabeled_mask[relabeled_action]:
                    raise RuntimeError("policy selected illegal action")
                mapped_action = ACTION_MAP.index(relabeled_action)
                changed = int(mapped_action != normal_action)
                legal = np.sort(logits[mask])
                gap = float(legal[-1] - legal[-2]) if len(legal) > 1 else float("inf")
                if arm == "shared_edge" and (max_error > 1e-5 or (changed and gap > 1e-5)):
                    raise RuntimeError("shared actor lost true-state equivariance")
                record = dict(arm=arm, train_seed=train, case=case,
                              physical_seed=physical, tick=tick,
                              normal_action=normal_action,
                              permuted_mapped_action=mapped_action,
                              action_changed=changed,
                              max_abs_logit_error=max_error,
                              legal_score_gap=gap)
                writer.writerow(record)
                item = summary[arm]
                item["states"] += 1
                item["action_changed"] += changed
                item["unambiguous_changes"] += int(changed and gap > 1e-5)
                item["max_abs_logit_error"] = max(item["max_abs_logit_error"], max_error)
                item["mean_abs_logit_error_sum"] += float(errors.mean())
                observation, _, done, _, _ = env.step(normal_action)
            if not done or abs(env.episode_metrics()["objective"] - expected[(arm, train, case, physical)]) > 1e-9:
                raise RuntimeError("selected-policy replay differs from campaign")
    for arm in ARMS:
        item = summary[arm]
        item["mean_abs_logit_error"] = item.pop("mean_abs_logit_error_sum") / item["states"]
    write_json(output / "summary.json", dict(
        label="post hoc true-state relabeling; no new cost claim",
        machine_order=MACHINE_ORDER, worker_order=WORKER_ORDER, by_arm=summary))
    write_json(output / "manifest.json", dict(
        status="COMPLETED", source_run=str(campaign), source_commit=manifest["git"]["commit"],
        checkpoints=checkpoints, episodes=len(grid), decisions=len(grid) * 48,
        trace_sha256=sha(output / "trace.csv"), summary_sha256=sha(output / "summary.json")))
    print(json.dumps({"status": "COMPLETED", "output_dir": str(output), "summary": summary}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
