"""Read-only same-weight residual-capacity intervention on selected v5 trajectories."""
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
from ht_pdm_fjsp.maintenance_framework_v5_policy import ACTIONS

MACHINE_ORDER = (2, 0, 3, 1)  # new index -> old index
WORKER_ORDER = (1, 0)
ACTION_MAP = tuple(ACTIONS.index(tuple(sorted(
    (MACHINE_ORDER.index(machine), WORKER_ORDER.index(worker))
    for machine, worker in matching))) for matching in ACTIONS)


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
    if manifest["status"] != "COMPLETED" or manifest["profile"] not in ("smoke", "full"):
        raise RuntimeError("requires a completed v5 campaign")
    selection = json.loads((campaign / "selection.json").read_text())
    registry = json.loads((campaign / "seed_registry.json").read_text())
    checkpoint = campaign / selection["postdecision"]["checkpoint"]
    if sha(checkpoint) != selection["postdecision"]["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint hash mismatch")
    torch.set_num_threads(1)
    model = MaskablePPO.load(checkpoint, device="cpu")
    actor = model.policy.action_net
    if actor.variant != "postdecision":
        raise RuntimeError("selected actor variant mismatch")
    panel = registry["sealed_test"] if manifest["profile"] == "full" else registry["smoke"]
    seeds = range(panel["start"], panel["start"] + panel["count"])
    expected = {}
    with (campaign / "episodes.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["controller"] == "postdecision":
                expected[(row["case"], int(row["seed"]))] = float(row["objective"])
    grid = [(case, seed) for case in CASES for seed in seeds]
    if len(expected) != len(grid):
        raise RuntimeError("incomplete selected-policy evaluation panel")
    fields = ("case", "seed", "tick", "normal_action", "zero_penalty_action",
              "decision_changed", "queued", "free_workers", "legal_actions",
              "max_legal_penalty", "chosen_penalty")
    by_case = {}
    permutation = dict(states=0, max_abs_logit_error=0.0,
                       non_tie_action_mismatches=0, ambiguous_ties=0)
    trace = output / "trace.csv"
    with trace.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for case, seed in tqdm(grid, desc="v5 common-state diagnostic", unit="episode"):
            env = FrameworkEnv(configs()[case])
            observation, _ = env.reset(seed=seed)
            for tick in range(48):
                mask = env.action_masks()
                tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
                with torch.no_grad():
                    normal = actor(tensor).numpy()[0]
                    _, machine, worker, edge = actor.split_observation(tensor)
                    penalty = actor.postdecision_penalty(machine, worker, edge).numpy()[0]
                    actor.variant = "edge_only"
                    try:
                        zero_penalty = actor(tensor).numpy()[0]
                    finally:
                        actor.variant = "postdecision"
                if np.max(np.abs(normal - (zero_penalty - penalty))) > 1e-5:
                    raise RuntimeError("intervention did not isolate residual penalty")
                normal_action = int(model.predict(observation, action_masks=mask,
                                                  deterministic=True)[0])
                actor.variant = "edge_only"
                try:
                    zero_action = int(model.predict(observation, action_masks=mask,
                                                    deterministic=True)[0])
                finally:
                    actor.variant = "postdecision"
                if not mask[normal_action] or not mask[zero_action]:
                    raise RuntimeError("policy selected illegal action")
                permuted_obs, permuted_mask = permute_state(observation, mask)
                with torch.no_grad():
                    permuted_logits = actor(torch.as_tensor(permuted_obs, dtype=torch.float32).reshape(1, -1)).numpy()[0]
                error = float(np.max(np.abs(normal - permuted_logits[list(ACTION_MAP)])))
                permutation["max_abs_logit_error"] = max(permutation["max_abs_logit_error"], error)
                if error > 1e-5:
                    raise RuntimeError(f"permutation logit error {error}")
                permuted_action = int(model.predict(permuted_obs, action_masks=permuted_mask,
                                                    deterministic=True)[0])
                if permuted_action != ACTION_MAP[normal_action]:
                    legal = np.sort(normal[mask])
                    gap = float(legal[-1] - legal[-2]) if len(legal) > 1 else float("inf")
                    if gap > 1e-5:
                        permutation["non_tie_action_mismatches"] += 1
                        raise RuntimeError("permutation changed unambiguous action")
                    permutation["ambiguous_ties"] += 1
                permutation["states"] += 1
                machine_features = observation[2:34].reshape(4, 8)
                worker_features = observation[34:40].reshape(2, 3)
                queued = int(((machine_features[:, 2] > .5) & (machine_features[:, 5] < .5)).sum())
                changed = int(normal_action != zero_action)
                record = dict(case=case, seed=seed, tick=tick,
                              normal_action=normal_action, zero_penalty_action=zero_action,
                              decision_changed=changed, queued=queued,
                              free_workers=int((worker_features[:, 0] > .5).sum()),
                              legal_actions=int(mask.sum()),
                              max_legal_penalty=float(penalty[mask].max()),
                              chosen_penalty=float(penalty[normal_action]))
                writer.writerow(record)
                row = by_case.setdefault(case, dict(states=0, with_queue=0,
                    with_positive_legal_penalty=0, changed=0, changed_with_queue=0))
                row["states"] += 1
                row["with_queue"] += int(queued > 0)
                row["with_positive_legal_penalty"] += int(record["max_legal_penalty"] > 1e-6)
                row["changed"] += changed
                row["changed_with_queue"] += changed * int(queued > 0)
                observation, _, done, _, _ = env.step(normal_action)
            if not done:
                raise RuntimeError("incomplete replay")
            if abs(env.episode_metrics()["objective"] - expected[(case, seed)]) > 1e-9:
                raise RuntimeError("replay cost differs from source campaign")
    summary = dict(label="post hoc common-state intervention; no closed-loop cost claim",
                   by_case=by_case, permutation=permutation)
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", dict(
        status="COMPLETED", source_run=str(campaign), source_commit=manifest["git"]["commit"],
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        episodes=len(grid), decisions=len(grid) * 48,
        trace_sha256=sha(trace), summary_sha256=sha(output / "summary.json")))
    print(json.dumps({"status": "COMPLETED", "output_dir": str(output),
                      "by_case": by_case, "permutation": permutation}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
