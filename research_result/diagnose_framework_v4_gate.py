"""Read-only common-state temporal-gate intervention on completed v4 TEST trajectories."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from torch.nn import functional as F
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, FrameworkEnv, configs
from ht_pdm_fjsp.maintenance_framework_v4_policy import ACTIONS


MACHINE_ORDER = (2, 0, 3, 1)  # new index -> old index
WORKER_ORDER = (1, 0)
ACTION_MAP = tuple(ACTIONS.index(tuple(sorted(
    (MACHINE_ORDER.index(machine), WORKER_ORDER.index(worker))
    for machine, worker in matching))) for matching in ACTIONS)


def permute_state(observation, mask):
    """Relabel a true observation, including the idle/busy worker assignment."""
    result = observation.copy()
    result[2:34] = observation[2:34].reshape(4, 8)[list(MACHINE_ORDER)].reshape(-1)
    workers = observation[34:40].reshape(2, 3)[list(WORKER_ORDER)].copy()
    for worker in workers:
        old_machine = round(float(worker[2]) * 5) - 1
        worker[2] = 0.0 if old_machine < 0 else (MACHINE_ORDER.index(old_machine) + 1) / 5
    result[34:40] = workers.reshape(-1)
    result[40:56] = observation[40:56].reshape(4, 2, 2)[list(MACHINE_ORDER)][:, list(WORKER_ORDER)].reshape(-1)
    new_mask = np.zeros_like(mask)
    new_mask[list(ACTION_MAP)] = mask
    return result, new_mask


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(campaign, output, selected_arm="temporal", ablated_arm="current_queue"):
    if (selected_arm, ablated_arm) not in (("temporal", "current_queue"), ("current_queue", "edge_only")):
        raise ValueError("unsupported intervention pair")
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((campaign / "manifest.json").read_text())
    if manifest["status"] != "COMPLETED" or manifest["profile"] != "full":
        raise RuntimeError("requires completed full campaign")
    selection = json.loads((campaign / "selection.json").read_text())
    registry = json.loads((campaign / "seed_registry.json").read_text())
    checkpoint = campaign / selection[selected_arm]["checkpoint"]
    if digest(checkpoint) != selection[selected_arm]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    torch.set_num_threads(1)
    model = MaskablePPO.load(checkpoint, device="cpu")
    actor = model.policy.action_net
    if actor.variant != selected_arm:
        raise RuntimeError("selected checkpoint variant mismatch")
    coefficient = float(F.softplus(actor.log_opportunity_scale).item())
    seeds = range(registry["sealed_test"]["start"],
                  registry["sealed_test"]["start"] + registry["sealed_test"]["count"])
    expected = {}
    with (campaign / "episodes.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["controller"] == selected_arm:
                expected[(row["case"], int(row["seed"]))] = float(row["objective"])
    grid = [(case, seed) for case in CASES for seed in seeds]
    if len(expected) != len(grid):
        raise RuntimeError("incomplete source temporal TEST grid")
    fields = ("case", "seed", "tick", "normal_action", "ablated_action",
              "argmax_changed", "near_competitors", "queued_pm", "queued_cm",
              "free_workers", "legal_actions", "max_abs_legal_logit_delta")
    trace = output / "trace.csv"
    summaries = {}
    permutation = dict(max_abs_logit_error=0.0, deterministic_mismatches=0,
                       ambiguous_ties=0, states=0)
    with trace.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for case, seed in tqdm(grid, desc="Opened TEST gate diagnostic", unit="episode"):
            env = FrameworkEnv(configs()[case])
            obs, _ = env.reset(seed=seed)
            for tick in range(48):
                mask = env.action_masks()
                tensor = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
                with torch.no_grad():
                    normal = actor(tensor).detach().cpu().numpy()[0]
                    actor.variant = ablated_arm
                    try:
                        ablated_logits = actor(tensor).detach().cpu().numpy()[0]
                    finally:
                        actor.variant = selected_arm
                normal_choice = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
                actor.variant = ablated_arm
                try:
                    ablated_choice = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
                finally:
                    actor.variant = selected_arm
                if not mask[normal_choice] or not mask[ablated_choice]:
                    raise RuntimeError("policy selected masked action")
                permuted_obs, permuted_mask = permute_state(obs, mask)
                with torch.no_grad():
                    permuted_logits = actor(torch.as_tensor(permuted_obs, dtype=torch.float32).reshape(1, -1)).numpy()[0]
                error = float(np.max(np.abs(normal - permuted_logits[list(ACTION_MAP)])))
                permutation["max_abs_logit_error"] = max(permutation["max_abs_logit_error"], error)
                if error > 1e-5:
                    raise RuntimeError(f"actor permutation logit error {error}")
                permuted_choice = int(model.predict(permuted_obs, action_masks=permuted_mask,
                                                    deterministic=True)[0])
                if permuted_choice != ACTION_MAP[normal_choice]:
                    legal_scores = np.sort(normal[mask])
                    gap = float(legal_scores[-1] - legal_scores[-2]) if len(legal_scores) > 1 else float("inf")
                    if gap > 1e-5:
                        permutation["deterministic_mismatches"] += 1
                        raise RuntimeError("policy permutation changed unambiguous choice")
                    permutation["ambiguous_ties"] += 1
                permutation["states"] += 1
                machine = obs[2:34].reshape(4, 8)
                near = ((machine[:, 0] >= 2 / 8 - 1e-6)
                        & (machine[:, 0] < 3 / 8 - 1e-6)
                        & (machine[:, 2] < .5) & (machine[:, 5] < .5)
                        & (machine[:, 1] < .5))
                row = dict(case=case, seed=seed, tick=tick,
                           normal_action=normal_choice, ablated_action=ablated_choice,
                           argmax_changed=int(normal_choice != ablated_choice),
                           near_competitors=int(near.sum()),
                           queued_pm=int(((machine[:, 2] > .5) & (machine[:, 3] < .5)).sum()),
                           queued_cm=int(((machine[:, 2] > .5) & (machine[:, 3] > .5)).sum()),
                           free_workers=int((obs[34:40].reshape(2, 3)[:, 0] > .5).sum()),
                           legal_actions=int(mask.sum()),
                           max_abs_legal_logit_delta=float(np.max(np.abs(
                               normal[mask] - ablated_logits[mask]))))
                writer.writerow(row)
                item = summaries.setdefault(case, dict(states=0, disagreements=0,
                    near_states=0, disagreements_with_near=0))
                item["states"] += 1
                item["disagreements"] += row["argmax_changed"]
                item["near_states"] += int(row["near_competitors"] > 0)
                item["disagreements_with_near"] += (
                    row["argmax_changed"] * int(row["near_competitors"] > 0))
                obs, _, done, _, _ = env.step(normal_choice)
            if not done:
                raise RuntimeError("incomplete policy replay")
            if abs(env.episode_metrics()["objective"] - expected[(case, seed)]) > 1e-9:
                raise RuntimeError("policy replay cost differs from campaign")
    write_json(output / "summary.json", dict(
        label="post hoc common-state intervention, no cost claim",
        selected_arm=selected_arm, ablated_arm=ablated_arm,
        opportunity_scale=coefficient, by_case=summaries,
        permutation=permutation))
    write_json(output / "manifest.json", dict(
        status="COMPLETED", source_run=str(campaign), source_commit=manifest["git"]["commit"],
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        episodes=len(grid), decisions=len(grid) * 48,
        trace_sha256=digest(trace), summary_sha256=digest(output / "summary.json")))
    print(json.dumps({"status": "COMPLETED", "output_dir": str(output),
                      "summary": summaries}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-arm", choices=("temporal", "current_queue"), default="temporal")
    parser.add_argument("--ablated-arm", choices=("current_queue", "edge_only"), default="current_queue")
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.output_dir.resolve(), args.selected_arm, args.ablated_arm)


if __name__ == "__main__":
    main()
