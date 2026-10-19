"""Rollout-teacher imitation and matched PPO follow-up on maintenance dispatch."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import (
    CASES, HORIZON, FrameworkEnv, LearnedController, configs, evaluate_episode,
)
from ht_pdm_fjsp.maintenance_framework_v1_campaign import (
    FULL_STEPS, ProgressCallback, add_row, check_checkpoint, git_info,
    model_new, sha, write_json,
)
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
from ht_pdm_fjsp.maintenance_request_policies import choose
from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests

ROOT = Path(__file__).resolve().parents[2]
VERSION = "maintenance_framework_v3_teacher_ppo"
PROTOCOL = ROOT / "research_result/framework_v3_protocol_vi.md"
REGISTRY = ROOT / "configs/maintenance_framework_v3_seed_registry.json"
SOURCES = tuple(ROOT / path for path in (
    "src/ht_pdm_fjsp/maintenance_framework_v1.py",
    "src/ht_pdm_fjsp/maintenance_framework_v1_campaign.py",
    "src/ht_pdm_fjsp/maintenance_framework_v3_campaign.py",
    "src/ht_pdm_fjsp/maintenance_dispatch.py",
    "src/ht_pdm_fjsp/maintenance_requests.py",
    "src/ht_pdm_fjsp/maintenance_request_policies.py",
    "research_result/framework_v3_protocol_vi.md",
    "configs/maintenance_framework_v3_seed_registry.json",
    "pyproject.toml", "uv.lock",
))
CONTROLLERS = ("fcfs", "risk_matching", "rollout", "scratch_ppo", "warm_bc", "teacher_ppo")
LEARNED = ("scratch_ppo", "warm_bc", "teacher_ppo")


class LoggedFrameworkEnv(FrameworkEnv):
    """Record raw episode seeds to audit paired exogenous training streams."""

    def __init__(self, config):
        self.episode_seeds = []
        super().__init__(config)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.episode_seeds.append(self.seed_raw)
        return observation, info


class V3RuleController:
    def __init__(self, name, forecast_seed, domain, check_timeout=lambda: None):
        if name not in ("fcfs", "risk_matching", "rollout"):
            raise ValueError(name)
        self.name = name
        self.forecast_seed = forecast_seed
        self.domain = domain
        self.check_timeout = check_timeout
        self.forecast_transitions = 0

    def choose(self, config, state, tick, *, case, seed):
        forecast = role_seed(self.forecast_seed, f"{VERSION}:{self.domain}:{case}:{seed}:{tick}")
        submissions, matching, diagnostic = choose(
            config, state, tick, "threshold", self.name,
            policy_horizon=HORIZON, forecast_seed=forecast,
            scenarios=16, forecast_horizon=6, check_timeout=self.check_timeout,
        )
        self.forecast_transitions += diagnostic["forecast_transitions"]
        return submissions, matching


def seed_registry():
    registry = json.loads(REGISTRY.read_text())
    historical = set()
    for path in (ROOT / "configs").glob("*seed_registry.json"):
        if path != REGISTRY:
            historical |= registered_integers(json.loads(path.read_text()))
    def panel(name):
        spec = registry[name]
        return list(range(spec["start"], spec["start"] + spec["count"]))
    groups = [panel(name) for name in ("training", "development", "sealed_test", "smoke", "teacher_bank")]
    groups += [[registry["forecast_seed"]], [registry["bootstrap_seed"]]]
    flat = [seed for group in groups for seed in group]
    if len(flat) != len(set(flat)) or set(flat) & historical:
        raise RuntimeError("v3 seed overlap with historical registry or split")
    if registry["teacher_bank"]["count"] != registry["training"]["count"] * registry["teacher_bank"]["per_training_seed"]:
        raise RuntimeError("teacher bank grouping mismatch")
    return registry, groups


def source_info():
    info = git_info()
    relative = str(PROTOCOL.relative_to(ROOT))
    paths = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", relative],
                                    cwd=ROOT, text=True).splitlines()
    paths += subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", relative],
                                     cwd=ROOT, text=True).splitlines()
    info["blocking_paths"] = sorted(set(info["blocking_paths"] + paths))
    return info


def policy_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def seed_sequence_digest(seeds):
    return hashlib.sha256(json.dumps(seeds, separators=(",", ":")).encode()).hexdigest()


def create_teacher_bank(out, train_seed, physical_seeds, config, forecast_seed, deadline_check, bar):
    observations, masks, actions, episode_ids, ticks = [], [], [], [], []
    wall_start = time.perf_counter()
    forecast_transitions = 0
    teacher_costs = []
    selection_seconds = 0.
    env = FrameworkEnv(config)
    for raw_seed in physical_seeds:
        deadline_check()
        env.reset(seed=raw_seed)
        teacher = V3RuleController("rollout", forecast_seed, "bank", deadline_check)
        for tick in range(HORIZON):
            observation = env._observation()
            mask = env.action_masks()
            start = time.perf_counter()
            submissions, matching = teacher.choose(config, env.inner.state, tick,
                                                    case="reciprocal_scale2", seed=raw_seed)
            selection_seconds += time.perf_counter() - start
            admitted = submit_requests(config, env.inner.state, tick, submissions)
            if matching not in legal_matchings(config, admitted):
                raise RuntimeError("illegal teacher label")
            index = env.actions.index(matching)
            if not mask[index]:
                raise RuntimeError("teacher label masked by environment")
            observations.append(observation)
            masks.append(mask)
            actions.append(index)
            episode_ids.append(raw_seed)
            ticks.append(tick)
            env.step(index)
        if abs(env.episode_return + env.inner.metrics["objective"]) > 1e-9:
            raise RuntimeError("teacher reward/cost mismatch")
        teacher_costs.append(env.inner.metrics["objective"])
        forecast_transitions += teacher.forecast_transitions
        bar.update(1)
    bank_dir = out / "teacher_bank"
    bank_dir.mkdir(exist_ok=True)
    path = bank_dir / f"{train_seed}.npz"
    np.savez_compressed(path, observations=np.asarray(observations, np.float32),
                        masks=np.asarray(masks, bool), actions=np.asarray(actions, np.int64),
                        episode_seeds=np.asarray(episode_ids, np.int64), ticks=np.asarray(ticks, np.int16))
    return path, {"training_seed": train_seed, "physical_seeds": list(physical_seeds),
        "episodes": len(physical_seeds), "labels": len(actions), "sha256": sha(path),
        "mean_teacher_cost": float(np.mean(teacher_costs)),
        "forecast_transitions": forecast_transitions,
        "selection_seconds": selection_seconds, "wall_seconds": time.perf_counter() - wall_start}


def pretrain_actor(model, bank_path, train_seed, epochs, progress_path, deadline_check):
    data = np.load(bank_path)
    observations = torch.as_tensor(data["observations"], dtype=torch.float32, device=model.device)
    masks = torch.as_tensor(data["masks"], dtype=torch.bool, device=model.device)
    actions = torch.as_tensor(data["actions"], dtype=torch.long, device=model.device)
    if len(observations) != len(actions) or not bool(masks.gather(1, actions[:, None]).all()):
        raise RuntimeError("illegal/incomplete teacher bank")
    rng = np.random.default_rng(role_seed(train_seed, VERSION + ":bc_shuffle"))
    optimizer = model.policy.optimizer
    model.policy.set_training_mode(True)
    started = time.perf_counter()
    updates = 0
    for epoch in tqdm(range(1, epochs + 1), desc=f"BC {train_seed}", unit="epoch"):
        deadline_check()
        nll_total = 0.
        for indices in np.array_split(rng.permutation(len(actions)), math.ceil(len(actions) / 256)):
            index = torch.as_tensor(indices, dtype=torch.long, device=model.device)
            _, log_probability, _ = model.policy.evaluate_actions(
                observations[index], actions[index], action_masks=masks[index])
            loss = -log_probability.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite BC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            nll_total += float(loss.detach()) * len(indices)
            updates += 1
        add_row(progress_path, {"train_seed": train_seed, "epoch": epoch,
                                "mean_nll": nll_total / len(actions), "optimizer_updates": updates,
                                "elapsed_seconds": time.perf_counter() - started})
    optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)
    model.policy.set_training_mode(False)
    return {"epochs": epochs, "labels": len(actions), "optimizer_updates": updates,
            "wall_seconds": time.perf_counter() - started}


def paired_summary(rows, bootstrap_seed):
    result = {}
    by_case = {}
    metrics = ("objective", "H_availability", "H_failures", "queued_cm_ticks",
               "queued_pm_ticks", "H_pending", "end_pending", "H_in_service",
               "end_in_service", "maintenance_cost", "unavailability_cost",
               "failure_cost", "tail_cost", "selection_seconds", "episode_wall_seconds",
               "forecast_transitions")
    for case in CASES:
        by_controller = {}
        for name in CONTROLLERS:
            panel = sorted((row for row in rows if row["case"] == case and row["controller"] == name),
                           key=lambda row: int(row["seed"]))
            if not panel:
                raise RuntimeError("empty evaluation cell")
            by_controller[name] = panel
            result[f"{case}/{name}"] = {"count": len(panel),
                "means": {metric: float(np.mean([float(row[metric]) for row in panel])) for metric in metrics}}
        by_case[case] = by_controller
        teacher = by_controller["teacher_ppo"]
        for comparator in ("scratch_ppo", "warm_bc", "fcfs", "risk_matching", "rollout"):
            other = by_controller[comparator]
            if [int(row["seed"]) for row in teacher] != [int(row["seed"]) for row in other]:
                raise RuntimeError("unpaired evaluation grid")
            differences = np.asarray([float(a["objective"]) - float(b["objective"])
                                      for a, b in zip(teacher, other)])
            key = f"{case}/teacher_minus_{comparator}"
            primary = case == "reciprocal_scale2" and comparator in ("scratch_ppo", "warm_bc")
            rng = np.random.default_rng(role_seed(bootstrap_seed, f"{VERSION}:{key}"))
            indices = rng.integers(0, len(differences), size=(20000, len(differences)))
            ci = np.quantile(differences[indices].mean(axis=1),
                             [.0125, .9875] if primary else [.025, .975])
            result[key] = {"paired_mean": float(differences.mean()),
                           "ci": ci.tolist(), "ci_level": .975 if primary else .95,
                           "wins": int((differences < 0).sum()),
                           "ties": int((differences == 0).sum()),
                           "losses": int((differences > 0).sum()),
                           "confirmed": bool(differences.mean() < 0 and ci[1] < 0) if primary else None}
    result["co_primary_both_confirmed"] = bool(
        result["reciprocal_scale2/teacher_minus_scratch_ppo"]["confirmed"] and
        result["reciprocal_scale2/teacher_minus_warm_bc"]["confirmed"])
    return result


def run(profile, output_dir):
    if profile not in ("smoke", "full"):
        raise ValueError(profile)
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError("refusing existing run directory")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    deadline = started + (21600 if profile == "full" else 1200)
    registry, groups = seed_registry()
    train_seeds, dev_seeds, test_seeds, smoke_seeds, bank_seeds = groups[:5]
    git = source_info()
    if profile == "full" and git["blocking_paths"]:
        raise RuntimeError(f"full run requires committed inputs: {git['blocking_paths']}")
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in SOURCES}
    config = configs()["reciprocal_scale2"]
    all_configs = configs()
    manifest = {"status": "RUNNING", "phase": "preflight", "profile": profile,
        "protocol": VERSION, "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": git, "seed_registry": registry, "source_hashes": source_hashes,
        "invocation": [sys.executable, *sys.argv], "cwd": str(ROOT),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": torch.__version__, "device": "cpu"},
        "deadline_seconds": 21600 if profile == "full" else 1200}
    write_json(out / "manifest.json", manifest)
    shutil.copy2(PROTOCOL, out / "protocol.md")
    shutil.copy2(REGISTRY, out / "seed_registry.json")
    write_json(out / "resolved_config.json", {"cases": {name: asdict(cfg) for name, cfg in all_configs.items()},
        "train_case": "reciprocal_scale2",
        "steps": list(FULL_STEPS if profile == "full" else (1024,)),
        "teacher_bank_episodes_per_replica": 32 if profile == "full" else 2,
        "bc_epochs": 10 if profile == "full" else 2,
        "bc_batch_size": 256, "bc_learning_rate": .0003,
        "ppo": {"n_steps": 1024, "batch_size": 256, "n_epochs": 5,
                "gamma": .99, "gae_lambda": .95, "clip_range": .2,
                "learning_rate": .0003, "ent_coef": .01, "vf_coef": .5,
                "max_grad_norm": .5, "net_arch": [64, 64]}})
    def deadline_check():
        if time.monotonic() >= deadline:
            raise TimeoutError("v3 six-hour campaign deadline")
    try:
        train = train_seeds if profile == "full" else smoke_seeds[:1]
        dev = dev_seeds if profile == "full" else smoke_seeds
        evaluation = test_seeds if profile == "full" else smoke_seeds
        steps = FULL_STEPS if profile == "full" else (1024,)
        bank_count = 32 if profile == "full" else 2
        bc_epochs = 10 if profile == "full" else 2
        manifest["phase"] = "teacher_bank"
        write_json(out / "manifest.json", manifest)
        banks, bank_meta = {}, {}
        with tqdm(total=len(train) * bank_count, desc="Teacher bank", unit="episode") as bar:
            for index, train_seed in enumerate(train):
                assigned = bank_seeds[index * 32:index * 32 + bank_count]
                bank_path, metadata = create_teacher_bank(out, train_seed, assigned, config,
                    registry["forecast_seed"], deadline_check, bar)
                banks[train_seed] = bank_path
                bank_meta[str(train_seed)] = metadata
                write_json(out / "teacher_bank_meta.json", bank_meta)
        manifest["phase"] = "training"
        write_json(out / "manifest.json", manifest)
        candidates = {name: [] for name in LEARNED}
        training_audit = {}
        training_seed_sequences = {}
        for train_seed in train:
            deadline_check()
            scratch_dir = out / "arms" / "scratch_ppo"
            teacher_dir = out / "arms" / "teacher_ppo"
            for arm_dir in (scratch_dir, teacher_dir):
                arm_dir.mkdir(parents=True, exist_ok=True)
                progress = arm_dir / "training_progress.csv"
                if not progress.exists():
                    with progress.open("x", newline="") as stream:
                        csv.writer(stream).writerow(["train_seed", "steps", "checkpoint", "epoch_seconds"])
            scratch_env = LoggedFrameworkEnv(config)
            scratch = model_new(scratch_env, train_seed)
            initial_digest = policy_digest(scratch)
            scratch_env.episode_seeds.clear()
            scratch_start = time.perf_counter()
            scratch.learn(total_timesteps=steps[-1],
                          callback=ProgressCallback(scratch_dir, train_seed, steps, deadline),
                          progress_bar=False)
            scratch_seconds = time.perf_counter() - scratch_start
            scratch_sequence = list(scratch_env.episode_seeds)
            for step in steps:
                path = scratch_dir / "checkpoints" / str(train_seed) / f"{step}.zip"
                if not path.is_file():
                    raise RuntimeError("missing scratch checkpoint")
                candidates["scratch_ppo"].append((train_seed, step, path))
            scratch_env.close()
            del scratch
            teacher_env = LoggedFrameworkEnv(config)
            teacher = model_new(teacher_env, train_seed)
            teacher_initial_digest = policy_digest(teacher)
            if teacher_initial_digest != initial_digest:
                raise RuntimeError("scratch and teacher initial weights differ")
            bc_info = pretrain_actor(teacher, banks[train_seed], train_seed, bc_epochs,
                                     out / "bc_progress.csv", deadline_check)
            warm_dir = out / "arms" / "warm_bc"
            warm_dir.mkdir(parents=True, exist_ok=True)
            warm_path = warm_dir / f"{train_seed}.zip"
            teacher.save(warm_path)
            candidates["warm_bc"].append((train_seed, 0, warm_path))
            teacher_env.episode_seeds.clear()
            teacher_start = time.perf_counter()
            teacher.learn(total_timesteps=steps[-1],
                          callback=ProgressCallback(teacher_dir, train_seed, steps, deadline),
                          progress_bar=False)
            teacher_seconds = time.perf_counter() - teacher_start
            teacher_sequence = list(teacher_env.episode_seeds)
            if teacher_sequence != scratch_sequence:
                raise RuntimeError("paired training physical seed streams differ")
            for step in steps:
                path = teacher_dir / "checkpoints" / str(train_seed) / f"{step}.zip"
                if not path.is_file():
                    raise RuntimeError("missing teacher PPO checkpoint")
                candidates["teacher_ppo"].append((train_seed, step, path))
            teacher_env.close()
            del teacher
            training_seed_sequences[str(train_seed)] = {"scratch_ppo": scratch_sequence,
                "teacher_ppo": teacher_sequence}
            write_json(out / "training_seed_sequences.json", training_seed_sequences)
            training_audit[str(train_seed)] = {"scratch_initial_policy_sha256": initial_digest,
                "teacher_initial_policy_sha256": teacher_initial_digest,
                "training_episode_seed_count": len(scratch_sequence),
                "physical_sequence_sha256": seed_sequence_digest(scratch_sequence),
                "paired_sequence_equal": True, "scratch_ppo_wall_seconds": scratch_seconds,
                "teacher_ppo_wall_seconds": teacher_seconds, "bc": bc_info}
            write_json(out / "training_audit.json", training_audit)
        manifest["phase"] = "development"
        write_json(out / "manifest.json", manifest)
        development = []
        total_dev = sum(len(group) for group in candidates.values()) * len(dev)
        with tqdm(total=total_dev, desc="Development", unit="episode") as bar:
            for arm in LEARNED:
                for train_seed, step, path in candidates[arm]:
                    model = check_checkpoint(path, config, dev[0])
                    for physical_seed in dev:
                        deadline_check()
                        controller = LearnedController(model)
                        controller.name = arm
                        row = evaluate_episode(config, "reciprocal_scale2", physical_seed, controller)
                        row.update(train_seed=train_seed, checkpoint_step=step,
                                   checkpoint=str(path.relative_to(out)))
                        development.append(row)
                        add_row(out / "development.partial.csv", row)
                        bar.update(1)
        shutil.copy2(out / "development.partial.csv", out / "development.csv")
        selections = {}
        for arm in LEARNED:
            scores = []
            for train_seed, step, path in candidates[arm]:
                cell = [row["objective"] for row in development
                        if row["controller"] == arm and row["train_seed"] == train_seed
                        and row["checkpoint_step"] == step]
                if len(cell) != len(dev):
                    raise RuntimeError("incomplete development candidate")
                scores.append((float(np.mean(cell)), train_seed, step, path))
            value, seed, step, path = min(scores)
            selections[arm] = {"train_seed": seed, "steps": step,
                "checkpoint": str(path.relative_to(out)), "checkpoint_sha256": sha(path),
                "development_mean_cost": value,
                "candidate_scores": [{"mean_cost": score, "train_seed": train_seed,
                    "steps": candidate_step, "checkpoint": str(candidate_path.relative_to(out))}
                    for score, train_seed, candidate_step, candidate_path in scores]}
        write_json(out / "selection.json", selections)
        manifest["phase"] = "sealed_evaluation" if profile == "full" else "smoke_evaluation"
        manifest["selection_sha256"] = sha(out / "selection.json")
        write_json(out / "manifest.json", manifest)
        models = {arm: check_checkpoint(out / item["checkpoint"], config, dev[0])
                  for arm, item in selections.items()}
        grid = [(case, controller, seed) for case in CASES
                for controller in CONTROLLERS for seed in evaluation]
        episodes = []
        with tqdm(grid, desc="Sealed evaluation" if profile == "full" else "Smoke evaluation",
                  unit="episode") as bar:
            for case, controller_name, physical_seed in bar:
                deadline_check()
                if controller_name in LEARNED:
                    controller = LearnedController(models[controller_name])
                    controller.name = controller_name
                else:
                    controller = V3RuleController(controller_name, registry["forecast_seed"],
                                                  "evaluation", deadline_check)
                row = evaluate_episode(all_configs[case], case, physical_seed, controller)
                if abs(row["episode_return"] + row["objective"]) > 1e-9:
                    raise RuntimeError("evaluation reward/cost mismatch")
                if abs(row["objective"] - sum(row[key] for key in
                       ("maintenance_cost", "unavailability_cost", "failure_cost"))) > 1e-9:
                    raise RuntimeError("evaluation component mismatch")
                if (row["requests_created"] - row["requests_completed"] !=
                        row["end_pending"] + row["end_in_service"]):
                    raise RuntimeError("evaluation request conservation")
                episodes.append(row)
                add_row(out / "episodes.partial.csv", row)
        shutil.copy2(out / "episodes.partial.csv", out / "episodes.csv")
        expected = len(CASES) * len(CONTROLLERS) * len(evaluation)
        if len(episodes) != expected or any(row["invalid_assignments"] != 0 for row in episodes):
            raise RuntimeError("incomplete evaluation grid or invalid assignment")
        if source_hashes != {str(path.relative_to(ROOT)): sha(path) for path in SOURCES}:
            raise RuntimeError("source changed during campaign")
        if any(sha(banks[seed]) != bank_meta[str(seed)]["sha256"] for seed in train):
            raise RuntimeError("teacher bank changed during campaign")
        if any(sha(out / item["checkpoint"]) != item["checkpoint_sha256"]
               for item in selections.values()):
            raise RuntimeError("selected checkpoint changed during campaign")
        summary = paired_summary(episodes, registry["bootstrap_seed"])
        write_json(out / "summary.json", summary)
        write_json(out / "audit.json", {"status": "PASS", "teacher_labels_legal": True,
            "initial_weights_equal": True, "paired_training_physical_seeds_equal": True,
            "checkpoint_reload": True, "reward_components_request_reconciliation": True,
            "zero_invalid_assignments": True, "source_bank_checkpoint_hashes_stable": True,
            "development_episodes": len(development), "evaluation_episodes": len(episodes),
            "sealed_panel_opened_after_selection": profile == "full"})
        manifest.update(status="COMPLETED", phase="done", elapsed_seconds=time.monotonic() - started,
            counts={"teacher_bank_episodes": sum(item["episodes"] for item in bank_meta.values()),
                    "teacher_labels": sum(item["labels"] for item in bank_meta.values()),
                    "bc_optimizer_updates": sum(item["bc"]["optimizer_updates"] for item in training_audit.values()),
                    "ppo_training_steps": len(train) * len(("scratch_ppo", "teacher_ppo")) * steps[-1],
                    "development": len(development), "evaluation": len(episodes),
                    "ppo_checkpoints": len(candidates["scratch_ppo"]) + len(candidates["teacher_ppo"]),
                    "warm_checkpoints": len(candidates["warm_bc"])},
            finished_utc=datetime.now(timezone.utc).isoformat(),
            artifact_hashes={path.name: sha(path) for path in out.iterdir()
                             if path.is_file() and path.name != "manifest.json"})
        write_json(out / "manifest.json", manifest)
        return out
    except (TimeoutError, KeyboardInterrupt) as exc:
        manifest.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic() - started)
        write_json(out / "manifest.json", manifest)
        raise
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic() - started)
        write_json(out / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = run(args.profile, args.output_dir)
    print(json.dumps({"status": "COMPLETED", "output_dir": str(out)}))


if __name__ == "__main__":
    main()
