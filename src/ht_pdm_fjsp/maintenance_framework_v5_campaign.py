"""Prelocked scratch-RL comparison of post-decision residual capacity actors."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
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
from ht_pdm_fjsp.maintenance_framework_v5_policy import PostdecisionCapacityPolicy
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
from ht_pdm_fjsp.maintenance_request_policies import choose
from ht_pdm_fjsp.maintenance_requests import RequestEnv

ROOT = Path(__file__).resolve().parents[2]
VERSION = "maintenance_framework_v5_postdecision_capacity"
PROTOCOL = ROOT / "research_result/framework_v5_protocol_vi.md"
PROPOSAL = ROOT / "research_result/method_v5_proposal_vi.md"
REGISTRY = ROOT / "configs/maintenance_framework_v5_seed_registry.json"
LEARNED = ("flat", "edge_only", "additive_queue", "postdecision")
CONTROLLERS = ("fcfs", "risk_matching", "rollout", *LEARNED)
SOURCES = tuple(ROOT / path for path in (
    "src/ht_pdm_fjsp/maintenance_framework_v1.py",
    "src/ht_pdm_fjsp/maintenance_framework_v1_campaign.py",
    "src/ht_pdm_fjsp/maintenance_framework_v5_policy.py",
    "src/ht_pdm_fjsp/maintenance_framework_v5_campaign.py",
    "src/ht_pdm_fjsp/maintenance_dispatch.py",
    "src/ht_pdm_fjsp/maintenance_requests.py",
    "src/ht_pdm_fjsp/maintenance_request_policies.py",
    "research_result/method_v5_proposal_vi.md",
    "research_result/framework_v5_protocol_vi.md",
    "configs/maintenance_framework_v5_seed_registry.json",
    "pyproject.toml", "uv.lock",
))


class PairedMixtureEnv(FrameworkEnv):
    """Independent case and physical seed streams, identical across arms."""

    def __init__(self, training_seed: int):
        self.cases = configs()
        self.case_rng = np.random.default_rng(role_seed(training_seed, VERSION + ":case"))
        self.physical_rng = np.random.default_rng(role_seed(training_seed, VERSION + ":physical_sequence"))
        self.episode_tape: list[tuple[str, int]] = []
        super().__init__(self.cases[CASES[0]])

    def reset(self, *, seed=None, options=None):
        case = CASES[int(self.case_rng.integers(len(CASES)))]
        raw_seed = int(self.physical_rng.integers(0, 2**31))
        self.config = self.cases[case]
        self.inner = RequestEnv(self.config)
        observation, info = super().reset(seed=raw_seed, options=options)
        self.episode_tape.append((case, raw_seed))
        return observation, info


class V4RuleController:
    def __init__(self, name, forecast_seed, check_timeout=lambda: None):
        if name not in ("fcfs", "risk_matching", "rollout"):
            raise ValueError(name)
        self.name = name
        self.forecast_seed = forecast_seed
        self.forecast_transitions = 0
        self.check_timeout = check_timeout

    def choose(self, config, state, tick, *, case, seed):
        forecast = role_seed(self.forecast_seed, f"{VERSION}:forecast:{case}:{seed}:{tick}")
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
    groups = [panel(key) for key in ("training", "development", "sealed_test", "smoke")]
    groups += [[registry["forecast_seed"]], [registry["bootstrap_seed"]]]
    flat = [value for group in groups for value in group]
    if len(flat) != len(set(flat)) or set(flat) & historical:
        raise RuntimeError("v5 seed overlap with historical registry or split")
    return registry, groups


def source_info():
    info = git_info()
    relative = [str(path.relative_to(ROOT)) for path in (PROPOSAL, PROTOCOL, REGISTRY)]
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--", *relative],
        cwd=ROOT, text=True,
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *relative],
        cwd=ROOT, text=True,
    ).splitlines()
    info["blocking_paths"] = sorted(set(info["blocking_paths"] + changed + untracked))
    return info


def policy_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def model_for_arm(arm, env, seed):
    if arm == "flat":
        return model_new(env, seed)
    return MaskablePPO(
        PostdecisionCapacityPolicy, env, policy_kwargs={"variant": arm},
        seed=seed, device="cpu", verbose=0, n_steps=1024,
        batch_size=256, n_epochs=5, gamma=.99, gae_lambda=.95,
        clip_range=.2, learning_rate=3e-4, ent_coef=.01,
        vf_coef=.5, max_grad_norm=.5,
    )


def summarize(rows, bootstrap_seed):
    result = {}
    panels = {}
    metrics = (
        "objective", "H_availability", "H_failures", "queued_cm_ticks",
        "queued_pm_ticks", "H_pending", "end_pending", "H_in_service",
        "end_in_service", "maintenance_cost", "unavailability_cost",
        "failure_cost", "tail_cost", "selection_seconds",
        "episode_wall_seconds", "forecast_transitions",
    )
    for case in CASES:
        panels[case] = {}
        for controller in CONTROLLERS:
            panel = sorted(
                (row for row in rows if row["case"] == case and row["controller"] == controller),
                key=lambda row: int(row["seed"]),
            )
            if not panel:
                raise RuntimeError("empty evaluation cell")
            panels[case][controller] = panel
            result[f"{case}/{controller}"] = {
                "count": len(panel),
                "means": {key: float(np.mean([float(row[key]) for row in panel])) for key in metrics},
            }
        pairs = (("postdecision", "additive_queue"), ("additive_queue", "edge_only"),
                 ("postdecision", "edge_only"), ("postdecision", "flat"),
                 ("postdecision", "risk_matching"), ("postdecision", "rollout"),
                 ("postdecision", "fcfs"))
        for subject, comparator in pairs:
            lhs = panels[case][subject]
            rhs = panels[case][comparator]
            if [int(row["seed"]) for row in lhs] != [int(row["seed"]) for row in rhs]:
                raise RuntimeError("unpaired case panel")
            delta = np.asarray([float(a["objective"]) - float(b["objective"]) for a, b in zip(lhs, rhs)])
            key = f"{case}/{subject}_minus_{comparator}"
            rng = np.random.default_rng(role_seed(bootstrap_seed, f"{VERSION}:{key}"))
            indices = rng.integers(0, len(delta), size=(20000, len(delta)))
            ci = np.quantile(delta[indices].mean(axis=1), [.025, .975])
            result[key] = {
                "paired_mean": float(delta.mean()), "ci95_descriptive": ci.tolist(),
                "wins": int((delta < 0).sum()), "ties": int((delta == 0).sum()),
                "losses": int((delta > 0).sum()),
            }
    seeds = [int(row["seed"]) for row in panels[CASES[0]]["postdecision"]]
    for subject, comparator in pairs:
        values = []
        for index, seed in enumerate(seeds):
            differences = []
            for case in CASES:
                lhs = panels[case][subject][index]
                rhs = panels[case][comparator][index]
                if int(lhs["seed"]) != seed or int(rhs["seed"]) != seed:
                    raise RuntimeError("unpaired cross-case panel")
                differences.append(float(lhs["objective"]) - float(rhs["objective"]))
            values.append(float(np.mean(differences)))
        delta = np.asarray(values)
        key = f"pooled_{subject}_minus_{comparator}"
        rng = np.random.default_rng(role_seed(bootstrap_seed, f"{VERSION}:{key}"))
        indices = rng.integers(0, len(delta), size=(20000, len(delta)))
        ci = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        primary = subject == "postdecision" and comparator == "additive_queue"
        result[key] = {
            "paired_mean": float(delta.mean()), "cluster_bootstrap_ci95": ci.tolist(),
            "confirmed": bool(delta.mean() < 0 and ci[1] < 0) if primary else None,
            "seed_count": len(seeds), "cases_per_seed": len(CASES),
            "wins": int((delta < 0).sum()), "ties": int((delta == 0).sum()),
            "losses": int((delta > 0).sum()),
        }
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
    deadline = started + (14400 if profile == "full" else 1200)
    registry, groups = seed_registry()
    train_seeds, dev_seeds, test_seeds, smoke_seeds = groups[:4]
    git = source_info()
    if profile == "full" and git["blocking_paths"]:
        raise RuntimeError(f"full run requires committed inputs: {git['blocking_paths']}")
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in SOURCES}
    all_configs = configs()
    manifest = {
        "status": "RUNNING", "phase": "preflight", "profile": profile,
        "protocol": VERSION, "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": git, "seed_registry": registry, "source_hashes": source_hashes,
        "invocation": [sys.executable, *sys.argv], "cwd": str(ROOT),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": torch.__version__, "device": "cpu"},
        "deadline_seconds": 14400 if profile == "full" else 1200,
    }
    write_json(out / "manifest.json", manifest)
    shutil.copy2(PROTOCOL, out / "protocol.md")
    shutil.copy2(PROPOSAL, out / "method_proposal.md")
    shutil.copy2(REGISTRY, out / "seed_registry.json")
    steps = FULL_STEPS if profile == "full" else (1024,)
    write_json(out / "resolved_config.json", {
        "cases": {name: asdict(config) for name, config in all_configs.items()},
        "train_case_distribution": "uniform over four fixed cases",
        "steps": list(steps), "arms": list(LEARNED),
        "ppo": {"n_steps": 1024, "batch_size": 256, "n_epochs": 5,
                "gamma": .99, "gae_lambda": .95, "clip_range": .2,
                "learning_rate": .0003, "ent_coef": .01, "vf_coef": .5,
                "max_grad_norm": .5},
    })
    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("v5 four-hour campaign deadline")
    try:
        train = train_seeds if profile == "full" else smoke_seeds[:1]
        dev = dev_seeds if profile == "full" else smoke_seeds
        evaluation = test_seeds if profile == "full" else smoke_seeds
        manifest["phase"] = "training"
        write_json(out / "manifest.json", manifest)
        candidates = {arm: [] for arm in LEARNED}
        training_audit = {}
        training_tapes = {}
        model_parameters = {}
        for training_seed in train:
            check_deadline()
            reference_tape = None
            initial_structured_hash = None
            training_tapes[str(training_seed)] = {}
            training_audit[str(training_seed)] = {}
            for arm in LEARNED:
                arm_dir = out / "arms" / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                progress = arm_dir / "training_progress.csv"
                if not progress.exists():
                    with progress.open("x", newline="") as stream:
                        csv.writer(stream).writerow(["train_seed", "steps", "checkpoint", "epoch_seconds"])
                env = PairedMixtureEnv(training_seed)
                model = model_for_arm(arm, env, training_seed)
                model_parameters[arm] = sum(param.numel() for param in model.policy.parameters())
                initial_hash = policy_digest(model)
                if arm != "flat":
                    if initial_structured_hash is None:
                        initial_structured_hash = initial_hash
                    elif initial_hash != initial_structured_hash:
                        raise RuntimeError("structured arms' initial weights differ")
                start_arm = time.perf_counter()
                model.learn(
                    total_timesteps=steps[-1],
                    callback=ProgressCallback(arm_dir, training_seed, steps, deadline),
                    progress_bar=False,
                )
                elapsed_arm = time.perf_counter() - start_arm
                tape = list(env.episode_tape)
                if reference_tape is None:
                    reference_tape = tape
                elif tape != reference_tape:
                    raise RuntimeError("training case/physical tapes differ across arms")
                training_tapes[str(training_seed)][arm] = tape
                training_audit[str(training_seed)][arm] = {
                    "initial_policy_sha256": initial_hash,
                    "training_episode_count": len(tape),
                    "training_tape_sha256": hashlib.sha256(
                        json.dumps(tape, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "wall_seconds": elapsed_arm,
                }
                for step in steps:
                    path = arm_dir / "checkpoints" / str(training_seed) / f"{step}.zip"
                    if not path.is_file():
                        raise RuntimeError(f"missing checkpoint {path}")
                    candidates[arm].append((training_seed, step, path))
                write_json(out / "training_tapes.json", training_tapes)
                write_json(out / "training_audit.json", training_audit)
                env.close()
                del model
        manifest["phase"] = "development"
        write_json(out / "manifest.json", manifest)
        development = []
        total_dev = sum(len(group) for group in candidates.values()) * len(CASES) * len(dev)
        with tqdm(total=total_dev, desc="Development", unit="episode") as bar:
            for arm in LEARNED:
                for training_seed, step, path in candidates[arm]:
                    model = check_checkpoint(path, all_configs[CASES[0]], dev[0])
                    for case in CASES:
                        for seed in dev:
                            check_deadline()
                            controller = LearnedController(model)
                            controller.name = arm
                            row = evaluate_episode(all_configs[case], case, seed, controller)
                            row.update(train_seed=training_seed, checkpoint_step=step,
                                       checkpoint=str(path.relative_to(out)))
                            development.append(row)
                            add_row(out / "development.partial.csv", row)
                            bar.update(1)
        shutil.copy2(out / "development.partial.csv", out / "development.csv")
        selections = {}
        for arm in LEARNED:
            scores = []
            for training_seed, step, path in candidates[arm]:
                cell = [float(row["objective"]) for row in development
                        if row["controller"] == arm and row["train_seed"] == training_seed
                        and row["checkpoint_step"] == step]
                if len(cell) != len(CASES) * len(dev):
                    raise RuntimeError("incomplete development candidate")
                scores.append((float(np.mean(cell)), training_seed, step, path))
            value, training_seed, step, path = min(scores)
            selections[arm] = {
                "train_seed": training_seed, "steps": step,
                "checkpoint": str(path.relative_to(out)), "checkpoint_sha256": sha(path),
                "development_mean_cost": value,
                "candidate_scores": [
                    {"mean_cost": score, "train_seed": seed, "steps": candidate_step,
                     "checkpoint": str(candidate_path.relative_to(out))}
                    for score, seed, candidate_step, candidate_path in scores
                ],
            }
        write_json(out / "selection.json", selections)
        manifest["phase"] = "sealed_evaluation" if profile == "full" else "smoke_evaluation"
        manifest["selection_sha256"] = sha(out / "selection.json")
        write_json(out / "manifest.json", manifest)
        models = {arm: check_checkpoint(out / item["checkpoint"], all_configs[CASES[0]], dev[0])
                  for arm, item in selections.items()}
        episodes = []
        grid = [(case, controller, seed) for case in CASES
                for controller in CONTROLLERS for seed in evaluation]
        with tqdm(grid, desc="Sealed evaluation" if profile == "full" else "Smoke evaluation",
                  unit="episode") as bar:
            for case, controller_name, seed in bar:
                check_deadline()
                if controller_name in LEARNED:
                    controller = LearnedController(models[controller_name])
                    controller.name = controller_name
                else:
                    controller = V4RuleController(
                        controller_name, registry["forecast_seed"], check_deadline,
                    )
                row = evaluate_episode(all_configs[case], case, seed, controller)
                if abs(row["episode_return"] + row["objective"]) > 1e-9:
                    raise RuntimeError("evaluation reward/cost mismatch")
                if abs(row["objective"] - sum(row[key] for key in
                       ("maintenance_cost", "unavailability_cost", "failure_cost"))) > 1e-9:
                    raise RuntimeError("evaluation cost components mismatch")
                if row["requests_created"] - row["requests_completed"] != (
                        row["end_pending"] + row["end_in_service"]):
                    raise RuntimeError("evaluation request conservation mismatch")
                episodes.append(row)
                add_row(out / "episodes.partial.csv", row)
        shutil.copy2(out / "episodes.partial.csv", out / "episodes.csv")
        expected = len(CASES) * len(CONTROLLERS) * len(evaluation)
        if len(episodes) != expected or any(row["invalid_assignments"] != 0 for row in episodes):
            raise RuntimeError("incomplete evaluation or invalid assignment")
        if source_hashes != {str(path.relative_to(ROOT)): sha(path) for path in SOURCES}:
            raise RuntimeError("source changed during campaign")
        if any(sha(out / item["checkpoint"]) != item["checkpoint_sha256"]
               for item in selections.values()):
            raise RuntimeError("selected checkpoint changed during campaign")
        summary = summarize(episodes, registry["bootstrap_seed"])
        write_json(out / "summary.json", summary)
        write_json(out / "audit.json", {
            "status": "PASS", "training_tapes_equal": True,
            "structured_initial_weights_equal": True, "checkpoint_reload": True,
            "source_checkpoint_hashes_stable": True,
            "development_episodes": len(development),
            "evaluation_episodes": len(episodes),
            "reward_components_request_reconciliation": True,
            "zero_invalid_assignments": True,
            "selection_before_sealed_panel": profile == "full",
        })
        manifest.update(
            status="COMPLETED", phase="done",
            elapsed_seconds=time.monotonic() - started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            counts={
                "ppo_training_steps": len(train) * len(LEARNED) * steps[-1],
                "ppo_checkpoints": sum(len(group) for group in candidates.values()),
                "development": len(development), "evaluation": len(episodes),
            },
            policy_parameters=model_parameters,
            artifact_hashes={path.name: sha(path) for path in out.iterdir()
                             if path.is_file() and path.name != "manifest.json"},
        )
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
    output = run(args.profile, args.output_dir)
    print(json.dumps({"status": "COMPLETED", "output_dir": str(output)}))


if __name__ == "__main__":
    main()
