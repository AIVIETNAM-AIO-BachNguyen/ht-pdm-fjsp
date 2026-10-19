"""Fixed-budget configuration-mixture follow-up for the maintenance RL framework."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import gymnasium as gym
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
from ht_pdm_fjsp.maintenance_requests import RequestEnv

ROOT = Path(__file__).resolve().parents[2]
VERSION = "maintenance_framework_v2_mixture"
PROTOCOL = ROOT / "research_result/framework_v2_protocol_vi.md"
REGISTRY = ROOT / "configs/maintenance_framework_v2_seed_registry.json"
ANCHOR = ROOT / "research_result/framework_v1_full_20261019T153014Z/checkpoints/62000002/199680.zip"
ANCHOR_SHA256 = "6f1400b2e11990d86dbd14d137b85c2f2ac73ccf1c1c380bee8d03a779727cb9"
SOURCES = tuple(ROOT / path for path in (
    "src/ht_pdm_fjsp/maintenance_framework_v1.py",
    "src/ht_pdm_fjsp/maintenance_framework_v1_campaign.py",
    "src/ht_pdm_fjsp/maintenance_framework_v2_campaign.py",
    "src/ht_pdm_fjsp/maintenance_dispatch.py",
    "src/ht_pdm_fjsp/maintenance_requests.py",
    "src/ht_pdm_fjsp/maintenance_request_policies.py",
    "research_result/framework_v2_protocol_vi.md",
    "configs/maintenance_framework_v2_seed_registry.json",
    "pyproject.toml", "uv.lock",
))
CONTROLLERS = ("fcfs", "risk_matching", "rollout", "anchor_ppo", "mixture_ppo")


class MixtureEnv(FrameworkEnv):
    """Choose one registered N4/K2 configuration per episode, uniformly."""

    def __init__(self):
        self.case_configs = configs()
        self.case_counts = {case: 0 for case in CASES}
        super().__init__(self.case_configs[CASES[0]])

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        case = CASES[int(self.np_random.integers(len(CASES)))]
        self.config = self.case_configs[case]
        self.inner = RequestEnv(self.config)
        self.case_counts[case] += 1
        return super().reset(seed=None, options=options)


class V2RuleController:
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
    groups = [panel(name) for name in ("training", "development", "sealed_test", "smoke")]
    groups += [[registry["forecast_seed"]], [registry["bootstrap_seed"]]]
    flat = [seed for group in groups for seed in group]
    if len(flat) != len(set(flat)) or set(flat) & historical:
        raise RuntimeError("v2 seed overlap with historical registry or split")
    return registry, groups


def source_info():
    info = git_info()
    paths = subprocess.check_output([
        "git", "diff", "--name-only", "HEAD", "--", str(PROTOCOL.relative_to(ROOT))
    ], cwd=ROOT, text=True).splitlines()
    paths += subprocess.check_output([
        "git", "ls-files", "--others", "--exclude-standard", "--", str(PROTOCOL.relative_to(ROOT))
    ], cwd=ROOT, text=True).splitlines()
    info["blocking_paths"] = sorted(set(info["blocking_paths"] + paths))
    return info


def summarize(rows, bootstrap_seed):
    rng = np.random.default_rng(bootstrap_seed)
    result = {}
    by_case = {}
    metrics = ("objective", "H_availability", "H_failures", "queued_cm_ticks",
               "queued_pm_ticks", "H_pending", "end_pending", "H_in_service",
               "end_in_service", "maintenance_cost", "unavailability_cost",
               "failure_cost", "tail_cost", "selection_seconds", "episode_wall_seconds",
               "forecast_transitions")
    for case in CASES:
        by_controller = {}
        for controller in CONTROLLERS:
            panel = sorted((row for row in rows if row["case"] == case and row["controller"] == controller),
                           key=lambda row: int(row["seed"]))
            if not panel:
                raise RuntimeError("empty evaluation cell")
            by_controller[controller] = panel
            result[f"{case}/{controller}"] = {"count": len(panel),
                "means": {key: float(np.mean([float(row[key]) for row in panel])) for key in metrics}}
        by_case[case] = by_controller
        learned = by_controller["mixture_ppo"]
        for comparator in ("anchor_ppo", "fcfs", "risk_matching", "rollout"):
            other = by_controller[comparator]
            if [int(row["seed"]) for row in learned] != [int(row["seed"]) for row in other]:
                raise RuntimeError("unpaired evaluation grid")
            delta = np.array([float(a["objective"]) - float(b["objective"])
                              for a, b in zip(learned, other)])
            draws = rng.integers(0, len(delta), size=(20000, len(delta)))
            ci = np.quantile(delta[draws].mean(axis=1), [.025, .975])
            result[f"{case}/mixture_minus_{comparator}"] = {
                "paired_mean": float(delta.mean()), "bootstrap_ci95_descriptive": ci.tolist(),
                "wins": int((delta < 0).sum()), "ties": int((delta == 0).sum()),
                "losses": int((delta > 0).sum())}
    seeds = [int(row["seed"]) for row in by_case[CASES[0]]["mixture_ppo"]]
    pooled = []
    for index, seed in enumerate(seeds):
        differences = []
        for case in CASES:
            mix = by_case[case]["mixture_ppo"][index]
            anchor = by_case[case]["anchor_ppo"][index]
            if int(mix["seed"]) != seed or int(anchor["seed"]) != seed:
                raise RuntimeError("cross-case seed mismatch")
            differences.append(float(mix["objective"]) - float(anchor["objective"]))
        pooled.append(float(np.mean(differences)))
    values = np.asarray(pooled)
    draws = rng.integers(0, len(values), size=(20000, len(values)))
    ci = np.quantile(values[draws].mean(axis=1), [.025, .975])
    result["primary_case_balanced_mixture_minus_anchor"] = {
        "paired_mean": float(values.mean()), "cluster_bootstrap_ci95": ci.tolist(),
        "positive_improvement_confirmed": bool(values.mean() < 0 and ci[1] < 0),
        "seed_count": len(values), "cases_per_seed": len(CASES),
        "wins": int((values < 0).sum()), "ties": int((values == 0).sum()),
        "losses": int((values > 0).sum())}
    return result


def run(profile, output_dir):
    if profile not in ("smoke", "full"):
        raise ValueError(profile)
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError("refusing existing run directory")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    start = time.monotonic()
    deadline = start + (21600 if profile == "full" else 1200)
    registry, groups = seed_registry()
    train_seeds, dev_seeds, test_seeds, smoke_seeds = groups[:4]
    git = source_info()
    if profile == "full" and git["blocking_paths"]:
        raise RuntimeError(f"full run requires committed inputs: {git['blocking_paths']}")
    if not ANCHOR.is_file() or sha(ANCHOR) != ANCHOR_SHA256:
        raise RuntimeError("frozen v1 checkpoint missing or hash mismatch")
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in SOURCES}
    cfgs = configs()
    manifest = {"status": "RUNNING", "phase": "preflight", "profile": profile,
        "protocol": VERSION, "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": git, "seed_registry": registry, "source_hashes": hashes,
        "anchor_checkpoint": str(ANCHOR.relative_to(ROOT)), "anchor_sha256": ANCHOR_SHA256,
        "invocation": [sys.executable, *sys.argv], "cwd": str(ROOT),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": torch.__version__, "device": "cpu"},
        "deadline_seconds": 21600 if profile == "full" else 1200}
    write_json(out / "manifest.json", manifest)
    shutil.copy2(PROTOCOL, out / "protocol.md")
    shutil.copy2(REGISTRY, out / "seed_registry.json")
    write_json(out / "resolved_config.json", {"cases": {name: asdict(cfg) for name, cfg in cfgs.items()},
        "training_case_sampling": "uniform per episode", "steps": list(FULL_STEPS),
        "hyperparameters": {"n_steps": 1024, "batch_size": 256, "n_epochs": 5,
            "gamma": .99, "gae_lambda": .95, "clip_range": .2,
            "learning_rate": .0003, "ent_coef": .01, "vf_coef": .5,
            "max_grad_norm": .5, "net_arch": [64, 64]}})
    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("campaign deadline")
    try:
        manifest["phase"] = "training"
        write_json(out / "manifest.json", manifest)
        steps = FULL_STEPS if profile == "full" else (1024,)
        seeds = train_seeds if profile == "full" else smoke_seeds[:1]
        with (out / "training_progress.csv").open("x", newline="") as stream:
            csv.writer(stream).writerow(["train_seed", "steps", "checkpoint", "epoch_seconds"])
        paths = []
        case_counts = {}
        for seed in seeds:
            check_deadline()
            env = MixtureEnv()
            model = model_new(env, seed)
            callback = ProgressCallback(out, seed, steps, deadline)
            model.learn(total_timesteps=steps[-1], callback=callback, progress_bar=False)
            case_counts[str(seed)] = env.case_counts
            for step in steps:
                checkpoint = out / "checkpoints" / str(seed) / f"{step}.zip"
                if not checkpoint.is_file():
                    raise RuntimeError("missing checkpoint")
                paths.append((seed, step, checkpoint))
            env.close()
        write_json(out / "training_case_counts.json", case_counts)
        manifest["phase"] = "development"
        write_json(out / "manifest.json", manifest)
        panel = dev_seeds if profile == "full" else smoke_seeds
        development = []
        with tqdm(total=len(paths) * len(CASES) * len(panel), desc="Development", unit="episode") as bar:
            for train_seed, step, checkpoint in paths:
                model = check_checkpoint(checkpoint, cfgs[CASES[0]], panel[0])
                for case in CASES:
                    for seed in panel:
                        check_deadline()
                        row = evaluate_episode(cfgs[case], case, seed, LearnedController(model))
                        row.update(train_seed=train_seed, checkpoint_step=step,
                                   checkpoint=str(checkpoint.relative_to(out)))
                        development.append(row)
                        add_row(out / "development.partial.csv", row)
                        bar.update(1)
        shutil.copy2(out / "development.partial.csv", out / "development.csv")
        scores = []
        for train_seed, step, checkpoint in paths:
            cell = [row["objective"] for row in development
                    if row["train_seed"] == train_seed and row["checkpoint_step"] == step]
            if len(cell) != len(CASES) * len(panel):
                raise RuntimeError("incomplete development selection cell")
            scores.append((float(np.mean(cell)), train_seed, step, checkpoint))
        score, train_seed, step, selected = min(scores)
        selection = {"train_seed": train_seed, "steps": step,
            "checkpoint": str(selected.relative_to(out)), "development_case_balanced_mean_cost": score,
            "candidate_scores": [{"mean_cost": value, "train_seed": seed,
                "steps": tick, "checkpoint": str(path.relative_to(out))}
                for value, seed, tick, path in scores],
            "selection_rule": "lowest four-case development mean; tie smaller training seed then earlier step"}
        write_json(out / "selection.json", selection)
        manifest["phase"] = "sealed_evaluation" if profile == "full" else "smoke_evaluation"
        manifest["selected_checkpoint"] = selection["checkpoint"]
        write_json(out / "manifest.json", manifest)
        mixture_model = check_checkpoint(selected, cfgs[CASES[0]], panel[0])
        anchor_model = check_checkpoint(ANCHOR, cfgs[CASES[0]], panel[0])
        eval_seeds = test_seeds if profile == "full" else smoke_seeds
        grid = [(case, controller, seed) for case in CASES for controller in CONTROLLERS for seed in eval_seeds]
        rows = []
        with tqdm(grid, desc="Sealed evaluation" if profile == "full" else "Smoke evaluation",
                  unit="episode") as bar:
            for case, controller_name, seed in bar:
                check_deadline()
                if controller_name in ("anchor_ppo", "mixture_ppo"):
                    model = anchor_model if controller_name == "anchor_ppo" else mixture_model
                    controller = LearnedController(model)
                    controller.name = controller_name
                else:
                    controller = V2RuleController(controller_name, registry["forecast_seed"], check_deadline)
                row = evaluate_episode(cfgs[case], case, seed, controller)
                if abs(row["episode_return"] + row["objective"]) > 1e-9:
                    raise RuntimeError("reward/cost mismatch")
                if abs(row["objective"] - sum(row[key] for key in
                       ("maintenance_cost", "unavailability_cost", "failure_cost"))) > 1e-9:
                    raise RuntimeError("cost component mismatch")
                if (row["requests_created"] - row["requests_completed"] !=
                        row["end_pending"] + row["end_in_service"]):
                    raise RuntimeError("request conservation")
                rows.append(row)
                add_row(out / "episodes.partial.csv", row)
        shutil.copy2(out / "episodes.partial.csv", out / "episodes.csv")
        expected = len(CASES) * len(CONTROLLERS) * len(eval_seeds)
        if len(rows) != expected or any(row["invalid_assignments"] != 0 for row in rows):
            raise RuntimeError("incomplete grid or infeasible assignment")
        if hashes != {str(path.relative_to(ROOT)): sha(path) for path in SOURCES} or sha(ANCHOR) != ANCHOR_SHA256:
            raise RuntimeError("source or comparator changed during campaign")
        summary = summarize(rows, registry["bootstrap_seed"])
        write_json(out / "summary.json", summary)
        write_json(out / "audit.json", {"status": "PASS", "episodes": len(rows),
            "development_episodes": len(development), "checkpoints": len(paths),
            "zero_invalid_assignments": True, "reward_cost_components_request_reconciliation": True,
            "source_and_anchor_hashes_stable": True, "checkpoint_reload": True,
            "sealed_panel_opened_after_selection": profile == "full"})
        manifest.update(status="COMPLETED", phase="done", elapsed_seconds=time.monotonic() - start,
            counts={"episodes": len(rows), "development": len(development),
                    "training_steps": len(seeds) * steps[-1], "checkpoints": len(paths)},
            finished_utc=datetime.now(timezone.utc).isoformat(),
            artifact_hashes={path.name: sha(path) for path in out.iterdir()
                             if path.is_file() and path.name != "manifest.json"})
        write_json(out / "manifest.json", manifest)
        return out
    except (TimeoutError, KeyboardInterrupt) as exc:
        manifest.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic() - start)
        write_json(out / "manifest.json", manifest)
        raise
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic() - start)
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
