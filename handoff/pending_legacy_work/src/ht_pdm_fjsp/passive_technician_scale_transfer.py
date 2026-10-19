"""Zero-shot scale-transfer experiment for passive shared technicians."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_baselines import _action_diagnostics, _obs_tensor
from ht_pdm_fjsp.passive_technician_marl import (
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
    PassiveConfig,
    PassiveTechnicianEnv,
    dispatcher_action,
)
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    PassiveValueDecomposition,
    ReplayBuffer,
    ValueTrainSettings,
    evaluate_value_decomposition,
    train_value_decomposition_checkpoints,
)


TRAIN_SEEDS = tuple(range(31, 36))
DEV_SEEDS = tuple(range(501, 551))
SEALED_SEEDS = tuple(range(601, 701))
FULL_BUDGETS = (5_000, 10_000, 20_000)
SMOKE_BUDGETS = (8, 16, 32)
SCALABLE_METHODS = ("pooled_set_qmix", "relational_set_qmix")
FIELDNAMES = (
    "cell", "policy", "budget", "seed", "train_seed", "objective", "failures",
    "jobs", "collisions", "waiting", "invalid_requests", "busy_requests",
    "unique_joint_actions", "defer_fraction", "action_steps", "request_count",
)


def source_config() -> PassiveConfig:
    return PassiveConfig(
        machines=3,
        technicians=2,
        horizon=12,
        failure_age=5,
        max_age=8,
        service_time=((2, 4), (4, 2), (3, 3)),
        semantics="fifo_queue",
    )


def _seeded_config(machines: int, technicians: int, topology: str, seed: int) -> PassiveConfig:
    rng = random.Random(seed + machines * 1_003 + technicians * 10_007)
    service = [[float("inf")] * technicians for _ in range(machines)]
    if topology == "complete":
        edges = [(machine, technician) for machine in range(machines) for technician in range(technicians)]
    elif topology == "sparse":
        edges = []
        for machine in range(machines):
            edges.append((machine, machine % technicians))
            if technicians > 1 and rng.random() < 0.35:
                edges.append((machine, rng.randrange(technicians)))
        covered = {technician for _, technician in edges}
        for technician in set(range(technicians)) - covered:
            edges.append((technician % machines, technician))
    else:
        raise ValueError(topology)
    for machine, technician in set(edges):
        service[machine][technician] = rng.randint(2, 5)
    return PassiveConfig(
        machines=machines,
        technicians=technicians,
        horizon=18,
        failure_age=5,
        max_age=8,
        service_time=tuple(tuple(row) for row in service),
        semantics="fifo_queue",
    )


def evaluation_cells(seed: int) -> dict[str, PassiveConfig]:
    return {
        "source_3m2t": source_config(),
        "scale_5m3t_complete": _seeded_config(5, 3, "complete", seed),
        "scale_5m3t_sparse": _seeded_config(5, 3, "sparse", seed),
        "scale_8m4t_complete": _seeded_config(8, 4, "complete", seed),
        "scale_8m4t_sparse": _seeded_config(8, 4, "sparse", seed),
    }


class ScalableSetQMix(nn.Module):
    """Permutation-invariant monotonic mixer for variable machine/tech counts."""

    def __init__(self, mode: str, hidden: int = 64) -> None:
        super().__init__()
        if mode not in {"pooled_set_qmix", "relational_set_qmix"}:
            raise ValueError(mode)
        self.mode = mode
        self.hidden = hidden
        self.machine_encoder = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.technician_encoder = nn.Sequential(nn.Linear(6, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.edge_encoder = nn.Sequential(nn.Linear(9, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.edge_value = nn.Linear(hidden, 1)
        self.defer_value = nn.Sequential(nn.Linear(3 + 2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.global_encoder = nn.Sequential(nn.Linear(18, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.agent_weight = nn.Sequential(nn.Linear(3 + hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.global_value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def agent_q(self, local: torch.Tensor) -> torch.Tensor:
        batch, machines, features = local.shape
        own = local[..., :3]
        slots = local[..., 3:].reshape(batch, machines, -1, 6)
        machine_hidden = self.machine_encoder(own)
        technician_hidden = self.technician_encoder(slots)
        if self.mode == "pooled_set_qmix":
            q_edges = (machine_hidden.unsqueeze(2) * technician_hidden).sum(-1) / math.sqrt(self.hidden)
            pool = torch.cat((technician_hidden.mean(2), technician_hidden.amax(2)), dim=-1)
            q_defer = self.defer_value(torch.cat((own, pool), dim=-1))
        else:
            own_edges = own.unsqueeze(2).expand(-1, -1, slots.shape[2], -1)
            edge_hidden = self.edge_encoder(torch.cat((own_edges, slots), dim=-1))
            q_edges = self.edge_value(edge_hidden).squeeze(-1)
            pool = torch.cat((edge_hidden.mean(2), edge_hidden.amax(2)), dim=-1)
            q_defer = self.defer_value(torch.cat((own, pool), dim=-1))
        return torch.cat((q_defer, q_edges), dim=-1)

    def _global(self, local: torch.Tensor) -> torch.Tensor:
        batch, machines, features = local.shape
        own = local[..., :3]
        slots = local[..., 3:].reshape(batch, machines, -1, 6)
        machine_stats = torch.cat((own.mean(1), own.amax(1)), dim=-1)
        edge_stats = torch.cat((slots.flatten(1, 2).mean(1), slots.flatten(1, 2).amax(1)), dim=-1)
        return self.global_encoder(torch.cat((machine_stats, edge_stats), dim=-1))

    def total_q(self, local: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        values = self.agent_q(local)
        chosen = values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        global_state = self._global(local)
        expanded = global_state.unsqueeze(1).expand(-1, local.shape[1], -1)
        weights = torch.nn.functional.softplus(self.agent_weight(torch.cat((local[..., :3], expanded), dim=-1)).squeeze(-1))
        return (weights * chosen).sum(-1) + self.global_value(global_state).squeeze(-1)

    def save(self, path: Path, seed: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "mode": self.mode, "hidden": self.hidden, "seed": seed}, path)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "ScalableSetQMix":
        payload = torch.load(path, map_location=device, weights_only=True)
        model = cls(str(payload["mode"]), int(payload["hidden"])).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def _train_scalable(
    mode: str, seed: int, budgets: tuple[int, ...], root: Path,
    settings: ValueTrainSettings, device: torch.device,
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    config = source_config()
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    online = ScalableSetQMix(mode, settings.hidden_dim).to(device)
    target = ScalableSetQMix(mode, settings.hidden_dim).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=settings.learning_rate)
    replay = ReplayBuffer(settings.replay_capacity, config.machines, config.observation_dim, config.technicians + 1)
    checkpoints: dict[int, Path] = {}
    progress: list[dict[str, Any]] = []
    global_step = 0
    for episode in tqdm(range(1, max(budgets) + 1), desc=f"{mode}/{seed}", unit="episode", leave=False):
        env = PassiveTechnicianEnv(config, seed=seed * 100_000 + episode - 1)
        observations = env.reset()
        done = False
        episode_objective = 0.0
        losses: list[float] = []
        while not done:
            local = _obs_tensor(observations, config).numpy()
            masks = np.asarray(env.action_masks(), dtype=np.bool_)
            epsilon = max(settings.epsilon_end, settings.epsilon_start - (settings.epsilon_start - settings.epsilon_end) * global_step / max(1, int(settings.episodes * 12 * settings.epsilon_fraction)))
            with torch.no_grad():
                local_tensor = torch.as_tensor(local, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = online.agent_q(local_tensor)[0].masked_fill(~torch.as_tensor(masks, dtype=torch.bool, device=device), -1e9)
                greedy = q_values.argmax(-1).cpu().tolist()
            actions = tuple(int(rng.choice(np.flatnonzero(masks[m]))) if rng.random() < epsilon else int(greedy[m]) for m in range(config.machines))
            next_observations, reward, done, _ = env.step(actions)
            next_local = _obs_tensor(next_observations, config).numpy()
            next_masks = np.asarray(env.action_masks(), dtype=np.bool_)
            replay.add(local, masks, actions, float(reward), next_local, next_masks, done)
            observations = next_observations
            episode_objective -= float(reward)
            global_step += 1
            if replay.size >= settings.learning_starts and global_step % settings.train_frequency == 0:
                batch = replay.sample(settings.batch_size, rng)
                batch_local = torch.as_tensor(batch["local"], dtype=torch.float32, device=device)
                batch_masks = torch.as_tensor(batch["masks"], dtype=torch.bool, device=device)
                batch_actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
                next_local_tensor = torch.as_tensor(batch["next_local"], dtype=torch.float32, device=device)
                next_masks_tensor = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=device)
                chosen = online.total_q(batch_local, batch_actions)
                with torch.no_grad():
                    next_values = online.agent_q(next_local_tensor).masked_fill(~next_masks_tensor, -1e9)
                    next_actions = next_values.argmax(-1)
                    next_q = target.total_q(next_local_tensor, next_actions)
                    done_tensor = torch.as_tensor(batch["dones"], dtype=torch.bool, device=device)
                    target_q = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device) + settings.gamma * (~done_tensor).float() * next_q
                loss = (chosen - target_q).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if global_step % settings.target_update_interval == 0:
                    target.load_state_dict(online.state_dict())
        progress.append({"policy": mode, "train_seed": seed, "episode": episode, "objective": episode_objective, "loss": float(np.mean(losses)) if losses else 0.0, "epsilon": epsilon})
        if episode in set(budgets):
            path = root / f"budget_{episode}" / "model.pt"
            online.save(path, seed)
            checkpoints[episode] = path
    return checkpoints, progress


def _evaluate(config: PassiveConfig, policy: str, model: Any, seed: int, train_seed: int | str, budget: int, device: torch.device) -> dict[str, Any]:
    env = PassiveTechnicianEnv(config, seed=seed)
    observations = env.reset()
    actions_seen: set[tuple[int, ...]] = set()
    defer = steps = 0
    done = False
    while not done:
        if policy == "skill_aware_fifo":
            actions = dispatcher_action(env)
        elif policy == "standard_qmix":
            row = evaluate_value_decomposition(config, model, seed, int(train_seed), device)
            return {**row, "cell": "source_3m2t", "policy": policy, "budget": budget}
        else:
            local = _obs_tensor(observations, config).to(device).unsqueeze(0)
            masks = torch.as_tensor(env.action_masks(), dtype=torch.bool, device=device)
            with torch.no_grad():
                q = model.agent_q(local)[0].masked_fill(~masks, -1e9)
                actions = tuple(int(a) for a in q.argmax(-1).cpu().tolist())
        actions_seen.add(tuple(actions))
        defer += sum(action == 0 for action in actions)
        steps += 1
        observations, _, done, _ = env.step(actions)
    return {"cell": "", "policy": policy, "budget": budget, "seed": seed, "train_seed": train_seed, **env.metrics, **_action_diagnostics(config, actions_seen, defer, steps)}


def _exact_sign_flip(values: list[float]) -> float:
    observed = abs(statistics.fmean(values))
    stats = [abs(statistics.fmean([value * sign for value, sign in zip(values, signs)])) for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(value >= observed - 1e-12 for value in stats) / len(stats)


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_value))
        adjusted[name] = running
    return adjusted


def _write_csv(rows: list[dict[str, Any]], path: Path, fields: tuple[str, ...] | None = None) -> None:
    if not rows and fields is None:
        return
    names = list(fields) if fields else list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _profile(name: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if name == "smoke":
        return (31,), (501, 502, 503), SMOKE_BUDGETS
    if name == "full":
        return TRAIN_SEEDS, DEV_SEEDS, FULL_BUDGETS
    raise ValueError(name)


def run(args: argparse.Namespace) -> Path:
    train_seeds, eval_seeds, budgets = _profile(args.profile)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = replace(ValueTrainSettings(episodes=max(budgets)), replay_capacity=10_000, batch_size=64, learning_starts=64)
    cells = evaluation_cells(eval_seeds[0])
    manifest = {
        "status": "RUNNING", "experiment": "passive_technician_scale_transfer", "profile": args.profile,
        "methods": ["skill_aware_fifo", "standard_qmix", *SCALABLE_METHODS],
        "train_seeds": list(train_seeds), "development_seeds": list(eval_seeds),
        "sealed_test_seeds": list(SEALED_SEEDS), "checkpoint_budgets": list(budgets),
        "training_config": asdict(source_config()), "evaluation_cells_template": {key: asdict(value) for key, value in cells.items()},
        "hypothesis": "Relational Set-QMIX has lower mean objective than pooled Set-QMIX on each prespecified out-of-distribution scale/topology family at the final checkpoint.",
        "primary_metric": "mean objective per training seed averaged over common development seeds",
        "objective_version": OBJECTIVE_VERSION, "observation_version": OBSERVATION_VERSION,
        "requested_device": args.device, "resolved_device": str(device), "git_revision": _git_revision(),
        "sealed_test_evaluated": False, "started_at": datetime.now(UTC).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    config_snapshot = {"training_config": asdict(source_config()), "evaluation_cells_template": {key: asdict(value) for key, value in cells.items()}, "train_seeds": train_seeds, "development_seeds": eval_seeds, "sealed_test_seeds": SEALED_SEEDS, "budgets": budgets, "settings": asdict(settings)}
    (output / "benchmark_config.json").write_text(json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n")
    rows: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    source = source_config()
    for seed in tqdm(eval_seeds, desc="evaluate FIFO references", unit="episode"):
        for cell_name, config in evaluation_cells(seed).items():
            for policy in ("skill_aware_fifo",):
                row = _evaluate(config, policy, None, seed, "", 0, device)
                row["cell"] = cell_name
                rows.append(row)
        _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)

    for method in ("standard_qmix", *SCALABLE_METHODS):
        for train_seed in tqdm(train_seeds, desc=f"train {method}", unit="seed"):
            root = output / "models" / method / f"train_seed_{train_seed}"
            if method == "standard_qmix":
                checkpoints, train_rows = train_value_decomposition_checkpoints(source, "qmix", train_seed, (max(budgets),), root, settings, device)
                checkpoint_map = {budget: checkpoints[max(budgets)] for budget in budgets}
                progress.extend({**item, "cell": "source_3m2t", "policy": method} for item in train_rows)
            else:
                checkpoint_map, train_rows = _train_scalable(method, train_seed, budgets, root, settings, device)
                progress.extend({**item, "cell": "source_3m2t", "policy": method} for item in train_rows)
            for budget in budgets:
                if method == "standard_qmix" and budget != max(budgets):
                    continue
                checkpoint = checkpoint_map[budget]
                model = PassiveValueDecomposition.load(checkpoint, source, device) if method == "standard_qmix" else ScalableSetQMix.load(checkpoint, device)
                for seed in tqdm(eval_seeds, desc=f"evaluate {method}/{train_seed}/{budget}", unit="episode", leave=False):
                    for cell_name, config in evaluation_cells(seed).items():
                        if method == "standard_qmix" and cell_name != "source_3m2t":
                            continue
                        row = _evaluate(config, method, model, seed, train_seed, budget, device)
                        row["cell"] = cell_name
                        rows.append(row)
                    _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)

    _write_csv(rows, output / "episodes.csv", FIELDNAMES)
    _write_csv(rows, output / "coordination.csv", FIELDNAMES)
    _write_csv(progress, output / "training_progress.csv")
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["cell"], row["policy"], int(row["budget"]), str(row["train_seed"])), []).append(row)
    budget_summary = []
    for (cell, policy, budget, train_seed), group in sorted(grouped.items()):
        budget_summary.append({"cell": cell, "policy": policy, "budget": budget, "train_seed": train_seed, "evaluation_count": len(group), "objective_mean": statistics.fmean(float(row["objective"]) for row in group), "failures_mean": statistics.fmean(float(row["failures"]) for row in group), "jobs_mean": statistics.fmean(float(row["jobs"]) for row in group), "waiting_mean": statistics.fmean(float(row["waiting"]) for row in group), "invalid_requests_mean": statistics.fmean(float(row["invalid_requests"]) for row in group)})
    _write_csv(budget_summary, output / "budget_summary.csv")
    final = max(budgets)
    contrasts = {}
    raw_ps = {}
    for cell_name in (name for name in cells if name != "source_3m2t"):
        deltas = []
        for seed in train_seeds:
            left = next(float(row["objective_mean"]) for row in budget_summary if row["cell"] == cell_name and row["policy"] == "relational_set_qmix" and int(row["budget"]) == final and row["train_seed"] == str(seed))
            right = next(float(row["objective_mean"]) for row in budget_summary if row["cell"] == cell_name and row["policy"] == "pooled_set_qmix" and int(row["budget"]) == final and row["train_seed"] == str(seed))
            deltas.append(left - right)
        rng = np.random.default_rng(20260926)
        samples = rng.choice(np.asarray(deltas), size=(20_000, len(deltas)), replace=True).mean(axis=1)
        contrast = {"training_seed_count": len(deltas), "mean_delta_relational_minus_pooled": statistics.fmean(deltas), "training_seed_deltas": deltas, "bootstrap_95_ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))], "exact_sign_flip_p": _exact_sign_flip(deltas)}
        contrasts[cell_name] = contrast
        raw_ps[cell_name] = contrast["exact_sign_flip_p"]
    adjusted = _holm(raw_ps)
    for cell_name, p_value in adjusted.items():
        contrasts[cell_name]["holm_adjusted_p"] = p_value
    expected = len(eval_seeds) * len(cells) + len(eval_seeds) * len(train_seeds) * (len(budgets) * len(SCALABLE_METHODS) * len(cells) + 1)
    audits = {"expected_episode_count": expected, "episode_count": len(rows), "all_expected_rows_present": len(rows) == expected, "invalid_requests_zero": all(float(row["invalid_requests"]) == 0 for row in rows), "nonnegative_objectives": all(float(row["objective"]) >= 0 for row in rows), "all_training_seeds_present": all(str(row["train_seed"]) for row in rows if row["policy"] != "skill_aware_fifo"), "sealed_panel_closed": True}
    summary = {"primary_metric": manifest["primary_metric"], "final_budget": final, "budget_summary": budget_summary, "primary_ood_contrasts": contrasts, "audits": audits}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest.update({"status": "COMPLETED", "finished_at": datetime.now(UTC).isoformat(), "episode_count": len(rows), "expected_episode_count": expected, "checkpoint_count": sum(path.is_file() for path in output.rglob("model.pt")), "audit_gates": audits, "outputs": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audits": audits, "primary_ood_contrasts": contrasts}, indent=2, sort_keys=True))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
