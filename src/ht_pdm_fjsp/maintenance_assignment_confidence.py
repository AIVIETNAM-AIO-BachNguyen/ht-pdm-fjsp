"""Prospective assignment coverage x empirical confidence-gate experiment."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_assignment_confidence_benchmark import ROOT, VERSION, ARMS, NAMES, settings, metadata, seed_audit
from ht_pdm_fjsp.maintenance_assignment_confidence_data import collect, make_dataset, forecast
from ht_pdm_fjsp.maintenance_assignment_confidence_policy import ConfidenceController, calibrate
from ht_pdm_fjsp.maintenance_assignment_confidence_training import fit
from ht_pdm_fjsp.maintenance_assignment_confidence_evaluation import CountingJournal, evaluate
from ht_pdm_fjsp.maintenance_assignment_confidence_statistics import summarize, diagnostic_summary
from ht_pdm_fjsp.maintenance_assignment_value_source import verify_source, sha, parameters_sha
from ht_pdm_fjsp.maintenance_assignment_value_forks import candidates, choose, paired_gain
from ht_pdm_fjsp.maintenance_autoregressive_policy import feature_batch
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_opportunity_cost import write_json, write_csv, compact
from ht_pdm_fjsp.maintenance_waiting import ENV_VERSION


def counts(cfg):
    n, c = len(cfg["model_seeds"]), len(cfg["conditions"])
    trajectories = lambda split: n * c * len(cfg[f"{split}_cohorts"]) * len(cfg[f"{split}_shocks"])
    source = {split: trajectories(split) for split in ("train", "development", "calibration")}
    diagnostic_trajectories = n * c * min(cfg["diagnostic_profiles"], len(cfg["test_cohorts"]))
    states = len(cfg["landmarks"]) + len(cfg["probe_landmarks"])
    upper = sum(source[s] * 21 * cfg[f"{s}_scenarios"] * sum(60 - t for t in cfg["landmarks"] + cfg["probe_landmarks"])
        for s in source)
    upper += diagnostic_trajectories * 2 * (cfg["selection_scenarios"] + cfg["evaluation_scenarios"]) * sum(60 - t for t in cfg["landmarks"] + cfg["probe_landmarks"])
    episodes = trajectories("test") * len(NAMES)
    return dict(source_models=n, trained_heads=n * len(ARMS) * cfg["ensemble_members"],
        train_states=source["train"] * states, development_states=source["development"] * states,
        calibration_states=source["calibration"] * states, diagnostic_states=diagnostic_trajectories * states,
        natural_source_transitions=(sum(source.values()) + diagnostic_trajectories) * 48,
        probe_prefix_transitions=(sum(source.values()) + diagnostic_trajectories) * len(cfg["probe_landmarks"]) * 2,
        test_episodes=episodes, test_base_transitions=episodes * 48, test_tail_transitions=episodes * 12,
        logged_test_transitions=n * c * len(NAMES) * min(len(cfg["test_cohorts"]), cfg["audit_profiles"]) * min(len(cfg["test_shocks"]), cfg["audit_shocks"]) * 60,
        optimizer_steps=n * len(ARMS) * cfg["ensemble_members"] * cfg["epochs"] * ((cfg["draws_per_epoch"] + cfg["minibatch"] - 1) // cfg["minibatch"]),
        max_forecast_transitions=upper)


@torch.no_grad()
def diagnose(output, cfg, records, models, controllers, journals, stats):
    rows = []
    for si, record in enumerate(tqdm(records, desc="Held-out assignment root checks", unit="state")):
        identity = record["identity"]; seed = identity["model_seed"]; tick = identity["tick"]
        c, state, original = record["config"], record["state"], record["original"]
        roots0, groups, _ = candidates(c, state, original)
        roots = tuple(roots0[i] for i in groups["assignment"])
        if len(roots) > 2:
            raise RuntimeError("N4/K2 same-machine assignment class exceeded")
        actions = {}; audits = {}
        for name, ctrl in controllers[seed].items():
            actions[name] = ctrl.select(c, state, 48 - tick, 48)[0]
            audits[name] = ctrl.last_decode_audit[0]
            if actions[name] not in roots:
                raise RuntimeError("assignment changed source timing or CM")
        journals["root_predictions"].add({**identity, "roots": compact(roots), "actions": compact(actions),
            "audits": compact(audits), "recorded_before_forecasts": True})
        row = {**identity, "exposure": int(len(roots) > 1), "roots": compact(roots),
            "selection_forecast": "", "evaluation_forecast": "", "reference_gain": 0., "reference_mc_se": 0.}
        if len(roots) > 1:
            selection, p1 = forecast(output, cfg, record, roots, models[seed], "ranking_selection", cfg["selection_scenarios"], stats, str(si))
            evaluation, p2 = forecast(output, cfg, record, roots, models[seed], "ranking_evaluation", cfg["evaluation_scenarios"], stats, str(si))
            ref = choose(selection["costs"], tuple(range(len(roots))))
            row["reference_gain"], row["reference_mc_se"] = paired_gain(evaluation["costs"], 0, ref)
            row.update(selection_forecast=p1, evaluation_forecast=p2)
        for name, action in actions.items():
            root = roots.index(action)
            gain, se = paired_gain(evaluation["costs"], 0, root) if len(roots) > 1 else (0., 0.)
            audit = audits[name]; alt = CATALOGUE.index(roots[1]) if len(roots) > 1 else audit["original_id"]
            k = audit["candidates"].index(alt)
            observed = float((evaluation["costs"][0] - evaluation["costs"][1]).mean() / cfg["reward_scale"]) if len(roots) > 1 else 0.
            prediction = audit["predicted_gains"][k]
            row.update({f"{name}_gain": gain, f"{name}_mc_se": se, f"{name}_changed": int(root != 0),
                f"{name}_predicted_alt_gain": prediction, f"{name}_measured_alt_gain": observed,
                f"{name}_squared_error": (prediction - observed) ** 2,
                f"{name}_ensemble_sd": audit["ensemble_sd"][k], f"{name}_margin": audit["margin"]})
        journals["root_states"].add(row); rows.append(row)
    return rows


def run(output, source_dir, profile="smoke", device="cpu"):
    if device != "cpu":
        raise ValueError("CPU only")
    output = Path(output); cfg = settings(profile); start = time.perf_counter()
    if output.exists() and any(output.iterdir()):
        raise ValueError("new empty timestamped output directory required; resume unsupported")
    output.mkdir(parents=True, exist_ok=True); torch.set_num_threads(1)
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    now = lambda: datetime.now(timezone.utc).isoformat()
    manifest = dict(protocol=VERSION, status="RUNNING", started=now(), commit=git("rev-parse", "HEAD"),
        dirty=bool(git("status", "--porcelain")), settings=cfg, expected_counts=counts(cfg), device=device,
        platform=platform.platform(), python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
        environment_version=ENV_VERSION, full_test_outcomes_opened=False, source_outcomes_previously_opened=True,
        thread_environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    journals = {}; stats = defaultdict(int); write_json(output / "manifest.json", manifest)
    try:
        if profile == "full" and manifest["dirty"]:
            raise ValueError("full requires clean committed code")
        if profile == "full" and any(v != "1" for v in manifest["thread_environment"].values()):
            raise ValueError("full requires OMP/OPENBLAS/MKL_NUM_THREADS=1")
        models, source, source_audit = verify_source(source_dir, cfg)
        manifest["source_audit"] = source_audit
        before = {seed: parameters_sha(model) for seed, model in models.items()}
        manifest["seed_audit"] = seed_audit(cfg)
        all_panels, physical_registry, audit = metadata(); panels = all_panels[profile]
        manifest["physical_split_audit"] = audit
        write_json(output / "source_contract.json", source)
        write_json(output / "resolved_config.json", dict(settings=cfg, physical_profiles=physical_registry[profile]))
        shutil.copyfile(ROOT / f"configs/{VERSION}_seed_registry.json", output / "seed_registry.json")
        shutil.copyfile(ROOT / f"docs/{VERSION}_plan.md", output / "plan.md")
        snapshot = output / "source_snapshot"; snapshot.mkdir()
        for path in sorted((ROOT / "src/ht_pdm_fjsp").glob("*.py")):
            shutil.copyfile(path, snapshot / path.name)
        for path in sorted((ROOT / "configs").glob("*.json")):
            shutil.copyfile(path, snapshot / path.name)
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copyfile(ROOT / name, snapshot / name)
        names = ("source_trajectories", "probe_prefixes", "label_states", "label_pairs", "training_progress", "training_draws",
            "calibration_predictions", "episodes.partial", "decisions", "pm_choices", "requests", "carryover", "coordination",
            "root_predictions", "root_states")
        journals = {name: CountingJournal(output / f"{name}.csv") for name in names}
        print(json.dumps(dict(protocol=VERSION, settings=cfg, budgets=counts(cfg)), indent=2), flush=True)
        datasets = {}; records = {}
        for split in ("train", "development", "calibration"):
            records[split] = collect(panels, cfg, split, models, journals["source_trajectories"], journals["probe_prefixes"])
            datasets[split] = make_dataset(output, cfg, split, records[split], models, journals, stats)
        controllers = {}; selection = {}; calibration = {}; coverage = []
        for seed, train_seed in tqdm(list(zip(cfg["model_seeds"], cfg["train_seeds"], strict=True)), desc="Multi-source ensemble training"):
            heads, chosen = fit(output, cfg, train_seed, seed, source["models"][str(seed)]["sha256"],
                datasets["train"][seed], datasets["development"][seed], journals, stats)
            selection[str(seed)] = chosen; controllers[seed] = {}; calibration[str(seed)] = {}
            for arm in ARMS:
                margins, calibration_rows = calibrate(heads[arm], datasets["calibration"][seed], cfg)
                calibration[str(seed)][arm] = margins
                for row in calibration_rows:
                    journals["calibration_predictions"].add(dict(source_seed=seed, arm=arm, **{k: compact(v) if isinstance(v, list) else v for k, v in row.items()}))
                for suffix in ("ungated", "gated"):
                    name = f"{arm}_{suffix}"
                    controllers[seed][name] = ConfidenceController(models[seed], heads[arm], name, margins, cfg)
            for split in datasets:
                d = datasets[split][seed]
                for cond, origin in itertools.product(range(2), range(2)):
                    selected = (d["conditions"] == cond) & (d["origins"] == origin)
                    labels = selected.nonzero().flatten()
                    coverage.append(dict(source_seed=seed, split=split, condition=cfg["conditions"][cond],
                        origin="probe" if origin else "natural", labels=len(labels),
                        labels_by_tick={str(t): int((selected & (d["ticks"] == t)).sum()) for t in sorted(set(d["ticks"].tolist()))},
                        labels_at_tick_zero=int((selected & (d["ticks"] == 0)).sum()),
                        states_with_labels=len(d["pairs"][labels, 0].unique())))
        write_json(output / "training_coverage.json", coverage)
        write_json(output / "architecture.json", dict(
            frozen_source_parameters={str(seed): sum(p.numel() for p in model.parameters()) for seed, model in models.items()},
            parameters_per_head=sum(p.numel() for p in controllers[cfg["model_seeds"][0]]["broad_gated"].heads[0].parameters()),
            members_per_ensemble=cfg["ensemble_members"], trained_ensembles_per_source=len(ARMS),
            timing="same source PM machine set and exact CM at intervention; closed-loop trajectories can diverge",
            uncertainty="member disagreement + separate empirical calibration; not a proven confidence bound"))
        write_json(output / "checkpoint_selection.json", dict(locked_before_test=True, criterion="common broad true dev MSE; earliest epoch tie", selections=selection))
        write_json(output / "calibration.json", dict(locked_before_test=True, parameters=calibration,
            method="90th percentile empirical overoptimism; ensemble SD + paired MCSE; no confidence guarantee"))
        if stats["optimizer_steps"] != counts(cfg)["optimizer_steps"]:
            raise RuntimeError("matched optimization count mismatch")
        manifest["full_test_outcomes_opened"] = profile == "full"; write_json(output / "manifest.json", manifest)
        episodes = []; totals = defaultdict(lambda: defaultdict(float))
        for seed, name in tqdm(list(itertools.product(cfg["model_seeds"], NAMES)), desc="Sealed five-controller multi-seed evaluation"):
            ctrl = models[seed] if name == "frozen_ppo" else controllers[seed][name]
            episodes.extend(evaluate(panels, cfg, "test", name, seed, ctrl, journals=journals, time_totals=totals))
        root_records = collect(panels, cfg, "test", models, journals["source_trajectories"], journals["probe_prefixes"],
            cohorts=cfg["test_cohorts"][:cfg["diagnostic_profiles"]], shocks=cfg["test_shocks"][:1])
        diagnostics = diagnose(output, cfg, root_records, models, controllers, journals, stats)
        summary, cells = summarize(episodes, cfg)
        write_json(output / "summary.json", summary); write_json(output / "diagnostic_summary.json", diagnostic_summary(diagnostics, cfg))
        write_csv(output / "episodes.csv", episodes); write_csv(output / "profile_differences.csv", cells)
        write_csv(output / "time_profiles.csv", [dict(condition=cond, controller=name, model_seed=seed, time=t, phase=phase,
            episodes=int(v["episodes"]), **{k: x / v["episodes"] for k, x in v.items() if k != "episodes"})
            for (cond, name, seed, t, phase), v in sorted(totals.items())])
        if before != {seed: parameters_sha(model) for seed, model in models.items()}:
            raise RuntimeError("frozen PPO mutated")
        actual = dict(source_models=len(models), trained_heads=len(models) * len(ARMS) * cfg["ensemble_members"],
            train_states=len(records["train"]), development_states=len(records["development"]), calibration_states=len(records["calibration"]),
            diagnostic_states=len(diagnostics), natural_source_transitions=journals["source_trajectories"].rows,
            probe_prefix_transitions=journals["probe_prefixes"].rows, test_episodes=len(episodes),
            test_base_transitions=len(episodes) * 48, test_tail_transitions=len(episodes) * 12,
            logged_test_transitions=journals["decisions"].rows, optimizer_steps=stats["optimizer_steps"])
        if actual != {k: v for k, v in counts(cfg).items() if k != "max_forecast_transitions"} or stats["forecast_transitions"] > counts(cfg)["max_forecast_transitions"]:
            raise RuntimeError("state/episode/compute count mismatch")
        if journals["label_states"].rows != sum(len(r) for r in records.values()) or journals["label_pairs"].rows != sum(len(d["pairs"]) for ds in datasets.values() for d in ds.values()):
            raise RuntimeError("dataset journals mismatch")
        if profile == "smoke" and stats["forecast_futures"] != stats["scalar_replayed_futures"]:
            raise RuntimeError("smoke forecast replay incomplete")
        manifest.update(status="COMPLETED", audit="PASS", actual_counts=actual, effective_compute=dict(stats),
            frozen_ppo_audit="PASS", elapsed_seconds=time.perf_counter() - start)
    except BaseException as error:
        manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for journal in journals.values():
            journal.close()
        manifest["finished"] = now()
        manifest["artifacts"] = {p.relative_to(output).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        write_json(output / "manifest.json", manifest)
    print(json.dumps(dict(status=manifest["status"], audit=manifest["audit"], output_dir=str(output), elapsed_seconds=manifest["elapsed_seconds"])), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), default="smoke")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.source_dir, args.profile, args.device)


if __name__ == "__main__":
    main()
