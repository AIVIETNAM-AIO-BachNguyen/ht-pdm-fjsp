"""Independent timing/technician counterfactuals for the current frozen decoder."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
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

from ht_pdm_fjsp import maintenance_autoregressive_benchmark as benchmark
from ht_pdm_fjsp import maintenance_autoregressive_pilot as pilot
from ht_pdm_fjsp import maintenance_boundary_sensitivity as boundary
from ht_pdm_fjsp import maintenance_joint_rl as historical
from ht_pdm_fjsp import maintenance_opportunity_cost as prior
from ht_pdm_fjsp.maintenance_ar_diagnostic_forks import (
    CONTRASTS, COMPONENTS, candidates, choose, paired_gain, policy_scores, fork_roots, scalar_replay)
from ht_pdm_fjsp.maintenance_autoregressive_policy import CONTEXT
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_timing_statistics import crossed_ratio
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

VERSION = "maintenance_ar_decision_diagnostic_v1"
ROOT = Path(__file__).resolve().parents[2]
PHASES = ("selection", "evaluation")
RAW_KEYS = ("cohorts", "physical_shocks", "selection_seed", "evaluation_seed", "analysis_seed")
KEYS = ("condition", "cohort", "shock", "model_seed", "tick")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameters_sha(model):
    return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def settings(profile):
    if profile not in ("full", "smoke"):
        raise ValueError("unknown profile")
    registry = json.loads((ROOT/f"configs/{VERSION}_seed_registry.json").read_text())
    if registry["protocol"] != VERSION:
        raise ValueError("seed protocol mismatch")
    full = profile == "full"
    return dict(profile=profile, **registry["panels"][profile],
        model_seeds=list(range(16100000, 16100003)) if full else [16100000],
        conditions=["dense", "skill_mask"], landmarks=[0, 6, 18, 30, 42] if full else [0, 42],
        selection_scenarios=32 if full else 2, evaluation_scenarios=64 if full else 2,
        horizon=48, tail_ticks=12, bootstrap_samples=20000 if full else 200,
        primary_confidence=1-.05/6, max_roots=7, training_steps=0, torch_threads=1,
        hard_timeout_seconds=7200 if full else 120)


def counts(cfg):
    trajectories = len(cfg["model_seeds"])*len(cfg["cohorts"])*len(cfg["physical_shocks"])*len(cfg["conditions"])
    return dict(models=len(cfg["model_seeds"]), trajectories=trajectories, trajectory_ticks=trajectories*48,
        states=trajectories*len(cfg["landmarks"]), training_steps=0,
        max_forecast_transitions=trajectories*sum(60-t for t in cfg["landmarks"])*7*
            (cfg["selection_scenarios"]+cfg["evaluation_scenarios"]))


def physical_seed(cohort, shock):
    return role_seed(shock, f"{VERSION}:physical:{cohort}")


def forecast_seed(cfg, identity, phase):
    if phase not in PHASES:
        raise ValueError("unknown forecast phase")
    name = ":".join(str(identity[k]) for k in KEYS)
    return role_seed(cfg[f"{phase}_seed"], f"{VERSION}:{phase}:{name}")


def seed_audit(cfg):
    historical_raw = historical.integers(historical.exclusions()["historical_raw_panels"])
    own = ROOT/f"configs/{VERSION}_seed_registry.json"
    registry_hashes = {}
    for path in (ROOT/"configs").glob("*seed_registry.json"):
        if path != own:
            historical_raw |= historical.integers(json.loads(path.read_text()))
            registry_hashes[path.name] = sha(path)
    fresh, derived = set(), []
    # Include all source pilot metadata RNG roles, never execute its trajectories.
    old_roles = set()
    _, _, training = benchmark.metadata()
    for mode in ("full", "smoke"):
        old = benchmark.settings(mode)
        for seed in old["train_seeds"]:
            old_roles.update(role_seed(seed, f"{benchmark.VERSION}:{role}") for role in ("initialization", "actions", "minibatches"))
        for config, _, cs, es, _ in training[mode].values():
            old_roles.update((cs, es, role_seed(cs, f"{benchmark.VERSION}:roles"), role_seed(cs, f"{benchmark.VERSION}:initial_scenario")))
        for split in ("development", "test"):
            for cohort in old[f"{split}_cohorts"]:
                old_roles.update((role_seed(cohort, f"{benchmark.VERSION}:roles"), role_seed(cohort, f"{benchmark.VERSION}:initial_scenario")))
                old_roles.update(role_seed(shock, f"{benchmark.VERSION}:physical:{cohort}") for shock in old[f"{split}_shocks"])
        old_roles.add(role_seed(old["analysis_seed"], f"{benchmark.VERSION}:crossed_bootstrap"))
        panel = settings(mode)
        for key in RAW_KEYS:
            group = historical.integers(panel[key])
            if group & (fresh | historical_raw | old_roles):
                raise ValueError("fresh raw seed collision")
            fresh |= group
        for cohort in panel["cohorts"]:
            derived.extend((role_seed(cohort, f"{benchmark.VERSION}:roles"), role_seed(cohort, f"{benchmark.VERSION}:initial_scenario")))
            derived.extend(physical_seed(cohort, shock) for shock in panel["physical_shocks"])
        for values in itertools.product(panel["conditions"], panel["cohorts"], panel["physical_shocks"], panel["model_seeds"], panel["landmarks"]):
            identity = dict(zip(KEYS, values, strict=True))
            derived.extend(forecast_seed(panel, identity, phase) for phase in PHASES)
        derived.append(role_seed(panel["analysis_seed"], f"{VERSION}:crossed_bootstrap"))
    if (cfg != settings(cfg["profile"]) or len(derived) != len(set(derived)) or
            set(derived) & (fresh | historical_raw | old_roles)):
        raise ValueError("derived RNG/settings collision")
    return dict(status="PASS", unique_derived_roles=len(derived), own_registry_sha256=sha(own),
        historical_registry_hashes=registry_hashes, source_pilot_roles_checked=len(old_roles),
        intentional_pairing="physical across replicas/conditions; roots within phase; independent selection/evaluation")


@lru_cache(maxsize=1)
def metadata():
    blocked = set(benchmark.historical_pool())
    old_panels, _, old_training = benchmark.metadata()
    for values in old_panels.values():
        blocked.update(boundary.signature(c) for c in values.values())
    for values in old_training.values():
        blocked.update(boundary.signature(record[0]) for record in values.values())
    excluded = len(blocked)
    panels, registry = {}, {}
    for mode in ("full", "smoke"):
        cfg = settings(mode)
        panels[mode], registry[mode] = {}, []
        for cohort in cfg["cohorts"]:
            for attempt in range(10000):
                generated = {cond: benchmark.make_config(cohort, cond, 60, attempt) for cond in cfg["conditions"]}
                sigs = {boundary.signature(c) for c in generated.values()}
                if len(sigs) == 2 and not sigs & blocked:
                    break
            else:
                raise RuntimeError("fresh profile pool exhausted")
            blocked.update(sigs)
            for cond, config in generated.items():
                panels[mode][cohort, cond] = config
                registry[mode].append(dict(cohort=cohort, condition=cond, generator_attempt=attempt,
                    config=asdict(config), **benchmark.scenario(config, cohort)))
    return panels, registry, dict(status="PASS", excluded_signatures=excluded, outcomes_used=False,
        full_smoke_signatures_disjoint=True, fresh_signatures=sum(len(v) for v in registry.values()))


def verify_source(source_dir, cfg, contract=None):
    """Pin provenance and model bytes; do not use source evaluation outcomes."""
    source_dir = Path(source_dir)
    if contract is None:
        contract = json.loads((ROOT/f"configs/{VERSION}_sources.json").read_text())
    if contract["protocol"] != VERSION or contract["source_protocol"] != pilot.VERSION:
        raise ValueError("source contract mismatch")
    for name, expected in contract["files"].items():
        if sha(source_dir/name) != expected:
            raise ValueError(f"source hash mismatch: {name}")
    manifest = json.loads((source_dir/"manifest.json").read_text())
    if (manifest["status"] != "COMPLETED" or manifest["audit"] != "PASS" or
            manifest["commit"] != contract["source_commit"] or manifest["dirty"] or
            manifest["settings"] != benchmark.settings("full")):
        raise ValueError("source completion/commit/settings mismatch")
    verified = 0
    for name, record in manifest["artifacts"].items():
        path = source_dir/name
        if not path.resolve().is_relative_to(source_dir.resolve()) or sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"source artifact mismatch: {name}")
        verified += 1
    for path in (source_dir/"source_snapshot").glob("*.py"):
        current = ROOT/"src/ht_pdm_fjsp"/path.name
        if current.exists() and sha(current) != sha(path):
            raise ValueError(f"source environment/policy code changed: {path.name}")
    selection = json.loads((source_dir/"checkpoint_selection.json").read_text())
    if not selection["locked_before_test"]:
        raise ValueError("source checkpoint selection was not locked")
    models = {}
    for seed in cfg["model_seeds"]:
        pinned = contract["models"][str(seed)]
        selected = selection["selections"][f"{CONTEXT}:{seed}"]
        if (selected["selected_checkpoint"] != pinned["path"] or selected["update"] != pinned["update"] or
                selected["learned_steps"] != pinned["learned_steps"] or
                pilot.choose_checkpoint(selected["candidates"])["update"] != pinned["update"]):
            raise ValueError("source model selection mismatch")
        path = source_dir/pinned["path"]
        if sha(path) != pinned["sha256"]:
            raise ValueError("source model hash mismatch")
        model, payload = pilot.load_checkpoint(path)
        if (payload["algorithm"] != CONTEXT or payload["train_seed"] != seed or
                payload["update"] != pinned["update"] or payload["learned_steps"] != pinned["learned_steps"] or
                payload["settings"] != benchmark.settings("full")):
            raise ValueError("source checkpoint identity/settings mismatch")
        models[seed] = model
    return models, contract, dict(status="PASS", source_commit=contract["source_commit"],
        verified_source_artifacts=verified, model_count=len(models), source_outcomes_previously_opened=True)


def summarize(rows, cfg):
    index = {tuple(row[k] for k in KEYS): row for row in rows}
    expected = set(itertools.product(cfg["conditions"], cfg["cohorts"], cfg["physical_shocks"], cfg["model_seeds"], cfg["landmarks"]))
    if len(rows) != len(index) or set(index) != expected:
        raise RuntimeError("incomplete/duplicate state grid")
    n, p = len(cfg["model_seeds"]), len(cfg["cohorts"])
    rng = np.random.default_rng(role_seed(cfg["analysis_seed"], f"{VERSION}:crossed_bootstrap"))
    rd = rng.integers(n, size=(cfg["bootstrap_samples"], n))
    pd = rng.integers(p, size=(cfg["bootstrap_samples"], p))
    primary, cells, strata = {}, [], []
    for condition, contrast in itertools.product(cfg["conditions"], CONTRASTS):
        num, den = np.zeros((n, p)), np.zeros((n, p))
        for i, seed in enumerate(cfg["model_seeds"]):
            for j, cohort in enumerate(cfg["cohorts"]):
                group = [index[condition, cohort, shock, seed, tick] for shock, tick in itertools.product(cfg["physical_shocks"], cfg["landmarks"])]
                for row in group:
                    exposure, gain = row[f"{contrast}_exposure"], row[f"{contrast}_gain"]
                    if exposure not in (0, 1) or not np.isfinite(gain) or (not exposure and gain != 0):
                        raise ValueError("invalid conditional numerator/exposure")
                num[i, j] = sum(row[f"{contrast}_gain"] for row in group)
                den[i, j] = sum(row[f"{contrast}_exposure"] for row in group)
                cells.append(dict(condition=condition, contrast=contrast, model_seed=seed, cohort=cohort,
                    numerator=float(num[i, j]), exposure=int(den[i, j])))
        draws = crossed_ratio(num, den, rd, pd)
        finite = draws[np.isfinite(draws)]
        empty = float(1-len(finite)/len(draws))
        estimate = float(num.sum()/den.sum()) if den.sum() else None
        alpha = (1-cfg["primary_confidence"])/2
        ci = np.quantile(finite, [alpha, 1-alpha]).tolist() if estimate is not None and empty <= .05 else None
        status = "INSUFFICIENT" if ci is None else "POSITIVE_CORRECTION_GAIN" if ci[0] > 0 else "NEGATIVE_CORRECTION_GAIN" if ci[1] < 0 else "UNRESOLVED"
        eligible = [row for row in rows if row["condition"] == condition and row[f"{contrast}_exposure"]]
        total = sum(row["condition"] == condition for row in rows)
        primary[f"{condition}:{contrast}"] = dict(estimate=estimate, confidence=cfg["primary_confidence"],
            confidence_interval=ci, exposure=len(eligible), total_states=total, coverage=len(eligible)/total,
            switched=sum(row[f"{contrast}_selected"] != 0 for row in eligible), empty_bootstrap_fraction=empty,
            replica_gains={str(seed): float(num[i].sum()/den[i].sum()) if den[i].sum() else None for i, seed in enumerate(cfg["model_seeds"])},
            status="ENGINEERING_ONLY" if cfg["profile"] == "smoke" else status,
            positive_means="lower remaining H48+12 cost after selection-chosen one-step correction")
        for tick in cfg["landmarks"]:
            group = [row for row in eligible if row["tick"] == tick]
            strata.append(dict(condition=condition, contrast=contrast, tick=tick, exposure=len(group),
                mean_gain=float(np.mean([row[f"{contrast}_gain"] for row in group])) if group else None))
    secondary = {}
    for condition in cfg["conditions"]:
        group = [row for row in rows if row["condition"] == condition and row["joint_exposure"]]
        secondary[condition] = dict(exposure=len(group),
            joint_vs_best_single_mean=float(np.mean([row["joint_vs_best_single_gain"] for row in group])) if group else None,
            defer_all_mean=float(np.mean([row["defer_all_gain"] for row in group])) if group else None,
            joint_selected_both_changed=sum(row["joint_selected_both_changed"] for row in group),
            exploratory_only=True)
    return dict(primary_contrasts=primary, secondary=secondary, strata=strata,
        uncertainty="paired crossed 3-replica/profile percentile bootstrap, conditional on fixed forecast tapes; paired MC SE saved per state",
        multiplicity="six exploratory contrasts, approximate Bonferroni99.1667%; three replicas limit calibration",
        scope="one-step simulator-assisted corrections on this frozen policy's visited states; not new RL training or a deployable oracle"), cells


def evaluate(config, state, identity, model, cfg, journals):
    original, info, masses, sequences, prefixes = policy_scores(config, state, 48-identity["tick"], model)
    roots, groups, no_pm = candidates(config, state, original)
    selection = fork_roots(config, state, roots, model, forecast_seed(cfg, identity, "selection"),
        cfg["selection_scenarios"], identity["tick"], record_trace=cfg["profile"] == "smoke")
    selected = {name: choose(selection["costs"], eligible) for name, eligible in groups.items()}
    # Fix all choices before obtaining independent evaluation outcomes.
    evaluation = fork_roots(config, state, roots, model, forecast_seed(cfg, identity, "evaluation"),
        cfg["evaluation_scenarios"], identity["tick"], record_trace=cfg["profile"] == "smoke")
    values = (selection, evaluation)
    if cfg["profile"] == "smoke":
        for value in values:
            scalar_replay(config, state, value, model, identity["tick"])
    row = dict(**identity, state=prior.compact(asdict(state)), original=prior.compact(original),
        token_trace=prior.compact(info["token_trace"]), policy_value=info["value"],
        roots=prior.compact(roots), unique_roots=len(roots), no_pm_index=no_pm,
        original_matching_probability=float(masses[original]),
        marginal_argmax_matching=prior.compact(max(masses, key=masses.get)),
        first_stop_probability=prefixes[0]["stop_probability"],
        pm_starts=sum(not state.failed[m] for m, _ in original),
        young_pm_starts=sum(not state.failed[m] and state.ages[m]+1 < config.failure_age for m, _ in original),
        physical_seed=physical_seed(identity["cohort"], identity["shock"]),
        selection_seed=forecast_seed(cfg, identity, "selection"), evaluation_seed=forecast_seed(cfg, identity, "evaluation"),
        remaining_ticks=60-identity["tick"], forecast_transitions=sum(v["transitions"] for v in values))
    for name, eligible in groups.items():
        i = selected[name]
        gain, se = paired_gain(evaluation["costs"], 0, i)
        row.update({f"{name}_candidates": prior.compact(eligible), f"{name}_exposure": int(len(eligible) > 1),
            f"{name}_selected": i, f"{name}_gain": gain, f"{name}_mc_se": se,
            f"{name}_matching_probability": float(masses[roots[i]]),
            f"{name}_exploratory_positive_mc95": int(i != 0 and gain-1.96*se > 0)})
    gain, se = paired_gain(evaluation["costs"], selected["best_single"], selected["joint"])
    row.update(joint_vs_best_single_gain=gain, joint_vs_best_single_mc_se=se,
        joint_selected_both_changed=int(selected["joint"] not in groups["best_single"]))
    gain, se = paired_gain(evaluation["costs"], 0, no_pm)
    row.update(defer_all_gain=gain, defer_all_mc_se=se)
    for sequence in sequences:
        journals["sequence_scores"].add(dict(**identity, trace=prior.compact(sequence["trace"]),
            action=prior.compact(sequence["action"]), log_probability=sequence["log_probability"], probability=sequence["probability"]))
    for prefix in prefixes:
        journals["prefix_scores"].add(dict(**identity, **{k: prior.compact(v) if isinstance(v, list) else v for k, v in prefix.items()}))
    for phase, value in zip(PHASES, values, strict=True):
        for i, root in enumerate(roots):
            delta, se = paired_gain(value["costs"], 0, i)
            journals["root_values"].add(dict(**identity, phase=phase, root_index=i, action=prior.compact(root),
                matching_probability=float(masses[root]), scenarios=value["costs"].shape[1],
                cost_mean=float(value["costs"][i].mean()), baseline_minus_root_mean=delta, paired_mc_se=se,
                base_mean=float(value["base"][i].mean()), tail_mean=float(value["tail"][i].mean()),
                **{k: float(value["components"][k][i].mean()) for k in COMPONENTS}))
        journals["rollout_audits"].add(dict(**identity, phase=phase, transitions=value["transitions"],
            **value["counts"], feasibility="PASS", scalar_replay="PASS" if cfg["profile"] == "smoke" else "NOT_RUN"))
    return row, values


def save_forecasts(path, identity, values):
    """Variable roots/futures per file; no padding is interpreted as data."""
    data = {"identity": np.asarray([str(identity[k]) for k in KEYS])}
    for phase, value in zip(PHASES, values, strict=True):
        for key in ("costs", "base", "tail", "shocks", "first"):
            data[f"{phase}_{key}"] = value[key]
        for k in COMPONENTS:
            data[f"{phase}_{k}"] = value["components"][k]
        for stage in ("cutoff", "terminal"):
            for k, v in value[stage].items():
                data[f"{phase}_{stage}_{k}"] = v
    np.savez_compressed(path, **data)


def run(output, model_source_dir, profile="smoke", device="cpu"):
    if device != "cpu":
        raise ValueError("CPU only")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("new/empty artifact directory required")
    output.mkdir(parents=True, exist_ok=True)
    cfg = settings(profile)
    torch.set_num_threads(1)
    started = time.perf_counter()
    now = lambda: datetime.now(timezone.utc).isoformat()
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    manifest = dict(protocol=VERSION, status="RUNNING", started=now(), commit=git("rev-parse", "HEAD"),
        dirty=bool(git("status", "--porcelain")), settings=cfg, expected_counts=counts(cfg), device=device,
        python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__), platform=platform.platform(),
        new_full_outcomes_opened=False, source_outcomes_previously_opened=True,
        thread_environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    prior.write_json(output/"manifest.json", manifest)
    journals, rows = {}, []
    totals = dict(transitions=0, tail_rows=0, decision_rows=0, near_tie_scalar_rows=0)
    try:
        if profile == "full" and (manifest["dirty"] or any(v != "1" for v in manifest["thread_environment"].values())):
            raise ValueError("full requires clean committed code and OMP/OPENBLAS/MKL_NUM_THREADS=1")
        models, contract, manifest["source_audit"] = verify_source(model_source_dir, cfg)
        manifest["seed_audit"] = seed_audit(cfg)
        panels, registry, manifest["physical_audit"] = metadata()
        panels, registry = panels[profile], registry[profile]
        prior.write_json(output/"source_contract.json", contract)
        prior.write_json(output/"resolved_config.json", dict(settings=cfg, physical_profiles=registry))
        shutil.copyfile(ROOT/f"configs/{VERSION}_seed_registry.json", output/"seed_registry.json")
        shutil.copyfile(ROOT/f"docs/{VERSION}_plan.md", output/"plan.md")
        snapshot = output/"source_snapshot"
        snapshot.mkdir()
        for path in list((ROOT/"src/ht_pdm_fjsp").glob("*.py")) + list((ROOT/"configs").glob("*.json")) + [ROOT/"pyproject.toml", ROOT/"uv.lock"]:
            shutil.copyfile(path, snapshot/path.name)
        initial = {}
        for seed in cfg["model_seeds"]:
            destination = output/"frozen_models"/f"seed_{seed}.pt"
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(Path(model_source_dir)/contract["models"][str(seed)]["path"], destination)
            restored, _ = pilot.load_checkpoint(destination)
            if sha(destination) != contract["models"][str(seed)]["sha256"] or parameters_sha(restored) != parameters_sha(models[seed]):
                raise RuntimeError("frozen model roundtrip mismatch")
            for (cohort, _), config in panels.items():
                state = benchmark.initial_state(config, cohort)
                if restored.select(config, state, 48, 48) != models[seed].select(config, state, 48, 48):
                    raise RuntimeError("restored model action mismatch")
            models[seed], initial[seed] = restored, parameters_sha(restored)
        for name in ("trajectories", "states.partial", "root_values", "sequence_scores", "prefix_scores", "rollout_audits"):
            journals[name] = pilot.CountingJournal(output/f"{name}.csv")
        (output/"forecasts").mkdir()
        manifest["new_full_outcomes_opened"] = profile == "full"
        prior.write_json(output/"manifest.json", manifest)
        print(json.dumps(dict(protocol=VERSION, budgets=counts(cfg), training="NONE"), indent=2), flush=True)
        combos = list(itertools.product(cfg["model_seeds"], cfg["cohorts"], cfg["physical_shocks"], cfg["conditions"]))
        with tqdm(total=counts(cfg)["states"], desc="Independent PM counterfactuals", unit="state") as progress:
            for seed, cohort, shock, condition in tqdm(combos, desc="Frozen multi-seed trajectories", unit="trajectory"):
                config = panels[cohort, condition]
                env = WaitingEnv(config)
                env.reset(physical_seed(cohort, shock), benchmark.initial_state(config, cohort))
                for tick in range(48):
                    if time.perf_counter()-started > cfg["hard_timeout_seconds"]:
                        raise TimeoutError("fixed diagnostic hard timeout exceeded")
                    state = env.state
                    identity = dict(condition=condition, cohort=cohort, shock=shock, model_seed=seed, tick=tick)
                    if tick in cfg["landmarks"]:
                        row, values = evaluate(config, state, identity, models[seed], cfg, journals)
                        row["forecast_file"] = f"forecasts/state_{len(rows):05d}.npz"
                        save_forecasts(output/row["forecast_file"], identity, values)
                        rows.append(row)
                        journals["states.partial"].add(row)
                        for value in values:
                            totals["transitions"] += value["transitions"]
                            for k, v in value["counts"].items():
                                totals[k] += v
                        progress.update()
                    action, _ = models[seed].select(config, state, 48-tick, 48)
                    validate_matching(config, state, action)
                    _, reward, _, info = env.step(action)
                    journals["trajectories"].add(dict(**identity, state=prior.compact(asdict(state)),
                        action=prior.compact(action), cost=-reward, following=prior.compact(asdict(env.state)), info=prior.compact(info)))
        if any(parameters_sha(models[s]) != value for s, value in initial.items()):
            raise RuntimeError("frozen parameters changed")
        summary, cells = summarize(rows, cfg)
        prior.write_json(output/"summary.json", summary)
        prior.write_csv(output/"states.csv", rows)
        prior.write_csv(output/"profile_cells.csv", cells)
        prior.write_csv(output/"strata.csv", summary["strata"])
        roots = sum(row["unique_roots"] for row in rows)
        scenarios = cfg["selection_scenarios"]+cfg["evaluation_scenarios"]
        expected_transitions = sum(row["unique_roots"]*scenarios*(60-row["tick"]) for row in rows)
        expected_decisions = sum(row["unique_roots"]*scenarios*(47-row["tick"]) for row in rows)
        if (len(rows) != counts(cfg)["states"] or journals["trajectories"].rows != counts(cfg)["trajectory_ticks"] or
                journals["states.partial"].rows != len(rows) or journals["root_values"].rows != 2*roots or
                journals["rollout_audits"].rows != 2*len(rows) or totals["transitions"] != expected_transitions or
                totals["transitions"] > counts(cfg)["max_forecast_transitions"] or
                totals["tail_rows"] != roots*scenarios*12 or totals["decision_rows"] != expected_decisions):
            raise RuntimeError("budget/clock journal count mismatch")
        manifest.update(status="COMPLETED", audit="PASS", frozen_parameter_hashes={str(s): v for s, v in initial.items()},
            actual_counts=dict(models=len(models), trajectories=len(combos), trajectory_ticks=journals["trajectories"].rows,
                states=len(rows), unique_roots=roots, training_steps=0, forecast_files=len(rows),
                forecast_transitions=totals["transitions"], tail_rows=totals["tail_rows"],
                decision_rows=totals["decision_rows"], near_tie_scalar_rows=totals["near_tie_scalar_rows"],
                sequence_score_rows=journals["sequence_scores"].rows, prefix_score_rows=journals["prefix_scores"].rows))
    except BaseException as error:
        manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for journal in journals.values():
            journal.close()
        manifest["finished"] = now()
        manifest["elapsed_seconds"] = time.perf_counter()-started
        manifest["artifacts"] = {str(path.relative_to(output)): dict(sha256=sha(path), bytes=path.stat().st_size)
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        prior.write_json(output/"manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--model-source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = run(args.output_dir, args.model_source_dir, args.profile, args.device)
    print(json.dumps(dict(status=manifest["status"], audit=manifest["audit"], elapsed_seconds=manifest["elapsed_seconds"], actual_counts=manifest["actual_counts"]), indent=2))


if __name__ == "__main__":
    main()
