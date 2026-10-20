"""DEV-only on-policy decomposition of matching choice and canonical-mask headroom."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import (
    CASES, HORIZON, FrameworkEnv, catalogue, configs, vector_observation,
)
from ht_pdm_fjsp.maintenance_framework_v7_policy import canonical_mask
from ht_pdm_fjsp.maintenance_request_policies import choose, request_action, risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings, state_from_dict, submit_requests, validate_state

ROOT = Path(__file__).resolve().parent.parent
SOURCE_RUN = ROOT / "research_result/framework_v7_full_20261020T012531Z"
VERSION = "v8_current_request_headroom_dev_v1"
ARMS = ("shared_canonical", "shared_free", "flat_free")
TRAIN_SEEDS = tuple(range(68000000, 68000005))
DEV_SEEDS = tuple(range(69100000, 69100032))
SMOKE_SEEDS = tuple(range(69110000, 69110002))
FORECAST_ANCHOR = 69120000
SOURCES = (
    "research_result/diagnose_v8_headroom.py",
    "research_result/v8_headroom_diagnostic_plan_vi.md",
    "src/ht_pdm_fjsp/maintenance_framework_v1.py",
    "src/ht_pdm_fjsp/maintenance_framework_v6_policy.py",
    "src/ht_pdm_fjsp/maintenance_framework_v7_policy.py",
    "src/ht_pdm_fjsp/maintenance_requests.py",
    "src/ht_pdm_fjsp/maintenance_request_policies.py",
    "src/ht_pdm_fjsp/maintenance_dispatch.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def clock_bin(tick):
    if 0 <= tick <= 14:
        return "early"
    if 15 <= tick <= 29:
        return "middle"
    if 30 <= tick <= 41:
        return "late"
    return None


def state_rank(case, arm, train_seed, physical_seed, tick):
    payload = f"{VERSION}:{case}:{arm}:{train_seed}:{physical_seed}:{tick}"
    return hashlib.sha256(payload.encode()).hexdigest()


def source_hashes():
    return {item: sha(ROOT / item) for item in SOURCES}


def clean_sources():
    modified = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *SOURCES],
                                       cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard",
                                         "--", *SOURCES], cwd=ROOT, text=True).splitlines()
    return not modified and not untracked


def load_models():
    source_manifest = json.loads((SOURCE_RUN / "manifest.json").read_text())
    assert source_manifest["status"] == "COMPLETED"
    selection_path = SOURCE_RUN / "selection.json"
    if sha(selection_path) != source_manifest["selection_sha256"]:
        raise RuntimeError("V7 selection hash mismatch")
    selected = json.loads(selection_path.read_text())
    models = {}
    checkpoints = {}
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            item = selected[arm][str(seed)]
            path = SOURCE_RUN / item["checkpoint"]
            if sha(path) != item["checkpoint_sha256"]:
                raise RuntimeError("V7 checkpoint hash mismatch")
            models[(arm, seed)] = MaskablePPO.load(path, device="cpu")
            checkpoints[f"{arm}/{seed}"] = dict(path=str(path), sha256=sha(path))
    return source_manifest, sha(selection_path), models, checkpoints


def current_actions(config, state, tick):
    submissions = request_action(config, state, tick, "threshold")
    admitted = submit_requests(config, state, tick, submissions)
    actions = catalogue(config.machines, config.technicians)
    legal = set(legal_matchings(config, admitted))
    mask = np.asarray([action in legal for action in actions], dtype=bool)
    canonical = canonical_mask(config, actions, mask)
    observation = vector_observation(config, admitted, tick)
    return submissions, admitted, actions, mask, canonical, observation


def choose_policy(model, arm, observation, mask, canonical):
    used = canonical if arm == "shared_canonical" else mask
    action = int(model.predict(observation, action_masks=used, deterministic=True)[0])
    if not used[action]:
        raise RuntimeError("policy selected illegal/masked action")
    return action


def collect(profile, models, output, deadline):
    seeds = DEV_SEEDS if profile == "dev" else SMOKE_SEEDS
    cap = 4 if profile == "dev" else 1
    all_configs = configs()
    candidates = {}
    seen = {}
    episodes = []
    episode_fields = ("case", "source_arm", "train_seed", "physical_seed", "objective",
                      "H_availability", "H_failures", "H_pending", "end_pending",
                      "queued_cm_ticks", "H_service_starts", "service_starts", "selection_seconds")
    grid = [(case, arm, train_seed, seed)
            for case in CASES for arm in ARMS for train_seed in TRAIN_SEEDS for seed in seeds]
    with (output / "episodes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, episode_fields)
        writer.writeheader()
        for case, arm, train_seed, seed in tqdm(grid, desc="Headroom on-policy trajectories", unit="episode"):
            if time.monotonic() > deadline:
                raise TimeoutError("headroom trajectory deadline")
            env = FrameworkEnv(all_configs[case])
            env.reset(seed=seed)
            starts = 0
            selection_time = 0.0
            model = models[(arm, train_seed)]
            for tick in range(HORIZON):
                state = env.inner.state
                submissions, admitted, actions, mask, canonical, obs = current_actions(env.config, state, tick)
                started = time.perf_counter()
                choice = choose_policy(model, arm, obs, mask, canonical)
                selection_time += time.perf_counter() - started
                bin_name = clock_bin(tick)
                if bin_name is not None:
                    stratum = f"{case}/{arm}/{train_seed}/{bin_name}"
                    seen[stratum] = seen.get(stratum, 0) + 1
                    if mask.sum() >= 2:
                        candidates.setdefault(stratum, []).append(dict(
                            case=case, source_arm=arm, train_seed=train_seed,
                            physical_seed=seed, tick=tick, clock_bin=bin_name,
                            source_action=choice, original=asdict(state),
                            rank=state_rank(case, arm, train_seed, seed, tick),
                        ))
                starts += len(actions[choice])
                _, _, done, _, _ = env.step(choice)
            if not done:
                raise RuntimeError("incomplete H episode")
            metrics = env.episode_metrics()
            row = dict(case=case, source_arm=arm, train_seed=train_seed,
                       physical_seed=seed, H_service_starts=starts,
                       selection_seconds=selection_time,
                       **{key: metrics[key] for key in episode_fields
                          if key not in ("case", "source_arm", "train_seed", "physical_seed",
                                         "H_service_starts", "selection_seconds")})
            writer.writerow(row)
            episodes.append(row)
    sampling = {}
    selected = []
    for case in CASES:
        for arm in ARMS:
            for train_seed in TRAIN_SEEDS:
                for bin_name in ("early", "middle", "late"):
                    stratum = f"{case}/{arm}/{train_seed}/{bin_name}"
                    available = candidates.get(stratum, [])
                    kept = sorted(available, key=lambda row: row["rank"])[:cap]
                    sampling[stratum] = dict(seen=seen.get(stratum, 0), eligible=len(available),
                                             selected=len(kept), cap=cap)
                    selected.extend(kept)
    return selected, sampling, episodes


def select_best(scores, indices, baseline_index):
    order = [int(baseline_index)] if baseline_index in indices else []
    order += [int(index) for index in sorted(indices) if index != baseline_index]
    if not order:
        raise ValueError("empty allowed action set")
    best = order[0]
    for index in order[1:]:
        if scores[index] < scores[best] - 1e-10:
            best = index
    return best


def category(source, planner, actions):
    if source == planner:
        return "match"
    if not actions[source]:
        return "under_service"
    if not actions[planner]:
        return "over_service"
    first = {machine for machine, _ in actions[source]}
    second = {machine for machine, _ in actions[planner]}
    return "assignment" if first == second else "machine_subset"


def gap_components(scores, chosen, full_best, allowed_best):
    actor = scores[chosen] - scores[allowed_best]
    mask = scores[allowed_best] - scores[full_best]
    full = scores[chosen] - scores[full_best]
    if min(actor, mask, full) < -1e-8 or abs(actor + mask - full) > 1e-8:
        raise RuntimeError("root-regret algebra failed")
    return actor, mask, full


def evaluate(profile, selected, models, output, deadline):
    all_configs = configs()
    scenarios = 16 if profile == "dev" else 4
    fields = ("state_id", "case", "source_arm", "train_seed", "physical_seed", "tick", "clock_bin",
              "policy_arm", "source_action", "policy_action", "planner_action", "risk_action",
              "category", "cardinality_delta", "planner_excluded_by_canonical",
              "actor_gap", "mask_gap", "full_regret", "risk_regret")
    summary = dict(states=0, forecast_transitions=0,
                   by_source={})
    with (output / "states.jsonl").open("x") as state_stream, \
            (output / "choices.csv").open("x", newline="") as choices_stream:
        writer = csv.DictWriter(choices_stream, fields)
        writer.writeheader()
        for state_id, item in enumerate(tqdm(selected, desc="Headroom root-value probes", unit="state")):
            if time.monotonic() > deadline:
                raise TimeoutError("headroom root-value deadline")
            config = all_configs[item["case"]]
            state = state_from_dict(item["original"])
            tick = item["tick"]
            validate_state(config, state, tick)
            submissions, admitted, actions, mask, canonical, obs = current_actions(config, state, tick)
            if mask.sum() < 2 or clock_bin(tick) != item["clock_bin"]:
                raise RuntimeError("selected source state no longer eligible")
            forecast_seed = role_seed(FORECAST_ANCHOR, f"{VERSION}:{item['rank']}")
            _, planner_matching, diagnostic = choose(
                config, state, tick, "threshold", "rollout",
                policy_horizon=HORIZON, forecast_seed=forecast_seed,
                scenarios=scenarios, forecast_horizon=6,
            )
            scores = {actions.index(tuple(row["matching"])): float(row["expected_cost"])
                      for row in diagnostic["root_scores"]}
            if set(scores) != set(np.flatnonzero(mask)):
                raise RuntimeError("planner root-score grid incomplete")
            risk = actions.index(risk_matching(config, admitted, tick))
            full_best = select_best(scores, set(np.flatnonzero(mask)), risk)
            allowed_best = select_best(scores, set(np.flatnonzero(canonical)), risk)
            planner = actions.index(planner_matching)
            if planner != full_best:
                raise RuntimeError("planner tie/argmin rule mismatch")
            source_choice = choose_policy(models[(item["source_arm"], item["train_seed"])],
                                          item["source_arm"], obs, mask, canonical)
            if source_choice != item["source_action"]:
                raise RuntimeError("source action replay mismatch")
            record = dict(state_id=state_id, **item, forecast_seed=forecast_seed,
                          submissions=submissions, legal_actions=list(map(int, np.flatnonzero(mask))),
                          canonical_actions=list(map(int, np.flatnonzero(canonical))),
                          planner_action=planner, best_canonical_action=allowed_best,
                          risk_action=risk, scores=scores,
                          forecast_transitions=diagnostic["forecast_transitions"])
            state_stream.write(json.dumps(record, sort_keys=True) + "\n")
            summary["states"] += 1
            summary["forecast_transitions"] += diagnostic["forecast_transitions"]
            stratum = f"{item['case']}/{item['source_arm']}/{item['train_seed']}"
            cell = summary["by_source"].setdefault(stratum, dict(
                states=0, source_full_regret_sum=0.0, source_actor_gap_sum=0.0,
                source_mask_gap_sum=0.0, source_gap_ge_1=0, source_gap_ge_5=0,
                source_gap_ge_10=0, mask_excluded_planner=0,
                source_categories={name: 0 for name in
                                   ("match", "under_service", "over_service", "assignment", "machine_subset")}))
            cell["states"] += 1
            for arm in ARMS:
                prediction = choose_policy(models[(arm, item["train_seed"])], arm, obs, mask, canonical)
                allowed = allowed_best if arm == "shared_canonical" else full_best
                actor_gap, mask_gap, full_regret = gap_components(scores, prediction, full_best, allowed)
                risk_regret = scores[risk] - scores[full_best]
                chosen_category = category(prediction, planner, actions)
                writer.writerow(dict(state_id=state_id, case=item["case"],
                                     source_arm=item["source_arm"], train_seed=item["train_seed"],
                                     physical_seed=item["physical_seed"], tick=tick,
                                     clock_bin=item["clock_bin"], policy_arm=arm,
                                     source_action=source_choice, policy_action=prediction,
                                     planner_action=planner, risk_action=risk,
                                     category=chosen_category,
                                     cardinality_delta=len(actions[prediction])-len(actions[planner]),
                                     planner_excluded_by_canonical=int(not canonical[planner]),
                                     actor_gap=actor_gap, mask_gap=mask_gap,
                                     full_regret=full_regret, risk_regret=risk_regret))
                if arm == item["source_arm"]:
                    cell["source_full_regret_sum"] += full_regret
                    cell["source_actor_gap_sum"] += actor_gap
                    cell["source_mask_gap_sum"] += mask_gap
                    for threshold in (1, 5, 10):
                        cell[f"source_gap_ge_{threshold}"] += full_regret >= threshold
                    cell["mask_excluded_planner"] += not canonical[planner]
                    cell["source_categories"][chosen_category] += 1
    return summary


def run(profile, output):
    if profile not in ("smoke", "dev"):
        raise ValueError(profile)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
    previous = set()
    for registry in (ROOT / "configs").glob("*seed_registry.json"):
        previous |= registered_integers(json.loads(registry.read_text()))
    reserved = set((*DEV_SEEDS, *SMOKE_SEEDS, FORECAST_ANCHOR))
    if reserved & previous or len(reserved) != len(DEV_SEEDS) + len(SMOKE_SEEDS) + 1:
        raise RuntimeError("headroom diagnostic seed overlap")
    if profile == "dev" and not clean_sources():
        raise RuntimeError("DEV headroom requires committed source and plan")
    started = time.monotonic()
    deadline = started + (300 if profile == "smoke" else 2400)
    parent, selection_hash, models, checkpoints = load_models()
    manifest = dict(version=VERSION, status="RUNNING", phase="preflight", profile=profile,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                    source_run=str(SOURCE_RUN), source_run_commit=parent["git"]["commit"],
                    selection_sha256=selection_hash, checkpoints=checkpoints,
                    source_hashes=source_hashes(),
                    seeds=dict(dev=list(DEV_SEEDS), smoke=list(SMOKE_SEEDS),
                               forecast_anchor=FORECAST_ANCHOR),
                    config_cases={key: asdict(value) for key, value in configs().items()},
                    runtime=dict(python=platform.python_version(), platform=platform.platform(),
                                 torch=torch.__version__),
                    timeout_seconds=300 if profile == "smoke" else 2400)
    save_json(output / "manifest.json", manifest)
    try:
        manifest["phase"] = "trajectories"
        save_json(output / "manifest.json", manifest)
        selected, sampling, episodes = collect(profile, models, output, deadline)
        save_json(output / "sampling.json", sampling)
        manifest["phase"] = "root_values"
        save_json(output / "manifest.json", manifest)
        summary = evaluate(profile, selected, models, output, deadline)
        summary["sampling"] = sampling
        summary["episodes"] = len(episodes)
        save_json(output / "summary.json", summary)
        if source_hashes() != manifest["source_hashes"]:
            raise RuntimeError("source changed during run")
        manifest.update(status="COMPLETED", phase="done", states=len(selected),
                        episodes=len(episodes), elapsed_seconds=time.monotonic()-started,
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        artifact_hashes={name: sha(output / name) for name in
                                         ("episodes.csv", "sampling.json", "states.jsonl",
                                          "choices.csv", "summary.json")})
        save_json(output / "manifest.json", manifest)
        print(json.dumps(dict(status="COMPLETED", output_dir=str(output),
                              episodes=len(episodes), states=len(selected))))
    except Exception as exc:
        manifest.update(status="FAILED", phase="failed", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic()-started)
        save_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "dev"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.profile, args.output_dir.resolve())


if __name__ == "__main__":
    main()
