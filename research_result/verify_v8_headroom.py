"""Independent V8 on-policy headroom source, root-value and decomposition audit."""
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

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import (
    CASES, HORIZON, FrameworkEnv, catalogue, configs, vector_observation,
)
from ht_pdm_fjsp.maintenance_framework_v7_policy import canonical_mask
from ht_pdm_fjsp.maintenance_request_policies import choose, request_action, risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings, state_from_dict, submit_requests, validate_state

ROOT = Path(__file__).resolve().parent.parent
VERSION = "v8_current_request_headroom_dev_v1"
ARMS = ("shared_canonical", "shared_free", "flat_free")
TRAIN = tuple(range(68000000, 68000005))
BINS = (("early", 0, 14), ("middle", 15, 29), ("late", 30, 41))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left, right, label):
    if not np.isclose(float(left), float(right), atol=1e-8, rtol=0):
        raise AssertionError(f"{label}: {left} != {right}")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def rank(case, arm, train_seed, physical_seed, tick):
    value = f"{VERSION}:{case}:{arm}:{train_seed}:{physical_seed}:{tick}"
    return hashlib.sha256(value.encode()).hexdigest()


def bin_at(tick):
    return next((name for name, lo, hi in BINS if lo <= tick <= hi), None)


def current(config, state, tick):
    submission = request_action(config, state, tick, "threshold")
    admitted = submit_requests(config, state, tick, submission)
    actions = catalogue()
    legal = set(legal_matchings(config, admitted))
    mask = np.asarray([action in legal for action in actions], dtype=bool)
    return submission, admitted, actions, mask, canonical_mask(config, actions, mask), \
        vector_observation(config, admitted, tick)


def predict(model, arm, obs, mask, canonical):
    allowed = canonical if arm == "shared_canonical" else mask
    index = int(model.predict(obs, action_masks=allowed, deterministic=True)[0])
    assert allowed[index]
    return index


def load_models(manifest, parent):
    source_manifest = json.loads((parent / "manifest.json").read_text())
    assert source_manifest["status"] == "COMPLETED"
    assert source_manifest["git"]["commit"] == manifest["source_run_commit"]
    selection_path = parent / "selection.json"
    assert sha(selection_path) == source_manifest["selection_sha256"] == manifest["selection_sha256"]
    selection = json.loads(selection_path.read_text())
    models = {}
    for arm in ARMS:
        for seed in TRAIN:
            item = selection[arm][str(seed)]
            path = parent / item["checkpoint"]
            digest = sha(path)
            assert digest == item["checkpoint_sha256"] == manifest["checkpoints"][f"{arm}/{seed}"]["sha256"]
            models[(arm, seed)] = MaskablePPO.load(path, device="cpu")
    return models


def regenerate(profile, models, episodes):
    seeds = tuple(range(69100000, 69100032)) if profile == "dev" else tuple(range(69110000, 69110002))
    cap = 4 if profile == "dev" else 1
    episode_map = {(row["case"], row["source_arm"], int(row["train_seed"]),
                    int(row["physical_seed"])): row for row in episodes}
    assert len(episode_map) == len(episodes) == len(CASES) * len(ARMS) * len(TRAIN) * len(seeds)
    candidates = {}
    observations = {}
    grid = [(case, arm, train_seed, seed) for case in CASES for arm in ARMS
            for train_seed in TRAIN for seed in seeds]
    for case, arm, train_seed, seed in tqdm(grid, desc="Independent headroom trajectory replay", unit="episode"):
        env = FrameworkEnv(configs()[case])
        env.reset(seed=seed)
        starts = 0
        for tick in range(HORIZON):
            state = env.inner.state
            _, _, actions, mask, canonical, obs = current(env.config, state, tick)
            action = predict(models[(arm, train_seed)], arm, obs, mask, canonical)
            bin_name = bin_at(tick)
            if bin_name and mask.sum() >= 2:
                key = f"{case}/{arm}/{train_seed}/{bin_name}"
                candidates.setdefault(key, []).append(dict(
                    case=case, source_arm=arm, train_seed=train_seed,
                    physical_seed=seed, tick=tick, clock_bin=bin_name,
                    rank=rank(case, arm, train_seed, seed, tick),
                    source_action=action, original=state,
                ))
            starts += len(actions[action])
            _, _, done, _, _ = env.step(action)
        assert done
        metrics = env.episode_metrics()
        row = episode_map[(case, arm, train_seed, seed)]
        for field in ("objective", "H_availability", "H_failures", "H_pending",
                      "end_pending", "queued_cm_ticks", "service_starts"):
            close(row[field], metrics[field], f"episode {case}/{arm}/{train_seed}/{seed}/{field}")
        assert int(row["H_service_starts"]) == starts
        assert float(row["selection_seconds"]) >= 0
        close(metrics["objective"], sum(metrics[k] for k in
              ("maintenance_cost", "unavailability_cost", "failure_cost")), "cost parts")
        assert metrics["requests_created"] - metrics["requests_completed"] == (
            metrics["end_pending"] + metrics["end_in_service"])
    sampling = {}
    selected = []
    for case in CASES:
        for arm in ARMS:
            for train_seed in TRAIN:
                for name, lo, hi in BINS:
                    key = f"{case}/{arm}/{train_seed}/{name}"
                    found = candidates.get(key, [])
                    keep = sorted(found, key=lambda item: item["rank"])[:cap]
                    sampling[key] = dict(seen=len(seeds) * (hi-lo+1), eligible=len(found),
                                         selected=len(keep), cap=cap)
                    selected.extend(keep)
    return selected, sampling


def best(scores, allowed, risk):
    ordering = ([risk] if risk in allowed else []) + [i for i in sorted(allowed) if i != risk]
    assert ordering
    choice = ordering[0]
    for candidate in ordering[1:]:
        if scores[candidate] < scores[choice] - 1e-10:
            choice = candidate
    return choice


def class_of(chosen, planner, actions):
    if chosen == planner:
        return "match"
    if not actions[chosen]:
        return "under_service"
    if not actions[planner]:
        return "over_service"
    return ("assignment" if {m for m, _ in actions[chosen]} ==
            {m for m, _ in actions[planner]} else "machine_subset")


def verify(run):
    torch.set_num_threads(1)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED" and manifest["version"] == VERSION
    profile = manifest["profile"]
    assert profile in ("smoke", "dev")
    assert manifest["seeds"]["dev"] == list(range(69100000, 69100032))
    assert manifest["seeds"]["smoke"] == list(range(69110000, 69110002))
    assert manifest["seeds"]["forecast_anchor"] == 69120000
    for path, digest in manifest["source_hashes"].items():
        assert sha(ROOT / path) == digest
    for name, digest in manifest["artifact_hashes"].items():
        assert sha(run / name) == digest
    models = load_models(manifest, Path(manifest["source_run"]))
    episodes = read_csv(run / "episodes.csv")
    selected, sampling = regenerate(profile, models, episodes)
    assert sampling == json.loads((run / "sampling.json").read_text())
    states = [json.loads(line) for line in (run / "states.jsonl").read_text().splitlines()]
    choices = read_csv(run / "choices.csv")
    assert len(states) == manifest["states"] == len(selected)
    assert len(choices) == len(states) * len(ARMS)
    choice_map = {(int(row["state_id"]), row["policy_arm"]): row for row in choices}
    assert len(choice_map) == len(choices)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["states"] == len(states) and summary["episodes"] == manifest["episodes"] == len(episodes)
    assert summary["sampling"] == sampling
    total_forecast = 0
    by_source = {}
    for state_id, (raw, expected) in enumerate(tqdm(zip(states, selected), total=len(states),
                                                     desc="Independent headroom root audit", unit="state")):
        assert raw["state_id"] == state_id
        for key in ("case", "source_arm", "train_seed", "physical_seed", "tick", "clock_bin", "rank", "source_action"):
            assert raw[key] == expected[key]
        state = state_from_dict(raw["original"])
        assert state == expected["original"]
        config = configs()[raw["case"]]
        tick = raw["tick"]
        validate_state(config, state, tick)
        submission, admitted, actions, mask, canonical, obs = current(config, state, tick)
        assert raw["submissions"] == list(submission)
        assert raw["legal_actions"] == list(map(int, np.flatnonzero(mask)))
        assert raw["canonical_actions"] == list(map(int, np.flatnonzero(canonical)))
        forecast = role_seed(69120000, f"{VERSION}:{raw['rank']}")
        assert raw["forecast_seed"] == forecast
        _, planner_matching, diagnostics = choose(
            config, state, tick, "threshold", "rollout", policy_horizon=HORIZON,
            forecast_seed=forecast, scenarios=16 if profile == "dev" else 4,
            forecast_horizon=6)
        scores = {actions.index(tuple(row["matching"])): float(row["expected_cost"])
                  for row in diagnostics["root_scores"]}
        assert set(scores) == set(np.flatnonzero(mask)) == {int(k) for k in raw["scores"]}
        for action, value in scores.items():
            close(raw["scores"][str(action)], value, "root score")
        risk = actions.index(risk_matching(config, admitted, tick))
        full_best = best(scores, set(scores), risk)
        restricted_best = best(scores, set(map(int, np.flatnonzero(canonical))), risk)
        assert raw["planner_action"] == actions.index(planner_matching) == full_best
        assert raw["best_canonical_action"] == restricted_best
        assert raw["risk_action"] == risk
        assert raw["forecast_transitions"] == diagnostics["forecast_transitions"]
        total_forecast += diagnostics["forecast_transitions"]
        source = f"{raw['case']}/{raw['source_arm']}/{raw['train_seed']}"
        cell = by_source.setdefault(source, dict(
            states=0, source_full_regret_sum=0.0, source_actor_gap_sum=0.0,
            source_mask_gap_sum=0.0, source_gap_ge_1=0, source_gap_ge_5=0,
            source_gap_ge_10=0, mask_excluded_planner=0,
            source_categories={name: 0 for name in
                               ("match", "under_service", "over_service", "assignment", "machine_subset")}))
        cell["states"] += 1
        for arm in ARMS:
            row = choice_map[(state_id, arm)]
            assert row["case"] == raw["case"] and row["source_arm"] == raw["source_arm"]
            assert int(row["train_seed"]) == raw["train_seed"]
            assert int(row["physical_seed"]) == raw["physical_seed"] and int(row["tick"]) == tick
            chosen = predict(models[(arm, raw["train_seed"])], arm, obs, mask, canonical)
            assert int(row["policy_action"]) == chosen
            assert int(row["source_action"]) == raw["source_action"]
            assert int(row["planner_action"]) == full_best and int(row["risk_action"]) == risk
            permitted_best = restricted_best if arm == "shared_canonical" else full_best
            actor_gap = scores[chosen] - scores[permitted_best]
            mask_gap = scores[permitted_best] - scores[full_best]
            full_gap = scores[chosen] - scores[full_best]
            assert min(actor_gap, mask_gap, full_gap) >= -1e-8
            close(actor_gap + mask_gap, full_gap, "gap algebra")
            for key, value in (("actor_gap", actor_gap), ("mask_gap", mask_gap),
                               ("full_regret", full_gap),
                               ("risk_regret", scores[risk] - scores[full_best])):
                close(row[key], value, key)
            assert row["category"] == class_of(chosen, full_best, actions)
            assert int(row["cardinality_delta"]) == len(actions[chosen]) - len(actions[full_best])
            assert int(row["planner_excluded_by_canonical"]) == int(not canonical[full_best])
            if arm == raw["source_arm"]:
                assert chosen == raw["source_action"]
                cell["source_full_regret_sum"] += full_gap
                cell["source_actor_gap_sum"] += actor_gap
                cell["source_mask_gap_sum"] += mask_gap
                for threshold in (1, 5, 10):
                    cell[f"source_gap_ge_{threshold}"] += full_gap >= threshold
                cell["mask_excluded_planner"] += not canonical[full_best]
                cell["source_categories"][row["category"]] += 1
    assert summary["forecast_transitions"] == total_forecast
    assert set(summary["by_source"]) == set(by_source)
    for source, item in by_source.items():
        reported = summary["by_source"][source]
        assert item["states"] == reported["states"]
        assert item["source_categories"] == reported["source_categories"]
        for key, value in item.items():
            if key != "source_categories":
                close(reported[key], value, f"summary {source}/{key}")
    return dict(status="PASS", run=str(run), profile=profile,
                episodes=len(episodes), states=len(states), choices=len(choices),
                forecast_transitions=total_forecast,
                checks=["source/artifact/checkpoint hashes", "full independent trajectory and episode-cost replay",
                        "hash-selected clock strata", "complete common-bank root scores",
                        "same-state learned/risk actions", "exact actor/mask regret decomposition and categories"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = verify(args.run_dir.resolve())
    (output / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
