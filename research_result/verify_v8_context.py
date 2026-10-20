"""Independent source-trajectory, pair, planner and actor audit for V8 DEV probe."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
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
from ht_pdm_fjsp.maintenance_requests import (
    legal_matchings, state_from_dict, submit_requests, transition, validate_state,
)

ROOT = Path(__file__).resolve().parent.parent
VERSION = "v8_absent_request_context_dev_v1"
ARMS = ("shared_free", "shared_canonical", "flat_free")
SOURCES = ("risk_matching", *(f"shared_canonical/{seed}" for seed in range(68000000, 68000005)))
TRAIN = tuple(range(68000000, 68000005))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(case, source, seed, tick, machine):
    return hashlib.sha256(f"{VERSION}:{case}:{source}:{seed}:{tick}:{machine}".encode()).hexdigest()


def close(left, right, label):
    if not np.isclose(float(left), float(right), rtol=0, atol=1e-8):
        raise AssertionError(f"{label}: {left} != {right}")


def load_models(source_run, manifest):
    selection_path = source_run / "selection.json"
    assert sha(selection_path) == manifest["selection_sha256"]
    selection = json.loads(selection_path.read_text())
    models = {}
    for arm in ARMS:
        for seed in TRAIN:
            item = selection[arm][str(seed)]
            path = source_run / item["checkpoint"]
            assert sha(path) == item["checkpoint_sha256"] == manifest["checkpoints"][f"{arm}/{seed}"]["sha256"]
            models[(arm, seed)] = MaskablePPO.load(path, device="cpu")
    return models


def independent_candidates(config, state, tick):
    if tick < 4 or tick > 41:
        return None
    submission = request_action(config, state, tick, "threshold")
    admitted = submit_requests(config, state, tick, submission)
    if not any(request is not None and request.started_at is None for request in admitted.requests):
        return None
    if len(legal_matchings(config, admitted)) < 2:
        return None
    for machine in range(config.machines):
        if (state.requests[machine] is None and not state.physical.failed[machine]
                and machine not in state.physical.assigned and state.physical.ages[machine] <= 1):
            return machine
    return None


def regenerate_selected(profile, seeds, models):
    cap = 16 if profile == "dev" else 2
    cases = configs()
    actions = catalogue()
    candidates = {f"{case}/{source}": [] for case in CASES for source in SOURCES}
    grid = [(case, source, seed) for case in CASES for source in SOURCES for seed in seeds]
    for case, source, seed in tqdm(grid, desc="Independent source replay", unit="episode"):
        env = FrameworkEnv(cases[case])
        env.reset(seed=seed)
        for tick in range(HORIZON):
            state = env.inner.state
            machine = independent_candidates(env.config, state, tick)
            if machine is not None:
                candidates[f"{case}/{source}"].append(dict(
                    case=case, source=source, physical_seed=seed, tick=tick,
                    changed_machine=machine, rank=rank(case, source, seed, tick, machine),
                    original=state,
                ))
            submission = request_action(env.config, state, tick, "threshold")
            admitted = submit_requests(env.config, state, tick, submission)
            legal = legal_matchings(env.config, admitted)
            if source == "risk_matching":
                matching = risk_matching(env.config, admitted, tick)
            else:
                training_seed = int(source.split("/")[1])
                mask = np.asarray([action in legal for action in actions], dtype=bool)
                restricted = canonical_mask(env.config, actions, mask)
                observation = vector_observation(env.config, admitted, tick)
                index = int(models[("shared_canonical", training_seed)].predict(
                    observation, action_masks=restricted, deterministic=True)[0])
                assert restricted[index]
                matching = actions[index]
            assert matching in legal
            env.step(actions.index(matching))
    chosen = {}
    counts = {}
    for stratum, items in candidates.items():
        keep = sorted(items, key=lambda item: item["rank"])[:cap]
        counts[stratum] = dict(seen=len(seeds) * HORIZON, eligible=len(items),
                               selected=len(keep), cap=cap)
        chosen.update({(item["case"], item["source"], item["physical_seed"],
                        item["tick"], item["changed_machine"]): item for item in keep})
    return chosen, counts


def verify(run):
    torch.set_num_threads(1)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED" and manifest["version"] == VERSION
    profile = manifest["profile"]
    assert profile in ("smoke", "dev")
    seeds = tuple(range(69000000, 69000032)) if profile == "dev" else tuple(range(69010000, 69010002))
    assert manifest["seeds"]["dev"] == list(range(69000000, 69000032))
    assert manifest["seeds"]["smoke"] == list(range(69010000, 69010002))
    assert manifest["seeds"]["forecast_anchor"] == 69020000
    for relative, digest in manifest["source_hashes"].items():
        assert sha(ROOT / relative) == digest
    for name, digest in manifest["artifact_hashes"].items():
        assert sha(run / name) == digest
    source_run = Path(manifest["source_run"])
    source_manifest = json.loads((source_run / "manifest.json").read_text())
    assert source_manifest["status"] == "COMPLETED"
    assert source_manifest["git"]["commit"] == manifest["source_run_commit"]
    models = load_models(source_run, manifest)
    expected_selected, expected_counts = regenerate_selected(profile, seeds, models)
    sampling = json.loads((run / "sampling.json").read_text())
    assert sampling == expected_counts
    pairs = [json.loads(line) for line in (run / "pairs.jsonl").read_text().splitlines()]
    assert len(pairs) == manifest["pairs"] == len(expected_selected)
    with (run / "policy.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(pairs) * len(ARMS) * len(TRAIN)
    policy_cells = {(int(row["pair_id"]), row["arm"], int(row["train_seed"])): row for row in rows}
    assert len(policy_cells) == len(rows)
    actions = catalogue()
    cases = configs()
    counters = dict(planner_action_flips=0, planner_material_flips_1=0,
                    planner_material_flips_5=0, planner_material_flips_10=0)
    for pair_id, pair in enumerate(tqdm(pairs, desc="Independent paired audit", unit="pair")):
        assert pair["pair_id"] == pair_id
        key = (pair["case"], pair["source"], pair["physical_seed"],
               pair["tick"], pair["changed_machine"])
        assert key in expected_selected
        source_state = expected_selected.pop(key)["original"]
        original = state_from_dict(pair["original"])
        changed = state_from_dict(pair["changed"])
        assert source_state == original
        config = cases[pair["case"]]
        tick = pair["tick"]
        validate_state(config, original, tick)
        validate_state(config, changed, tick)
        machine = pair["changed_machine"]
        assert original.physical.ages[machine] <= 1 and changed.physical.ages[machine] == 2
        assert replace(changed, physical=replace(changed.physical, ages=original.physical.ages)) == original
        assert pair["rank"] == rank(*key)
        original_submission = request_action(config, original, tick, "threshold")
        changed_submission = request_action(config, changed, tick, "threshold")
        assert original_submission == changed_submission
        assert pair["original_submissions"] == pair["changed_submissions"] == list(original_submission)
        original_admitted = submit_requests(config, original, tick, original_submission)
        changed_admitted = submit_requests(config, changed, tick, changed_submission)
        legal = legal_matchings(config, original_admitted)
        assert legal == legal_matchings(config, changed_admitted)
        mask = np.asarray([action in legal for action in actions], dtype=bool)
        canonical = canonical_mask(config, actions, mask)
        assert pair["legal_actions"] == list(np.flatnonzero(mask))
        assert pair["canonical_actions"] == list(np.flatnonzero(canonical))
        forecast_seed = role_seed(69020000, f"{VERSION}:{pair['rank']}")
        assert pair["forecast_seed"] == forecast_seed
        recorded = pair["planner"]
        replanned = []
        for state in (original, changed):
            _, match, detail = choose(config, state, tick, "threshold", "rollout",
                                      policy_horizon=HORIZON, forecast_seed=forecast_seed,
                                      scenarios=16 if profile == "dev" else 4, forecast_horizon=6)
            replanned.append((actions.index(match),
                              {actions.index(tuple(item["matching"])): item["expected_cost"]
                               for item in detail["root_scores"]},
                              detail["forecast_transitions"]))
        old, new = replanned
        assert recorded["original_action"] == old[0] and recorded["changed_action"] == new[0]
        assert recorded["forecast_transitions"] == old[2] + new[2]
        for observed, actual in ((recorded["original_scores"], old[1]),
                                 (recorded["changed_scores"], new[1])):
            assert {int(key) for key in observed} == set(actual)
            for action, value in actual.items():
                close(observed[str(action)], value, "planner root score")
        old_advantage = old[1][new[0]] - old[1][old[0]]
        new_advantage = new[1][old[0]] - new[1][new[0]]
        close(recorded["original_advantage"], old_advantage, "old flip advantage")
        close(recorded["changed_advantage"], new_advantage, "new flip advantage")
        counters["planner_action_flips"] += old[0] != new[0]
        for threshold in (1, 5, 10):
            flagged = old[0] != new[0] and min(old_advantage, new_advantage) >= threshold
            assert recorded[f"material_flip_{threshold}"] == flagged
            counters[f"planner_material_flips_{threshold}"] += flagged
        zero_events = (False,) * config.machines
        follow_old, _, _ = transition(config, original, tick, original_submission,
                                      actions[old[0]], zero_events)
        follow_new, _, _ = transition(config, changed, tick, changed_submission,
                                      actions[old[0]], zero_events)
        assert machine not in request_action(config, follow_old, tick + 1, "threshold")
        assert machine in request_action(config, follow_new, tick + 1, "threshold")
        obs = (vector_observation(config, original_admitted, tick),
               vector_observation(config, changed_admitted, tick))
        for arm in ARMS:
            used_mask = canonical if arm == "shared_canonical" else mask
            for seed in TRAIN:
                row = policy_cells[(pair_id, arm, seed)]
                assert row["case"] == pair["case"] and row["source"] == pair["source"]
                assert int(row["physical_seed"]) == pair["physical_seed"]
                assert int(row["tick"]) == tick and int(row["changed_machine"]) == machine
                model = models[(arm, seed)]
                prediction = [int(model.predict(value, action_masks=used_mask,
                                                deterministic=True)[0]) for value in obs]
                assert used_mask[prediction[0]] and used_mask[prediction[1]]
                assert prediction == [int(row["original_action"]), int(row["changed_action"])]
                assert int(row["action_changed"]) == (prediction[0] != prediction[1])
                assert int(row["planner_material_flip_5"]) == recorded["material_flip_5"]
                for j, scores in enumerate((old[1], new[1])):
                    regret = scores[prediction[j]] - min(scores.values())
                    close(row[("original", "changed")[j] + "_forecast_regret"], regret,
                          "policy forecast regret")
                if arm.startswith("shared_"):
                    with torch.no_grad():
                        logits = [model.policy.action_net(torch.as_tensor(value[None])).numpy()[0]
                                  for value in obs]
                    error = np.max(np.abs(logits[0][mask] - logits[1][mask]))
                    close(row["max_legal_logit_error"], error, "shared actor logit error")
                    assert error <= 1e-6
                else:
                    assert row["max_legal_logit_error"] == "nan"
    assert not expected_selected
    summary = json.loads((run / "summary.json").read_text())
    assert summary["sampling"] == sampling and summary["pairs"] == len(pairs)
    for key, value in counters.items():
        assert summary[key] == value
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        item = summary["policy"][arm]
        assert item["states"] == len(subset)
        assert item["action_flips"] == sum(int(row["action_changed"]) for row in subset)
        assert item["material_planner_flip_action_flips"] == sum(
            int(row["action_changed"]) * int(row["planner_material_flip_5"]) for row in subset)
        for field in ("original", "changed"):
            close(item[field + "_regret_mean"], np.mean([float(row[field + "_forecast_regret"])
                                                       for row in subset]), "mean forecast regret")
    return dict(status="PASS", source_run=str(run), profile=profile,
                pairs=len(pairs), policy_rows=len(rows), planner_counts=counters,
                checks=["source/artifact/checkpoint hashes", "independently regenerated full source trajectories",
                        "hash-ranked selected state pairs", "valid paired request/physical state and root legality",
                        "common-bank planner root scores and material flips", "all policy actions/logits/regrets"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = verify(args.run_dir.resolve())
    (out / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
