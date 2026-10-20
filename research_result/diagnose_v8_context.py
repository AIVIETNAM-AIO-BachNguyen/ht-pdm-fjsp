"""Prelocked DEV-only paired-state probe of absent-request machine context."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
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
from ht_pdm_fjsp.maintenance_requests import (
    legal_matchings, submit_requests, transition, validate_state,
)
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers

ROOT = Path(__file__).resolve().parent.parent
VERSION = "v8_absent_request_context_dev_v1"
PLAN = ROOT / "research_result/v8_context_diagnostic_plan_vi.md"
V7_RUN = ROOT / "research_result/framework_v7_full_20261020T012531Z"
ACTOR_ARMS = ("shared_free", "shared_canonical", "flat_free")
SOURCE_ARMS = ("risk_matching", *(f"shared_canonical/{seed}" for seed in range(68000000, 68000005)))
DEV_SEEDS = tuple(range(69000000, 69000032))
SMOKE_SEEDS = tuple(range(69010000, 69010002))
FORECAST_ANCHOR = 69020000
SOURCE_FILES = (
    "research_result/diagnose_v8_context.py",
    "research_result/v8_context_diagnostic_plan_vi.md",
    "src/ht_pdm_fjsp/maintenance_framework_v1.py",
    "src/ht_pdm_fjsp/maintenance_framework_v6_policy.py",
    "src/ht_pdm_fjsp/maintenance_framework_v7_policy.py",
    "src/ht_pdm_fjsp/maintenance_requests.py",
    "src/ht_pdm_fjsp/maintenance_request_policies.py",
    "src/ht_pdm_fjsp/maintenance_dispatch.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def matching_index(actions, matching) -> int:
    return actions.index(tuple(matching))


def selected_absent_machine(config, state, time: int):
    """Return a valid original/age-2 pair or None, before observing outcomes."""
    if not 4 <= time <= 41:
        return None
    submission = request_action(config, state, time, "threshold")
    admitted = submit_requests(config, state, time, submission)
    if not any(r is not None and r.started_at is None for r in admitted.requests):
        return None
    if len(legal_matchings(config, admitted)) < 2:
        return None
    busy = set(state.physical.assigned)
    candidates = [m for m in range(config.machines)
                  if state.requests[m] is None and not state.physical.failed[m]
                  and m not in busy and state.physical.ages[m] <= 1]
    if not candidates:
        return None
    machine = candidates[0]
    ages = list(state.physical.ages)
    ages[machine] = 2
    changed = replace(state, physical=replace(state.physical, ages=tuple(ages)))
    validate_state(config, state, time)
    validate_state(config, changed, time)
    changed_submission = request_action(config, changed, time, "threshold")
    changed_admitted = submit_requests(config, changed, time, changed_submission)
    if submission != changed_submission or admitted.requests != changed_admitted.requests:
        raise RuntimeError("root request changed under below-threshold age perturbation")
    if legal_matchings(config, admitted) != legal_matchings(config, changed_admitted):
        raise RuntimeError("root legal matching changed under paired intervention")
    if state.physical.remaining != changed.physical.remaining or state.physical.assigned != changed.physical.assigned:
        raise RuntimeError("paired technician state differs")
    return machine, changed


def next_tick_request_difference(config, original, changed, time, root):
    submissions = request_action(config, original, time, "threshold")
    zero_events = (False,) * config.machines
    original_next, _, _ = transition(config, original, time, submissions, root, zero_events)
    changed_next, _, _ = transition(config, changed, time, submissions, root, zero_events)
    old_next = request_action(config, original_next, time + 1, "threshold")
    new_next = request_action(config, changed_next, time + 1, "threshold")
    changed_machine = next(m for m, (a, b) in enumerate(zip(original.physical.ages,
                                                               changed.physical.ages)) if a != b)
    return changed_machine not in old_next and changed_machine in new_next


def age_pair_rank(case, source, seed, tick, machine):
    identifier = f"{VERSION}:{case}:{source}:{seed}:{tick}:{machine}"
    return hashlib.sha256(identifier.encode()).hexdigest()


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def committed_inputs():
    relative = list(SOURCE_FILES)
    modified = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *relative],
                                       cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard",
                                         "--", *relative], cwd=ROOT, text=True).splitlines()
    return not modified and not untracked


def candidate_sources(selection, checkpoints):
    models = {}
    for source in SOURCE_ARMS:
        if source == "risk_matching":
            continue
        seed = source.split("/")[1]
        models[source] = checkpoints[("shared_canonical", seed)]
    return models


def load_checkpoints():
    manifest = json.loads((V7_RUN / "manifest.json").read_text())
    if manifest["status"] != "COMPLETED" or manifest["protocol"] != "maintenance_framework_v7_canonical_assignment":
        raise RuntimeError("requires completed V7 source run")
    selection_path = V7_RUN / "selection.json"
    if sha(selection_path) != manifest["selection_sha256"]:
        raise RuntimeError("V7 checkpoint selection hash mismatch")
    selection = json.loads(selection_path.read_text())
    models = {}
    provenance = {}
    for arm in ACTOR_ARMS:
        for seed in range(68000000, 68000005):
            item = selection[arm][str(seed)]
            checkpoint = V7_RUN / item["checkpoint"]
            if sha(checkpoint) != item["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch {arm}/{seed}")
            models[(arm, str(seed))] = MaskablePPO.load(checkpoint, device="cpu")
            provenance[f"{arm}/{seed}"] = dict(path=str(checkpoint), sha256=sha(checkpoint))
    return manifest, selection_path, models, provenance


def choose_source_action(source, models, config, state, time):
    submission = request_action(config, state, time, "threshold")
    admitted = submit_requests(config, state, time, submission)
    actions = catalogue(config.machines, config.technicians)
    legal = set(legal_matchings(config, admitted))
    mask = np.asarray([action in legal for action in actions], dtype=bool)
    if source == "risk_matching":
        return submission, risk_matching(config, admitted, time)
    seed = source.split("/")[1]
    restricted = canonical_mask(config, actions, mask)
    observation = vector_observation(config, admitted, time)
    index = int(models[("shared_canonical", seed)].predict(
        observation, action_masks=restricted, deterministic=True)[0])
    if not restricted[index]:
        raise RuntimeError("source model selected masked action")
    return submission, actions[index]


def collect_states(profile, models, deadline):
    seeds = DEV_SEEDS if profile == "dev" else SMOKE_SEEDS
    cap = 16 if profile == "dev" else 2
    all_configs = configs()
    eligible = {}
    seen = {}
    grid = [(case, source, seed) for case in CASES for source in SOURCE_ARMS for seed in seeds]
    with tqdm(grid, desc="Context source trajectories", unit="episode") as bar:
        for case, source, seed in bar:
            if time.monotonic() > deadline:
                raise TimeoutError("context diagnostic trajectory deadline")
            env = FrameworkEnv(all_configs[case])
            env.reset(seed=seed)
            stratum = f"{case}/{source}"
            for tick in range(HORIZON):
                state = env.inner.state
                seen[stratum] = seen.get(stratum, 0) + 1
                pair = selected_absent_machine(env.config, state, tick)
                if pair is not None:
                    machine, changed = pair
                    eligible.setdefault(stratum, []).append(dict(
                        case=case, source=source, physical_seed=seed, tick=tick,
                        changed_machine=machine, original=asdict(state),
                        changed=asdict(changed),
                        rank=age_pair_rank(case, source, seed, tick, machine),
                    ))
                submissions, matching = choose_source_action(source, models, env.config, state, tick)
                if matching not in legal_matchings(env.config, submit_requests(env.config, state, tick, submissions)):
                    raise RuntimeError("source selected illegal matching")
                env.step(matching_index(env.actions, matching))
    selected = []
    counts = {}
    for stratum in sorted(seen):
        found = eligible.get(stratum, [])
        retained = sorted(found, key=lambda item: item["rank"])[:cap]
        counts[stratum] = dict(seen=seen[stratum], eligible=len(found), selected=len(retained), cap=cap)
        selected.extend(retained)
    return selected, counts


def compare_planner(config, original, changed, tick, forecast_seed, scenarios, actions):
    results = []
    for state in (original, changed):
        _, chosen, diagnostics = choose(config, state, tick, "threshold", "rollout",
                                        policy_horizon=HORIZON, forecast_seed=forecast_seed,
                                        scenarios=scenarios, forecast_horizon=6)
        scores = {matching_index(actions, entry["matching"]): float(entry["expected_cost"])
                  for entry in diagnostics["root_scores"]}
        if len(scores) < 2:
            raise RuntimeError("missing planner root scores")
        results.append((matching_index(actions, chosen), scores, diagnostics["forecast_transitions"]))
    left, right = results
    old_advantage = left[1][right[0]] - left[1][left[0]]
    new_advantage = right[1][left[0]] - right[1][right[0]]
    return dict(original_action=left[0], changed_action=right[0],
                original_scores=left[1], changed_scores=right[1],
                original_advantage=old_advantage, changed_advantage=new_advantage,
                forecast_transitions=left[2] + right[2],
                material_flip_1=left[0] != right[0] and min(old_advantage, new_advantage) >= 1.0,
                material_flip_5=left[0] != right[0] and min(old_advantage, new_advantage) >= 5.0,
                material_flip_10=left[0] != right[0] and min(old_advantage, new_advantage) >= 10.0)


def evaluate_pairs(profile, pairs, models, output, deadline):
    all_configs = configs()
    actions = catalogue()
    scenarios = 16 if profile == "dev" else 4
    policy_fields = ("pair_id", "case", "source", "physical_seed", "tick", "changed_machine",
                     "arm", "train_seed", "original_action", "changed_action", "action_changed",
                     "max_legal_logit_error", "original_forecast_regret", "changed_forecast_regret",
                     "planner_material_flip_5")
    summary = dict(pairs=0, planner_action_flips=0, planner_material_flips_1=0,
                   planner_material_flips_5=0, planner_material_flips_10=0,
                   policy={arm: dict(states=0, action_flips=0, material_planner_flip_action_flips=0,
                                     original_regret_sum=0.0, changed_regret_sum=0.0,
                                     max_legal_logit_error=0.0) for arm in ACTOR_ARMS})
    pair_path = output / "pairs.jsonl"
    policy_path = output / "policy.csv"
    with pair_path.open("x") as pair_stream, policy_path.open("x", newline="") as policy_stream:
        writer = csv.DictWriter(policy_stream, policy_fields)
        writer.writeheader()
        for identifier, pair in enumerate(tqdm(pairs, desc="Context paired evaluation", unit="pair")):
            if time.monotonic() > deadline:
                raise TimeoutError("context diagnostic evaluation deadline")
            config = all_configs[pair["case"]]
            from ht_pdm_fjsp.maintenance_requests import state_from_dict
            original = state_from_dict(pair["original"])
            changed = state_from_dict(pair["changed"])
            tick = pair["tick"]
            validate_state(config, original, tick)
            validate_state(config, changed, tick)
            original_submit = request_action(config, original, tick, "threshold")
            changed_submit = request_action(config, changed, tick, "threshold")
            original_admitted = submit_requests(config, original, tick, original_submit)
            changed_admitted = submit_requests(config, changed, tick, changed_submit)
            legal = set(legal_matchings(config, original_admitted))
            if original_submit != changed_submit or legal != set(legal_matchings(config, changed_admitted)):
                raise RuntimeError("paired root feasibility changed")
            mask = np.asarray([action in legal for action in actions], dtype=bool)
            canonical = canonical_mask(config, actions, mask)
            seed = role_seed(FORECAST_ANCHOR, f"{VERSION}:{pair['rank']}")
            planner = compare_planner(config, original, changed, tick, seed, scenarios, actions)
            if not next_tick_request_difference(config, original, changed, tick, actions[planner["original_action"]]):
                raise RuntimeError("paired context did not create next-tick request difference")
            record = dict(pair_id=identifier, **pair, forecast_seed=seed,
                          planner=planner, original_submissions=original_submit,
                          changed_submissions=changed_submit,
                          legal_actions=[i for i, ok in enumerate(mask) if ok],
                          canonical_actions=[i for i, ok in enumerate(canonical) if ok],
                          next_tick_request_changed=True)
            pair_stream.write(json.dumps(record, sort_keys=True) + "\n")
            summary["pairs"] += 1
            summary["planner_action_flips"] += planner["original_action"] != planner["changed_action"]
            for threshold in (1, 5, 10):
                summary[f"planner_material_flips_{threshold}"] += planner[f"material_flip_{threshold}"]
            obs0 = vector_observation(config, original_admitted, tick)
            obs1 = vector_observation(config, changed_admitted, tick)
            for arm in ACTOR_ARMS:
                policy_mask = canonical if arm == "shared_canonical" else mask
                for train_seed in range(68000000, 68000005):
                    model = models[(arm, str(train_seed))]
                    selected = []
                    for observation in (obs0, obs1):
                        action = int(model.predict(observation, action_masks=policy_mask,
                                                   deterministic=True)[0])
                        if not policy_mask[action]:
                            raise RuntimeError("policy selected masked action")
                        selected.append(action)
                    error = float("nan")
                    if arm.startswith("shared_"):
                        with torch.no_grad():
                            left_logits = model.policy.action_net(torch.as_tensor(obs0[None])).numpy()[0]
                            right_logits = model.policy.action_net(torch.as_tensor(obs1[None])).numpy()[0]
                        error = float(np.max(np.abs(left_logits[mask] - right_logits[mask])))
                        if error > 1e-6:
                            raise RuntimeError("shared actor changed legal logits under absent-machine context")
                    old_regret = planner["original_scores"][selected[0]] - min(planner["original_scores"].values())
                    new_regret = planner["changed_scores"][selected[1]] - min(planner["changed_scores"].values())
                    if min(old_regret, new_regret) < -1e-8:
                        raise RuntimeError("negative forecast regret")
                    writer.writerow(dict(pair_id=identifier, case=pair["case"], source=pair["source"],
                                         physical_seed=pair["physical_seed"], tick=tick,
                                         changed_machine=pair["changed_machine"], arm=arm,
                                         train_seed=train_seed, original_action=selected[0],
                                         changed_action=selected[1], action_changed=int(selected[0] != selected[1]),
                                         max_legal_logit_error=error,
                                         original_forecast_regret=old_regret,
                                         changed_forecast_regret=new_regret,
                                         planner_material_flip_5=int(planner["material_flip_5"])))
                    cell = summary["policy"][arm]
                    cell["states"] += 1
                    cell["action_flips"] += selected[0] != selected[1]
                    cell["material_planner_flip_action_flips"] += (
                        selected[0] != selected[1] and planner["material_flip_5"])
                    cell["original_regret_sum"] += old_regret
                    cell["changed_regret_sum"] += new_regret
                    if arm.startswith("shared_"):
                        cell["max_legal_logit_error"] = max(cell["max_legal_logit_error"], error)
    for arm, cell in summary["policy"].items():
        if cell["states"]:
            cell["original_regret_mean"] = cell["original_regret_sum"] / cell["states"]
            cell["changed_regret_mean"] = cell["changed_regret_sum"] / cell["states"]
    return summary


def run(profile, output):
    if profile not in ("smoke", "dev"):
        raise ValueError(profile)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + (300 if profile == "smoke" else 2400)
    torch.set_num_threads(1)
    registry_values = set()
    for path in (ROOT / "configs").glob("*seed_registry.json"):
        registry_values |= registered_integers(json.loads(path.read_text()))
    diagnostic_values = set((*DEV_SEEDS, *SMOKE_SEEDS, FORECAST_ANCHOR))
    if diagnostic_values & registry_values or len(diagnostic_values) != len(DEV_SEEDS) + len(SMOKE_SEEDS) + 1:
        raise RuntimeError("diagnostic seed collision")
    if profile == "dev" and not committed_inputs():
        raise RuntimeError("full DEV diagnostic requires committed source and plan")
    source_manifest, selection_path, models, checkpoint_provenance = load_checkpoints()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = dict(version=VERSION, status="RUNNING", phase="preflight", profile=profile,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    git_commit=git_commit, source_run=str(V7_RUN),
                    source_run_commit=source_manifest["git"]["commit"],
                    selection_sha256=sha(selection_path), source_hashes=source_hashes(),
                    checkpoints=checkpoint_provenance,
                    seeds=dict(dev=list(DEV_SEEDS), smoke=list(SMOKE_SEEDS),
                               forecast_anchor=FORECAST_ANCHOR),
                    config_cases={key: asdict(value) for key, value in configs().items()},
                    runtime=dict(python=platform.python_version(), platform=platform.platform(),
                                 torch=torch.__version__), timeout_seconds=300 if profile == "smoke" else 2400)
    write_json(output / "manifest.json", manifest)
    try:
        manifest["phase"] = "trajectories"
        write_json(output / "manifest.json", manifest)
        pairs, counts = collect_states(profile, models, deadline)
        write_json(output / "sampling.json", counts)
        manifest["phase"] = "paired_evaluation"
        write_json(output / "manifest.json", manifest)
        summary = evaluate_pairs(profile, pairs, models, output, deadline)
        summary["sampling"] = counts
        write_json(output / "summary.json", summary)
        if source_hashes() != manifest["source_hashes"]:
            raise RuntimeError("source changed during diagnostic")
        manifest.update(status="COMPLETED", phase="done", pairs=len(pairs),
                        elapsed_seconds=time.monotonic() - started,
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        artifact_hashes={name: sha(output / name) for name in
                                         ("sampling.json", "pairs.jsonl", "policy.csv", "summary.json")})
        write_json(output / "manifest.json", manifest)
        print(json.dumps(dict(status="COMPLETED", output_dir=str(output), pairs=len(pairs))))
    except Exception as exc:
        manifest.update(status="FAILED", phase="failed", error=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.monotonic() - started)
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "dev"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.profile, args.output_dir.resolve())


if __name__ == "__main__":
    main()
