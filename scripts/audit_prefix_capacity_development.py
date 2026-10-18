"""Independent read-only recomputation of the selected-pair capacity prefix DEV gate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

from tqdm.auto import tqdm


ARMS = ("reservation_ppo", "mask_only_ppo", "independent_ppo")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit(source, output):
    started = time.monotonic()
    manifest = json.loads((source / "manifest.json").read_text())
    config = json.loads((source / "resolved_config.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    original_audit = json.loads((source / "audit.json").read_text())
    choices = json.loads((source / "checkpoint_choices.json").read_text())
    assert manifest['protocol'] == 'technician_prefix_capacity_v3'
    assert manifest["status"] == "COMPLETED" and manifest["audit_status"] == "PASS"
    assert manifest["profile"] == "development" and not manifest["test_opened"]
    assert original_audit["status"] == "PASS" and not config["test_seeds"]
    for name, digest in tqdm(manifest["source_seal"].items(), desc="Source seals"):
        assert sha(source / name) == digest, name
    for name, metadata in tqdm(manifest["output_bindings"].items(), desc="Output bindings"):
        path = source / name
        assert path.stat().st_size == metadata["bytes"] and sha(path) == metadata["sha256"], name
    data = rows(source / "development_episodes.csv")
    train = config["train_seeds"]
    scales = config["scales"]
    seeds = config["development_seeds"]
    checkpoints = [0, *config["checkpoints"]]
    expected = {(n, arm, t, step, seed) for n in scales
                for arm in ((*ARMS, "ppo", "mappo") if n == 4 else ARMS)
                for t in train for step in (checkpoints if arm in ARMS else [491520])
                for seed in seeds}
    key = lambda r: (int(r["n"]), r["method"], int(r["train_seed"]),
                     int(r["checkpoint_steps"]), int(r["seed"]))
    table = {key(r): r for r in data}
    assert len(data) == len(expected) and set(table) == expected
    populations = {}
    for identity, row in table.items():
        n, arm, t, step, seed = identity
        label = n, seed
        assert populations.setdefault(label, row["population_sha256"]) == row["population_sha256"]
        assert float(row["invalid_assignments"]) == 0
    chosen = {}
    means = {}
    replica = {}
    health = {}
    for n in scales:
        means[n] = {}
        replica[n] = {}
        health[n] = {}
        for arm in ((*ARMS, "ppo", "mappo") if n == 4 else ARMS):
            costs = []
            replica[n][arm] = {}
            selected_rows = []
            for t in train:
                options = checkpoints if arm in ARMS else [491520]
                by_step = {step: statistics.fmean(float(table[n, arm, t, step, seed]["extended_cost"])
                                                   for seed in seeds) for step in options}
                step = min(by_step, key=lambda s: (by_step[s], s))
                assert choices[f"{n}:{arm}:{t}"]["steps"] == step
                chosen[f"{n}:{arm}:{t}"] = step
                costs.append(by_step[step])
                replica[n][arm][t] = dict(step=step,cost=by_step[step],
                    cost_per_machine=by_step[step] / n)
                selected_rows.extend(table[n, arm, t, step, seed] for seed in seeds)
            means[n][arm] = statistics.fmean(costs)
            health[n][arm] = {name: statistics.fmean(float(r[name]) for r in selected_rows)
                for name in ("availability", "normalized_pending", "normalized_failures", "invalid_assignments")}
            assert abs(means[n][arm] - summary["means"][str(n)][arm]) < 1e-9
    gains = {n: 1 - means[n]["reservation_ppo"] / means[n]["mask_only_ppo"] for n in scales}
    pooled = 1 - sum(means[n]["reservation_ppo"] / n for n in scales) / sum(
        means[n]["mask_only_ppo"] / n for n in scales)
    assert abs(pooled - summary["pooled_gain_full_vs_mask"]) < 1e-12
    for n in scales:
        assert abs(gains[n] - summary["scale_gains"][str(n)]) < 1e-12
    health_guards = {n: dict(
        availability=health[n]["reservation_ppo"]["availability"] >= health[n]["mask_only_ppo"]["availability"] - .01,
        pending=health[n]["reservation_ppo"]["normalized_pending"] <= health[n]["mask_only_ppo"]["normalized_pending"] + .025,
        failures=health[n]["reservation_ppo"]["normalized_failures"] <= health[n]["mask_only_ppo"]["normalized_failures"] + .25,
        invalid=health[n]["reservation_ppo"]["invalid_assignments"] == 0 and health[n]["mask_only_ppo"]["invalid_assignments"] == 0)
        for n in scales}
    passed = pooled >= .005 and all(v >= -.01 for v in gains.values()) and all(
        all(bound.values()) for bound in health_guards.values())
    assert passed == summary["development_screen_pass"]
    training = json.loads((source / "training_audit.json").read_text())
    assert len(training) == len(choices)
    for model in training:
        if model["reference"]:
            assert model["learned_steps"] == 0 and model["foundation_training_steps"] == 491520
        else:
            assert model["learned_steps"] == config["env_steps"]
            assert model["warm_state_exposures"] == config["teacher_bank_steps"]
            assert model['training_episodes']==config['env_steps']//48
            assert model['optimizer_steps']>0 and model['changed']
    initials=json.loads((source/'initial_parameter_hashes.json').read_text())
    for n in scales:
        for t in train:
            assert len({initials[f'{n}:{arm}:{t}'] for arm in ARMS})==1
    training_grid=set();tapes={}
    with (source/'training_episodes.csv').open(newline='') as stream:
        for row in tqdm(csv.DictReader(stream),desc='Paired PPO training tapes',unit='episode'):
            n,arm,t,w,e=(int(row['n']),row['algorithm'],int(row['train_seed']),
                         int(row['worker']),int(row['episode_index']))
            if arm not in ARMS:continue
            key=(n,t,w,e);identity=key+(arm,)
            assert identity not in training_grid
            training_grid.add(identity)
            tape=row['physical_seed'],row['population_sha256']
            if key in tapes:assert tapes[key]==tape
            else:tapes[key]=tape
            assert abs(float(row['episode_reward'])*20+float(row['extended_cost']))<1e-5
            assert float(row['invalid_assignments'])==0
    expected_training={(n,t,w,e,arm) for n in scales for t in train
        for w in range(config['vector_envs'])
        for e in range(config['env_steps']//(48*config['vector_envs'])) for arm in ARMS}
    assert training_grid==expected_training
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),output / Path(__file__).name)
    report = dict(status="PASS",verdict="DEV_GATE_PASS" if passed else "DEV_GATE_FAIL",
        source_artifact=str(source),source_git=manifest["git"],source_manifest_sha256=sha(source / "manifest.json"),
        verifier_sha256=sha(output / Path(__file__).name),finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started,episodes=len(data),populations=len(populations),
        training_models=len(training),training_episodes=len(training_grid),
        paired_training_tapes=len(tapes),chosen_steps=chosen,means=means,replica=replica,
        scale_gains=gains,pooled_gain=pooled,health_guards=health_guards,
        test_opened=False)
    (output / "audit.json").write_text(json.dumps(report,indent=2,sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.artifact.resolve(),args.output_dir.resolve()),sort_keys=True))


if __name__ == "__main__":
    main()
