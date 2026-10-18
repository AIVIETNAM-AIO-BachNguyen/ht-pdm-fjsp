"""Independent read-only verifier for teacher-transfer test artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition
from ht_pdm_fjsp.technician_teacher_transfer_env import VERSION, deadline, graph_observation, scaled_config
from ht_pdm_fjsp.technician_teacher_transfer_policy import ReservationPolicy


GRAPH_ARMS = ("teacher_reservation_ppo", "scratch_reservation_ppo")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def state_from(raw):
    return DispatchState(**{name: tuple(values) for name, values in json.loads(raw).items()})


def decode(config, state, method, submitted, tick):
    """Reimplement the coordination mapping without ResourceEnv.resolve."""
    obs = graph_observation(config, state, tick)
    mask = obs["pair_mask"]
    n, k = mask.shape
    if method == "deadline":
        assert submitted is None
        return deadline(config, state)
    token = np.asarray(submitted, dtype=int)
    if method in GRAPH_ARMS:
        assert token.shape == (k,)
        assert ((token >= 0) & (token <= n)).all()
        requests = [(int(m) - 1, j) for j, m in enumerate(token) if m]
        assert all(mask[m, j] for m, j in requests)
        assert len({m for m, _ in requests}) == len(requests)
        return tuple(sorted(requests))
    raise AssertionError(f"unknown method {method}")


def load_graph_model(root, choices, n, arm, seed):
    choice = choices[f"{n}:{arm}:{seed}"]
    path = root / choice["path"]
    assert digest(path) == choice["sha256"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["protocol"] == VERSION
    assert payload["algorithm"] == arm and payload["train_seed"] == seed
    assert payload["steps"] == choice["steps"] and payload["n"] == n
    model = ReservationPolicy(arm, payload["hidden"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def independently_resample(rows, config, coordination):
    """Recompute the registered crossed intervals from raw episode rows."""
    scales, train, tests = config["scales"], config["train_seeds"], config["test_seeds"]
    names = ("normalized_cost", "availability", "normalized_pending", "normalized_failures")
    arrays = {n: {arm: np.asarray([[[float(rows[n, arm, t, s][name]) for name in names]
                                    for s in tests] for t in train])
                  for arm in GRAPH_ARMS} for n in scales}
    rng = np.random.default_rng(role_seed(config["analysis_seed"], VERSION + ":independent_verifier_bootstrap"))
    samples = 40000 if config["profile"] == "confirmation" else 200
    draws = {"pooled_gain_teacher_vs_scratch": []}
    for n in scales:
        draws[f"N{n}_gain_teacher_vs_scratch"] = []
        for name in ("availability", "pending_per_machine", "failures_per_machine"):
            draws[f"N{n}_{name}_teacher_minus_scratch"] = []
    for start in tqdm(range(0, samples, 128), desc="Independent crossed bootstrap", unit="batch"):
        b = min(128, samples - start)
        t_indices = rng.integers(len(train), size=(b, len(train)))
        t_weights = (t_indices[:, :, None] == np.arange(len(train))).mean(1)
        summed_teacher = np.zeros(b)
        summed_scratch = np.zeros(b)
        for n in scales:
            e_indices = rng.integers(len(tests), size=(b, len(tests)))
            e_weights = (e_indices[:, :, None] == np.arange(len(tests))).mean(1)
            teacher = np.einsum("bt,bs,tsd->bd", t_weights, e_weights, arrays[n][GRAPH_ARMS[0]])
            scratch = np.einsum("bt,bs,tsd->bd", t_weights, e_weights, arrays[n][GRAPH_ARMS[1]])
            summed_teacher += teacher[:, 0]
            summed_scratch += scratch[:, 0]
            draws[f"N{n}_gain_teacher_vs_scratch"].extend((1 - teacher[:, 0] / scratch[:, 0]).tolist())
            for name, index in (("availability", 1), ("pending_per_machine", 2), ("failures_per_machine", 3)):
                draws[f"N{n}_{name}_teacher_minus_scratch"].extend((teacher[:, index] - scratch[:, index]).tolist())
        draws["pooled_gain_teacher_vs_scratch"].extend((1 - summed_teacher / summed_scratch).tolist())
    alpha = (.025 if config["profile"] == "confirmation" else .05) / 13
    bounds = {name: dict(lower=float(np.quantile(values, alpha)), upper=float(np.quantile(values, 1 - alpha)))
              for name, values in draws.items()}
    points = {name: float(np.mean(values)) for name, values in draws.items()}
    pooled_point = 1 - sum(arrays[n][GRAPH_ARMS[0]][:, :, 0].mean() for n in scales) / sum(
        arrays[n][GRAPH_ARMS[1]][:, :, 0].mean() for n in scales)
    gains=[1-arrays[n][GRAPH_ARMS[0]][:, :, 0].mean()/arrays[n][GRAPH_ARMS[1]][:, :, 0].mean()
           for n in scales]
    guards = {"pooled_teacher_gain": pooled_point >= .01 and bounds["pooled_gain_teacher_vs_scratch"]["lower"] > 0,
              "all_scales_point_noninferior": all(
                  gain>=-.01 for gain in gains),
              "two_scales_positive": sum(gain>0 for gain in gains)>=2}
    for n in scales:
        guards[f"N{n}_health"] = (bounds[f"N{n}_availability_teacher_minus_scratch"]["lower"] >= -.01 and
            bounds[f"N{n}_pending_per_machine_teacher_minus_scratch"]["upper"] <= .025 and
            bounds[f"N{n}_failures_per_machine_teacher_minus_scratch"]["upper"] <= .25)
    times = [float(r["decision_ms"]) for r in coordination if int(r["n"]) == max(scales)
             and r["method"] == GRAPH_ARMS[0]]
    guards['largest_select_latency'] = float(np.quantile(times,.95))<=50
    guards = {name: bool(value) for name, value in guards.items()}
    return dict(draws=samples,alpha_per_endpoint=alpha,bounds=bounds,
                bootstrap_draw_means=points,guards=guards,goal_pass=all(guards.values()))


def verify(root, output):
    started = time.monotonic()
    manifest = load_json(root / "manifest.json")
    assert manifest["status"] == "COMPLETED" and manifest["audit_status"] == "PASS"
    assert manifest["protocol"] == VERSION and manifest["test_opened"]
    for name, expected in manifest["source_seal"].items():
        assert digest(root / name) == expected, f"source seal: {name}"
    for name, metadata in tqdm(manifest["output_bindings"].items(), desc="Output bindings"):
        path = root / name
        assert path.stat().st_size == metadata["bytes"] and digest(path) == metadata["sha256"], name
    config = load_json(root / "resolved_config.json")
    summary = load_json(root / "summary.json")
    choices = load_json(root / "checkpoint_choices.json")
    episodes = load_rows(root / "episodes.csv")
    coordination = load_rows(root / "coordination.csv")
    decisions = load_rows(root / "decisions.csv")
    scales, train, tests = config["scales"], config["train_seeds"], config["test_seeds"]
    methods = {n: (*GRAPH_ARMS, "deadline") for n in scales}
    expected = {(n, arm, t, s) for n in scales for arm in methods[n]
                for t in ([-1] if arm == "deadline" else train) for s in tests}
    episode_key = lambda row: (int(row["n"]), row["method"], int(row["train_seed"]), int(row["seed"]))
    rows = {episode_key(row): row for row in episodes}
    assert len(episodes) == len(expected) and set(rows) == expected
    coord = {(episode_key(row), int(row["tick"])): row for row in coordination}
    assert len(coord) == 48 * len(rows)
    assert len(decisions) == 60 * len(rows)
    torch.set_num_threads(1)
    models = {}
    tapes = {}
    counts = dict(physical_ticks=0, matched_feasibility_states=0,
                  matched_feasibility_stages=0, teacher_trajectory_states=0,
                  scratch_trajectory_states=0, invalid_assignments=0)
    accumulated = {}
    for row in tqdm(decisions, desc="Independent physical and teacher/scratch replay", unit="tick"):
        key = episode_key(row)
        n, arm, train_seed, seed = key
        tick = int(row["tick"])
        config_n = scaled_config(n)
        if key not in tapes:
            env = WaitingEnv(config_n)
            physical_seed = role_seed(seed, f"{VERSION}:N{n}:physical")
            env.reset(physical_seed)
            tapes[key] = (env.state, env.events, 0)
            accumulated[key] = 0.0
        prior, events, next_tick = tapes[key]
        assert tick == next_tick and state_from(row["state"]) == prior
        event = tuple(json.loads(row["events"]))
        assert event == tuple(events[tick])
        pairs = tuple(tuple(x) for x in json.loads(row["action"]))
        if tick < 48:
            record = coord[key, tick]
            obs = graph_observation(config_n, prior, tick)
            assert hashlib.sha256(obs["state"].tobytes()).hexdigest() == record["observation_sha256"]
            submitted = json.loads(record["submitted"])
            assert decode(config_n, prior, arm, submitted, tick) == pairs
            if arm in GRAPH_ARMS:
                model_key = (n, train_seed)
                if model_key not in models:
                    models[model_key] = tuple(load_graph_model(root, choices, n, a, train_seed)
                                              for a in GRAPH_ARMS)
                teacher, scratch = models[model_key]
                batch = tensors({name: np.asarray(value)[None] for name, value in obs.items()})
                actions = torch.as_tensor(np.asarray(submitted)[None], dtype=torch.long)
                teacher_trace, scratch_trace = [], []
                with torch.no_grad():
                    teacher.trace(batch, recorded=actions, audit=teacher_trace)
                    scratch.trace(batch, recorded=actions, audit=scratch_trace)
                assert len(teacher_trace) == len(scratch_trace) == config_n.technicians
                reserved = set()
                for j, (a, b) in enumerate(zip(teacher_trace, scratch_trace, strict=True)):
                    reference = np.r_[True, [bool(obs["pair_mask"][m, j]) and m not in reserved for m in range(n)]]
                    assert np.array_equal(a["legal"].numpy()[0], reference)
                    for field in ("legal", "resource", "stats"):
                        assert torch.equal(a[field], b[field]), (key, tick, j, field)
                    counts["matched_feasibility_stages"] += 1
                    if submitted[j]:
                        reserved.add(submitted[j] - 1)
                counts["matched_feasibility_states"] += 1
                counts["teacher_trajectory_states"] += int(arm == GRAPH_ARMS[0])
                counts["scratch_trajectory_states"] += int(arm == GRAPH_ARMS[1])
        else:
            assert deadline(config_n, prior) == pairs
        following, cost, info = transition(config_n, prior, pairs, event)
        assert asdict(following) == {k: tuple(v) for k, v in json.loads(row["following"]).items()}
        assert abs(cost - float(row["cost"])) < 1e-8
        accumulated[key] += cost
        counts["physical_ticks"] += 1
        counts["invalid_assignments"] += info["invalid_assignments"]
        tapes[key] = (following, events, tick + 1)
    for key, total in accumulated.items():
        assert tapes[key][2] == 60
        assert abs(total - float(rows[key]["extended_cost"])) < 1e-7
    assert counts["invalid_assignments"] == 0
    expected_trajectory_states = len(train) * len(tests) * len(scales) * 48
    assert counts["teacher_trajectory_states"] == expected_trajectory_states
    assert counts["scratch_trajectory_states"] == expected_trajectory_states
    assert counts["matched_feasibility_states"] == 2 * expected_trajectory_states
    means = {}
    for n in scales:
        means[n] = {arm: np.mean([float(rows[n, arm, t, s]["normalized_cost"])
                                  for t in train for s in tests]) for arm in GRAPH_ARMS}
        estimate = 1 - means[n][GRAPH_ARMS[0]] / means[n][GRAPH_ARMS[1]]
        assert abs(estimate - summary["intervals"][f"N{n}_gain_teacher_vs_scratch"]["estimate"]) < 1e-12
    pooled = 1 - sum(means[n][GRAPH_ARMS[0]] for n in scales) / sum(means[n][GRAPH_ARMS[1]] for n in scales)
    assert abs(pooled - summary["intervals"]["pooled_gain_teacher_vs_scratch"]["estimate"]) < 1e-12
    inference = independently_resample(rows, config, coordination)
    if config["profile"] == "confirmation":
        assert inference["guards"] == summary["guards"], "independent inference gate disagreement"
    secondary = load_rows(root / "secondary_episodes.csv")
    secondary_report = load_json(root / "secondary_summary.json")
    secondary_expected = {(n, arm, t, s, condition)
        for n in scales for arm in GRAPH_ARMS for t in train for s in tests
        for condition in (("final", "zero_shot") if arm == GRAPH_ARMS[0] else ("final",))}
    secondary_key = lambda r: (int(r["n"]),r["method"],int(r["train_seed"]),
                               int(r["seed"]),r["condition"])
    assert len(secondary) == len(secondary_expected)
    assert {secondary_key(r) for r in secondary} == secondary_expected
    for row in secondary:
        n,arm,t,s,condition = secondary_key(row)
        assert row["population_sha256"] == rows[n,arm,t,s]["population_sha256"]
        assert int(float(row["invalid_assignments"])) == 0
        assert abs(float(row["normalized_cost"])-float(row["extended_cost"])/n) < 1e-10
    assert secondary_report["episodes"] == len(secondary)
    for n in scales:
        for arm in GRAPH_ARMS:
            for condition in (("final", "zero_shot") if arm == GRAPH_ARMS[0] else ("final",)):
                panel=[r for r in secondary if secondary_key(r)[0]==n and
                       r["method"]==arm and r["condition"]==condition]
                found=float(np.mean([float(r["normalized_cost"]) for r in panel]))
                reported=secondary_report["results"][str(n)][arm][condition]["means"]["normalized_cost"]
                assert abs(found-reported)<1e-12
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    report = dict(status="PASS", source_artifact=str(root), source_manifest_sha256=digest(root / "manifest.json"),
                  verifier_sha256=digest(output / Path(__file__).name), finished_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - started, counts=counts,
                  secondary_episodes=len(secondary),
                  means=means, pooled_gain_teacher_vs_scratch=pooled,
                  independent_inference=inference,
                  source_git=manifest["git"], source_summary_goal_pass=summary["goal_pass"])
    (output / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact.resolve(), args.output_dir.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
