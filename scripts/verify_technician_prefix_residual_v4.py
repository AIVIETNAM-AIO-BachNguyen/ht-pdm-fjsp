"""Independent read-only verifier for B4 direct-prefix test artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed
from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue
from ht_pdm_fjsp.maintenance_skill_contention_screen import quality, urgency
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition
from ht_pdm_fjsp.technician_prefix_residual_v4_env import VERSION, deadline, graph_observation, scaled_config
from ht_pdm_fjsp.technician_prefix_residual_v4_policy import ReservationPolicy


GRAPH_ARMS = ("reservation_ppo", "mask_only_ppo", "independent_ppo")


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
    if method == "ppo":
        assert isinstance(submitted, int)
        return catalogue(4, 2)[submitted]
    token = np.asarray(submitted, dtype=int)
    if method in GRAPH_ARMS:
        assert token.shape == (k,)
        assert ((token >= 0) & (token <= n)).all()
        requests = [(int(m) - 1, j) for j, m in enumerate(token) if m]
        assert all(mask[m, j] for m, j in requests)
        if method != "independent_ppo":
            assert len({m for m, _ in requests}) == len(requests)
        pairs = []
        for m in range(n):
            options = [p for p in requests if p[0] == m]
            if options:
                pairs.append(max(options, key=lambda p: (quality(config, *p), -p[1])))
        return tuple(sorted(pairs))
    if method == "mappo":
        assert token.shape == (n,)
        assert ((token >= 0) & (token <= k)).all()
        requests = [(m, int(j) - 1) for m, j in enumerate(token) if j]
        assert all(mask[m, j] for m, j in requests)
        pairs = []
        for j in range(k):
            options = [p for p in requests if p[1] == j]
            if options:
                pairs.append(max(options, key=lambda p: (urgency(config, state, p[0]), -p[0])))
        return tuple(sorted(pairs))
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
                  for arm in ("reservation_ppo", "mask_only_ppo")} for n in scales}
    rng = np.random.default_rng(role_seed(config["analysis_seed"], VERSION + ":independent_verifier_bootstrap"))
    samples = 40000 if config["profile"] == "confirmation" else 200
    draws = {"pooled_gain_full_vs_mask": []}
    for n in scales:
        draws[f"N{n}_gain_full_vs_mask"] = []
        for name in ("availability", "pending_per_machine", "failures_per_machine"):
            draws[f"N{n}_{name}_full_minus_mask"] = []
    for start in tqdm(range(0, samples, 128), desc="Independent crossed bootstrap", unit="batch"):
        b = min(128, samples - start)
        t_indices = rng.integers(len(train), size=(b, len(train)))
        t_weights = (t_indices[:, :, None] == np.arange(len(train))).mean(1)
        summed_full = np.zeros(b)
        summed_mask = np.zeros(b)
        for n in scales:
            e_indices = rng.integers(len(tests), size=(b, len(tests)))
            e_weights = (e_indices[:, :, None] == np.arange(len(tests))).mean(1)
            full = np.einsum("bt,bs,tsd->bd", t_weights, e_weights, arrays[n]["reservation_ppo"])
            mask = np.einsum("bt,bs,tsd->bd", t_weights, e_weights, arrays[n]["mask_only_ppo"])
            summed_full += full[:, 0]
            summed_mask += mask[:, 0]
            draws[f"N{n}_gain_full_vs_mask"].extend((1 - full[:, 0] / mask[:, 0]).tolist())
            for name, index in (("availability", 1), ("pending_per_machine", 2), ("failures_per_machine", 3)):
                draws[f"N{n}_{name}_full_minus_mask"].extend((full[:, index] - mask[:, index]).tolist())
        draws["pooled_gain_full_vs_mask"].extend((1 - summed_full / summed_mask).tolist())
    alpha = (.025 if config["profile"] == "confirmation" else .05) / 17
    bounds = {name: dict(lower=float(np.quantile(values, alpha)), upper=float(np.quantile(values, 1 - alpha)))
              for name, values in draws.items()}
    points = {name: float(np.mean(values)) for name, values in draws.items()}
    pooled_point = 1 - sum(arrays[n]["reservation_ppo"][:, :, 0].mean() for n in scales) / sum(
        arrays[n]["mask_only_ppo"][:, :, 0].mean() for n in scales)
    guards = {"pooled_prefix_gain": pooled_point >= .01 and bounds["pooled_gain_full_vs_mask"]["lower"] > 0,
              "all_scales_point_noninferior": all(
                  1 - arrays[n]["reservation_ppo"][:, :, 0].mean() / arrays[n]["mask_only_ppo"][:, :, 0].mean() >= -.01
                  for n in scales)}
    for n in scales:
        guards[f"N{n}_health"] = (bounds[f"N{n}_availability_full_minus_mask"]["lower"] >= -.01 and
            bounds[f"N{n}_pending_per_machine_full_minus_mask"]["upper"] <= .025 and
            bounds[f"N{n}_failures_per_machine_full_minus_mask"]["upper"] <= .25)
        times = [float(r["decision_ms"]) for r in coordination if int(r["n"]) == n and r["method"] == "reservation_ppo"]
        guards[f"N{n}_latency"] = float(np.quantile(times, .95)) <= 50
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
    methods = {n: (*GRAPH_ARMS, "ppo", "mappo", "deadline") if n == 4 else (*GRAPH_ARMS, "deadline") for n in scales}
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
    counts = dict(physical_ticks=0, prefix_states=0, prefix_stages=0,
                  nonzero_full_prefix_stages=0, nonzero_full_residual_stages=0,
                  same_state_residual_choice_changes=0, invalid_assignments=0)
    accumulated = {}
    for row in tqdm(decisions, desc="Independent physical and prefix replay", unit="tick"):
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
            if arm == "reservation_ppo":
                model_key = (n, train_seed)
                if model_key not in models:
                    models[model_key] = tuple(load_graph_model(root, choices, n, a, train_seed)
                                              for a in ("reservation_ppo", "mask_only_ppo"))
                full, masked = models[model_key]
                batch = tensors({name: np.asarray(value)[None] for name, value in obs.items()})
                actions = torch.as_tensor(np.asarray(submitted)[None], dtype=torch.long)
                full_trace, masked_trace = [], []
                with torch.no_grad():
                    full.trace(batch, recorded=actions, audit=full_trace)
                    masked.trace(batch, recorded=actions, audit=masked_trace)
                assert len(full_trace) == len(masked_trace) == config_n.technicians
                edge, _, _, _ = full.encode(batch)
                reserved = set()
                for j, (a, b) in enumerate(zip(full_trace, masked_trace, strict=True)):
                    reference = np.r_[True, [bool(obs["pair_mask"][m, j]) and m not in reserved for m in range(n)]]
                    assert np.array_equal(a["legal"].numpy()[0], reference)
                    for field in ("legal", "resource", "stats"):
                        assert torch.equal(a[field], b[field]), (key, tick, j, field)
                    assert torch.count_nonzero(b["prefix"]) == 0
                    assert torch.count_nonzero(b["candidate_capacity"]) == 0
                    assert torch.count_nonzero(b["wait_capacity"]) == 0
                    assert torch.count_nonzero(b["prefix_residual"]) == 0
                    expected_residual = (
                        full.query(edge[:, :, j]) * full.key(a["prefix"])
                    ).sum(-1) / math.sqrt(full.hidden)
                    assert torch.allclose(a["prefix_residual"][:, 1:], expected_residual, atol=1e-6)
                    assert torch.allclose(a["full_logits"] - a["base_logits"],
                                          a["prefix_residual"], atol=1e-6, rtol=0)
                    legal = a["legal"]
                    full_choice = a["full_logits"].masked_fill(~legal, -1e9).argmax(-1)
                    base_choice = a["base_logits"].masked_fill(~legal, -1e9).argmax(-1)
                    counts["nonzero_full_residual_stages"] += int(torch.count_nonzero(a["prefix_residual"]) > 0)
                    counts["same_state_residual_choice_changes"] += int(torch.any(full_choice != base_choice))
                    previous = [(submitted[h] - 1, h) for h in range(j) if submitted[h]]
                    busy = sum(float(obs["edges"][m, h, 1]) * 3 for m, h in previous) / config_n.technicians
                    weak = sum(float(obs["edges"][m, h, 1]) * 3 * (1 - float(obs["edges"][m, h, 2]))
                               for m, h in previous) / config_n.technicians
                    expected_wait = torch.tensor([[busy, weak]])
                    expected_candidate = torch.as_tensor(obs["edges"][:, j, 1] * 3)[:, None] * expected_wait[0]
                    assert torch.allclose(a["wait_capacity"], expected_wait, atol=1e-6)
                    assert torch.allclose(a["candidate_capacity"][0], expected_candidate, atol=1e-6)
                    counts["nonzero_full_prefix_stages"] += int(torch.count_nonzero(a["prefix"]) > 0)
                    counts["prefix_stages"] += 1
                    if submitted[j]:
                        reserved.add(submitted[j] - 1)
                counts["prefix_states"] += 1
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
    assert counts["prefix_states"] == len(train) * len(tests) * len(scales) * 48
    means = {}
    for n in scales:
        means[n] = {arm: np.mean([float(rows[n, arm, t, s]["normalized_cost"])
                                  for t in train for s in tests]) for arm in GRAPH_ARMS}
        estimate = 1 - means[n]["reservation_ppo"] / means[n]["mask_only_ppo"]
        assert abs(estimate - summary["intervals"][f"N{n}_gain_full_vs_mask"]["estimate"]) < 1e-12
    pooled = 1 - sum(means[n]["reservation_ppo"] for n in scales) / sum(means[n]["mask_only_ppo"] for n in scales)
    assert abs(pooled - summary["intervals"]["pooled_gain_full_vs_mask"]["estimate"]) < 1e-12
    inference = independently_resample(rows, config, coordination)
    if config["profile"] == "confirmation":
        assert inference["guards"] == summary["guards"], "independent inference gate disagreement"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    report = dict(status="PASS", source_artifact=str(root), source_manifest_sha256=digest(root / "manifest.json"),
                  verifier_sha256=digest(output / Path(__file__).name), finished_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - started, counts=counts,
                  means=means, pooled_gain_full_vs_mask=pooled,
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
