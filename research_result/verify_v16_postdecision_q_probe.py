"""Independent all-row feature, checkpoint-choice, saved-Q and CI audit for V16."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, catalogue, configs, vector_observation
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings, state_from_dict, submit_requests

ROOT = Path(__file__).resolve().parent.parent
ACTIONS = catalogue()
SEEDS = tuple(range(72600000, 72600005))
STARTS = (72950000, 72950001, 72950002)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def same(a, b):
    assert np.isclose(float(a), float(b), rtol=0, atol=1e-7), (a, b)


class IndependentHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(74, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def independent_features(config, state, action, obs):
    incidence = [float((m, j) in action) for m in range(config.machines)
                 for j in range(config.technicians)]
    served = [float(any(m == pair[0] for pair in action)) for m in range(config.machines)]
    before = [int(t) for t in state.physical.remaining]
    after = before[:]
    for machine, worker in action:
        assert after[worker] == 0 and config.service_time[machine][worker] > 0
        after[worker] = config.service_time[machine][worker]

    def release(worker_times):
        earliest = []
        for machine in range(config.machines):
            options = [worker_times[j] for j in range(config.technicians)
                       if config.service_time[machine][j] > 0]
            earliest.append(max(1, min(options)))
        return np.asarray([*worker_times, *earliest], dtype=np.float32) / 12

    prefix = np.concatenate((obs, np.asarray(incidence + served, np.float32)))
    return np.concatenate((prefix, release(after))), np.concatenate((prefix, release(before)))


def confidence(cube):
    random = np.random.default_rng(72950003)
    samples = np.empty(20000)
    for offset in tqdm(range(0, 20000, 1000), desc="Independent V16 CI", unit="chunk"):
        selected = random.integers(0, 5, size=(1000, 5))
        ranks = random.integers(0, 64, size=(1000, 5, 6, 4, 64))
        samples[offset:offset + 1000] = np.take_along_axis(cube[selected], ranks, axis=-1).mean((1, 2, 3, 4))
    return np.quantile(samples, [.025, .975])


def audit(source, probe, output):
    started = time.monotonic()
    deadline = started + 1800
    manifest = json.loads((probe / "manifest.json").read_text())
    profile = manifest["profile"]
    assert profile in ("smoke", "dev") and manifest["status"] == "COMPLETED"
    source_manifest = json.loads((source / "manifest.json").read_text())
    assert source_manifest["status"] == "COMPLETED"
    assert source_manifest["profile"] == profile
    assert Path(manifest["source_run"]).resolve() == source
    assert digest(source / "manifest.json") == manifest["source_manifest_sha256"]
    assert digest(source / "counterfactual_states.csv") == manifest["source_q_sha256"]
    for name, expected in source_manifest["source_hashes"].items():
        assert digest(ROOT / name) == expected
    for name, expected in source_manifest["artifact_hashes"].items():
        assert digest(source / name) == expected
    for name, expected in manifest["artifact_hashes"].items():
        assert digest(probe / name) == expected
    assert digest(ROOT / "research_result/v16_postdecision_q_probe_protocol_vi.md") == manifest["protocol_sha256"]
    assert digest(ROOT / "research_result/v16_postdecision_release_value_proposal_vi.md") == manifest["proposal_sha256"]
    assert digest(ROOT / "research_result/run_v16_postdecision_q_probe.py") == manifest["script_sha256"]
    if profile == "dev":
        main = Path(manifest["source_audit"])
        assert digest(main / "independent_audit.json") == manifest["source_audit_sha256"]
        report = json.loads((main / "independent_audit.json").read_text())
        assert report["status"] == "PASS" and Path(report["run"]).resolve() == source
        seeds, starts, expected_rows = SEEDS, STARTS, 7680 * 3
    else:
        assert manifest["source_audit"] is None
        seeds, starts, expected_rows = (72620000,), STARTS[:1], 4
    raw = [r for r in records(source / "counterfactual_states.csv") if r["arm"] == "count_calibrated"]
    key = lambda r: (int(r["train_seed"]), int(r["iteration"]), r["case"], r["rank"])
    original = {key(r): r for r in raw}
    assert len(original) == len(raw) == (7680 if profile == "dev" else 12)
    rows = records(probe / "heldout_rows.csv")
    assert len(rows) == expected_rows
    counts = {k: 0 for k in original}
    models = {}
    torch.set_num_threads(1)
    assert len(manifest["checkpoint_hashes"]) == len(seeds) * len(starts) * 2
    for name, expected in manifest["checkpoint_hashes"].items():
        cp = probe / name
        assert digest(cp) == expected
        tokens = Path(name).parts
        assert len(tokens) == 4 and tokens[0] == "checkpoints"
        fold, init, arm = int(tokens[1]), int(tokens[2]), Path(tokens[3]).stem
        assert fold in seeds and init in starts and arm in ("post", "pre")
        model = IndependentHead()
        model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
        assert sum(p.numel() for p in model.parameters()) == 9025
        model.eval()
        models[(fold, init, arm)] = model
    trace = records(probe / "train_trace.csv")
    assert len(trace) == len(models) * 6
    assert len({(int(r["fold_seed"]), int(r["init_seed"]), r["arm"], int(r["epoch"])) for r in trace}) == len(trace)
    assert all(np.isfinite(float(r["loss"])) for r in trace)
    observed = set()
    configs_by_case = configs()
    for row in tqdm(rows, desc="Independent V16 root/action replay", unit="row"):
        if time.monotonic() > deadline:
            raise TimeoutError("V16 independent audit")
        root_key = key(row)
        assert root_key in original and int(row["init_seed"]) in starts
        unique = (root_key, int(row["init_seed"]))
        assert unique not in observed
        observed.add(unique)
        counts[root_key] += 1
        raw_row = original[root_key]
        config = configs_by_case[row["case"]]
        state = state_from_dict(json.loads(raw_row["original_state"]))
        tick = int(raw_row["tick"])
        admitted = submit_requests(config, state, tick,
                                   request_action(config, state, tick, "threshold"))
        legal = sorted(ACTIONS.index(a) for a in legal_matchings(config, admitted))
        assert legal == json.loads(raw_row["legal"]) == json.loads(row["legal"])
        costs = {int(a): float(q) for a, q in json.loads(raw_row["scores"]).items()}
        assert costs == {int(a): float(q) for a, q in json.loads(row["q"]).items()}
        assert set(legal) == set(costs)
        assert int(row["source_action"]) == int(raw_row["source_action"])
        same(row["source_q"], costs[int(row["source_action"])])
        same(row["best_q"], min(costs.values()))
        assert int(row["physical_seed"]) == int(raw_row["physical_seed"])
        assert int(row["tick"]) == tick
        assert row["service_legal"] == str(any(ACTIONS[a] for a in legal))
        obs = vector_observation(config, admitted, tick)
        arrays = {"post": [], "pre": []}
        for action in legal:
            post, pre = independent_features(config, admitted, ACTIONS[action], obs)
            arrays["post"].append(post)
            arrays["pre"].append(pre)
        for arm in ("post", "pre"):
            model = models[(int(row["train_seed"]), int(row["init_seed"]), arm)]
            with torch.no_grad():
                scores = model(torch.from_numpy(np.asarray(arrays[arm])).reshape(1, len(legal), 74))
            chosen = legal[int(np.argmin(scores.numpy()[0]))]
            assert chosen == int(row[f"{arm}_action"]) and chosen in legal
            same(row[f"{arm}_q"], costs[chosen])
        same(row["delta"], float(row["post_q"]) - float(row["pre_q"]))
    if profile == "dev":
        assert all(v == 3 for v in counts.values())
    else:
        assert sum(v == 1 for v in counts.values()) == 4
    summary = json.loads((probe / "summary.json").read_text())
    assert summary["rows"] == len(rows) and summary["parameter_count"] == 9025
    delta = np.array([float(r["delta"]) for r in rows])
    same(summary["mean_delta"], delta.mean())
    for arm in ("post", "pre", "source"):
        same(summary[f"{arm}_q"], np.mean([float(r[f"{arm}_q"]) for r in rows]))
    assert summary["changed"] == sum(r["post_action"] != r["pre_action"] for r in rows)
    service = [r for r in rows if r["service_legal"] == "True"]
    post_wait = np.mean([int(r["post_action"]) == 0 for r in service])
    source_wait = np.mean([int(r["source_action"]) == 0 for r in service])
    same(summary["post_wait_rate"], post_wait)
    same(summary["source_wait_rate"], source_wait)
    if profile == "dev":
        cube = np.empty((5, 6, 4, 64))
        for si, seed in enumerate(SEEDS):
            for u in range(1, 7):
                for ci, case in enumerate(CASES):
                    group = [r for r in rows if int(r["train_seed"]) == seed
                             and int(r["iteration"]) == u and r["case"] == case]
                    assert len(group) == 192
                    ranks = sorted({r["rank"] for r in group})
                    assert len(ranks) == 64
                    cube[si, u - 1, ci] = [np.mean([float(r["delta"]) for r in group
                                                     if r["rank"] == rank]) for rank in ranks]
        interval = confidence(cube)
        for got, expected in zip(summary["ci95"], interval):
            same(got, expected)
        for si, seed in enumerate(SEEDS):
            same(summary["by_seed"][str(seed)], cube[si].mean())
        gate = bool(cube.mean() <= -1 and sum(cube.mean(axis=(1, 2, 3)) < 0) >= 4
                    and interval[1] < 0 and summary["changed_fraction"] >= .01
                    and summary["post_q"] <= summary["source_q"] - 1
                    and post_wait <= source_wait + .02 and post_wait <= .10)
        assert summary["diagnostic_gate"] == gate
    else:
        assert summary["ci95"] is None and summary["diagnostic_gate"] is False
    report = dict(status="PASS", source=str(source), probe=str(probe), profile=profile,
                  checked_rows=len(rows), checked_checkpoints=len(models),
                  independently_retrained=False, mean_delta=float(delta.mean()),
                  wall_seconds=time.monotonic() - started)
    (output / "independent_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    audit(args.source_dir.resolve(), args.probe_dir.resolve(), output)


if __name__ == "__main__":
    main()
