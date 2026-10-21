"""Held-out V16 post/pre-decision release representation screen on audited V14 Q."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, catalogue, configs, vector_observation
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings, state_from_dict, submit_requests

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "research_result/v16_postdecision_q_probe_protocol_vi.md"
PROPOSAL = ROOT / "research_result/v16_postdecision_release_value_proposal_vi.md"
ACTIONS = catalogue()
SEEDS = tuple(range(72600000, 72600005))
STARTS = (72950000, 72950001, 72950002)
BOOTSTRAP_SEED = 72950003
EPOCHS = 6
DIM = 74
WAIT = ACTIONS.index(())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def features(cfg, state, action):
    """Two matched 74-D action inputs, using only admitted current state."""
    pre = list(state.physical.remaining)
    post = pre.copy()
    incidence = np.zeros((cfg.machines, cfg.technicians), np.float32)
    started = np.zeros(cfg.machines, np.float32)
    for machine, worker in action:
        if pre[worker] != 0 or cfg.service_time[machine][worker] <= 0:
            raise ValueError("illegal service commitment")
        incidence[machine, worker] = 1
        started[machine] = 1
        post[worker] = cfg.service_time[machine][worker]

    def release(vector):
        earliest = [max(1, min(vector[j] for j in range(cfg.technicians)
                               if cfg.service_time[m][j] > 0))
                    for m in range(cfg.machines)]
        return np.asarray([*vector, *earliest], np.float32) / 12

    common = np.concatenate((incidence.ravel(), started))
    return common, release(post), release(pre)


class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIM, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def bank(run, profile, deadline):
    source_manifest = json.loads((run / "manifest.json").read_text())
    assert source_manifest["status"] == "COMPLETED"
    assert source_manifest["profile"] == profile
    for name, digest in source_manifest["source_hashes"].items():
        assert sha(ROOT / name) == digest
    for name, digest in source_manifest["artifact_hashes"].items():
        assert sha(run / name) == digest
    raw = [r for r in read_csv(run / "counterfactual_states.csv")
           if r["arm"] == "count_calibrated"]
    if profile == "dev":
        seeds, updates, per_cell = SEEDS, 6, 64
    else:
        seeds, updates, per_cell = (72620000,), 1, 3
    assert len(raw) == len(seeds) * updates * len(CASES) * per_cell
    groups = {}
    for row in raw:
        groups.setdefault((int(row["train_seed"]), int(row["iteration"]), row["case"]), []).append(row)
    expected = {(s, u, c) for s in seeds for u in range(1, updates + 1) for c in CASES}
    assert set(groups) == expected
    for group in groups.values():
        group.sort(key=lambda r: r["rank"])
        assert len(group) == per_cell and len({r["rank"] for r in group}) == per_cell

    ordered = [r for s in seeds for u in range(1, updates + 1) for c in CASES
               for r in groups[(s, u, c)]]
    n = len(ordered)
    x = {arm: np.empty((n, len(ACTIONS), DIM), np.float32) for arm in ("post", "pre")}
    q = np.zeros((n, len(ACTIONS)), np.float32)
    mask = np.zeros((n, len(ACTIONS)), bool)
    source_q = np.empty(n, np.float64)
    metadata = []
    config_map = configs()
    for i, row in enumerate(tqdm(ordered, desc="V16 reconstruct Q roots", unit="state")):
        if time.monotonic() > deadline:
            raise TimeoutError("V16 bank reconstruction")
        cfg = config_map[row["case"]]
        original = state_from_dict(json.loads(row["original_state"]))
        tick = int(row["tick"])
        admitted = submit_requests(cfg, original, tick,
                                   request_action(cfg, original, tick, "threshold"))
        legal = sorted(ACTIONS.index(a) for a in legal_matchings(cfg, admitted))
        assert legal == json.loads(row["legal"]) and len(legal) > 1
        costs = {int(k): float(v) for k, v in json.loads(row["scores"]).items()}
        assert set(costs) == set(legal) and all(np.isfinite(v) for v in costs.values())
        source = int(row["source_action"])
        assert source in costs
        obs = vector_observation(cfg, admitted, tick)
        for action in legal:
            common, post, pre = features(cfg, admitted, ACTIONS[action])
            assert len(common) == 12 and len(post) == len(pre) == 6
            x["post"][i, action] = np.concatenate((obs, common, post))
            x["pre"][i, action] = np.concatenate((obs, common, pre))
            q[i, action] = costs[action]
            mask[i, action] = True
        x["post"][i, ~mask[i]] = 0
        x["pre"][i, ~mask[i]] = 0
        source_q[i] = costs[source]
        metadata.append(dict(train_seed=int(row["train_seed"]), iteration=int(row["iteration"]),
                             case=row["case"], rank=row["rank"], physical_seed=int(row["physical_seed"]),
                             tick=tick, source_action=source, source_q=costs[source],
                             legal=json.dumps(legal), q=json.dumps(costs, sort_keys=True)))
    assert all(np.isfinite(value).all() for value in x.values())
    return x, q, mask, source_q, metadata


def fit(x, q, mask, train_index, start_seed, deadline):
    torch.manual_seed(start_seed)
    model = ValueHead()
    assert sum(p.numel() for p in model.parameters()) == 9025
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    xx = torch.from_numpy(x)
    qq = torch.from_numpy(q)
    mm = torch.from_numpy(mask)
    losses = []
    for epoch in range(EPOCHS):
        rng = np.random.default_rng(start_seed + epoch)
        order = rng.permutation(train_index)
        epoch_loss = []
        for left in range(0, len(order), 256):
            if time.monotonic() > deadline:
                raise TimeoutError("V16 network fit")
            idx = order[left:left + 256]
            target = torch.softmax((-qq[idx] / 10).masked_fill(~mm[idx], -1e9), dim=-1)
            logits = -model(xx[idx]).masked_fill(~mm[idx], 1e9)
            loss = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_loss)))
    model.eval()
    return model, losses


def predict(model, x, mask, indices):
    results = []
    times = []
    with torch.no_grad():
        for left in range(0, len(indices), 256):
            idx = indices[left:left + 256]
            begin = time.perf_counter()
            scores = model(torch.from_numpy(x[idx])).numpy()
            scores[~mask[idx]] = np.inf
            assert np.isfinite(scores[mask[idx]]).all()
            results.extend(np.argmin(scores, axis=1).tolist())
            times.extend([((time.perf_counter() - begin) / len(idx))] * len(idx))
    return np.asarray(results, np.int64), times


def bootstrap(cube, draws):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result = np.empty(draws)
    ns, nu, nc, nr = cube.shape
    for left in tqdm(range(0, draws, 1000), desc="V16 hierarchical bootstrap", unit="chunk"):
        count = min(1000, draws - left)
        seeds = rng.integers(0, ns, size=(count, ns))
        ranks = rng.integers(0, nr, size=(count, ns, nu, nc, nr))
        result[left:left + count] = np.take_along_axis(cube[seeds], ranks, axis=-1).mean(axis=(1, 2, 3, 4))
    return [float(v) for v in np.quantile(result, [.025, .975])]


def run_probe(run, out, profile, audit):
    start = time.monotonic()
    deadline = start + (1800 if profile == "dev" else 300)
    if profile == "dev":
        report = json.loads((audit / "independent_audit.json").read_text())
        assert report["status"] == "PASS" and Path(report["run"]).resolve() == run
    else:
        assert audit is None
    torch.set_num_threads(1)
    x, q, mask, source_q, meta = bank(run, profile, deadline)
    seeds = SEEDS if profile == "dev" else (72620000,)
    rows = []
    trace = []
    checkpoints = {}
    for fold, seed in enumerate(tqdm(seeds, desc="V16 held-out folds", unit="fold")):
        held = np.flatnonzero(np.asarray([m["train_seed"] == seed for m in meta]))
        train = np.flatnonzero(np.asarray([m["train_seed"] != seed for m in meta]))
        if profile == "smoke":
            train, held = held[:8], held[8:]
        assert len(held) == (1536 if profile == "dev" else 4)
        for init in (STARTS if profile == "dev" else STARTS[:1]):
            selected, elapsed = {}, {}
            for arm in ("post", "pre"):
                model, losses = fit(x[arm], q, mask, train, init, deadline)
                choices, times = predict(model, x[arm], mask, held)
                selected[arm] = choices
                elapsed[arm] = times
                path = Path("checkpoints") / str(seed) / str(init) / f"{arm}.pt"
                (out / path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out / path)
                checkpoints[str(path)] = sha(out / path)
                for epoch, loss in enumerate(losses, 1):
                    trace.append(dict(fold_seed=seed, init_seed=init, arm=arm, epoch=epoch, loss=loss))
            for local, index in enumerate(held):
                row = meta[index].copy()
                post, pre = (int(selected[arm][local]) for arm in ("post", "pre"))
                assert mask[index, post] and mask[index, pre]
                row.update(init_seed=init, post_action=post, pre_action=pre,
                           post_q=float(q[index, post]), pre_q=float(q[index, pre]),
                           delta=float(q[index, post] - q[index, pre]),
                           best_q=float(q[index, mask[index]].min()),
                           service_legal=bool(any(mask[index, a] and bool(ACTIONS[a])
                                                  for a in range(len(ACTIONS)))),
                           post_seconds=elapsed["post"][local], pre_seconds=elapsed["pre"][local])
                rows.append(row)
    write_csv(out / "heldout_rows.csv", rows)
    write_csv(out / "train_trace.csv", trace)
    result = np.asarray([r["delta"] for r in rows], np.float64)
    if profile == "dev":
        cube = np.empty((5, 6, len(CASES), 64), np.float64)
        for si, seed in enumerate(SEEDS):
            for ui in range(1, 7):
                for ci, case in enumerate(CASES):
                    cell = [r for r in rows if r["train_seed"] == seed and r["iteration"] == ui
                            and r["case"] == case]
                    assert len(cell) == 64 * len(STARTS)
                    by_rank = {}
                    for row in cell:
                        by_rank.setdefault(row["rank"], []).append(row["delta"])
                    assert len(by_rank) == 64 and all(len(v) == len(STARTS) for v in by_rank.values())
                    cube[si, ui - 1, ci] = [np.mean(by_rank[k]) for k in sorted(by_rank)]
        ci = bootstrap(cube, 20000)
    else:
        cube, ci = None, None
    changed = sum(r["post_action"] != r["pre_action"] for r in rows)
    service = [r for r in rows if r["service_legal"]]
    wait_post = np.mean([r["post_action"] == WAIT for r in service])
    wait_source = np.mean([r["source_action"] == WAIT for r in service])
    by_seed = {str(seed):float(np.mean([r["delta"] for r in rows if r["train_seed"] == seed]))
               for seed in seeds}
    summary = dict(status="COMPLETED", profile=profile, rows=len(rows), starts=STARTS if profile == "dev" else STARTS[:1],
                   parameter_count=9025, epochs=EPOCHS, mean_delta=float(result.mean()), ci95=ci,
                   by_seed=by_seed,
                   by_case={case:float(np.mean([r["delta"] for r in rows if r["case"] == case])) for case in CASES
                            if any(r["case"] == case for r in rows)},
                   by_start={str(init):float(np.mean([r["delta"] for r in rows if r["init_seed"] == init]))
                             for init in (STARTS if profile == "dev" else STARTS[:1])},
                   post_q=float(np.mean([r["post_q"] for r in rows])),
                   pre_q=float(np.mean([r["pre_q"] for r in rows])),
                   source_q=float(np.mean([r["source_q"] for r in rows])),
                   post_regret=float(np.mean([r["post_q"] - r["best_q"] for r in rows])),
                   pre_regret=float(np.mean([r["pre_q"] - r["best_q"] for r in rows])),
                   changed=changed, changed_fraction=changed / len(rows),
                   service_rows=len(service), post_wait_rate=float(wait_post), source_wait_rate=float(wait_source),
                   post_inference_seconds=float(np.mean([r["post_seconds"] for r in rows])),
                   pre_inference_seconds=float(np.mean([r["pre_seconds"] for r in rows])))
    summary["diagnostic_gate"] = bool(profile == "dev" and summary["mean_delta"] <= -1
        and sum(v < 0 for v in by_seed.values()) >= 4 and ci[1] < 0
        and summary["changed_fraction"] >= .01 and summary["post_q"] <= summary["source_q"] - 1
        and wait_post <= wait_source + .02 and wait_post <= .10)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = dict(status="COMPLETED", profile=profile, source_run=str(run),
                    source_manifest_sha256=sha(run / "manifest.json"),
                    source_q_sha256=sha(run / "counterfactual_states.csv"),
                    source_audit=str(audit) if audit else None,
                    source_audit_sha256=sha(audit / "independent_audit.json") if audit else None,
                    protocol_sha256=sha(PROTOCOL), proposal_sha256=sha(PROPOSAL),
                    script_sha256=sha(Path(__file__)), start_utc=datetime.now(timezone.utc).isoformat(),
                    wall_seconds=time.monotonic() - start, checkpoint_hashes=checkpoints,
                    artifact_hashes={name:sha(out / name) for name in ("heldout_rows.csv", "train_trace.csv", "summary.json")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "dev"), required=True)
    parser.add_argument("--main-audit-dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    run_probe(args.run_dir.resolve(), out, args.profile,
              args.main_audit_dir.resolve() if args.main_audit_dir else None)


if __name__ == "__main__":
    main()
