"""Matched ensemble optimization budgets, common development selection."""
import copy
import hashlib
import math
import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_assignment_opportunity_benchmark import VERSION, ARMS
from ht_pdm_fjsp.maintenance_assignment_value_policy import RelativeValueHead, CONTRACT
from ht_pdm_fjsp.maintenance_assignment_value_training import predictions, mse
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_opportunity_cost import compact


def draw_indices(data, arm, size, rng):
    if arm not in ARMS or size < 1:
        raise ValueError("known arm and positive budget required")
    groups = []
    for cond in (0, 1):
        base = data["conditions"] == cond
        bins = sorted(set((data["ticks"][base] // 12).tolist())) if arm != "landmark_low" else [None]
        if not bins or not base.any():
            raise RuntimeError("missing natural condition labels; no resampling")
        groups.append([(base if clock is None else base & (data["ticks"] // 12 == clock)).nonzero().flatten().numpy() for clock in bins])
    indices = []
    for _ in range(size):
        clocks = groups[int(rng.integers(2))]
        pool = clocks[int(rng.integers(len(clocks)))]
        indices.append(int(pool[int(rng.integers(len(pool)))]))
    return torch.tensor(indices, dtype=torch.long)


def fit(output, cfg, train_seed, source_seed, source_hash, train, dev, journals, stats):
    result = {arm: [] for arm in ARMS}; selections = {}; initial_hashes = []
    for member in tqdm(range(cfg["ensemble_members"]), desc=f"Ensemble PPO {source_seed}", unit="member"):
        torch.manual_seed(role_seed(train_seed, f"{VERSION}:initialization:{member}"))
        first = RelativeValueHead(cfg["hidden"])
        heads = {a: copy.deepcopy(first) for a in ARMS}
        initial_hash = hashlib.sha256(b"".join(v.numpy().tobytes() for v in first.state_dict().values())).hexdigest()
        initial_hashes.append(initial_hash)
        optimizers = {a: torch.optim.Adam(h.parameters(), lr=cfg["learning_rate"]) for a, h in heads.items()}
        candidates = {a: [] for a in ARMS}
        for epoch in tqdm(range(1, cfg["epochs"] + 1), desc=f"Matched head training {source_seed}/{member}", leave=False):
            epoch_draws = {}
            for arm, head in heads.items():
                # Same member/epoch RNG stream; indexing distributions differ by design.
                rng = np.random.default_rng(role_seed(train_seed, f"{VERSION}:draws:{member}:{epoch}"))
                data = train[arm]
                draws = draw_indices(data, arm, cfg["draws_per_epoch"], rng)
                epoch_draws[arm] = draws
                if arm == "opportunity_high" and not torch.equal(draws, epoch_draws["opportunity_low"]):
                    raise RuntimeError("low/high opportunity draws lost pairing")
                journals["training_draws"].add(dict(source_seed=source_seed, member=member, epoch=epoch,
                    arm=arm, draw_indices=compact(draws.tolist()), unique_labels=len(draws.unique()),
                    probe_draws=int(data["origins"][draws].sum()), late_draws=int((data["ticks"][draws] >= 12).sum()),
                    draws_by_clock_bin=compact(torch.bincount(data["ticks"][draws]//12, minlength=4).tolist())))
                head.train(); total = 0.
                for start in range(0, len(draws), cfg["minibatch"]):
                    ids = draws[start:start + cfg["minibatch"]]
                    loss = torch.nn.functional.smooth_l1_loss(predictions(head, data, ids), data["targets"][ids])
                    optimizers[arm].zero_grad(); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise RuntimeError("nonfinite optimization")
                    optimizers[arm].step(); total += float(loss.detach()) * len(ids)
                    stats["optimizer_steps"] += 1
                head.eval(); row = dict(source_seed=source_seed, train_seed=train_seed, member=member,
                    arm=arm, epoch=epoch, train_smooth_l1=total / len(draws), common_development_mse=mse(head, dev),
                    draws=len(draws), initial_sha256=initial_hash)
                if not math.isfinite(row["common_development_mse"]):
                    raise RuntimeError("nonfinite development predictions")
                journals["training_progress"].add(row)
                if epoch in cfg["checkpoints"]:
                    path = output / arm / f"source_{source_seed}" / f"member_{member}_epoch_{epoch}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = dict(protocol=VERSION, feature_contract=CONTRACT, settings=cfg, arm=arm,
                        source_seed=source_seed, source_sha256=source_hash, train_seed=train_seed,
                        member=member, epoch=epoch, state_dict=head.state_dict())
                    torch.save(payload, path)
                    restored = RelativeValueHead(cfg["hidden"])
                    restored.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
                    if not torch.equal(predictions(head, dev), predictions(restored, dev)):
                        raise RuntimeError("checkpoint prediction roundtrip")
                    candidates[arm].append({**row, "checkpoint": path.relative_to(output).as_posix()})
        for arm in ARMS:
            chosen = min(candidates[arm], key=lambda r: (r["common_development_mse"], r["epoch"]))
            payload = torch.load(output / chosen["checkpoint"], weights_only=True)
            if payload["source_sha256"] != source_hash or payload["settings"] != cfg or payload["feature_contract"] != CONTRACT:
                raise RuntimeError("checkpoint source/protocol binding")
            restored = RelativeValueHead(cfg["hidden"]); restored.load_state_dict(payload["state_dict"])
            restored_hash = hashlib.sha256(b"".join(v.numpy().tobytes() for v in restored.state_dict().values())).hexdigest()
            if restored_hash == initial_hash:
                raise RuntimeError("selected head never changed from initialization")
            result[arm].append(restored.eval())
            selections[f"{arm}:{member}"] = dict(selected=chosen, candidates=candidates[arm])
    if len(set(initial_hashes)) != cfg["ensemble_members"]:
        raise RuntimeError("ensemble members share identical initialization")
    return result, selections
