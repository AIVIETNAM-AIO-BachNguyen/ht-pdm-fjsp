"""Ensemble relative-cost decisions with separately calibrated empirical guards."""
import numpy as np
import torch

from ht_pdm_fjsp.maintenance_assignment_value_policy import RelativeValueHead, inputs, allowed_mask, canonical_tokens
from ht_pdm_fjsp.maintenance_autoregressive_policy import CONTEXT, STOP, feature_batch
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE


@torch.no_grad()
def pair_values(heads, data):
    pairs = data["pairs"]
    feature = {k: data[k][pairs[:, 0]] for k in ("raw_pairs", "context")}
    rows = torch.arange(len(pairs))
    return torch.stack([h(feature)[rows, pairs[:, 2]] - h(feature)[rows, pairs[:, 1]] for h in heads])


def calibrate(heads, data, cfg):
    values = pair_values(heads, data).numpy()
    rows = data["pairs"].numpy()
    original = data["original_ids"][data["pairs"][:, 0]].numpy()
    result = {}; records = []
    for ci, condition in enumerate(cfg["conditions"]):
        errors = []
        for i, (_, left, right) in enumerate(rows):
            if int(data["conditions"][i]) != ci or original[i] not in (left, right):
                continue
            # Cost gain orig-alt: right-left cost if right is original, opposite otherwise.
            sign = 1 if original[i] == right else -1
            member_gain = sign * values[:, i]
            mean = float(member_gain.mean()); sd = float(member_gain.std(ddof=1))
            observed = sign * float(data["targets"][i]); se = float(data["mc_se"][i])
            adjusted = mean - cfg["gate_sd_weight"] * sd
            error = adjusted - (observed - cfg["gate_mc_weight"] * se)
            errors.append(error)
            records.append(dict(condition=condition, pair_index=i, state_index=int(rows[i, 0]),
                origin="probe" if data["origins"][i] else "natural", tick=int(data["ticks"][i]),
                member_gains=member_gain.tolist(), predicted_gain=mean, ensemble_sd=sd,
                observed_gain=observed, mc_se=se, empirical_overoptimism=error))
        enabled = len(errors) >= cfg["calibration_minimum"]
        margin = max(0., float(np.quantile(errors, cfg["gate_quantile"], method="higher"))) if enabled else None
        result[condition] = dict(enabled=enabled, margin=margin, samples=len(errors),
            minimum=cfg["calibration_minimum"], quantile=cfg["gate_quantile"],
            semantics="empirical guard; not a statistical confidence/safety guarantee")
    return result, records


def choose_assignment(member_scores, original, ids, gated, calibration, sd_weight=1., tolerance=1e-5):
    scores = np.asarray(member_scores)
    if scores.ndim != 2 or len(scores) < 2 or not np.isfinite(scores).all() or original not in ids:
        raise ValueError("finite ensemble and original feasible action required")
    delta = scores[:, original, None] - scores[:, ids]
    gain = delta.mean(0); sd = delta.std(0, ddof=1)
    adjusted = gain - sd_weight * sd if gated else gain
    eligible = [k for k, action in enumerate(ids) if action != original and
        (not gated or calibration["enabled"]) and
        adjusted[k] > (calibration["margin"] if gated and calibration["enabled"] else 0.) + tolerance]
    chosen = min(eligible, key=lambda k: (-adjusted[k], ids[k])) if eligible else ids.index(original)
    audit = dict(original_id=original, chosen_id=ids[chosen], candidates=ids,
        member_gains=delta.T.tolist(), predicted_gains=gain.tolist(), ensemble_sd=sd.tolist(),
        adjusted_gains=adjusted.tolist(), gated=gated,
        margin=calibration["margin"] if gated else 0., calibration_enabled=calibration["enabled"],
        changed=ids[chosen] != original, near_tie=False,
        rejection_reason="none" if ids[chosen] != original else "no_alternative" if len(ids) == 1 else
            "insufficient_calibration" if gated and not calibration["enabled"] else "insufficient_predicted_gain")
    # Recompute threshold-sensitive and argmax-sensitive decisions in scalar mode.
    threshold = (calibration["margin"] or 0.) + tolerance if gated else tolerance
    audit["near_tie"] = bool(np.any(np.abs(adjusted - threshold) <= 2e-5) or
        len(eligible) > 1 and abs(sorted(adjusted[eligible])[-1] - sorted(adjusted[eligible])[-2]) <= 2e-5)
    return ids[chosen], audit


class ConfidenceController:
    assignment_value_controller = True
    algorithm = CONTEXT

    def __init__(self, base, heads, name, calibration, cfg):
        self.base = base; self.heads = [h.eval() for h in heads]; self.name = name
        self.calibration = calibration; self.cfg = cfg; self.gated = name.endswith("_gated")
        self.learned_horizon, self.tail_ticks = base.learned_horizon, base.tail_ticks
        self.last_decode_audit = None

    @torch.no_grad()
    def sample(self, batch, generator=None, deterministic=False):
        if not deterministic:
            raise ValueError("deterministic empirical policy improvement only")
        tokens = self.base.greedy_tokens(batch)
        encoded = self.base.encode(batch); live = batch["pair_mask"].clone()
        active = torch.ones(len(tokens), dtype=torch.bool); count = torch.zeros(len(tokens), dtype=torch.long)
        near = torch.zeros(len(tokens), dtype=torch.bool)
        for step in range(3):
            dist, _ = self.base.token_distribution(batch, encoded, live, count)
            top = dist.logits.topk(2, 1).values
            near |= active & (top[:, 0] - top[:, 1] <= 1.1e-5)
            tok = tokens[:, step]; chosen = active & (tok >= 0) & (tok < STOP)
            rows = torch.arange(len(tokens))[chosen]
            live[rows, tok[chosen] // 2, :] = False; live[rows, :, tok[chosen] % 2] = False
            count += chosen.long(); active = chosen
        for i in near.nonzero().flatten().tolist():
            tokens[i] = self.base.greedy_tokens({k: v[i:i+1] for k, v in batch.items()})[0]
        originals = [tuple(sorted((t // 2, t % 2) for t in row if 0 <= t < STOP)) for row in tokens.tolist()]
        original_ids = torch.tensor([CATALOGUE.index(a) for a in originals])
        mask = allowed_mask(batch, original_ids)
        data = inputs(batch, self.base)
        scores = torch.stack([h(data) for h in self.heads]).numpy()
        output, audits = [], []
        for i, original in enumerate(original_ids.tolist()):
            # Structural service eligibility is the first raw edge feature; availability
            # is transient and cannot be used to infer the experimental condition.
            condition = "skill_mask" if (batch["edges"][i, ..., 0] < .5).any() else "dense"
            ids = mask[i].nonzero().flatten().tolist()
            chosen, audit = choose_assignment(scores[:, i], original, ids, self.gated,
                self.calibration[condition], self.cfg["gate_sd_weight"], self.cfg["tie_tolerance"])
            if audit["near_tie"]:
                sub = inputs({k: v[i:i+1] for k, v in batch.items()}, self.base)
                scalar = torch.stack([h(sub)[0] for h in self.heads]).numpy()
                chosen, audit = choose_assignment(scalar, original, ids, self.gated,
                    self.calibration[condition], self.cfg["gate_sd_weight"], self.cfg["tie_tolerance"])
            audit.update(condition=condition, original=originals[i], selected=CATALOGUE[chosen],
                near_tie=bool(near[i]) or audit["near_tie"])
            output.append(canonical_tokens(CATALOGUE[chosen])); audits.append(audit)
        action = torch.tensor(output, dtype=torch.long)
        logprob, _, value = self.base.evaluate(batch, action)
        self.last_decode_audit = audits
        return action, logprob, value

    def decode(self, c, state, action):
        return self.base.decode(c, state, action)

    @torch.no_grad()
    def select(self, c, state, remaining, horizon):
        if horizon != self.learned_horizon:
            raise ValueError("original learned horizon required")
        action, logprob, value = self.sample(feature_batch([c], [state], [remaining], horizon), deterministic=True)
        return self.decode(c, state, action[0]), dict(token_trace=action[0].tolist(),
            log_probability=float(logprob[0]), value=float(value[0]), assignment_decode=self.last_decode_audit[0])
