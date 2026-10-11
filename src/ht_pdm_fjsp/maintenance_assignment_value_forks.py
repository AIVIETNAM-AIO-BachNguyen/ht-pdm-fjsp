"""CM-preserving root interventions with frozen autoregressive RL continuation."""
from dataclasses import asdict
import itertools

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_autoregressive_policy import CONTEXT, STOP, feature_batch
from ht_pdm_fjsp.maintenance_assignment_policy import tie_argmax
from ht_pdm_fjsp.maintenance_dispatch import validate_matching, validate_state
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays, batch_tick, feasible, membership, reference_batch
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_policy_forks import COMPONENTS, batch_features, state_at
from ht_pdm_fjsp.maintenance_timing_forks import assert_batch_feasible
from ht_pdm_fjsp.maintenance_waiting import transition

CONTRASTS = ("timing", "assignment", "joint")


def candidates(config, state, original):
    """Only retain/omit the original PM machines; preserve every CM edge."""
    validate_matching(config, state, original)
    cm = tuple(edge for edge in original if state.failed[edge[0]])
    pm = tuple(edge for edge in original if not state.failed[edge[0]])
    machines = {m for m, _ in pm}
    allowed = []
    for action in feasible(config, state):
        corrective = tuple(edge for edge in action if state.failed[edge[0]])
        preventive = tuple(edge for edge in action if not state.failed[edge[0]])
        if corrective == cm and {m for m, _ in preventive} <= machines:
            allowed.append(action)
    roots = tuple(dict.fromkeys([original, *allowed]))
    timing = [i for i, action in enumerate(roots)
              if all(state.failed[m] or (m, j) in pm for m, j in action)]
    assignment = [i for i, action in enumerate(roots)
                  if {m for m, _ in action if not state.failed[m]} == machines]
    groups = dict(timing=tuple(timing), assignment=tuple(assignment), joint=tuple(range(len(roots))))
    groups["best_single"] = tuple(sorted(set(timing) | set(assignment)))
    no_pm = roots.index(cm)
    if not 1 <= len(roots) <= 7 or any(group[0] != 0 for group in groups.values()):
        raise RuntimeError("N4/K2 candidate contract")
    for root in roots:
        validate_matching(config, state, root)
    return roots, groups, no_pm


def choose(costs, eligible):
    """Original root wins ties; selection data alone enter this choice."""
    costs = np.asarray(costs)
    if (costs.ndim != 2 or not np.isfinite(costs).all() or not eligible or
            eligible[0] != 0 or len(set(eligible)) != len(eligible) or
            any(i < 0 or i >= len(costs) for i in eligible)):
        raise ValueError("invalid selection data/group")
    means = costs.mean(1)
    return min(eligible, key=lambda i: (means[i], i))


def paired_gain(costs, left, right):
    values = np.asarray(costs)[left] - np.asarray(costs)[right]
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite paired futures required")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def ar_batch_features(config, states, remaining, horizon=48):
    data = batch_features(config, states, remaining, horizon)
    owners = states["assigned"][:, None, :] == np.arange(4)[None, :, None]
    data.pop("joint_mask")
    data["pair_mask"] = torch.from_numpy(
        (np.asarray(config.service_time) > 0)[None] & ~owners.any(2)[:, :, None]
        & (states["remaining"] == 0)[:, None, :])
    return data


@torch.no_grad()
def rl_batch(config, states, remaining, model):
    if model.algorithm != CONTEXT:
        raise ValueError("frozen context decoder required")
    data = ar_batch_features(config, states, remaining, model.learned_horizon)
    if getattr(model,"assignment_value_controller",False):
        tokens,_,_=model.sample(data,deterministic=True)
        selected=np.zeros((len(tokens),4,2),dtype=bool)
        for step in range(3):
            tok=tokens[:,step].numpy();rows=np.flatnonzero((tok>=0)&(tok<STOP))
            selected[rows,tok[rows]//2,tok[rows]%2]=True
        # Repeat near value ties scalarly; scalar wrapper also handles PPO ties.
        near=np.array([r["near_tie"] for r in model.last_decode_audit],bool)
        for i in np.flatnonzero(near):
            action,_=model.select(config,state_at(states,int(i)),remaining,model.learned_horizon)
            selected[i]=membership(4,2)[CATALOGUE.index(action)].astype(bool)
        assert_batch_feasible(config,states,selected)
        return selected,dict(decision_rows=len(tokens),near_tie_scalar_rows=int(near.sum()))
    encoded = model.encode(data)
    live = data["pair_mask"].clone()
    size = len(live)
    count = torch.zeros(size, dtype=torch.long)
    active = torch.ones(size, dtype=torch.bool)
    near = torch.zeros(size, dtype=torch.bool)
    selected = np.zeros((size, 4, 2), dtype=bool)
    for _ in range(3):
        distribution, _ = model.token_distribution(data, encoded, live, count)
        top = distribution.logits.topk(2, dim=1).values
        near |= active & (top[:, 0] - top[:, 1] <= 1e-5)
        token = tie_argmax(distribution.logits)
        chosen = active & (token < STOP)
        rows = torch.arange(size)[chosen]
        machines, workers = token[chosen] // 2, token[chosen] % 2
        selected[rows.numpy(), machines.numpy(), workers.numpy()] = True
        live[rows, machines, :] = False
        live[rows, :, workers] = False
        count += chosen.long()
        active = chosen
        if not active.any():
            break
    if active.any():
        raise RuntimeError("missing forced STOP")
    # Batched floating-point arithmetic can perturb argmax ties. Replay the
    # entire ordered sequence scalarly for any affected row.
    for i in np.flatnonzero(near.numpy()):
        action, _ = model.select(config, state_at(states, int(i)), remaining, model.learned_horizon)
        selected[i] = membership(4, 2)[CATALOGUE.index(action)].astype(bool)
    assert_batch_feasible(config, states, selected)
    return selected, dict(decision_rows=size, near_tie_scalar_rows=int(near.sum()))



def fork_roots(config, state, roots, model, seed, scenarios, tick, record_trace=False):
    h, tail_ticks = model.learned_horizon, model.tail_ticks
    if scenarios < 2 or not 0 <= tick < h or model.algorithm != CONTEXT:
        raise ValueError("two futures, original learned clock and context decoder required")
    validate_state(config, state)
    roots = tuple(roots)
    if not roots or len(roots) > 21 or len(set(roots)) != len(roots):
        raise ValueError("one to twenty-one unique roots required")
    for root in roots:
        validate_matching(config, state, root)
    n, physical = len(roots), h+tail_ticks-tick
    root_masks = membership(4, 2)[[CATALOGUE.index(root) for root in roots]].astype(bool)
    shocks = np.random.default_rng(seed).random((physical, scenarios, 4)) < config.failure_probability
    states = arrays(state, n*scenarios)
    base, tail = np.zeros(n*scenarios), np.zeros(n*scenarios)
    components = {k: np.zeros(n*scenarios) for k in COMPONENTS}
    counts = dict(decision_rows=0, near_tie_scalar_rows=0, tail_rows=0)
    cutoff, trace = None, []
    first = None
    for offset in range(physical):
        t = tick+offset
        if offset == 0:
            selected = np.repeat(root_masks, scenarios, axis=0)
        elif t < h:
            selected, audit = rl_batch(config, states, h-t, model)
            for k, value in audit.items():
                counts[k] += value
        else:
            selected = reference_batch(config, states, h+tail_ticks-t, "deadline_matching")
        assert_batch_feasible(config, states, selected)
        if record_trace:
            trace.append(selected.reshape(n, scenarios, 4, 2).copy())
        servicing = selected.any(2) | (states["assigned"][:, None, :] == np.arange(4)[None, :, None]).any(2)
        starts = selected.any(2)
        maintenance = (starts & ~states["failed"]).sum(1)*config.preventive_cost + (starts & states["failed"]).sum(1)*config.corrective_cost
        unavailable = (states["failed"] | servicing).sum(1)*config.unavailable_cost
        waiting = (states["failed"] & ~servicing & (states["pending_wait"]+1 > config.waiting_limit)).sum(1)*config.waiting_price
        following, cost = batch_tick(config, states, selected, np.tile(shocks[offset], (n, 1)))
        failure = (following["failed"] & ~states["failed"]).sum(1)*config.failure_cost
        if not np.array_equal(cost, maintenance+unavailable+waiting+failure):
            raise RuntimeError("component reconciliation failed")
        for k, values in zip(COMPONENTS, (maintenance, unavailable, failure, waiting), strict=True):
            components[k] += values
        if offset == 0:
            first = cost.reshape(n, scenarios).copy()
        if t < h:
            base += cost
        else:
            tail += cost
            counts["tail_rows"] += len(cost)
        states = following
        if t == h-1:
            cutoff = {k: v.reshape(n, scenarios, -1).copy() for k, v in states.items()}
    if cutoff is None or counts["tail_rows"] != n*scenarios*tail_ticks or counts["decision_rows"] != n*scenarios*(h-1-tick):
        raise RuntimeError("original-clock count mismatch")
    return dict(roots=roots, costs=(base+tail).reshape(n, scenarios), base=base.reshape(n, scenarios),
        tail=tail.reshape(n, scenarios), components={k: v.reshape(n, scenarios) for k, v in components.items()},
        cutoff=cutoff, terminal={k: v.reshape(n, scenarios, -1).copy() for k, v in states.items()},
        first=first, shocks=shocks, transitions=n*scenarios*physical, counts=counts,
        trace=np.stack(trace) if record_trace else None)


def scalar_replay(config, state, values, model, tick):
    h, tail_ticks = model.learned_horizon, model.tail_ticks
    from ht_pdm_fjsp.maintenance_opportunity_cost import rule_action
    for i, root in enumerate(values["roots"]):
        for scenario in range(values["shocks"].shape[1]):
            current, base, tail = state, 0., 0.
            components = {k: 0. for k in COMPONENTS}
            for offset in range(h+tail_ticks-tick):
                t = tick+offset
                action = root if offset == 0 else model.select(config, current, h-t, h)[0] if t < h else rule_action(config, current, h+tail_ticks-t, "deadline_matching")
                if values["trace"] is not None:
                    np.testing.assert_array_equal(membership(4, 2)[CATALOGUE.index(action)].astype(bool), values["trace"][offset, i, scenario])
                current, cost, info = transition(config, current, action, values["shocks"][offset, scenario])
                if t < h:
                    base += cost
                else:
                    tail += cost
                for k in COMPONENTS:
                    components[k] += info[k]
                if offset == 0 and cost != values["first"][i, scenario]:
                    raise RuntimeError("scalar root cost mismatch")
                if t == h-1:
                    for k, v in asdict(current).items():
                        np.testing.assert_array_equal(v, values["cutoff"][k][i, scenario])
            if base != values["base"][i, scenario] or tail != values["tail"][i, scenario]:
                raise RuntimeError("scalar base/tail mismatch")
            for k in COMPONENTS:
                if components[k] != values["components"][k][i, scenario]:
                    raise RuntimeError("scalar component mismatch")
            for k, v in asdict(current).items():
                np.testing.assert_array_equal(v, values["terminal"][k][i, scenario])
    return True
