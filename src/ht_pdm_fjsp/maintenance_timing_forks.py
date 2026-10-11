"""Common-random-number root forks with a frozen rule continuation."""
from dataclasses import asdict
import math
import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import validate_matching, validate_state
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays, batch_tick, feasible, reference_batch
from ht_pdm_fjsp.maintenance_waiting import transition

CONTINUATION = 'deadline_matching'


def without_preventive(state, action):
    return tuple((m, j) for m, j in action if state.failed[m])


def kind(state, action):
    if not action:
        return 'WAIT'
    preventive = sum(not state.failed[m] for m, _ in action)
    return 'PM' if preventive == len(action) else 'CM' if preventive == 0 else 'MIXED'


def assert_batch_feasible(c, state, selected):
    if (selected.sum(1) > 1).any() or (selected.sum(2) > 1).any():
        raise RuntimeError('matching conflict')
    valid = np.asarray(c.service_time)[None] > 0
    valid = np.broadcast_to(valid, selected.shape).copy()
    valid &= (state['remaining'] == 0)[:, None, :]
    assigned = state['assigned']
    for m in range(c.machines):
        valid[:, m, :] &= ~(assigned == m).any(1)[:, None]
    if (selected & ~valid).any():
        raise RuntimeError('busy or incompatible fork action')


def fork_values(c, state, seed, scenarios, ticks):
    if scenarios < 2 or ticks < 1:
        raise ValueError('at least two scenarios and positive window required')
    validate_state(c, state)
    candidates = feasible(c, state)
    a = len(candidates)
    roots = np.zeros((a, c.machines, c.technicians), bool)
    for i, action in enumerate(candidates):
        validate_matching(c, state, action)
        for m, j in action:
            roots[i, m, j] = True
    # Private forecast RNG, no access to the live environment event tape.
    shocks = np.random.default_rng(seed).random((ticks, scenarios, c.machines)) < c.failure_probability
    batch = arrays(state, a * scenarios)
    totals = np.zeros(a * scenarios)
    components = {k: np.zeros(a * scenarios) for k in ['maintenance_cost', 'unavailability_cost', 'failure_cost', 'waiting_cost']}
    first = None
    for t in range(ticks):
        selected = np.repeat(roots, scenarios, 0) if t == 0 else reference_batch(c, batch, ticks-t, CONTINUATION)
        assert_batch_feasible(c, batch, selected)
        servicing = (selected.any(2)).copy()
        for m in range(c.machines):
            servicing[:, m] |= (batch['assigned'] == m).any(1)
        preventive = selected.any(2) & ~batch['failed']
        corrective = selected.any(2) & batch['failed']
        maintenance = preventive.sum(1)*c.preventive_cost + corrective.sum(1)*c.corrective_cost
        unavailable = (batch['failed'] | servicing).sum(1)*c.unavailable_cost
        waiting = (batch['failed'] & ~servicing & (batch['pending_wait']+1 > c.waiting_limit)).sum(1)*c.waiting_price
        following, cost = batch_tick(c, batch, selected, np.tile(shocks[t], (a, 1)))
        failure = (following['failed'] & ~batch['failed']).sum(1)*c.failure_cost
        if not np.allclose(maintenance+unavailable+waiting+failure, cost, atol=1e-10):
            raise RuntimeError('fork component reconciliation')
        for key, value in zip(components, [maintenance, unavailable, failure, waiting], strict=True):
            components[key] += value
        if t == 0:
            first = cost.reshape(a, scenarios).mean(1)
        totals += cost
        batch = following
    costs = totals.reshape(a, scenarios)
    if not np.isfinite(costs).all():
        raise RuntimeError('nonfinite fork costs')
    return dict(candidates=candidates, costs=costs, means=costs.mean(1),
        mc_se=costs.std(1, ddof=1)/math.sqrt(scenarios), first_step_mean=first,
        component_means={k: v.reshape(a, scenarios).mean(1) for k, v in components.items()},
        shocks=shocks, transitions=a*scenarios*ticks, physical_actions_valid=True)


def paired_delta(values, left, right):
    lookup = {a: i for i, a in enumerate(values['candidates'])}
    samples = values['costs'][lookup[left]] - values['costs'][lookup[right]]
    return float(samples.mean()), float(samples.std(ddof=1)/math.sqrt(len(samples)))


def scalar_replay(c, state, values):
    """Engineering check of EVERY action/scenario against scalar physical dynamics."""
    from ht_pdm_fjsp.maintenance_opportunity_cost import rule_action
    for i, action in enumerate(values['candidates']):
        for j in range(values['shocks'].shape[1]):
            current = state
            total = 0.
            for t in range(len(values['shocks'])):
                pairs = action if t == 0 else rule_action(c, current, len(values['shocks'])-t, CONTINUATION)
                current, cost, _ = transition(c, current, pairs, values['shocks'][t, j])
                total += cost
            if total != values['costs'][i, j]:
                raise RuntimeError('scalar fork replay mismatch')
    return True
