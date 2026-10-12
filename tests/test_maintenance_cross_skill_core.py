"""Independent matching enumeration and physical reference-value checks."""

from dataclasses import replace
from functools import lru_cache
import itertools
import json
import math

import pytest

from ht_pdm_fjsp import maintenance_cross_skill_core as core
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_matching
from ht_pdm_fjsp.maintenance_waiting import transition


def idle(ages=(3, 3, 3, 3), failed=(False,) * 4, waits=(0,) * 4):
    return DispatchState(ages, failed, waits, (0, 0), (-1, -1))


def test_locked_family_reciprocal_specialization_controls_and_serialization():
    specs = core.specs()
    assert len(specs) == 10 and len({s["candidate"] for s in specs}) == 10
    assert sum(s["selectable"] for s in specs) == 8
    for spec in specs:
        c = core.make_config(json.loads(json.dumps(spec)))
        assert c.horizon == 60 and c.machines == 4 and c.technicians == 2
        assert (c.failure_age, c.failure_probability, c.waiting_price) == (4, .45, 12.)
        assert isinstance(c.service_time, tuple) and isinstance(c.service_time[0], tuple)
        if spec["selectable"]:
            group0 = [m for m in range(4) if c.service_time[m][0] == 1]
            assert len(group0) == (2 if spec["candidate"].endswith("balanced") else 3)
            for m in range(4):
                fast = min(range(2), key=lambda j: c.service_time[m][j])
                expert = max(range(2), key=lambda j: c.restoration[m][j])
                assert fast != expert and c.restoration[m][expert] == 1.
                assert c.restoration[m][fast] in (.25, .5)
        elif spec["candidate"] == "old_fast1_slow325":
            assert c.service_time == ((1, 0), (1, 3), (1, 2), (1, 5))
        else:
            assert c.service_time == ((1, 1),) * 4
    assert len(core.specs("smoke")) == 4
    # Callers cannot mutate a later specification result through an earlier one.
    specs[0]["config"]["failure_probability"] = 0.
    assert core.specs()[0]["config"]["failure_probability"] == .45
    with pytest.raises(ValueError, match="profile"):
        core.specs("unknown")


@pytest.mark.parametrize("base", core.frontier.CHEAP)
def test_exact_subset_pool_against_independent_permutation_catalogue(base):
    states = [idle(), idle((3, 0, 5, 1)), idle((0, 1, 2, 1)),
        replace(idle(), remaining=(2, 0), assigned=(0, -1)),
        replace(idle(), remaining=(2, 1), assigned=(0, 1))]
    for spec in core.specs("smoke"):
        c = core.make_config(spec)
        for state in states:
            for tick in (0, 42, 47):
                pool = core.subset_pool(c, state, tick, base)
                assert pool[0] == core.frontier.cheap(c, state, tick, base)
                subset = tuple(sorted(m for m, _ in pool[0]))
                expected = set()
                for workers in itertools.permutations(range(2), len(subset)):
                    action = tuple(zip(subset, workers))
                    try:
                        validate_matching(c, state, action)
                    except ValueError:
                        continue
                    expected.add(action)
                assert set(pool) == expected and len(pool) == len(expected)
                if not subset:
                    assert pool == ((),)
                    assert core.reference_indices(c, state, tick, pool) == {
                        "baseline": 0, "quality": 0, "endpoint_dp": 0}
                    assert core.completion_span(c, state, pool) == 0.


def scalar_isolated(c, machine, state, horizon, worker=None):
    """Expected scalar cost; enumerate shocks and future legal starts directly."""
    one = replace(c, machines=1, service_time=(c.service_time[machine],),
        restoration=(c.restoration[machine],))
    initial = DispatchState((state.ages[machine],), (state.failed[machine],),
        (state.pending_wait[machine],), (0, 0), (-1, -1))

    @lru_cache(None)
    def value(h, x):
        if not h:
            return 0.
        actions = [()]
        if 0 not in x.assigned and (x.failed[0] or x.ages[0] >= one.failure_age - 1):
            actions += [((0, j),) for j in range(2) if one.service_time[0][j]
                and x.remaining[j] == 0]
        return min(q(h, x, a) for a in actions)

    def q(h, x, action):
        total = 0.
        for event, probability in ((False, 1 - one.failure_probability),
            (True, one.failure_probability)):
            following, cost, _ = transition(one, x, action, (event,))
            total += probability * (cost + value(h - 1, following))
        return total

    return q(horizon, initial, () if worker is None else ((0, worker),))


def test_quality_and_endpoint_references_match_manual_scores_and_scalar_physics():
    c = core.make_config(core.specs()[0])
    state = idle((3, 5, 8, 3), (False, True, True, False), (0, 7, 1, 0))
    # Deliberately retain a baseline assignment that neither heuristic must favor.
    pool = (((0, 0), (2, 1)), ((0, 1), (2, 0)))
    tick = 56
    scores = [sum(c.service_time[m][j] / c.restoration[m][j] for m, j in a)
        for a in pool]
    oracle = [sum(scalar_isolated(c, m, state, 4) -
        scalar_isolated(c, m, state, 4, j) for m, j in a) for a in pool]
    refs = core.reference_indices(c, state, tick, pool)
    assert refs["baseline"] == 0
    assert refs["quality"] == min(range(2), key=lambda i: (scores[i], i)) == 1
    assert refs["endpoint_dp"] == max(range(2), key=lambda i: (oracle[i], -i))
    # Verify every DP matching score against an independently enumerated simulator.
    gains = core.dp.tables(c).gains
    for i, action in enumerate(pool):
        actual = sum(gains[4, m, state.ages[m], int(state.failed[m]),
            min(state.pending_wait[m], c.waiting_limit), j] for m, j in action)
        assert actual == pytest.approx(oracle[i], abs=1e-10)


def test_exchangeability_and_exact_score_ties_keep_baseline_first():
    c = core.make_config(next(s for s in core.specs() if s["candidate"] == "exchangeable"))
    state = idle()
    pool = (((0, 1), (1, 0)), ((0, 0), (1, 1)))
    for tick in (0, 47, 59):
        assert core.reference_indices(c, state, tick, pool) == {
            "baseline": 0, "quality": 0, "endpoint_dp": 0}
    assert core.completion_span(c, state, pool) == 0.


def test_completion_span_rounding_matches_real_scalar_service_completions():
    c = core.make_config(core.specs()[0])
    state = idle()
    pool = (((0, 0),), ((0, 1),))
    totals = []
    for action in pool:
        x = state
        for tick in range(c.service_time[0][action[0][1]]):
            x, _, _ = transition(c, x, action if tick == 0 else (), (False,) * 4)
        totals.append(x.ages[0])
    assert totals == [2, 0] and core.completion_span(c, state, pool) == 2.
    # Nominal .875 versus1 is equivalent at age3, but differs at capped age8.
    rounded = replace(c, restoration=((.875, 1.),) * 4)
    assert core.completion_span(rounded, state, pool) == 0.
    assert core.completion_span(rounded, idle((8, 3, 3, 3)), pool) == 1.
    pair_pool = (((0, 0), (2, 1)), ((0, 1), (2, 0)))
    manual = [sum(math.floor(state.ages[m] * (1 - c.restoration[m][j]))
        for m, j in a) for a in pair_pool]
    assert core.completion_span(c, state, pair_pool) == max(manual) - min(manual)


def test_invalid_reference_pools_are_rejected():
    c = core.make_config(core.specs()[0]);state = idle()
    for pool in ((), ((), ()), ((), ((0, 0),)), (((0, 0), (1, 0)),)):
        with pytest.raises(ValueError):
            core.reference_indices(c, state, 0, pool)
    with pytest.raises(ValueError, match="unurgent"):
        core.completion_span(c, idle((0, 3, 3, 3)), (((0, 0),),))
    with pytest.raises(ValueError, match="tick"):
        core.reference_indices(c, state, 60, ((),))
