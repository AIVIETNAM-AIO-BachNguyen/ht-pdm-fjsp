"""Independent physical-transition checks for the isolated DP surrogate."""

from dataclasses import replace
from functools import lru_cache
import numpy as np
import pytest

from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_matching
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays
from ht_pdm_fjsp.maintenance_restricted_dp import RestrictedDP, batch, select
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, transition


def config(probability=.45, restoration=.75):
    return WaitingConfig(1, 2, 8, 3, probability, ((2, 3),),
        ((restoration, 1.),), waiting_limit=2, waiting_price=12.)


def oracle(c):
    """Enumerate events/actions using the real one-machine scalar transition.

    Waiting remains unbounded, and the recursion checks urgency every step.
    This does not use the DP's recurrence, tables or vector simulator.
    """
    @lru_cache(None)
    def value(h, state):
        if h == 0:
            return 0.
        actions = [()]
        if 0 not in state.assigned and (state.failed[0] or state.ages[0] >= c.failure_age - 1):
            actions += [((0, j),) for j, duration in enumerate(c.service_time[0])
                if duration and state.remaining[j] == 0]
        return min(q(h, state, action) for action in actions)

    def q(h, state, action):
        total = 0.
        for event, probability in ((False, 1 - c.failure_probability), (True, c.failure_probability)):
            if probability:
                following, cost, _ = transition(c, state, action, (event,))
                total += probability * (cost + value(h - 1, following))
        return total
    return value, q


def state(age, failed=False, wait=0, remaining=0, worker=0, workers=2):
    return DispatchState((age,), (failed,), (wait,),
        tuple(remaining if j == worker else 0 for j in range(workers)),
        tuple(0 if remaining and j == worker else -1 for j in range(workers)))


@pytest.mark.parametrize("probability", [0., .45, 1.])
@pytest.mark.parametrize("restoration", [.5, .75, 1.])
def test_against_independent_scalar_expectation(probability, restoration):
    c = config(probability, restoration)
    dp = RestrictedDP(c)
    expected, q = oracle(c)
    cases = [state(0), state(1), state(2), state(5), state(2, True),
        state(5, True, 1), state(5, True, 2), state(5, True, 9),
        state(5, True, 9, remaining=2), state(5, remaining=1, worker=1)]
    for h in (1, 2, 4, 6):
        for s in cases:
            j = next((j for j, r in enumerate(s.remaining) if r), -1)
            assert dp.value(h, 0, s.ages[0], s.failed[0], s.pending_wait[0],
                max(s.remaining), j) == pytest.approx(expected(h, s), abs=1e-10)
            if j == -1 and (s.failed[0] or s.ages[0] >= c.failure_age - 1):
                for worker in range(c.technicians):
                    independent_gain = q(h, s, ()) - q(h, s, ((0, worker),))
                    assert dp.gains[h, 0, s.ages[0], int(s.failed[0]), min(s.pending_wait[0], c.waiting_limit), worker] == pytest.approx(independent_gain, abs=1e-10)


def test_early_pm_is_prohibited_and_terminal_completion_has_no_extra_value():
    c = config(1.)
    dp = RestrictedDP(c)
    assert np.isneginf(dp.gains[:, :, :c.failure_age - 1, 0]).all()
    assert dp.select(state(1), 8, "isolated_endpoint") == ()
    # An ongoing failed job completing exactly at the horizon pays only downtime.
    assert dp.value(2, 0, 7, True, 99, remaining=2, worker=0) == 2 * c.unavailable_cost
    expected, _ = oracle(c)
    assert dp.value(3, 0, 7, True, 99, remaining=2, worker=0) == pytest.approx(
        expected(3, state(7, True, 99, remaining=2)))


@pytest.mark.parametrize("limit", [0, 2, 4])
def test_unbounded_failed_wait_is_equivalent_after_limit(limit):
    c = replace(config(), waiting_limit=limit)
    dp = RestrictedDP(c)
    expected, _ = oracle(c)
    for h in (1, 4, 6):
        for wait in (limit, limit + 1, limit + 17):
            assert dp.value(h, 0, 3, True, wait) == pytest.approx(expected(h, state(3, True, wait)))
        assert dp.value(h, 0, 3, True, limit) == dp.value(h, 0, 3, True, limit + 17)


def test_wait_first_ties_and_partial_subsets_are_feasible():
    c = WaitingConfig(3, 2, 8, 3, .45, ((1, 0), (0, 4), (1, 2)),
        ((.75, .75),) * 3, waiting_limit=2, waiting_price=12.)
    dp = RestrictedDP(c)
    s = DispatchState((3, 3, 0), (True, True, False), (3, 0, 0), (0, 0), (-1, -1))
    # At horizon1, overdue CM has gain10; non-overdue CM has gain-2.
    assert dp.select(s, 1, "isolated_endpoint") == ((0, 0),)
    validate_matching(c, s, dp.select(s, 1, "isolated_endpoint"))
    busy = replace(s, remaining=(1, 0), assigned=(2, -1))
    assert dp.select(busy, 1, "isolated_endpoint") == ()
    tied = replace(c, preventive_cost=0., corrective_cost=0., unavailable_cost=0.,
        failure_cost=0., waiting_price=0.)
    assert RestrictedDP(tied).select(s, 8, "isolated_endpoint") == ()


def test_label_permutation_preserves_unique_physical_matching():
    c = WaitingConfig(2, 2, 8, 3, .45, ((1, 4), (3, 1)), ((.75, .75),) * 2,
        waiting_limit=2, waiting_price=12.)
    s = DispatchState((3, 5), (True, True), (4, 3), (0, 0), (-1, -1))
    chosen = RestrictedDP(c).select(s, 6, "isolated_endpoint")
    assert chosen == ((0, 0), (1, 1))
    for machine_order, worker_order in (((1, 0), (0, 1)), ((0, 1), (1, 0)), ((1, 0), (1, 0))):
        permuted_c = replace(c,
            service_time=tuple(tuple(c.service_time[m][j] for j in worker_order) for m in machine_order),
            restoration=tuple(tuple(c.restoration[m][j] for j in worker_order) for m in machine_order))
        permuted_s = DispatchState(tuple(s.ages[m] for m in machine_order),
            tuple(s.failed[m] for m in machine_order), tuple(s.pending_wait[m] for m in machine_order),
            (0, 0), (-1, -1))
        result = RestrictedDP(permuted_c).select(permuted_s, 6, "isolated_endpoint")
        restored = tuple(sorted((machine_order[m], worker_order[j]) for m, j in result))
        assert restored == chosen


def test_vector_scalar_consistency_horizon_clip_and_fingerprints():
    c = config()
    dp = RestrictedDP(c)
    states = [state(0), state(2), state(5, True, 20), state(5, True, 20, 2)]
    vector = {key: np.concatenate([arrays(s)[key] for s in states]) for key in arrays(states[0])}
    for name in ("isolated_h6", "isolated_endpoint"):
        for h in (1, 6, 8):
            result = dp.batch(vector, h, name)
            assert result.dtype == bool and result.shape == (4, 1, 2)
            assert np.array_equal(result, batch(c, vector, h, name))
            for index, s in enumerate(states):
                action = tuple((int(m), int(j)) for m, j in zip(*np.nonzero(result[index])))
                assert action == select(c, s, h, name) == dp.select(s, h, name)
                validate_matching(c, s, action)
    assert np.array_equal(dp.batch(vector, 8, "isolated_h6"), dp.batch(vector, 6, "isolated_endpoint"))
    assert dp.fingerprints() == RestrictedDP(c).fingerprints()
    assert all(len(digest) == 64 for digest in dp.fingerprints().values())
    with pytest.raises(ValueError):
        dp.batch(vector, 0, "isolated_endpoint")
    with pytest.raises(ValueError):
        dp.batch(vector, 9, "isolated_endpoint")
    with pytest.raises(ValueError):
        dp.batch(vector, 1, "unknown")
