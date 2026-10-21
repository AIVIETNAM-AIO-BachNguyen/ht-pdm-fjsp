"""Physics-level checks for the V15 first-compatible-release hazard feature."""
import sys
from pathlib import Path

import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import DispatchConfig, DispatchState
from ht_pdm_fjsp.maintenance_requests import RequestState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research_result'))
from run_v15_release_hazard_probe import ACTIONS, exposure_features, choose, bootstrap  # noqa: E402


def request_state(ages, failed, remaining, assigned):
    n = len(ages)
    return RequestState(DispatchState(tuple(ages), tuple(failed), (0,) * n,
                                      tuple(remaining), tuple(assigned)),
                        (None,) * n, (0,) * n)


def test_current_tick_hazard_and_age_agnostic_control_diverge():
    cfg = DispatchConfig(2, 1, 64, 4, .5, ((2,), (2,)), ((.75,), (.75,)))
    state = request_state((0, 3), (False, False), (0,), (-1,))
    wait = ACTIONS.index(())
    young = ACTIONS.index(((0, 0),))
    at_risk = ACTIONS.index(((1, 0),))
    got = exposure_features(cfg, state, np.ones(2), [wait, young, at_risk])
    assert got[wait] == (0., 0.)
    assert np.allclose(got[young], (.25, -.25))
    assert np.allclose(got[at_risk], (-.5, -.25))


def test_specialist_compatibility_busy_worker_and_failed_exclusion():
    cfg = DispatchConfig(3, 2, 64, 4, .5,
                         ((2, 0), (2, 0), (0, 2)), ((.75, .75),) * 3)
    wait = ACTIONS.index(())
    serve_other = ACTIONS.index(((2, 1),))
    state = request_state((3, 1, 0), (False, False, False), (2, 0), (1, -1))
    got = exposure_features(cfg, state, np.ones(3), [wait, serve_other])
    assert np.allclose(got[serve_other], (0., -.5))
    failed = request_state((3, 1, 0), (True, False, False), (2, 0), (1, -1))
    got_failed = exposure_features(cfg, failed, np.array([1., 0., 0.]), [wait, serve_other])
    assert got_failed[serve_other] == (0., 0.)


def test_action_tie_and_paired_bootstrap_zero_cube():
    wait = ACTIONS.index(())
    one = ACTIONS.index(((0, 0),))
    assert choose(np.zeros(len(ACTIONS)), [wait, one], {wait: 0., one: 0.}, 0.) == min(wait, one)
    assert bootstrap(np.zeros((1, 1, 1, 3)), 100) == (0., 0.)
