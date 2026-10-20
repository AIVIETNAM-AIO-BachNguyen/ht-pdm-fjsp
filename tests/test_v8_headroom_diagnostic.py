"""Root-value decomposition, taxonomy and current-action mask gates."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_framework_v1 import catalogue, configs
from ht_pdm_fjsp.maintenance_requests import initial_state, submit_requests

_spec = importlib.util.spec_from_file_location(
    "diagnose_v8_headroom", Path(__file__).resolve().parents[1] / "research_result/diagnose_v8_headroom.py"
)
assert _spec is not None and _spec.loader is not None
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)


def test_clock_bin_and_rank_determinism():
    assert [_probe.clock_bin(t) for t in (0, 14, 15, 29, 30, 41, 42)] == [
        "early", "early", "middle", "middle", "late", "late", None,
    ]
    assert _probe.state_rank("x", "shared_free", 1, 2, 3) == _probe.state_rank(
        "x", "shared_free", 1, 2, 3)
    assert _probe.state_rank("x", "shared_free", 1, 2, 3) != _probe.state_rank(
        "x", "shared_free", 1, 2, 4)


def test_exact_regret_decomposition_and_tie_rule():
    scores = {0: 10.0, 1: 7.0, 2: 9.0}
    assert _probe.select_best(scores, {0, 1, 2}, baseline_index=0) == 1
    assert _probe.select_best(scores, {0, 2}, baseline_index=0) == 2
    assert _probe.gap_components(scores, chosen=0, full_best=1, allowed_best=2) == (1.0, 2.0, 3.0)
    assert _probe.select_best({0: 5.0, 1: 5.0}, {0, 1}, baseline_index=1) == 1
    with pytest.raises(RuntimeError, match="root-regret algebra"):
        _probe.gap_components(scores, chosen=1, full_best=0, allowed_best=2)


def test_taxonomy_and_current_legal_mask():
    actions = catalogue()
    wait = actions.index(())
    a = actions.index(((0, 0),))
    b = actions.index(((0, 1),))
    c = actions.index(((1, 0),))
    assert _probe.category(a, a, actions) == "match"
    assert _probe.category(wait, a, actions) == "under_service"
    assert _probe.category(a, wait, actions) == "over_service"
    assert _probe.category(a, b, actions) == "assignment"
    assert _probe.category(a, c, actions) == "machine_subset"
    config = configs()["reciprocal_scale2"]
    physical = DispatchState(
        ages=(3, 0, 0, 0), failed=(False,) * 4, pending_wait=(0,) * 4,
        remaining=(0, 0), assigned=(-1, -1),
    )
    state = submit_requests(config, initial_state(config, physical), 5, (0,))
    submissions, admitted, present_actions, legal, canonical, observation = _probe.current_actions(
        config, state, 5)
    assert submissions == () and admitted.requests[0] is not None
    assert legal.shape == canonical.shape == (21,) and legal[wait] and canonical[wait]
    assert legal[a] and legal[b] and canonical[a] and not canonical[b]
    assert observation.shape == (56,)
