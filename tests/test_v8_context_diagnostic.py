"""Valid paired current states and structural actor blindness probe."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_framework_v1 import catalogue, configs, vector_observation
from ht_pdm_fjsp.maintenance_framework_v6_policy import EdgeSharingActor
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import (
    initial_state, legal_matchings, submit_requests, validate_state,
)
_spec = importlib.util.spec_from_file_location(
    "diagnose_v8_context", Path(__file__).resolve().parents[1] / "research_result/diagnose_v8_context.py"
)
assert _spec is not None and _spec.loader is not None
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)
compare_planner = _probe.compare_planner
next_tick_request_difference = _probe.next_tick_request_difference
selected_absent_machine = _probe.selected_absent_machine


def paired_state():
    config = configs()["reciprocal_scale2"]
    physical = DispatchState(
        ages=(3, 0, 1, 0), failed=(False,) * 4,
        pending_wait=(0,) * 4, remaining=(0, 0), assigned=(-1, -1),
    )
    base = initial_state(config, physical)
    state = submit_requests(config, base, 5, (0,))
    return config, state


def test_absent_request_age_pair_keeps_root_actions_and_changes_next_request():
    config, state = paired_state()
    machine, changed = selected_absent_machine(config, state, 5)
    assert machine == 1 and state.physical.ages[1] == 0 and changed.physical.ages[1] == 2
    validate_state(config, changed, 5)
    assert request_action(config, state, 5, "threshold") == request_action(config, changed, 5, "threshold")
    assert legal_matchings(config, state) == legal_matchings(config, changed)
    assert next_tick_request_difference(config, state, changed, 5, ())


def test_shared_actor_legal_logits_are_context_invariant_but_planner_scores_can_read_pair():
    config, state = paired_state()
    _, changed = selected_absent_machine(config, state, 5)
    obs = vector_observation(config, state, 5)
    other = vector_observation(config, changed, 5)
    assert not torch.equal(torch.as_tensor(obs), torch.as_tensor(other))
    actor = EdgeSharingActor("shared_edge")
    actions = catalogue()
    legal = legal_matchings(config, state)
    indices = [actions.index(action) for action in legal]
    with torch.no_grad():
        left = actor(torch.as_tensor(obs[None]))[0]
        right = actor(torch.as_tensor(other[None]))[0]
    torch.testing.assert_close(left[indices], right[indices], atol=0, rtol=0)
    planner = compare_planner(config, state, changed, 5, 69020000, 4, actions)
    assert set(planner["original_scores"]) == set(planner["changed_scores"]) == set(indices)
    assert planner["forecast_transitions"] == 2 * len(indices) * 4 * 6
