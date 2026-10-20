"""Focused schedule and divergence checks for V12 planner intervention."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('v12_sparse',
    Path(__file__).resolve().parents[1] / 'research_result/diagnose_v12_sparse_planner.py')
assert _spec is not None and _spec.loader is not None
_diagnostic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diagnostic)


def test_one_action_and_four_tick_wait_schedule():
    one = next(i for i, action in enumerate(_diagnostic.ACTIONS) if action)
    assert _diagnostic.forced_actions(None, 0, 'one') == {}
    assert _diagnostic.forced_actions(4, 0, 'actor') == {}
    assert _diagnostic.forced_actions(4, one, 'one') == {4: one}
    assert _diagnostic.forced_actions(4, one, 'wait4') == {4: one}
    assert _diagnostic.forced_actions(4, 0, 'one') == {4: 0}
    assert _diagnostic.forced_actions(4, 0, 'wait4') == {4: 0, 5: 0, 6: 0, 7: 0}
    assert _diagnostic.forced_actions(47, 0, 'wait4') == {47: 0}


def test_category_distinguishes_wait_and_assignment():
    actions = _diagnostic.ACTIONS
    wait = actions.index(())
    service = actions.index(((0, 0),))
    other_assignment = actions.index(((0, 1),))
    assert _diagnostic.category(service, wait) == 'actor_service_planner_wait'
    assert _diagnostic.category(wait, service) == 'actor_wait_planner_service'
    assert _diagnostic.category(service, other_assignment) == 'different_technician_assignment'
