"""Correction budget counts disagreements, not matching planner queries."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('v12_budget',
    Path(__file__).resolve().parents[1] / 'research_result/diagnose_v12_planner_budget.py')
assert _spec is not None and _spec.loader is not None
_budget = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_budget)


def test_correction_budget_counts_only_action_changes():
    choose = _budget.corrected_choice
    assert choose(3, 4, 0, 0) == (3, 0, False)
    assert choose(3, 3, 0, 2) == (3, 0, False)
    assert choose(3, 4, 0, 2) == (4, 1, True)
    assert choose(5, 6, 1, 2) == (6, 2, True)
    assert choose(7, 8, 2, 2) == (7, 2, False)


def test_fixed_budget_grid():
    assert _budget.BUDGETS == (0, 1, 2, 4)
    assert _budget.BOOTSTRAP == 71590000
