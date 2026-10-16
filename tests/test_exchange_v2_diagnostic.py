"""Diagnostic contrasts must distinguish trace activity from net allocation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('exchange_v2_diagnostic',
    Path(__file__).resolve().parents[1]/'scripts/diagnose_exchange_v2_allocation.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


@pytest.mark.parametrize('trace,arm,final,changed,cancelled,cardinality,duration,risk', [
    ([1,2,2,2], 'unrestricted_exchange_ppo', [1,2], 0,1,0,0,0),
    ([1,2,2,0], 'exchange_ppo', [2,1], 1,0,0,3,0),
    ([0,2,2,0], 'exchange_ppo', [2,0], 1,0,0,1,0),
    ([1,0,0,2], 'self_repair_ppo', [1,2], 1,0,1,1,-1),
    ([1,2,3,0], 'exchange_ppo', [0,2], 1,0,-1,-1,1),
    ([1,2,0,0], 'construct_only_ppo', [1,2], 0,0,0,0,0),
])
def test_trace_activity_net_changes_and_static_service_descriptors(
        trace,arm,final,changed,cancelled,cardinality,duration,risk):
    cfg = SimpleNamespace(failure_age=8,service_time=((1,3),(2,1)),
                          restoration=((.5,.9),(.75,.4)))
    state = SimpleNamespace(assigned=(-1,-1),failed=(False,False),ages=(7,7))
    row = diagnostic.trace_summary(np.ones((2,2),dtype=bool),np.array(trace),arm,cfg,state)
    assert row['final_plan'] == final
    assert row['trace_net_changed'] == changed
    assert row['trace_exchange_net_unchanged'] == cancelled
    assert row['trace_cardinality_difference'] == cardinality
    assert row['trace_duration_difference'] == duration
    assert row['trace_risk_unserved_difference'] == risk
    if trace == [1,2,2,0]:
        assert row['trace_restoration_difference'] == pytest.approx(.75)


@pytest.mark.parametrize('arm', ['ppo','normalized_ppo','independent_ppo','completion_reference_ppo'])
def test_noncomparable_trace_is_inapplicable_not_a_zero_effect(arm):
    assert diagnostic.trace_summary(None,None,arm,None,None) is None


@pytest.mark.parametrize('arm', ['exchange_ppo','inverse_mask_ppo'])
def test_locked_or_immediate_inverse_reversal_is_rejected(arm):
    cfg = SimpleNamespace(failure_age=8,service_time=((1,3),(2,1)),
                          restoration=((.5,.9),(.75,.4)))
    state = SimpleNamespace(assigned=(-1,-1),failed=(False,False),ages=(7,7))
    with pytest.raises(ValueError):
        diagnostic.trace_summary(np.ones((2,2),dtype=bool),np.array([1,2,2,2]),arm,cfg,state)


@pytest.mark.parametrize('arm', ['unrestricted_exchange_ppo','inverse_mask_ppo'])
def test_added_controls_have_comparable_feasible_trace_descriptors(arm):
    cfg = SimpleNamespace(failure_age=8,service_time=((1,3),(2,1)),
                          restoration=((.5,.9),(.75,.4)))
    state = SimpleNamespace(assigned=(-1,-1),failed=(False,False),ages=(7,7))
    row=diagnostic.trace_summary(np.ones((2,2),dtype=bool),np.array([1,2,2,0]),arm,cfg,state)
    assert row['trace_net_changed']==1 and row['trace_exchange_net_unchanged']==0
    assert row['trace_duration_difference']==3 and row['final_plan']==[2,1]
