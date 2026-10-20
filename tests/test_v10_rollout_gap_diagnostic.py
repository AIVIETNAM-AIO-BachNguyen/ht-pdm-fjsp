"""Mutually exclusive timing/priority/assignment classes and sample tie rules."""
import importlib.util
from pathlib import Path

_spec=importlib.util.spec_from_file_location('v10_rollout_gap_diagnostic',
    Path(__file__).resolve().parents[1]/'research_result/diagnose_v10_rollout_gap.py')
assert _spec is not None and _spec.loader is not None
_d=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)


def index(action):return _d.ACTIONS.index(action)


def test_action_classes_separate_wait_cardinality_subset_and_assignment():
    wait=index(())
    one=index(((0,0),))
    two=index(((0,0),(1,1)))
    other=index(((2,0),))
    swapped=index(((0,1),(1,0)))
    assert _d.action_class(wait,one)=='wait_vs_service'
    assert _d.action_class(one,wait)=='service_vs_wait'
    assert _d.action_class(one,two)=='lower_cardinality'
    assert _d.action_class(two,one)=='higher_cardinality'
    assert _d.action_class(one,other)=='machine_subset'
    assert _d.action_class(two,swapped)=='technician_assignment'
    assert _d.action_class(one,one)=='match'


def test_rank_bins_and_risk_first_root_ties():
    assert [_d.clock_bin(t) for t in (0,15,16,31,32,47)]==[
        'early','early','middle','middle','late','late']
    assert _d.rank('case','all_action',1,2,3)==_d.rank('case','all_action',1,2,3)
    assert _d.rank('case','all_action',1,2,3)!=_d.rank('case','all_action',1,2,4)
    assert _d.best_root({0:5.,1:5.,2:8.},risk=1)==1
    assert _d.best_root({0:5.,1:4.,2:8.},risk=0)==1
