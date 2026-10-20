"""First-divergence action classes and common-bank full-horizon Q."""
import importlib.util
from pathlib import Path

_spec=importlib.util.spec_from_file_location('first_divergence',
    Path(__file__).resolve().parents[1]/'research_result/diagnose_v10_first_divergence.py')
assert _spec is not None and _spec.loader is not None
_d=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_d)


def test_divergence_categories():
    index=_d.ACTIONS.index
    wait=index(())
    one=index(((0,0),))
    another=index(((0,1),))
    second=index(((1,0),))
    two=index(((0,0),(1,1)))
    assert _d.category(wait,one)=='wait_vs_service'
    assert _d.category(one,wait)=='service_vs_wait'
    assert _d.category(one,two)=='lower_cardinality'
    assert _d.category(two,one)=='higher_cardinality'
    assert _d.category(one,another)=='technician_assignment'
    assert _d.category(one,second)=='machine_subset'


def test_full_horizon_bank_is_deterministic_and_extends_six_ticks():
    cfg=_d.configs()['interchangeable_scale1']
    env=_d.FrameworkEnv(cfg);env.reset(seed=70410000)
    state=env.inner.state
    _,_,legal,_=_d.legal_present(cfg,state,0)
    assert len(legal)>=1
    seed,bank=_d.bank_for(cfg,'interchangeable_scale1',70400000,70410000,0,1)
    assert (seed,bank)==_d.bank_for(cfg,'interchangeable_scale1',70400000,70410000,0,1)
    assert seed!=_d.bank_for(cfg,'interchangeable_scale1',70400001,70410000,0,1)[0]
    short,full,steps=_d.forecast_root(cfg,state,0,legal[0],bank,lambda:None)
    assert steps==cfg.horizon
    assert full>=short>=0
