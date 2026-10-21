import numpy as np
from ht_pdm_fjsp.maintenance_framework_v1 import catalogue,configs
from ht_pdm_fjsp.maintenance_requests import initial_state
from research_result.diagnose_v17_first_divergence import bank,bootstrap,kind,root_cost


def test_category_covers_wait_machine_and_assignment():
    actions=catalogue();wait=actions.index(())
    service=[i for i,a in enumerate(actions) if len(a)==1]
    assert kind(service[0],wait)=='actor_service_rollout_wait'
    assert kind(wait,service[0])=='actor_wait_rollout_service'
    assert kind(service[0],service[0])=='same'
    pairs=[(i,j) for i in service for j in service if i!=j]
    assert any(kind(i,j)=='machine_set' for i,j in pairs)
    assert any(kind(i,j)=='assignment' for i,j in pairs)


def test_common_bank_and_no_future_information():
    cfg=configs()['reciprocal_scale2']
    seed,a=bank(cfg,'reciprocal_scale2',73000000,73011000,4,2)
    other_seed,b=bank(cfg,'reciprocal_scale2',73000000,73011000,4,2)
    assert seed==other_seed and a==b
    assert len(a)==2 and len(a[0]['events'])==cfg.horizon-4
    assert len(a[0]['uniforms'])==48-4
    state=initial_state(cfg)
    # Risk continuation requires no actor and is deterministic given the tape.
    root=catalogue().index(())
    _,full=bank(cfg,'reciprocal_scale2',73000000,73011000,0,1)
    one=root_cost(cfg,state,0,root,full[0],None,'risk',lambda:None)
    two=root_cost(cfg,state,0,root,full[0],None,'risk',lambda:None)
    assert one==two and one[2]==cfg.horizon


def test_bootstrap_constant_cube():
    np.testing.assert_allclose(bootstrap(np.full((5,16),7.0)),[7.0,7.0])
