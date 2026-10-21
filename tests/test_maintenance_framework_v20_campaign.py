"""V20 matched scratch-RL budget and greedy-continuation invariants."""
import numpy as np
import torch
from ht_pdm_fjsp.maintenance_framework_v1 import configs,vector_observation
from ht_pdm_fjsp.maintenance_framework_v20_campaign import (
    ACTIONS,ARMS,actor_digest,actor_new,forecast_cost,legal_present,
    masked_probabilities,seed_registry,update_actor,
)
from ht_pdm_fjsp.maintenance_requests import initial_state


def test_same_active_actor_and_factorized_legal_probabilities():
    torch.set_num_threads(1)
    cfg=configs()['reciprocal_scale2'];obs=vector_observation(cfg,initial_state(cfg),0)
    left=actor_new(73700000,ARMS[0]);right=actor_new(73700000,ARMS[1])
    assert actor_digest(left)==actor_digest(right)
    assert sum(p.numel() for p in left.parameters())==sum(p.numel() for p in right.parameters())==3219
    p=masked_probabilities(left,obs,(0,1,2))
    assert np.isclose(p.sum(),1) and np.all(p[3:]==0) and np.all(p[:3]>=0)


def test_both_arms_use_identical_q_and_respect_kl():
    torch.set_num_threads(1)
    cfg=configs()['reciprocal_scale2'];obs=vector_observation(cfg,initial_state(cfg),0)
    records=[dict(observation=obs,legal=(0,1,2),source_action=1) for _ in range(4)]
    q=[{0:10.,1:25.,2:30.} for _ in records]
    for arm in ARMS:
        actor=actor_new(73700000,arm)
        info=update_actor(actor,records,q,arm,73700000,1,lambda:None,epochs=1)
        assert info['updates']>=1 and info['kl']<=.0200001
        assert np.isfinite(info['new_wait_mean'])
        assert info['wait_target_mean']>.5


def test_forecast_continuation_is_greedy_and_ignores_uniform_tape():
    torch.set_num_threads(1)
    cfg=configs()['interchangeable_scale1'];state=initial_state(cfg)
    _,_,legal,obs=legal_present(cfg,state,0)
    actor=actor_new(73700000,ARMS[0]).eval()
    record=dict(original_state=state,tick=0)
    events=tuple(tuple(False for _ in range(cfg.machines)) for _ in range(cfg.horizon))
    value1,n1=forecast_cost(actor,cfg,record,legal[0],events,tuple(0. for _ in range(48)),lambda:None,'own')
    value2,n2=forecast_cost(actor,cfg,record,legal[0],events,tuple(.99 for _ in range(48)),lambda:None,'own')
    assert n1==n2==cfg.horizon and value1==value2


def test_seed_registry_new_roles_and_generated_training_seeds():
    reg,groups=seed_registry()
    assert reg['campaign_version']=='maintenance_framework_v20_activation_margin_dev_v1'
    assert list(map(len,groups))==[5,8,16,2]
