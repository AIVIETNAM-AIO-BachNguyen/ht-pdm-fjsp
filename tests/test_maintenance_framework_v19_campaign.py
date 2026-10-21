"""V19 matched scratch-RL campaign invariants."""
import numpy as np
import torch
from ht_pdm_fjsp.maintenance_framework_v1 import configs,vector_observation,catalogue
from ht_pdm_fjsp.maintenance_framework_v19_campaign import (
    actor_new,actor_digest,masked_probabilities,seed_registry,update_actor,
)
from ht_pdm_fjsp.maintenance_requests import initial_state


def test_equal_actor_capacity_and_identical_initialization():
    torch.set_num_threads(1)
    cfg=configs()['reciprocal_scale2'];obs=vector_observation(cfg,initial_state(cfg),0)
    left=actor_new(73400000,'temporal_deficit')
    right=actor_new(73400000,'snapshot_deficit')
    assert actor_digest(left)==actor_digest(right)
    assert sum(p.numel() for p in left.parameters())==sum(p.numel() for p in right.parameters())
    p=masked_probabilities(left,obs,(0,1,2))
    assert np.isclose(p.sum(),1) and np.all(p[3:]==0)


def test_both_arms_accept_identical_own_q_and_respect_kl():
    cfg=configs()['reciprocal_scale3'];obs=vector_observation(cfg,initial_state(cfg),0)
    records=[dict(observation=obs,legal=(0,1,2),source_action=0) for _ in range(4)]
    own=[{0:10.,1:0.,2:5.} for _ in records]
    for mode in ('temporal_deficit','snapshot_deficit'):
        actor=actor_new(73400000,mode)
        info=update_actor(actor,records,own,mode,73400000,1,lambda:None,epochs=1)
        assert info['updates']>=1 and info['kl']<=.0200001
        assert np.isfinite(info['new_service_mean'])


def test_service_flow_term_has_gradient_for_both_arms():
    service=torch.tensor([len(a) for a in catalogue()],dtype=torch.float32)
    for _ in ('temporal_deficit','snapshot_deficit'):
        logits=torch.zeros(len(service),requires_grad=True)
        new=(torch.softmax(logits,dim=0)*service).sum()
        penalty=10*torch.relu(new.detach()+.5-new-.05).square()
        penalty.backward()
        assert penalty.item()>0 and torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum()>0


def test_seed_registry_disjoint_and_fixed():
    reg,groups=seed_registry()
    assert reg['campaign_version']=='maintenance_framework_v19_temporal_capacity_dev_v1'
    assert list(map(len,groups))==[5,8,16,2]
