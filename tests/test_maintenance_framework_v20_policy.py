import torch
import pytest

from ht_pdm_fjsp.maintenance_framework_v14_policy import OpportunityCostActor
from ht_pdm_fjsp.maintenance_framework_v20_policy import (
    ACTIONS,WAIT,ActivationMarginActor,activation_targets,conditional_service_gain,
    factorized_probabilities,
)


def test_capacity_and_legal_probability_conservation():
    actor=ActivationMarginActor()
    control=OpportunityCostActor('count_calibrated')
    assert sum(p.numel() for p in actor.parameters())==sum(p.numel() for p in control.parameters())==3219
    logits=torch.randn(3,len(ACTIONS),requires_grad=True)
    legal=torch.zeros_like(logits,dtype=torch.bool)
    legal[:,WAIT]=True;legal[0,1:5]=True;legal[1,2]=True
    p,pwait,conditional=factorized_probabilities(logits,legal)
    assert torch.allclose(p.sum(dim=1),torch.ones(3))
    assert torch.all(p[~legal]==0)
    assert p[2,WAIT]==1 and conditional[2].sum()==0
    assert torch.allclose(p[:,WAIT],pwait)
    assert torch.allclose(conditional[:2].sum(dim=1),torch.ones(2))
    (-p[0,WAIT].log()).backward()
    assert torch.isfinite(logits.grad).all()


def test_wait_target_and_matching_gain_signs():
    q=torch.zeros((2,len(ACTIONS)))
    legal=torch.zeros_like(q,dtype=torch.bool)
    legal[:,WAIT]=True;legal[:,1]=True;legal[:,2]=True
    q[0,WAIT]=5;q[0,1]=15;q[0,2]=20
    q[1,WAIT]=20;q[1,1]=5;q[1,2]=15
    target,weight=activation_targets(q,legal)
    assert target[0]>.5 and target[1]<.5
    assert torch.allclose(weight,torch.tensor([1.,1.5]))
    conditional=torch.zeros_like(q);conditional[:,1]=1
    gain=conditional_service_gain(conditional,q,legal)
    assert gain[0]>0 and gain[1]>0


def test_invalid_wait_mask_rejected():
    logits=torch.zeros((1,len(ACTIONS)));legal=torch.zeros_like(logits,dtype=torch.bool)
    with pytest.raises(ValueError,match='WAIT'):factorized_probabilities(logits,legal)
