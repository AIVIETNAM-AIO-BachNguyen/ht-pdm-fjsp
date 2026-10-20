"""V10 assignment contrast and matched all-action ablation."""
import numpy as np
import torch

from ht_pdm_fjsp import maintenance_framework_v10_campaign as v10


def test_assignment_groups_and_conditional_gain_ignore_subset_probability():
    wait=v10.ACTIONS.index(())
    worker0=v10.ACTIONS.index(((0,0),))
    worker1=v10.ACTIONS.index(((0,1),))
    other=v10.ACTIONS.index(((1,0),))
    groups=v10.assignment_groups((wait,worker0,worker1,other))
    assert groups==((worker0,worker1),)
    raw=torch.zeros((1,len(v10.ACTIONS)),requires_grad=True)
    q=torch.zeros_like(raw);q[0,worker0]=20.;q[0,worker1]=0.
    gain=v10.conditional_assignment_gain(raw,q,[groups])
    gain.backward()
    assert abs(float(gain.detach()))<1e-8
    assert raw.grad[0,worker1]>0
    assert raw.grad[0,worker0]<0
    assert raw.grad[0,wait]==0


def test_assignment_credit_changes_update_on_identical_bank():
    wait=v10.ACTIONS.index(())
    worker0=v10.ACTIONS.index(((0,0),))
    worker1=v10.ACTIONS.index(((0,1),))
    legal=(wait,worker0,worker1)
    record=dict(observation=np.zeros(56,dtype=np.float32),legal=legal,source_action=worker0)
    q={wait:30.,worker0:20.,worker1:0.}
    joint=v10.actor_new(70400000);assignment=v10.actor_new(70400000)
    assert v10.actor_digest(joint)==v10.actor_digest(assignment)
    no_timeout=lambda:None
    a=v10.update_actor(joint,[record],[q],'joint_credit',70400000,1,no_timeout,epochs=1)
    b=v10.update_actor(assignment,[record],[q],'assignment_credit',70400000,1,no_timeout,epochs=1)
    assert a['eligible_assignment_groups']==b['eligible_assignment_groups']==1
    assert a['updates']>=1 and b['updates']>=1
    assert v10.actor_digest(joint)!=v10.actor_digest(assignment)


def test_v10_seed_registry_disjoint_from_prior():
    reg,groups=v10.seed_registry()
    assert reg['campaign_version']==v10.VERSION
    assert tuple(groups[0])==tuple(range(70400000,70400005))
    assert not set(groups[0])&set(groups[1])
