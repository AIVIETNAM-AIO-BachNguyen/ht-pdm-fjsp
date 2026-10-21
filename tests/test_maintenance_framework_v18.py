import numpy as np
import torch

from ht_pdm_fjsp.maintenance_framework_v1 import configs,vector_observation
from ht_pdm_fjsp.maintenance_framework_v17_policy import ACTIONS
from ht_pdm_fjsp.maintenance_framework_v18_campaign import (
    actor_new,actor_digest,bounded_values,masked_probabilities,seed_registry,update_actor,
)
from ht_pdm_fjsp.maintenance_requests import initial_state


def test_bounded_continuation_disagreement_is_centered_and_clipped():
    own={0:100.,1:110.,2:125.}
    risk={0:200.,1:245.,2:180.}
    bounded,raw,correction=bounded_values(own,risk)
    own_mean=np.mean(list(own.values()));risk_mean=np.mean(list(risk.values()))
    for action in own:
        assert np.isclose(raw[action],risk[action]-risk_mean-own[action]+own_mean)
        assert np.isclose(bounded[action],own[action]+.5*np.clip(raw[action],-40,40))
        assert abs(correction[action])<=20
    assert any(abs(x)==20 for x in correction.values())
    shifted,_,_=bounded_values({a:v+25 for a,v in own.items()},
                               {a:v-90 for a,v in risk.items()})
    for action in own:assert np.isclose(shifted[action]-bounded[action],25)


def test_equal_actor_capacity_and_distinct_credit():
    torch.set_num_threads(1)
    cfg=configs()['reciprocal_scale2'];obs=vector_observation(cfg,initial_state(cfg),0)
    bounded=actor_new(73300000,'bounded_disagreement')
    control=actor_new(73300000,'own_rank')
    assert actor_digest(bounded)==actor_digest(control)
    assert sum(p.numel() for p in bounded.parameters())==9025
    p=masked_probabilities(bounded,obs,(0,1,2))
    assert np.isclose(p.sum(),1) and np.all(p[3:]==0)
    records=[dict(observation=obs,legal=(0,1,2),source_action=0) for _ in range(4)]
    own=[{0:10.,1:0.,2:5.} for _ in records]
    transformed=[{0:10.,1:20.,2:5.} for _ in records]
    for actor,arm in ((bounded,'bounded_disagreement'),(control,'own_rank')):
        info=update_actor(actor,records,own,transformed,arm,73300000,1,lambda:None,epochs=1)
        assert info['updates']>=1 and info['kl']<=.0200001
        assert np.isfinite(info['new_service_mean'])
    assert actor_digest(bounded)!=actor_digest(control)


def test_service_flow_term_has_gradient_for_both_arms():
    # Deliberately large service decline activates the same soft penalty in each arm.
    service=torch.tensor([len(a) for a in ACTIONS],dtype=torch.float32)

    for arm in ('bounded_disagreement','own_rank'):
        logits=torch.zeros(len(ACTIONS),requires_grad=True)
        new=(torch.softmax(logits,dim=0)*service).sum()
        old=new.detach()+.5
        penalty=10*torch.relu(old-new-.05).square()
        penalty.backward()
        assert penalty.item()>0 and torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum()>0


def test_seed_registry_disjoint_and_fixed():
    reg,groups=seed_registry()
    assert reg['campaign_version']=='maintenance_framework_v18_continuation_disagreement_dev_v1'
    assert list(map(len,groups))==[5,8,16,2]
