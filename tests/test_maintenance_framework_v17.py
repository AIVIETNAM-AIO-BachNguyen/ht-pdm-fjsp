import numpy as np
import torch

from ht_pdm_fjsp.maintenance_framework_v1 import configs, vector_observation
from ht_pdm_fjsp.maintenance_requests import initial_state
from ht_pdm_fjsp.maintenance_framework_v17_policy import ACTIONS, FullMatchingActor
from ht_pdm_fjsp.maintenance_framework_v17_campaign import (
    actor_new, actor_digest, masked_probabilities, seed_registry, update_actor,
)
from research_result.run_v16_postdecision_q_probe import features


def test_exact_pre_release_features_and_active_capacity():
    cfg=configs()['specialist_generalist_scale2']
    state=initial_state(cfg)
    obs=vector_observation(cfg,state,0)
    actor=FullMatchingActor()
    actual=actor.feature_tensor(torch.tensor(obs).reshape(1,56))[0].detach().numpy()
    assert sum(p.numel() for p in actor.parameters())==9025
    for action,matching in enumerate(ACTIONS):
        if any(cfg.service_time[m][j]==0 for m,j in matching):
            continue
        common,_,pre=features(cfg,state,matching)
        expected=np.concatenate((obs,common,pre))
        np.testing.assert_allclose(actual[action],expected,atol=1e-7)
    score=actor(torch.tensor(obs).reshape(1,56)).sum()
    score.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())


def test_paired_initialization_mask_and_distinct_credit_update():
    cfg=configs()['reciprocal_scale2'];state=initial_state(cfg)
    obs=vector_observation(cfg,state,0)
    rank=actor_new(73000000,'rank_projection')
    expected=actor_new(73000000,'expected_q')
    assert actor_digest(rank)==actor_digest(expected)
    probs=masked_probabilities(rank,obs,(0,1,2))
    assert np.isclose(probs.sum(),1) and np.all(probs[3:]==0)
    records=[dict(observation=obs,legal=(0,1,2),source_action=0) for _ in range(4)]
    q=[{0:10.,1:0.,2:5.} for _ in records]
    for actor,arm in ((rank,'rank_projection'),(expected,'expected_q')):
        info=update_actor(actor,records,q,arm,73000000,1,lambda:None,epochs=1)
        assert info['updates']>=1 and info['kl']<=.0200001
    assert actor_digest(rank)!=actor_digest(expected)


def test_v17_seed_registry_disjoint_from_prior():
    reg,groups=seed_registry()
    assert reg['campaign_version']=='maintenance_framework_v17_rank_projection_dev_v1'
    assert list(map(len,groups))==[5,8,16,2]
