"""Core V9 paired-counterfactual credit and source contract tests."""
from dataclasses import replace

import numpy as np
import torch

from ht_pdm_fjsp.maintenance_framework_v1 import configs, catalogue
from ht_pdm_fjsp.maintenance_framework_v9_campaign import (
    ACTIONS, actor_digest, actor_new, clock_bin, counterfactual_values,
    episode_seed, legal_present, masked_probabilities, scenario_bank,
    seed_registry, state_rank, update_actor,
)
from ht_pdm_fjsp.maintenance_requests import initial_state


def test_seed_roles_rank_and_actor_start_are_deterministic():
    registry, panels = seed_registry()
    assert [len(panel) for panel in panels] == [5,16,2]
    assert len(set(panels[0]) & set(panels[1])) == 0
    assert actor_digest(actor_new(70300000)) == actor_digest(actor_new(70300000))
    assert episode_seed(70300000,1,'reciprocal_scale2',0) != episode_seed(
        70300000,1,'reciprocal_scale2',1)
    assert state_rank(1,2,'case',3,4) != state_rank(1,2,'case',3,5)
    assert [clock_bin(t) for t in (0,15,16,31,32,47)] == [0,0,1,1,2,2]
    assert registry['forecast_seed'] not in panels[1]


def test_common_bank_is_deterministic_and_does_not_use_source_tape():
    record=dict(case='reciprocal_scale2',physical_seed=123,tick=11)
    seed,bank=scenario_bank(70320000,70300000,1,record,2)
    assert (seed,bank)==scenario_bank(70320000,70300000,1,record,2)
    assert len(bank)==2
    assert len(bank[0][0])==configs()['reciprocal_scale2'].horizon-11
    assert len(bank[0][1])==48-11
    assert bank!=scenario_bank(70320000,70300000,1,{**record,'tick':12},2)[1]


def test_all_action_update_credits_better_legal_matching_and_stays_in_kl_region():
    actor=actor_new(70300000)
    obs=np.zeros(56,np.float32)
    legal=(0,1)
    records=[dict(observation=obs,legal=legal,source_action=0) for _ in range(64)]
    values=[{0:20.0,1:10.0} for _ in records]
    before=masked_probabilities(actor,obs,legal)[1]
    result=update_actor(actor,records,values,'all_action',70300000,1,lambda:None)
    after=masked_probabilities(actor,obs,legal)[1]
    assert result['updates']>0
    assert result['kl']<=.020001
    assert after>before
    assert len(ACTIONS)==len(catalogue())==21


def test_source_present_state_uses_threshold_admission_and_legal_mask():
    cfg=configs()['reciprocal_scale2']
    state=initial_state(cfg)
    submissions,admitted,legal,obs=legal_present(cfg,state,0)
    assert submissions==()
    assert legal==(0,)
    assert obs.shape==(56,)
    assert admitted==state


def test_fixed_training_physical_seeds_have_no_dev_or_smoke_overlap():
    _,panels=seed_registry()
    seeds=[episode_seed(train,iteration,case,index) for train in panels[0]
           for iteration in range(1,7) for case in configs() for index in range(16)]
    assert len(seeds)==len(set(seeds))==5*6*4*16
    assert not set(seeds)&(set(panels[1])|set(panels[2]))
