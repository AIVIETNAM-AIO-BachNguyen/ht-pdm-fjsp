import numpy as np
import pytest
import torch
from ht_pdm_fjsp.technician_capacity_service_choice_diagnostic import (
    core_trace,under_prefix,executed_pairs,ppo_observation,slot_description,summarize,METHODS)
from ht_pdm_fjsp.technician_capacity_factorized_policy import FactorizedCapacityPolicy
from ht_pdm_fjsp.technician_capacity_factorized_experiment import mechanism_states,learner


def observations():return mechanism_states('engineering')[0]['observation']

@pytest.mark.parametrize('arm',METHODS[:3])
def test_instrumentation_preserves_actions_parameters_and_restores_methods(arm):
    torch.set_num_threads(1);torch.manual_seed(19);m=FactorizedCapacityPolicy(arm);obs=observations()
    initial=learner.parameter_digest(m);before=m.select(obs);a,traces=core_trace(m,obs)
    assert np.array_equal(a,before) and len(traces)==2 and 'conditionals' not in m.__dict__
    assert learner.parameter_digest(m)==initial and all(p.grad is None for p in m.parameters())
    assert slot_description(traces[1],traces[1]['log'],traces[1]['support'])['legal_count']==2


def test_matched_prefix_recomputes_target_encoding_and_preserves_support():
    torch.set_num_threads(1);m=FactorizedCapacityPolicy('residual_capacity_ppo');other=FactorizedCapacityPolicy('flat_capacity_ppo')
    obs=observations();_,traces=core_trace(m,obs);context=traces[1]
    calls=[];hook=other.encoder.register_forward_hook(lambda *args:calls.append(1))
    try:log,support=under_prefix(other,context)
    finally:hook.remove()
    assert len(calls)==1 and torch.equal(support,context['support'])
    assert abs(float(log.exp().sum())-1)<1e-6


def test_independent_arbitration_uses_quality_then_index_and_baseline_masks():
    obs=observations();assert executed_pairs(obs,np.array([1,1,0]),'independent_capacity_ppo')==[(0,0)]
    with pytest.raises(ValueError):executed_pairs(obs,np.array([1,1,0]),'residual_capacity_ppo')
    with pytest.raises(ValueError):executed_pairs(obs,np.array([4,0,0]),'residual_capacity_ppo')
    assert ppo_observation(obs)['joint_mask'].sum()==7


def test_eligibility_thresholds_and_missing_contexts():
    own=[dict(method=a,train_seed=1,state_index=i,starts=1,long_starts=int(a=='residual_capacity_ppo'),service_ticks_committed=2 if a=='residual_capacity_ppo' else 1) for a in METHODS for i in range(12)]
    matched=[dict(method=a,train_seed=1,state_index=i,min_legal_duration=2,go_mode=int(a=='residual_capacity_ppo'),probability_go=.9 if a=='residual_capacity_ppo' else .1) for a in METHODS[:2] for i in range(12)]
    assert summarize(own,matched)['legal_option_followup_motivated']
    assert not summarize(own,[r for r in matched if r['state_index']<11])['legal_option_followup_motivated']
    with pytest.raises(ValueError):summarize(own,matched[:-1])
