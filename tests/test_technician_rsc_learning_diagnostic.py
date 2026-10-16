import pytest
import torch
from ht_pdm_fjsp.technician_rsc_learning_diagnostic import inspect_policy
from ht_pdm_fjsp.technician_residual_capacity_experiment import mechanism_states,CORE_ARMS,learner
from ht_pdm_fjsp.technician_residual_capacity_policy import ResidualCapacityPolicy

@pytest.mark.parametrize('arm',CORE_ARMS)
def test_diagnostic_zero_initial_branch_no_mutation_or_grad_and_finite(arm):
    torch.set_num_threads(1);torch.manual_seed(43);m=ResidualCapacityPolicy(arm)
    obs=[r['observation'] for r in mechanism_states('development')]
    before=learner.parameter_digest(m);r=inspect_policy(m,obs,obs)
    assert r['same_weight_zero_message_tv']==0
    assert r['capacity_input_logodds_gradient_l2']==0
    assert r['capacity_embedding_machine12_l2']==0
    assert r['initial_corpus_mean_zero_message_tv']==0
    assert before==learner.parameter_digest(m) and m.algorithm==arm
    assert all(p.grad is None for p in m.parameters())
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in m.modules())


def test_detector_can_measure_active_message_path_and_saturated_bid():
    torch.set_num_threads(1);torch.manual_seed(43);m=ResidualCapacityPolicy('residual_capacity_ppo')
    torch.nn.init.normal_(m.capacity_encoder[-1].weight,std=.2)
    obs=[r['observation'] for r in mechanism_states('development')]
    r=inspect_policy(m,obs,obs)
    assert r['same_weight_zero_message_tv']>1e-6
    assert r['capacity_input_logodds_gradient_l2']>1e-6
    assert r['capacity_embedding_machine12_l2']>1e-6
    with torch.no_grad():m.bid[0].bias.fill_(30)
    saturated=inspect_policy(m,obs,obs)
    assert saturated['actor_tanh_saturation']['bid.1']==1
    assert saturated['capacity_input_logodds_gradient_l2']==0
