import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_capacity_onpolicy_credit_diagnostic as d


def test_credit_trace_identity_terminal_and_lambda_zero():
    rng=np.random.default_rng(12);reward=rng.normal(size=(48,4));value=rng.normal(size=(48,4))
    returns,a=d.credit_arrays(reward,value,1.)
    np.testing.assert_allclose(a,returns-value,atol=1e-12)
    _,one=d.credit_arrays(reward,value,0.)
    np.testing.assert_allclose(one[:-1],reward[:-1]+value[1:]-value[:-1],atol=1e-12)
    np.testing.assert_allclose(one[-1],reward[-1]-value[-1],atol=1e-12)
    with pytest.raises(ValueError):d.credit_arrays(reward[:-1],value[:-1],.95)
    with pytest.raises(ValueError):d.credit_arrays(reward,value,1.1)


def test_statistics_accurate_value_and_crossfit_reference():
    reward=np.random.default_rng(1).normal(size=(48,64));returns=np.cumsum(reward[::-1],axis=0)[::-1]
    stats,arrays=d.credit_statistics(reward,returns)
    assert stats['critic_explained_variance']==1 and stats['prediction_rmse']==0
    assert stats['lambda1_max_identity_error']<1e-10
    np.testing.assert_allclose(arrays['time_reference'][:,::2],np.repeat(returns[:,1::2].mean(1,keepdims=True),32,axis=1))
    shifted,_=d.credit_statistics(reward,returns+2)
    assert shifted['critic_explained_variance']==1 and shifted['prediction_mean_error']==2
    assert shifted['critic_mse_skill']<1


def test_dispersion_and_gradient_cosine_math():
    g=np.array([[1.,0],[3.,0]])
    assert d.block_dispersion(g)['dispersion']==.25
    assert d.cosine([1,2],[2,4])==pytest.approx(1)
    assert d.cosine([0,0],[1,2])==0


@pytest.mark.parametrize('method',('residual_capacity_ppo','ppo','normalized_ppo'))
def test_stochastic_collection_reward_tail_replay_and_frozen_parameters(method):
    torch.set_num_threads(1);c=d.factored.settings('engineering')
    m=d.factored.make_model(method,4,c,c['train_seeds'][0]);before=d.factored.learner.parameter_digest(m)
    data=d.collect(m,method,[230700001,230700002],torch.Generator().manual_seed(44),lambda:None)
    batch,actions,logs,values,reward,episodes,decisions=data
    assert reward.shape==(48,2) and len(episodes)==2 and len(decisions)==96
    assert episodes[0]['physical_ticks']==60 and episodes[0]['invalid_assignments']==0
    with torch.no_grad():log,_,value=m.evaluate(batch,actions)
    torch.testing.assert_close(log,logs,atol=2e-5,rtol=0)
    torch.testing.assert_close(value,torch.tensor(values.reshape(-1)),atol=.002304,rtol=1e-6)
    assert before==d.factored.learner.parameter_digest(m) and all(p.grad is None for p in m.parameters())


def test_block_gradients_are_complete_frozen_and_use_episode_blocks():
    torch.set_num_threads(1);c=d.factored.settings('engineering');t=c['train_seeds'][0]
    m=d.factored.make_model('residual_capacity_ppo',4,c,t)
    obs=d.factored.mechanism_states('engineering')[0]['observation']
    batch=d.factored.learner.tensors([obs]*3072)
    with torch.no_grad():actions=m.sample(batch,torch.Generator().manual_seed(44))[0]
    before=d.factored.learner.parameter_digest(m)
    stats,arrays=d.credit_statistics(np.random.default_rng(1).normal(size=(48,64)),np.zeros((48,64)))
    result,vectors=d.gradient_diagnostic(m,batch,actions,arrays)
    assert vectors['mc'].shape==(4,11423) and vectors['gae'].shape==(4,11423)
    assert all(np.isfinite(v).all() for v in vectors.values())
    assert -1.00001<=result['mean_gradient_cosine']<=1.00001
    assert before==d.factored.learner.parameter_digest(m) and all(p.grad is None for p in m.parameters())


def test_every_candidate_replica_must_meet_all_eligibility_checks():
    rows=[dict(family=f,method='residual_capacity_ppo',train_seed=t,critic_explained_variance=.8,
        absolute_mean_error_in_return_sd=.02,gradient=dict(dispersion_ratio=.5,mean_gradient_cosine=.9,mc=dict(mean_gradient_squared_norm=1.)))
        for f in d.ARMS for t in range(3)]
    assert d.eligibility(rows)['preferred_followup']=='rsc'
    rows[0]['absolute_mean_error_in_return_sd']=.11
    assert d.eligibility(rows)['preferred_followup']=='factored'
    rows[-1]['gradient']['mean_gradient_cosine']=.79
    assert d.eligibility(rows)['preferred_followup'] is None
    with pytest.raises(ValueError):d.eligibility(rows[:-1])


def test_registered_seed_panels_disjoint_and_fully_derived():
    audit=d.seed_audit()
    assert audit['status']=='PASS' and audit['raw_unique']==67 and audit['derived_unique']==70
    assert audit['source_family_historical_derived_checked']>150000
