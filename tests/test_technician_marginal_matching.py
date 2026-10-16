"""Physical probability must be the exact mixture, with feasible canonical paths."""
import itertools
import math
import numpy as np
import pytest
import torch

from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors
from ht_pdm_fjsp.technician_marginal_matching_env import CORE_ARMS, ResourceEnv, decode_matching
from ht_pdm_fjsp.technician_marginal_matching_policy import MarginalMatchingPolicy


def model(arm='marginal_matching_ppo'):
    with torch.random.fork_rng():
        torch.manual_seed(101)
        return MarginalMatchingPolicy(arm)


def dense():
    obs = ResourceEnv(200000000, 4, 'marginal_matching_ppo').reset()
    obs['pair_mask'][:] = True
    obs['machines'][:, 3:5] = 0
    obs['edges'][..., 3] = 0
    obs['technicians'][:, 0] = 1
    return tensors([obs])


def matchings():
    return [a for a in itertools.product(range(5), repeat=2)
            if len([v for v in a if v]) == len(set(v for v in a if v))]


@pytest.mark.parametrize('arm', CORE_ARMS)
def test_physical_distribution_and_both_components_normalize_on_21_matchings(arm):
    m,b=model(arm),dense()
    table=torch.stack([m.components(b,torch.tensor([a]))[0][0] for a in matchings()])
    assert len(table)==21
    torch.testing.assert_close(table.exp().sum(0),torch.ones(2),atol=2e-6,rtol=0)
    torch.testing.assert_close(m.physical_log(table).exp().sum(),torch.tensor(1.),atol=2e-6,rtol=0)


@pytest.mark.parametrize('arm', CORE_ARMS)
def test_sample_replay_scalar_value_and_parameter_counts(arm):
    m=model(arm);b={k:v.repeat(16,*([1]*(v.ndim-1))) for k,v in dense().items()}
    a,old,v=m.sample(b,generator=torch.Generator().manual_seed(19));new,entropy,newv=m.evaluate(b,a)
    assert a.shape==(16,3) and old.shape==new.shape==v.shape==entropy.shape==(16,)
    torch.testing.assert_close(old,new,atol=2e-5,rtol=0)
    torch.testing.assert_close(v,newv,atol=2e-5*115.2,rtol=1e-6)
    groups=m.gradient_groups()
    assert sum(p.numel() for p in groups['actor'])==9803
    assert sum(p.numel() for p in groups['critic'])==8587
    assert sum(p.numel() for p in m.parameters())==18390
    for row in a:
        decode_matching(np.ones((4,2),bool),row.numpy())


def test_primary_ablation_shares_physical_sampling_deployment_but_uses_correct_scores():
    marginal,latent=model(),model('latent_order_ppo');b=dense()
    a,log,_=marginal.sample(b,generator=torch.Generator().manual_seed(12))
    other,latentlog,_=latent.sample(b,generator=torch.Generator().manual_seed(12))
    assert torch.equal(a,other)
    components=marginal.components(b,a[:,:-1])[0]
    torch.testing.assert_close(log,torch.logsumexp(components,1)-math.log(2))
    torch.testing.assert_close(latentlog,components.gather(1,a[:,-1:]).squeeze(1)-math.log(2))
    assert torch.equal(marginal.greedy(b),latent.greedy(b))


def test_marginal_gradient_is_posterior_weighted_component_gradient():
    m=model();logs=m.components(dense(),torch.tensor([[1,2]]))[0]
    actor=m.gradient_groups()['actor']
    g=torch.autograd.grad((torch.logsumexp(logs,1)-math.log(2)).sum(),actor,retain_graph=True)
    expected=torch.autograd.grad((logs.softmax(1).detach()*logs).sum(),actor)
    for x,y in zip(g,expected):
        torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('arm', CORE_ARMS)
def test_forced_wait_and_equal_core_slot_compute(arm,monkeypatch):
    m=model(arm);b=dense();b['pair_mask'][:]=False
    calls=[];original=m.conditionals
    def counted(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(m,'conditionals',counted)
    action,log,_=m.sample(b,generator=torch.Generator().manual_seed(5))
    assert len(calls)==6 and not action[:,:-1].any()
    assert float(log[0])==pytest.approx(-math.log(2) if arm=='latent_order_ppo' else 0.)
    calls.clear();_,entropy,_=m.evaluate(b,action)
    assert len(calls)==4 and float(entropy.detach()[0])==0
    calls.clear();m.greedy(b)
    assert len(calls)==12


def test_no_prefix_removes_calendar_messages_but_not_reservation_masks():
    captured={}
    for arm in ('marginal_matching_ppo','no_prefix_ppo'):
        m=model(arm);raw=[]
        hook=m.calendar_encoder.register_forward_pre_hook(lambda module,args:raw.append(args[0].detach().clone()))
        m.components(dense(),torch.tensor([[1,2]]));hook.remove();captured[arm]=raw
        with pytest.raises(ValueError,match='duplicate'):
            m.components(dense(),torch.tensor([[1,1]]))
    assert all(torch.equal(captured['no_prefix_ppo'][0],x) for x in captured['no_prefix_ppo'])
    assert not torch.equal(captured['marginal_matching_ppo'][0],captured['marginal_matching_ppo'][1])


def test_machine_and_two_technician_swap_equivariance_of_physical_likelihood():
    m=model();b=dense();a=torch.tensor([[1,2]])
    original=m.physical_log(m.components(b,a)[0]);v=m.physical_value(b)
    for machines,techs in ((torch.tensor([2,0,3,1]),torch.tensor([0,1])),
                            (torch.arange(4),torch.tensor([1,0]))):
        changed={k:x.clone() for k,x in b.items()}
        changed['machines']=b['machines'][:,machines]
        changed['technicians']=b['technicians'][:,techs]
        changed['edges']=b['edges'][:,machines][:,:,techs]
        changed['pair_mask']=b['pair_mask'][:,machines][:,:,techs]
        inverse=torch.argsort(machines)
        translated=inverse[(a-1).clamp(min=0)]+1
        translated=torch.where(a>0,translated,0)[:,techs]
        torch.testing.assert_close(m.physical_log(m.components(changed,translated)[0]),original,atol=2e-6,rtol=0)
        torch.testing.assert_close(m.physical_value(changed),v,atol=2e-5*115.2,rtol=1e-6)


def test_sample_frequencies_match_the_marginal_not_one_component():
    m=model();b=dense();all_actions=matchings()
    with torch.no_grad():
        probabilities=torch.stack([m.physical_log(m.components(b,torch.tensor([a]))[0])[0] for a in all_actions]).exp().numpy()
    big={k:v.repeat(4000,*([1]*(v.ndim-1))) for k,v in b.items()}
    a=m.sample(big,generator=torch.Generator().manual_seed(110))[0][:,:-1].numpy()
    empirical=np.array([np.mean((a==row).all(1)) for row in all_actions])
    assert np.max(np.abs(empirical-probabilities))<.025
    assert np.abs(empirical-probabilities).sum()/2<.07


def test_auxiliary_order_does_not_change_physical_transition():
    envs=[ResourceEnv(200000000,4,'marginal_matching_ppo') for _ in range(2)]
    obs=[e.reset() for e in envs];chosen=model().select(obs[0]);results=[]
    for o,e in enumerate(envs):
        action=chosen.copy();action[-1]=o;results.append(e.step(action))
    assert envs[0].env.state==envs[1].env.state
    assert results[0][1:3]==results[1][1:3]


@pytest.mark.parametrize('submitted', [[1,1,0],[1,2,2],[5,0,0],[-1,0,0],[1.,2.,0.],[1,2]])
def test_decoder_rejects_invalid_vectors(submitted):
    with pytest.raises(ValueError):
        decode_matching(np.ones((4,2),bool),submitted)


def test_experiment_environment_rejects_larger_sizes():
    with pytest.raises(ValueError,match='N4/K2'):
        ResourceEnv(200000000,8,'marginal_matching_ppo')
