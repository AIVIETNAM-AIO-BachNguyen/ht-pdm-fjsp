"""Mechanism mathematics, likelihood, information and physical coordination."""
import itertools
import numpy as np
import pytest
import torch
from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors
from ht_pdm_fjsp.technician_completion_policy import initial_calendar
from ht_pdm_fjsp.technician_residual_capacity_policy import (
    ResidualCapacityPolicy, CORE_ARMS, capacity_summaries)
from ht_pdm_fjsp.technician_residual_capacity_env import ResourceEnv
from ht_pdm_fjsp.technician_residual_capacity_experiment import mechanism_states


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def model(arm=CORE_ARMS[0],active=False):
    with torch.random.fork_rng():
        torch.manual_seed(101)
        m=ResidualCapacityPolicy(arm)
        if active:torch.nn.init.normal_(m.capacity_encoder[-1].weight,std=.1)
        return m


def batch():
    return tensors([mechanism_states('engineering')[0]['observation']])


def first_log(m,b,reserved=None):
    cal,planned=initial_calendar(b)
    return m.conditionals(b,m.encoder(m.raw_pairs(b)),cal,planned,
        torch.zeros(len(cal),dtype=torch.long),0,
        torch.zeros((len(cal),4),dtype=torch.bool) if reserved is None else reserved)[0]


def test_hand_computed_capacity_conservation_and_no_later_roles():
    b=batch();reserved=torch.zeros((1,4),dtype=torch.bool)
    true,static=capacity_summaries(b,reserved,torch.tensor([0]))
    torch.testing.assert_close(true,torch.tensor([[[0.,1.,.5],[0.,0.,.5],[0.,0.,0.],[0.,0.,0.]]]))
    torch.testing.assert_close(static.sum(1),torch.tensor([[1.,2.,2.]]))
    assert not capacity_summaries(b,reserved,torch.tensor([1]))[0].any()
    reserved[0,0]=True
    changed,_=capacity_summaries(b,reserved,torch.tensor([0]))
    torch.testing.assert_close(changed,torch.tensor([[[0.,0.,0.],[0.,0.,1.],[0.,0.,0.],[0.,0.,0.]]]))


def test_busy_technician_can_contribute_when_service_fits_but_busy_machine_cannot():
    b=batch();b['technicians'][0,1,0]=0;b['technicians'][0,1,1]=1/12
    # Later busy technician needs1 tick before servicing M1(2 ticks), M2(4).
    true,_=capacity_summaries(b,torch.zeros((1,4),dtype=torch.bool),torch.tensor([0]))
    torch.testing.assert_close(true[0],torch.tensor([[0.,0.,1.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]))
    b['machines'][0,0,3]=1
    assert not capacity_summaries(b,torch.zeros((1,4),dtype=torch.bool),torch.tensor([0]))[0].any()


def test_integer_tick_rounding_and_irrelevance_of_unobserved_metadata():
    b=batch();a=capacity_summaries(b,torch.zeros((1,4),dtype=torch.bool),torch.tensor([0]))[0]
    b['edges'][...,1]+=1e-8;b['unobserved_future_events']=torch.ones(1,60,4)*999
    torch.testing.assert_close(a,capacity_summaries(b,torch.zeros((1,4),dtype=torch.bool),torch.tensor([0]))[0])


@pytest.mark.parametrize('arm',CORE_ARMS)
def test_sample_replay_scalar_critic_capacity_and_conditional_call_budget(arm,monkeypatch):
    m=model(arm,True);b={k:v.repeat(16,*([1]*(v.ndim-1))) for k,v in batch().items()}
    count=[];orig=m.conditionals
    def call(*args,**kwargs):count.append(1);return orig(*args,**kwargs)
    monkeypatch.setattr(m,'conditionals',call)
    a,log,v=m.sample(b,generator=torch.Generator().manual_seed(19))
    assert len(count)==2 and not a[:,-1].any()
    count.clear();new,entropy,newv=m.evaluate(b,a)
    assert len(count)==2 and log.shape==entropy.shape==v.shape==(16,)
    torch.testing.assert_close(log,new,atol=2e-5,rtol=0)
    torch.testing.assert_close(v,newv)
    count.clear();m.greedy(b);assert len(count)==2
    groups=m.gradient_groups()
    assert sum(p.numel() for p in groups['actor'])==11423
    assert sum(p.numel() for p in groups['critic'])==8587
    assert sum(p.numel() for p in m.parameters())==20010
    m.evaluate(b,a)[0].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in groups['actor'])


@pytest.mark.parametrize('arm',CORE_ARMS)
def test_joint_proposal_distribution_normalizes(arm):
    m=model(arm,True);b=batch();b['pair_mask'][:]=True
    actions=[a for a in itertools.product(range(5),repeat=2)
        if arm=='independent_capacity_ppo' or len([x for x in a if x])==len(set(x for x in a if x))]
    log=torch.stack([m.evaluate(b,torch.tensor([[*a,0]]))[0][0] for a in actions])
    assert len(actions)==(25 if arm=='independent_capacity_ppo' else 21)
    torch.testing.assert_close(log.exp().sum(),torch.tensor(1.),atol=2e-6,rtol=0)


def test_same_initial_tensors_and_probabilities_but_distinct_active_messages():
    models=[model(a) for a in CORE_ARMS]
    for m in models[1:]:
        assert all(torch.equal(a,b) for a,b in zip(models[0].parameters(),m.parameters()))
        torch.testing.assert_close(first_log(models[0],batch()),first_log(m,batch()),atol=0,rtol=0)
    assert not torch.equal(first_log(model(active=True),batch()),first_log(model('no_capacity_ppo',True),batch()))
    assert not torch.equal(first_log(model(active=True),batch()),first_log(model('initial_capacity_ppo',True),batch()))
    assert not torch.equal(first_log(model(active=True),batch()),first_log(model('permuted_capacity_ppo',True),batch()))


@pytest.mark.parametrize('arm',CORE_ARMS)
def test_forced_wait_and_fixed_auxiliary_validation(arm):
    m=model(arm);b=batch();b['pair_mask'][:]=False
    a,log,_=m.sample(b,generator=torch.Generator().manual_seed(5))
    assert not a.any() and float(log[0])==0 and float(m.evaluate(b,a)[1][0].detach())==0
    a[:,-1]=1
    with pytest.raises(ValueError,match='fixed zero'):m.evaluate(b,a)


def test_independent_removes_calendar_and_reservation_but_true_retains_them():
    b=batch();a=torch.tensor([[1,1,0]])
    with pytest.raises(ValueError,match='duplicate'):model().evaluate(b,a)
    assert torch.isfinite(model('independent_capacity_ppo').evaluate(b,a)[0]).all()
    cal,planned=initial_calendar(b);reserved=torch.zeros((1,4),dtype=torch.bool)
    m=model('independent_capacity_ppo',True);base=m.encoder(m.raw_pairs(b))
    before=m.conditionals(b,base,cal,planned,torch.ones(1,dtype=torch.long),1,reserved)[0]
    cal[:]=999;planned[:]=1;reserved[:]=True
    after=m.conditionals(b,base,cal,planned,torch.ones(1,dtype=torch.long),1,reserved)[0]
    torch.testing.assert_close(before,after,atol=0,rtol=0)


def test_arbitration_reports_conflicts_without_retry_and_rejects_invalids():
    env=ResourceEnv(220700001,4,'independent_capacity_ppo');env.reset()
    from ht_pdm_fjsp.maintenance_dispatch import DispatchState
    env.env.state=DispatchState((3,3,0,0),(False,)*4,(0,)*4,(0,0),(-1,-1))
    pairs,diag=env.resolve(np.array([1,1,0]))
    # Both quality scores .5, lower technician index wins.
    assert pairs==((0,0),) and diag['conflicts']==diag['rejected_proposals']==1
    assert diag['executed_starts']==1 and diag['order_forward']==1
    for bad in ([1,2,1],[1.,2.,0.],[5,0,0],[-1,0,0],[1,2]):
        with pytest.raises(ValueError):env.resolve(np.asarray(bad))
    candidate=ResourceEnv(220700001,4,'residual_capacity_ppo');candidate.reset();candidate.env.state=env.env.state
    with pytest.raises(ValueError,match='duplicate'):candidate.resolve(np.array([1,1,0]))
    with pytest.raises(ValueError,match='N4/K2'):ResourceEnv(220700001,8,'residual_capacity_ppo')


def test_candidate_machine_permutation_equivariance_with_active_capacity():
    m=model(active=True);b=batch();permutation=torch.tensor([2,0,3,1]);inverse=torch.argsort(permutation)
    changed={k:v.clone() for k,v in b.items()}
    for key in ('machines','edges','pair_mask'):changed[key]=b[key][:,permutation]
    a=torch.tensor([[1,2,0]]);translated=torch.tensor([[inverse[0]+1,inverse[1]+1,0]])
    torch.testing.assert_close(m.evaluate(b,a)[0],m.evaluate(changed,translated)[0],atol=2e-6,rtol=0)
    torch.testing.assert_close(m.physical_value(b),m.physical_value(changed))


def test_probe_states_valid_disjoint_and_no_physical_execution():
    dev,cf=mechanism_states('development'),mechanism_states('confirmation')
    assert len(dev)==len(cf)==16 and {r['tick'] for r in dev}.isdisjoint({r['tick'] for r in cf})
    for r in [*dev,*cf]:
        assert r['observation']['pair_mask'].sum()==4
        assert r['state']['ages'][0]==r['state']['ages'][1]
