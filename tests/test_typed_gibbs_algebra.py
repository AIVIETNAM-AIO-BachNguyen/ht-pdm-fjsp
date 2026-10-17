import itertools,math
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_typed_gibbs as g


def enumerate_matchings(n,types,available,eligible):
    result=[]
    for action in itertools.product(range(len(types)+1),repeat=n):
        used=[a-1 for a in action if a]
        if len(used)!=len(set(used)) or any(not available[j] for j in used):continue
        if any(a and not eligible[i,types[a-1]] for i,a in enumerate(action)):continue
        result.append(action)
    return torch.tensor(result,dtype=torch.long)


def reference(utility,types,actions):
    selected=[]
    for action in actions:
        selected.append(sum((utility[i,types[a-1]] for i,a in enumerate(action.tolist()) if a),utility.new_zeros(())))
    energies=torch.stack(selected);logz=torch.logsumexp(energies,0);logp=energies-logz;entropy=-(logp.exp()*logp).sum();return logz,logp,entropy


@pytest.mark.parametrize('n,types',[(2,[0,1]),(3,[0,0,1]),(4,[0,0,1,1]),(5,[0,0,1,1])])
def test_every_labeled_density_partition_entropy_full_gradients_and_physical_map(n,types):
    torch.set_num_threads(1);types=torch.tensor(types);caps=tuple(int((types==t).sum()) for t in (0,1));utility=((torch.arange(n*2,dtype=torch.float64).reshape(n,2)*.37).sin()+.1).requires_grad_();eligible=torch.ones((n,2),dtype=torch.bool);available=torch.ones(len(types),dtype=torch.bool);actions=enumerate_matchings(n,types,available,eligible);table=g.policy_tables(utility[None],caps,eligible[None]);root=g.root_values(table,types,available[None]);z,lp,h=reference(utility,types,actions);torch.testing.assert_close(root['log_partition'][0],z,atol=1e-10,rtol=0);torch.testing.assert_close(root['entropy'][0],h,atol=1e-10,rtol=0)
    # Batched same utilities for ALL physical outcomes; no source count reference.
    many=g.policy_tables(utility[None].expand(len(actions),-1,-1),caps,eligible[None].expand(len(actions),-1,-1));actual=g.matching_log_probability(many,types,available[None].expand(len(actions),-1),actions);torch.testing.assert_close(actual,lp,atol=1e-10,rtol=0);assert float(actual.exp().sum().detach())==pytest.approx(1.,abs=1e-10)
    for actual_value,expected_value in ((root['log_partition'][0],z),(root['entropy'][0],h),(actual[len(actions)//2],lp[len(actions)//2])):
        a=torch.autograd.grad(actual_value,utility,retain_graph=True)[0];b=torch.autograd.grad(expected_value,utility,retain_graph=True)[0];torch.testing.assert_close(a,b,atol=1e-9,rtol=0)
    chosen=g.maximum_matching(table,types,available[None]);selected=g.matching_log_probability(table,types,available[None],chosen)[0];torch.testing.assert_close(selected,lp.max(),atol=1e-10,rtol=0);torch.testing.assert_close(root['max_log_probability'][0],lp.max(),atol=1e-10,rtol=0)


@pytest.mark.parametrize('mode',['gibbs','myopic'])
def test_unavailable_workers_type_masks_and_normalized_myopic_entropy_map(mode):
    torch.set_num_threads(1);types=torch.tensor([0,0,1,1]);available=torch.tensor([False,True,False,True]);utility=torch.tensor([[.7,-.5],[.4,1.2],[-.8,.9]],dtype=torch.float64,requires_grad=True);eligible=torch.tensor([[True,False],[True,True],[False,True]]);actions=enumerate_matchings(3,types,available,eligible);table=g.policy_tables(utility[None],(2,2),eligible[None],mode);root=g.root_values(table,types,available[None]);many=g.policy_tables(utility[None].expand(len(actions),-1,-1),(2,2),eligible[None].expand(len(actions),-1,-1),mode);lp=g.matching_log_probability(many,types,available[None].expand(len(actions),-1),actions);assert float(lp.exp().sum().detach())==pytest.approx(1.,abs=1e-10);entropy=-(lp.exp()*lp).sum();torch.testing.assert_close(root['entropy'][0],entropy,atol=1e-10,rtol=0)
    expected=torch.autograd.grad(entropy,utility,retain_graph=True)[0];actual=torch.autograd.grad(root['entropy'].sum(),utility)[0];assert torch.isfinite(actual).all();torch.testing.assert_close(actual,expected,atol=1e-9,rtol=0);chosen=g.maximum_matching(table,types,available[None]);torch.testing.assert_close(g.matching_log_probability(table,types,available[None],chosen)[0],lp.max(),atol=1e-10,rtol=0)
    with pytest.raises(ValueError):g.matching_log_probability(table,types,available[None],torch.tensor([[1,0,0]]))


def test_all24_orders_same_gibbs_policy_and_nontrivial_myopic_contrast():
    torch.set_num_threads(1);types=torch.tensor([0,1]);available=torch.ones((1,2),dtype=torch.bool);u=torch.tensor([[1.2,-.8],[-.7,1.8],[.4,.3],[.2,-.2]],dtype=torch.float64);eligible=torch.ones((4,2),dtype=torch.bool);actions=enumerate_matchings(4,types,available[0],eligible);canonical=g.policy_tables(u[None].expand(21,-1,-1),(1,1));lp=g.matching_log_probability(canonical,types,available.expand(21,-1),actions)
    for permutation in itertools.permutations(range(4)):
        ix=torch.tensor(permutation);table=g.policy_tables(u[ix][None].expand(21,-1,-1),(1,1));actual=g.matching_log_probability(table,types,available.expand(21,-1),actions[:,ix]);torch.testing.assert_close(actual,lp,atol=1e-10,rtol=0)
    forward=g.policy_tables(u[None].expand(21,-1,-1),(1,1),mode='myopic');reverse=g.policy_tables(u.flip(0)[None].expand(21,-1,-1),(1,1),mode='myopic');a=g.matching_log_probability(forward,types,available.expand(21,-1),actions);b=g.matching_log_probability(reverse,types,available.expand(21,-1),actions.flip(1));assert float((a.exp()-b.exp()).abs().sum()/2)>1e-3


def test_zero_utilities_identical_uniform21_and_group_multiplicity_not_physical_map():
    torch.set_num_threads(1);types=torch.tensor([0,1]);available=torch.ones((21,2),dtype=torch.bool);eligible=torch.ones((4,2),dtype=torch.bool);actions=enumerate_matchings(4,types,available[0],eligible)
    for mode in ('gibbs','myopic'):
        table=g.policy_tables(torch.zeros((21,4,2),dtype=torch.float64),(1,1),mode=mode);torch.testing.assert_close(g.matching_log_probability(table,types,available,actions),torch.full((21,),-math.log(21),dtype=torch.float64),atol=1e-10,rtol=0)
    types=torch.tensor([0,0,1]);available=torch.ones((1,3),dtype=torch.bool);table=g.policy_tables(torch.tensor([[[.1,.2]]],dtype=torch.float64),(2,1));assert g.maximum_matching(table,types,available).tolist()==[[3]]


def test_empty_capacities_extreme_utilities_and_invalid_contracts():
    torch.set_num_threads(1)
    for mode in ('gibbs','myopic'):
        utility=torch.tensor([[[1000.,-1000.],[-900.,800.]]],dtype=torch.float64,requires_grad=True);table=g.policy_tables(utility,(1,1),mode=mode);root=g.root_values(table,torch.tensor([0,1]),torch.ones((1,2),dtype=torch.bool));assert all(torch.isfinite(v).all() for v in root.values());grad=torch.autograd.grad(root['entropy'].sum()+root['log_partition'].sum(),utility)[0];assert torch.isfinite(grad).all()
        table=g.policy_tables(torch.zeros((1,2,2),dtype=torch.float64),(0,0),mode=mode);root=g.root_values(table,torch.empty(0,dtype=torch.long),torch.empty((1,0),dtype=torch.bool));assert all(v.item()==0 for v in root.values());assert g.maximum_matching(table,torch.empty(0,dtype=torch.long),torch.empty((1,0),dtype=torch.bool)).tolist()==[[0,0]]
    with pytest.raises(ValueError):g.policy_tables(torch.tensor([[[float('nan'),0.]]]),(1,1))
    with pytest.raises(ValueError):g.policy_tables(torch.zeros((1,2,3)),(1,1))


@pytest.mark.parametrize('mode',['gibbs','myopic'])
def test_ppo_ratio_full_gradient_against_independent_conditional_enumeration(mode):
    # The reference sums explicit legal suffix completions, without production DP.
    types=torch.tensor([0,0,1]);available=torch.tensor([True,True,True]);eligible=torch.tensor([[True,True],[True,False],[True,True]])
    actions=enumerate_matchings(3,types,available,eligible)
    utility=torch.tensor([[.35,-.8],[.6,.4],[-.2,1.1]],dtype=torch.float64,requires_grad=True)
    def explicit(u):
        if mode=='gibbs':return reference(u,types,actions)[1]
        result=[]
        for action in actions.tolist():
            free=set(range(3));lp=u.new_zeros(())
            for i,token in enumerate(action):
                weights=[];tokens=[0]+[j+1 for j in sorted(free) if eligible[i,types[j]]]
                for choice in tokens:
                    following=free-({choice-1} if choice else set())
                    count=len(enumerate_matchings(3-i-1,types,torch.tensor([j in following for j in range(3)]),eligible[i+1:]))
                    weights.append(math.log(count)+(u[i,types[choice-1]] if choice else u.new_zeros(())))
                logits=torch.stack(weights);lp=lp+logits[tokens.index(token)]-torch.logsumexp(logits,0)
                if token:free.remove(token-1)
            result.append(lp)
        return torch.stack(result)
    table=g.policy_tables(utility[None].expand(len(actions),-1,-1),(2,1),eligible[None].expand(len(actions),-1,-1),mode)
    actual=g.matching_log_probability(table,types,available[None].expand(len(actions),-1),actions)
    expected=explicit(utility);torch.testing.assert_close(actual,expected,atol=1e-10,rtol=0)
    old=explicit(utility.detach()*.7).detach();advantage=torch.linspace(-1.,1.,len(actions),dtype=torch.float64)
    def loss(log):
        ratio=(log-old).exp();return torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
    a=torch.autograd.grad(loss(actual),utility,retain_graph=True)[0];b=torch.autograd.grad(loss(expected),utility)[0]
    torch.testing.assert_close(a,b,atol=1e-9,rtol=0)


def test_labeled_worker_contracts_and_unavailable_all_wait():
    table=g.policy_tables(torch.ones((1,2,2),dtype=torch.float64),(1,1));types=torch.tensor([0,1]);available=torch.zeros((1,2),dtype=torch.bool)
    assert g.maximum_matching(table,types,available).tolist()==[[0,0]]
    assert g.matching_log_probability(table,types,available,torch.zeros((1,2),dtype=torch.long)).item()==0
    with pytest.raises(ValueError):g.root_values(table,torch.tensor([0,0]),available)
    with pytest.raises(ValueError):g.root_values(table,types,available.float())
    with pytest.raises(ValueError):g.matching_log_probability(table,types,~available,torch.tensor([[1,1]]))
