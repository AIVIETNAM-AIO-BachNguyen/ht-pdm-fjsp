import itertools
from collections import defaultdict
import numpy as np
import pytest
import torch
from ht_pdm_fjsp.technician_broker_removal_credit import conditional_removals,expected_removal_baseline,factor_log_probs


def decode(u,priority):
    result=[0]*len(u)
    for token in set(u)-{0}:
        j=max((j for j,t in enumerate(u) if t==token),key=lambda j:(priority[token-1,j],-j));result[j]=token
    return tuple(result)


def baseline(i,matching):
    return np.sin(.3*(i+1)+sum((j+1)*t*t*.2 for j,t in enumerate(matching)))+np.prod(np.array(matching)+1)*.013


@pytest.mark.parametrize('n,k,masked',[(2,2,False),(3,3,False),(3,3,True),(2,4,False),(2,4,True)])
def test_every_preimage_removal_law_and_conditional_credit_gradient(n,k,masked):
    torch.set_num_threads(1);legal=np.ones((n,k),bool)
    if masked:legal[:,-1]=False;legal[0,0]=False
    priority=np.array([[(j+m)%3 for j in range(k)] for m in range(n)],float)
    raw=torch.tensor(np.cos(np.arange(k*(n+1)).reshape(k,n+1)*.7),dtype=torch.float64,requires_grad=True);support=torch.tensor(np.c_[np.ones(k,bool),legal.T]);logp=raw.masked_fill(~support,-1e9).log_softmax(-1);groups=defaultdict(list)
    for u in itertools.product(*[[0]+[m+1 for m in range(n) if legal[m,j]] for j in range(k)]):
        log=sum(logp[j,t] for j,t in enumerate(u));mass=log.exp();groups[decode(u,priority)].append((u,mass,log))
    zero=torch.zeros_like(raw);baseline_surrogate=0.;total_mass=0.
    for matching,preimages in groups.items():
        physical=torch.tensor([matching],dtype=torch.long);weights,matchings=conditional_removals(logp[None],physical,torch.tensor(legal[None]),torch.tensor(priority[None]));torch.testing.assert_close(weights.sum(-1),torch.ones((1,k),dtype=torch.float64),atol=1e-12,rtol=0)
        values=torch.tensor([[[baseline(i,tuple(a.tolist())) for a in matchings[0,i]] for i in range(k)]],dtype=torch.float64);b=expected_removal_baseline(weights,values)[0];mass=sum(p for u,p,log in preimages);total_mass+=float(mass.detach());factors=factor_log_probs(logp[None],physical,torch.tensor(legal[None]),torch.tensor(priority[None]))[0];torch.testing.assert_close(factors.sum().exp(),mass,atol=1e-12,rtol=1e-12)
        reward=np.cos(sum((i+1)*token for i,token in enumerate(matching))*.4);reference=0.
        for i in range(k):
            conditional=defaultdict(float)
            for u,p,log in preimages:
                removed=list(u);removed[i]=0;alternative=decode(removed,priority);conditional[alternative]+=float((p/mass).detach());reference=reference+(p/mass).detach()*logp[i,u[i]]*(reward-baseline(i,alternative));baseline_surrogate=baseline_surrogate+p.detach()*logp[i,u[i]]*baseline(i,alternative)
            actual=defaultdict(float)
            for r in range(k+1):actual[tuple(matchings[0,i,r].tolist())]+=float(weights[0,i,r].detach())
            for a in set(actual)|set(conditional):assert actual[a]==pytest.approx(conditional[a],abs=1e-12)
            assert float(b[i].detach())==pytest.approx(sum(prob*baseline(i,a) for a,prob in conditional.items()),abs=1e-12)
        rb=(factors*(reward-b.detach())).sum();a=torch.autograd.grad(reference,raw,retain_graph=True)[0];c=torch.autograd.grad(rb,raw,retain_graph=True)[0];torch.testing.assert_close(a,c,atol=1e-10,rtol=1e-10)
    assert total_mass==pytest.approx(1,abs=1e-12);zero=torch.autograd.grad(baseline_surrogate,raw)[0];torch.testing.assert_close(zero,torch.zeros_like(raw),atol=1e-10,rtol=0)


def test_removing_winner_can_promote_a_loser_and_idle_removal_changes_nothing():
    p=torch.tensor([[[.1,.8,.1],[.2,.7,.1],[.4,.5,.1]]],dtype=torch.float64);physical=torch.tensor([[1,0,0]]);legal=torch.ones((1,2,3),dtype=torch.bool);priority=torch.tensor([[[3.,2.,1.],[3.,2.,1.]]]);weights,matchings=conditional_removals(p.log(),physical,legal,priority)
    torch.testing.assert_close(weights[0,0],torch.tensor([(2/9)*(4/9),0.,7/9,(2/9)*(5/9)],dtype=torch.float64),atol=1e-12,rtol=0)
    assert matchings[0,0,2].tolist()==[0,1,0] and matchings[0,0,3].tolist()==[0,0,1]
    assert weights[0,1,0]==1 and matchings[0,1,0].tolist()==[1,0,0]


def test_forced_wait_and_invalid_physical_actions():
    logp=torch.tensor([[[0.,-1e9,-1e9],[0.,-1e9,-1e9]]],dtype=torch.float64,requires_grad=True);legal=torch.zeros((1,2,2),dtype=torch.bool);priority=torch.ones((1,2,2));physical=torch.zeros((1,2),dtype=torch.long);weights,matching=conditional_removals(logp,physical,legal,priority)
    assert matching.count_nonzero()==0 and weights[:,:,0].eq(1).all() and weights[:,:,1:].eq(0).all()
    with pytest.raises(ValueError,match='legal physical'):conditional_removals(logp,torch.tensor([[1,0]]),legal,priority)


def test_actor_baseline_must_detach_conditional_weights_to_avoid_extra_gradient():
    raw=torch.tensor([[.2,.9,-.3],[.4,.1,.8]],dtype=torch.float64,requires_grad=True);logp=raw.log_softmax(-1);priority=np.array([[2.,1.],[1.,2.]]);legal=torch.ones((1,2,2),dtype=torch.bool);groups=defaultdict(list)
    for u in itertools.product(range(3),repeat=2):groups[decode(u,priority)].append(sum(logp[j,t] for j,t in enumerate(u)).exp())
    detached=0.;leaking=0.
    for matching,terms in groups.items():
        physical=torch.tensor([matching]);weights,matchings=conditional_removals(logp[None],physical,legal,torch.tensor(priority[None]));values=torch.tensor([[[baseline(i,tuple(a.tolist())) for a in matchings[0,i]] for i in range(2)]],dtype=torch.float64);b=expected_removal_baseline(weights,values)[0];f=factor_log_probs(logp[None],physical,legal,torch.tensor(priority[None]))[0];mass=sum(terms).detach();detached+=mass*(f*b.detach()).sum();leaking+=mass*(f*b).sum()
    correct=torch.autograd.grad(detached,raw,retain_graph=True)[0];wrong=torch.autograd.grad(leaking,raw)[0];torch.testing.assert_close(correct,torch.zeros_like(raw),atol=1e-10,rtol=0);assert wrong.norm()>1e-7
