"""Deterministic algebra fixtures only: no environment, checkpoint or seed panel."""
import importlib.util,itertools
from pathlib import Path
import pytest
import torch

spec=importlib.util.spec_from_file_location('broker_algebra',Path(__file__).resolve().parents[1]/'scripts/broker_marginal_algebra.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def decode(proposal,ranks):
    # Independent grouping implementation used only by exhaustive tiny fixtures.
    result=[0]*len(proposal)
    for token in set(proposal)-{0}:
        group=[j for j,t in enumerate(proposal) if t==token]
        j=min(group,key=lambda j:int(ranks[token-1,j]));result[j]=token
    return tuple(result)


def fixture(n,k,masked=False):
    logits=torch.sin(torch.arange(k*(n+1),dtype=torch.float64).reshape(k,n+1)*.73).requires_grad_()
    ranks=torch.tensor([[(j+m)%k for j in range(k)] for m in range(n)])
    legal=torch.ones((n,k),dtype=torch.bool)
    if masked:
        legal[0,-1]=False;legal[:,0]=False
        if n>1 and k>2:legal[1,1]=False
    support=torch.cat([torch.ones((k,1),dtype=torch.bool),legal.T],1)
    log=torch.log_softmax(logits.masked_fill(~support,-torch.inf),1)
    groups={};latent={}
    for u in itertools.product(range(n+1),repeat=k):
        if any(t and not legal[t-1,j] for j,t in enumerate(u)):continue
        mass=log[torch.arange(k),torch.tensor(u)].sum().exp();s=decode(u,ranks)
        groups[s]=groups.get(s,0)+mass;latent[u]=(s,mass)
    return logits,log,ranks,legal,groups,latent


@pytest.mark.parametrize('n,k',[(1,2),(2,3),(3,3),(2,4)])
@pytest.mark.parametrize('masked',[False,True])
def test_mass_gradient_and_normalization_match_exhaustive_proposal_sum(n,k,masked):
    torch.set_num_threads(1);logits,log,ranks,legal,groups,_=fixture(n,k,masked);masses=[]
    for s,expected in groups.items():
        actual=a.physical_logprob(log,s,ranks,legal).exp();masses.append(actual)
        torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-12)
        g=torch.autograd.grad(actual,logits,retain_graph=True)[0];reference=torch.autograd.grad(expected,logits,retain_graph=True)[0]
        torch.testing.assert_close(g,reference,atol=2e-12,rtol=2e-12)
    total=torch.stack(masses).sum();torch.testing.assert_close(total,torch.ones_like(total),atol=2e-12,rtol=0)
    assert torch.autograd.grad(total,logits)[0].abs().max()<2e-12


def test_posterior_score_gradient_mean_and_variance():
    logits,log,ranks,legal,groups,latent=fixture(2,3);physical_scores={};latent_scores={}
    for s in groups:physical_scores[s]=torch.autograd.grad(a.physical_logprob(log,s,ranks,legal),logits,retain_graph=True)[0].flatten().detach()
    for u in latent:latent_scores[u]=torch.autograd.grad(log[torch.arange(3),torch.tensor(u)].sum(),logits,retain_graph=True)[0].flatten().detach()
    for s,q in groups.items():
        posterior=sum(p.detach()/q.detach()*latent_scores[u] for u,(out,p) in latent.items() if out==s)
        torch.testing.assert_close(physical_scores[s],posterior,atol=2e-12,rtol=2e-12)
    advantage={s:float(sum((j+1)*m for j,m in enumerate(s)))-2.5 for s in groups}
    latent_mean=sum(p.detach()*advantage[s]*latent_scores[u] for u,(s,p) in latent.items())
    physical_mean=sum(q.detach()*advantage[s]*physical_scores[s] for s,q in groups.items())
    torch.testing.assert_close(latent_mean,physical_mean,atol=2e-12,rtol=2e-12)
    latent_variance=sum(p.detach()*((advantage[s]*latent_scores[u]-latent_mean)**2).sum() for u,(s,p) in latent.items())
    physical_variance=sum(q.detach()*((advantage[s]*physical_scores[s]-physical_mean)**2).sum() for s,q in groups.items())
    conditional_variance=sum(p.detach()*((advantage[s]*(latent_scores[u]-physical_scores[s]))**2).sum() for u,(s,p) in latent.items())
    torch.testing.assert_close(latent_variance-physical_variance,conditional_variance,atol=2e-10,rtol=2e-12)
    assert float(conditional_variance)>0


def test_ratio_conditional_expectation_and_clipped_surrogate_jensen():
    _,old,ranks,legal,groups,latent=fixture(2,3)
    new=torch.log_softmax(-old.detach()*2+torch.arange(9,dtype=torch.float64).reshape(3,3)*.6,1)
    def clipped(r,adv):return torch.minimum(r*adv,r.clamp(.8,1.2)*adv)
    for s,q in groups.items():
        physical_ratio=(a.physical_logprob(new,s,ranks,legal)-q.log()).exp()
        ratios=[]
        for u,(out,p) in latent.items():
            if out==s:
                r=(new[torch.arange(3),torch.tensor(u)].sum()-p.log()).exp();ratios.append((p.detach()/q.detach(),r))
        torch.testing.assert_close(physical_ratio,sum(w*r for w,r in ratios),atol=2e-12,rtol=2e-12)
        for adv in (-2.,1.5):assert float((clipped(physical_ratio,adv)-sum(w*clipped(r,adv) for w,r in ratios)).detach())>=-2e-12


def test_alias_split_counterexample_and_kl_contraction():
    old=torch.tensor([[.4,.6],[.8,.2]],dtype=torch.float64).log();new=torch.tensor([[.4,.6],[.2,.8]],dtype=torch.float64).log()
    ranks=torch.tensor([[0,1]]);legal=torch.ones((1,2),dtype=torch.bool);s=(1,0)
    torch.testing.assert_close(a.physical_logprob(old,s,ranks,legal),a.physical_logprob(new,s,ranks,legal),atol=2e-12,rtol=0)
    # Winner0's alias class has two latent vectors; physical ratio1, clipped latent conditional mean .44.
    conditional=sum(float(old[1,j].exp())*min(float((new[1,j]-old[1,j]).exp()),1.2) for j in (0,1))
    assert conditional==pytest.approx(.44)
    states={(0,0),(1,0),(0,1)};physical_kl=0.
    for s in states:
        log_old=a.physical_logprob(old,s,ranks,legal);log_new=a.physical_logprob(new,s,ranks,legal)
        physical_kl+=float(log_old.exp()*(log_old-log_new))
    latent_kl=float((old.exp()*(old-new)).sum());assert physical_kl<latent_kl


def test_extreme_logits_forced_wait_and_invalid_matchings():
    ranks=torch.tensor([[0,1],[1,0]]);legal=torch.ones((2,2),dtype=torch.bool)
    log=torch.log_softmax(torch.tensor([[0.,-10000.,10000.],[0.,10000.,-10000.]],dtype=torch.float64,requires_grad=True),1)
    value=a.physical_logprob(log,(1,0),ranks,legal);assert torch.isfinite(value) and value<-10000
    assert torch.isfinite(torch.autograd.grad(value,log)[0]).all()
    for s in [(1,1),(3,0),(1.,0.),(1,)]:
        with pytest.raises(ValueError):a.physical_logprob(log,s,ranks,legal)
    with pytest.raises(ValueError):a.physical_logprob(log,(0,0),torch.zeros_like(ranks),legal)
    legal[:]=False
    with pytest.raises(ValueError):a.physical_logprob(log,(1,0),ranks,legal)
    forced=torch.tensor([[0.,-torch.inf,-torch.inf],[0.,-torch.inf,-torch.inf]],dtype=torch.float64)
    assert a.physical_logprob(forced,(0,0),ranks,legal).item()==0


def test_rejects_unmasked_or_unnormalized_categorical_input():
    ranks=torch.tensor([[0,1]]);legal=torch.tensor([[True,False]])
    with pytest.raises(ValueError):a.physical_logprob(torch.tensor([[.5,.5],[.5,.5]]).log(),(0,0),ranks,legal)
    legal[:]=True
    with pytest.raises(ValueError):a.physical_logprob(torch.zeros((2,2)),(0,0),ranks,legal)
