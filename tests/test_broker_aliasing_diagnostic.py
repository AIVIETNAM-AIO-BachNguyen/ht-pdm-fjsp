"""Only deterministic arithmetic fixtures; no frozen research results read."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
x=module('alias_diag','diagnose_broker_aliasing.py');a=module('marginal_algebra','broker_marginal_algebra.py')


@pytest.mark.parametrize('n,masked',[(1,False),(2,False),(4,False),(4,True)])
def test_analytic_physical_score_matches_autodiff_and_exhaustive_moments(n,masked):
    torch.set_num_threads(1);z=torch.sin(torch.arange(2*(n+1),dtype=torch.float64)).reshape(2,n+1).requires_grad_();legal=np.ones((n,2),bool)
    if masked:legal[:,0]=False;legal[0,1]=False
    support=torch.tensor(np.c_[np.ones(2,bool),legal.T]);logp=torch.log_softmax(z.masked_fill(~support,-torch.inf),1)
    ranks=np.array([[m%2,(m+1)%2] for m in range(n)]);gram=np.eye(2*(n+1))
    for matching in [(0,0),*[(m+1,j) if j==0 else (0,m+1) for m in range(n) for j in range(2)]]:
        if any(t and not legal[t-1,j] for j,t in enumerate(matching)):continue
        logmass,score=x.physical_score(logp.detach().numpy(),matching,ranks,legal)
        actual=a.physical_logprob(logp,matching,ranks,legal);expected=torch.autograd.grad(actual,z,retain_graph=True)[0].detach().numpy().ravel()
        assert logmass==pytest.approx(float(actual.detach()),abs=2e-12)
        np.testing.assert_allclose(score,expected,atol=2e-12,rtol=2e-12)
    metrics=x.noise_metrics(logp.detach().numpy(),gram,ranks,legal)
    assert metrics['removed_fisher_trace']>=0 and 0<=metrics['conflict_probability']<=1
    if masked:assert metrics['removed_fraction']==pytest.approx(0,abs=2e-12)


def test_full_shared_actor_parameter_gram_matches_direct_policy_score():
    shapes={'actor.0.weight':(3,7),'actor.0.bias':(3,),'actor.2.weight':(3,3),'actor.2.bias':(3,),'actor.4.weight':(3,3),'actor.4.bias':(3,)}
    weights={name:(torch.sin(torch.arange(np.prod(shape),dtype=torch.float64)*.2).reshape(shape)*.3).requires_grad_() for name,shape in shapes.items()};inputs=np.cos(np.arange(14).reshape(2,7)*.3)
    logits,gram=x.actor_logits_gram(inputs,weights);logp=logits-np.logaddexp.reduce(logits,axis=1)[:,None];score=-np.exp(logp);score[[0,1],[1,0]]+=1
    z=torch.tensor(inputs,dtype=torch.float64)
    for i in (0,2,4):
        z=z@weights[f'actor.{i}.weight'].T+weights[f'actor.{i}.bias']
        if i!=4:z=z.tanh()
    log=torch.log_softmax(z,1);grads=torch.autograd.grad(log[0,1]+log[1,0],tuple(weights.values()))
    exact=sum(float((g*g).sum()) for g in grads)
    assert float(score.ravel()@gram@score.ravel())==pytest.approx(exact,abs=2e-12,rel=2e-12)
    assert all(w.grad is None for w in weights.values())


def test_inputs_reconstruct_exact_independent_zero_prefix():
    helper=x.load_helper();state=dict(ages=[3,4,5,6],failed=[False,True,False,False],pending_wait=[0,2,0,0],remaining=[0,0],assigned=[-1,-1]);obs=helper.observation(state,7);inputs=x.inputs_for(obs)
    assert inputs.shape==(2,103) and (inputs[:,90:102]==0).all()
    np.testing.assert_array_equal(inputs[:,:71],np.tile(obs['state'],(2,1)));np.testing.assert_array_equal(inputs[:,-1],[0,.5])


def test_aggregate_ratio_of_sums_and_locked_primary_time():
    records=[dict(train_seed=t,checkpoint_steps=step,seed=s,tick=tick,proposal_fisher_trace=1.,physical_fisher_trace=.8,removed_fisher_trace=.2,
        proposal_entropy_per_technician=.2,conflict_probability=.1) for t in x.TRAINS for step in x.STEPS for s in x.DEV for tick in range(0,48,3)]
    result=x.summarize(records);assert result['future_trial_screen_pass'] and len(result['summaries'])==27
    row=next(r for r in records if r['train_seed']==x.TRAINS[0] and r['checkpoint_steps']==122880);row.update(proposal_fisher_trace=10000.,physical_fisher_trace=10000.,removed_fisher_trace=0.)
    result=x.summarize(records);assert not result['future_trial_screen_pass']
    assert result['summaries'][f'{x.TRAINS[0]}:122880']['removed_fraction']==pytest.approx(511*.2/10511)
    with pytest.raises(AssertionError):x.summarize(records[:-1])


def test_actual_actor_shape_gram_and_masked_moments():
    shapes={'actor.0.weight':(64,103),'actor.0.bias':(64,),'actor.2.weight':(64,64),'actor.2.bias':(64,),'actor.4.weight':(5,64),'actor.4.bias':(5,)}
    weights={name:(torch.sin(torch.arange(np.prod(shape),dtype=torch.float64)*.2).reshape(shape)*.03).requires_grad_() for name,shape in shapes.items()}
    logits,gram=x.actor_logits_gram(np.cos(np.arange(206).reshape(2,103)*.3),weights)
    assert logits.shape==(2,5) and gram.shape==(10,10) and sum(p.numel() for p in weights.values())==11141
    assert np.linalg.eigvalsh(gram).min()>-2e-10
    logp=logits-np.logaddexp.reduce(logits,axis=1)[:,None];ranks=np.array([[0,1],[0,1],[0,1],[1,0]])
    metrics=x.noise_metrics(logp,gram,ranks,np.ones((4,2),bool));assert metrics['proposal_actions']==25 and metrics['physical_actions']==21
    assert metrics['proposal_fisher_trace']>=metrics['physical_fisher_trace']>0


def test_independent_direct_parameter_verifier_matches_gram_moments():
    v=module('alias_direct_verifier','verify_broker_aliasing_diagnostic.py')
    shapes={'actor.0.weight':(3,7),'actor.0.bias':(3,),'actor.2.weight':(3,3),'actor.2.bias':(3,),'actor.4.weight':(3,3),'actor.4.bias':(3,)}
    weights={name:(torch.cos(torch.arange(np.prod(shape),dtype=torch.float64)*.2).reshape(shape)*.3).requires_grad_() for name,shape in shapes.items()};inputs=np.sin(np.arange(14).reshape(2,7)*.3)
    logits,gram=x.actor_logits_gram(inputs,weights);logp=logits-np.logaddexp.reduce(logits,axis=1)[:,None];ranks=np.array([[0,1],[1,0]])
    for legal in [np.ones((2,2),bool),np.array([[True,False],[False,True]]),np.zeros((2,2),bool)]:
        support=np.c_[np.ones(2,bool),legal.T];z=np.where(support,logits,-np.inf);logp=z-np.logaddexp.reduce(z,axis=1)[:,None]
        metrics=x.noise_metrics(logp,gram,ranks,legal);proposal,physical=v.direct_fisher(inputs,weights,legal,ranks)
        assert proposal==pytest.approx(metrics['proposal_fisher_trace'],abs=2e-12,rel=2e-12)
        assert physical==pytest.approx(metrics['physical_fisher_trace'],abs=2e-12,rel=2e-12)
