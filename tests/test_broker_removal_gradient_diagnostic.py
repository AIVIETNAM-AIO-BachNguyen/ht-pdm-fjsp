import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_broker_removal_gradient_diagnostic as d
from ht_pdm_fjsp.technician_broker_removal_policy import baseline_cache


def fixture():
    torch.set_num_threads(1);c=d.e.settings('engineering');m=d.e.make_model(d.e.CORE[0],c,c['train_seeds'][0]).double();m.critic.requires_grad_(False);m.removal.requires_grad_(False);o=d.DiagnosticEnv(340700000,4,d.e.CORE[0]).reset();return m,o


@pytest.mark.parametrize('proposal',[False,True])
def test_closed_vjp_full_network_finite_difference_and_autograd(proposal):
    model,obs=fixture();h=d.reference()[1];observations=[obs]*4;batch=d.double_batch(observations);a,_,_=model.sample(batch,torch.Generator().manual_seed(340700001));adv=np.array([[1.,-1.],[2.,.5],[-.5,3.],[.3,-.1]])
    def score():
        if proposal:return model.proposal_log_probs(batch)[0].gather(-1,a[...,None]).squeeze(-1)
        return model.evaluate(batch,a)[0]
    objective=(score()*torch.tensor(adv)).mean();expected=torch.cat([v.ravel() for v in torch.autograd.grad(objective,tuple(model.actor.parameters()))]).numpy();w={k:v.detach().numpy().copy() for k,v in model.state_dict().items()};actual=d.numpy_vjp(observations,a.numpy(),adv,w,h,proposal);np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10)
    parameter=model.actor[4].bias;original=parameter[0].item();delta=1e-5
    with torch.no_grad():parameter[0]=original+delta
    plus=float((score()*torch.tensor(adv)).mean().detach())
    with torch.no_grad():parameter[0]=original-delta
    minus=float((score()*torch.tensor(adv)).mean().detach())
    assert (plus-minus)/(2*delta)==pytest.approx(actual[-5],rel=1e-7,abs=1e-9)


def test_variance_and_locked_all_checkpoint_rule():
    a=np.array([[1.,2.],[3.,4.],[5.,6.]]);assert d.trace_covariance(a)==8
    m=d.metrics(dict(state=a,conditional=a*.9,proposal=a*1.1));assert m['gain_vs_state']==pytest.approx(.19)
    records=[dict(train_seed=t,steps=s,gain_vs_state=.06) for t in d.TRAINS for s in d.STEPS];_,checks=d.primary(records);assert all(v for row in checks.values() for v in row.values())
    records[0]['gain_vs_state']=-.01;_,checks=d.primary(records);assert not checks[str(d.TRAINS[0])]['material_mean_gain']
    with pytest.raises(AssertionError):d.primary(records[:-1])


@pytest.mark.parametrize('learned',[False,True])
def test_complete_engineering_episode_replay_stochastic_baseline_and_gradients(tmp_path,learned):
    model,_=fixture()
    if learned:
        with torch.no_grad():model.removal[0].weight[:,71:79].copy_(torch.arange(512).reshape(64,8)/1000-.25)
    result=d.collect(model,[340700000],340700001,tmp_path/'paired',lambda:None);assert result['optimizer_updates']==0 and result['physical_transitions']==60 and result['policy_decisions']==48
    assert max(result['gradient_max_errors'].values())<1e-9
    with np.load(tmp_path/'paired/gradients.npz') as f:
        assert f['state'].shape==(1,11141)
        if not learned:np.testing.assert_allclose(f['state'],f['conditional'],atol=1e-12,rtol=1e-12)
        else:assert np.max(abs(f['state']-f['conditional']))>1e-5


def test_seed_panels_fresh_and_engineering_only():
    result=d.seed_audit(d.reference()[1]);assert result['raw']==146 and not result['confirmation_states_constructed']
