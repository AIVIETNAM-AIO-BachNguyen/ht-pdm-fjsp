import importlib.util,itertools
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors
from ht_pdm_fjsp.technician_broker_marginal_policy import CORE_ARMS,BrokerMarginalPolicy,priorities,physical_tokens,physical_logprob
from ht_pdm_fjsp.technician_raw_state_reservation_policy import RawStateReservationPolicy

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('broker_unit_math',ROOT/'scripts/broker_marginal_algebra.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)

def batch(n=4,k=2):
    dim=6*n+3*k+4*n*k+9;edges=torch.zeros((1,n,k,4),dtype=torch.float64);edges[...,1]=1/12;edges[...,2]=.5
    return dict(state=torch.sin(torch.arange(dim,dtype=torch.float64))[None],technicians=torch.zeros((1,k,3),dtype=torch.float64),edges=edges,pair_mask=torch.ones((1,n,k),dtype=torch.bool))


@pytest.mark.parametrize('arm',CORE_ARMS)
def test_exact_pushforward_proposal_compatibility_gradients_and_aliases(arm):
    torch.set_num_threads(1);model=BrokerMarginalPolicy(arm).double();b=batch();log,_=model.proposal_log_probs(b);priority=priorities(b,arm)
    proposals=torch.tensor(list(itertools.product(range(5),repeat=2)),dtype=torch.long);expanded={k:v.expand(len(proposals),*v.shape[1:]).clone() for k,v in b.items()}
    physical=physical_tokens(proposals,expanded['pair_mask'],priorities(expanded,arm));actual,_,_=model.evaluate(expanded,proposals);unique={}
    for i,s in enumerate(physical.tolist()):unique.setdefault(tuple(s),i)
    assert len(unique)==21 and sum(p.numel() for p in model.parameters())==19974
    if arm in (CORE_ARMS[0],CORE_ARMS[2]):
        mass=actual[list(unique.values())].exp().sum();torch.testing.assert_close(mass,torch.ones_like(mass),atol=2e-12,rtol=0)
        grads=torch.autograd.grad(mass,tuple(model.actor.parameters()),retain_graph=True);assert max(float(g.abs().max()) for g in grads)<2e-12
        for s,first in unique.items():
            group=[i for i,t in enumerate(physical.tolist()) if tuple(t)==s]
            torch.testing.assert_close(actual[group],actual[first].expand(len(group)),atol=2e-12,rtol=0)
            ranks=np.empty((4,2),int)
            for m in range(4):
                for rank,j in enumerate(sorted(range(2),key=lambda j:(-float(priority[0,m,j]),j))):ranks[m,j]=rank
            expected=reference.physical_logprob(log[0],s,ranks,b['pair_mask'][0])
            torch.testing.assert_close(actual[first],expected,atol=2e-12,rtol=0)
    else:
        base=RawStateReservationPolicy('independent_raw_ppo').double();base.load_state_dict(model.state_dict());old=base.evaluate(expanded,proposals)[0]
        torch.testing.assert_close(actual,old,atol=2e-12,rtol=0)


@pytest.mark.parametrize('arm',CORE_ARMS)
def test_sample_replay_forced_wait_critic_and_frozen_likelihood_execution(arm):
    model=BrokerMarginalPolicy(arm).double();b=batch();g=torch.Generator().manual_seed(17);actions,old_log,old_value=model.sample(b,g)
    log,entropy,value=model.evaluate(b,actions);torch.testing.assert_close(old_log,log,atol=0,rtol=0);torch.testing.assert_close(old_value,value,atol=0,rtol=0)
    assert entropy.item()>=0
    b['pair_mask'][:]=False;actions,old_log,_=model.sample(b,g);log,entropy,_=model.evaluate(b,actions)
    assert actions.tolist()==[[0,0]] and old_log.item()==log.item()==entropy.item()==0
    with pytest.raises(ValueError):model.evaluate(b,torch.tensor([[1,0]]))
    with pytest.raises(ValueError):model.evaluate(b,torch.zeros((1,2)))


def test_quality_double_comparison_and_id_rank_diverge_on_near_float32_tie():
    b=batch(1,2);b['edges']=b['edges'].float();b['edges'][0,0,:,1]=torch.tensor([1/12,3/12]);b['edges'][0,0,:,2]=torch.tensor([.1,.3]);u=torch.tensor([[1,1]])
    assert float((b['edges'][0,0,0,2]/1))==float((b['edges'][0,0,1,2]/3))
    assert physical_tokens(u,b['pair_mask'],priorities(b,CORE_ARMS[0])).tolist()==[[0,1]]
    assert physical_tokens(u,b['pair_mask'],priorities(b,CORE_ARMS[2])).tolist()==[[1,0]]
    with pytest.raises(ValueError):physical_logprob(torch.zeros((1,2,2)),u,b['pair_mask'],priorities(b,CORE_ARMS[0]))


def test_polynomial_shapes_no_catalogue_and_no_learning_likelihood_at_select(monkeypatch):
    import ht_pdm_fjsp.maintenance_heterogeneous_model as catalogue
    import ht_pdm_fjsp.technician_broker_marginal_policy as module
    def forbidden(*args,**kwargs):raise AssertionError('forbidden deployment helper')
    monkeypatch.setattr(catalogue,'catalogue',forbidden)
    for n,k in ((8,4),(32,16)):
        model=BrokerMarginalPolicy(CORE_ARMS[0],n,k).double();b=batch(n,k)
        action,log,_=model.sample(b,torch.Generator().manual_seed(17));assert action.shape==(1,k) and torch.isfinite(log).all()
    model=BrokerMarginalPolicy(CORE_ARMS[0]);obs={k:v[0].numpy() for k,v in batch().items()}
    monkeypatch.setattr(module,'physical_logprob',forbidden);monkeypatch.setattr(model.critic,'forward',forbidden)
    assert model.select(obs).shape==(2,)


def test_numpy_torch_decoders_agree_all_proposals_and_near_tie():
    from ht_pdm_fjsp.technician_broker_marginal_policy import decode_proposals
    b=batch();b['edges']=b['edges'].float();b['edges'][0,0,:,1]=torch.tensor([1/12,3/12]);b['edges'][0,0,:,2]=torch.tensor([.1,.3])
    for arm in CORE_ARMS:
        for tokens in itertools.product(range(5),repeat=2):
            pairs,diag=decode_proposals(np.array(tokens),b['pair_mask'][0].numpy(),arm,b['edges'][0].numpy())
            physical=physical_tokens(torch.tensor([tokens]),b['pair_mask'],priorities(b,arm))[0].tolist()
            assert pairs==tuple(sorted((t-1,j) for j,t in enumerate(physical) if t))
            assert diag['rejected_proposals']==int(tokens[0]>0 and tokens[0]==tokens[1])


def test_matched_ratio_modes_change_only_learning_likelihood_at_fixed_weights():
    b=batch();m=BrokerMarginalPolicy(CORE_ARMS[0]).double();p=BrokerMarginalPolicy(CORE_ARMS[1]).double();p.load_state_dict(m.state_dict());u=torch.tensor([[1,1]])
    ml,me,mv=m.evaluate(b,u);pl,pe,pv=p.evaluate(b,u)
    torch.testing.assert_close(me,pe,atol=0,rtol=0);torch.testing.assert_close(mv,pv,atol=0,rtol=0)
    assert ml.item()>pl.item()
    obs={key:v[0].numpy() for key,v in b.items()}
    # select uses float32 tensors, so fixed-weight float32 copies are compared.
    m.float();p.float();assert np.array_equal(m.select(obs),p.select(obs))


def test_alias_gradient_and_ratio_are_invariant_to_discarded_proposal():
    m=BrokerMarginalPolicy(CORE_ARMS[0]).double();b=batch();b={k:v.expand(2,*v.shape[1:]).clone() for k,v in b.items()};u=torch.tensor([[1,0],[1,1]])
    old=m.evaluate(b,u)[0];g0=torch.autograd.grad(old[0],tuple(m.actor.parameters()),retain_graph=True);g1=torch.autograd.grad(old[1],tuple(m.actor.parameters()))
    for a,c in zip(g0,g1):torch.testing.assert_close(a,c,atol=2e-12,rtol=2e-12)
    with torch.no_grad():m.actor[4].bias.add_(torch.linspace(-.1,.1,5,dtype=torch.float64))
    new=m.evaluate(b,u)[0];ratio=(new-old.detach()).exp();torch.testing.assert_close(ratio[0],ratio[1],atol=2e-12,rtol=0)


@pytest.mark.parametrize('arm',CORE_ARMS)
def test_actual_environment_decoder_physics_cutoff_and_reward(arm):
    from dataclasses import asdict
    from ht_pdm_fjsp.technician_broker_marginal_env import ResourceEnv
    spec=importlib.util.spec_from_file_location('broker_primitive_fixture',ROOT/'scripts/verify_raw_reservation_budget_development.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    torch.set_num_threads(1);env=ResourceEnv(290700001,4,arm);obs=env.reset();state,events,population=v.initial_events(env.physical_seed);assert population==env.population
    model=BrokerMarginalPolicy(arm);reward=0.;cost=0.;cutoff=None
    for tick in range(48):
        assert state=={k:list(a) for k,a in asdict(env.env.state).items()}
        action=model.select(obs);pairs,diag=env.resolve(action)
        expected,metrics=v.primitive_step(state,pairs,events[tick]);cost+=metrics['objective']
        obs,r,done,info=env.step(action);reward+=r
        assert info['coordination']==diag
        if tick<47:state=expected
        else:
            cutoff=cost;state=expected
            for tail in range(48,60):
                from ht_pdm_fjsp.technician_reservation_env import deadline
                from ht_pdm_fjsp.maintenance_dispatch import DispatchState
                pairs=deadline(env.config,DispatchState(**{k:tuple(a) for k,a in state.items()}))
                state,metrics=v.primitive_step(state,pairs,events[tail]);cost+=metrics['objective']
    assert done and cost==info['episode_metrics']['extended_cost'] and cutoff==info['episode_metrics']['policy_cost']
    assert -20*reward==pytest.approx(cost,abs=1e-5) and info['episode_metrics']['invalid_assignments']==0


def test_fresh_paired_environment_tapes_and_no_core_catalogue(monkeypatch):
    from ht_pdm_fjsp.technician_broker_marginal_env import ResourceEnv
    from ht_pdm_fjsp.technician_raw_reservation_budget_env import ResourceEnv as OldEnv
    import ht_pdm_fjsp.maintenance_heterogeneous_model as base
    names=(*CORE_ARMS,'ppo','normalized_ppo','mappo','full_mappo');populations=[]
    for arm in names:
        env=ResourceEnv(290700001,4,arm);env.reset();populations.append(env.population)
    assert len(set(populations))==1
    old=OldEnv(290700001,4,'independent_raw_ppo');old.reset();assert env.physical_seed!=old.physical_seed
    def forbidden(*args):raise AssertionError('core enumerated matching catalogue')
    monkeypatch.setattr(base,'catalogue',forbidden)
    for arm in CORE_ARMS:
        env=ResourceEnv(290700001,4,arm);obs=env.reset();env.step(BrokerMarginalPolicy(arm).select(obs))


@pytest.mark.parametrize('n,k',[(2,3),(2,4)])
@pytest.mark.parametrize('arm',[CORE_ARMS[0],CORE_ARMS[2]])
def test_vectorized_exact_density_and_gradient_for_more_than_two_technicians(n,k,arm):
    b=batch(n,k)
    for m in range(n):
        for j in range(k):
            b['edges'][0,m,j,1]=(1+j%3)/12;b['edges'][0,m,j,2]=.1+.1*((j+m)%k)
    model=BrokerMarginalPolicy(arm,n,k).double();proposals=torch.tensor(list(itertools.product(range(n+1),repeat=k)),dtype=torch.long)
    expanded={key:v.expand(len(proposals),*v.shape[1:]).clone() for key,v in b.items()};physical=physical_tokens(proposals,expanded['pair_mask'],priorities(expanded,arm));log=model.evaluate(expanded,proposals)[0];unique={}
    for i,s in enumerate(physical.tolist()):unique.setdefault(tuple(s),i)
    single=model.proposal_log_probs(b)[0][0];priority=priorities(b,arm)[0];ranks=np.empty((n,k),int)
    for m in range(n):
        for rank,j in enumerate(sorted(range(k),key=lambda j:(-float(priority[m,j]),j))):ranks[m,j]=rank
    for s,i in unique.items():torch.testing.assert_close(log[i],reference.physical_logprob(single,s,ranks,b['pair_mask'][0]),atol=2e-12,rtol=2e-12)
    mass=log[list(unique.values())].exp().sum();torch.testing.assert_close(mass,torch.ones_like(mass),atol=2e-12,rtol=0)
    gradients=torch.autograd.grad(mass,tuple(model.actor.parameters()));assert max(float(g.abs().max()) for g in gradients)<2e-12


def test_critic_is_action_independent_and_core_proposal_rng_is_matched():
    models=[BrokerMarginalPolicy(a) for a in CORE_ARMS];b={key:v.float() if v.dtype!=torch.bool else v for key,v in batch().items()}
    for m in models[1:]:m.load_state_dict(models[0].state_dict())
    actions=[m.sample(b,torch.Generator().manual_seed(21))[0] for m in models]
    assert all(torch.equal(actions[0],a) for a in actions[1:])
    same={key:v.expand(2,*v.shape[1:]).clone() for key,v in b.items()};u=torch.tensor([[1,0],[1,1]])
    assert torch.equal(models[0].evaluate(same,u)[2],models[0].evaluate(same,u.flip(0))[2])
