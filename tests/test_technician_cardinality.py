from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_cardinality_experiment as x
from ht_pdm_fjsp.technician_cardinality_policy import opportunity_prices
from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors,parameter_digest
from ht_pdm_fjsp.maintenance_dispatch import DispatchState,validate_matching
from ht_pdm_fjsp.maintenance_waiting import transition
from ht_pdm_fjsp.technician_reservation_env import deadline

@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)

def urgent(n=4,a='scarcity_ppo'):
    env=x.ResourceEnv(68100000,n,a);env.reset();k=n//2
    env.env.state=DispatchState((3,)*n,(False,)*n,(0,)*n,(0,)*k,(-1,)*k)
    return env

@pytest.mark.parametrize('arm',['scarcity_ppo','reservation_ppo','flat_stop_ppo'])
def test_exact_trace_mass_and_physical_action_support(arm):
    env=urgent(a=arm);batch=tensors([env.observe()]);model=x.make_model(arm,4,x.settings('smoke'),68000000)
    mass=0.;support=set();count=0
    for trace in itertools.product(range(9),repeat=2):
        try:pairs,_=env.resolve(np.array(trace))
        except ValueError:
            with pytest.raises(ValueError):model.evaluate(batch,torch.tensor([trace]))
            continue
        log,entropy,_=model.evaluate(batch,torch.tensor([trace]));mass+=float(log.detach().exp());support.add(pairs);count+=1
        assert torch.isfinite(entropy).all()
    assert count==33 and len(support)==21 and mass==pytest.approx(1.,abs=2e-6)

def test_stop_empty_resources_and_noncanonical_suffix():
    env=urgent();env.env.state=replace(env.env.state,ages=(0,)*4)
    model=x.make_model('scarcity_ppo',4,x.settings('smoke'),68000000);batch=tensors([env.observe()])
    action,log,value=model.sample(batch,torch.Generator().manual_seed(3))
    assert not action.any() and not log.any()
    assert torch.equal(log,model.evaluate(batch,action)[0])
    with pytest.raises(ValueError):env.resolve(np.array([0,1]))
    with pytest.raises(ValueError):env.resolve(np.array([1,1]))

def test_scarce_worker_message_and_finite_gradient():
    bids=torch.tensor([[[1.,0.],[1.,0.],[1.,0.],[0.,0.]]],requires_grad=True)
    wait=torch.full((1,4),-2.,requires_grad=True)
    legal=torch.tensor([[[True,False],[True,False],[True,False],[True,True]]])
    duration=torch.ones_like(bids);gate=torch.tensor(0.,requires_grad=True)
    price=opportunity_prices(bids,wait,legal,duration,gate)
    assert price[0,3,0]>price[0,3,1] and price.max()<=.5
    assert price[0,3,0]>opportunity_prices(bids,wait,legal,duration*.25,gate)[0,3,0]
    price.square().sum().backward()
    assert torch.isfinite(bids.grad).all() and bids.grad.abs().sum()>0
    assert torch.isfinite(wait.grad).all() and torch.isfinite(gate.grad)
    assert not opportunity_prices(bids.detach(),wait.detach(),torch.zeros_like(legal),duration,gate.detach()).any()
    singleton=legal.clone();singleton[:,1:]=False
    assert not opportunity_prices(bids.detach(),wait.detach(),singleton,duration,gate.detach()).any()

@pytest.mark.parametrize('n',[4,8,16,32])
def test_equal_initialization_capacity_and_sample_replay(n):
    c=x.settings('smoke');models=[x.make_model(a,n,c,68000000) for a in x.GRAPH_ARMS]
    assert len({parameter_digest(m) for m in models})==1
    assert {sum(p.numel() for p in m.parameters()) for m in models}=={19781}
    for model in models:
        env=urgent(n,model.algorithm);batch=tensors([env.observe()])
        action,log,value=model.sample(batch,torch.Generator().manual_seed(42));other,_,v=model.evaluate(batch,action)
        assert torch.equal(log,other) and torch.equal(value,v)
        (-other.mean()+v.square().mean()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        validate_matching(env.config,env.env.state,env.resolve(action[0].numpy())[0])

def test_no_identity_features_permutation_equivariance():
    env=urgent();obs=env.observe();m=np.array([2,0,3,1]);t=np.array([1,0]);permuted={k:v.copy() for k,v in obs.items()}
    permuted['machines']=obs['machines'][m];permuted['technicians']=obs['technicians'][t]
    permuted['edges']=obs['edges'][m][:,t];permuted['pair_mask']=obs['pair_mask'][m][:,t]
    for arm in ('scarcity_ppo','reservation_ppo'):
        model=x.make_model(arm,4,x.settings('smoke'),68000000);old=model.encode(tensors([obs]));new=model.encode(tensors([permuted]))
        assert torch.allclose(old[2][:,m][:,:,t],new[2],atol=2e-6,rtol=0)
        trace=[1,4];mapped=[int(np.where(m==(z-1)//2)[0][0])*2+int(np.where(t==(z-1)%2)[0][0])+1 for z in trace]
        a=model.evaluate(tensors([obs]),torch.tensor([trace]));b=model.evaluate(tensors([permuted]),torch.tensor([mapped]))
        assert torch.allclose(a[0],b[0],atol=2e-6,rtol=0) and torch.allclose(a[2],b[2],atol=2e-6,rtol=0)

def test_no_future_events_and_no_large_catalogue(monkeypatch):
    import ht_pdm_fjsp.maintenance_heterogeneous_model as h
    monkeypatch.setattr(h,'catalogue',lambda *a:(_ for _ in ()).throw(AssertionError('catalogue forbidden')))
    for n in (8,16,32):
        for a in x.GRAPH_ARMS:
            env=urgent(n,a);obs=env.observe();env.env.events=((True,)*n,)*60
            assert all(np.array_equal(v,env.observe()[k]) for k,v in obs.items())
            model=x.make_model(a,n,x.settings('smoke'),68000000)
            assert env.actions is None
            validate_matching(env.config,env.env.state,env.resolve(model.select(obs))[0])
            validate_matching(env.config,env.env.state,deadline(env.config,env.env.state))

@pytest.mark.parametrize('arm',x.arms(4))
def test_checkpoint_and_sixty_tick_replay(tmp_path,arm):
    c=x.settings('smoke');env=urgent(a=arm);model=x.make_model(arm,4,c,68000000);probe=env.observe()
    path=tmp_path/'checkpoint.pt';x.save_model(model,path,4,c,68000000,0,probe)
    with pytest.raises(FileExistsError):x.save_model(model,path,4,c,68000000,0,probe)
    with pytest.raises(RuntimeError):x.load_model(path,4,c,68000000,arm,192)
    state=env.env.state;cost=reward=0.
    for tick in range(48):
        action=model.select(env.observe());pairs,_=env.resolve(action);state,v,_=transition(env.config,state,pairs,env.env.events[tick]);cost+=v
        _,r,done,info=env.step(action);reward+=r
        if tick<47:assert state==env.env.state
    for tick in range(48,60):
        state,v,_=transition(env.config,state,deadline(env.config,state),env.env.events[tick]);cost+=v
    assert done and state==env.env.state and cost==info['episode_metrics']['extended_cost'] and -20*reward==pytest.approx(cost)

def test_independent_collision_and_full_information_control():
    env=urgent(a='independent_ppo');pairs,diag=env.resolve(np.array([1,1,1,1]))
    assert len(pairs)==1 and diag['rejected_proposals']==3 and diag['conflicts']==1
    assert 'full_local' not in env.observe()
    obs=urgent(a='full_mappo').observe();assert obs['full_local'].shape==(4,75)
    assert np.array_equal(obs['full_local'][:,:71],np.tile(obs['state'],(4,1)))

def test_fresh_seed_roles_and_confirmation_guard(tmp_path):
    assert x.seed_audit()['status']=='PASS'
    c=x.settings('confirmation');assert c['teacher_assistance'] is False and c['pretraining_steps']==0
    assert c['confirmation_round']==2 and c['confidence']==1-(.05/6)/35
    assert c['bootstrap_samples']==200000
    with pytest.raises(ValueError):x.validate_prerequisite(None,c,{})
    d=tmp_path/'dev';d.mkdir()
    for name,data in [('manifest.json',dict(protocol=x.VERSION,status='COMPLETED',audit_status='PASS',test_opened=False)),('summary.json',dict(development_screen_pass=False)),('resolved_config.json',dict(profile='development'))]:
        (d/name).write_text(json.dumps(data))
    with pytest.raises(ValueError):x.validate_prerequisite(d,c,{})

def test_matched_serial_spawn_random_training_no_teacher(tmp_path,monkeypatch):
    import ht_pdm_fjsp.technician_reservation_distillation as teacher
    for name in ('load_warm','freeze_teachers','prepare_warm_job'):
        monkeypatch.setattr(teacher,name,lambda *a,**kw:(_ for _ in ()).throw(AssertionError('teacher forbidden')))
    c=x.settings('smoke');c['development_seeds']=c['development_seeds'][:1]
    import time
    left=tmp_path/'serial';right=tmp_path/'spawn';left.mkdir();right.mkdir()
    serial=x.fit_job((c,4,'scarcity_ppo',68000000,str(left),time.monotonic()))
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:
        parallel=pool.submit(x.fit_job,(c,4,'scarcity_ppo',68000000,str(right),time.monotonic())).result(timeout=120)
    assert serial['initial']==parallel['initial']
    assert serial['choice']['sha256']==parallel['choice']['sha256']
    assert serial['audit']['parameter_sha256']==parallel['audit']['parameter_sha256']
    assert serial['audit']['learned_steps']==384 and serial['audit']['pretraining_steps']==0
    assert serial['audit']['from_scratch'] and not list(tmp_path.rglob('warm*'))
    import shutil,csv
    shutil.copyfile(left/'N4/scarcity_ppo/68000000/training_episodes.csv',left/'training_episodes.csv')
    c=dict(c,scales=[4],test_seeds=[]);monkeypatch.setattr(x,'arms',lambda n:('scarcity_ppo',))
    audited=x.audit_run(left,c,serial['devrows'],[],[serial['audit']],{'choice':serial['choice']})
    assert audited['status']=='PASS' and not audited['teacher_assistance']
    with (left/'training_episodes.csv').open() as f:rows=list(csv.DictReader(f))
    rows[0]['physical_seed']=str(int(rows[0]['physical_seed'])+1)
    with (left/'training_episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    with pytest.raises(RuntimeError,match='seed namespace'):x.audit_run(left,c,serial['devrows'],[],[serial['audit']],{'choice':serial['choice']})

def test_complete_statistical_family_null_and_missing_cell():
    c=dict(x.settings('smoke'),scales=[4,8,16,32],train_seeds=[1,2],test_seeds=[3,4],bootstrap_samples=200)
    rows=[dict(n=n,method=a,train_seed=t,seed=s,normalized_cost=100.,availability=.7,normalized_pending=0.,normalized_failures=0.)
          for n in c['scales'] for a,t in [(a,t) for a in x.arms(n) for t in c['train_seeds']]+[('deadline',-1)] for s in c['test_seeds']]
    latencies={f'{n}:{a}':[1.,2.] for n in c['scales'] for a in (*x.arms(n),'deadline')}
    summary=x.summarize(rows,c,latencies)
    assert summary['family_size']==35 and not summary['goal_pass'] and not summary['guards']['price_mechanism']
    assert summary['intervals']['pooled_price_gain']['estimate']==pytest.approx(0.)
    with pytest.raises(RuntimeError,match='test grid'):x.summarize(rows[:-1],c,latencies)


def test_group_clipping_does_not_rescale_actor_for_large_critic():
    from ht_pdm_fjsp.maintenance_balanced_algorithms import clip_gradient_groups
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.actor=torch.nn.Parameter(torch.tensor(0.));self.critic=torch.nn.Parameter(torch.tensor(0.))
        def gradient_groups(self):return {'actor':[self.actor],'critic':[self.critic]}
    model=Toy();model.actor.grad=torch.tensor(.2);model.critic.grad=torch.tensor(100.)
    clip_gradient_groups(model,.5,True)
    assert model.actor.grad==pytest.approx(.2) and model.critic.grad==pytest.approx(.5,abs=1e-6)
    model.actor.grad=torch.tensor(.2);model.critic.grad=torch.tensor(100.)
    clip_gradient_groups(model,.5,False)
    assert model.actor.grad<.002

@pytest.mark.parametrize('grouped',[False,True])
def test_gradient_logging_does_not_change_learning(grouped):
    from ht_pdm_fjsp.maintenance_balanced_algorithms import fit
    c=dict(x.settings('smoke'),group_gradient_clipping=grouped)
    digests=[]
    for logging in (False,True):
        model=x.make_model('scarcity_ppo',4,c,68000000)
        audit=fit(model,dict(c,log_gradient_groups=logging),68000000,
            lambda seed,worker:x.ResourceEnv(seed,4,'scarcity_ppo',worker,True))
        digests.append(audit['parameter_sha256'])
    assert digests[0]==digests[1]

@pytest.mark.parametrize('arm',['scarcity_ppo','flat_stop_ppo'])
def test_counterfactual_observer_preserves_model_identity_and_actual_diagnostics(arm,monkeypatch):
    model=x.make_model(arm,4,x.settings('smoke'),68000000)
    digest=parameter_digest(model);original_select=model.select;actual=[]
    def select(obs,disable_prices=False):
        action=original_select(obs,disable_prices=disable_prices)
        if not disable_prices:actual.append(dict(model.last_diagnostics))
        return action
    monkeypatch.setattr(model,'select',select)
    x.evaluate(arm,4,68000000,68100000,model)
    assert model.algorithm==arm
    assert len(actual)==48
    assert model.last_diagnostics==actual[-1]
    assert parameter_digest(model)==digest

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_hierarchical_gate_is_invariant_to_edge_count_and_empty_support(dtype):
    from ht_pdm_fjsp.technician_cardinality_policy import broker_log_probabilities
    for count in (1,8,512):
        stop=torch.tensor([-.7],dtype=dtype,requires_grad=True)
        edges=torch.zeros((1,count),dtype=dtype,requires_grad=True)
        legal=torch.ones_like(edges,dtype=torch.bool)
        log=broker_log_probabilities(stop,edges,legal,True)
        assert log.exp().sum().detach().item()==pytest.approx(1.,abs=1e-6)
        # FP32 normalization can round by several ULPs across512 alternatives.
        tolerance=1e-6 if dtype==torch.float32 else 1e-12
        assert log[0,0].exp().item()==pytest.approx(torch.sigmoid(stop).item(),abs=tolerance)
        (-log[0,1]).backward()
        assert torch.isfinite(stop.grad).all() and torch.isfinite(edges.grad).all()
        empty=broker_log_probabilities(stop.detach(),edges.detach(),torch.zeros_like(legal),True)
        assert empty[0,0]==0 and empty.exp()[0,1:].sum()==0

def test_conditional_greedy_follows_go_mass_even_when_flat_mode_is_stop():
    from ht_pdm_fjsp.technician_cardinality_policy import broker_log_probabilities,greedy_broker_action
    stop=torch.tensor([-1.,0.,1.]);edges=torch.full((3,8),-2.);legal=torch.ones_like(edges,dtype=torch.bool)
    h=broker_log_probabilities(stop,edges,legal,True)
    flat=broker_log_probabilities(stop,edges,legal,False)
    assert h[0,0].exp()<.5 and h[0].argmax()==0
    assert greedy_broker_action(stop,h,legal,True).tolist()==[1,0,0]
    assert greedy_broker_action(stop,flat,legal,False).tolist()==[0,0,0]
    legal[0]=False
    assert greedy_broker_action(stop,h,legal,True)[0]==0

def test_flat_control_matches_original_categorical_and_preserves_unpriced_factorization():
    from ht_pdm_fjsp.technician_cardinality_policy import broker_log_probabilities
    stop=torch.tensor([-.8]);edges=torch.tensor([[.2,.7,-.2]])
    legal=torch.tensor([[True,False,True]])
    log=broker_log_probabilities(stop,edges,legal,False)
    original=torch.distributions.Categorical(logits=torch.tensor([[-.8,.2,-1e9,-.2]])).logits
    assert torch.equal(log,original)
    model=x.make_model('flat_stop_ppo',4,x.settings('smoke'),68000000)
    obs=urgent(a='flat_stop_ppo').observe();old=model.algorithm
    model.select(obs,disable_prices=True)
    assert model.algorithm==old and model.last_diagnostics['price_mean_abs']==0

def test_both_registered_mechanisms_are_required():
    c=x.settings('smoke');means={n:{a:np.array([100.,.7,0.,0.]) for a in (*x.arms(n),'deadline')} for n in (4,8,16,32)}
    point=x.estimates(means,[4,8,16,32]);intervals={k:dict(lower=v,upper=v) for k,v in point.items()}
    timing={f'{n}:scarcity_ppo':dict(p95_ms=1.) for n in means}
    checks=x.guards(intervals,list(means),timing)
    assert not checks['price_mechanism'] and not checks['cardinality_mechanism']
    assert not any('optional_cardinality' in k for k in checks)
