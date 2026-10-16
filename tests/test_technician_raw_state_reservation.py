import itertools,time
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_raw_state_reservation_experiment as x
from ht_pdm_fjsp.technician_raw_state_reservation_policy import CORE_ARMS,RawStateReservationPolicy,decode_tokens
from ht_pdm_fjsp.maintenance_dispatch import DispatchState


def model_obs(arm=CORE_ARMS[0]):
    torch.set_num_threads(1);c=x.settings('engineering');t=c['train_seeds'][0];model=x.make_model(arm,c,t)
    env=x.ResourceEnv(270700001,4,arm);env.env.state=DispatchState((3,3,3,3),(False,)*4,(0,)*4,(0,0),(-1,-1));env.env.time=0
    return model,env.observe()


@pytest.mark.parametrize('arm',CORE_ARMS)
def test_exact_conditional_density_normalization_gradients_and_recorded_replay(arm):
    model,obs=model_obs(arm)
    actions=torch.tensor([[a,b] for a,b in itertools.product(range(5),repeat=2) if arm==CORE_ARMS[3] or not(a==b and a>0)],dtype=torch.long)
    batch=x.learner.tensors([obs]*len(actions));log,_,value=model.evaluate(batch,actions)
    mass=log.exp().sum();assert float(mass.detach())==pytest.approx(1,abs=1e-6)
    mass.backward();assert max(float(p.grad.abs().max()) for p in model.actor.parameters())<2e-6
    assert all(p.grad is None for p in model.critic.parameters())
    torch.testing.assert_close(value,value[:1].expand_as(value),atol=2e-5,rtol=0)
    # Same state tensor/layout, different recorded actions: critic is bit-identical.
    alternate_value=model.evaluate(batch,actions.flip(0))[2]
    assert torch.equal(value,alternate_value)
    generator=torch.Generator().manual_seed(81);sample,old_log,old_value=model.sample(batch,generator)
    current_log,_,current_value=model.evaluate(batch,sample)
    torch.testing.assert_close(old_log,current_log,atol=2e-5,rtol=0);torch.testing.assert_close(old_value,current_value,atol=.002304,rtol=1e-6)
    assert sample.shape==(len(actions),2) and sample.dtype==torch.long
    if arm!=CORE_ARMS[3]:
        assert not ((sample[:,0]==sample[:,1]) & (sample[:,0]>0)).any()
        with pytest.raises(ValueError):model.evaluate(x.learner.tensors([obs]),torch.tensor([[1,1]]))
    with pytest.raises(ValueError):model.evaluate(batch,sample.float())


@pytest.mark.parametrize('arm',CORE_ARMS)
def test_prefix_inputs_support_reverse_indexing_and_independent_proposals(arm):
    model,obs=model_obs(arm);captured=[]
    handle=model.actor.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    recorded=torch.tensor([[1,2]])
    try:model.evaluate(x.learner.tensors([obs]),recorded)
    finally:handle.remove()
    assert len(captured)==2
    for c in captured:torch.testing.assert_close(c[0,:71],torch.tensor(obs['state']),atol=0,rtol=0)
    if arm in (CORE_ARMS[1],CORE_ARMS[3]):assert all((c[:,90:102]==0).all() for c in captured)
    elif arm==CORE_ARMS[0]:assert captured[1][0,90]==1 and captured[1][0,98]==1
    else:assert captured[1][0,93]==1 and captured[1][0,99]==1
    expected=[.5,0] if arm==CORE_ARMS[2] else [0,.5]
    assert [float(c[0,-1]) for c in captured]==expected
    if arm==CORE_ARMS[3]:model.evaluate(x.learner.tensors([obs]),torch.tensor([[1,1]]))


def test_busy_forced_wait_rng_contract_and_no_critic_at_deployment(monkeypatch):
    model,obs=model_obs();obs['pair_mask'][:]=False;b=x.learner.tensors([obs])
    actions,log,_=model.sample(b,torch.Generator().manual_seed(3));_,entropy,_=model.evaluate(b,actions)
    assert (actions==0).all() and log.item()==0 and entropy.item()==0
    with pytest.raises(ValueError):model.sample(b)
    with pytest.raises(ValueError):model.sample(b,torch.Generator().manual_seed(3),epsilon=.1)
    def forbidden(*args):raise AssertionError('critic invoked at deployment')
    monkeypatch.setattr(model.critic,'forward',forbidden);assert (model.select(obs)==0).all()


def test_private_critics_equal_fresh_initial_tensors_capacities_and_ppo_pair():
    c=x.settings('engineering');t=c['train_seeds'][0];models={a:x.make_model(a,c,t) for a in x.ARMS}
    assert len({x.learner.parameter_digest(models[a]) for a in CORE_ARMS})==1
    assert all(sum(p.numel() for p in models[a].parameters())==19974 for a in CORE_ARMS)
    critics=[m.critic.net if isinstance(m.critic,x.reference.ValueUnits) else m.critic for m in models.values()]
    assert len({x.raw_tensor_digest(m) for m in critics})==1
    assert len({next(m.parameters()).data_ptr() for m in critics})==8
    assert x.raw_tensor_digest(models['ppo'])==x.raw_tensor_digest(models['normalized_ppo'])
    assert sum(p.numel() for p in models[CORE_ARMS[0]].actor.parameters())==11141
    assert sum(p.numel() for p in critics[0].parameters())==8833


def test_arbitration_invalid_tokens_and_checkpoint_roundtrip(tmp_path):
    model,obs=model_obs();legal=obs['pair_mask'];edges=obs['edges']
    assert decode_tokens(np.array([1,1]),legal,CORE_ARMS[3],edges)[0]==((0,0),)
    with pytest.raises(ValueError):decode_tokens(np.array([1,1]),legal,CORE_ARMS[0],edges)
    with pytest.raises(ValueError):decode_tokens(np.array([1.,2.]),legal,CORE_ARMS[0],edges)
    with pytest.raises(ValueError):decode_tokens(np.array([5,0]),legal,CORE_ARMS[0],edges)
    c=x.settings('engineering');t=c['train_seeds'][0];path=tmp_path/'model.pt'
    x.save_model(model,path,CORE_ARMS[0],c,t,0,obs);loaded=x.load_model(path,CORE_ARMS[0],c,t,0)
    assert np.array_equal(model.select(obs),loaded.select(obs))
    with pytest.raises(FileExistsError):x.save_model(model,path,CORE_ARMS[0],c,t,0,obs)


def test_polynomial_size_and_no_catalogue_on_core_path(monkeypatch):
    import ht_pdm_fjsp.maintenance_heterogeneous_model as catalogue
    def forbidden(*args):raise AssertionError('enumerated matching space')
    monkeypatch.setattr(catalogue,'catalogue',forbidden)
    model,obs=model_obs();env=x.ResourceEnv(270700001,4,CORE_ARMS[0]);o=env.reset();env.step(model.select(o))
    for n,k in ((8,4),(32,16)):
        m=RawStateReservationPolicy(CORE_ARMS[0],n,k)
        expected=64*(m.input_dim+m.state_dim+n+6)+2*64**2+n+2
        assert sum(p.numel() for p in m.parameters())==expected
        b={'state':torch.zeros((2,m.state_dim)),'technicians':torch.zeros((2,k,3)),'edges':torch.zeros((2,n,k,4)),'pair_mask':torch.ones((2,n,k),dtype=torch.bool)}
        a=m.sample(b,torch.Generator().manual_seed(4))[0]
        assert a.shape==(2,k)
        for row in a:
            chosen=row[row>0].tolist();assert len(chosen)==len(set(chosen))


def test_seed_panels_and_physical_pairing():
    audit=x.seed_audit();assert audit['status']=='PASS' and audit['raw_unique']==366 and audit['derived_unique']==82438
    assert x.settings()['test_seeds']==[]
    populations=[]
    for a in x.ARMS:
        env=x.ResourceEnv(270700001,4,a);env.reset();populations.append(env.population)
    assert len(set(populations))==1


def test_end_to_end_fixed_correctness_budget_all_arms_audits_and_frozen_modes(tmp_path):
    """Disposable engineering fixture, not a separate research smoke experiment."""
    c=x.settings('engineering');c.update(env_steps=96,vector_envs=2,rollout=96,epochs=1,minibatch=48,checkpoints=[96])
    t=c['train_seeds'][0];start=time.monotonic();results=[x.fit_job((c,a,t,str(tmp_path),start)) for a in x.ARMS]
    choices={f"{r['method']}:{t}":r['choice'] for r in results};dev=[v for r in results for v in r['development']]
    x.write_json(tmp_path/'checkpoint_choices.json',choices)
    for key in ('training_episodes','training_progress'):x.merge_journals(tmp_path,key,results)
    assert x.audit_training(tmp_path,c,results,choices,dev,lambda:None)['training_episodes']==16
    rows=[];journals={key:x.Journal(tmp_path/f'{key}.csv') for key in ('coordination','decisions')}
    try:
        for label in (*x.ARMS,*x.FROZEN,'deadline'):
            a=CORE_ARMS[0] if label in x.FROZEN else label;choice=choices[f'{a}:{t}'] if a!='deadline' else None
            model=x.load_model(tmp_path/choice['path'],a,c,t,choice['steps']) if choice else None
            before=x.learner.parameter_digest(model) if model else None
            for s in c['development_seeds']:
                rows.append(x.evaluate(label,t if label!='deadline' else -1,s,model,journals['coordination'],journals['decisions']))
                if model:assert model.algorithm==a
            if model:assert before==x.learner.parameter_digest(model)
    finally:
        for j in journals.values():j.close()
    verified=x.audit_selected(tmp_path,c,rows,lambda:None);assert verified['physical_transitions']==1440
    latency=x.isolated_latency(tmp_path,c,choices,lambda:None);assert latency['samples']==144 and (tmp_path/'isolated_timing_samples.csv').exists()
    assert len(x.point_summary(rows,latency)['checks'])==13


def test_serial_and_process_training_identical_tensors_and_choices(tmp_path):
    c=x.settings('engineering');c.update(env_steps=96,vector_envs=2,rollout=96,epochs=1,minibatch=48,checkpoints=[96]);t=c['train_seeds'][0];start=time.monotonic()
    arms=(CORE_ARMS[0],'normalized_ppo');serial=[x.fit_job((c,a,t,str(tmp_path/'serial'),start)) for a in arms]
    with x.ProcessPoolExecutor(max_workers=2,mp_context=x.mp.get_context('spawn')) as pool:
        parallel=list(pool.map(x.fit_job,[(c,a,t,str(tmp_path/'parallel'),start) for a in arms]))
    for s,p in zip(serial,parallel):
        assert s['audit']['parameter_sha256']==p['audit']['parameter_sha256']
        assert s['choice']['steps']==p['choice']['steps'] and s['choice']['development_cost']==p['choice']['development_cost']


def test_gates_preserve_mechanism_and_strict_health_limits():
    rows=[dict(method=a,normalized_cost=101 if a in (CORE_ARMS[3],'mappo','full_mappo') else 100,
        availability=.67,normalized_pending=0,normalized_failures=.1,invalid_assignments=0) for a in (*x.ARMS,*x.FROZEN,'deadline')]
    assert x.point_summary(rows,dict(p95_ms=1))['development_screen_pass']
    rows[0]['normalized_failures']=.4
    assert not x.point_summary(rows,dict(p95_ms=1))['checks']['normalized_failures_delta_vs_ppo']
