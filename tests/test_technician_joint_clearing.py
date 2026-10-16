import itertools,json,time
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_joint_clearing_experiment as x
from ht_pdm_fjsp.technician_joint_clearing_policy import CORE_ARMS,JointClearingPolicy,decode_utility,quantized_utilities


def all_pairs(n,k):
    for tokens in itertools.product(range(n+1),repeat=k):
        chosen=[(m-1,j) for j,m in enumerate(tokens) if m]
        if len({m for m,j in chosen})==len(chosen):yield tuple(sorted(chosen))


def test_flow_exact_partial_matching_against_enumeration_only_in_test():
    rng=np.random.default_rng(42)
    for _ in range(100):
        mask=rng.random((4,2))>.3;u=np.where(mask,rng.normal(size=(4,2)),0);q=quantized_utilities(u,mask)
        pairs,diag=decode_utility(u,mask,CORE_ARMS[0]);opt=max(sum(q[m,j] for m,j in p) for p in all_pairs(4,2) if all(mask[m,j] for m,j in p))
        assert diag['quantized_utility']==opt and sum(q[m,j] for m,j in pairs)==opt
        for arm in CORE_ARMS[1:]:assert decode_utility(u,mask,arm)[1]['quantized_utility']<=opt


def test_joint_substitution_trap_for_both_serial_orders_and_arbitration():
    mask=np.ones((2,2),bool);u=np.array([[9.,8.],[8.,0.]])
    assert decode_utility(u,mask,CORE_ARMS[0])[0]==((0,1),(1,0))
    assert decode_utility(u,mask,CORE_ARMS[1])[0]==((0,0),)
    assert decode_utility(u[:,::-1],mask,CORE_ARMS[2])[1]['quantized_utility']==9_000_000
    pairs,diag=decode_utility(u,mask,CORE_ARMS[3]);assert pairs==((0,0),) and diag['rejected_proposals']==1
    assert decode_utility(np.ones((2,2)),mask,CORE_ARMS[3])[0]==((0,0),)


def test_wait_mask_quantization_and_decoder_input_contract():
    mask=np.ones((4,2),bool)
    for arm in CORE_ARMS:
        assert decode_utility(np.zeros((4,2)),mask,arm)[0]==()
        assert decode_utility(-np.ones((4,2)),mask,arm)[0]==()
        assert decode_utility(np.zeros((4,2)),np.zeros((4,2),bool),arm)[0]==()
    with pytest.raises(ValueError):quantized_utilities(np.ones((4,2)),np.zeros((4,2),bool))
    with pytest.raises(ValueError):quantized_utilities(np.full((4,2),np.nan),mask)
    with pytest.raises(ValueError):quantized_utilities(np.full((4,2),1e7),mask)
    assert quantized_utilities(np.full((4,2),.4e-6),mask).max()==0


def model_obs(arm=CORE_ARMS[0]):
    torch.set_num_threads(1);c=x.settings('engineering');t=c['train_seeds'][0]
    model=x.make_model(arm,c,t);obs=x.reference.mechanism_states('engineering')[0]['observation']
    return model,obs


def test_gaussian_density_support_entropy_gradients_rng_and_full_float_actions():
    m,obs=model_obs();batch=x.learner.tensors([obs]*3)
    a,log,value=m.sample(batch,torch.Generator().manual_seed(123));mu,ls,mask=m.distribution(batch)
    assert a.dtype==torch.float32 and (a[~mask]==0).all() and not torch.equal(a,a.round())
    expected=torch.where(mask,torch.distributions.Normal(mu,ls.exp()).log_prob(a),torch.zeros_like(a)).sum((1,2))
    torch.testing.assert_close(log,expected)
    check,entropy,v=m.evaluate(batch,a);torch.testing.assert_close(check,log);torch.testing.assert_close(v,value)
    assert entropy.shape==(3,) and float(entropy[0].detach())==pytest.approx(.5*np.log(2*np.pi*np.e))
    again=m.sample(batch,torch.Generator().manual_seed(123))[0];assert torch.equal(a,again)
    (-check.mean()-.01*entropy.mean()).backward()
    assert m.log_scale.grad is not None and torch.isfinite(m.log_scale.grad)
    broken=a.clone();broken[~mask]=1
    with pytest.raises(ValueError):m.evaluate(batch,broken)
    with pytest.raises(ValueError):m.sample(batch)


def test_forced_wait_zero_density_entropy_and_no_critic_in_select():
    m,obs=model_obs();obs={k:v.copy() for k,v in obs.items()};obs['pair_mask'][:]=False
    b=x.learner.tensors([obs]);a,log,_=m.sample(b,torch.Generator().manual_seed(1));_,entropy,_=m.evaluate(b,a)
    assert (a==0).all() and log.item()==0 and entropy.item()==0
    def forbidden(*args):raise AssertionError('critic invoked at deployment')
    m.physical_value=forbidden
    assert (m.select(obs)==0).all()


def test_capacity_and_random_tensors_equal_across_all_core_and_ppo_pair():
    c=x.settings('engineering');t=c['train_seeds'][0];models=[x.make_model(a,c,t) for a in CORE_ARMS]
    assert {sum(p.numel() for p in m.parameters()) for m in models}=={19843}
    assert len({x.learner.parameter_digest(m) for m in models})==1
    assert x.raw_tensor_digest(x.make_model('ppo',c,t))==x.raw_tensor_digest(x.make_model('normalized_ppo',c,t))
    assert all(len(m.gradient_groups()['actor'])+len(m.gradient_groups()['critic'])==len(list(m.parameters())) for m in models)


def test_checkpoint_roundtrip_and_frozen_decoder_support(tmp_path):
    m,obs=model_obs();c=x.settings('engineering');t=c['train_seeds'][0];p=tmp_path/'checkpoint.pt'
    meta=x.save_model(m,p,CORE_ARMS[0],c,t,0,obs);loaded=x.load_model(p,CORE_ARMS[0],c,t,0)
    assert meta['sigma']==1 and np.array_equal(m.select(obs),loaded.select(obs))
    for a in CORE_ARMS:decode_utility(m.select(obs),obs['pair_mask'],a)
    with pytest.raises(FileExistsError):x.save_model(m,p,CORE_ARMS[0],c,t,0,obs)


def test_graph_dimension_and_row_equivariance_without_action_catalogue(monkeypatch):
    m,obs=model_obs();batch=x.learner.tensors([obs]);order=torch.tensor([2,0,3,1]);permuted={k:v.clone() for k,v in batch.items()}
    for key in ('machines','edges','pair_mask'):permuted[key]=permuted[key][:,order]
    torch.testing.assert_close(m.distribution(permuted)[0],m.distribution(batch)[0][:,order],atol=1e-6,rtol=0)
    import ht_pdm_fjsp.maintenance_heterogeneous_model as catalogue
    def forbidden(*args):raise AssertionError('enumerated joint actions')
    monkeypatch.setattr(catalogue,'catalogue',forbidden)
    env=x.ResourceEnv(250700001,4,CORE_ARMS[0]);o=env.reset();env.step(m.select(o))
    larger={k:v.clone() for k,v in batch.items()}
    larger['machines']=larger['machines'].repeat(1,2,1);larger['technicians']=larger['technicians'].repeat(1,2,1)
    larger['edges']=larger['edges'].repeat(1,2,2,1);larger['pair_mask']=larger['pair_mask'].repeat(1,2,2)
    assert m.distribution(larger)[0].shape==(1,8,4)
    decode_utility(m.distribution(larger)[0][0].detach().numpy(),larger['pair_mask'][0].numpy(),CORE_ARMS[0])


@pytest.mark.parametrize('arm',CORE_ARMS[:1]+CORE_ARMS[3:])
def test_float_latent_actions_survive_actual_ppo_updates_and_tail(arm):
    torch.set_num_threads(1);c=x.settings('engineering');t=c['train_seeds'][0];m=x.make_model(arm,c,t)
    c.update(env_steps=192,vector_envs=2,rollout=96,epochs=1,minibatch=48,checkpoints=[96,192],
        rng_namespace=f'{x.VERSION}:N4:graph:',value_loss_normalizer=x.value_unit(4),
        behavior_value_atol=2e-5*x.value_unit(4),group_gradient_clipping=True,log_gradient_groups=True)
    updates=[];audit=x.learner.fit(m,c,t,lambda seed,w:x.ResourceEnv(seed,4,arm,w,True),on_checkpoint=lambda model,step,audit:updates.append(step))
    assert updates==[96,192] and audit['changed'] and audit['optimizer_steps']==4 and audit['physical_ticks']==240
    assert audit['behavior_normalized_max_error']<=1


def test_physical_pairing_across_methods_and_no_scientific_test_in_dev():
    tapes=[]
    for arm in x.ARMS:
        env=x.ResourceEnv(250700001,4,arm);env.reset();tapes.append(env.population)
    assert len(set(tapes))==1 and x.settings()['test_seeds']==[]
    assert x.settings('confirmation')['test_seeds'][0]==250500000
    audit=x.seed_audit();assert audit['status']=='PASS' and audit['raw_unique']==366 and audit['derived_unique']==82429


def test_locked_gates_independent_improvement_serial_noninferiority_health():
    rows=[dict(method=a,normalized_cost=100 if a not in ('independent_clearing_ppo','mappo','full_mappo') else 101,
        availability=.67,normalized_pending=0,normalized_failures=.1,invalid_assignments=0)
        for a in (*x.ARMS,'deadline',*x.FROZEN)]
    s=x.point_summary(rows,dict(p95_ms=1));assert len(s['checks'])==13 and s['development_screen_pass']
    rows[0]['normalized_failures']=.4
    assert not x.point_summary(rows,dict(p95_ms=1))['checks']['normalized_failures_delta_vs_ppo']
    rows[0]['normalized_failures']=.1;rows[0]['normalized_cost']=102
    assert not x.point_summary(rows,dict(p95_ms=1))['checks']['gain_vs_forward_clearing_ppo']
    rows[0]['normalized_cost']=100;rows[x.ARMS.index('mappo')]['normalized_cost']=100
    assert not x.point_summary(rows,dict(p95_ms=1))['checks']['gain_vs_mappo']


def test_end_to_end_engineering_budget_selection_and_scalar_audits(tmp_path):
    """Disposable integration fixture, not a research smoke/quality experiment."""
    c=x.settings('engineering');c.update(env_steps=96,vector_envs=2,rollout=96,epochs=1,minibatch=48,checkpoints=[96])
    t=c['train_seeds'][0];start=time.monotonic();results=[x.fit_job((c,a,t,str(tmp_path),start)) for a in x.ARMS]
    choices={f"{r['method']}:{t}":r['choice'] for r in results};dev=[v for r in results for v in r['development']]
    x.merge_journals(tmp_path,'training_episodes',results);x.merge_journals(tmp_path,'training_progress',results)
    audit=x.audit_training(tmp_path,c,results,choices,dev,lambda:None)
    assert audit['training_episodes']==16 and audit['development_episodes']==32
    rows=[];journals={key:x.Journal(tmp_path/f'{key}.csv') for key in ('coordination','decisions')}
    try:
        for label in (*x.ARMS,*x.FROZEN,'deadline'):
            a=x.CORE_ARMS[0] if label in x.FROZEN else label
            choice=choices[f'{a}:{t}'] if a!='deadline' else None
            model=x.load_model(tmp_path/choice['path'],a,c,t,choice['steps']) if choice else None
            for s in c['development_seeds']:rows.append(x.evaluate(label,t if label!='deadline' else -1,s,model,journals['coordination'],journals['decisions']))
    finally:
        for j in journals.values():j.close()
    verified=x.audit_selected(tmp_path,c,rows,lambda:None)
    assert verified['physical_transitions']==24*60 and verified['coordination_decisions']==24*48
    latency=x.isolated_latency(tmp_path,c,choices,lambda:None)
    assert latency['samples']==144 and latency['p95_ms']>0
    summary=x.point_summary(rows,latency)
    assert len(summary['checks'])==13 and summary['test_opened'] is False and summary['goal_pass'] is False


def test_serial_and_spawn_workers_have_identical_tensors_and_choices(tmp_path):
    c=x.settings('engineering');c.update(env_steps=96,vector_envs=2,rollout=96,epochs=1,minibatch=48,checkpoints=[96])
    t=c['train_seeds'][0];start=time.monotonic();arms=(CORE_ARMS[0],'ppo')
    serial=[x.fit_job((c,a,t,str(tmp_path/'serial'),start)) for a in arms]
    with x.ProcessPoolExecutor(max_workers=2,mp_context=x.mp.get_context('spawn')) as pool:
        parallel=list(pool.map(x.fit_job,[(c,a,t,str(tmp_path/'parallel'),start) for a in arms]))
    for before,after in zip(serial,parallel):
        assert before['audit']['parameter_sha256']==after['audit']['parameter_sha256']
        assert before['choice']['steps']==after['choice']['steps'] and before['choice']['development_cost']==after['choice']['development_cost']
