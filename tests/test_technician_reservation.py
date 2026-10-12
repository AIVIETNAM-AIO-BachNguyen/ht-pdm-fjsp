from dataclasses import asdict,replace
import itertools
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_reservation_experiment as x
from ht_pdm_fjsp import technician_reservation_env as e
from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors,parameter_digest
from ht_pdm_fjsp.maintenance_balanced_benchmark_env import BenchmarkEnv,observation
from ht_pdm_fjsp.maintenance_dispatch import DispatchState,validate_matching
from ht_pdm_fjsp.maintenance_waiting import transition


def urgent(n=4,method='reservation_ppo'):
    env=e.ResourceEnv(51700123,n,method);env.reset();k=n//2
    env.env.state=DispatchState((3,)*n,(False,)*n,(0,)*n,(0,)*k,(-1,)*k)
    return env


def test_small_masks_and_exact_state_match_original():
    env=urgent();original=BenchmarkEnv(1);original.reset();original.inner.env.state=env.env.state
    old=observation(original.inner);new=env.observe()
    for key in ('state','local','local_mask'):assert np.array_equal(old[key],new[key])
    for a in original.inner.actions:
        token=np.zeros(2,dtype=int)
        for m,j in a:token[j]=m+1
        pairs,diag=env.resolve(token)
        assert pairs==a and not diag['conflicts']


def test_no_catalogue_at_large_scale(monkeypatch):
    import ht_pdm_fjsp.maintenance_heterogeneous_model as h
    monkeypatch.setattr(h,'catalogue',lambda *a:(_ for _ in ()).throw(AssertionError('catalogue forbidden')))
    for n in (8,16,32):
        env=urgent(n);obs=env.observe();assert obs['pair_mask'].shape==(n,n//2)
        action=e.deadline(env.config,env.env.state);validate_matching(env.config,env.env.state,action)
        assert len(action)==n//2
        model=x.make_model('reservation_ppo',n,x.settings('smoke'),51700000)
        for _ in range(48):_,_,done,info=env.step(model.select(env.observe()))
        assert done and info['episode_metrics']['physical_ticks']==60


def test_trace_probability_normalizes_and_replays():
    torch.set_num_threads(1);env=urgent();batch=tensors([env.observe()]);cfg=x.settings('smoke')
    for arm in ('reservation_ppo','independent_ppo'):
        model=x.make_model(arm,4,cfg,51700000);mass=0.
        for tokens in itertools.product(range(5),repeat=2):
            recorded=torch.tensor([tokens])
            if arm=='reservation_ppo' and tokens[0] and tokens[0]==tokens[1]:
                with pytest.raises(ValueError):model.evaluate(batch,recorded)
                continue
            log,entropy,_=model.evaluate(batch,recorded);mass+=float(log.detach().exp())
            assert entropy.isfinite().all()
        assert mass==pytest.approx(1.,abs=1e-6)
        act,log,val=model.sample(batch,torch.Generator().manual_seed(42))
        check,_,checkval=model.evaluate(batch,act)
        assert torch.equal(log,check) and torch.equal(val,checkval)
        loss=-(check.mean())+checkval.square().mean();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_equal_initial_parameters_ablation_and_no_future_inputs():
    cfg=x.settings('smoke');a=x.make_model('reservation_ppo',4,cfg,51700000);b=x.make_model('independent_ppo',4,cfg,51700000)
    assert parameter_digest(a)==parameter_digest(b)
    env=urgent();before=env.observe();env.env.events=((True,)*4,)*60
    assert all(np.array_equal(before[k],env.observe()[k]) for k in before)
    original=env.env.state;env.env.state=replace(original,ages=(3,3,3,3),remaining=(0,2),assigned=(-1,1))
    mask=env.observe()['pair_mask'];assert not mask[1].any() and not mask[:,1].any()
    with pytest.raises(ValueError):env.step(np.array([1,2]))


def test_duplicate_proposals_lost_without_reassignment():
    env=urgent(method='independent_ppo');pairs,d=env.resolve(np.array([1,1]))
    assert pairs==((0,0),) and d['conflicts']==d['rejected_proposals']==1
    assert env.resolve(np.array([1,2]))[1]['conflicts']==0


def test_sixty_tick_scalar_physics_and_tail_reconciliation():
    env=urgent();state=env.env.state;total=0.;reward=0.
    for tick in range(48):
        candidates=np.argwhere(env.observe()['pair_mask']);token=np.zeros(2,dtype=int)
        if len(candidates):m,j=candidates[-1];token[j]=m+1
        action,_=env.resolve(token);state,cost,_=transition(env.config,state,action,env.env.events[tick]);total+=cost
        _,r,done,info=env.step(token);reward+=r
        if tick<47:assert env.env.state==state
    for tick in range(48,60):
        action=e.deadline(env.config,state);state,cost,_=transition(env.config,state,action,env.env.events[tick]);total+=cost
    assert done and env.env.state==state and -20*reward==pytest.approx(total)
    assert info['episode_metrics']['extended_cost']==total


@pytest.mark.parametrize('n,arm',[(4,a) for a in x.ARMS]+[(8,'mappo'),(8,'reservation_ppo')])
def test_checkpoint_roundtrip(tmp_path,n,arm):
    cfg=x.settings('smoke');model=x.make_model(arm,n,cfg,51700000);probe=e.ResourceEnv(51800000,n,arm).reset()
    path=tmp_path/'model.pt';x.save_model(model,path,n,cfg,51700000,192,probe)
    loaded=x.load_model(path,n,cfg,51700000,arm,192)
    assert np.array_equal(model.select(probe),loaded.select(probe))
    with pytest.raises(FileExistsError):x.save_model(model,path,n,cfg,51700000,192,probe)


def test_panels_and_exact_family():
    assert x.seed_audit()['status']=='PASS'
    assert x.settings('development')['test_seeds']==[]
    assert len(x.settings('confirmation')['train_seeds'])==5
    means={n:{a:np.array([250.,.67,0.,.1]) for a in (*x.arms(n),'deadline')} for n in (4,8,16,32)}
    assert len(x.estimates(means,[4,8,16,32]))==24


def test_refuse_overwrite(tmp_path):
    (tmp_path/'prior').write_text('keep')
    with pytest.raises(FileExistsError):x.run('smoke',tmp_path)


def test_value_gradients_cannot_change_actor_encoder():
    model=x.make_model('reservation_ppo',4,x.settings('smoke'),53800000)
    env=urgent();batch=tensors([env.observe()]);action,_,_=model.sample(batch,torch.Generator().manual_seed(1))
    _,_,value=model.evaluate(batch,action);value.square().mean().backward()
    assert all(p.grad is None for p in model.edge_encoder.parameters())
    assert all(p.grad is None for p in model.actor.parameters())
    assert any(p.grad is not None for p in model.value_edge_encoder.parameters())


def test_serial_and_spawn_training_identical(tmp_path):
    import time,multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    cfg=x.settings('smoke');cfg=dict(cfg,env_steps=192,checkpoints=[192],development_seeds=cfg['development_seeds'][:1])
    serial=x.fit_job((cfg,4,'reservation_ppo',53800000,str(tmp_path/'serial'),time.monotonic()))
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:
        parallel=pool.submit(x.fit_job,(cfg,4,'reservation_ppo',53800000,str(tmp_path/'parallel'),time.monotonic())).result()
    assert serial['initial']==parallel['initial']
    assert serial['audit']['parameter_sha256']==parallel['audit']['parameter_sha256']
    assert serial['choice']['development_cost']==parallel['choice']['development_cost']
    assert serial['choice']['steps']==parallel['choice']['steps']==192
    for key in ('learned_steps','optimizer_steps','training_episodes','behavior_max_error'):
        assert serial['audit'][key]==parallel['audit'][key]


def test_mechanism_gate_requires_every_large_scale_and_small_noninferiority():
    cfg=dict(x.settings('smoke'),scales=[4,8,16,32])
    rows=[];latencies={}
    for n in cfg['scales']:
        for arm in (*x.arms(n),'deadline'):
            latencies[f'{n}:{arm}']=[1.]*4
            cost=100. if arm in ('reservation_ppo','ppo','deadline') or n==4 and arm=='independent_ppo' else 110.
            replicas=[-1] if arm=='deadline' else cfg['train_seeds']
            for t in replicas:
                for s in cfg['test_seeds']:
                    rows.append(dict(n=n,method=arm,train_seed=t,seed=s,normalized_cost=cost,
                        availability=.67,normalized_pending=0.,normalized_failures=.1))
    summary=x.summarize(rows,cfg,latencies)
    assert not summary['guards']['small_mechanism'] and summary['guards']['small_ablation_noninferiority']
    assert summary['goal_pass']
    damaged=[dict(r,normalized_cost=100.) if r['n']==16 and r['method']=='independent_ppo' else r for r in rows]
    assert not x.summarize(damaged,cfg,latencies)['goal_pass']
    damaged=[dict(r,normalized_pending=.1) if r['n']==8 and r['method']=='reservation_ppo' else r for r in rows]
    assert not x.summarize(damaged,cfg,latencies)['goal_pass']
