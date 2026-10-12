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
    cfg=x.settings('smoke');cfg=dict(cfg,env_steps=192,checkpoints=[192],train_seeds=[53800000],development_seeds=cfg['development_seeds'][:1])
    from ht_pdm_fjsp.technician_reservation_distillation import freeze_teachers,prepare_warm_job,DEFAULT_TEACHERS
    from ht_pdm_fjsp.maintenance_opportunity_cost import write_json
    for name in ('serial','parallel'):
        directory=tmp_path/name;directory.mkdir()
        freeze_teachers(DEFAULT_TEACHERS,directory,cfg)
        result=prepare_warm_job((cfg,53800000,str(directory),time.monotonic()))
        write_json(directory/'warm_choices.json',{'53800000':result})
    serial=x.fit_job((cfg,4,'reservation_ppo',53800000,str(tmp_path/'serial'),time.monotonic()))
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:
        parallel=pool.submit(x.fit_job,(cfg,4,'reservation_ppo',53800000,str(tmp_path/'parallel'),time.monotonic())).result()
    assert serial['initial']==parallel['initial']
    assert serial['audit']['parameter_sha256']==parallel['audit']['parameter_sha256']
    assert serial['choice']['development_cost']==parallel['choice']['development_cost']
    assert serial['choice']['steps']==parallel['choice']['steps']
    assert serial['choice']['steps'] in (0,192)
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


def test_teacher_matching_labels_cover_all_catalogue_actions():
    from ht_pdm_fjsp.technician_reservation_distillation import matching_tokens
    env=urgent();ppo=e.ResourceEnv(1,4,'ppo')
    tokens=matching_tokens(np.arange(len(ppo.actions)),ppo.actions)
    for action,target in zip(ppo.actions,tokens):assert env.resolve(target)[0]==action


def test_frozen_teacher_hash_and_metadata(tmp_path):
    from ht_pdm_fjsp.technician_reservation_distillation import freeze_teachers,load_reference,DEFAULT_TEACHERS
    cfg=x.settings('smoke');report=freeze_teachers(DEFAULT_TEACHERS,tmp_path,cfg)
    meta=report['references'][f"ppo:{cfg['train_seeds'][0]}"];model=load_reference(tmp_path,meta)
    assert parameter_digest(model)==meta['parameter_sha256']
    path=tmp_path/meta['path'];path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(RuntimeError,match='frozen teacher hash'):load_reference(tmp_path,meta)


def test_distillation_budget_labels_returns_and_disjoint_streams(tmp_path):
    import time,csv,json
    from ht_pdm_fjsp.technician_reservation_distillation import freeze_teachers,prepare_warm_job,load_reference,audit_bank,DEFAULT_TEACHERS
    from ht_pdm_fjsp.maintenance_opportunity_cost import write_json
    cfg=x.settings('smoke');seed=cfg['train_seeds'][0]
    refs=freeze_teachers(DEFAULT_TEACHERS,tmp_path,cfg)['references']
    result=prepare_warm_job((cfg,seed,str(tmp_path),time.monotonic()))
    assert result['bank']['state_exposures']==192 and result['bank']['physical_ticks']==240
    r,i=result['arms'].values()
    assert r['initial_parameter_sha256']==i['initial_parameter_sha256']
    assert r['bank_sha256']==i['bank_sha256'] and r['teacher_sha256']==i['teacher_sha256']
    assert r['augmentation_sha256']==i['augmentation_sha256']
    assert r['optimizer_steps']==i['optimizer_steps']==8
    assert r['state_presentations']==i['state_presentations']==384
    directory=tmp_path/'warm'/str(seed)
    rows=list(csv.DictReader((directory/'bank_episodes.csv').open()))
    env=e.ResourceEnv(seed,4,'reservation_ppo',0,True);env.reset()
    assert all(int(row['worker'])>=1000 and int(row['physical_seed'])!=env.physical_seed for row in rows)
    data=dict(np.load(directory/'bank.npz'));data['returns']=data['returns'].copy();data['returns'][0]+=1
    with pytest.raises(RuntimeError,match='return audit'):
        audit_bank(data,load_reference(tmp_path,refs[f'ppo:{seed}']),e.ResourceEnv(seed,4,'ppo').actions,cfg)
    write_json(tmp_path/'warm_choices.json',{str(seed):result})
    from ht_pdm_fjsp.technician_reservation_distillation import load_warm
    for n in (4,8,16,32):
        model=x.make_model('reservation_ppo',n,cfg,seed);load_warm(model,tmp_path,seed)
        assert parameter_digest(model)==r['parameter_sha256']


def test_reference_evaluation_never_refits_or_reselects(tmp_path,monkeypatch):
    import time
    from ht_pdm_fjsp.technician_reservation_distillation import freeze_teachers,DEFAULT_TEACHERS
    cfg=x.settings('smoke');seed=cfg['train_seeds'][0];freeze_teachers(DEFAULT_TEACHERS,tmp_path,cfg)
    monkeypatch.setattr(x.learner,'fit',lambda *args,**kw:(_ for _ in ()).throw(AssertionError('reference refit forbidden')))
    for arm in ('ppo','mappo'):
        result=x.fit_job((cfg,4,arm,seed,str(tmp_path),time.monotonic()))
        assert result['audit']['reference'] and result['audit']['learned_steps']==0
        assert result['audit']['foundation_training_steps']==491520
        assert {row['checkpoint_steps'] for row in result['devrows']}=={491520}


def test_graph_attention_capacity_independent_of_scale():
    cfg=x.settings('smoke');counts=[]
    for n in (4,8,16,32):
        model=x.make_model('reservation_ppo',n,cfg,55800000)
        counts.append(sum(p.numel() for p in model.parameters()))
        env=urgent(n);action=model.select(env.observe());env.resolve(action)
    assert len(set(counts))==1


def test_teacher_renaming_preserves_physics_matching_and_observations():
    from ht_pdm_fjsp.technician_reservation_distillation import rename_presentation
    env=urgent();env.env.state=DispatchState((3.,6.,4.,3.),(False,True,False,True),(0,2,0,1),(0,0),(-1,-1))
    original=env.env.state;obs=env.observe();pairs=((0,0),(2,1));labels=torch.tensor([[1,3]])
    events=(True,False,False,True);following,cost,components=transition(env.config,original,pairs,events)
    for seed in range(12):
        batch,target,machines,workers=rename_presentation(tensors([obs]),labels,torch.Generator().manual_seed(seed))
        mo=machines[0].tolist();wo=workers[0].tolist();inverse=np.argsort(mo)
        def permute_state(state):
            return DispatchState(tuple(state.ages[m] for m in mo),tuple(state.failed[m] for m in mo),
                tuple(state.pending_wait[m] for m in mo),tuple(state.remaining[j] for j in wo),
                tuple(int(inverse[state.assigned[j]]) if state.assigned[j]>=0 else -1 for j in wo))
        cfg=replace(env.config,service_time=tuple(tuple(env.config.service_time[m][j] for j in wo) for m in mo),
            restoration=tuple(tuple(env.config.restoration[m][j] for j in wo) for m in mo))
        state=permute_state(original);expected=e.graph_observation(cfg,state,0)
        for key in ('machines','technicians','edges','pair_mask','global_features','state'):
            assert np.array_equal(batch[key][0].numpy(),expected[key])
        renamed=tuple(sorted((int(m)-1,j) for j,m in enumerate(target[0]) if m))
        actual,newcost,newcomponents=transition(cfg,state,renamed,tuple(events[m] for m in mo))
        assert actual==permute_state(following) and newcost==cost and newcomponents==components
        for arm in ('reservation_ppo','independent_ppo'):
            model=x.make_model(arm,4,x.settings('smoke'),57800000);model.evaluate(batch,target)


def test_public_remaining_worker_pool_excludes_consumed_wait_slot_in_both_arms():
    env=urgent();batch=tensors([env.observe()]);cfg=x.settings('smoke')
    for arm in ('reservation_ppo','independent_ppo'):
        model=x.make_model(arm,4,cfg,57800000);edge,_,_,_=model.encode(batch);contexts=[]
        handle=model.actor.register_forward_pre_hook(lambda module,args:contexts.append(args[0].detach().clone()))
        model.evaluate(batch,torch.tensor([[0,0]]));handle.remove()
        expected=edge[:,:,1].mean(1)
        assert torch.allclose(contexts[1][:,:,2*model.hidden:3*model.hidden],expected[:,None].expand(-1,4,-1))


def test_value_scales_with_public_machine_count():
    cfg=x.settings('smoke')
    for n in (4,8,16,32):
        model=x.make_model('reservation_ppo',n,cfg,57800000)
        with torch.no_grad():model.critic[-1].weight.zero_();model.critic[-1].bias.fill_(1.)
        value=model.encode(tensors([urgent(n).observe()]))[-1]
        assert float(value.detach()[0])==n/4


def test_float32_replay_guard_separates_value_rounding_and_actor_drift():
    from ht_pdm_fjsp.maintenance_balanced_algorithms import behavior_replay_audit
    cfg=x.settings('smoke');log=torch.tensor([-2.]);value=torch.tensor([-1024.])
    rounded=torch.nextafter(value,torch.tensor([float('inf')]))
    with pytest.raises(RuntimeError,match='behavior policy mismatch'):
        behavior_replay_audit(log,log,rounded,value,{})
    audit=behavior_replay_audit(log,log,rounded,value,cfg)
    assert audit['behavior_log_max_error']==0 and audit['behavior_value_max_error']>2e-6
    assert audit['behavior_normalized_max_error']<=1
    with pytest.raises(RuntimeError,match='behavior policy mismatch'):
        behavior_replay_audit(log+.001,log,value,value,cfg)
    with pytest.raises(RuntimeError,match='behavior policy mismatch'):
        behavior_replay_audit(log,log,value+.1,value,cfg)


def test_pool_failure_is_visible_before_shutdown_and_cancels_pending(tmp_path):
    import json,time
    from concurrent.futures import Future
    failed=Future();failed.set_exception(RuntimeError('known numerical diagnostic'));pending=Future()
    manifest={'status':'RUNNING'}
    with pytest.raises(RuntimeError,match='known numerical diagnostic'):
        list(x.pool_results([failed,pending],tmp_path,manifest,time.monotonic(),'Expected technical failure'))
    assert pending.cancelled()
    assert json.loads((tmp_path/'manifest.json').read_text())['status']=='FAILED'
    assert json.loads((tmp_path/'abort.json').read_text())['reason']=='TECHNICAL_FAILURE'
    cfg=x.settings('smoke')
    with pytest.raises(RuntimeError,match='sibling job failed'):
        x.fit_job((cfg,4,'reservation_ppo',cfg['train_seeds'][0],str(tmp_path),time.monotonic()))
    from ht_pdm_fjsp.technician_reservation_distillation import prepare_warm_job
    with pytest.raises(RuntimeError,match='sibling job failed'):
        prepare_warm_job((cfg,cfg['train_seeds'][0],str(tmp_path),time.monotonic()))
