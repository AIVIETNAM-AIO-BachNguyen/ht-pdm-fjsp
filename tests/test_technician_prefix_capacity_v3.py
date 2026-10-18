import numpy as np
import pytest
import torch

from ht_pdm_fjsp import technician_prefix_capacity_v3_experiment as x
from ht_pdm_fjsp import technician_prefix_capacity_v3_env as e
from ht_pdm_fjsp.maintenance_balanced_algorithms import parameter_digest, tensors
from ht_pdm_fjsp.maintenance_dispatch import DispatchState


def test_relational_prefix_and_mask_share_feasible_stages():
    torch.set_num_threads(1)
    cfg=x.settings('smoke')
    models={arm:x.make_model(arm,4,cfg,67800000) for arm in x.GRAPH_ARMS}
    assert len({parameter_digest(model) for model in models.values()})==1
    env=e.ResourceEnv(67900000,4,'reservation_ppo');env.reset()
    env.env.state=DispatchState((3,)*4,(False,)*4,(0,)*4,(0,)*2,(-1,)*2)
    batch=tensors([env.observe()]);action=torch.tensor([[1,2]])
    traces={}
    for arm in ('reservation_ppo','mask_only_ppo'):
        traces[arm]=[]
        models[arm].trace(batch,recorded=action,audit=traces[arm])
    for full,masked in zip(traces['reservation_ppo'],traces['mask_only_ppo'],strict=True):
        for field in ('legal','resource','stats'):
            assert torch.equal(full[field],masked[field])
        assert not torch.count_nonzero(masked['prefix'])
        assert not torch.count_nonzero(masked['candidate_capacity'])
        assert not torch.count_nonzero(masked['wait_capacity'])
        assert torch.isfinite(full['prefix']).all()
    assert traces['reservation_ppo'][1]['prefix'].shape==(1,4,64)
    assert torch.count_nonzero(traces['reservation_ppo'][1]['prefix'])
    assert torch.count_nonzero(traces['reservation_ppo'][1]['candidate_capacity'])
    assert torch.count_nonzero(traces['reservation_ppo'][1]['wait_capacity'])
    assert not traces['reservation_ppo'][1]['legal'][0,1]
    assert env.resolve(action[0].numpy())[0]==((0,0),(1,1))


def test_empty_prefix_wait_and_duplicate_contract():
    cfg=x.settings('smoke');env=e.ResourceEnv(67900000,4,'reservation_ppo');env.reset()
    env.env.state=DispatchState((3,)*4,(False,)*4,(0,)*4,(0,)*2,(-1,)*2)
    batch=tensors([env.observe()]);full=x.make_model('reservation_ppo',4,cfg,67800000)
    trace=[]
    full.trace(batch,recorded=torch.tensor([[0,1]]),audit=trace)
    assert not torch.count_nonzero(trace[1]['prefix'])
    assert not torch.count_nonzero(trace[1]['candidate_capacity'])
    assert not torch.count_nonzero(trace[1]['wait_capacity'])
    assert torch.isfinite(full.evaluate(batch,torch.tensor([[0,1]]))[0]).all()
    for arm in ('reservation_ppo','mask_only_ppo'):
        with pytest.raises(ValueError,match='illegal recorded'):
            x.make_model(arm,4,cfg,67800000).evaluate(batch,torch.tensor([[1,1]]))
    x.make_model('independent_ppo',4,cfg,67800000).evaluate(batch,torch.tensor([[1,1]]))


def test_selected_edge_attention_depends_on_candidate_at_n8():
    cfg=x.settings('smoke');env=e.ResourceEnv(67900000,8,'reservation_ppo');env.reset()
    env.env.state=DispatchState((3,)*8,(False,)*8,(0,)*8,(0,)*4,(-1,)*4)
    batch=tensors([env.observe()]);action=torch.tensor([[1,2,3,4]])
    full=x.make_model('reservation_ppo',8,cfg,67800000)
    masked=x.make_model('mask_only_ppo',8,cfg,67800000)
    ft=[];mt=[]
    full.trace(batch,recorded=action,audit=ft)
    masked.trace(batch,recorded=action,audit=mt)
    assert torch.equal(ft[2]['legal'],mt[2]['legal'])
    assert ft[2]['prefix'].shape==(1,8,64)
    assert not torch.count_nonzero(mt[2]['prefix'])
    assert not torch.allclose(ft[2]['prefix'][0,4],ft[2]['prefix'][0,5])


def test_fresh_panels_and_sealed_gate(tmp_path):
    assert x.seed_audit()['status']=='PASS'
    assert x.settings('development')['scales']==[4,8]
    assert not x.settings('development')['test_seeds']
    assert len(x.settings('confirmation')['train_seeds'])==5
    with pytest.raises(ValueError,match='development artifact'):
        x.run('confirmation',tmp_path/'unopened')


def test_primary_summary_requires_matched_mask_gain():
    cfg=dict(x.settings('smoke'),scales=[4,8],train_seeds=[67800000],test_seeds=[68000000,68000001])
    rows=[];latencies={}
    for n in cfg['scales']:
        for arm in (*x.arms(n),'deadline'):
            latencies[f'{n}:{arm}']=[1.]*4
            cost={'reservation_ppo':100.,'mask_only_ppo':102.,'independent_ppo':110.}.get(arm,100.)
            for train in ([-1] if arm=='deadline' else cfg['train_seeds']):
                for seed in cfg['test_seeds']:
                    rows.append(dict(n=n,method=arm,train_seed=train,seed=seed,
                        normalized_cost=cost,availability=.67,normalized_pending=0.,normalized_failures=.1))
    assert x.summarize(rows,cfg,latencies)['goal_pass']
    for row in rows:
        if row['method']=='mask_only_ppo':row['normalized_cost']=100.
    assert not x.summarize(rows,cfg,latencies)['goal_pass']


def test_independent_resampling_requires_matched_gain():
    from scripts.verify_technician_prefix_capacity_v3 import independently_resample
    config=dict(profile='smoke',scales=[4,8],train_seeds=[67800000],
        test_seeds=[68000000,68000001],analysis_seed=68100000)
    rows={};coordination=[]
    for n in config['scales']:
        for seed in config['test_seeds']:
            for arm,cost in (('reservation_ppo',98.),('mask_only_ppo',100.)):
                rows[n,arm,67800000,seed]=dict(normalized_cost=cost,availability=.67,
                    normalized_pending=0.,normalized_failures=.1)
            coordination.append(dict(n=n,method='reservation_ppo',decision_ms=1.))
    assert independently_resample(rows,config,coordination)['goal_pass']
    for key,row in rows.items():
        if key[1]=='reservation_ppo':row['normalized_cost']=100.
    assert not independently_resample(rows,config,coordination)['goal_pass']


def test_capacity_scalars_match_selected_pair_physics():
    cfg=x.settings('smoke');env=e.ResourceEnv(67900000,4,'reservation_ppo');env.reset()
    env.env.state=DispatchState((3,)*4,(False,)*4,(0,)*4,(0,)*2,(-1,)*2)
    obs=env.observe();batch=tensors([obs]);recorded=torch.tensor([[1,2]])
    full=x.make_model('reservation_ppo',4,cfg,67800000);audit=[]
    full.trace(batch,recorded=recorded,audit=audit)
    duration=obs['edges'][0,0,1]*3
    restoration=obs['edges'][0,0,2]
    expected=torch.tensor([duration/2,duration*(1-restoration)/2])
    assert torch.allclose(audit[1]['wait_capacity'][0],expected)
    for m in range(4):
        assert torch.allclose(audit[1]['candidate_capacity'][0,m],expected*obs['edges'][m,1,1]*3)


def test_confirmation_source_guard_rejects_changed_dev_snapshot(tmp_path, monkeypatch):
    prior=tmp_path/'dev';snapshot=prior/'source_snapshot';snapshot.mkdir(parents=True)
    live=tmp_path/'live';live.mkdir()
    for name,content in [('protocol.md','locked protocol'),('seed_registry.json','locked seeds')]:
        (prior/name).write_text(content)
        (live/name).write_text(content)
    (snapshot/'physics.py').write_text('physics = 1')
    (live/'physics.py').write_text('physics = 1')
    (snapshot/'technician_prefix_capacity_v3_experiment.py').write_text('DEV runner')
    (live/'technician_prefix_capacity_v3_experiment.py').write_text('confirmation guard')
    monkeypatch.setattr(x,'PLAN',live/'protocol.md')
    monkeypatch.setattr(x,'REGISTRY',live/'seed_registry.json')
    monkeypatch.setattr(x,'__file__',str(live/'technician_prefix_capacity_v3_experiment.py'))
    x.verify_development_source(prior)
    (live/'physics.py').write_text('physics = 2')
    with pytest.raises(RuntimeError,match='frozen learning source changed'):
        x.verify_development_source(prior)
    (live/'physics.py').write_text('physics = 1')
    (live/'seed_registry.json').write_text('changed seeds')
    with pytest.raises(RuntimeError,match='protocol or seed registry changed'):
        x.verify_development_source(prior)


def test_independent_dev_gate_requires_bound_pass(tmp_path):
    import hashlib
    import json
    prior=tmp_path/'dev';prior.mkdir()
    (prior/'manifest.json').write_text('manifest')
    audit=tmp_path/'auditor';audit.mkdir()
    verifier=audit/'audit_prefix_capacity_development.py';verifier.write_text('independent auditor')
    report=dict(status='PASS',verdict='DEV_GATE_PASS',test_opened=False,
        source_artifact=str(prior),source_manifest_sha256=hashlib.sha256(b'manifest').hexdigest(),
        verifier_sha256=hashlib.sha256(verifier.read_bytes()).hexdigest())
    (audit/'audit.json').write_text(json.dumps(report))
    assert x.verify_independent_development_audit(prior,audit)['verifier_sha256']==report['verifier_sha256']
    with pytest.raises(ValueError,match='independent development audit'):
        x.verify_independent_development_audit(prior,None)
    report['verdict']='DEV_GATE_FAIL';(audit/'audit.json').write_text(json.dumps(report))
    with pytest.raises(RuntimeError,match='independent development audit failed'):
        x.verify_independent_development_audit(prior,audit)
