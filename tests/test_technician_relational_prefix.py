import numpy as np
import pytest
import torch

from ht_pdm_fjsp import technician_relational_prefix_experiment as x
from ht_pdm_fjsp import technician_relational_prefix_env as e
from ht_pdm_fjsp.maintenance_balanced_algorithms import parameter_digest, tensors
from ht_pdm_fjsp.maintenance_dispatch import DispatchState


def test_relational_prefix_and_mask_share_feasible_stages():
    torch.set_num_threads(1)
    cfg=x.settings('smoke')
    models={arm:x.make_model(arm,4,cfg,63800000) for arm in x.GRAPH_ARMS}
    assert len({parameter_digest(model) for model in models.values()})==1
    env=e.ResourceEnv(63900000,4,'reservation_ppo');env.reset()
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
        assert torch.isfinite(full['prefix']).all()
    assert traces['reservation_ppo'][1]['prefix'].shape==(1,4,64)
    assert torch.count_nonzero(traces['reservation_ppo'][1]['prefix'])
    assert not traces['reservation_ppo'][1]['legal'][0,1]
    assert env.resolve(action[0].numpy())[0]==((0,0),(1,1))


def test_empty_prefix_wait_and_duplicate_contract():
    cfg=x.settings('smoke');env=e.ResourceEnv(63900000,4,'reservation_ppo');env.reset()
    env.env.state=DispatchState((3,)*4,(False,)*4,(0,)*4,(0,)*2,(-1,)*2)
    batch=tensors([env.observe()]);full=x.make_model('reservation_ppo',4,cfg,63800000)
    trace=[]
    full.trace(batch,recorded=torch.tensor([[0,1]]),audit=trace)
    assert not torch.count_nonzero(trace[1]['prefix'])
    assert torch.isfinite(full.evaluate(batch,torch.tensor([[0,1]]))[0]).all()
    for arm in ('reservation_ppo','mask_only_ppo'):
        with pytest.raises(ValueError,match='illegal recorded'):
            x.make_model(arm,4,cfg,63800000).evaluate(batch,torch.tensor([[1,1]]))
    x.make_model('independent_ppo',4,cfg,63800000).evaluate(batch,torch.tensor([[1,1]]))


def test_selected_edge_attention_depends_on_candidate_at_n8():
    cfg=x.settings('smoke');env=e.ResourceEnv(63900000,8,'reservation_ppo');env.reset()
    env.env.state=DispatchState((3,)*8,(False,)*8,(0,)*8,(0,)*4,(-1,)*4)
    batch=tensors([env.observe()]);action=torch.tensor([[1,2,3,4]])
    full=x.make_model('reservation_ppo',8,cfg,63800000)
    masked=x.make_model('mask_only_ppo',8,cfg,63800000)
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
    cfg=dict(x.settings('smoke'),scales=[4,8],train_seeds=[63800000],test_seeds=[64000000,64000001])
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
    from scripts.verify_technician_relational_prefix_v2 import independently_resample
    config=dict(profile='smoke',scales=[4,8],train_seeds=[63800000],
        test_seeds=[64000000,64000001],analysis_seed=64100000)
    rows={};coordination=[]
    for n in config['scales']:
        for seed in config['test_seeds']:
            for arm,cost in (('reservation_ppo',98.),('mask_only_ppo',100.)):
                rows[n,arm,63800000,seed]=dict(normalized_cost=cost,availability=.67,
                    normalized_pending=0.,normalized_failures=.1)
            coordination.append(dict(n=n,method='reservation_ppo',decision_ms=1.))
    assert independently_resample(rows,config,coordination)['goal_pass']
    for key,row in rows.items():
        if key[1]=='reservation_ppo':row['normalized_cost']=100.
    assert not independently_resample(rows,config,coordination)['goal_pass']
