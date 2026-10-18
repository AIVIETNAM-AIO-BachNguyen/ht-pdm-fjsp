import numpy as np
import pytest
import torch

from ht_pdm_fjsp import technician_prefix_experiment as x
from ht_pdm_fjsp import technician_prefix_env as e
from ht_pdm_fjsp.maintenance_balanced_algorithms import parameter_digest,tensors
from ht_pdm_fjsp.maintenance_dispatch import DispatchState


def test_full_and_mask_only_share_every_feasible_stage():
    torch.set_num_threads(1)
    cfg=x.settings('smoke')
    models={arm:x.make_model(arm,4,cfg,60800000) for arm in x.GRAPH_ARMS}
    assert len({parameter_digest(model) for model in models.values()})==1
    env=e.ResourceEnv(60900000,4,'reservation_ppo');env.reset()
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
    assert torch.count_nonzero(traces['reservation_ppo'][1]['prefix'])
    assert not traces['reservation_ppo'][1]['legal'][0,1]
    assert env.resolve(action[0].numpy())[0]==((0,0),(1,1))


def test_mask_only_forbids_duplicate_while_independent_can_propose_it():
    cfg=x.settings('smoke');env=e.ResourceEnv(60900000,4,'reservation_ppo');env.reset()
    env.env.state=DispatchState((3,)*4,(False,)*4,(0,)*4,(0,)*2,(-1,)*2)
    batch=tensors([env.observe()]);duplicate=torch.tensor([[1,1]])
    for arm in ('reservation_ppo','mask_only_ppo'):
        with pytest.raises(ValueError,match='illegal recorded'):
            x.make_model(arm,4,cfg,60800000).evaluate(batch,duplicate)
    x.make_model('independent_ppo',4,cfg,60800000).evaluate(batch,duplicate)


def test_fresh_panels_and_sealed_gate(tmp_path):
    assert x.seed_audit()['status']=='PASS'
    assert x.settings('development')['scales']==[4,8]
    assert not x.settings('development')['test_seeds']
    assert len(x.settings('confirmation')['train_seeds'])==5
    with pytest.raises(ValueError,match='development artifact'):
        x.run('confirmation',tmp_path/'unopened')


def test_primary_summary_requires_matched_prefix_gain():
    cfg=dict(x.settings('smoke'),scales=[4,8],train_seeds=[60800000],test_seeds=[61000000,61000001])
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


def test_independent_verifier_bootstrap_detects_prefix_gain():
    from scripts.verify_technician_prefix_mechanism import independently_resample
    config=dict(profile='smoke',scales=[4,8],train_seeds=[60800000],
        test_seeds=[61000000,61000001],analysis_seed=61100000)
    rows={}
    coordination=[]
    for n in config['scales']:
        for seed in config['test_seeds']:
            for arm,cost in (('reservation_ppo',98.),('mask_only_ppo',100.)):
                rows[n,arm,60800000,seed]=dict(normalized_cost=cost,availability=.67,
                    normalized_pending=0.,normalized_failures=.1)
            coordination.append(dict(n=n,method='reservation_ppo',decision_ms=1.))
    assert independently_resample(rows,config,coordination)['goal_pass']
    for key,row in rows.items():
        if key[1]=='reservation_ppo':row['normalized_cost']=100.
    assert not independently_resample(rows,config,coordination)['goal_pass']
