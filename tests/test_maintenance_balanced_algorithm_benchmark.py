import json
import numpy as np
import pytest
import torch

from ht_pdm_fjsp import maintenance_balanced_algorithm_benchmark as bench
from ht_pdm_fjsp.maintenance_balanced_benchmark_env import BenchmarkEnv


def test_locked_profiles_and_seed_disjointness():
    smoke=bench.settings('smoke'); full=bench.settings('full')
    assert len(full['train_seeds'])==5 and len(full['test_seeds'])==256
    assert full['env_steps']*4*5==9830400
    assert full['env_steps']//48*60*4*5==12288000
    assert len(full['development_seeds'])*5*4*5==6400
    assert (5*4+5)*len(full['test_seeds'])==6400
    assert all(s%full['rollout']==0 for s in full['checkpoints'])
    assert not set(smoke['test_seeds'])&set(full['test_seeds'])
    assert bench.seed_audit(smoke)['status']=='PASS'
    bad=dict(smoke,env_steps=smoke['env_steps']+48)
    with pytest.raises(ValueError): bench.seed_audit(bad)


@pytest.mark.parametrize('algorithm',bench.ALGORITHMS)
def test_checkpoint_roundtrip_and_corruption(tmp_path,algorithm):
    cfg=bench.settings('smoke'); seed=cfg['train_seeds'][0]
    model=bench.make_model(algorithm,cfg,seed)
    env=BenchmarkEnv(cfg['development_seeds'][0]); obs=env.reset()
    rng=torch.random.get_rng_state().clone()
    path=tmp_path/'model.pt'
    metadata=bench.checkpoint(model,path,cfg,seed,192,obs)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert metadata['sha256']==bench.sha(path)
    with pytest.raises(FileExistsError): bench.checkpoint(model,path,cfg,seed,192,obs)
    loaded=bench.load_model(path,cfg,algorithm,seed,192)
    assert np.array_equal(loaded.select(obs),model.select(obs))
    payload=torch.load(path,weights_only=True)
    payload['parameter_sha256']='wrong'; torch.save(payload,path)
    with pytest.raises(RuntimeError,match='parameter hash'): bench.load_model(path,cfg,algorithm,seed,192)


def test_refuse_overwrite_before_training(tmp_path):
    (tmp_path/'prior.json').write_text('{}')
    with pytest.raises(FileExistsError): bench.run('smoke',tmp_path)
    assert json.loads((tmp_path/'prior.json').read_text())=={}
    with pytest.raises(ValueError,match='CPU'): bench.run('smoke',tmp_path,'cuda')


def test_evaluation_population_identity_and_terminal_fields():
    cfg=bench.settings('smoke'); train_seed=cfg['train_seeds'][0]; seed=cfg['test_seeds'][0]
    model=bench.make_model('ppo',cfg,train_seed)
    row=bench.evaluate_episode('ppo',train_seed,seed,cfg,model)
    env=BenchmarkEnv(seed); env.reset()
    assert row['population_sha256']==env.initial_population
    assert row['physical_ticks']==60 and row['invalid_assignments']==0
    assert row['extended_cost']==row['policy_cost']+row['tail_cost']


def test_all_controller_scalar_replay_and_aggregate_corruption(tmp_path):
    cfg=bench.settings('smoke')
    cfg=dict(cfg,test_seeds=cfg['test_seeds'][:1],scenarios=2,audit_seeds=1)
    train_seed=cfg['train_seeds'][0]; rows=[]; choices={}
    decisions=bench.Journal(tmp_path/'testdecisions.csv'); coord=bench.Journal(tmp_path/'coordination.csv')
    env=BenchmarkEnv(cfg['development_seeds'][0]); obs=env.reset()
    try:
        for method in (*bench.ALGORITHMS,*bench.BASELINES):
            if method in bench.ALGORITHMS:
                model=bench.make_model(method,cfg,train_seed); path=tmp_path/f'{method}.pt'
                metadata=bench.checkpoint(model,path,cfg,train_seed,0,obs)
                choices[f'{method}:{train_seed}']=dict(metadata,path=path.name,steps=0)
                replica=train_seed
            else: model=None; replica=-1
            rows.append(bench.evaluate_episode(method,replica,cfg['test_seeds'][0],cfg,model,decisions,coord))
    finally: decisions.close(); coord.close()
    audit=bench.replay_test(tmp_path,cfg,rows,choices,lambda:None)
    assert audit['status']=='PASS' and audit['physical_ticks']==540 and audit['policy_records']==432
    rows[-1]=dict(rows[-1],waiting_cost=rows[-1]['waiting_cost']+1)
    with pytest.raises(RuntimeError,match='aggregate replay'):
        bench.replay_test(tmp_path,cfg,rows,choices,lambda:None)
