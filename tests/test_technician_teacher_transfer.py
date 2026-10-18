import json
import numpy as np
import pytest
import torch

from ht_pdm_fjsp import technician_teacher_transfer_experiment as x
from ht_pdm_fjsp import technician_teacher_transfer_env as e
from ht_pdm_fjsp.maintenance_balanced_algorithms import parameter_digest, tensors
from ht_pdm_fjsp.maintenance_dispatch import DispatchState


def test_teacher_and_scratch_same_initial_reservation_architecture():
    torch.set_num_threads(1)
    cfg=x.settings('smoke')
    teacher=x.make_model('teacher_reservation_ppo',8,cfg,62800000)
    scratch=x.make_model('scratch_reservation_ppo',8,cfg,62800000)
    assert parameter_digest(teacher)==parameter_digest(scratch)
    assert sum(p.numel() for p in teacher.parameters())==sum(p.numel() for p in scratch.parameters())
    env=e.ResourceEnv(62900000,8,'teacher_reservation_ppo');env.reset()
    env.env.state=DispatchState((3,)*8,(False,)*8,(0,)*8,(0,)*4,(-1,)*4)
    obs=env.observe();batch=tensors([obs]);action=torch.tensor([[1,2,3,4]])
    traces=[]
    for model in (teacher,scratch):
        audit=[];model.trace(batch,recorded=action,audit=audit);traces.append(audit)
    for a,b in zip(*traces,strict=True):
        assert torch.equal(a['legal'],b['legal'])
    assert env.resolve(action[0].numpy())[0]==((0,0),(1,1),(2,2),(3,3))


def test_fresh_seed_panel_and_development_gate(tmp_path):
    assert x.seed_audit()['status']=='PASS'
    assert x.settings('development')['test_seeds']==[]
    assert x.settings('confirmation')['scales']==[8,16,32]
    with pytest.raises(ValueError,match='development artifact'):
        x.run('confirmation',tmp_path/'sealed')


def test_teacher_primary_gain_and_health_requirements():
    cfg=dict(x.settings('smoke'),scales=[8,16,32],train_seeds=[62800000],
        test_seeds=[63000000,63000001])
    rows=[];latencies={}
    for n in cfg['scales']:
        for arm in (*x.ARMS,'deadline'):
            latencies[f'{n}:{arm}']=[1.]*4
            for train in ([-1] if arm=='deadline' else cfg['train_seeds']):
                for seed in cfg['test_seeds']:
                    rows.append(dict(n=n,method=arm,train_seed=train,seed=seed,
                        normalized_cost=98. if arm=='teacher_reservation_ppo' else 100.,
                        availability=.67,normalized_pending=0.,normalized_failures=.1))
    positive=x.summarize(rows,cfg,latencies)
    assert positive['goal_pass']
    json.dumps(positive,allow_nan=False)
    for row in rows:
        if row['method']=='teacher_reservation_ppo':row['normalized_cost']=100.
    assert not x.summarize(rows,cfg,latencies)['goal_pass']


def test_development_reports_fixed_final_and_auc():
    cfg=dict(x.settings('smoke'),scales=[8,16,32],train_seeds=[62800000],
        development_seeds=[62900000,62900001])
    rows=[]
    for n in cfg['scales']:
        for arm in x.ARMS:
            for step in [0,*cfg['checkpoints']]:
                for seed in cfg['development_seeds']:
                    rows.append(dict(n=n,method=arm,train_seed=62800000,checkpoint_steps=step,
                        seed=seed,extended_cost=n*(98. if arm=='teacher_reservation_ppo' else 100.),
                        availability=.67,normalized_pending=0.,normalized_failures=.1,
                        invalid_assignments=0))
    summary=x.development_summary(rows,cfg)
    assert summary['development_screen_pass']
    assert summary['pooled_gain_teacher_vs_scratch']==pytest.approx(.02)
    assert summary['learning_auc'][8]['teacher_reservation_ppo']['mean']==pytest.approx(98.)
    assert summary['fixed_final_cost_per_machine'][8]['scratch_reservation_ppo']['mean']==pytest.approx(100.)


@pytest.mark.parametrize('n',(16,32))
def test_large_teacher_and_scratch_assignments_without_catalogue(monkeypatch,n):
    import ht_pdm_fjsp.maintenance_heterogeneous_model as old
    monkeypatch.setattr(old,'catalogue',lambda *args:(_ for _ in ()).throw(AssertionError('large catalogue')))
    cfg=x.settings('smoke')
    for arm in x.ARMS:
        env=e.ResourceEnv(62900000,n,arm);obs=env.reset()
        model=x.make_model(arm,n,cfg,62800000)
        for _ in range(48):
            obs,_,done,info=env.step(model.select(obs))
        assert done and info['episode_metrics']['invalid_assignments']==0


def test_independent_confirmation_resampling_requires_teacher_gain():
    from scripts.verify_teacher_transfer_confirmation import independently_resample
    config=dict(profile='smoke',scales=[8,16,32],train_seeds=[62800000],
        test_seeds=[63000000,63000001],analysis_seed=63100000)
    rows={};coordination=[]
    for n in config['scales']:
        for seed in config['test_seeds']:
            for arm,cost in (('teacher_reservation_ppo',98.),('scratch_reservation_ppo',100.)):
                rows[n,arm,62800000,seed]=dict(normalized_cost=cost,
                    availability=.67,normalized_pending=0.,normalized_failures=.1)
            coordination.append(dict(n=n,method='teacher_reservation_ppo',decision_ms=1.))
    assert independently_resample(rows,config,coordination)['goal_pass']
    for key,row in rows.items():
        if key[1]=='teacher_reservation_ppo':row['normalized_cost']=100.
    assert not independently_resample(rows,config,coordination)['goal_pass']


def test_confirmation_requires_exact_independent_dev_pass(tmp_path):
    import hashlib
    prior=tmp_path/'dev';prior.mkdir();(prior/'manifest.json').write_text('frozen A1 manifest')
    audit=tmp_path/'audit';audit.mkdir()
    verifier=audit/'audit_teacher_transfer_development.py';verifier.write_text('independent A1 auditor')
    report=dict(status='PASS',profile='development',test_opened=False,
        development_screen_pass=True,source_artifact=str(prior),
        source_manifest_sha256=hashlib.sha256(b'frozen A1 manifest').hexdigest(),
        verifier_sha256=hashlib.sha256(verifier.read_bytes()).hexdigest())
    (audit/'audit.json').write_text(json.dumps(report))
    assert x.verify_independent_development_audit(prior,audit)['verifier_sha256']==report['verifier_sha256']
    with pytest.raises(ValueError,match='independent DEV audit'):
        x.verify_independent_development_audit(prior,None)
    report['development_screen_pass']=False;(audit/'audit.json').write_text(json.dumps(report))
    with pytest.raises(RuntimeError,match='independent DEV audit failed'):
        x.verify_independent_development_audit(prior,audit)
