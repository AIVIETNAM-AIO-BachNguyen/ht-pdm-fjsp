"""Pilot budgets, checkpoint selection and local PPO/replay artifact contract."""
import json
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import maintenance_joint_learning_mac as s


def test_prospective_panels_budgets_and_checkpoint_selection():
    cfg=s.settings('full')
    assert len(cfg['train_seeds'])==3 and cfg['steps']==65536 and cfg['steps']%(cfg['envs']*cfg['n_steps'])==0
    assert s.raw_audit(cfg)['status']=='PASS'
    records=[dict(steps=16384,development_cost=500),dict(steps=32768,development_cost=480),dict(steps=49152,development_cost=480)]
    assert s.checkpoint_choice(records)['steps']==32768
    with pytest.raises(ValueError):s.checkpoint_choice([dict(steps=0,development_cost=1)])
    with pytest.raises(ValueError,match='locked'):s.raw_audit({**cfg,'steps':65537})


def test_crossed_bootstrap_keeps_initial_and_fixed_baseline_pairing():
    cfg=s.settings('full')
    rows=[]
    for seed in cfg['test_seeds']:
        for name,train in [('ppo',t) for t in cfg['train_seeds']]+[('initial',t) for t in cfg['train_seeds']]+[(b,-1) for b in s.BASELINES]:
            cost=100 if name=='ppo' else 200 if name=='initial' else 110
            rows.append(dict(controller=name,train_seed=train,seed=seed,extended_cost=cost,policy_cost=cost,tail_cost=0,availability=.7,terminal_pending=0,tail_pending=0,failures=0,policy_failures=0,waiting_cost=0,policy_overdue_ticks=0,forecast_transitions=0,planning_seconds=0))
    result=s.summarize(cfg,rows)
    assert result['training_replicas']==3 and result['status']=='EXPLORATORY_LEARNING_PROBE'
    assert result['contrasts']['gain_vs_initial']['estimate']==pytest.approx(.5)
    assert result['contrasts']['gain_vs_initial']['ci95']==pytest.approx([.5,.5])
    assert result['contrasts']['gain_vs_deadline']['estimate']==pytest.approx(1-100/110)
    with pytest.raises(RuntimeError,match='grid'):s.summarize(cfg,rows[:-1])


def test_real_smoke_train_save_reload_select_test_replay_and_no_overwrite(tmp_path):
    out=tmp_path/'smoke';m=s.run('smoke',out)
    assert m['status']=='COMPLETED' and m['audit']=='PASS' and m['scientific_status']=='ENGINEERING_ONLY'
    assert m['training_steps']==512 and m['test_episodes']==12 and not m['test_outcomes_opened']
    audit=json.loads((out/'audit.json').read_text())
    assert audit['physical_transitions']==720 and audit['training_completed_episodes']==10 and audit['training_unique_physical_seeds']==12
    assert audit['checkpoints']==5 and all(r['learned_ticks']==16 for r in audit['training_partial_trajectories'])
    choices=json.loads((out/'selections.json').read_text());assert set(choices)=={'41400000'} and choices['41400000']['steps']>0
    assert (out/'episodes.csv').read_bytes()==(out/'episodes.partial.csv').read_bytes()
    for name,binding in m['files'].items():assert s.frontier.old.screen.sha(out/name)==binding['sha256']
    with pytest.raises(FileExistsError):s.run('smoke',out)
    # The replay must bind reported tail economics and checkpoint identity.
    rows=[]
    for record in s.frontier.read_csv(out/'episodes.csv'):
        rows.append({k:v if k=='controller' else None if v=='' else float(v) if '.' in v or 'e' in v.lower() else int(v) for k,v in record.items()})
    seed=41400000
    models={('ppo',seed):s.MaskablePPO.load(choices[str(seed)]['path'],device='cpu'),
            ('initial',seed):s.MaskablePPO.load(out/f'seed_{seed}'/'initial.zip',device='cpu')}
    rows[0]['tail_cost']+=1
    with pytest.raises(RuntimeError):s.replay(out,s.settings('smoke'),rows,models,lambda:None)
    rows[0]['tail_cost']-=1;rows[0]['checkpoint_steps']=0
    with pytest.raises(RuntimeError,match='checkpoint selection identity'):s.replay(out,s.settings('smoke'),rows,models,lambda:None)


def test_full_dirty_refusal_and_hard_timeout_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(s.frontier.old.screen,'git_metadata',lambda:dict(commit='x',dirty=True))
    with pytest.raises(RuntimeError,match='clean committed'):s.run('full',tmp_path/'full')
    original=s.settings
    monkeypatch.setattr(s,'settings',lambda profile:{**original(profile),'max_seconds':-1})
    monkeypatch.setattr(s,'raw_audit',lambda cfg:dict(status='PASS'))
    with pytest.raises(TimeoutError):s.run('smoke',tmp_path/'timeout')
    assert json.loads((tmp_path/'timeout'/'manifest.json').read_text())['status']=='FAILED'


def test_checkpoint_reload_does_not_restart_training_random_streams(tmp_path):
    import random
    from stable_baselines3.common.vec_env import DummyVecEnv
    cfg=s.settings('smoke');vec=DummyVecEnv([lambda:s.JointLearningEnv(s.c_for(cfg),s.VERSION,0,True)])
    model=s.MaskablePPO('MlpPolicy',vec,seed=29,device='cpu',n_steps=32,batch_size=32)
    random.seed(700);np.random.seed(800);torch.manual_seed(900)
    py,npstate,ts=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    expected=(random.random(),np.random.random(),torch.rand(4))
    random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(ts)
    s.save_model(tmp_path/'test.zip',model,cfg,41400000,0)
    actual=(random.random(),np.random.random(),torch.rand(4))
    assert actual[0]==expected[0] and actual[1]==expected[1] and torch.equal(actual[2],expected[2])
    vec.close()


def test_training_audit_rejects_missing_episode_and_wrong_stream():
    cfg=s.settings('smoke');seed=cfg['train_seeds'][0];done=[];partial=[]
    for worker in range(2):
        for index in range(5):
            done.append(dict(train_seed=seed,worker=worker,episode_index=index,stream_seed=seed+worker,physical_ticks=60,
                physical_seed=s.frontier.role_seed(seed+worker,f'{s.VERSION}:training_physical:worker:{worker}:episode:{index}')))
        partial.append(dict(train_seed=seed,worker=worker,episode_index=5,stream_seed=seed+worker,physical_ticks=16,learned_ticks=16,completed=False,
            physical_seed=s.frontier.role_seed(seed+worker,f'{s.VERSION}:training_physical:worker:{worker}:episode:5')))
    assert s.training_audit(cfg,done,partial)[str(seed)]['learned_steps']==512
    with pytest.raises(RuntimeError,match='step accounting'):s.training_audit(cfg,done[:-1],partial)
    partial[0]['stream_seed']+=1
    with pytest.raises(RuntimeError,match='stream identity'):s.training_audit(cfg,done,partial)
