"""Engineering contracts for same-state forks; synthetic checkpoints are not evidence."""
import csv
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ht_pdm_fjsp import maintenance_timing_diagnostic as e
from ht_pdm_fjsp import maintenance_timing_forks as forks
from ht_pdm_fjsp import maintenance_timing_statistics as stats
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition
from ht_pdm_fjsp.maintenance_joint_policy import JOINT, CATALOGUE


@pytest.fixture
def source_fixture(tmp_path):
    """A hash-pinned synthetic source tests IO/physics; never a trained-result claim."""
    source = tmp_path/'source';source.mkdir()
    cfg = e.joint.settings('full');models = {};selections = {}
    for seed in cfg['train_seeds']:
        model = e.joint.model_for(cfg,seed,JOINT)
        name = f'{JOINT}/seed_{seed}/model.pt';path = source/name;path.parent.mkdir(parents=True)
        torch.save(dict(protocol=e.joint.VERSION,feature_contract=e.joint.FEATURE_CONTRACT,
            environment_version=e.joint.ENV_VERSION,algorithm=JOINT,train_seed=seed,settings=cfg,
            update=20,learned_steps=61440,state_dict=model.state_dict()),path)
        models[str(seed)] = dict(selected_checkpoint=name,update=20,learned_steps=61440,sha256=e.sha(path))
        selections[f'{JOINT}:{seed}'] = dict(selected_checkpoint=name,update=20,learned_steps=61440)
    rules = dict(dense='adaptive_isolated_matching',skill_mask='deadline_matching')
    e.prior.write_json(source/'checkpoint_selection.json',dict(locked_before_test=True,selections=selections,selected_rules=rules))
    _,profiles = e.joint.build_panels(cfg)
    e.prior.write_json(source/'resolved_config.json',dict(physical_profiles=profiles))
    snapshot = source/'source_snapshot';snapshot.mkdir()
    path = e.ROOT/'src/ht_pdm_fjsp/maintenance_waiting.py'
    (snapshot/path.name).write_bytes(path.read_bytes())
    manifest = dict(protocol=e.joint.VERSION,status='COMPLETED',audit='PASS',dirty=False,commit='fixture',
        settings=cfg,actual_counts=e.joint.counts(cfg),artifacts={str(p.relative_to(source)):
            dict(sha256=e.sha(p),bytes=p.stat().st_size) for p in source.rglob('*') if p.is_file()})
    e.prior.write_json(source/'manifest.json',manifest)
    locked = dict(protocol=e.VERSION,source_run='synthetic_engineering',commit='fixture',
        manifest_sha256=e.sha(source/'manifest.json'),models=models,selected_rules=rules)
    return source,locked


def test_exact_locked_counts_and_raw_roles():
    assert e.counts(e.settings('full')) == dict(models=10,trajectories=640,trajectory_ticks=30720,
        states=2560,max_forecast_transitions=41287680)
    assert e.counts(e.settings('smoke')) == dict(models=1,trajectories=4,trajectory_ticks=16,
        states=8,max_forecast_transitions=4032)
    assert e.settings('smoke')['horizon']==48
    assert e.seed_audit(e.settings('full'))['unique_derived_roles']==1303
    with pytest.raises(ValueError,match='settings changed'):
        e.seed_audit({**e.settings('full'),'forecast_ticks':25})
    assert e.physical_seed(13100000,13200000)!=e.forecast_seed(e.settings('full'),13100000,13200000,12100000,JOINT,6)


def test_physical_source_training_and_smoke_full_separation():
    full,_ = e.build_panels(e.settings('full'));smoke,_ = e.build_panels(e.settings('smoke'))
    fs = {e.boundary.signature(c) for c in full.values()};ss = {e.boundary.signature(c) for c in smoke.values()}
    assert len(fs)==16 and len(ss)==2
    assert not fs&ss and not (fs|ss)&e.exclusion_pool()
    assert len(e.exclusion_pool())>50000
    cfg = e.joint.settings('full');old,_ = e.joint.build_panels(cfg)
    _,forbidden = e.joint.seed_audit(cfg);sampler = e.joint.training_sampler(cfg,old,forbidden)
    for seed,index in [(12100000,0),(12100004,198),(12100009,5119)]:
        assert e.boundary.signature(sampler(seed,index)[0]) in e.exclusion_pool()
    for cohort in e.settings('full')['cohorts']:
        dense,masked = full[cohort,'dense'],full[cohort,'skill_mask']
        assert dense.restoration==masked.restoration
        assert (dense.failure_age,dense.failure_probability)==(masked.failure_age,masked.failure_probability)
        assert sum(d==0 for row in masked.service_time for d in row)==2


@pytest.mark.parametrize('condition',['dense','skill_mask'])
def test_same_state_all_roots_common_shocks_scalar_replay_and_future_privacy(condition):
    c = e.joint.make_config(13500000,condition,48)
    env = WaitingEnv(c);env.reset(47)
    for tick in range(8):
        state = env.state
        v = forks.fork_values(c,state,31+tick,8,9)
        assert forks.scalar_replay(c,state,v)
        assert v['candidates']==e.joint.feasible(c,state)
        assert np.allclose(sum(v['component_means'].values()),v['means'])
        private = env.events;env.events=tuple((True,)*4 for _ in private)
        again = forks.fork_values(c,state,31+tick,8,9)
        np.testing.assert_array_equal(v['costs'],again['costs'])
        env.events=private
        a,b = v['candidates'][0],v['candidates'][-1]
        d,se = forks.paired_delta(v,a,b);rd,rse = forks.paired_delta(v,b,a)
        assert d==-rd and se==rse
        assert forks.paired_delta(v,a,a)==(0.,0.)
        assert state==env.state
        env.step(e.prior.rule_action(c,state,48-tick,'deadline_matching'))


def test_pm_interventions_keep_corrective_pair_and_worker_and_busy_repairs():
    c = e.joint.make_config(13500000,'dense',48)
    state = DispatchState((4,3,1,0),(True,False,False,False),(2,0,0,0),(0,0),(-1,-1))
    root = ((0,1),(1,0))
    assert forks.without_preventive(state,root)==((0,1),)
    assert forks.kind(state,root)=='MIXED'
    v = forks.fork_values(c,state,12,7,8)
    assert forks.scalar_replay(c,state,v)
    corrective_only = forks.without_preventive(state,root)
    assert forks.paired_delta(v,root,corrective_only)==forks.paired_delta(v,root,tuple(p for p in root if p!=(1,0)))
    busy = DispatchState((4,3,1,0),(True,False,False,False),(2,0,0,0),(2,3),(0,1))
    forced = forks.fork_values(c,busy,12,7,8)
    assert forced['candidates']==((),)
    assert forks.scalar_replay(c,busy,forced)
    assert forced['first_step_mean'][0]>=12
    with pytest.raises(ValueError):forks.fork_values(c,state,12,1,8)
    with pytest.raises(ValueError):forks.fork_values(c,state,12,7,0)


def statistic_fixture():
    cfg = dict(profile='full',conditions=['dense','skill_mask'],cohorts=[1,2],physical_shocks=[3],
        model_seeds=[4,5],origins=[JOINT,'best_rule'],landmarks=[6],analysis_seed=13400000,bootstrap_samples=400)
    rows = [dict(condition=cond,cohort=cohort,shock=shock,model_seed=seed,origin=origin,tick=tick,
        pm_delta=0.,pm_exposure=0,rule_delta=0.,choice_exposure=1,actions_agree=True,rl_kind='WAIT',rule_kind='WAIT')
        for cond,cohort,shock,seed,origin,tick in itertools.product(cfg['conditions'],cfg['cohorts'],cfg['physical_shocks'],cfg['model_seeds'],cfg['origins'],cfg['landmarks'])]
    return cfg,rows


def test_bootstrap_conditional_ratio_not_mean_cell_ratios():
    nums = np.array([[100.,0.],[20.,30.]])
    dens = np.array([[10.,0.],[1.,3.]])
    rd = np.array([[0,1],[0,0],[1,1]]);pd = np.array([[0,1],[1,1],[0,0]])
    actual = stats.crossed_ratio(nums,dens,rd,pd)
    assert actual[0]==150/14 and np.isnan(actual[1]) and actual[2]==20
    cfg,rows = statistic_fixture()
    for r in rows:
        if r['cohort']==1:r.update(pm_delta=3.,pm_exposure=1)
    result,cells = stats.summarize(rows,cfg)
    assert result['primary_contrasts']['dense:pm_deferral']['estimate']==3
    assert result['primary_contrasts']['dense:pm_deferral']['confidence_interval'] is None
    assert result['primary_contrasts']['dense:pm_deferral']['empty_bootstrap_fraction']>.05
    assert len(cells)==16
    with pytest.raises(RuntimeError):stats.summarize(rows[:-1],cfg)
    with pytest.raises(RuntimeError):stats.summarize(rows+[rows[0]],cfg)
    rows[0]['pm_exposure']=0
    with pytest.raises(ValueError):stats.summarize(rows,cfg)


def test_frozen_pinned_input_integrity(source_fixture):
    source,locked = source_fixture
    models,audit = e.verify_source(source,locked)
    assert len(models)==10 and audit['status']=='PASS'
    with pytest.raises(ValueError,match='manifest hash'):
        e.verify_source(source,{**locked,'manifest_sha256':'bad'})
    wrong = json.loads(json.dumps(locked));wrong['selected_rules']['dense']='deadline_matching'
    with pytest.raises(ValueError,match='rule changed'):e.verify_source(source,wrong)
    wrong = json.loads(json.dumps(locked));wrong['models']['12100000']['update']=40
    with pytest.raises(ValueError,match='selection'):e.verify_source(source,wrong)
    path = source/locked['models']['12100000']['selected_checkpoint'];path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='artifact mismatch'):e.verify_source(source,locked)


def test_preflight_full_dirty_missing_source_and_nonempty_never_open_outcomes(tmp_path,monkeypatch):
    monkeypatch.setattr(e.platform,'platform',lambda:'engineering fixture')
    monkeypatch.setattr(e.subprocess,'check_output',lambda args,**kw: 'dirty\n' if 'status' in args else 'commit\n')
    out = tmp_path/'full'
    with pytest.raises(ValueError,match='clean'):e.run(out,tmp_path/'absent','full')
    m = json.loads((out/'manifest.json').read_text())
    assert m['status']=='FAILED' and not m['new_full_panel_outcomes_opened']
    with pytest.raises(ValueError,match='new/empty'):e.run(out,tmp_path/'absent')
    smoke = tmp_path/'missing'
    with pytest.raises(FileNotFoundError):e.run(smoke,tmp_path/'absent')
    assert not json.loads((smoke/'manifest.json').read_text())['new_full_panel_outcomes_opened']
    monkeypatch.setattr(e.subprocess,'check_output',lambda args,**kw:'' if 'status' in args else 'commit\n')
    monkeypatch.setenv('OMP_NUM_THREADS','2')
    with pytest.raises(ValueError,match='NUM_THREADS'):e.run(tmp_path/'threads',tmp_path/'absent','full')


def test_end_to_end_smoke_round_trip_auditable_costs_and_repeatability(tmp_path,monkeypatch,source_fixture):
    source,locked = source_fixture;monkeypatch.setattr(e,'contract',lambda:locked)
    first = tmp_path/'first';second = tmp_path/'second'
    m = e.run(first,source)
    assert m['status']=='COMPLETED' and m['audit']=='PASS'
    assert m['actual_counts']==e.counts(e.settings('smoke'))
    assert not m['new_full_panel_outcomes_opened'] and m['learning'].startswith('NOT_APPLICABLE')
    for name,record in m['artifacts'].items():
        assert e.sha(first/name)==record['sha256'] and (first/name).stat().st_size==record['bytes']
    e.run(second,source)
    for name in ['states.csv','root_values.csv','preventive_pairs.csv','trajectories.csv','summary.json']:
        assert (first/name).read_bytes()==(second/name).read_bytes()
    padded = np.load(first/'raw_forecast_costs.npz')['costs']
    assert padded.shape==(8,21,4)
    states = list(csv.DictReader((first/'states.csv').open()))
    roots = list(csv.DictReader((first/'root_values.csv').open()))
    for i,row in enumerate(states):
        selected = [r for r in roots if all(r[k]==row[k] for k in ('condition','cohort','shock','model_seed','origin','tick'))]
        assert len(selected)==int(row['candidates'])
        for r in selected:
            samples = json.loads(r['scenario_costs'])
            np.testing.assert_array_equal(padded[i,int(r['action_index'])],samples)
            assert np.mean(samples)==float(r['mean_cost'])
        assert np.isnan(padded[i]).all(1).sum()==21-len(selected)
        state = DispatchState(**{k:tuple(v) for k,v in json.loads(row['state']).items()})
        c = e.build_panels(e.settings('smoke'))[0][int(row['cohort']),row['condition']]
        v = forks.fork_values(c,state,int(row['forecast_seed']),4,6)
        assert forks.scalar_replay(c,state,v)
        assert tuple(tuple(p) for p in json.loads(row['no_pm_action']))==forks.without_preventive(state,tuple(tuple(p) for p in json.loads(row['rl_action'])))
    trajectory = list(csv.DictReader((first/'trajectories.csv').open()))
    assert len(trajectory)==16 and all(float(r['cost'])>=0 for r in trajectory)
    for r in trajectory:
        c = e.build_panels(e.settings('smoke'))[0][int(r['cohort']),r['condition']]
        env = WaitingEnv(c);env.reset(e.physical_seed(int(r['cohort']),int(r['shock'])))
        state = DispatchState(**{k:tuple(v) for k,v in json.loads(r['state']).items()})
        following,cost,_ = transition(c,state,tuple(tuple(p) for p in json.loads(r['action'])),env.events[int(r['tick'])])
        assert float(r['cost'])==cost and json.loads(r['following'])==json.loads(json.dumps(asdict(following)))
    for seed in e.settings('smoke')['model_seeds']:
        assert e.sha(first/'frozen_models'/f'seed_{seed}.pt')==locked['models'][str(seed)]['sha256']


def test_interruption_keeps_partial_states_failed_manifest(tmp_path,monkeypatch,source_fixture):
    source,locked = source_fixture;monkeypatch.setattr(e,'contract',lambda:locked)
    original = e.evaluate_state;calls = 0
    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls==2:raise KeyboardInterrupt('fixture interruption')
        return original(*args)
    monkeypatch.setattr(e,'evaluate_state',interrupted)
    out = tmp_path/'interrupted'
    with pytest.raises(KeyboardInterrupt):e.run(out,source)
    m = json.loads((out/'manifest.json').read_text())
    assert m['status']=='FAILED' and not m['new_full_panel_outcomes_opened']
    assert len(list(csv.DictReader((out/'states.partial.csv').open())))==1
    assert 'states.partial.csv' in m['artifacts'] and not (out/'states.csv').exists()
