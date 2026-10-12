"""Fair action constraints, scalar forecast oracle and prospective artifact gates."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from ht_pdm_fjsp import maintenance_joint_frontier_mac as s
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import transition, WaitingEnv


def idle(ages=(1,3,3,2)):
    return DispatchState(ages,(False,)*4,(0,)*4,(0,0),(-1,-1))


def test_shared_admissibility_includes_wait_partials_and_strong_baselines():
    cfg=s.settings('smoke');c=s.make_config(cfg);state=idle()
    space=s.old.screen.action_space(c,state)[2]
    for method in s.CHEAP:
        a=s.cheap(c,state,0,method)
        assert a in space and all(state.failed[m] or state.ages[m]>=3 for m,_ in a)
        for subset in (False,True):
            pool=s.roots(c,state,0,method,subset)
            assert pool[0]==a and () in pool and set(pool)<=set(space)
    assert s.cheap(c,idle((0,1,2,1)),0,'isolated_endpoint')==()


@pytest.mark.parametrize('base',s.CHEAP)
@pytest.mark.parametrize('probability',(0.,.45,1.))
def test_endpoint_forecast_matches_independent_scalar_continuation(base,probability):
    cfg=s.settings('smoke');c=replace(s.make_config(cfg),failure_probability=probability)
    state=idle();actions=((),((1,0),(2,1)));tick=42;scenarios=4;seed=726
    b=s.forecast(c,state,actions,tick,scenarios,seed,base)
    shocks=np.random.default_rng(seed).random((60-tick,scenarios,4))<probability
    expected=np.zeros((len(actions),scenarios))
    for i,action in enumerate(actions):
        for j in range(scenarios):
            x=state
            for offset in range(60-tick):
                a=action if offset==0 else s.cheap(c,x,tick+offset,base if tick+offset<48 else 'deadline_matching')
                x,cost,_=transition(c,x,a,shocks[offset,j]);expected[i,j]+=cost
    np.testing.assert_allclose(b['samples'],expected,atol=1e-10,rtol=0)
    assert b['transitions']==len(actions)*scenarios*(60-tick)


def test_subset_ablation_holds_each_subset_min_duration_with_baseline_exception():
    cfg=s.settings('smoke');c=s.make_config(cfg);x=idle((3,3,3,3))
    full=s.roots(c,x,0,'isolated_endpoint');restricted=s.roots(c,x,0,'isolated_endpoint',True)
    assert restricted[0]==full[0] and set(restricted)<=set(full)
    for a in restricted[1:]:
        group=[b for b in full if tuple(m for m,_ in b)==tuple(m for m,_ in a)]
        assert sum(c.service_time[m][j] for m,j in a)==min(sum(c.service_time[m][j] for m,j in b) for b in group)


def test_development_selection_cannot_read_confirmation():
    cfg=s.settings('smoke');rows=[dict(method=m,seed=seed,extended_cost=10 if m=='isolated_h6' else 20) for seed in cfg['development_seeds'] for m in s.CHEAP]
    result=s.choose_baseline(cfg,rows)
    assert result['base']=='isolated_h6' and not result['confirmation_outcomes_opened']
    with pytest.raises(RuntimeError,match='grid'):s.choose_baseline(cfg,rows[:-1])
    with pytest.raises(RuntimeError,match='grid'):s.choose_baseline(cfg,rows+[rows[0]])


def arithmetic(cfg,joint=90):
    rows=[dict(method=m,seed=seed,phase='confirmation',physical_seed=s.physical_seed(seed),extended_cost=joint if m=='joint_rollout' else 100,
        availability=.7,terminal_pending=0,policy_cost=joint if m=='joint_rollout' else 100) for seed in cfg['confirmation_seeds'] for m in s.METHODS]
    local=[dict(seed=seed,tick=t,**{f'{m}_{k}':0 for m in s.PLANNERS for k in ('gain','changed')}) for seed in cfg['confirmation_seeds'] for t in cfg['landmarks']]
    return rows,local


def test_registered_eight_interval_family_joint_primary_health_and_no_fallback():
    cfg=s.settings('full');rows,local=arithmetic(cfg)
    result=s.summarize(cfg,'isolated_endpoint',rows,local)
    assert result['scientific_status']=='JOINT_SETTING_CONFIRMED' and len(result['comparisons'])==8
    assert all(x['confidence']==1-.05/8 for x in result['comparisons'].values())
    rows,local=arithmetic(cfg,99.5)
    assert s.summarize(cfg,'isolated_endpoint',rows,local)['scientific_status']=='NOT_CONFIRMED'
    rows,local=arithmetic(cfg)
    for r in rows:
        if r['method']=='joint_rollout': r['availability']=.2
    assert s.summarize(cfg,'isolated_endpoint',rows,local)['scientific_status']=='NOT_CONFIRMED'
    cfg=s.settings('smoke');rows,local=arithmetic(cfg)
    assert s.summarize(cfg,'deadline_matching',rows,local)['scientific_status']=='ENGINEERING_ONLY'


def test_seeds_fresh_roles_split_and_locked():
    cfg=s.settings('smoke');a=s.seed_audit(cfg)
    assert a['status']=='PASS' and not set(cfg['development_seeds'])&set(cfg['confirmation_seeds'])
    assert s.bank_seed(cfg,cfg['confirmation_seeds'][0],0)!=s.bank_seed(cfg,cfg['confirmation_seeds'][0],0,'holdout')
    with pytest.raises(ValueError,match='locked'):s.seed_audit({**cfg,'scenarios':16})


def test_smoke_complete_replay_source_binding_overwrite_and_metric_tamper(tmp_path):
    out=tmp_path/'run';m=s.run('smoke',out)
    assert m['status']=='COMPLETED' and m['audit']=='PASS' and not m['confirmation_outcomes_opened']
    assert m['actual_counts']==dict(development=6,confirmation=10,physical_transitions=960,landmarks=14)
    assert m['scientific_status']=='ENGINEERING_ONLY'
    for name,b in m['files'].items():
        assert s.old.screen.sha(out/name)==b['sha256'] and (out/name).stat().st_size==b['bytes']
    with pytest.raises(FileExistsError):s.run('smoke',out)
    cfg=s.settings('smoke');base=json.loads((out/'selection.json').read_text())['base']
    # Serialized row tampering must be detected independently from decision log.
    dev=[];rows=[];local=[]
    def typed(path,string_keys):
        result=[]
        for r in s.read_csv(path):
            result.append({k:v if k in string_keys else float(v) if '.' in v or 'e' in v.lower() else int(v) for k,v in r.items()})
        return result
    dev=typed(out/'development_episodes.csv',('method','phase'))
    rows=typed(out/'episodes.csv',('method','phase'));local=typed(out/'landmarks.csv',())
    rows[0]['availability']+=.1
    with pytest.raises(RuntimeError,match='replay|scalar'):s.replay(out/'decisions.csv',cfg,base,dev+rows,local)


def test_timeout_records_failure_and_full_requires_commit(tmp_path,monkeypatch):
    original=s.settings
    monkeypatch.setattr(s,'seed_audit',lambda cfg:{'status':'PASS'})
    monkeypatch.setattr(s,'settings',lambda profile:{**original(profile),'max_seconds':-1})
    with pytest.raises(TimeoutError):s.run('smoke',tmp_path/'timed')
    assert json.loads((tmp_path/'timed'/'manifest.json').read_text())['status']=='FAILED'
    monkeypatch.setattr(s.old.screen,'git_metadata',lambda:dict(commit='fake',dirty=True))
    with pytest.raises(RuntimeError,match='clean committed'):s.run('full',tmp_path/'full')


def test_development_replay_reference_stays_deadline_after_dp_wins(tmp_path):
    cfg=s.settings('smoke');rows=[];path=tmp_path/'decisions.csv';journal=s.Journal(path)
    try:
        for seed in cfg['development_seeds']:
            for method in s.CHEAP:
                r,_=s.episode(cfg,method,seed,'deadline_matching','development',journal);rows.append(r)
    finally: journal.close()
    audit=s.replay(path,cfg,'isolated_endpoint',rows,[])
    assert audit['status']=='PASS' and audit['episodes']==6


def test_joint_must_beat_every_cheap_comparator_not_only_deadline():
    cfg=s.settings('full');rows,local=arithmetic(cfg,90)
    for r in rows:
        if r['method']=='isolated_h6':r['extended_cost']=85
    result=s.summarize(cfg,'isolated_h6',rows,local)
    assert result['gates']['cost_vs_deadline_matching']
    assert not result['gates']['cost_vs_isolated_h6']
    assert result['scientific_status']=='NOT_CONFIRMED'


def test_logged_seed_and_root_flags_are_not_unchecked_provenance(tmp_path):
    cfg=s.settings('smoke');journal=s.Journal(tmp_path/'d.csv')
    try: row,states=s.episode(cfg,'deadline_matching',cfg['confirmation_seeds'][0],'deadline_matching','confirmation',journal)
    finally:journal.close()
    records=s.read_csv(tmp_path/'d.csv');records[0]['physical_seed']=str(int(records[0]['physical_seed'])+1)
    s.write_csv(tmp_path/'tampered.csv',records)
    with pytest.raises(RuntimeError,match='seed provenance'):s.replay(tmp_path/'tampered.csv',cfg,'deadline_matching',[row],[])
    seed=cfg['confirmation_seeds'][0];tick=0;c=s.make_config(cfg);root_rows=[]
    class Sink:
        def add(self,r):root_rows.append(r)
    local=s.landmark(cfg,states[tick],seed,tick,'deadline_matching',Sink())
    s.validate_root_rows(c,cfg,'deadline_matching',{(seed,tick):states[tick]},[local],root_rows)
    root_rows[0]['selected_joint_rollout']=1-int(root_rows[0]['selected_joint_rollout'])
    with pytest.raises(RuntimeError,match='root flag'):s.validate_root_rows(c,cfg,'deadline_matching',{(seed,tick):states[tick]},[local],root_rows)
