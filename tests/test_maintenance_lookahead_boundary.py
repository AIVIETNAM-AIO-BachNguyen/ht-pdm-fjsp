"""Boundary scoring, independent physical replay and prospective panel seals."""
import csv
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ht_pdm_fjsp import maintenance_lookahead_boundary as s
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition


def idle(ages=(3,3,3,3)):
    return DispatchState(ages,(False,)*4,(0,)*4,(0,0),(-1,-1))


def test_fixed_environment_horizons_budget_and_new_seed_panels():
    cfg=s.settings('smoke'); c=s.make_config(cfg)
    assert c.service_time==((1,0),(1,3),(1,2),(1,5)) and c.horizon==60
    assert len(s.roots(c,idle()))==17 and len(s.CONTROLLERS)==7
    assert cfg['train_seeds']==cfg['development_seeds']==[]
    assert s.settings('full')['scenarios']==128
    assert s.settings('full')['confidence']==1-.05/12
    assert s.forecast_cap(s.settings('full'))==2523365376
    assert s.score_ticks(cfg,'joint_h12',47)==12
    assert s.score_ticks(cfg,'joint_h24',47)==13
    assert s.score_ticks(cfg,'assignment_endpoint',0)==60
    assert s.score_ticks(cfg,'assignment_endpoint',47)==13
    audit=s.seed_audit(cfg)
    assert audit['raw_unique']==136 and audit['derived_unique']==7394
    assert audit['prior_derived_checked']==8000
    assert not set(cfg['evaluation_seeds']) & set(s.previous.settings('smoke')['evaluation_seeds'])
    with pytest.raises(ValueError,match='locked'): s.seed_audit({**cfg,'scenarios':64})
    with pytest.raises(ValueError): s.settings('other')
    with pytest.raises(ValueError): s.score_ticks(cfg,'joint_endpoint',48)


@pytest.mark.parametrize('probability',[0.,.45,1.])
def test_all_nested_values_equal_independent_scalar_rollouts(probability):
    cfg=s.settings('smoke'); c=replace(s.make_config(cfg),failure_probability=probability)
    state=DispatchState((3,5,3,2),(True,False,False,False),(6,0,0,0),(0,0),(-1,-1))
    actions=tuple(dict.fromkeys([s.myopic(c,state),(),((1,0),(2,1)),((1,1),(2,0))]))
    bank,scenarios=991,8
    result=s.checkpoint_values(c,state,actions,(12,24,60),scenarios,bank)
    shocks=np.random.default_rng(bank).random((60,scenarios,4))<probability
    expected={h:np.zeros((len(actions),scenarios)) for h in (12,24,60)}
    for i,action in enumerate(actions):
        for j in range(scenarios):
            following=state; total=0.
            for tick in range(60):
                chosen=action if tick==0 else s.myopic(c,following)
                following,cost,_=transition(c,following,chosen,shocks[tick,j]); total+=cost
                if tick+1 in expected: expected[tick+1][i,j]=total
    for h,value in result['values'].items():
        np.testing.assert_array_equal(value['samples'],expected[h])
        np.testing.assert_allclose(value['mc_se'],expected[h].std(1,ddof=1)/np.sqrt(scenarios))
    assert result['transitions']==len(actions)*scenarios*60


def test_shared_prefixes_match_old_algorithm_and_terminal_decomposition():
    cfg=s.settings('smoke'); c=s.make_config(cfg); state=idle((0,3,3,3))
    actions=s.roots(c,state)
    values=s.checkpoint_values(c,state,actions,(12,24,60),8,555)['values']
    np.testing.assert_array_equal(values[12]['samples'],s.previous.forecast_values(c,state,actions,12,8,555)['samples'])
    shorter=s.checkpoint_values(c,state,actions,(24,),8,555)['values'][24]
    np.testing.assert_array_equal(values[24]['samples'],shorter['samples'])
    terminal=values[60]['samples']-values[12]['samples']
    assert (terminal>=0).all()
    np.testing.assert_array_equal(values[12]['samples']+terminal,values[60]['samples'])


def test_prior_recorded_periodic_state_rejects_one_tick_saving_when_window_extends():
    # Prior panel observation is a regression fixture, not a new evaluation seed.
    cfg=s.settings('smoke'); c=s.make_config(cfg); state=idle((1,3,3,2))
    baseline=s.myopic(c,state); swapped=((1,1),(2,0))
    assert baseline==((1,0),(2,1))
    values=s.checkpoint_values(c,state,(baseline,swapped),(12,24,60),8,777)['values']
    np.testing.assert_array_equal(values[12]['means'],[102.,101.])
    assert s.minimum(values[12]['means'],(0,1))==1
    assert s.minimum(values[24]['means'],(0,1))==0
    assert s.minimum(values[60]['means'],(0,1))==0
    assert (values[12]['mc_se']==0).all()


@pytest.mark.parametrize('name',s.PLANNERS)
def test_controller_restrictions_no_real_future_and_scores_reconcile(name):
    cfg=s.settings('smoke'); c=s.make_config(cfg); seed=cfg['evaluation_seeds'][0]
    env=WaitingEnv(c); env.reset(s.physical_seed(seed),idle((0,3,3,3)))
    before=env.state; action,diag=s.select(c,before,name,cfg,seed,0)
    env.events=[(False,)*4]*60
    assert s.select(c,env.state,name,cfg,seed,0)==(action,diag)
    assert env.state==before and action in s.roots(c,before)
    if name.startswith('assignment_'):
        assert tuple(m for m,j in action)==tuple(m for m,j in s.myopic(c,before))
    assert diag['planned'] and diag['selection_seed']!=s.physical_seed(seed)
    assert diag['selected_mean']<=diag['myopic_mean']+1e-10
    for prefix in ('selected','myopic'):
        assert diag[f'{prefix}_mean']==pytest.approx(diag[f'{prefix}_prefix12']+diag[f'{prefix}_terminal'])
    if name.endswith('h12'): assert diag['selected_terminal']==diag['myopic_terminal']==0.


def test_forced_choices_common_tail_and_baseline_ties_skip_forecasts(monkeypatch):
    cfg=s.settings('smoke'); c=s.make_config(cfg); state=idle((3,3,0,0))
    def no_forecast(*a,**k): raise AssertionError('must not forecast')
    monkeypatch.setattr(s,'checkpoint_values',no_forecast)
    for name in s.CONTROLLERS:
        action,diag=s.select(c,state,name,cfg,39300000,48)
        assert action==s.myopic(c,state) and not diag['planned']
    for name in ('assignment_h12','assignment_h24','assignment_endpoint'):
        assert not s.select(c,state,name,cfg,39300000,0)[1]['planned']
    actions=s.roots(c,idle((0,3,3,0)))
    assert all(m in (1,2) for a in actions for m,j in a)
    assert s.minimum(np.zeros(len(actions)),s.eligible_indices(actions,True))==0
    with pytest.raises(ValueError): s.minimum(np.array([np.nan]),(0,))


def test_holdout_bank_cannot_select_and_endpoint_gain_includes_negative_values(monkeypatch):
    cfg=s.settings('smoke'); c=s.make_config(cfg); state=idle((0,3,3,0)); calls=[]
    chosen=s.eligible_indices(s.roots(c,state),True)[-1]
    def synthetic(c,state,roots,checkpoints,scenarios,seed,check_timeout=None):
        calls.append(seed); values={}
        for h in checkpoints:
            costs=np.full((len(roots),scenarios),100.*h)
            costs[chosen]+= -10. if seed==s.forecast_seed(cfg,39300000,0) else 10.
            values[h]=dict(samples=costs,means=costs.mean(1),mc_se=np.zeros(len(roots)))
        return dict(values=values,transitions=len(roots)*scenarios*max(checkpoints))
    monkeypatch.setattr(s,'checkpoint_values',synthetic)
    row=s.landmark(cfg,state,39300000,0)
    assert calls==[s.forecast_seed(cfg,39300000,0),s.forecast_seed(cfg,39300000,0,'holdout')]
    for name in s.PLANNERS:
        assert row[f'{name}_root']==chosen and row[f'{name}_endpoint_gain']==-10.


def synthetic(cfg):
    costs=(100.,99.,98.,95.,101.,100.,96.)
    rows=[dict(controller=p,seed=seed,physical_seed=s.physical_seed(seed),extended_cost=costs[i],
        availability=.7,terminal_pending=0.) for seed in cfg['evaluation_seeds'] for i,p in enumerate(s.CONTROLLERS)]
    local=[dict(seed=seed,tick=tick,**{f'{name}_endpoint_gain':2. for name in s.PLANNERS},
        **{f'{name}_disagreement_rate':1 for name in s.PLANNERS}) for seed in cfg['evaluation_seeds'] for tick in cfg['landmarks']]
    return rows,local


def test_paired_ratio_of_means_twelve_interval_family_and_primary_guards():
    cfg=s.settings('smoke'); rows,local=synthetic(cfg); result=s.summarize(cfg,rows,local)
    assert result['scientific_status']=='ENGINEERING_ONLY'
    assert result['primary']=='assignment_endpoint_vs_myopic' and len(result['comparisons'])==12
    assert result['comparisons']['assignment_endpoint_vs_myopic']['ci']==pytest.approx([.04,.04])
    assert result['secondary_assignment_boundary_repair']
    for row in rows:
        if row['seed']==cfg['evaluation_seeds'][0]:
            if row['controller']=='myopic_matching': row['extended_cost']=200.
            elif row['controller']=='assignment_endpoint': row['extended_cost']=180.
    result=s.summarize(cfg,rows,local)
    assert result['comparisons']['assignment_endpoint_vs_myopic']['estimate']==pytest.approx(1-117/125)
    rows,local=synthetic(cfg)
    for row in rows:
        if row['controller']=='assignment_endpoint': row.update(availability=.65,terminal_pending=1.)
    result=s.summarize(cfg,rows,local)
    assert result['gates']['assignment']==dict(practical_cost=True,availability=False,pending=False)
    assert all(result['gates']['joint'].values())
    with pytest.raises(RuntimeError,match='grid'): s.summarize(cfg,rows[:-1],local)
    with pytest.raises(RuntimeError,match='landmark'): s.summarize(cfg,rows,local[:-1])


def test_full_classification_uses_primary_scope_not_joint_or_repair():
    cfg=s.settings('full'); rows,local=synthetic(cfg)
    assert s.summarize(cfg,rows,local)['scientific_status']=='ASSIGNMENT_HEADROOM_CONFIRMED'
    for row in rows:
        if row['controller']=='assignment_endpoint': row['extended_cost']=100.
    result=s.summarize(cfg,rows,local)
    assert result['scientific_status']=='NOT_CONFIRMED' and result['secondary_joint_headroom']
    assert result['secondary_assignment_boundary_repair']


def test_timeout_and_dirty_full_do_not_open_evaluation_panel(tmp_path,monkeypatch):
    cfg=s.settings('smoke'); c=s.make_config(cfg)
    def expired(): raise TimeoutError('test deadline')
    with pytest.raises(TimeoutError): s.checkpoint_values(c,idle(),s.roots(c,idle()),(12,60),8,123,expired)
    monkeypatch.setattr(s.screen,'git_metadata',lambda:dict(commit='test',branch='test',dirty=True))
    with pytest.raises(RuntimeError,match='clean'): s.run('full',tmp_path/'full')
    assert not (tmp_path/'full').exists()
    counter=[-601.]
    def clock(): counter[0]+=601.; return counter[0]
    monkeypatch.setattr(s,'time',SimpleNamespace(monotonic=clock,perf_counter=lambda:0.))
    with pytest.raises(TimeoutError): s.run('smoke',tmp_path/'timeout')
    manifest=json.loads((tmp_path/'timeout/manifest.json').read_text())
    assert manifest['status']=='FAILED' and not manifest['full_evaluation_outcomes_opened']


def typed(path):
    with path.open() as f: raw=list(csv.DictReader(f))
    return [{k:v if k in ('controller','state') or k.endswith('_action') else float(v) if '.' in v or 'e' in v.lower() else int(v)
        for k,v in r.items()} for r in raw]


def test_complete_smoke_grid_replay_hashes_and_tamper_detection(tmp_path):
    cfg=s.settings('smoke'); output=tmp_path/'run'; manifest=s.run('smoke',output)
    assert manifest['status']=='COMPLETED' and manifest['audit']=='PASS'
    assert manifest['actual_counts']==manifest['expected_counts']==dict(episodes=28,physical_transitions=1680,landmarks=28)
    assert manifest['scientific_status']=='ENGINEERING_ONLY' and not manifest['full_evaluation_outcomes_opened']
    assert manifest['forecast_transitions']<=manifest['forecast_cap']
    for name,record in manifest['files'].items():
        data=(output/name).read_bytes()
        assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
    original=(output/'manifest.json').read_bytes()
    with pytest.raises(FileExistsError): s.run('smoke',output)
    assert (output/'manifest.json').read_bytes()==original
    with pytest.raises(ValueError): s.run('smoke',tmp_path/'gpu',device='cuda')
    audit=json.loads((output/'audit.json').read_text())
    assert audit['audited_landmarks']==28 and audit['paired_tapes'] and audit['metrics_replayed']
    assert audit['forecast_transitions']==audit['replay_forecast_transitions']
    rows,local=typed(output/'episodes.csv'),typed(output/'landmarks.csv')
    with (output/'landmark_roots.csv').open() as f: root_rows=list(csv.DictReader(f))
    with (output/'decisions.csv').open() as f: decisions=list(csv.DictReader(f))
    diag=json.loads(decisions[1]['diagnostic']); diag['candidate_count']=99
    decisions[1]['diagnostic']=json.dumps(diag); tampered=tmp_path/'bad.csv'
    with tampered.open('w',newline='') as f:
        writer=csv.DictWriter(f,list(decisions[0])); writer.writeheader(); writer.writerows(decisions)
    with pytest.raises(RuntimeError,match='replay'): s.replay_audit(tampered,cfg,rows,local,root_rows)
    root_rows[0]['selection_endpoint_mean']='999999'
    with pytest.raises(RuntimeError): s.replay_audit(output/'decisions.csv',cfg,rows,local,root_rows)
