"""Non-clairvoyant forecasts, paired outcomes, ablation and artifact integrity."""

import csv
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ht_pdm_fjsp import maintenance_myopic_lookahead as s
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays, reference_batch
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition


def idle(ages=(3, 3, 3, 3)):
    return DispatchState(ages, (False,)*4, (0,)*4, (0,0), (-1,-1))


def test_fixed_environment_mask_budget_and_full_smoke_split():
    cfg = s.settings("smoke"); c = s.make_config(cfg)
    assert c.service_time == ((1,0),(1,3),(1,2),(1,5))
    assert c.restoration == ((.75,.75),)*4 and c.horizon == 60
    assert c.failure_age == 4 and c.failure_probability == .45 and c.waiting_price == 12
    assert len(s.roots(c, idle())) == 17
    assert cfg['lookahead_ticks'] == s.settings('full')['lookahead_ticks'] == 12
    assert s.settings('full')['scenarios'] == 128
    assert not set(cfg['evaluation_seeds']) & set(s.settings('full')['evaluation_seeds'])
    assert len(s.settings('full')['evaluation_seeds']) == 128 and cfg['train_seeds'] == []
    with pytest.raises(ValueError): s.settings('other')


@pytest.mark.parametrize('probability', [0., .45, 1.])
def test_vector_forecasts_equal_scalar_rollouts_with_wait_failure_and_completion(probability):
    cfg = s.settings('smoke'); c = replace(s.make_config(cfg), failure_probability=probability)
    state = DispatchState((3,5,3,2),(True,False,False,False),(6,0,0,0),(0,0),(-1,-1))
    actions = s.roots(c, state)
    # Include WAIT and both assignments of a shared subset as well as myopic.
    actions = tuple(dict.fromkeys([actions[0], (), ((1,0),(2,1)), ((1,1),(2,0))]))
    bank, horizon, scenarios = 991, 12, 8
    actual = s.forecast_values(c, state, actions, horizon, scenarios, bank)
    shocks = np.random.default_rng(bank).random((horizon,scenarios,4)) < probability
    expected = np.zeros((len(actions),scenarios))
    for i, action in enumerate(actions):
        for j in range(scenarios):
            following = state
            for tick in range(horizon):
                chosen = action if tick == 0 else s.myopic(c, following)
                following, cost, _ = transition(c, following, chosen, shocks[tick,j])
                expected[i,j] += cost
    np.testing.assert_array_equal(actual['samples'],expected)
    assert actual['transitions'] == len(actions)*scenarios*horizon
    np.testing.assert_allclose(actual['means'],expected.mean(1),atol=0,rtol=0)
    np.testing.assert_allclose(actual['mc_se'],expected.std(1,ddof=1)/np.sqrt(scenarios))


def test_batch_continuation_exactly_matches_original_v3_on_real_reachable_states():
    cfg = s.settings('smoke'); c = s.make_config(cfg)
    for seed in cfg['evaluation_seeds']:
        env = WaitingEnv(c); env.reset(s.physical_seed(seed))
        for tick in range(60):
            expected = s.screen.choose(c,env.state,'deadline_matching')
            batch = reference_batch(c,arrays(env.state),12,'deadline_matching')[0]
            actual = tuple((int(m),int(j)) for m,j in zip(*np.nonzero(batch)))
            assert actual == expected == s.myopic(c,env.state)
            env.step(expected)


def test_root_restriction_baseline_ties_and_no_extra_early_prevention():
    cfg = s.settings('smoke'); c = s.make_config(cfg)
    actions = s.roots(c,idle((0,3,3,0)))
    assert all(m in (1,2) for a in actions for m,j in a)
    allowed = s.eligible_indices(actions,True)
    assert len(allowed) == 2 and allowed[0] == 0
    assert all(tuple(m for m,j in actions[i]) == tuple(m for m,j in actions[0]) for i in allowed)
    assert s.minimum(np.zeros(len(actions)),allowed) == 0
    costs = np.full(len(actions),100.); costs[0] = 99.; costs[allowed[1]] = 98.
    assert s.minimum(costs,allowed) == allowed[1]
    with pytest.raises(ValueError): s.minimum(costs,(1,))
    with pytest.raises(ValueError): s.minimum(np.array([np.nan]),(0,))


def test_assignment_shortcut_forced_subset_and_common_tail_do_not_forecast(monkeypatch):
    cfg = s.settings('smoke'); c = s.make_config(cfg); state = idle((3,3,0,0))
    def no_forecast(*a,**k): raise AssertionError('must not forecast')
    monkeypatch.setattr(s,'forecast_values',no_forecast)
    action, diag = s.select(c,state,'assignment_lookahead',cfg,36300000,0)
    assert action == s.myopic(c,state) and not diag['planned']
    for name in s.CONTROLLERS:
        action, diag = s.select(c,state,name,cfg,36300000,48)
        assert action == s.myopic(c,state) and diag['forecast_transitions'] == 0
    with pytest.raises(ValueError): s.select(c,state,'other',cfg,36300000,0)


def test_planner_ignores_physical_future_and_does_not_mutate_current_state():
    cfg = s.settings('smoke'); c = s.make_config(cfg)
    env = WaitingEnv(c); env.reset(s.physical_seed(cfg['evaluation_seeds'][0]),idle((0,3,3,3)))
    before = env.state
    first = s.select(c,env.state,'joint_lookahead',cfg,cfg['evaluation_seeds'][0],0)
    env.events = [(False,)*4]*60
    second = s.select(c,env.state,'joint_lookahead',cfg,cfg['evaluation_seeds'][0],0)
    assert first == second and env.state == before
    assert first[1]['selection_seed'] != s.physical_seed(cfg['evaluation_seeds'][0])


def test_selection_and_holdout_landmark_banks_separate_and_heldout_does_not_select(monkeypatch):
    cfg = s.settings('smoke'); c = s.make_config(cfg); state = idle((0,3,3,0))
    actions = s.roots(c,state); calls=[]
    def synthetic(c,state,roots,horizon,scenarios,seed,check_timeout=None):
        calls.append(seed)
        costs=np.full((len(roots),scenarios),100.)
        # Selection prefers last action; holdout makes it worse than baseline.
        costs[-1] = 90. if seed == s.forecast_seed(cfg,36300000,0) else 110.
        return dict(samples=costs,means=costs.mean(1),mc_se=np.zeros(len(roots)),transitions=len(roots)*scenarios*horizon)
    monkeypatch.setattr(s,'forecast_values',synthetic)
    row=s.landmark(cfg,state,36300000,0)
    assert calls == [s.forecast_seed(cfg,36300000,0),s.forecast_seed(cfg,36300000,0,'holdout')]
    assert row['joint_root'] == len(actions)-1 and row['joint_heldout_gain'] == -10.
    assert row['selection_seed'] != row['holdout_seed']


def test_assignment_controller_cannot_switch_machine_subset():
    cfg = s.settings('smoke'); c=s.make_config(cfg)
    for state in (idle(),idle((0,3,3,0)),idle((0,3,3,3))):
        action, diag = s.select(c,state,'assignment_lookahead',cfg,36300000,1)
        assert tuple(m for m,j in action) == tuple(m for m,j in s.myopic(c,state))
        assert action in s.screen.action_space(c,state)[2]


def test_seed_roles_new_locked_and_disjoint_from_three_screens():
    cfg=s.settings('smoke'); audit=s.seed_audit(cfg)
    assert audit['raw_unique']==136 and audit['derived_unique']==7394
    assert audit['prior_derived_checked']==606
    with pytest.raises(ValueError,match='locked'): s.seed_audit({**cfg,'scenarios':64})
    with pytest.raises(ValueError): s.forecast_seed(cfg,36300000,60)


def synthetic(cfg,costs=(100.,95.,98.),availability=(.7,.7,.7),pending=(0.,0.,0.)):
    rows=[dict(controller=p,seed=seed,physical_seed=s.physical_seed(seed),extended_cost=costs[i],
        availability=availability[i],terminal_pending=pending[i])
        for seed in cfg['evaluation_seeds'] for i,p in enumerate(s.CONTROLLERS)]
    local=[dict(seed=seed,tick=tick,joint_heldout_gain=2.,assignment_heldout_gain=1.,
        joint_disagreement_rate=1,assignment_disagreement_rate=1)
        for seed in cfg['evaluation_seeds'] for tick in cfg['landmarks']]
    return rows,local


def test_bootstrap_is_paired_ratio_of_means_and_guards_bind_even_with_cost_gain():
    cfg=s.settings('smoke'); rows,local=synthetic(cfg)
    result=s.summarize(cfg,rows,local)
    assert result['scientific_status']=='ENGINEERING_ONLY' and all(result['gates'].values())
    assert len(result['comparisons'])==5
    assert result['comparisons']['joint_vs_myopic']['estimate']==pytest.approx(.05)
    assert result['comparisons']['joint_vs_myopic']['ci']==pytest.approx([.05,.05])
    # Varied denominators: ratio of mean costs differs from mean percentages.
    for row in rows:
        if row['seed']==cfg['evaluation_seeds'][0]:
            row['extended_cost']={'myopic_matching':200.,'joint_lookahead':180.,'assignment_lookahead':190.}[row['controller']]
    result=s.summarize(cfg,rows,local)
    assert result['comparisons']['joint_vs_myopic']['estimate']==pytest.approx(1-116.25/125.)
    rows,local=synthetic(cfg,availability=(.7,.65,.7),pending=(0.,1.,0.))
    result=s.summarize(cfg,rows,local)
    assert result['gates']['practical_cost'] and not result['gates']['availability'] and not result['gates']['pending']
    with pytest.raises(RuntimeError,match='grid'): s.summarize(cfg,rows[:-1],local)
    with pytest.raises(RuntimeError,match='landmark'): s.summarize(cfg,rows,local[:-1])


def test_full_result_classification_on_synthetic_arithmetic_not_full_trajectories():
    cfg=s.settings('full')
    rows,local=synthetic(cfg)
    assert s.summarize(cfg,rows,local)['scientific_status']=='JOINT_HEADROOM_CONFIRMED'
    rows,local=synthetic(cfg,costs=(100.,100.1,98.))
    result=s.summarize(cfg,rows,local)
    assert result['scientific_status']=='NOT_CONFIRMED' and result['comparisons']['joint_vs_myopic']['ci'][1]<0


def test_initial_busy_worker_forecast_stays_feasible_and_matches_scalar():
    cfg=s.settings('smoke'); c=s.make_config(cfg)
    state=DispatchState((3,4,3,0),(False,True,False,False),(0,0,0,0),(0,3),(-1,1))
    actions=s.roots(c,state)
    assert all(j==0 and m!=1 for action in actions for m,j in action)
    actual=s.forecast_values(c,state,actions,6,8,777)
    shocks=np.random.default_rng(777).random((6,8,4))<c.failure_probability
    for i,action in enumerate(actions):
        for scenario in range(8):
            following=state; expected=0.
            for tick in range(6):
                chosen=action if tick==0 else s.myopic(c,following)
                following,cost,_=transition(c,following,chosen,shocks[tick,scenario]); expected+=cost
            assert actual['samples'][i,scenario]==expected


def test_forecast_timeout_callback_interrupts_before_transition():
    cfg=s.settings('smoke'); c=s.make_config(cfg)
    def expired(): raise TimeoutError('test deadline')
    with pytest.raises(TimeoutError): s.forecast_values(c,idle(),s.roots(c,idle()),12,8,123,expired)


def test_run_records_timeout_and_dirty_full_stops_before_opening_panel(tmp_path,monkeypatch):
    cfg=s.settings('smoke')
    monkeypatch.setattr(s.screen,'git_metadata',lambda:dict(commit='test',branch='test',dirty=True))
    with pytest.raises(RuntimeError,match='clean'): s.run('full',tmp_path/'full')
    assert not (tmp_path/'full').exists()
    counter=[-601.]
    def expired_clock(): counter[0]+=601.; return counter[0]
    monkeypatch.setattr(s,'time',SimpleNamespace(monotonic=expired_clock,perf_counter=lambda:0.))
    with pytest.raises(TimeoutError): s.run('smoke',tmp_path/'timeout')
    manifest=json.loads((tmp_path/'timeout/manifest.json').read_text())
    assert manifest['status']=='FAILED' and not manifest['full_evaluation_outcomes_opened']
    assert 'TimeoutError' in manifest['error']


def test_complete_smoke_replay_forecasts_artifacts_and_tamper_detection(tmp_path):
    cfg=s.settings('smoke'); output=tmp_path/'run'; manifest=s.run('smoke',output)
    assert manifest['status']=='COMPLETED' and manifest['audit']=='PASS'
    assert manifest['actual_counts']==manifest['expected_counts']==dict(episodes=12,physical_transitions=720,landmarks=28)
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
    # Rebuild typed records for replay; changing a recorded action must be caught.
    with (output/'episodes.csv').open() as f: rows=list(csv.DictReader(f))
    rows=[{k:v if k=='controller' else int(v) if k in ('seed','physical_seed') else float(v) for k,v in r.items()} for r in rows]
    with (output/'landmarks.csv').open() as f: local=list(csv.DictReader(f))
    text={'state','joint_action','assignment_action'}
    ints={'seed','tick','roots','selection_seed','holdout_seed','forecast_transitions','joint_root','assignment_root'}
    local=[{k:v if k in text else int(v) if k in ints else float(v) for k,v in r.items()} for r in local]
    with (output/'landmark_roots.csv').open() as f: root_rows=list(csv.DictReader(f))
    with (output/'decisions.csv').open() as f: decisions=list(csv.DictReader(f))
    decisions[0]['diagnostic']=json.dumps(dict(planned=True,candidate_count=99,selection_seed=0,
        selected_mean=0,myopic_mean=0,forecast_transitions=0))
    tampered=tmp_path/'bad.csv'
    with tampered.open('w',newline='') as f:
        writer=csv.DictWriter(f,list(decisions[0])); writer.writeheader(); writer.writerows(decisions)
    with pytest.raises(RuntimeError,match='replay'): s.replay_audit(tampered,cfg,rows,local,root_rows)
