"""Engineering checks for interventions, independent selection and complete clocks."""
import csv
from dataclasses import replace
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ht_pdm_fjsp import maintenance_decision_headroom as e
from ht_pdm_fjsp import maintenance_headroom_forks as f
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv
from ht_pdm_fjsp.maintenance_joint_policy import JOINT


def policy():
    return e.joint.model_for(e.joint.settings('full'), 12100000, JOINT)


def config():
    return e.joint.make_config(15600000, 'dense', 48)


def test_registered_budget_rng_and_fail_closed(tmp_path):
    cfg = e.settings('full')
    assert e.counts(cfg) == dict(models=10, trajectories=640, trajectory_ticks=30720, states=2560,
                               training_steps=0, max_forecast_transitions=26542080)
    assert e.counts(e.settings('smoke'))['states'] == 4
    assert e.seed_audit(cfg)['unique_derived_roles'] == 2583
    row = dict(cohort=cfg['cohorts'][0], shock=cfg['physical_shocks'][0], model_seed=cfg['model_seeds'][0], origin=JOINT, tick=42)
    seeds = [e.forecast_seed(cfg, row, p) for p in e.PHASES] + [e.physical_seed(row['cohort'], row['shock'])]
    assert len(set(seeds)) == 3
    with pytest.raises(ValueError):
        e.forecast_seed(cfg, row, 'future')
    with pytest.raises(ValueError):
        e.seed_audit(dict(cfg, selection_seed=15400000))
    (tmp_path/'keep').write_text('original')
    with pytest.raises(ValueError, match='new/empty'):
        e.run(tmp_path, tmp_path/'absent')
    assert (tmp_path/'keep').read_text() == 'original'
    empty = tmp_path/'failure'
    with pytest.raises(FileNotFoundError):
        e.run(empty, tmp_path/'absent')
    assert json.loads((empty/'manifest.json').read_text())['status'] == 'FAILED'


def test_pm_defer_preserves_cm_and_assignment_machine_set():
    c = config()
    s = DispatchState((4, 3, 1, 0), (True, False, False, False), (2, 0, 0, 0), (0, 0), (-1, -1))
    root = ((0, 1), (1, 0))
    actions, groups = e.candidates(c, s, root)
    assert actions[0] == root and actions[groups['timing'][1]] == ((0, 1),)
    assert len(actions) == 3 and len(groups['assignment']) == 2
    assert all({m for m, _ in actions[i]} == {0, 1} for i in groups['assignment'])
    assert actions[groups['assignment'][1]] == ((0, 0), (1, 1))
    actions, groups = e.candidates(c, s, ((0, 1),))
    assert groups['timing'] == (0,) and len(groups['assignment']) == 2
    busy = replace(s, remaining=(4, 0), assigned=(2, -1))
    actions, groups = e.candidates(c, busy, ((1, 1),))
    assert groups['assignment'] == (0,) and actions[groups['timing'][1]] == ()
    mask = replace(c, service_time=((0, 2), (3, 4), (2, 3), (3, 4)))
    actions, groups = e.candidates(mask, s, root)
    assert groups['assignment'] == (0,)
    both_busy = replace(s, remaining=(4, 2), assigned=(0, 1))
    assert e.candidates(c, both_busy, ()) == (((),), dict(timing=(0,), assignment=(0,)))


def test_selection_independence_ties_and_negative_holdout_retained():
    selection = np.array([[10., 10.], [7., 7.], [1., 1.]])
    chosen = e.choose(selection, (0, 1))
    assert chosen == 1  # Candidate2 has a different machine set and is excluded.
    evaluation = np.array([[10., 10.], [12., 14.], [0., 0.]])
    gain, se = e.heldout_gain(evaluation, chosen)
    assert gain == -3. and se == 1.
    evaluation[1] = 1000  # Holdout cannot change the previously chosen index.
    assert chosen == 1 and e.heldout_gain(evaluation, chosen)[0] == -990.
    assert e.choose(np.ones((3, 2)), (0, 2)) == 0
    assert e.heldout_gain(evaluation, 0) == (0., 0.)
    with pytest.raises(ValueError):
        e.choose(selection, (1,))


@pytest.mark.parametrize('condition', ['dense', 'skill_mask'])
@pytest.mark.parametrize('tick', [6, 42, 47])
def test_scalar_crn_cost_components_and_cutoff(condition, tick):
    c = e.joint.make_config(15600000, condition, 48)
    env = WaitingEnv(c)
    env.reset(15600010)
    s = env.state
    model = policy()
    root = model.select(c, s, 48-tick, 48)[0]
    actions, _ = e.candidates(c, s, root)
    before = e.previous.parameters_sha(model)
    v = f.fork_roots(c, s, actions, model, 15600020, 4, tick, record_trace=True)
    assert f.scalar_replay(c, s, v, model, tick)
    np.testing.assert_array_equal(v['costs'], v['base'] + v['tail'])
    np.testing.assert_array_equal(sum(v['component_costs'][0].values()), v['costs'][0])
    assert v['transitions'] == len(actions)*4*(60-tick)
    assert v['counters']['tail_transition_rows'] == len(actions)*4*12
    assert v['counters']['batched_decision_rows'] == len(actions)*4*(47-tick)
    assert e.previous.parameters_sha(model) == before and env.state == s
    env.events = tuple((True,)*4 for _ in env.events)
    repeat = f.fork_roots(c, s, actions, model, 15600020, 4, tick)
    np.testing.assert_array_equal(v['costs'], repeat['costs'])
    assert not np.isnan(e.padded(v, 'costs', 6)[:len(actions), :4]).any()
    assert np.isnan(e.padded(v, 'costs', 6)[:, 4:]).all()
    other = f.fork_roots(c, s, actions, model, 15600030, 4, tick)
    assert not np.array_equal(v['shocks'], other['shocks'])


def test_service_crosses_cutoff_and_illegal_roots():
    c = replace(config(), service_time=((6, 6),)*4)
    s = DispatchState((4, 3, 1, 0), (True, False, False, False), (2, 0, 0, 0), (0, 0), (-1, -1))
    actions, _ = e.candidates(c, s, ((0, 1), (1, 0)))
    v = f.fork_roots(c, s, actions, policy(), 15600020, 4, 47, record_trace=True)
    assert f.scalar_replay(c, s, v, policy(), 47)
    np.testing.assert_array_equal(v['cutoff'][0]['remaining'][0], np.full((4, 2), 5))
    np.testing.assert_array_equal(v['cutoff'][0]['assigned'][0], np.tile([1, 0], (4, 1)))
    for roots, scenarios, tick in [((), 4, 47), (((), ()), 4, 47), (((),), 1, 47), (((),), 4, 48)]:
        with pytest.raises(ValueError):
            f.fork_roots(c, s, roots, policy(), 1, scenarios, tick)


def statistic_fixture():
    cfg = dict(profile='full', conditions=['dense', 'skill_mask'], cohorts=[1, 2], physical_shocks=[3],
               model_seeds=[4, 5], origins=[JOINT, 'best_rule'], landmarks=[6], analysis_seed=15500000, bootstrap_samples=400)
    rows = [dict(zip(e.KEYS, identity), timing_gain=0., timing_exposure=0, timing_selected=0,
                 assignment_gain=0., assignment_exposure=0, assignment_selected=0)
            for identity in itertools.product(cfg['conditions'], cfg['cohorts'], cfg['physical_shocks'], cfg['model_seeds'], cfg['origins'], cfg['landmarks'])]
    return cfg, rows


def test_statistics_keeps_no_switch_eligible_and_signed_gain():
    cfg, rows = statistic_fixture()
    summary, cells = e.summarize(rows, cfg)
    assert all(v['status']=='INSUFFICIENT' for v in summary['primary_contrasts'].values())
    for row in rows:
        row.update(timing_exposure=1)
        if row['origin'] == JOINT:
            row.update(timing_gain=-6., timing_selected=1)
    summary, _ = e.summarize(rows, cfg)
    v = summary['primary_contrasts']['dense:timing']
    assert v['estimate'] == -3. and v['status'] == 'NEGATIVE_CORRECTION_GAIN'
    assert v['exposure'] == 8 and v['switched'] == 4 and v['coverage'] == 1
    cfg['profile'] = 'smoke'
    assert all(v['status']=='ENGINEERING_ONLY' for v in e.summarize(rows, cfg)[0]['primary_contrasts'].values())
    with pytest.raises(RuntimeError):
        e.summarize(rows[:-1], cfg)
    with pytest.raises(RuntimeError):
        e.summarize(rows+[rows[0]], cfg)
    rows[0]['timing_exposure'] = 0
    with pytest.raises(ValueError):
        e.summarize(rows, cfg)


def test_runner_artifacts_reconstruct_heldout_choices(tmp_path, monkeypatch):
    cfg = e.settings('smoke')
    model = policy()
    source = tmp_path/'source'
    (source/'frozen_models').mkdir(parents=True)
    seed = cfg['model_seeds'][0]
    path = source/'frozen_models'/f'seed_{seed}.pt'
    torch.save(dict(protocol=e.joint.VERSION, feature_contract=e.joint.FEATURE_CONTRACT,
        environment_version=e.joint.ENV_VERSION, algorithm=JOINT, train_seed=seed, settings=e.joint.settings('full'),
        update=80, learned_steps=245760, state_dict=model.state_dict()), path)
    locked = json.loads((e.ROOT/f'configs/{e.VERSION}_sources.json').read_text())
    locked['joint_source_contract']['models'][str(seed)]['sha256'] = e.sha(path)
    real_loads = e.json.loads
    def loads(text, *args, **kwargs):
        data = real_loads(text, *args, **kwargs)
        if isinstance(data, dict) and data.get('protocol') == e.VERSION and 'joint_source_contract' in data:
            return locked
        return data
    monkeypatch.setattr(e.json, 'loads', loads)
    monkeypatch.setattr(e.source, 'verify_source', lambda *args: ({seed: model}, {}, [], {'status': 'PASS'}))
    panels = {(cfg['cohorts'][0], cond): e.joint.make_config(15600000, cond, 48) for cond in cfg['conditions']}
    monkeypatch.setattr(e, 'build_panels', lambda _: (panels, [], {'status': 'PASS'}))
    out = tmp_path/'smoke'
    result = e.run(out, source)
    assert result['status'] == 'COMPLETED' and result['audit'] == 'PASS'
    assert result['actual_counts']['states'] == 4 and result['actual_counts']['trajectory_ticks'] == 192
    assert result['new_full_physical_outcomes_opened'] is False
    rows = list(csv.DictReader((out/'states.csv').open()))
    with np.load(out/'raw_forecast_costs.npz') as data:
        for n, row in enumerate(rows):
            for contrast in e.CONTRASTS:
                selected = int(row[f'{contrast}_selected'])
                choices = tuple(json.loads(row[f'{contrast}_candidates']))
                assert selected == e.choose(data['costs'][n, 0, :int(row['unique_roots'])], choices)
                assert e.heldout_gain(data['costs'][n, 1], selected)[0] == float(row[f'{contrast}_gain'])
            valid = data['valid'][n]
            np.testing.assert_array_equal(data['costs'][n][valid], (data['base'][n]+data['tail'][n])[valid])
            np.testing.assert_array_equal(data['costs'][n][valid], sum(data[k][n] for k in e.COMPONENTS)[valid])
    assert all(e.sha(out/name) == record['sha256'] for name, record in result['artifacts'].items())
