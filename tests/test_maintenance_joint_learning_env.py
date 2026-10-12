"""Physical accounting and seed/action contracts of the bounded learning wrapper."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed, validate_matching
from ht_pdm_fjsp.maintenance_joint_learning_env import JointLearningEnv
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, transition
from ht_pdm_fjsp import maintenance_skill_contention_screen as screen


def config():
    return WaitingConfig(4, 2, 60, 4, .45, ((1, 0), (1, 3), (1, 2), (1, 5)),
        ((.75, .75),) * 4, waiting_limit=4, waiting_price=12.)


class Journal:
    def __init__(self): self.rows = []
    def add(self, record): self.rows.append(record)


def test_sixty_tick_independent_scalar_rollout_tail_collapse_and_accounting():
    c = config();journal = Journal();gym = JointLearningEnv(c, journal=journal)
    obs, initial = gym.reset(seed=810012)
    independent = WaitingEnv(c);independent.reset(role_seed(810012, 'maintenance_joint_learning_mac_v1:physical'))
    state = independent.state;costs = [];reward = 0.;all_info = [];cutoff = None
    assert initial['physical_seed'] == gym.physical_seed
    for tick in range(48):
        # Alternate waiting with a physically valid rule to exercise failures,
        # blocked resources and pending waiting, not only routine no-failure PM.
        action = () if tick % 3 == 0 else screen.choose(c, gym.env.state, 'deadline_matching')
        action_index = gym.actions.index(action)
        assert gym.action_masks()[action_index]
        state, cost, scalar_info = transition(c, state, action, independent.events[tick])
        costs.append(cost);all_info.append(scalar_info)
        obs, r, terminated, truncated, info = gym.step(action_index);reward += r
        assert terminated == (tick == 47) and not truncated
        assert gym.observation_space.contains(obs) and obs.dtype == np.float32
        if tick < 47:
            assert gym.env.time == tick + 1 and gym.env.state == state
            assert r == -cost / 20.
        else:
            cutoff = state
            for tail_tick in range(48, 60):
                tail = screen.choose(c, state, 'deadline_matching')
                validate_matching(c, state, tail)
                state, tail_cost, scalar_info = transition(c, state, tail, independent.events[tail_tick])
                costs.append(tail_cost);all_info.append(scalar_info)
            assert gym.env.time == 60 and gym.env.state == state
            assert r == -sum(costs[47:]) / 20.
    metrics = info['episode_metrics']
    assert reward == pytest.approx(-sum(costs) / 20., abs=1e-12)
    assert metrics['extended_cost'] == sum(costs)
    assert metrics['policy_cost'] == sum(costs[:48])
    assert metrics['tail_cost'] == sum(costs[48:])
    assert metrics['physical_ticks'] == 60
    assert metrics['terminal_pending'] == sum(f and m not in cutoff.assigned for m, f in enumerate(cutoff.failed))
    assert metrics['tail_pending'] == sum(f and m not in state.assigned for m, f in enumerate(state.failed))
    assert metrics['availability'] == 1-sum(r['unavailable_ticks'] for r in all_info[:48])/192
    for key in gym.env.metrics:
        assert metrics[key] == sum(r[key] for r in all_info)
    assert len(journal.rows) == 60
    assert [r['tick'] for r in journal.rows] == list(range(60))
    assert [r['phase'] for r in journal.rows] == ['policy'] * 48 + ['tail'] * 12
    assert [r['cost'] for r in journal.rows] == costs
    for row in journal.rows:
        before = DispatchState(**{key: tuple(value) for key, value in json.loads(row['state']).items()})
        following, cost, _ = transition(c, before, tuple(map(tuple, json.loads(row['action']))), json.loads(row['events']))
        assert json.loads(json.dumps(asdict(following))) == json.loads(row['following']) and cost == row['cost']
    with pytest.raises(RuntimeError, match='reset required'):
        gym.step(0)


def test_exact_urgent_mask_catalogue_and_invalid_action_has_no_side_effects():
    gym = JointLearningEnv(config());gym.reset(seed=810013)
    gym.env.state = DispatchState((0, 2, 3, 5), (False, True, False, True),
        (0, 1, 0, 5), (0, 2), (-1, 3))
    assert len(gym.actions) == 21 and gym.actions[0] == ()
    mask = gym.action_masks()
    assert mask.dtype == bool and mask[0]
    # Independent legal-edge and urgency check, including occupied M3/T1.
    for index, action in enumerate(gym.actions):
        legal = all(m not in gym.env.state.assigned and gym.env.state.remaining[j] == 0
            and gym.config.service_time[m][j] > 0
            and (gym.env.state.failed[m] or gym.env.state.ages[m] >= 3) for m, j in action)
        assert mask[index] == legal
    before = (gym.env.state, gym.env.time, dict(gym.env.metrics), gym.policy_time)
    for bad in (-1, 21, 1.5, True, gym.actions.index(((0, 0),))):
        with pytest.raises(ValueError): gym.step(bad)
        assert (gym.env.state, gym.env.time, dict(gym.env.metrics), gym.policy_time) == before
    valid = gym.actions.index(((1, 0),))
    assert mask[valid]
    gym.step(valid)


def test_observation_shape_absent_edges_policy_clock_and_no_future_leakage():
    gym = JointLearningEnv(config());obs, _ = gym.reset(seed=810014)
    assert obs.shape == (71,) and np.isfinite(obs).all()
    edges = obs[30:62].reshape(4, 2, 4)
    assert np.array_equal(edges[0, 1], np.zeros(4))
    assert obs[62] == 48 / 12
    mask = gym.action_masks().copy()
    gym.env.events = tuple((True,) * 4 for _ in range(60))
    assert np.array_equal(obs, gym._observation())
    assert np.array_equal(mask, gym.action_masks())
    gym.env.events = tuple((False,) * 4 for _ in range(60))
    assert np.array_equal(obs, gym._observation())
    gym.step(0)
    assert gym._observation()[62] == np.float32(47 / 12)


def test_repeated_explicit_evaluation_seed_restarts_identical_episode():
    gym = JointLearningEnv(config(), namespace='probe_unit');obs, info = gym.reset(seed=810015)
    events = gym.env.events;state = gym.env.state
    assert info['physical_seed'] == role_seed(810015, 'probe_unit:physical')
    gym.step(0)
    again, repeated = gym.reset(seed=810015)
    assert np.array_equal(obs, again) and info == repeated
    assert gym.env.events == events and gym.env.state == state
    with pytest.raises(ValueError, match='explicit raw seed'): gym.reset()


def test_training_stream_advances_and_restarts_and_workers_are_separate():
    a = JointLearningEnv(config(), training=True, worker=0, namespace='probe_unit')
    b = JointLearningEnv(config(), training=True, worker=1, namespace='probe_unit')
    with pytest.raises(ValueError, match='initial explicit'): a.reset()
    first_obs, first = a.reset(seed=810016)
    _, second = a.reset()
    _, third = a.reset()
    assert [first['episode_index'], second['episode_index'], third['episode_index']] == [0, 1, 2]
    assert len({first['physical_seed'], second['physical_seed'], third['physical_seed']}) == 3
    assert first['physical_seed'] == role_seed(810016, 'probe_unit:training_physical:worker:0:episode:0')
    _, other_worker = b.reset(seed=810016)
    assert other_worker['physical_seed'] != first['physical_seed']
    restarted_obs, restarted = a.reset(seed=810016)
    assert restarted == first and np.array_equal(first_obs, restarted_obs)
    evaluation = JointLearningEnv(config(), namespace='probe_unit')
    _, eval_info = evaluation.reset(seed=810016)
    assert eval_info['physical_seed'] != first['physical_seed']


def test_configuration_contract_and_step_requires_reset():
    with pytest.raises(ValueError, match='N4/K2/H48'):
        JointLearningEnv(replace(config(), horizon=48))
    gym = JointLearningEnv(config())
    with pytest.raises(RuntimeError, match='reset required'): gym.step(0)
    with pytest.raises(ValueError): gym.reset(seed=-1)
    with pytest.raises(ValueError): gym.reset(seed=1, options={'initial': 'unsupported'})
