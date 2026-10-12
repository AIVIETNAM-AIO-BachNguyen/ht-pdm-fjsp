"""Independent physics, information and simultaneous-action contract checks."""
from dataclasses import asdict, replace

import numpy as np
import pytest

from ht_pdm_fjsp import maintenance_balanced_benchmark_env as b
from ht_pdm_fjsp import maintenance_balanced_search_mac_v2 as search
from ht_pdm_fjsp import maintenance_skill_contention_screen as screen
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition


def test_frozen_config_matches_selected_search_signature_exactly():
    selected = next(s for s in search.specifications('full') if s['candidate'] == 'balanced_a2_b4_u12')
    assert asdict(b.fixed_config()) == selected['config']
    c = b.fixed_config()
    assert c.service_time == ((1, 2), (1, 4), (2, 1), (4, 1))
    assert c.restoration == ((.5, 1.), (.5, 1.), (1., .5), (1., .5))
    assert (c.horizon, c.unavailable_cost, c.failure_age, c.failure_probability) == (60, 12., 4, .45)


def urgent_env():
    env = b.BenchmarkEnv(819100)
    env.reset()
    env.inner.env.state = DispatchState((3,) * 4, (False,) * 4, (0,) * 4, (0, 0), (-1, -1))
    return env


def test_all_twenty_one_joint_matchings_have_collision_free_proposals():
    env = urgent_env()
    obs = b.observation(env.inner)
    assert len(env.inner.actions) == 21 and obs['joint_mask'].all() and obs['local_mask'].all()
    for action in env.inner.actions:
        proposals = np.zeros(4, dtype=int)
        for machine, worker in action:
            proposals[machine] = worker + 1
        decoded, diagnostics = b.arbitrate(env.inner, proposals)
        assert decoded == action
        assert diagnostics['proposals'] == proposals.tolist()
        assert diagnostics['conflict_workers'] == diagnostics['rejected_proposals'] == 0
        validate_matching(env.inner.config, env.inner.env.state, decoded)


def test_local_features_exclude_peer_state_and_occupied_worker_failure():
    env = urgent_env()
    env.inner.env.state = DispatchState((3, 4, 2, 6), (False, True, False, False),
        (0, 4, 0, 0), (0, 2), (-1, 1))
    before = b.observation(env.inner)
    assert before['state'].shape == (71,) and before['local'].shape == (4, 33)
    assert before['local'].dtype == before['state'].dtype == np.float32
    assert np.isfinite(before['local']).all() and np.isfinite(before['state']).all()
    assert np.array_equal(before['local'][:, -4:], np.eye(4))
    assert np.all(before['local'][:, [8, 11]] == 0)
    # M1 is the other machine occupying workerB; its failure bit is hidden.
    env.inner.env.state = replace(env.inner.env.state, ages=(3, 8, 7, 0),
        failed=(False, False, True, True), pending_wait=(0, 0, 12, 7))
    after = b.observation(env.inner)
    assert np.array_equal(before['local'][0], after['local'][0])
    assert np.array_equal(before['local_mask'][0], after['local_mask'][0])
    assert not np.array_equal(before['state'], after['state'])
    env.inner.env.events = tuple((True,) * 4 for _ in range(60))
    assert all(np.array_equal(after[key], b.observation(env.inner)[key]) for key in after)


def test_mask_prohibits_early_pm_busy_machines_and_occupied_workers():
    env = urgent_env()
    env.inner.env.state = DispatchState((2, 3, 1, 4), (False, False, True, False),
        (0, 0, 2, 0), (0, 2), (-1, 3))
    obs = b.observation(env.inner)
    assert np.array_equal(obs['local_mask'], [[True, False, False], [True, True, False],
        [True, True, False], [True, False, False]])
    assert obs['joint_mask'][0]
    for action, legal in zip(env.inner.actions, obs['joint_mask']):
        expected = all(m in (1, 2) and j == 0 for m, j in action)
        assert legal == expected
    initial = env.inner.env.state, env.inner.env.time, dict(env.inner.env.metrics)
    for proposals in ([1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 1], [0., 0., 0., 0.], [0, 0, 0]):
        with pytest.raises(ValueError): env.step(proposals)
        assert (env.inner.env.state, env.inner.env.time, dict(env.inner.env.metrics)) == initial


def test_collision_priority_ties_preserve_losing_proposal_without_reassignment():
    env = urgent_env()
    env.inner.env.state = replace(env.inner.env.state, failed=(False, True, True, False),
        pending_wait=(0, 4, 5, 0))
    proposals = np.array([1, 1, 1, 0])
    matching, diagnostic = b.arbitrate(env.inner, proposals)
    assert matching == ((2, 0),)
    assert diagnostic['proposals'] == [1, 1, 1, 0]
    assert diagnostic['rejected_proposals'] == 2 and diagnostic['conflict_workers'] == 1
    assert diagnostic['executed_starts'] == 1 and diagnostic['unused_feasible_capacity'] == 1
    _, _, _, info = env.step(proposals)
    assert info['coordination']['proposals'] == proposals.tolist()
    assert env.inner.env.state.assigned[1] == -1
    env = urgent_env()
    assert b.arbitrate(env.inner, [0, 2, 2, 0])[0] == ((1, 1),)


def test_terminal_tail_and_team_reward_against_independent_scalar_sixty_ticks():
    env = b.BenchmarkEnv(819101);env.reset()
    c = b.fixed_config();physical = WaitingEnv(c);physical.reset(env.inner.physical_seed)
    state = physical.state;costs = [];infos = [];reward = 0.
    for tick in range(48):
        legal = screen.action_space(c, state)[2]
        action = legal[(tick * 7) % len(legal)]
        proposals = [0] * 4
        for machine, worker in action: proposals[machine] = worker + 1
        state, cost, info = transition(c, state, action, physical.events[tick]);costs.append(cost);infos.append(info)
        _, r, done, result = env.step(proposals if tick % 2 else env.inner.actions.index(action))
        reward += r
        assert done == (tick == 47)
        if tick < 47:
            assert env.inner.env.state == state and env.inner.env.time == tick + 1
        else:
            cutoff = state
            for tail_tick in range(48, 60):
                tail = screen.choose(c, state, 'deadline_matching')
                validate_matching(c, state, tail)
                state, cost, info = transition(c, state, tail, physical.events[tail_tick]);costs.append(cost);infos.append(info)
            assert r == -sum(costs[47:]) / 20.
    metrics = result['episode_metrics']
    assert env.inner.env.state == state and metrics['physical_ticks'] == 60
    assert reward == pytest.approx(-sum(costs) / 20., abs=1e-12)
    assert metrics['policy_cost'] == sum(costs[:48]) and metrics['tail_cost'] == sum(costs[48:])
    assert metrics['population_sha256'] == env.initial_population
    assert metrics['terminal_pending'] == sum(f and m not in cutoff.assigned for m, f in enumerate(cutoff.failed))
    for key in env.inner.env.metrics: assert metrics[key] == sum(row[key] for row in infos)


def test_training_population_hashes_paired_despite_different_algorithm_actions():
    hashes = {}
    for label in ('ppo', 'mappo', 'iql', 'qmix'):
        env = b.BenchmarkEnv(819102, worker=2, training=True)
        history = []
        for episode in range(3):
            env.reset();history.append((env.inner.physical_seed, env.initial_population))
            for tick in range(48):
                if label == 'ppo':
                    index = int(np.flatnonzero(env.inner.action_masks())[-1])
                    env.step(index)
                elif label == 'mappo':
                    env.step([0, 0, 0, 0])
                else:
                    mask = b.observation(env.inner)['local_mask']
                    env.step([int(np.flatnonzero(row)[-1]) for row in mask])
        hashes[label] = history
    assert all(history == hashes['ppo'] for history in hashes.values())
    assert len({seed for seed, _ in hashes['ppo']}) == 3
    other = b.BenchmarkEnv(819102, worker=3, training=True);other.reset()
    assert other.initial_population != hashes['ppo'][0][1]
    evaluation = b.BenchmarkEnv(819102);evaluation.reset()
    assert evaluation.inner.physical_seed != hashes['ppo'][0][0]
