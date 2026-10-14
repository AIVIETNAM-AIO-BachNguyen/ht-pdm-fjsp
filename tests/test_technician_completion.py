from dataclasses import replace
import itertools
import numpy as np
import pytest
import torch

from ht_pdm_fjsp.technician_completion_env import ResourceEnv, GRAPH_ARMS, VERSION
from ht_pdm_fjsp.technician_completion_policy import CompletionPolicy, initial_calendar, calendar_embeddings
from ht_pdm_fjsp.technician_completion_learning import prepare_advantage, allocation_objective, value_loss
from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors, parameter_digest, fit
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_matching


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def urgent(n=4, arm='completion_credit_ppo'):
    env = ResourceEnv(70100000, n, arm)
    env.reset()
    env.env.state = DispatchState((3,) * n, (False,) * n, (0,) * n,
                                 (0,) * (n // 2), (-1,) * (n // 2))
    return env


def model(arm):
    with torch.random.fork_rng():
        torch.manual_seed(70000000)
        return CompletionPolicy(arm)


@pytest.mark.parametrize('arm', GRAPH_ARMS[:3])
def test_unique_matching_mass_no_order_duplicates_and_wait_continuation(arm):
    env = urgent(arm=arm)
    m = model(arm)
    batch = tensors([env.observe()])
    support = set()
    mass = 0.
    for trace in itertools.product(range(5), repeat=2):
        if trace[0] == trace[1] != 0:
            with pytest.raises(ValueError):
                env.resolve(np.asarray(trace))
            with pytest.raises(ValueError):
                m.evaluate(batch, torch.tensor([trace]))
            continue
        pairs, _ = env.resolve(np.asarray(trace))
        assert pairs not in support
        support.add(pairs)
        log, entropy, values = m.evaluate(batch, torch.tensor([trace]))
        mass += float(log.detach().sum(1).exp())
        assert values.shape == entropy.shape == log.shape == (1, 2)
        assert torch.isfinite(entropy).all()
    assert len(support) == 21 and mass == pytest.approx(1., abs=2e-6)
    assert env.resolve(np.array([0, 2]))[0] == ((1, 1),)


@pytest.mark.parametrize('n', [4, 8, 16, 32])
def test_matched_weights_capacity_exact_replay_and_finite_gradients(n):
    models = [model(arm) for arm in GRAPH_ARMS]
    assert len({parameter_digest(m) for m in models}) == 1
    assert {sum(p.numel() for p in m.parameters()) for m in models} == {18444}
    for m in models:
        env = urgent(n, m.algorithm)
        batch = tensors([env.observe()])
        actions, log, value = m.sample(batch, torch.Generator().manual_seed(1))
        other, entropy, v = m.evaluate(batch, actions)
        assert torch.equal(log, other) and torch.equal(value, v)
        validate_matching(env.config, env.env.state, env.resolve(actions[0].numpy())[0])
        (-other.mean() - .01 * entropy.mean() + v.square().mean()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_busy_technician_forced_wait_is_zero_log_entropy():
    env = urgent()
    env.env.state = replace(env.env.state, remaining=(2, 0), assigned=(0, -1))
    m = model('completion_credit_ppo')
    batch = tensors([env.observe()])
    actions, log, value = m.sample(batch, torch.Generator().manual_seed(7))
    assert actions[0, 0] == 0 and log[0, 0] == 0
    assert m.evaluate(batch, actions)[1][0, 0] == 0
    with pytest.raises(ValueError):
        m.evaluate(batch, torch.tensor([[1, 2]]))


def test_existing_and_reserved_service_calendar_physical_completion_age():
    env = urgent()
    env.env.state = replace(env.env.state, ages=(7, 3, 3, 3), remaining=(2, 0), assigned=(0, -1))
    batch = tensors([env.observe()])
    calendar, planned = initial_calendar(batch)
    assert calendar[0, 0].tolist() == pytest.approx([1., 2 / 12, 3 / 8])
    assert planned[0, 0].tolist() == [1., 0.]
    env = urgent()
    m = model('completion_credit_ppo')
    captures = []
    handle = m.value_calendar_encoder.register_forward_pre_hook(
        lambda module, inputs: captures.append(inputs[0].detach().clone()))
    try:
        m.evaluate(tensors([env.observe()]), torch.tensor([[3, 1]]))
    finally:
        handle.remove()
    # Machine2 starts with technician0: time2, full restoration, age at finish0.
    assert captures[1][0, 2, 0, :3].tolist() == pytest.approx([1., 2 / 12, 0.])
    assert captures[0][0, 2, 0, :3].tolist() == pytest.approx([0., 0., 3 / 8])


def test_binary_calendar_cache_matches_direct_pair_mlp():
    m = model('completion_credit_ppo')
    calendar = torch.rand(2, 4, 3)
    planned = torch.randint(0, 2, (2, 4, 2)).float()
    cached = calendar_embeddings(m.calendar_encoder, calendar, planned)
    direct = m.calendar_encoder(torch.cat([calendar[:, :, None].expand(-1, -1, 2, -1),
                                         planned[..., None]], -1))
    assert torch.allclose(cached, direct, atol=2e-7, rtol=0)


@pytest.mark.parametrize('arm', GRAPH_ARMS)
def test_baseline_precedes_current_and_future_choices(arm):
    env = urgent(arm=arm)
    m = model(arm)
    batch = tensors([env.observe()])
    a = m.evaluate(batch, torch.tensor([[3, 1]]))[2]
    b = m.evaluate(batch, torch.tensor([[3, 2]]))[2]
    c = m.evaluate(batch, torch.tensor([[1, 2]]))[2]
    assert torch.equal(a, b)
    assert torch.equal(a[:, 0], c[:, 0])
    assert not torch.equal(a[:, 1], c[:, 1])


@pytest.mark.parametrize('arm', GRAPH_ARMS)
def test_machine_permutation_with_identical_technician_order(arm):
    env = urgent(arm=arm)
    obs = env.observe()
    order = np.array([2, 0, 3, 1])
    other = {key: value.copy() for key, value in obs.items()}
    other['machines'] = obs['machines'][order]
    other['edges'] = obs['edges'][order]
    other['pair_mask'] = obs['pair_mask'][order]
    trace = torch.tensor([[3, 1]])
    mapped = torch.tensor([[int(np.where(order == token - 1)[0][0]) + 1 for token in trace[0]]])
    m = model(arm)
    left = m.evaluate(tensors([obs]), trace)
    right = m.evaluate(tensors([other]), mapped)
    assert all(torch.allclose(a, b, atol=1e-6, rtol=0) for a, b in zip(left, right))


def test_independent_technician_contention_quality_arbitration():
    env = urgent(arm='independent_ppo')
    pairs, diagnostics = env.resolve(np.array([1, 1]))
    assert len(pairs) == 1 and diagnostics['conflicts'] == diagnostics['rejected_proposals'] == 1
    m = model('independent_ppo')
    assert torch.isfinite(m.evaluate(tensors([env.observe()]), torch.tensor([[1, 1]]))[0]).all()


def test_modes_use_same_prefix_critic_supervision_and_different_surrogates():
    returns = torch.tensor([2., 4.])
    values = torch.tensor([[0., 1.], [2., 4.]])
    prefix, _ = prepare_advantage(returns, values, 'prefix')
    state, _ = prepare_advantage(returns, values, 'state')
    joint, _ = prepare_advantage(returns, values, 'joint')
    assert prefix.shape == state.shape == (2, 2) and joint.shape == (2,)
    assert torch.equal(state[:, 0], state[:, 1]) and torch.equal(state[:, 0], joint)
    assert not torch.equal(prefix, state)
    assert value_loss(values, returns, 1.) == pytest.approx((4 + 1 + 4 + 0) / 4)
    old = torch.zeros(2, 2)
    new = torch.tensor([[.3, -.1], [-.4, .2]])
    advantage = torch.tensor([[1., -1.], [-2., 2.]])
    r = new.exp()
    expected = torch.minimum(r * advantage, r.clamp(.8, 1.2) * advantage).mean()
    assert allocation_objective(new, old, advantage, 'prefix', .2) == expected
    joint_advantage = torch.tensor([1., -2.])
    r = new.sum(1).exp()
    expected = torch.minimum(r * joint_advantage, r.clamp(.8, 1.2) * joint_advantage).mean()
    assert allocation_objective(new, old, joint_advantage, 'joint', .2) == expected


def test_no_future_events_or_large_catalogue(monkeypatch):
    import ht_pdm_fjsp.maintenance_heterogeneous_model as module
    monkeypatch.setattr(module, 'catalogue', lambda *args: (_ for _ in ()).throw(AssertionError('catalogue forbidden')))
    for n in (8, 16, 32):
        env = urgent(n)
        obs = env.observe()
        env.env.events = ((True,) * n,) * 60
        assert all(np.array_equal(value, env.observe()[key]) for key, value in obs.items())
        action = model('completion_credit_ppo').select(obs)
        validate_matching(env.config, env.env.state, env.resolve(action)[0])
        assert env.actions is None


@pytest.mark.parametrize('arm', GRAPH_ARMS)
def test_exact_budget_training_and_behavior_replay_for_each_learning_mode(arm):
    cfg = dict(env_steps=384, vector_envs=2, rollout=192, epochs=2, minibatch=48,
        learning_rate=.0003, gradient_clip=.5, gamma=1., checkpoints=[192, 384],
        clip=.2, entropy_weight=.01, value_weight=.5, torch_threads=1,
        rng_namespace=VERSION + ':engineering:', group_gradient_clipping=True,
        log_gradient_groups=True, behavior_log_atol=2e-5, behavior_log_rtol=0.,
        behavior_value_atol=2e-5, behavior_value_rtol=1e-6)
    m = model(arm)
    old = parameter_digest(m)
    audit = fit(m, cfg, 70000000, lambda seed, worker: ResourceEnv(seed, 4, arm, worker, True))
    assert audit['learned_steps'] == 384 and audit['optimizer_steps'] == 16
    assert audit['training_episodes'] == 8 and audit['physical_ticks'] == 480
    assert audit['behavior_normalized_max_error'] <= 1 and parameter_digest(m) != old


def test_advantages_are_frozen_behavior_baselines():
    values = torch.tensor([[0., 1.], [2., 3.]], requires_grad=True)
    returns = torch.tensor([2., 4.], requires_grad=True)
    for mode in ('prefix', 'state', 'joint'):
        advantage, _ = prepare_advantage(returns, values, mode)
        assert not advantage.requires_grad


def test_actor_only_inference_preserves_actions_and_skips_critic(monkeypatch):
    env = urgent()
    m = model('completion_credit_ppo')
    expected = m.sample(tensors([env.observe()]), deterministic=True)[0][0].numpy()
    monkeypatch.setattr(m.value_encoder, 'forward', lambda *args: (_ for _ in ()).throw(AssertionError('inference critic')))
    assert np.array_equal(m.select(env.observe()), expected)


def test_fresh_derived_seed_roles_and_round_not_reset():
    from ht_pdm_fjsp import technician_completion_experiment as x
    assert x.seed_audit()['status'] == 'PASS'
    c = x.settings('confirmation')
    assert c['hidden'] == 27 and not c['teacher_assistance'] and c['pretraining_steps'] == 0
    assert c['confirmation_round'] == 2 and c['confidence'] == 1 - (.05 / 6) / 35
    assert c['bootstrap_samples'] == 200000
    with pytest.raises(ValueError):
        x.validate_prerequisite(None, c, {})
    with pytest.raises(ValueError):
        x.validate_pilot(None, {})


def test_complete_statistical_family_null_and_missing_cell():
    from ht_pdm_fjsp import technician_completion_experiment as x
    c = dict(x.settings('smoke'), scales=[4, 8, 16, 32], train_seeds=[1, 2],
             test_seeds=[3, 4], bootstrap_samples=200)
    rows = [dict(n=n, method=a, train_seed=t, seed=s, normalized_cost=100.,
                 availability=.7, normalized_pending=0., normalized_failures=0.)
            for n in c['scales'] for a, t in [(a, t) for a in x.arms(n) for t in c['train_seeds']]
            + [('deadline', -1)] for s in c['test_seeds']]
    latencies = {f'{n}:{a}': [1., 2.] for n in c['scales'] for a in (*x.arms(n), 'deadline')}
    s = x.summarize(rows, c, latencies)
    assert s['family_size'] == 35 and not s['goal_pass']
    assert not s['guards']['credit_mechanism'] and not s['guards']['clipping_mechanism']
    assert s['intervals']['pooled_credit_gain']['estimate'] == 0
    assert s['intervals']['pooled_clipping_gain']['estimate'] == 0
    with pytest.raises(RuntimeError, match='test grid'):
        x.summarize(rows[:-1], c, latencies)


def test_saved_checkpoint_roundtrip_and_metadata_rejection(tmp_path):
    from ht_pdm_fjsp import technician_completion_experiment as x
    c = x.settings('smoke')
    env = urgent()
    m = x.make_model('completion_credit_ppo', 4, c, 70000000)
    path = tmp_path / 'checkpoint.pt'
    x.save_model(m, path, 4, c, 70000000, 0, env.observe())
    restored = x.load_model(path, 4, c, 70000000, 'completion_credit_ppo', 0)
    assert parameter_digest(restored) == parameter_digest(m)
    with pytest.raises(RuntimeError):
        x.load_model(path, 4, c, 70000000, 'state_credit_ppo', 0)
    with pytest.raises(FileExistsError):
        x.save_model(m, path, 4, c, 70000000, 0, env.observe())


def test_serial_spawn_scratch_training_determinism(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import time
    from ht_pdm_fjsp import technician_completion_experiment as x
    c = x.settings('smoke')
    c['development_seeds'] = c['development_seeds'][:1]
    left, right = tmp_path / 'serial', tmp_path / 'spawn'
    left.mkdir(); right.mkdir()
    args = (c, 4, 'completion_credit_ppo', 70000000)
    serial = x.fit_job((*args, str(left), time.monotonic()))
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn')) as pool:
        parallel = pool.submit(x.fit_job, (*args, str(right), time.monotonic())).result(timeout=120)
    assert serial['initial'] == parallel['initial']
    assert serial['choice']['sha256'] == parallel['choice']['sha256']
    assert serial['audit']['parameter_sha256'] == parallel['audit']['parameter_sha256']
    assert serial['audit']['from_scratch'] and serial['audit']['pretraining_steps'] == 0


def test_pilot_prerequisite_binds_source_and_cost_health_without_hiding_mechanism(tmp_path):
    import json
    from ht_pdm_fjsp import technician_completion_experiment as x
    c = x.settings('pilot')
    summary = dict(screen=dict(N4_gain_vs_ppo=True, N4_failures_per_machine_difference=True,
                              pooled_credit_gain=False, pooled_clipping_gain=False),
        selected_steps={f'{n}:completion_credit_ppo:{t}': 30720
                        for n in (4, 32) for t in c['train_seeds']})
    for name, data in [('resolved_config.json', c), ('audit.json', {'status': 'PASS'}), ('summary.json', summary)]:
        (tmp_path / name).write_text(json.dumps(data))
    manifest = dict(protocol=VERSION, status='COMPLETED', audit_status='PASS', test_opened=False,
                    source_seal={'source_snapshot/example.py': 'frozen'},
                    output_bindings={name: {'sha256': x.sha(tmp_path / name)}
                                     for name in ('resolved_config.json', 'audit.json', 'summary.json')})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    result = x.validate_pilot(tmp_path, {'source_snapshot/example.py': 'frozen'})
    assert result['status'] == 'PASS' and not any(result['mechanism_point_proxy_pass'].values())
    with pytest.raises(ValueError, match='source/protocol/registry'):
        x.validate_pilot(tmp_path, {'source_snapshot/example.py': 'changed'})
    summary['screen']['N4_failures_per_machine_difference'] = False
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='changed pilot artifact'):
        x.validate_pilot(tmp_path, {})
    manifest['output_bindings']['summary.json']['sha256'] = x.sha(tmp_path / 'summary.json')
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='cost/health'):
        x.validate_pilot(tmp_path, {})


def test_confirmation_rejects_failed_full_development(tmp_path):
    import json
    from ht_pdm_fjsp import technician_completion_experiment as x
    for name, data in [('resolved_config.json', {'profile': 'development'}),
                      ('summary.json', {'development_screen_pass': False}),
                      ('manifest.json', dict(protocol=VERSION, status='COMPLETED', audit_status='PASS', test_opened=False))]:
        (tmp_path / name).write_text(json.dumps(data))
    with pytest.raises(ValueError, match='development prerequisite fails'):
        x.validate_prerequisite(tmp_path, x.settings('confirmation'), {})


def test_original_ppo_training_is_bitwise_equal_to_frozen_v3_learner():
    import subprocess
    import types
    from pathlib import Path
    from ht_pdm_fjsp import maintenance_balanced_algorithms as current
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(['git', 'show',
        'ed0c784:src/ht_pdm_fjsp/maintenance_balanced_algorithms.py'], cwd=root, text=True)
    frozen = types.ModuleType('frozen_original_ppo_learner')
    exec(compile(source, 'frozen_original_ppo_learner', 'exec'), frozen.__dict__)
    cfg = dict(env_steps=384, vector_envs=2, rollout=192, epochs=2, minibatch=48,
        learning_rate=.0003, gradient_clip=.5, gamma=1., checkpoints=[192, 384],
        clip=.2, entropy_weight=.01, value_weight=.5, torch_threads=1,
        rng_namespace=VERSION + ':legacy_equivalence:')
    digests = []
    for implementation in (frozen, current):
        with torch.random.fork_rng():
            torch.manual_seed(70000000)
            m = implementation.Model('ppo', 64)
        audit = implementation.fit(m, cfg, 70000000,
            lambda seed, worker: ResourceEnv(seed, 4, 'ppo', worker, True))
        digests.append((implementation.parameter_digest(m), audit))
    assert digests[0] == digests[1]


def test_critic_preserves_resource_grouping_for_contention_alias():
    # Both states have identical multisets of raw pair features, but one puts
    # both urgent jobs on the same fastest technician; the other separates them.
    m = model('completion_credit_ppo')
    # Test representational capacity, not knowledge in a randomly initialized
    # critic. This witness uses current age/service inputs only and is never
    # trained, saved or used by any experiment policy.
    with torch.no_grad():
        for parameters in m.gradient_groups()['critic']:
            parameters.zero_()
        m.value_encoder[0].weight[0, 0] = 8.
        m.value_encoder[0].weight[0, 10] = -12.
        m.value_encoder[0].bias[0] = .5
        m.value_encoder[2].weight[0, 0] = 1.
        m.value_encoder[4].weight[0, 0] = 1.
        m.value_context[0].weight[0, 0] = 5.
        m.value_context[0].bias[0] = .7
        for module in (m.value_context, m.value_head):
            if module is m.value_head:
                module[0].weight[0, 0] = 1.
            module[2].weight[0, 0] = 1.
            module[4].weight[0, 0] = 1.
    predictions = []
    fast_capacities = []
    for ages in ((3, 3, 0, 0), (0, 3, 3, 0)):
        env = urgent()
        env.env.state = replace(env.env.state, ages=ages)
        fast = []
        for trace in itertools.product(range(5), repeat=2):
            try:
                pairs, _ = env.resolve(np.array(trace))
            except ValueError:
                continue
            if all(env.config.service_time[a][j] == 1 for a, j in pairs):
                fast.append(len(pairs))
        fast_capacities.append(max(fast))
        predictions.append(m.evaluate(tensors([env.observe()]), torch.tensor([[2, 0]]))[2])
    assert fast_capacities == [1, 2]
    assert (predictions[0] - predictions[1]).abs().min() > 1e-8


def test_critic_context_contains_known_decoding_roles_before_choices():
    env = urgent()
    m = model('completion_credit_ppo')
    captured = []
    handle = m.value_context.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone()))
    try:
        m.evaluate(tensors([env.observe()]), torch.tensor([[0, 2]]))
    finally:
        handle.remove()
    assert captured[0][0, :, -2:].tolist() == [[0., 1.], [.5, 1.]]
    assert captured[1][0, :, -2:].tolist() == [[0., 0.], [.5, 1.]]


def test_nonlinear_fusion_can_distinguish_reserved_failed_vs_healthy_job():
    env = urgent()
    env.env.state = replace(env.env.state, ages=(4, 4, 0, 0),
        failed=(True, False, False, False), pending_wait=(4, 0, 0, 0))
    m = model('completion_credit_ppo')
    # Capacity witness only, not a trained/saved model. Reserving either job has
    # identical duration/restored-age descriptors, but different current failure.
    with torch.no_grad():
        for parameter in m.gradient_groups()['critic']:
            parameter.zero_()
        m.value_encoder[0].weight[0, 1] = 1.
        m.value_calendar_encoder[0].weight[0, 3] = 1.
        for module in (m.value_encoder, m.value_calendar_encoder, m.value_context, m.value_head):
            if module in (m.value_context, m.value_head):
                module[0].weight[0, 0] = 1.
            module[2].weight[0, 0] = 1.
            module[4].weight[0, 0] = 1.
    batch = tensors([env.observe()])
    failed = m.evaluate(batch, torch.tensor([[1, 0]]))[2]
    healthy = m.evaluate(batch, torch.tensor([[2, 0]]))[2]
    assert torch.equal(failed[:, 0], healthy[:, 0])
    assert (failed[:, 1] - healthy[:, 1]).abs().min() > 1e-5
