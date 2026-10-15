import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pytest
import torch

from ht_pdm_fjsp import technician_exchange_experiment as x


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def panel(scales):
    return {n: {a: np.array([100. if a == 'exchange_ppo' else 110., .7, 0., 0.])
                for a in (*x.arms(n), 'deadline')} for n in scales}


def test_prospective_roles_counts_and_original_ppo_preserved():
    audit = x.seed_audit()
    assert audit['status'] == 'PASS' and audit['raw_unique'] == 387
    c = x.settings('confirmation')
    assert c['confirmation_round'] == 2 and c['confidence'] == 1-(.05/6)/43
    assert c['bootstrap_samples'] == 200000 and not c['teacher_assistance'] and not c['pretraining_steps']
    assert sum(len(x.arms(n)) for n in c['scales'])*len(c['train_seeds']) == 120
    originals = [x.make_model(a, 4, c, 120000000) for a in ('ppo', 'normalized_ppo')]
    assert all(torch.equal(a, b) for a, b in zip(originals[0].parameters(), originals[1].parameters()))
    with torch.random.fork_rng():
        torch.manual_seed(x.learner.role_seed(120000000, x.VERSION+':initialization:ppo'))
        raw = x.learner.Model('ppo', c['baseline_hidden'])
    assert x.learner.parameter_digest(raw) == x.learner.parameter_digest(originals[0])
    assert x.rng_role('ppo') == x.rng_role('normalized_ppo') == 'ppo'
    assert x.rng_role('completion_reference_ppo') != x.rng_role('exchange_ppo')


def test_factory_core_capacity_and_reference_is_fresh_v2_not_teacher():
    c = x.settings('smoke')
    models = [x.make_model(a, 4, c, 120000000) for a in x.CORE_ARMS]
    assert len({x.learner.parameter_digest(m) for m in models}) == 1
    assert {sum(p.numel() for p in m.parameters()) for m in models} == {19255}
    reference = x.make_model('completion_reference_ppo', 4, c, 120000000)
    assert reference.algorithm == 'completion_reference_ppo' and reference.allocation_learning_mode == 'prefix'
    assert sum(p.numel() for p in reference.parameters()) == 18444
    # All graph journals share a schema, including the external v2 reference.
    returns = torch.tensor([-10., -20.])
    assert set(models[0].return_diagnostics(returns, torch.ones(2, 4))) == set(reference.return_diagnostics(returns, torch.ones(2, 2)))


def test_43_endpoint_family_and_both_small_health_references_are_required():
    scales = [4, 8, 16, 32]
    means = panel(scales)
    points = x.estimates(means, scales)
    assert len(points) == 43
    checks = x.guards({k: dict(lower=v, upper=v) for k, v in points.items()}, scales,
                      {f'{n}:exchange_ppo': {'p95_ms': 1} for n in scales})
    assert all(checks.values())
    means[4]['exchange_ppo'][3] = means[4]['ppo'][3] = .4
    points = x.estimates(means, scales)
    checks = x.guards({k: dict(lower=v, upper=v) for k, v in points.items()}, scales,
                      {f'{n}:exchange_ppo': {'p95_ms': 1} for n in scales})
    assert checks['N4_health_vs_ppo'] and not checks['N4_health_vs_normalized_ppo']
    assert x.point_checks(points)['N4_failures_per_machine_difference_vs_ppo']
    assert not x.point_checks(points)['N4_failures_per_machine_difference_vs_normalized_ppo']


def test_bootstrap_null_missing_grid_and_inside_draw_pool():
    c = dict(x.settings('smoke'), scales=[4, 8, 16, 32], train_seeds=[1, 2], test_seeds=[3, 4], bootstrap_samples=200)
    rows = [dict(n=n, method=a, train_seed=t, seed=s, normalized_cost=100.,
                 availability=.7, normalized_pending=0., normalized_failures=0.)
            for n in c['scales'] for a, t in [(a, t) for a in x.arms(n) for t in c['train_seeds']]+[('deadline', -1)] for s in c['test_seeds']]
    latencies = {f'{n}:{a}': [1., 2.] for n in c['scales'] for a in (*x.arms(n), 'deadline')}
    summary = x.summarize(rows, c, latencies)
    assert summary['family_size'] == 43 and not summary['goal_pass']
    assert not summary['guards']['exchange_mechanism'] and not summary['guards']['repair_mechanism']
    assert summary['intervals']['pooled_exchange_gain']['estimate'] == 0
    with pytest.raises(RuntimeError, match='test grid'): x.summarize(rows[:-1], c, latencies)
    means = panel(c['scales'])
    means[8]['self_repair_ppo'][0] = 125.
    means[16]['self_repair_ppo'][0] = 100.
    means[32]['self_repair_ppo'][0] = 80.
    assert x.estimates(means, c['scales'])['pooled_exchange_gain'] == pytest.approx((.2+0.-.25)/3)


def test_passing_engineering_guards_cannot_claim_scientific_goal_completion():
    c = dict(x.settings('smoke'), scales=[4, 8, 16, 32], train_seeds=[1, 2], test_seeds=[3, 4], bootstrap_samples=200)
    rows = [dict(n=n, method=a, train_seed=t, seed=s,
                 normalized_cost=100. if a == 'exchange_ppo' else 110.,
                 availability=.7, normalized_pending=0., normalized_failures=0.)
            for n in c['scales'] for a, t in [(a, t) for a in x.arms(n) for t in c['train_seeds']]+[('deadline', -1)] for s in c['test_seeds']]
    latencies = {f'{n}:{a}': [1., 2.] for n in c['scales'] for a in (*x.arms(n), 'deadline')}
    s = x.summarize(rows, c, latencies)
    assert s['classification'] == 'ENGINEERING_ONLY' and s['engineering_guard_pass']
    assert not s['goal_pass']


@pytest.mark.parametrize('arm', (*x.CORE_ARMS, 'completion_reference_ppo', 'ppo', 'normalized_ppo', 'mappo', 'full_mappo'))
def test_checkpoint_tags_and_scalar_or_trace_dispatch(tmp_path, arm):
    c = x.settings('smoke');m = x.make_model(arm, 4, c, 120000000)
    env = x.ResourceEnv(120100000, 4, arm);obs = env.reset()
    action = m.select(obs);env.resolve(action)
    assert isinstance(action, int) if arm in ('ppo', 'normalized_ppo') else len(action) == (4 if arm in x.CORE_ARMS or arm in ('mappo', 'full_mappo') else 2)
    path = tmp_path/'zero.pt';meta = x.save_model(m, path, 4, c, 120000000, 0, obs)
    assert meta['value_probe']['value_unit'] == (1. if arm == 'ppo' else x.value_unit(4))
    other = x.load_model(path, 4, c, 120000000, arm, 0)
    assert x.learner.parameter_digest(m) == x.learner.parameter_digest(other)
    with pytest.raises(RuntimeError, match='metadata'):
        x.load_model(path, 4, c, 120000000, 'self_repair_ppo' if arm != 'self_repair_ppo' else 'exchange_ppo', 0)


def test_serial_spawn_exact_scratch_identity(tmp_path):
    c = dict(x.settings('smoke'), development_seeds=[120100000])
    left, right = tmp_path/'serial', tmp_path/'spawn';left.mkdir();right.mkdir()
    args = (c, 4, 'exchange_ppo', 120000000)
    serial = x.fit_job((*args, str(left), time.monotonic()))
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn')) as pool:
        parallel = pool.submit(x.fit_job, (*args, str(right), time.monotonic())).result(timeout=120)
    assert serial['initial'] == parallel['initial']
    assert serial['choice']['sha256'] == parallel['choice']['sha256']
    assert serial['audit']['parameter_sha256'] == parallel['audit']['parameter_sha256']
    assert serial['audit']['learned_steps'] == 384 and serial['audit']['optimizer_steps'] == 16


def prerequisite_fixture(path):
    c = x.settings('pilot');means = panel(c['scales'])
    for other in ('self_repair_ppo', 'construct_only_ppo'): means[32][other][0] = 100.
    point = x.estimates(means, c['scales']);screen = x.point_checks(point)
    selected = {f'{n}:{a}:{t}': 30720 for n in c['scales'] for a in x.arms(n) for t in c['train_seeds']}
    summary = dict(classification='DEVELOPMENT_ONLY', teacher_assistance=False, test_opened=False,
        estimates=point, screen=screen, development_screen_pass=all(screen.values()), selected_steps=selected)
    audits = [dict(n=n, algorithm=a, train_seed=t, learned_steps=c['env_steps'], physical_ticks=c['env_steps']//48*60,
        training_episodes=c['env_steps']//48, optimizer_steps=c['env_steps']//c['rollout']*c['epochs']*(c['rollout']//c['minibatch']),
        from_scratch=True, pretraining_steps=0, changed=True, behavior_normalized_max_error=0.)
        for n in c['scales'] for a in x.arms(n) for t in c['train_seeds']]
    for name, data in [('resolved_config.json', c), ('summary.json', summary), ('audit.json', {'status': 'PASS'}),
                       ('training_audit.json', audits), ('checkpoint_choices.json', {k: {'steps': v} for k, v in selected.items()})]:
        (path/name).write_text(json.dumps(data))
    manifest = dict(protocol=x.VERSION, status='COMPLETED', audit_status='PASS', test_opened=False,
        source_seal={'source_snapshot/example.py': 'frozen'},
        output_bindings={p.name: {'sha256': x.sha(p)} for p in path.iterdir()})
    (path/'manifest.json').write_text(json.dumps(manifest))
    return c, summary, audits, manifest


def test_pilot_bound_budget_source_grid_and_proxy_rules(tmp_path):
    c, summary, audits, manifest = prerequisite_fixture(tmp_path)
    result = x.validate_pilot(tmp_path, {'source_snapshot/example.py': 'frozen'})
    assert result['status'] == 'PASS' and not any(result['mechanism_point_proxy_pass'].values())
    with pytest.raises(ValueError, match='source/protocol'): x.validate_pilot(tmp_path, {'source_snapshot/example.py': 'changed'})
    audits[0]['learned_steps'] -= 1
    (tmp_path/'training_audit.json').write_text(json.dumps(audits))
    with pytest.raises(ValueError, match='training artifact'): x.validate_pilot(tmp_path, {})
    manifest['output_bindings']['training_audit.json']['sha256'] = x.sha(tmp_path/'training_audit.json')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='budget grid'): x.validate_pilot(tmp_path, {})
    del summary['screen']['N4_gain_vs_normalized_ppo']
    with pytest.raises(ValueError, match='incomplete'): x.validate_summary_grid(summary, c)


def test_false_summary_flags_do_not_override_numeric_health_limits(tmp_path):
    c, summary, _, _ = prerequisite_fixture(tmp_path)
    summary['estimates']['N32_failures_per_machine_difference_vs_deadline'] = .4
    with pytest.raises(ValueError, match='thresholds'): x.validate_summary_grid(summary, c)
    summary['screen'] = x.point_checks(summary['estimates'])
    summary['development_screen_pass'] = all(summary['screen'].values())
    x.validate_summary_grid(summary, c)


def test_confirmation_requires_real_full_development(tmp_path):
    for name, data in [('resolved_config.json', {'profile': 'development'}), ('summary.json', {'development_screen_pass': False}),
        ('manifest.json', dict(protocol=x.VERSION, status='COMPLETED', audit_status='PASS', test_opened=False))]:
        (tmp_path/name).write_text(json.dumps(data))
    with pytest.raises(ValueError, match='development prerequisite'): x.validate_prerequisite(tmp_path, x.settings('confirmation'), {})
