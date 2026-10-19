"""Read-only reconstruction of the skill screen, including tie diagnostics."""
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RUN = PROJECT / 'lab_results/maintenance_skill_contention_screen_v1_full_cpu_20261012T031404Z'
CODE = Path('/private/tmp/marl-maintenance-assignment-confidence-v1')
COMMIT = '8047eeb104dc9b9c3e833ae98b554e13d90a96b3'
sys.path.insert(0, str(CODE / 'src'))
from ht_pdm_fjsp import maintenance_skill_contention_screen as s
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv
from ht_pdm_fjsp.maintenance_dispatch import eligible_edges, role_seed
from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue


def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_rows(name):
    with (RUN / name).open() as stream:
        rows = list(csv.DictReader(stream))
    text = {'phase', 'candidate', 'policy'}
    ints = {'seed', 'physical_seed'}
    return [{k: (v if k in text else int(v) if k in ints else float(v)) for k, v in row.items()} for row in rows]


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for k in a: same(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert np.isclose(a, b, atol=1e-10, rtol=0), (a, b)
    else: assert a == b, (a, b)


manifest = json.loads((RUN / 'manifest.json').read_text())
assert manifest['status'] == 'COMPLETED' and manifest['audit'] == 'PASS'
assert manifest['git']['commit'] == COMMIT and not manifest['git']['dirty']
for name, record in manifest['files'].items():
    data = (RUN / name).read_bytes()
    assert len(data) == record['bytes']
    assert hashlib.sha256(data).hexdigest() == record['sha256']
bound_sources = {}
for p in (RUN / 'source_snapshot').glob('*.py'):
    committed = subprocess.check_output(['git', 'show', f'{COMMIT}:src/ht_pdm_fjsp/{p.name}'], cwd=CODE)
    assert p.read_bytes() == committed == (CODE / 'src/ht_pdm_fjsp' / p.name).read_bytes()
    bound_sources[p.name] = hashlib.sha256(committed).hexdigest()
for name, prefix in [('protocol.md', 'docs/maintenance_skill_contention_screen_v1_plan.md'),
                     ('seed_registry.json', 'configs/maintenance_skill_contention_screen_v1_seed_registry.json')]:
    assert (RUN / name).read_bytes() == subprocess.check_output(['git', 'show', f'{COMMIT}:{prefix}'], cwd=CODE)
raw_config = json.loads((RUN / 'resolved_config.json').read_text())
cfg, specs = raw_config['settings'], raw_config['candidates']
same(cfg, s.settings('full'))
same(specs, s.candidates(cfg))
old_summary = json.loads((RUN / 'summary.json').read_text())
selection = json.loads((RUN / 'selection.json').read_text())
assert hashlib.sha256((RUN / 'selection.json').read_bytes()).hexdigest() == manifest['selection_sha256']
same(selection, old_summary['selection'])
seeds = s.seed_audit(cfg)
same(seeds, json.loads((RUN / 'audit.json').read_text())['seed_audit'])
rows = read_rows('screening_episodes.csv') + read_rows('confirmation_episodes.csv')
for phase, chosen in [('screening', specs), ('confirmation', [x for x in specs if x['candidate'] in selection['confirmation_candidates']])]:
    group = [r for r in rows if r['phase'] == phase]
    expected = set(itertools.product([x['candidate'] for x in chosen], s.POLICIES, cfg[f'{phase}_seeds']))
    keys = [(r['candidate'], r['policy'], r['seed']) for r in group]
    assert len(keys) == len(expected) and set(keys) == expected
    assert len(group) == manifest['actual_counts'][phase] == manifest['expected_counts'][phase]
same(s.aggregate([r for r in rows if r['phase'] == 'confirmation']),
     {(r['candidate'], r['policy']): {k: v for k, v in r.items() if k not in ('candidate', 'policy')} for r in old_summary['confirmation_means']})

# Rebuild selection from seed means independently of the runner's rank function.
grouped = defaultdict(list)
for r in rows: grouped[r['phase'], r['candidate'], r['policy']].append(r)
screen_rank = []
for spec in specs:
    policy_means = [{key: sum(r[key] for r in grouped['screening', spec['candidate'], p]) / len(cfg['screening_seeds'])
        for key in ('conflict_rate', 'sensitive_choice_rate', 'availability', 'terminal_pending', 'extended_cost')} for p in s.POLICIES]
    worst = {key: min(v[key] for v in policy_means) for key in ('conflict_rate', 'sensitive_choice_rate', 'availability')}
    worst['terminal_pending'] = max(v['terminal_pending'] for v in policy_means)
    failures = [key for key, cutoff in cfg['limits'].items()
        if (worst[key] > cutoff if key == 'terminal_pending' else worst[key] < cutoff)]
    screen_rank.append(dict(candidate=spec['candidate'], topology=spec['topology'], pattern=spec['pattern'],
        selectable=spec['selectable'], eligible=spec['selectable'] and not failures,
        rejection=';'.join(failures) if spec['selectable'] else 'control',
        score=worst['conflict_rate'] * worst['sensitive_choice_rate'], **worst,
        mean_cost=sum(v['extended_cost'] for v in policy_means) / 2))
screen_rank.sort(key=lambda r: (-int(r['eligible']), -r['score'], r['candidate']))
same(screen_rank, old_summary['screening'])
eligible = [r for r in screen_rank if r['eligible']]
assert eligible[0]['candidate'] == selection['winner'] and eligible[1]['candidate'] == selection['runner_up']
assert selection['strongest_contention'] == min(screen_rank, key=lambda r: (-r['conflict_rate'], r['candidate']))['candidate']
assert selection['confirmation_candidates'] == [eligible[0]['candidate'], eligible[1]['candidate'],
    'overlap3__equal_control', 'partitioned_control__' + eligible[0]['pattern']]

# All recorded decision paths are separately reconstructed from stored seeds.
logged = s.replay_audit(RUN / 'decision_audit.jsonl', specs, cfg)
assert logged['replayed_transitions'] == 6960 and logged['audited_episodes'] == 116
spec_lookup = {r['candidate']: r for r in specs}
diagnostics, transitions = [], 0
for row in tqdm(rows, desc='reconstruct episodes and tie effects', unit='episode'):
    c = s.decode_config(spec_lookup[row['candidate']]); env = WaitingEnv(c)
    assert row['physical_seed'] == role_seed(row['seed'], f'{s.VERSION}:physical')
    env.reset(row['physical_seed'])
    base, extra, costs = Counter(), Counter(), Counter()
    terminal = None
    for tick in range(c.horizon):
        state = env.state
        occupied = set(state.assigned)
        demand = [m for m in range(c.machines) if m not in occupied and (state.failed[m] or state.ages[m] >= c.failure_age - 1)]
        edges = eligible_edges(c, state)
        actions = [a for a in catalogue(c.machines, c.technicians) if all(m in demand and edges[m][j] for m, j in a)]
        policy = row['policy'] if tick < cfg['horizon'] else 'deadline_matching'
        action = s.choose(c, state, policy)
        if tick < cfg['horizon']:
            fingerprints = defaultdict(set)
            for a in actions:
                fingerprints[tuple(m for m, j in a)].add(tuple((m, c.service_time[m][j], c.restoration[m][j]) for m, j in a))
            sensitive = {subset for subset, fingerprints in fingerprints.items() if len(fingerprints) > 1}
            largest = max(map(len, actions)); free = sum(r == 0 for r in state.remaining)
            best_sets, proposals = {}, []
            for m in demand:
                available = [j for j in range(c.technicians) if edges[m][j]]
                if available:
                    q = {j: c.restoration[m][j] / c.service_time[m][j] for j in available}
                    best = max(q.values())
                    best_sets[m] = {j for j in available if abs(q[j] - best) < 1e-12}
                    proposals.append(max(available, key=q.get))
            excess = sum(max(0, n - 1) for n in Counter(proposals).values())
            pref_largest = max(len(a) for a in actions if all(j in best_sets[m] for m, j in a))
            base.update(dict(conflict_rate=int(excess > 0), collision_excess=excess,
                sensitive_choice_rate=int(any(len(subset) == largest for subset in sensitive)),
                selected_sensitive_rate=int(tuple(m for m, _ in action) in sensitive),
                capacity_pressure_rate=int(len(demand) > free), skill_deficit_rate=int(largest < min(len(demand), free)), demand_count=len(demand)))
            extra.update(dict(tied_preference_rate=int(any(len(js) > 1 for js in best_sets.values())),
                tie_resolvable_conflict_rate=int(excess > 0 and pref_largest == len(proposals)),
                unavoidable_best_set_conflict_rate=int(pref_largest < len(proposals)),
                strict_preference_deficit_rate=int(pref_largest < largest)))
        _, _, _, info = env.step(action)
        transitions += 1
        if tick < cfg['horizon']: costs.update(info)
        if tick + 1 == cfg['horizon']:
            terminal = dict(terminal_pending=sum(env.state.failed[m] and m not in env.state.assigned for m in range(c.machines)),
                terminal_failed=sum(env.state.failed))
    reconstructed = dict(**{k: v / cfg['horizon'] for k, v in base.items()},
        availability=1 - costs['unavailable_ticks'] / (c.machines * cfg['horizon']),
        technician_utilization=costs['service_ticks'] / (c.technicians * cfg['horizon']), **terminal,
        tail_pending=sum(env.state.failed[m] and m not in env.state.assigned for m in range(c.machines)), tail_failed=sum(env.state.failed),
        policy_cost=costs['objective'], tail_cost=env.metrics['objective'] - costs['objective'], extended_cost=env.metrics['objective'],
        policy_failures=costs['failures'], policy_overdue_ticks=costs['overdue_waiting_ticks'],
        **{f'extended_{k}': env.metrics[k] for k in ('maintenance_cost', 'unavailability_cost', 'failure_cost', 'waiting_cost', 'invalid_assignments')})
    for key, value in reconstructed.items(): same(value, row[key])
    assert row['extended_invalid_assignments'] == 0
    assert abs(reconstructed['extended_cost'] - sum(reconstructed['extended_' + k] for k in
        ('maintenance_cost', 'unavailability_cost', 'failure_cost', 'waiting_cost'))) < 1e-8
    diag = dict(phase=row['phase'], candidate=row['candidate'], policy=row['policy'], seed=row['seed'],
        **{k: v / cfg['horizon'] for k, v in extra.items()})
    same(diag['tie_resolvable_conflict_rate'] + diag['unavoidable_best_set_conflict_rate'], row['conflict_rate'])
    diagnostics.append(diag)
assert transitions == 253440
s.write_csv(HERE / 'tie_diagnostics_episodes.csv', diagnostics)
diag_groups = defaultdict(list)
for r in diagnostics: diag_groups[r['phase'], r['candidate'], r['policy']].append(r)
diag_means = []
for key, group in diag_groups.items():
    fields = [k for k in group[0] if k not in ('phase', 'candidate', 'policy', 'seed')]
    record = dict(phase=key[0], candidate=key[1], policy=key[2], **{k: float(np.mean([r[k] for r in group])) for k in fields})
    source_group = grouped[key]
    record['original_conflict_rate'] = float(np.mean([r['conflict_rate'] for r in source_group]))
    record['tie_resolvable_fraction_of_reported_conflicts'] = record['tie_resolvable_conflict_rate'] / record['original_conflict_rate'] if record['original_conflict_rate'] else 0.
    diag_means.append(record)
s.write_csv(HERE / 'tie_diagnostics_means.csv', diag_means)

# A second bootstrap implementation uses counts/einsum instead of indexed arrays.
winner = selection['winner']; metrics = tuple(cfg['limits']); n = len(cfg['confirmation_seeds'])
lookup = {(r['candidate'], r['policy'], r['seed']): r for r in rows if r['phase'] == 'confirmation'}
values = np.asarray([[[lookup[winner, p, seed][k] for k in metrics] for p in s.POLICIES] for seed in cfg['confirmation_seeds']])
rng = np.random.default_rng(role_seed(cfg['analysis_seed'], f'{s.VERSION}:bootstrap'))
indices = rng.integers(0, n, size=(cfg['bootstrap_samples'], n))
weights = np.zeros((cfg['bootstrap_samples'], n), dtype=np.float64)
np.add.at(weights, (np.repeat(np.arange(len(indices)), n), indices.ravel()), 1 / n)
boot = np.einsum('bn,npm->bpm', weights, values, optimize=True)
alpha = (1 - cfg['confidence']) / 2
confirmed = {}
for j, key in enumerate(metrics):
    distribution = boot[:, :, j].max(1) if key == 'terminal_pending' else boot[:, :, j].min(1)
    lo, hi = np.quantile(distribution, [alpha, 1 - alpha])
    point = values.mean(0)[:, j].max() if key == 'terminal_pending' else values.mean(0)[:, j].min()
    passed = hi <= cfg['limits'][key] if key == 'terminal_pending' else lo >= cfg['limits'][key]
    confirmed[key] = dict(estimate=float(point), ci=[float(lo), float(hi)], pass_gate=bool(passed))
same(confirmed, old_summary['confirmation']['metrics'])
assert all(r['pass_gate'] for r in confirmed.values())

cost_comparisons = []
for candidate in selection['confirmation_candidates']:
    a = np.asarray([lookup[candidate, 'priority_greedy', seed]['extended_cost'] for seed in cfg['confirmation_seeds']])
    b = np.asarray([lookup[candidate, 'deadline_matching', seed]['extended_cost'] for seed in cfg['confirmation_seeds']])
    gain = (a - b).mean() / a.mean()
    draws = (weights @ (a - b)) / (weights @ a)
    ci = np.quantile(draws, [.025, .975])
    cost_comparisons.append(dict(candidate=candidate, reduction_vs_priority_greedy=float(gain),
        ci95=list(map(float, ci)), mean_cost_deadline=float(b.mean()), mean_cost_greedy=float(a.mean()),
        status='posthoc_exploratory; not pure assignment intervention or evidence of RL gain'))

results = dict(run=str(RUN), manifest=manifest, validation=dict(status='PASS', hashed_artifacts=len(manifest['files']),
    bytes=manifest['artifact_bytes'], source_bindings=bound_sources, exact_episode_grids=True, scalar_logged_replay=logged,
    reconstructed_episodes=len(rows), reconstructed_physical_transitions=transitions, all_episode_metrics_match=True,
    selection_reconstructed=True, independent_bootstrap_intervals_match=True, seeds=seeds['status']),
    selection=selection, confirmation=old_summary['confirmation'], screening_rank=screen_rank,
    confirmation_means=old_summary['confirmation_means'], tie_diagnostics=diag_means, cost_comparisons=cost_comparisons,
    selected_setting=spec_lookup[winner], eligible_settings=len(eligible))
dump('analysis.json', results)
print(json.dumps(results['validation'], indent=2))
print('winner diagnostics:', json.dumps([r for r in diag_means if r['phase'] == 'confirmation' and r['candidate'] == winner], indent=2))
print('cost contrasts:', json.dumps(cost_comparisons, indent=2))
