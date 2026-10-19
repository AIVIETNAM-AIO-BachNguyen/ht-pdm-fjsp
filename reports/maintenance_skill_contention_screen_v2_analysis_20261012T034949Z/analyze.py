"""Read-only v2 artifact validation, replay and exploratory mechanism diagnostics."""
from collections import defaultdict
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
RUN = PROJECT / 'lab_results/maintenance_skill_contention_screen_v2_full_cpu_20261012T034652Z'
CODE = Path('/private/tmp/marl-maintenance-assignment-confidence-v1')
COMMIT = '912ee749a556006e0164f7734450ccba9ef45bcb'
sys.path.insert(0, str(CODE / 'src'))
from ht_pdm_fjsp import maintenance_skill_contention_screen_v2 as s
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv
from ht_pdm_fjsp.maintenance_dispatch import role_seed


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for k in a: same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert np.isclose(a, b, atol=1e-10, rtol=0), (a, b)
    else: assert a == b, (a, b)


def dump(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_rows(name):
    with (RUN / name).open() as stream:
        rows = list(csv.DictReader(stream))
    text = {'phase', 'candidate', 'policy'}
    integers = {'seed', 'physical_seed'}
    return [{k: v if k in text else int(v) if k in integers else float(v) for k, v in r.items()} for r in rows]


manifest = json.loads((RUN / 'manifest.json').read_text())
assert manifest['status'] == 'COMPLETED' and manifest['audit'] == 'PASS'
assert manifest['git']['commit'] == COMMIT and not manifest['git']['dirty']
for name, record in manifest['files'].items():
    data = (RUN / name).read_bytes()
    assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
bindings = {}
for path in (RUN / 'source_snapshot').glob('*.py'):
    committed = subprocess.check_output(['git', 'show', f'{COMMIT}:src/ht_pdm_fjsp/{path.name}'], cwd=CODE)
    assert path.read_bytes() == committed == (CODE / 'src/ht_pdm_fjsp' / path.name).read_bytes()
    bindings[path.name] = hashlib.sha256(committed).hexdigest()
for name, original in [('protocol.md', f'docs/{s.VERSION}_plan.md'),
                       ('seed_registry.json', f'configs/{s.VERSION}_seed_registry.json')]:
    assert (RUN / name).read_bytes() == subprocess.check_output(['git', 'show', f'{COMMIT}:{original}'], cwd=CODE)
raw_cfg = json.loads((RUN / 'resolved_config.json').read_text())
cfg, specs = raw_cfg['settings'], raw_cfg['candidates']
same(cfg, s.settings('full')); same(specs, s.candidates(cfg))
summary = json.loads((RUN / 'summary.json').read_text())
selection = json.loads((RUN / 'selection.json').read_text())
same(selection, summary['selection'])
assert hashlib.sha256((RUN / 'selection.json').read_bytes()).hexdigest() == manifest['selection_sha256']
old_audit = json.loads((RUN / 'audit.json').read_text())
seeds = s.seed_audit(cfg); same(seeds, old_audit['seed_audit'])
screen = read_rows('screening_episodes.csv'); confirm = read_rows('confirmation_episodes.csv')
rows = screen + confirm
for phase, group, chosen in [('screening', screen, specs), ('confirmation', confirm,
        [r for r in specs if r['candidate'] in selection['confirmation_candidates']])]:
    expected = set(itertools.product([r['candidate'] for r in chosen], s.POLICIES, cfg[f'{phase}_seeds']))
    actual = [(r['candidate'], r['policy'], r['seed']) for r in group]
    assert len(actual) == len(expected) and set(actual) == expected
    assert all(r['physical_seed'] == role_seed(r['seed'], f'{s.VERSION}:physical') for r in group)
controls = s.control_audit(rows); same(controls, old_audit['control_audit'])
logged = s.replay_audit(RUN / 'decision_audit.jsonl', specs, cfg)
assert logged['replayed_transitions'] == 19200 and logged['audited_episodes'] == 320

# Independently compute ranking from stored episode means.
means = s.aggregate(screen); ranking = []
for spec in specs:
    group = [means[spec['candidate'], p] for p in s.POLICIES]
    metrics = {k: (max if k == 'terminal_pending' else min)(g[k] for g in group) for k in cfg['limits']}
    failures = [k for k, limit in cfg['limits'].items() if (metrics[k] > limit if k == 'terminal_pending' else metrics[k] < limit)]
    ranking.append(dict(candidate=spec['candidate'], topology=spec['topology'], pattern=spec['pattern'],
        selectable=spec['selectable'], eligible=spec['selectable'] and not failures,
        rejection=';'.join(failures) if spec['selectable'] else 'control', **metrics,
        mean_cost=float(np.mean([g['extended_cost'] for g in group]))))
ranking.sort(key=lambda r: (-int(r['eligible']), -r['skill_contention_rate'], -r['contested_choice_rate'], r['candidate']))
same(ranking, summary['screening'])
eligible = [r for r in ranking if r['eligible']]
assert selection['winner'] == eligible[0]['candidate'] and selection['runner_up'] == eligible[1]['candidate']
assert selection['confirmation_candidates'] == [selection['winner'], selection['runner_up'], *s.CONTROLS]
confirmation_means = s.aggregate(confirm)
same([dict(candidate=k[0], policy=k[1], **v) for k, v in confirmation_means.items()], summary['confirmation_means'])

# Independent metric oracle: enumerate assignments by worker recursion, without
# the runner's catalogue or preferred-edge/matching helpers.
def oracle(c, state):
    demand = [m for m in range(c.machines) if m not in state.assigned and (state.failed[m] or state.ages[m] >= 3)]
    free = [j for j in range(c.technicians) if state.remaining[j] == 0]
    valid = {m: [j for j in free if c.service_time[m][j] > 0] for m in demand}
    prefs = {}
    for m, js in valid.items():
        prefs[m] = {j for j in js if not any(
            c.service_time[m][k] <= c.service_time[m][j] and c.restoration[m][k] >= c.restoration[m][j]
            and (c.service_time[m][k] < c.service_time[m][j] or c.restoration[m][k] > c.restoration[m][j]) for k in js)}
    actions = []
    def visit(i, chosen):
        if i == len(free):
            actions.append(tuple(sorted(chosen))); return
        j = free[i]; visit(i + 1, chosen)
        for m in demand:
            if j in valid[m] and all(m != old for old, _ in chosen): visit(i + 1, chosen + [(m, j)])
    visit(0, [])
    best = [a for a in actions if all(j in prefs[m] for m, j in a)]
    f, p = max(map(len, actions)), max(map(len, best))
    grouped = defaultdict(set)
    for a in actions:
        grouped[tuple(m for m, _ in a)].add(tuple((m, c.service_time[m][j], c.restoration[m][j]) for m, j in a))
    preferred_subsets = {tuple(m for m, _ in a) for a in best if len(a) == f}
    joint = p < f and any(len(subset) == f and len(values) > 1 and subset not in preferred_subsets
                         for subset, values in grouped.items())
    return dict(skill_contention_rate=int(p < f), contested_choice_rate=int(joint),
        feasible_matching_cardinality=f, preferred_matching_cardinality=p, preferred_deficit=f - p)

# Reconstruct every stored episode. The oracle checks every measured state while
# original metrics, costs and terminal states are reconciled with episode CSVs.
original_metrics = s.state_metrics
oracle_ticks = 0
winner_diagnostics = []
active = None
def checked_metrics(c, state, selected, space=None):
    global oracle_ticks
    actual = original_metrics(c, state, selected, space)
    independent = oracle(c, state)
    same(independent, {k: actual[k] for k in independent}); oracle_ticks += 1
    if active['phase'] == 'confirmation' and active['candidate'] == selection['winner']:
        # Physical alternatives need not imply different expected economic value.
        # Count whether selected machines have distinct observable job states.
        subset = tuple(m for m, _ in selected)
        signatures = {(state.ages[m], state.failed[m], state.pending_wait[m]) for m in subset}
        selected_contested = actual['skill_contention_rate'] and actual['selected_sensitive_rate']
        winner_diagnostics.append(dict(policy=active['policy'], seed=active['seed'],
            strict=actual['skill_contention_rate'], joint=actual['contested_choice_rate'],
            selected_sensitive=actual['selected_sensitive_rate'], selected_contested=int(selected_contested),
            selected_contested_heterogeneous=int(selected_contested and len(signatures) > 1),
            selected_contested_symmetric=int(selected_contested and len(subset) > 1 and len(signatures) == 1),
            urgent_machines=actual['demand_count']))
    return actual
s.state_metrics = checked_metrics
spec_lookup = {r['candidate']: r for r in specs}
for row in tqdm(rows, desc='Reconstruct stored episodes', unit='episode'):
    active = row
    computed = s.one_episode(cfg, spec_lookup[row['candidate']], row['policy'], row['seed'], row['phase'])
    same(computed, row)
s.state_metrics = original_metrics
assert oracle_ticks == 10880 * 48

# Counts/einsum bootstrap differs from the runner's indexed-array implementation.
lookup = {(r['candidate'], r['policy'], r['seed']): r for r in confirm}
metrics = tuple(cfg['limits']); seeds_panel = cfg['confirmation_seeds']; n = len(seeds_panel)
winner = selection['winner']
values = np.asarray([[[lookup[winner, p, seed][k] for k in metrics] for p in s.POLICIES] for seed in seeds_panel])
rng = np.random.default_rng(role_seed(cfg['analysis_seed'], f'{s.VERSION}:bootstrap'))
indices = rng.integers(0, n, size=(cfg['bootstrap_samples'], n))
weights = np.zeros((len(indices), n))
np.add.at(weights, (np.repeat(np.arange(len(indices)), n), indices.ravel()), 1 / n)
boot = np.einsum('bn,npm->bpm', weights, values, optimize=True)
alpha = (1 - cfg['confidence']) / 2
confirmation = {}
for j, key in enumerate(metrics):
    upper = key == 'terminal_pending'
    dist = boot[:, :, j].max(1) if upper else boot[:, :, j].min(1)
    point = values.mean(0)[:, j].max() if upper else values.mean(0)[:, j].min()
    lo, hi = np.quantile(dist, [alpha, 1 - alpha])
    passed = hi <= cfg['limits'][key] if upper else lo >= cfg['limits'][key]
    confirmation[key] = dict(estimate=float(point), ci=[float(lo), float(hi)], pass_gate=bool(passed))
same(confirmation, summary['confirmation']['metrics'])
assert all(r['pass_gate'] for r in confirmation.values())

paired_identical = {}
costs = []
for candidate in selection['confirmation_candidates']:
    excluded = {'phase', 'candidate', 'policy', 'seed', 'physical_seed'}
    paired_identical[candidate] = all(all(lookup[candidate, s.POLICIES[0], seed][k] == lookup[candidate, s.POLICIES[1], seed][k]
        for k in lookup[candidate, s.POLICIES[0], seed] if k not in excluded) for seed in seeds_panel)
    a = np.asarray([lookup[candidate, 'priority_greedy', seed]['extended_cost'] for seed in seeds_panel])
    b = np.asarray([lookup[candidate, 'deadline_matching', seed]['extended_cost'] for seed in seeds_panel])
    ci = np.quantile((weights @ (a - b)) / (weights @ a), [.025, .975])
    costs.append(dict(candidate=candidate, cost_matching=float(b.mean()), cost_greedy=float(a.mean()),
        reduction_vs_greedy=float((a - b).mean() / a.mean()), ci95=ci.tolist(), inference='exploratory rule comparison, not RL benefit or isolated assignment effect'))

# Replay both winner rules together to test exact actions and full-state equality,
# rather than equating paths from matching aggregate metrics.
c = s.decode_config(spec_lookup[winner]); identical_transitions = 0
for seed in tqdm(seeds_panel, desc='Compare winner rule paths', unit='seed'):
    envs = [WaitingEnv(c), WaitingEnv(c)]
    for env in envs: env.reset(role_seed(seed, f'{s.VERSION}:physical'))
    for tick in range(c.horizon):
        assert envs[0].state == envs[1].state and envs[0].events[tick] == envs[1].events[tick]
        actions = [s.choose(c, env.state, policy if tick < cfg['horizon'] else 'deadline_matching') for env, policy in zip(envs, s.POLICIES)]
        assert actions[0] == actions[1]
        for env, action in zip(envs, actions): env.step(action)
        assert envs[0].state == envs[1].state
        identical_transitions += 1

diagnostics = []
for policy in s.POLICIES:
    group = [r for r in winner_diagnostics if r['policy'] == policy]
    counts = {k: sum(r[k] for r in group) for k in ('strict', 'joint', 'selected_sensitive', 'selected_contested',
        'selected_contested_heterogeneous', 'selected_contested_symmetric')}
    diagnostics.append(dict(policy=policy, measured_ticks=len(group), **counts,
        heterogeneous_fraction_of_selected_contested=counts['selected_contested_heterogeneous'] / counts['selected_contested']))
result = dict(run=str(RUN), manifest=manifest, validation=dict(status='PASS', hashed_files=len(manifest['files']),
    source_bindings=bindings, exact_episode_grids=True, seeds=seeds['status'], controls=controls,
    logged_replay=logged, reconstructed_episodes=len(rows), physical_transitions=len(rows) * 60,
    independent_metric_oracle_ticks=oracle_ticks, all_episode_metrics_match=True,
    selection_reconstructed=True, independent_confirmation_CIs_match=True,
    winner_identical_rule_paths_verified_seeds=n, winner_identical_rule_path_transitions=identical_transitions),
    selection=selection, ranking=ranking, eligible_count=len(eligible), confirmation=summary['confirmation'],
    confirmation_means=summary['confirmation_means'], selected_setting=spec_lookup[winner],
    same_paired_episode_metrics=paired_identical, costs=costs,
    winner_posthoc_state_symmetry_diagnostics=diagnostics)
dump('analysis.json', result)
print(json.dumps(result['validation'], indent=2))
print('symmetry diagnostic:', json.dumps(diagnostics, indent=2))
print('rule cost contrasts:', json.dumps(costs, indent=2))
