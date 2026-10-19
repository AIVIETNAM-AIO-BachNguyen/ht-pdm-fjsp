"""Read-only v3 artifact validation, replay and exploratory mechanism diagnostics."""
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
RUN = PROJECT / 'lab_results/maintenance_skill_contention_screen_v3_full_cpu_20261012T041544Z'
CODE = Path('/private/tmp/marl-maintenance-assignment-confidence-v1')
COMMIT = '19fb84d5d63977b8486afb6dde742b8864dd72ab'
sys.path.insert(0, str(CODE / 'src'))
from ht_pdm_fjsp import maintenance_skill_contention_screen_v3 as s
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
assert logged['replayed_transitions'] == 13920 and logged['audited_episodes'] == 232

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
ranking.sort(key=lambda r: (-int(r['eligible']), -r['skill_contention_rate'], -r['cost_sensitive_contention_rate'], -r['selected_cost_sensitive_contention_rate'], r['candidate']))
same(ranking, summary['screening'])
eligible = [r for r in ranking if r['eligible']]
assert not eligible and selection['status'] == 'NO_ELIGIBLE_SETTING'
assert selection['winner'] is None and selection['runner_up'] is None
assert selection['confirmation_candidates'] == list(s.CONTROLS)
assert summary['confirmation'] == dict(status='NOT_APPLICABLE', reason='NO_ELIGIBLE_SETTING', winner=None)
confirmation_means = s.aggregate(confirm)
same([dict(candidate=k[0], policy=k[1], **v) for k, v in confirmation_means.items()], summary['confirmation_means'])


# Freeze the historical seed registry bindings too.
for path in (RUN / 'source_snapshot').glob('*seed_registry.json'):
    committed = subprocess.check_output(['git', 'show', f'{COMMIT}:configs/{path.name}'], cwd=CODE)
    assert path.read_bytes() == committed
    bindings[path.name] = hashlib.sha256(committed).hexdigest()

# Independent exhaustive worker recursion, including all legal partial matchings.
# Does not use the experiment's action enumeration, preference or cost-span helpers.
def oracle(c, state, selected):
    demand = [m for m in range(c.machines) if m not in state.assigned and (state.failed[m] or state.ages[m] >= 3)]
    free = [j for j in range(c.technicians) if state.remaining[j] == 0]
    valid = {m: [j for j in free if c.service_time[m][j] > 0] for m in demand}
    prefs = {m: {j for j in js if not any(
        c.service_time[m][k] <= c.service_time[m][j] and c.restoration[m][k] >= c.restoration[m][j]
        and (c.service_time[m][k] < c.service_time[m][j] or c.restoration[m][k] > c.restoration[m][j])
        for k in js)} for m, js in valid.items()}
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
    grouped = defaultdict(list)
    physical = defaultdict(set)
    for a in actions:
        subset = tuple(m for m, _ in a)
        grouped[subset].append(c.unavailable_cost * sum(c.service_time[m][j] for m, j in a))
        physical[subset].add(tuple((m, c.service_time[m][j], c.restoration[m][j]) for m,j in a))
    preferred_subsets = {tuple(m for m, _ in a) for a in best if len(a) == f}
    joint = p < f and any(len(subset) == f and len(values) > 1 and subset not in preferred_subsets
        for subset, values in physical.items())
    spans = {subset: max(v) - min(v) for subset,v in grouped.items()}
    possible = max(v for subset,v in spans.items() if len(subset) == f)
    subset = tuple(m for m, _ in selected)
    selected_span = spans[subset]
    threshold = max(c.unavailable_cost, 1e-12)
    metrics = dict(skill_contention_rate=int(p < f), contested_choice_rate=int(joint),
        feasible_matching_cardinality=f, preferred_matching_cardinality=p, preferred_deficit=f-p,
        cost_sensitive_contention_rate=int(p < f and possible >= threshold),
        selected_cost_sensitive_contention_rate=int(p < f and len(subset) == f and selected_span >= threshold),
        committed_cost_span_max=possible, selected_committed_cost_span=selected_span)
    return metrics, grouped

original_metrics = s.state_metrics
oracle_ticks = 0
active = None
mechanisms = defaultdict(lambda: defaultdict(float))
examples = {}
def checked_metrics(c, state, selected, space=None):
    global oracle_ticks
    actual = original_metrics(c, state, selected, space)
    independent, grouped = oracle(c, state, selected)
    same(independent, {k: actual[k] for k in independent}); oracle_ticks += 1
    key = (active['phase'], active['candidate'], active['policy'])
    counts = mechanisms[key]; counts['measured_ticks'] += 1
    for k in ('skill_contention_rate','cost_sensitive_contention_rate','selected_cost_sensitive_contention_rate'):
        counts[k+'_ticks'] += actual[k]
    subset = tuple(m for m,j in selected)
    regret = c.unavailable_cost * sum(c.service_time[m][j] for m,j in selected) - min(grouped[subset])
    # Posthoc local obligation only; this does not measure expected episode gain.
    if actual['selected_cost_sensitive_contention_rate']:
        counts['selected_span_sum'] += actual['selected_committed_cost_span']
        counts['selected_local_regret_sum'] += regret
        counts['selected_nonminimal_cost_ticks'] += int(regret > 0)
    if actual['cost_sensitive_contention_rate'] and not actual['selected_cost_sensitive_contention_rate']:
        if len(selected) < actual['feasible_matching_cardinality']:
            counts['missed_underfilled_ticks'] += 1
        else:
            counts['missed_full_unique_subset_ticks'] += 1
            exclusive = any(sum(d>0 for d in c.service_time[m]) == 1 for m in subset)
            counts['missed_full_with_exclusive_machine_ticks'] += int(exclusive)
    if actual['cost_sensitive_contention_rate'] and key not in examples:
        examples[key] = dict(seed=active['seed'], ages=state.ages, failed=state.failed,
            remaining=state.remaining, pending_wait=state.pending_wait,
            selected=selected, same_subset_cost_ranges={str(k):[min(v),max(v)] for k,v in grouped.items() if len(k)==actual['feasible_matching_cardinality']},
            selected_local_regret=regret)
    return actual
s.state_metrics = checked_metrics
spec_lookup = {r['candidate']: r for r in specs}
for row in tqdm(rows, desc='Reconstruct and independently audit recorded episodes', unit='episode'):
    active = row
    computed = s.one_episode(cfg, spec_lookup[row['candidate']], row['policy'], row['seed'], row['phase'])
    same(computed, row)
s.state_metrics = original_metrics
assert len(rows) == 8064 and oracle_ticks == len(rows)*48
assert len(rows)*60 == old_audit['physical_transitions'] == 483840
assert manifest['expected_counts'] == manifest['actual_counts'] == dict(screening=6784,confirmation=1280)

per_rule = [dict(candidate=k[0],policy=k[1],**v) for k,v in means.items()]
mechanism_rows = [dict(phase=k[0],candidate=k[1],policy=k[2],**v,
    first_cost_sensitive_example=examples.get(k)) for k,v in mechanisms.items()]
from collections import Counter
selectable = [r for r in ranking if r['selectable']]
failures = dict(Counter(k for r in selectable for k in r['rejection'].split(';')))
result = dict(run=str(RUN), commit=COMMIT, validation=dict(status='PASS',hashed_files=len(manifest['files']),
    source_bindings=bindings,exact_episode_grids=True,seeds=seeds['status'],controls=controls,logged_replay=logged,
    reconstructed_episodes=len(rows),physical_transitions=len(rows)*60,independent_metric_oracle_ticks=oracle_ticks,
    all_episode_metrics_match=True,selection_reconstructed=True,confirmation_CIs='NOT_APPLICABLE_NO_ELIGIBLE_SETTING'),
    selection=selection,ranking=ranking,eligible_count=len(eligible),gate_failures=failures,
    confirmation=summary['confirmation'],confirmation_means=summary['confirmation_means'],
    screening_rule_means=per_rule,mechanism_diagnostics=mechanism_rows,candidate_specs=specs,
    exploratory_scope='posthoc mechanism counts and local committed-service obligation; no new experiment or confirmation')
dump('analysis.json',result)
print(json.dumps(result['validation'],indent=2))
print('gate failures:',failures)
for name in ['bottleneck2__fast2_slow634','bottleneck2__fast2_slow643','overlap3__fast1_slow325','overlap3__fast1_slow352']:
    print('DETAIL',name)
    print(json.dumps([r for r in per_rule if r['candidate']==name],indent=2))
    print(json.dumps([r for r in mechanism_rows if r['phase']=='screening' and r['candidate']==name],indent=2))
