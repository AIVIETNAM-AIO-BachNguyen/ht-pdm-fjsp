"""Read-only recomputation; no training, checkpoint inference, or raw edits."""
import collections
import csv
import json
import math
import pathlib
import statistics as stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'lab_results/ra_qmix_role_context_full_cpu_20261003T131518Z'
OUT = pathlib.Path(__file__).parent

def read(name):
    with (SOURCE / name).open() as f:
        return list(csv.DictReader(f))

models = read('model_metrics.csv')
paired = read('paired_seed_metrics.csv')
states = read('state_metrics.csv')
episodes = read('episodes.csv')
logs = read('training_progress.csv')
manifest = json.loads((SOURCE / 'manifest.json').read_text())
assert manifest['status'] == 'COMPLETED'
assert all((SOURCE / path).is_file() for path in manifest['outputs'])
assert len(models) == len(list(SOURCE.rglob('model.pt'))) == 150
assert len(states) == 67000 and len(logs) == 15000 and len(episodes) == 7500
keys = ['cell', 'algorithm', 'train_seed']
index = {tuple(r[k] for k in keys): r for r in models}
assert len(index) == 150
assert len({tuple(r[k] for k in keys + ['state_id']) for r in states}) == 67000
assert len({tuple(r[k] for k in keys + ['eval_seed']) for r in episodes}) == 7500
assert all(float(r['invalid_requests']) == 0 for r in episodes)
assert all(abs(float(r['objective']) - sum(float(r[k]) for k in
    ['maintenance_cost', 'queue_waiting_cost', 'failure_cost', 'downtime_cost'])) < 1e-10 for r in episodes)
assert all(math.isfinite(float(r[k])) for r in logs for k in
    ['normalized_mse', 'normalized_cf', 'raw_mse', 'raw_cf', 'gradient_norm'])
groups = collections.defaultdict(list)
for r in logs:
    groups[tuple(r[k] for k in keys)].append(int(r['update']))
assert len(groups) == 150 and all(sorted(v) == list(range(100, 10001, 100)) for v in groups.values())
assert all(float(r['raw_cf']) == float(r['normalized_cf']) == 0 for r in logs)
assert all(r['greedy_joint_agreement'] == 'True' for r in states if r['algorithm'].startswith('monotonic'))

def diagnostic(seed, arm, key='heldout_regret'):
    return stats.mean(float(index[(c, arm, seed)][key]) for c in ['shared_preference', 'pressure'])

for row in paired:
    seed = row['train_seed']
    for arm in ['monotonic_base', 'monotonic_role', 'monotonic_context', 'monotonic_role_context']:
        assert abs(diagnostic(seed, arm) - float(row[arm + '_regret'])) < 1e-12
    for name, a, b in [('role', 'monotonic_role', 'monotonic_base'),
                       ('peer_given_role', 'monotonic_role_context', 'monotonic_role')]:
        assert abs(diagnostic(seed, a) - diagnostic(seed, b) - float(row[name + '_regret_delta'])) < 1e-12
        assert abs(diagnostic(seed, a, 'undiscounted_policy_cost') - diagnostic(seed, b, 'undiscounted_policy_cost') - float(row[name + '_policy_cost_delta'])) < 1e-12

contrasts = {}
for name in ['role', 'peer_given_role', 'peer_without_role', 'factorial']:
    column = 'factorial_regret_interaction' if name == 'factorial' else name + '_regret_delta'
    values = [float(r[column]) for r in paired]
    mean = stats.mean(values)
    radius = 2.262157163 * stats.stdev(values) / math.sqrt(len(values))
    contrasts[name] = dict(mean=mean, ci95=[mean-radius, mean+radius], improving_seeds=sum(v < 0 for v in values))

means, mechanisms, feasibility = [], [], {}
verified_states = 0
for cell in ['nominal', 'shared_preference', 'pressure']:
    oracle_states = json.loads((SOURCE / cell / 'oracle_states.json').read_text())
    q = collections.defaultdict(dict)
    for r in read(cell + '/oracle_q.csv'):
        q[int(r['state_id'])][tuple(json.loads(r['actions']))] = float(r['q95'])
    feasibility[cell] = {}
    results = json.loads((SOURCE / cell / 'policy_feasibility.json').read_text())
    # Reconstruct input vectors independently from saved observations.
    for arm, result in results.items():
        feasibility[cell][arm] = result['status']
        if result['status'] in ['OPTIMAL', 'FEASIBLE']:
            lookup = {tuple(w['inputs']): w['action'] for w in result['witness_policy']}
            for st in oracle_states:
                actions = []
                for i in range(2):
                    identity = [int(i == j) for j in range(2)] if result['identity'] else [0, 0]
                    peer = st['observations'][1-i] if result['peer_information'] else [0] * 15
                    actions.append(lookup[tuple(st['observations'][i] + identity + peer)])
                values = q[st['state_id']]
                assert tuple(actions) in values
                assert max(values.values()) - values[tuple(actions)] <= 1e-8
                verified_states += 1
        elif result['method'] == 'direct_same_input_conflict':
            values = q[result['state_id']]
            optimum = max(values.values())
            optimal = [a for a, v in values.items() if optimum - v <= 1e-8]
            assert all(a[0] != a[1] for a in optimal)
    for arm in ['monotonic_base', 'monotonic_role', 'monotonic_context', 'monotonic_role_context', 'joint_q']:
        rr = [r for r in models if r['cell'] == cell and r['algorithm'] == arm]
        record = dict(cell=cell, algorithm=arm)
        for key in ['train_regret', 'heldout_regret', 'train_normalized_rmse', 'heldout_normalized_rmse',
                    'undiscounted_policy_cost', 'undiscounted_policy_gap']:
            record[key] = stats.mean(float(r[key]) for r in rr)
        means.append(record)
        critical = [r for r in states if r['cell'] == cell and r['algorithm'] == arm
                    and r['original_inputs_identical'] == 'True' and r['oracle_requires_unequal_actions'] == 'True']
        mechanisms.append(dict(cell=cell, algorithm=arm, state_seed_pairs=len(critical),
            unequal_actions=sum(len(set(json.loads(r['selected_actions']))) > 1 for r in critical),
            optimal_actions=sum(float(r['regret']) <= 1e-8 for r in critical)))

for filename, rows in [('cell_means.csv', means), ('symmetry_actions.csv', mechanisms)]:
    with (OUT / filename).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
result = dict(source=str(SOURCE), source_revision=manifest['git_revision'], status='ANALYZED',
    independent_checks='CSV arithmetic, file/key/count/log audits, reconstructed SAT lookup inputs and oracle optimality; no retraining or checkpoint inference',
    contrasts=contrasts, feasibility=feasibility, verified_sat_state_assignments=verified_states,
    pressure_role_unsat='Reported CP-SAT certificate; not independently re-solved here')
(OUT / 'analysis_checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
