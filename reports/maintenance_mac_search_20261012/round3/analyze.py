"""Independent integrity and count-weight bootstrap reconstruction; no reruns."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path('/Users/bachng/Coding/Reinforcement Learning/marl')
CHECKOUT = Path('/private/tmp/marl-maintenance-assignment-confidence-v1')
RUN = ROOT / 'artifacts/maintenance_joint_learning_mac_v1_full_cpu_20261012T064635Z'
OUT = ROOT / 'reports/maintenance_mac_search_20261012/round3'
VERSION = 'maintenance_joint_learning_mac_v1'

def read(name):
    return json.loads((RUN / name).read_text())

def rows(name):
    with (RUN / name).open() as handle:
        return list(csv.DictReader(handle))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def role(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'big')

manifest, summary, cfg = read('manifest.json'), read('summary.json'), read('resolved_config.json')['settings']
assert manifest['status'] == 'COMPLETED' and manifest['audit'] == 'PASS'
for name, binding in manifest['files'].items():
    path = RUN / name
    assert path.stat().st_size == binding['bytes'] and digest(path) == binding['sha256'], name
for name, seal in manifest['source_seal'].items():
    assert digest(RUN / name) == seal
    source = ('docs/maintenance_joint_learning_mac_v1_plan.md' if name == 'protocol.md'
              else 'configs/maintenance_joint_learning_mac_v1_seed_registry.json' if name == 'registry.json'
              else 'src/ht_pdm_fjsp/' + Path(name).name)
    committed = subprocess.check_output(['git', 'show', f"{manifest['git']['commit']}:{source}"], cwd=CHECKOUT)
    assert hashlib.sha256(committed).hexdigest() == seal, source
assert (RUN / 'episodes.partial.csv').read_bytes() == (RUN / 'episodes.csv').read_bytes()
assert digest(RUN / 'selections.json') == manifest['selection_sha256']
selection, dev, test = read('selections.json'), rows('development_episodes.csv'), rows('episodes.csv')
train, seeds = cfg['train_seeds'], cfg['test_seeds']
dev_expected = set(itertools.product(train, cfg['checkpoints'], cfg['development_seeds']))
assert len(dev) == len(dev_expected) == 384
assert {(int(r['train_seed']), int(r['checkpoint_steps']), int(r['seed'])) for r in dev} == dev_expected
for seed in train:
    costs = {step: np.mean([float(r['extended_cost']) for r in dev
                           if int(r['train_seed']) == seed and int(r['checkpoint_steps']) == step])
             for step in cfg['checkpoints']}
    choice = min(costs, key=lambda step: (costs[step], step))
    assert choice == selection[str(seed)]['steps']
    assert costs[choice] == selection[str(seed)]['development_cost']
names = [('ppo', s) for s in train] + [('initial', s) for s in train] + [(m, -1) for m in
         ('deadline_matching', 'isolated_h6', 'isolated_endpoint', 'joint_rollout')]
lookup = {(r['controller'], int(r['train_seed']), int(r['seed'])): r for r in test}
assert len(test) == len(lookup) == 1280
assert set(lookup) == {(m, s, raw) for m, s in names for raw in seeds}
for (name, seed, raw), r in lookup.items():
    assert int(r['physical_ticks']) == 60 and float(r['invalid_assignments']) == 0
    assert int(r['physical_seed']) == role(raw, f'{VERSION}:physical')
    assert float(r['extended_cost']) == float(r['policy_cost']) + float(r['tail_cost'])
    step = selection[str(seed)]['steps'] if name == 'ppo' else 0 if name == 'initial' else -1
    assert int(r['checkpoint_steps']) == step
training, partial = rows('training_episodes.csv'), read('training_partial.json')
observed = []
counts = {}
for seed in train:
    complete = [r for r in training if int(r['train_seed']) == seed]
    unfinished = [r for r in partial if int(r['train_seed']) == seed]
    assert len(complete) == 1360 and len(unfinished) == 8
    assert 48 * len(complete) + sum(int(r['learned_ticks']) for r in unfinished) == cfg['steps']
    for worker in range(8):
        group = [r for r in complete if int(r['worker']) == worker]
        tail = [r for r in unfinished if int(r['worker']) == worker]
        assert {int(r['episode_index']) for r in group} == set(range(170)) and len(tail) == 1
        assert int(tail[0]['episode_index']) == 170 and int(tail[0]['learned_ticks']) == 32
    for r in complete + unfinished:
        worker, episode = int(r['worker']), int(r['episode_index'])
        assert int(r['stream_seed']) == seed + worker
        assert int(r['physical_seed']) == role(seed + worker, f'{VERSION}:training_physical:worker:{worker}:episode:{episode}')
        observed.append(int(r['physical_seed']))
    assert all(float(r['invalid_assignments']) == 0 and int(r['physical_ticks']) == 60 for r in complete)
    physical = 60 * len(complete) + sum(int(r['physical_ticks']) for r in unfinished)
    assert physical == 81856
    counts[str(seed)] = dict(steps=65536, complete_episodes=1360, partial_workers=8, physical_ticks=physical)
assert len(observed) == len(set(observed)) == 4104
assert not set(observed) & {role(raw, f'{VERSION}:physical') for raw in seeds + cfg['development_seeds']}
fields = ('extended_cost', 'availability', 'terminal_pending')
matrix = lambda name: np.asarray([[[float(lookup[name, s, raw][k]) for k in fields] for raw in seeds] for s in train])
learned, initial = matrix('ppo'), matrix('initial')
base = np.asarray([[float(lookup['deadline_matching', -1, raw][k]) for k in fields] for raw in seeds])
rng = np.random.default_rng(role(cfg['analysis_seed'], f'{VERSION}:bootstrap'))
nr, ne, nb = len(train), len(seeds), cfg['bootstrap_samples']
replicas, episodes = rng.integers(nr, size=(nb, nr)), rng.integers(ne, size=(nb, ne))
rw = np.stack([(replicas == i).sum(1) for i in range(nr)], axis=1) / nr
ew = np.stack([(episodes == i).sum(1) for i in range(ne)], axis=1) / ne
a = np.einsum('br,be,rek->bk', rw, ew, learned, optimize=True)
b = np.einsum('br,be,rek->bk', rw, ew, initial, optimize=True)
c = ew @ base
distributions = dict(gain_vs_initial=1-a[:, 0]/b[:, 0], gain_vs_deadline=1-a[:, 0]/c[:, 0],
                     availability_difference=a[:, 1]-c[:, 1], pending_difference=a[:, 2]-c[:, 2])
point, initial_point, base_point = learned.mean((0, 1)), initial.mean((0, 1)), base.mean(0)
estimates = dict(gain_vs_initial=1-point[0]/initial_point[0], gain_vs_deadline=1-point[0]/base_point[0],
                 availability_difference=point[1]-base_point[1], pending_difference=point[2]-base_point[2])
contrasts = {}
for name, distribution in distributions.items():
    interval = np.quantile(distribution, [.025, .975])
    assert np.allclose(interval, summary['contrasts'][name]['ci95'], atol=1e-12, rtol=0), name
    assert np.isclose(estimates[name], summary['contrasts'][name]['estimate'], atol=1e-12, rtol=0)
    contrasts[name] = dict(estimate=float(estimates[name]), ci95=interval.tolist())
for mean in summary['means']:
    group = [r for r in test if r['controller'] == mean['controller'] and int(r['train_seed']) == mean['train_seed']]
    for field, value in mean.items():
        if field not in ('controller', 'train_seed'):
            assert np.isclose(np.mean([float(r[field]) for r in group]), value, atol=1e-12, rtol=0), field
result = dict(status='PASS', verification='file hashes/committed source/complete grids/frozen checkpoint choices/training provenance/count-weight crossed bootstrap',
              run=str(RUN), commit=manifest['git']['commit'], files_checked=len(manifest['files']), test_episodes=1280,
              development_episodes=384, training=counts, contrasts=contrasts, means=summary['means'],
              learned_aggregate=dict(zip(fields, point.tolist())), initial_aggregate=dict(zip(fields, initial_point.tolist())),
              remaining_cost_gap_to_rollout=float(point[0]-next(m['extended_cost'] for m in summary['means'] if m['controller']=='joint_rollout')),
              physical_replay_scope='Runner audit: every test tick and learned prediction; forecast replay first two test seeds only.')
(OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('status', 'files_checked', 'test_episodes', 'development_episodes', 'learned_aggregate', 'remaining_cost_gap_to_rollout', 'contrasts')}, indent=2))
