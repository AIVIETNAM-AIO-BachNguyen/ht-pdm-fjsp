"""Post hoc seed-cluster intervals; no change to the registered decision gates."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = Path('/private/tmp/marl-maintenance-assignment-confidence-v1')
sys.path.insert(0, str(CODE / 'src'))
from ht_pdm_fjsp.maintenance_dispatch import role_seed

analysis = json.loads((HERE / 'analysis.json').read_text())
run = Path(analysis['run'])
cfg = json.loads((run / 'resolved_config.json').read_text())['settings']
winner = analysis['selection']['winner']
with (HERE / 'tie_diagnostics_episodes.csv').open() as stream:
    rows = list(csv.DictReader(stream))
lookup = {(r['candidate'], r['policy'], int(r['seed'])): r
          for r in rows if r['phase'] == 'confirmation'}
policies = ('deadline_matching', 'priority_greedy')
n = len(cfg['confirmation_seeds'])
rng = np.random.default_rng(role_seed(cfg['analysis_seed'], 'maintenance_skill_contention_screen_v1:bootstrap'))
indices = rng.integers(0, n, size=(cfg['bootstrap_samples'], n))
result = {}
for metric in ('unavoidable_best_set_conflict_rate',
               'strict_preference_deficit_rate', 'tie_resolvable_conflict_rate'):
    values = np.asarray([[float(lookup[winner, p, seed][metric]) for p in policies]
                         for seed in cfg['confirmation_seeds']])
    draws = values[indices].mean(axis=1)
    result[metric] = dict(min_policy_estimate=float(values.mean(axis=0).min()),
        min_policy_ci9875=np.quantile(draws.min(axis=1), [.00625, .99375]).tolist(),
        per_policy=dict(zip(policies, values.mean(axis=0).tolist())))
(HERE / 'tie_diagnostic_intervals.json').write_text(json.dumps(result, indent=2) + '\n')
print('Post hoc diagnostic intervals reconstructed; registered gates unchanged.')
