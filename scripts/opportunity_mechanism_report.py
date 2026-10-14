"""Read-only diagnostics for completed, test-closed opportunity-message runs."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('--output-root', type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve()

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

manifest = json.loads((source / 'manifest.json').read_text())
assert manifest['protocol'] in ('technician_scarcity_ppo_v2','technician_scarcity_ppo_v3')
assert manifest['status'] == 'COMPLETED' and manifest['audit_status'] == 'PASS'
assert not manifest['test_opened']
assert manifest['profile'] in ('pilot', 'development')
for name, binding in manifest['output_bindings'].items():
    path = source / name
    assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256'], name
config = json.loads((source / 'resolved_config.json').read_text())
choices = json.loads((source / 'checkpoint_choices.json').read_text())
summary = json.loads((source / 'summary.json').read_text())
assert summary['classification'] == 'DEVELOPMENT_ONLY'
assert not config['teacher_assistance'] and config['pretraining_steps'] == 0
with (source / 'training_progress.csv').open() as stream:
    progress = list(csv.DictReader(stream))
with (source / 'development_episodes.csv').open() as stream:
    development = list(csv.DictReader(stream))

control = 'flat_stop_ppo' if manifest['protocol'].endswith('_v3') else 'global_clip_ppo'
arms = ('scarcity_ppo', 'reservation_ppo', 'independent_ppo', control)
metrics = ('actor_grad_norm', 'critic_grad_norm', 'actor_update_norm', 'critic_update_norm')
updates = []
checkpoints = []
for n in config['scales']:
    for arm in arms:
        for seed in config['train_seeds']:
            group = [r for r in progress if int(r['n']) == n and r['algorithm'] == arm
                     and int(r['train_seed']) == seed]
            assert group and int(group[-1]['learned_steps']) == config['env_steps']
            values = {key: np.asarray([float(r[key]) for r in group]) for key in metrics}
            assert all(np.isfinite(v).all() and (v >= 0).all() for v in values.values())
            row = dict(n=n, algorithm=arm, train_seed=seed, rollouts=len(group))
            row.update({key: float(v.mean()) for key, v in values.items()})
            updates.append(row)
            for step in (0, *config['checkpoints']):
                path = source / f'N{n}/{arm}/{seed}/checkpoints/step_{step:09d}.pt'
                payload = torch.load(path, map_location='cpu', weights_only=True)
                assert all(payload[k] == v for k, v in dict(
                    protocol=manifest['protocol'], n=n, algorithm=arm,
                    train_seed=seed, steps=step).items())
                gate = float(payload['state_dict']['message_strength'])
                assert math.isfinite(gate)
                rows = [r for r in development if int(r['n']) == n and r['method'] == arm
                        and int(r['train_seed']) == seed and int(r['checkpoint_steps']) == step]
                assert len(rows) == len(config['development_seeds'])
                key = f'{n}:{arm}:{seed}'
                checkpoints.append(dict(n=n, algorithm=arm, train_seed=seed, steps=step,
                    selected=choices[key]['steps'] == step, gate=gate,
                    maximum_price=.5 * float(torch.sigmoid(torch.tensor(gate))),
                    cost_per_machine=float(np.mean([float(r['normalized_cost']) for r in rows])),
                    price_mean_abs=float(np.mean([float(r['price_mean_abs']) for r in rows])),
                    price_active_fraction=float(np.mean([float(r['price_active_fraction']) for r in rows])),
                    matching_changes_per_episode=float(np.mean([
                        float(r['same_weight_unpriced_matching_changes']) for r in rows]))))

out = args.output_root / (manifest['protocol'] + '_mechanism_report_' +
                         datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
out.mkdir()
for name, rows in (('learning_updates.csv', updates), ('message_checkpoints.csv', checkpoints)):
    with (out / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
lines = ['# Opportunity-message learning diagnostics', '',
         'Exploratory DEVELOPMENT_ONLY; no scientific test opened.',
         f'Source: `{source}`; commit `{manifest["git"]["commit"]}`.', '',
         '| N | Arm | Mean raw actor norm | Mean raw critic norm | Mean actor step norm | Mean critic step norm |',
         '|---|---|---:|---:|---:|---:|']
for n in config['scales']:
    for arm in arms:
        panel = [r for r in updates if r['n'] == n and r['algorithm'] == arm]
        v = [np.mean([r[key] for r in panel]) for key in metrics]
        lines.append(f'| {n} | {arm} | ' + ' | '.join(f'{z:.6g}' for z in v) + ' |')
lines += ['', 'Raw gradient norms cannot establish parameter-step suppression under Adam.',
          'Actual step norms are Euclidean changes over each full parameter group,',
          'averaged across minibatches, rollouts and replicas. They do not measure',
          'policy-space movement or prove clipping caused a performance change.', '',
          'Message gates and same-weight interventions are descriptive. Independent',
          'training of the reservation and flat-STOP (v3) or global-clipping (v2)',
          'controls supplies the',
          'registered contrasts; only untouched confirmation seeds with locked',
          'intervals can support the scientific mechanism and efficacy claims.', '',
          'All replicas and checkpoints are retained in the CSVs, including step0',
          'and unselected checkpoints. Zero-priced control gates remain allocated',
          'for capacity matching and do not represent active communication.']
(out / 'findings.md').write_text('\n'.join(lines) + '\n')
shutil.copyfile(__file__, out / 'report_source.py')
(out / 'report_manifest.json').write_text(json.dumps(dict(
    classification='DEVELOPMENT_ONLY', source=str(source),
    source_manifest_sha256=sha(source / 'manifest.json'),
    script_sha256=sha(Path(__file__)),
    verified_bound_files=len(manifest['output_bindings']),
    outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}), indent=2) + '\n')
print(out)
