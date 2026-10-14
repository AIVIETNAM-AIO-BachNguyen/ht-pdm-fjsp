"""Frozen-checkpoint STOP calibration diagnostic on one already opened DEV seed.

No training or new seed panel. Capture first-slot probabilities without changing
the value returned by log_softmax. Test whether greedy STOP coexists with an
aggregate assignment probability above one half; this is descriptive, not causal.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import torch
from ht_pdm_fjsp import technician_opportunity_experiment as x

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('--output-root', type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve()
manifest = json.loads((source / 'manifest.json').read_text())
assert manifest['status'] == 'COMPLETED' and manifest['audit_status'] == 'PASS'
assert manifest['profile'] == 'pilot' and not manifest['test_opened']
assert manifest['protocol'] == x.VERSION
sha = x.sha
for name, binding in manifest['output_bindings'].items():
    p = source / name
    assert p.stat().st_size == binding['bytes'] and sha(p) == binding['sha256'], name
for p in Path(x.__file__).parent.glob('*.py'):
    assert sha(p) == manifest['source_seal']['source_snapshot/' + p.name], p
config = json.loads((source / 'resolved_config.json').read_text())
seed = config['development_seeds'][0]
torch.set_num_threads(1)
rows = []
probabilities = []
for n in config['scales']:
    for train in config['train_seeds']:
        for step in (0, *config['checkpoints']):
            path = source / f'N{n}/scarcity_ppo/{train}/checkpoints/step_{step:09d}.pt'
            model = x.load_model(path, n, config, train, 'scarcity_ppo', step)
            digest = x.learner.parameter_digest(model)
            env = x.ResourceEnv(seed, n, 'scarcity_ppo')
            obs = env.reset()
            stopped = disagreements = eligible = 0
            original = torch.log_softmax
            for tick in range(48):
                captures = []
                def capture(logits, *a, **kw):
                    result = original(logits, *a, **kw)
                    captures.append((logits.detach().clone(), result.detach().exp()))
                    return result
                try:
                    torch.log_softmax = capture
                    action = model.select(obs)
                finally:
                    torch.log_softmax = original
                assert len(captures) == n // 2
                logits, probability = captures[0]
                count = int((logits[0, 1:] > -1e8).sum())
                pstop = float(probability[0, 0])
                stop = int(action[0] == 0)
                discrepancy = bool(count and stop and pstop < .5)
                eligible += bool(count)
                stopped += bool(count and stop)
                disagreements += discrepancy
                probabilities.append(dict(n=n, train_seed=train, checkpoint_steps=step,
                    development_seed=seed, tick=tick, legal_edges=count,
                    greedy_stop=stop, stop_probability=pstop,
                    aggregate_go_probability=1-pstop,
                    largest_edge_probability=float(probability[0, 1:].max()),
                    greedy_stop_with_majority_go_mass=discrepancy))
                obs, _, done, info = env.step(action)
            assert done and x.learner.parameter_digest(model) == digest
            rows.append(dict(n=n, train_seed=train, checkpoint_steps=step,
                development_seed=seed, eligible_first_slots=eligible,
                greedy_stops_with_legal_edges=stopped,
                greedy_stops_with_majority_go_mass=disagreements,
                extended_cost=info['episode_metrics']['extended_cost'],
                checkpoint_sha256=sha(path)))
out = args.output_root / ('technician_scarcity_ppo_v2_stop_diagnostic_' +
                         datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
out.mkdir()
for name, data in (('checkpoints.csv', rows), ('probabilities.csv', probabilities)):
    with (out / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(data[0]))
        writer.writeheader()
        writer.writerows(data)
shutil.copyfile(__file__, out / 'diagnostic_source.py')
(out / 'manifest.json').write_text(json.dumps(dict(
    status='COMPLETED', classification='EXPLORATORY_DEV_ONLY_NO_TRAINING',
    source=str(source), source_manifest_sha256=sha(source / 'manifest.json'),
    development_seed=seed, scientific_test_opened=False,
    optimizer_steps=0, parameter_updates=0,
    script_sha256=sha(Path(__file__)),
    output_bindings={p.name: sha(p) for p in out.iterdir() if p.is_file()}), indent=2) + '\n')
print(out)
