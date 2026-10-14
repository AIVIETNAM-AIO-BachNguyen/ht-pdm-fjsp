"""Bound descriptive analysis of already recorded frozen v3 trajectories."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('--output-root', type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve()
manifest = json.loads((source / 'manifest.json').read_text())
assert manifest['status'] == 'COMPLETED' and manifest['profile'] == 'diagnostic'
assert manifest['episodes'] == 672 and manifest['independent_physical_ticks'] == 40320
assert not manifest['scientific_test_opened']
assert manifest['optimizer_steps'] == manifest['parameter_updates'] == 0
for name, binding in manifest['output_bindings'].items():
    path = source / name
    assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256'], name
run = Path(manifest['source'])
assert sha(run / 'manifest.json') == manifest['source_manifest_sha256']
run_manifest = json.loads((run / 'manifest.json').read_text())
binding = run_manifest['output_bindings']['physics.json']
assert (run / 'physics.json').stat().st_size == binding['bytes']
assert sha(run / 'physics.json') == binding['sha256']
physics = json.loads((run / 'physics.json').read_text())['4']
# All N4 edges are compatible; a free technician is a legal option for each
# recorded unassigned at-risk machine. Do not generalize this shortcut to skills.
assert all(v > 0 for row in physics['service_time'] for v in row)
states = {}
with (source / 'physical.csv').open() as stream:
    for row in csv.DictReader(stream):
        if row['phase'] == 'policy':
            key = tuple(row[k] for k in ('method', 'train_seed', 'development_seed', 'tick'))
            assert key not in states
            states[key] = json.loads(row['state'])
assert len(states) == 672 * 48
counts = defaultdict(Counter)
seen = set()
with (source / 'decisions.csv').open() as stream:
    for row in csv.DictReader(stream):
        key = tuple(row[k] for k in ('method', 'train_seed', 'development_seed', 'tick'))
        assert key not in seen
        seen.add(key)
        state = states[key]
        pairs = json.loads(row['executed'])
        used = {j for _, j in pairs}
        selected = {m for m, _ in pairs}
        expected_risk = [m for m in range(4) if not state['failed'][m] and
            state['ages'][m] >= physics['failure_age'] - 1 and
            m not in state['assigned'] and m not in selected]
        assert expected_risk == json.loads(row['unserved_at_risk'])
        free = [j for j, remaining in enumerate(state['remaining']) if remaining == 0]
        for label in ('unserved_at_risk', 'failed_unserved_at_risk'):
            for machine in json.loads(row[label]):
                counts[row['method']][label + ('_initial_free' if free else '_all_busy')] += 1
                if any(j not in used for j in free):
                    counts[row['method']][label + '_residual_free'] += 1
assert seen == set(states)
starts = defaultdict(Counter)
with (source / 'starts.csv').open() as stream:
    for row in csv.DictReader(stream):
        starts[row['method']]['all'] += 1
        starts[row['method']]['full_restoration'] += float(row['restoration']) == 1.
        starts[row['method']]['duration_gt_one'] += float(row['duration']) > 1
aggregate = json.loads((source / 'aggregate.json').read_text())
summary = {}
for method in aggregate:
    keys = [label + suffix for label in ('unserved_at_risk', 'failed_unserved_at_risk')
            for suffix in ('_initial_free', '_all_busy', '_residual_free')]
    values = {key: counts[method][key] / 96 for key in keys}
    values.update(full_restoration_start_fraction=starts[method]['full_restoration'] / starts[method]['all'],
                  longer_service_start_fraction=starts[method]['duration_gt_one'] / starts[method]['all'])
    summary[method] = values
out = args.output_root / ('technician_scarcity_ppo_v3_timing_report_' +
                         datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
out.mkdir(parents=True)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
shutil.copyfile(__file__, out / 'report_source.py')
(out / 'manifest.json').write_text(json.dumps(dict(
    classification='EXPLORATORY_DEV_ONLY_NO_TRAINING', source=str(source),
    source_manifest_sha256=sha(source / 'manifest.json'), script_sha256=sha(Path(__file__)),
    verified_input_bindings=len(manifest['output_bindings']),
    output_bindings={p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                     for p in out.iterdir() if p.is_file()}), indent=2) + '\n')
print(out)
