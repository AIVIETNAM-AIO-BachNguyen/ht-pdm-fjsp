"""Verify exact model equivalence between two completed engineering smoke runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(root, name):
    return json.loads((root / name).read_text())


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('four_worker_run', type=Path)
parser.add_argument('eight_worker_run', type=Path)
parser.add_argument('--output', type=Path, help='Fresh report path; existing reports are preserved')
args = parser.parse_args()
old, new = args.four_worker_run.resolve(), args.eight_worker_run.resolve()
destination = args.output.resolve() if args.output else new / 'execution_equivalence.json'
snapshot = destination.with_suffix('.py')
if destination.exists() or snapshot.exists():
    raise FileExistsError('Preserve the existing execution verification')
report = dict(classification='ENGINEERING_ONLY', old=str(old), new=str(new),
              verifier_sha256=sha(__file__), status='RUNNING')


def tensors_equal(left, right):
    a = torch.load(left, map_location='cpu', weights_only=True)
    b = torch.load(right, map_location='cpu', weights_only=True)
    assert a['parameter_sha256'] == b['parameter_sha256'], f'Parameter digest differs: {left}'
    assert a['state_dict'].keys() == b['state_dict'].keys(), f'Parameter inventory differs: {left}'
    assert all(torch.equal(a['state_dict'][k], b['state_dict'][k]) for k in a['state_dict']), f'Tensor differs: {left}'
    return dict(parameter_sha256=a['parameter_sha256'], old_file_sha256=sha(left), new_file_sha256=sha(right))


try:
    for root in (old, new):
        manifest = read(root, 'manifest.json')
        assert manifest['status'] == 'COMPLETED' and manifest['audit_status'] == 'PASS', 'Run is unverified'
        assert read(root, 'resolved_config.json')['profile'] == 'smoke', 'Scientific runs cannot be reused for this check'
        for name, binding in manifest['output_bindings'].items():
            assert sha(root / name) == binding['sha256'], f'Sealed input changed: {root / name}'
    a, b = read(old, 'resolved_config.json'), read(new, 'resolved_config.json')
    assert a.keys() == b.keys(), 'Configuration schema changed'
    differences = {k: [a[k], b[k]] for k in a if a[k] != b[k]}
    assert differences == {'training_processes': [4, 8]}, f'Learning configuration changed: {differences}'
    report['configuration_differences'] = differences
    files = sorted(p.name for p in (old / 'source_snapshot').glob('*.py'))
    assert files == sorted(p.name for p in (new / 'source_snapshot').glob('*.py')), 'Source inventory changed'
    for name in files:
        left, right = old / 'source_snapshot' / name, new / 'source_snapshot' / name
        if name == 'technician_reservation_experiment.py':
            text = left.read_text()
            original = '        training_processes=4,\n'
            changed = '        training_processes=8 if (smoke or final) else 4,\n'
            assert text.count(original) == 1 and text.replace(original, changed) == right.read_text(), 'Runner changed beyond concurrency'
        else:
            assert sha(left) == sha(right), f'Learning/physics source changed: {name}'
    paths = sorted(str(p.relative_to(old)) for p in old.glob('N*/**/checkpoints/*.pt'))
    assert paths == sorted(str(p.relative_to(new)) for p in new.glob('N*/**/checkpoints/*.pt')), 'Checkpoint inventory changed'
    report['checkpoints'] = {name: tensors_equal(old / name, new / name) for name in paths}
    warm_a, warm_b = read(old, 'warm_choices.json'), read(new, 'warm_choices.json')
    assert warm_a.keys() == warm_b.keys(), 'Student inventory changed'
    report['warm_checkpoints'] = {}
    report['teacher_banks'] = {}
    for seed in warm_a:
        left, right = old / 'warm' / seed / 'bank.npz', new / 'warm' / seed / 'bank.npz'
        assert sha(left) == sha(right) == warm_a[seed]['bank']['bank_sha256'] == warm_b[seed]['bank']['bank_sha256'], 'Teacher bank changed'
        report['teacher_banks'][seed] = sha(left)
        for arm in warm_a[seed]['arms']:
            name = f'warm/{seed}/{arm}_warm.pt'
            report['warm_checkpoints'][name] = tensors_equal(old / name, new / name)
    choices_a, choices_b = read(old, 'checkpoint_choices.json'), read(new, 'checkpoint_choices.json')
    assert choices_a.keys() == choices_b.keys(), 'Selection inventory changed'
    for name in choices_a:
        assert choices_a[name]['steps'] == choices_b[name]['steps'], 'Selected step changed'
        assert choices_a[name]['parameter_sha256'] == choices_b[name]['parameter_sha256'], 'Selected model changed'
    assert read(old, 'audit.json') == read(new, 'audit.json'), 'Budget/episode audit changed'
    def episodes(root):
        with (root / 'episodes.csv').open() as stream:
            return [{k: v for k, v in row.items() if k not in ('decision_mean_ms', 'decision_p95_ms', 'environment_mean_ms')}
                    for row in csv.DictReader(stream)]
    assert episodes(old) == episodes(new), 'Physical/quality outputs changed'
    report.update(status='PASS', checkpoints_compared=len(paths), warm_checkpoints_compared=len(report['warm_checkpoints']),
                  episode_outputs_compared=len(episodes(old)), scientific_tests_opened=False,
                  input_hashes={str(root / name): sha(root / name) for root in (old, new)
                                for name in ('manifest.json', 'resolved_config.json', 'audit.json', 'checkpoint_choices.json', 'episodes.csv')})
except Exception as error:
    report.update(status='FAILED', error=repr(error))
    raise
finally:
    destination.write_text(json.dumps(report, indent=2) + '\n')
    snapshot.write_bytes(Path(__file__).read_bytes())
    print(destination)
print(json.dumps({k: report[k] for k in ('status', 'checkpoints_compared', 'warm_checkpoints_compared', 'episode_outputs_compared')}))
