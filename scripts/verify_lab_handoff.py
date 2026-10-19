"""Verify the frozen lab handoff payload and compact completed-run capsule.

Standard library only. Read-only; does not execute historical analysis scripts,
train controllers or require the omitted raw decision trace.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = 'maintenance_request_lifecycle_v1_full_cpu_20261019T141931Z'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify(root: Path = ROOT) -> dict:
    package = json.loads((root/'handoff/package_manifest.json').read_text())
    require(package['schema_version'] == 1, 'Unsupported package schema')
    paths = [entry['path'] for entry in package['files']]
    require(len(paths) == len(set(paths)), 'Duplicate package paths')
    for entry in package['files']:
        path = root/entry['path']
        require(path.resolve().is_relative_to(root.resolve()), 'Path outside package')
        require(path.is_file(), f'Missing payload: {entry["path"]}')
        require(path.stat().st_size == entry['bytes'], f'Size mismatch: {entry["path"]}')
        require(digest(path) == entry['sha256'], f'Hash mismatch: {entry["path"]}')

    capsule = root/'evidence'/RUN
    manifest = json.loads((capsule/'manifest.json').read_text())
    require(manifest['status'] == 'COMPLETED', 'Original run was not completed')
    provenance = json.loads((root/'handoff/import_provenance.json').read_text())
    omitted = {entry['path']:entry for entry in provenance['omitted_raw_files']}
    require(set(omitted) == {'decisions.jsonl', 'episodes.partial.csv'}, 'Unexpected omissions')
    for name, expected in manifest['artifact_hashes'].items():
        if name in omitted:
            require(omitted[name]['sha256'] == expected, f'Omitted hash mismatch: {name}')
        else:
            require(digest(capsule/name) == expected, f'Original artifact hash mismatch: {name}')
    for name, expected in manifest['source_hashes'].items():
        copy = capsule/'source_snapshot'/Path(name).name if name.startswith('src/') else capsule/('protocol.md' if name.startswith('docs/') else 'seed_registry.json')
        require(digest(copy) == expected, f'Source snapshot hash mismatch: {name}')
        require(digest(root/name) == expected, f'Active diagnostic source changed: {name}')

    with (capsule/'episodes.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    configs = json.loads((capsule/'configs.json').read_text())
    expected_keys = set(itertools.product(configs, ('threshold','resource_aware'), ('fcfs','risk_matching','rollout'), range(61000000,61000064)))
    keys = {(r['cell'],r['requester'],r['dispatcher'],int(r['seed'])) for r in rows}
    require(len(rows) == len(keys) == 3456 and keys == expected_keys, 'Incomplete/duplicate episode grid')
    summary = json.loads((capsule/'summary.json').read_text())
    require(len(summary['means']) == 54 and len(summary['contrasts']) == 45, 'Summary shape mismatch')
    return dict(status='PASS', payload_files=len(paths), payload_bytes=sum(e['bytes'] for e in package['files']), episodes=len(rows), original_run=RUN, full_trace_in_git=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
