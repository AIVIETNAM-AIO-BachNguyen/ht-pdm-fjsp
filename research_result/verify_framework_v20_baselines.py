"""Replay frozen V7 and rule comparators from a completed V20 campaign.

This addendum does not alter the prelocked V20 campaign or its primary verifier.
It checks the comparator episode costs and operations with a fresh simulator replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, HORIZON, FrameworkEnv, catalogue, configs
from ht_pdm_fjsp.maintenance_framework_v20_campaign import (
    ROOT, V7, V7_ARMS, RULES, V7LearnedController, V7RuleController, v7_models,
)
from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests

ACTIONS = catalogue()
METRICS = (
    'objective', 'H_availability', 'H_failures', 'queued_cm_ticks',
    'queued_pm_ticks', 'H_pending', 'end_pending', 'service_starts',
    'maintenance_cost', 'unavailability_cost', 'failure_cost', 'tail_cost',
    'requests_created', 'requests_completed', 'end_in_service',
    'invalid_assignments',
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(row: dict[str, str], controller) -> float:
    cfg = configs()[row['case']]
    env = FrameworkEnv(cfg)
    env.reset(seed=int(row['seed']))
    legal_ticks = waits = 0
    max_error = 0.0
    for tick in range(HORIZON):
        submissions, matching = controller.choose(
            cfg, env.inner.state, tick, case=row['case'], seed=int(row['seed'])
        )
        admitted = submit_requests(cfg, env.inner.state, tick, submissions)
        legal = legal_matchings(cfg, admitted)
        if matching not in legal:
            raise AssertionError('baseline selected an illegal matching')
        if any(legal):
            legal_ticks += 1
            waits += int(len(matching) == 0)
        env.step(ACTIONS.index(matching))
    measured = env.episode_metrics()
    for key in METRICS:
        delta = abs(float(measured[key]) - float(row[key]))
        max_error = max(max_error, delta)
        if not np.isclose(measured[key], float(row[key]), atol=1e-7, rtol=0):
            raise AssertionError(f"{row['controller']} {row['case']} {row['seed']} {key}: {measured[key]} != {row[key]}")
    for key, actual in (
        ('episode_return', env.episode_return),
        ('legal_service_ticks', legal_ticks),
        ('legal_service_waits', waits),
        ('legal_service_wait_fraction', waits / legal_ticks if legal_ticks else 0),
    ):
        delta = abs(float(actual) - float(row[key]))
        max_error = max(max_error, delta)
        if not np.isclose(actual, float(row[key]), atol=1e-7, rtol=0):
            raise AssertionError(f"{row['controller']} {row['case']} {row['seed']} {key}: {actual} != {row[key]}")
    return max_error


def audit(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETED':
        raise ValueError('V20 campaign must be complete before baseline replay')
    if manifest['profile'] not in ('smoke', 'dev'):
        raise ValueError('unsupported profile')
    for name, sha in manifest['source_hashes'].items():
        if digest(ROOT / name) != sha:
            raise AssertionError(f'V20 source changed: {name}')
    v7, hashes, selection_hash = v7_models()
    if selection_hash != manifest['v7_selection_sha256']:
        raise AssertionError('V7 selection changed')
    for key, sha in hashes.items():
        if sha != manifest['prior_comparator_hashes'][key]:
            raise AssertionError(f'V7 checkpoint changed: {key}')
    with (run_dir / 'episodes.csv').open(newline='') as file:
        episodes = list(csv.DictReader(file))
    physical = range(73711000, 73711016) if manifest['profile'] == 'dev' else range(73720000, 73720002)
    expected = len(CASES) * len(physical) * (len(V7_ARMS) * 5 + len(RULES))
    selected = [r for r in episodes if r['controller'] in (*V7_ARMS, *RULES)]
    if len(selected) != expected:
        raise AssertionError(f'baseline row count {len(selected)} != {expected}')
    if len({(r['case'], r['controller'], r['train_seed'], r['seed']) for r in selected}) != expected:
        raise AssertionError('duplicate baseline row')
    if {int(r['seed']) for r in selected} != set(physical):
        raise AssertionError('baseline physical panel changed')
    start = time.monotonic()
    max_error = 0.0
    counts = {name: 0 for name in (*V7_ARMS, *RULES)}
    for row in tqdm(selected, desc='Independent V20 comparator replay', unit='episode'):
        name = row['controller']
        if name in V7_ARMS:
            seed = int(row['train_seed'])
            if seed not in range(68000000, 68000005):
                raise AssertionError('V7 training seed changed')
            controller = V7LearnedController(v7[(name, seed)], canonical=name == 'shared_canonical')
        else:
            if row['train_seed']:
                raise AssertionError('rule training seed should be blank')
            controller = V7RuleController(name, manifest['registry']['forecast_seed'], lambda: None)
        max_error = max(max_error, replay(row, controller))
        counts[name] += 1
    return {
        'status': 'PASS', 'run': str(run_dir), 'profile': manifest['profile'],
        'episodes_sha256': digest(run_dir / 'episodes.csv'),
        'baseline_episode_replays': len(selected), 'by_controller': counts,
        'max_absolute_metric_error': max_error,
        'wall_seconds': time.monotonic() - start,
        'scope': 'All frozen V7 and rule comparator episodes; cost, operations, legality and request accounting. Wall-clock selection times are not replayed.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    report = audit(args.run_dir.resolve())
    (args.output_dir / 'baseline_replay_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
