"""Descriptive per-case/seed/compute analysis for an independently audited V20 run.

Primary H1/H2 and original A/B gates remain exactly those in summary.json.
The intervals here are exploratory diagnostics, not new promotion tests.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

CASES = (
    'interchangeable_scale1', 'reciprocal_scale2',
    'reciprocal_scale3', 'specialist_generalist_scale2',
)
ARMS = ('activation_margin', 'generic_q')
METRICS = (
    'objective', 'H_failures', 'H_availability', 'queued_pm_ticks',
    'queued_cm_ticks', 'service_starts', 'legal_service_wait_fraction',
    'maintenance_cost', 'unavailability_cost', 'failure_cost', 'tail_cost',
    'selection_seconds',
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as file:
        return list(csv.DictReader(file))


def describe(run: Path, primary_audit: Path) -> dict:
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETED':
        raise ValueError('V20 run is not complete')
    audit = json.loads(primary_audit.read_text())
    if audit['status'] != 'PASS' or Path(audit['run']).resolve() != run.resolve():
        raise ValueError('primary independent audit must PASS for this run')
    summary = json.loads((run / 'summary.json').read_text())
    rows = csv_rows(run / 'episodes.csv')
    trace = csv_rows(run / 'training_trace.csv')
    registry = manifest['registry']
    seed_count = 5 if manifest['profile'] == 'dev' else 1
    train = tuple(range(registry['training']['start'], registry['training']['start'] + seed_count)) if manifest['profile'] == 'dev' else (registry['smoke']['start'],)
    physical_key = 'evaluation_development' if manifest['profile'] == 'dev' else 'smoke'
    physical_spec = registry[physical_key]
    physical = tuple(range(physical_spec['start'], physical_spec['start'] + physical_spec['count']))
    lookup = {(r['case'], r['controller'], r['train_seed'], int(r['seed'])): r for r in rows}
    if len(lookup) != len(rows):
        raise AssertionError('duplicate final evaluation row')
    controllers = tuple(sorted({r['controller'] for r in rows}))
    by_case = {}
    for case in CASES:
        cells = {}
        for controller in controllers:
            group = [r for r in rows if r['case'] == case and r['controller'] == controller]
            cells[controller] = {
                'episodes': len(group),
                'means': {metric: float(np.mean([float(r[metric]) for r in group])) for metric in METRICS},
            }
            if controller in ARMS:
                cells[controller]['cost_by_training_seed'] = {
                    str(seed): float(np.mean([float(lookup[(case, controller, str(seed), p)]['objective']) for p in physical]))
                    for seed in train
                }
        by_case[case] = cells
    # Crossed seed/tape bootstrap is descriptive; main prelocked 4-case CI stays in summary.json.
    rng = np.random.default_rng(int(registry['bootstrap_seed']) + 1)
    seed_draw = rng.integers(0, len(train), size=(20_000, len(train)))
    tape_draw = rng.integers(0, len(physical), size=(20_000, len(train), len(physical)))
    contrasts = {}
    for comparator in ('generic_q', 'count_calibrated', 'rollout'):
        contrasts[comparator] = {}
        for case in CASES:
            matrix = None
            if comparator == 'generic_q':
                matrix = np.asarray([
                    [float(lookup[(case, 'activation_margin', str(seed), p)]['objective']) -
                     float(lookup[(case, 'generic_q', str(seed), p)]['objective'])
                     for p in physical]
                    for seed in train
                ])
            elif comparator == 'count_calibrated':
                # Treat the locked five-replica V14 ensemble as a fixed comparator.
                matrix = np.asarray([
                    [float(lookup[(case, 'activation_margin', str(seed), p)]['objective']) -
                     float(np.mean([float(lookup[(case, comparator, str(v14_seed), p)]['objective'])
                                    for v14_seed in range(72600000, 72600005)]))
                     for p in physical]
                    for seed in train
                ])
            elif comparator == 'rollout':
                matrix = np.asarray([
                    [float(lookup[(case, 'activation_margin', str(seed), p)]['objective']) -
                     float(lookup[(case, comparator, '', p)]['objective'])
                     for p in physical]
                    for seed in train
                ])
            if matrix is None or matrix.shape != (len(train), len(physical)):
                raise AssertionError('paired contrast grid incomplete')
            ci = np.quantile(matrix[seed_draw[:, :, None], tape_draw].mean(axis=(1, 2)), [.025, .975])
            contrasts[comparator][case] = {
                'candidate_minus_comparator': float(matrix.mean()),
                'descriptive_ci95': [float(ci[0]), float(ci[1])],
                'training_seed_means': {str(seed): float(matrix[i].mean()) for i, seed in enumerate(train)},
            }
    if abs(float(np.mean([contrasts['generic_q'][case]['candidate_minus_comparator'] for case in CASES])) - float(summary['ablation']['mean'])) > 1e-7:
        raise AssertionError('descriptive contrast disagrees with audited primary ablation')
    compute = {}
    for arm in ARMS:
        cell = [r for r in trace if r['arm'] == arm]
        compute[arm] = {
            'updates': len(cell),
            'source_episodes': sum(int(r['source_episodes']) for r in cell),
            'source_steps': sum(int(r['source_steps']) for r in cell),
            'selected_states': sum(int(r['selected_states']) for r in cell),
            'forecast_transitions': sum(int(r['forecast_transitions']) for r in cell),
            'gradient_minibatches': sum(int(r['updates']) for r in cell),
            'counterfactual_plus_gradient_wall_seconds': sum(float(r['wall_seconds']) for r in cell),
            'kl_stops': sum(r['stopped_on_kl'].lower() == 'true' for r in cell),
        }
    return {
        'status': 'DESCRIPTIVE', 'created_utc': datetime.now(timezone.utc).isoformat(), 'run': str(run.resolve()),
        'run_manifest_sha256': sha(run / 'manifest.json'),
        'episodes_sha256': sha(run / 'episodes.csv'),
        'primary_audit_sha256': sha(primary_audit),
        'profile': manifest['profile'],
        'primary_gate_from_audited_summary': {
            'dev_go_gate': summary['dev_go_gate'],
            'ablation': summary['ablation'],
            'original_goal_distances': summary['original_goal_distances'],
        },
        'by_case': by_case, 'descriptive_per_case_contrasts': contrasts,
        'training_compute': compute,
        'limitation': 'Per-case CIs are exploratory, not prelocked gates; the frozen five-replica V14 ensemble is held fixed during bootstrap. Training wall seconds exclude source collection and final evaluation. Selection runtime is measured in the V20 campaign, not independently replayed.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--primary-audit', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    report = describe(args.run_dir.resolve(), args.primary_audit.resolve())
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'descriptive_analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Wrote descriptive analysis: {args.output_dir / 'descriptive_analysis.json'}")


if __name__ == '__main__':
    main()
