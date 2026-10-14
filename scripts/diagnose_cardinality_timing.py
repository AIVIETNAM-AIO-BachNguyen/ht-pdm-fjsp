"""Read-only selected N4 trajectories on the already opened v3 DEV panel."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import technician_cardinality_experiment as x
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import transition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--profile', choices=('smoke', 'diagnostic'), required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['protocol'] == x.VERSION
    assert manifest['status'] == 'COMPLETED' and manifest['audit_status'] == 'PASS'
    assert manifest['profile'] == 'development' and not manifest['test_opened']
    for name, binding in manifest['output_bindings'].items():
        path = source / name
        assert path.stat().st_size == binding['bytes'] and x.sha(path) == binding['sha256'], name
    for path in Path(x.__file__).parent.glob('*.py'):
        assert x.sha(path) == manifest['source_seal']['source_snapshot/' + path.name], path
    assert x.sha(x.PLAN) == manifest['source_seal']['protocol.md']
    assert x.sha(x.REGISTRY) == manifest['source_seal']['seed_registry.json']
    config = json.loads((source / 'resolved_config.json').read_text())
    assert not config['test_seeds'] and not config['teacher_assistance']
    assert config['pretraining_steps'] == 0
    choices = json.loads((source / 'checkpoint_choices.json').read_text())
    with (source / 'development_episodes.csv').open() as stream:
        original = {(int(r['n']), r['method'], int(r['train_seed']),
                     int(r['checkpoint_steps']), int(r['seed'])): r
                    for r in csv.DictReader(stream)}
    trains = config['train_seeds'][:1] if args.profile == 'smoke' else config['train_seeds']
    seeds = config['development_seeds'][:1] if args.profile == 'smoke' else config['development_seeds']
    methods = x.arms(4)
    started = time.monotonic()
    out = args.output_root / (x.VERSION + '_timing_' + args.profile + '_' +
                             datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    plan = Path(__file__).resolve().parents[1] / 'docs/technician_scarcity_ppo_v3_timing_diagnostic_plan.md'
    shutil.copyfile(__file__, out / 'diagnostic_source.py')
    shutil.copyfile(plan, out / 'diagnostic_plan.md')
    result = dict(status='RUNNING', classification='EXPLORATORY_DEV_ONLY_NO_TRAINING',
                  profile=args.profile, source=str(source), source_manifest_sha256=x.sha(source / 'manifest.json'),
                  script_sha256=x.sha(Path(__file__)), plan_sha256=x.sha(plan),
                  train_seeds=trains, development_seeds=seeds, scientific_test_opened=False,
                  optimizer_steps=0, parameter_updates=0, teacher_assistance=False,
                  verified_input_bindings=len(manifest['output_bindings']))
    x.write_json(out / 'manifest.json', result)
    journals = {name: x.Journal(out / (name + '.csv')) for name in ('decisions', 'starts', 'physical', 'episodes')}
    episodes = []
    populations = {}
    replays = 0
    torch.set_num_threads(1)

    def timeout():
        if time.monotonic() - started > 600:
            raise TimeoutError('registered diagnostic hard stop')

    try:
        jobs = [(method, train, seed) for method in methods for train in trains for seed in seeds]
        for method, train, seed in tqdm(jobs, desc='Frozen N4 DEV trajectories'):
            timeout()
            choice = choices[f'4:{method}:{train}']
            path = source / choice['path']
            assert x.sha(path) == choice['sha256']
            model = x.load_model(path, 4, config, train, method, choice['steps'])
            before_digest = x.learner.parameter_digest(model)
            tags = dict(method=method, train_seed=train, development_seed=seed,
                        checkpoint_steps=choice['steps'])
            records = []

            class PhysicalLog:
                def add(self, record):
                    records.append(record)
                    journals['physical'].add(dict(tags, **record))

            env = x.ResourceEnv(seed, 4, method, journal=PhysicalLog())
            obs = env.reset()
            assert populations.setdefault(seed, env.population) == env.population
            counts = dict(first_stop_with_legal=0, ticks_residual_legal=0,
                          unserved_at_risk=0, failed_unserved_at_risk=0,
                          preventive_starts=0, corrective_starts=0,
                          conflicts=0, rejected_proposals=0)
            service_sum = restoration_sum = age_sum = 0.
            for tick in range(48):
                timeout()
                state = env.env.state
                legal = obs['pair_mask'].copy()
                action = model.select(obs)
                pairs, coordination = env.resolve(action)
                selected = {m for m, _ in pairs}
                servicing = {m for m in state.assigned if m >= 0}
                risk = {m for m in range(4) if not state.failed[m] and
                        state.ages[m] >= env.config.failure_age - 1 and
                        m not in servicing and m not in selected}
                residual = legal.copy()
                for m, j in pairs:
                    residual[m, :] = False
                    residual[:, j] = False
                    counts['corrective_starts' if state.failed[m] else 'preventive_starts'] += 1
                    duration = env.config.service_time[m][j]
                    restoration = env.config.restoration[m][j]
                    service_sum += duration
                    restoration_sum += restoration
                    age_sum += state.ages[m]
                    journals['starts'].add(dict(tags, tick=tick, machine=m, technician=j,
                        age=state.ages[m], failed=state.failed[m], duration=duration, restoration=restoration))
                first_stop = bool(method in ('scarcity_ppo', 'reservation_ppo', 'flat_stop_ppo')
                                  and action[0] == 0 and legal.any())
                counts['first_stop_with_legal'] += first_stop
                counts['ticks_residual_legal'] += bool(residual.any())
                counts['unserved_at_risk'] += len(risk)
                counts['conflicts'] += coordination['conflicts']
                counts['rejected_proposals'] += coordination['rejected_proposals']
                offset = len(records)
                obs, _, done, info = env.step(action)
                # Tick47 also advances the deadline tail: inspect the policy record,
                # not the environment's final state, for this tick's failure change.
                record = records[offset]
                assert record['phase'] == 'policy' and record['tick'] == tick
                following = json.loads(record['following'])
                newly_failed = {m for m in risk if following['failed'][m]}
                counts['failed_unserved_at_risk'] += len(newly_failed)
                journals['decisions'].add(dict(tags, tick=tick,
                    submitted=json.dumps(int(action) if np.isscalar(action) else action.tolist()),
                    executed=json.dumps(pairs), legal_edges=int(legal.sum()),
                    residual_legal_edges=int(residual.sum()), first_stop_with_legal=first_stop,
                    unserved_at_risk=json.dumps(sorted(risk)),
                    failed_unserved_at_risk=json.dumps(sorted(newly_failed))))
            assert done and len(records) == 60
            for record in records:
                raw = json.loads(record['state'])
                state = DispatchState(**{k: tuple(v) for k, v in raw.items()})
                following, cost, _ = transition(env.config, state,
                    tuple(tuple(p) for p in json.loads(record['action'])),
                    tuple(json.loads(record['events'])))
                assert json.dumps(asdict(following), sort_keys=True) == json.dumps(json.loads(record['following']), sort_keys=True)
                assert abs(cost - record['cost']) < 1e-9
                replays += 1
            assert model.algorithm == method and x.learner.parameter_digest(model) == before_digest
            metrics = info['episode_metrics']
            expected = original[4, method, train, choice['steps'], seed]
            for key, value in metrics.items():
                if isinstance(value, str):
                    assert expected[key] == value, key
                else:
                    assert abs(float(expected[key]) - float(value)) < 1e-8, key
            starts = counts['preventive_starts'] + counts['corrective_starts']
            row = dict(tags, **counts, extended_cost=metrics['extended_cost'],
                       availability=metrics['availability'], pending=metrics['terminal_pending'],
                       policy_failures=metrics['policy_failures'],
                       mean_start_duration=service_sum / max(1, starts),
                       mean_start_restoration=restoration_sum / max(1, starts),
                       mean_start_age=age_sum / max(1, starts), checkpoint_sha256=x.sha(path))
            journals['episodes'].add(row)
            episodes.append(row)
        assert len(episodes) == len(jobs) and replays == 60 * len(jobs)
        aggregate = {}
        for method in methods:
            panel = [r for r in episodes if r['method'] == method]
            keys = [*counts, 'extended_cost', 'availability', 'pending', 'policy_failures',
                    'mean_start_duration', 'mean_start_restoration', 'mean_start_age']
            aggregate[method] = {k: float(np.mean([r[k] for r in panel])) for k in keys}
        x.write_json(out / 'aggregate.json', aggregate)
        result.update(status='COMPLETED', episodes=len(episodes), independent_physical_ticks=replays,
                      episode_metric_reconciliation='PASS', frozen_parameter_check='PASS',
                      elapsed_seconds=time.monotonic() - started)
    except Exception as exc:
        result.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for journal in journals.values():
            journal.close()
        result['output_bindings'] = {p.name: dict(sha256=x.sha(p), bytes=p.stat().st_size)
                                     for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}
        x.write_json(out / 'manifest.json', result)
    print(out)


if __name__ == '__main__':
    main()
