"""Read-only selected trajectories on the opened Exchange PPO pilot DEV panel."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import technician_exchange_v2_experiment as x
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import transition
from ht_pdm_fjsp.technician_exchange_v2_env import decode_trace


COORDINATED_TRACE_ARMS = ('exchange_ppo', 'unrestricted_exchange_ppo', 'inverse_mask_ppo', 'self_repair_ppo', 'construct_only_ppo')
TRACE_FIELDS = ('trace_net_changed', 'trace_exchange_net_unchanged',
                'trace_cardinality_difference', 'trace_duration_difference',
                'trace_restoration_difference', 'trace_risk_unserved_difference')


def trace_summary(mask, action, arm, config, state):
    if arm not in COORDINATED_TRACE_ARMS:
        return None
    n, k = mask.shape
    construction_trace = np.concatenate([np.asarray(action)[:k], np.zeros(k, dtype=np.int64)])
    construction, _ = decode_trace(mask, construction_trace, arm)
    final, counters = decode_trace(mask, action, arm)
    servicing = {m for m in state.assigned if m >= 0}
    risk = {m for m in range(n) if not state.failed[m] and
            state.ages[m] >= config.failure_age-1 and m not in servicing}
    def descriptors(plan):
        pairs = [(int(token)-1, j) for j, token in enumerate(plan) if token]
        covered = {m for m, _ in pairs}
        return (len(pairs), sum(config.service_time[m][j] for m,j in pairs),
                sum(config.restoration[m][j] for m,j in pairs), len(risk-covered))
    initial, following = descriptors(construction), descriptors(final)
    changed = int(not np.array_equal(construction, final))
    return dict(construction_plan=construction.tolist(), final_plan=final.tolist(),
        trace_net_changed=changed,
        trace_exchange_net_unchanged=int(counters['exchanges'] > 0 and not changed),
        trace_cardinality_difference=following[0]-initial[0],
        trace_duration_difference=following[1]-initial[1],
        trace_restoration_difference=following[2]-initial[2],
        trace_risk_unserved_difference=following[3]-initial[3])


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
    assert manifest['profile'] == 'pilot' and not manifest['test_opened']
    for name, binding in manifest['output_bindings'].items():
        path = source / name
        assert path.stat().st_size == binding['bytes'] and x.sha(path) == binding['sha256'], name
    for path in Path(x.__file__).parent.glob('*.py'):
        assert x.sha(path) == manifest['source_seal']['source_snapshot/' + path.name], path
    assert x.sha(x.PLAN) == manifest['source_seal']['protocol.md']
    assert x.sha(x.REGISTRY) == manifest['source_seal']['seed_registry.json']
    config = json.loads((source / 'resolved_config.json').read_text())
    assert config == x.settings('pilot')
    assert not config['test_seeds'] and not config['teacher_assistance']
    assert config['pretraining_steps'] == 0
    choices = json.loads((source / 'checkpoint_choices.json').read_text())
    for name,digest in json.loads((source/'seed_audit.json').read_text())['historical_registry_hashes'].items():
        assert x.sha(x.ROOT/'configs'/name) == digest
    with (source / 'development_episodes.csv').open() as stream:
        original = {(int(r['n']), r['method'], int(r['train_seed']),
                     int(r['checkpoint_steps']), int(r['seed'])): r
                    for r in csv.DictReader(stream)}
    with (source / 'development_reference.csv').open() as stream:
        for row in csv.DictReader(stream):
            original[int(row['n']), 'deadline', -1, 0, int(row['seed'])] = row
    trains = config['train_seeds'][:1] if args.profile == 'smoke' else config['train_seeds']
    seeds = config['development_seeds'][:1] if args.profile == 'smoke' else config['development_seeds']
    started = time.monotonic()
    git = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=x.ROOT,text=True).strip(),
               dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=x.ROOT,text=True).strip()))
    if args.profile == 'diagnostic' and git['dirty']:
        raise RuntimeError('commit verified diagnostic before full analysis')
    out = args.output_root / (x.VERSION + '_allocation_diagnostic_' + args.profile + '_' +
                             datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    plan = Path(__file__).resolve().parents[1] / 'docs/technician_exchange_ppo_v2_diagnostic_plan.md'
    shutil.copyfile(__file__, out / 'diagnostic_source.py')
    shutil.copyfile(plan, out / 'diagnostic_plan.md')
    result = dict(status='RUNNING', classification='EXPLORATORY_DEV_ONLY_NO_TRAINING',
                  profile=args.profile, source=str(source), source_manifest_sha256=x.sha(source / 'manifest.json'),
                  script_sha256=x.sha(Path(__file__)), plan_sha256=x.sha(plan),
                  train_seeds=trains, development_seeds=seeds, scientific_test_opened=False,
                  optimizer_steps=0, parameter_updates=0, teacher_assistance=False,
                  verified_input_bindings=len(manifest['output_bindings']))
    result['git'] = git
    input_seals = dict(manifest['source_seal'])
    x.write_json(out / 'manifest.json', result)
    journals = {name: x.Journal(out / (name + '.csv')) for name in ('decisions', 'starts', 'physical', 'episodes')}
    episodes = []
    populations = {}
    replays = 0
    torch.set_num_threads(1)

    def timeout():
        if time.monotonic() - started > 1200:
            raise TimeoutError('registered diagnostic hard stop')

    try:
        jobs = [(n, method, train, seed) for n in config['scales'] for method in x.arms(n) for train in trains for seed in seeds]
        jobs += [(n, 'deadline', -1, seed) for n in config['scales'] for seed in seeds]
        for n, method, train, seed in tqdm(jobs, desc='Frozen pilot DEV trajectories'):
            timeout()
            choice = choices.get(f'{n}:{method}:{train}', {'steps': 0})
            path = source / choice['path'] if method != 'deadline' else None
            if path is not None:
                assert x.sha(path) == choice['sha256']
            model = x.load_model(path, n, config, train, method, choice['steps']) if path is not None else None
            before_digest = x.learner.parameter_digest(model) if model is not None else None
            tags = dict(n=n, method=method, train_seed=train, development_seed=seed,
                        checkpoint_steps=choice['steps'])
            records = []

            class PhysicalLog:
                def add(self, record):
                    records.append(record)
                    journals['physical'].add(dict(tags, **record))

            env = x.ResourceEnv(seed, n, method, journal=PhysicalLog())
            obs = env.reset()
            assert populations.setdefault((n, seed), env.population) == env.population
            counts = dict(first_wait_with_legal=0, ticks_residual_legal=0,
                          unserved_at_risk=0, failed_unserved_at_risk=0, failed_risk_with_residual_legal=0, long_service_starts=0, short_service_starts=0,
                          preventive_starts=0, corrective_starts=0,
                          conflicts=0, rejected_proposals=0)
            trace_totals = dict.fromkeys(TRACE_FIELDS, 0.)
            trace_ticks = policy_service = policy_failed_waiting = policy_unavailable = 0
            service_sum = restoration_sum = age_sum = 0.
            for tick in range(48):
                timeout()
                state = env.env.state
                legal = obs['pair_mask'].copy()
                action = model.select(obs) if model is not None else None
                pairs, coordination = env.resolve(action)
                trace = trace_summary(legal, action, method, env.config, state)
                if trace is not None:
                    trace_ticks += 1
                    for key in TRACE_FIELDS: trace_totals[key] += trace[key]
                    final_pairs = tuple(sorted((int(token)-1,j) for j,token in enumerate(trace['final_plan']) if token))
                    assert final_pairs == pairs
                selected = {m for m, _ in pairs}
                servicing = {m for m in state.assigned if m >= 0}
                risk = {m for m in range(n) if not state.failed[m] and
                        state.ages[m] >= env.config.failure_age - 1 and
                        m not in servicing and m not in selected}
                residual = legal.copy()
                for m, j in pairs:
                    residual[m, :] = False
                    residual[:, j] = False
                    counts['corrective_starts' if state.failed[m] else 'preventive_starts'] += 1
                    counts['long_service_starts' if env.config.service_time[m][j] > 1 else 'short_service_starts'] += 1
                    duration = env.config.service_time[m][j]
                    restoration = env.config.restoration[m][j]
                    service_sum += duration
                    restoration_sum += restoration
                    age_sum += state.ages[m]
                    journals['starts'].add(dict(tags, tick=tick, machine=m, technician=j,
                        age=state.ages[m], failed=state.failed[m], duration=duration, restoration=restoration))
                first_wait = bool(method in x.GRAPH_ARMS and action[0] == 0 and legal[:, 0].any())
                counts['first_wait_with_legal'] += first_wait
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
                assert tuple(map(tuple,json.loads(record['action']))) == pairs
                assert info['coordination'] == coordination
                following = json.loads(record['following'])
                _, _, policy_components = transition(env.config, state, pairs, tuple(json.loads(record['events'])))
                policy_service += policy_components['service_ticks']
                policy_failed_waiting += policy_components['failed_waiting_ticks']
                policy_unavailable += policy_components['unavailable_ticks']
                newly_failed = {m for m in risk if following['failed'][m]}
                counts['failed_unserved_at_risk'] += len(newly_failed)
                counts['failed_risk_with_residual_legal'] += sum(bool(residual[m].any()) for m in newly_failed)
                journals['decisions'].add(dict(tags, tick=tick,
                    submitted=json.dumps(None if action is None else int(action) if np.isscalar(action) else action.tolist()),
                    executed=json.dumps(pairs), legal_edges=int(legal.sum()),
                    residual_legal_edges=int(residual.sum()), first_wait_with_legal=first_wait,
                    unserved_at_risk=json.dumps(sorted(risk)),
                    failed_unserved_at_risk=json.dumps(sorted(newly_failed)),
                    trace_applicable=trace is not None,
                    construction_plan=json.dumps(trace['construction_plan']) if trace is not None else '',
                    final_plan=json.dumps(trace['final_plan']) if trace is not None else '',
                    **{key: trace[key] if trace is not None else None for key in TRACE_FIELDS}))
            assert done and len(records) == 60
            physical_totals = {}
            for record in records:
                raw = json.loads(record['state'])
                state = DispatchState(**{k: tuple(v) for k, v in raw.items()})
                following, cost, components = transition(env.config, state,
                    tuple(tuple(p) for p in json.loads(record['action'])),
                    tuple(json.loads(record['events'])))
                assert json.dumps(asdict(following), sort_keys=True) == json.dumps(json.loads(record['following']), sort_keys=True)
                assert abs(cost - record['cost']) < 1e-9
                for key,value in components.items():physical_totals[key]=physical_totals.get(key,0.)+value
                replays += 1
            if model is not None:
                assert model.algorithm == method and x.learner.parameter_digest(model) == before_digest
            metrics = info['episode_metrics']
            assert all(abs(value-metrics[key])<1e-8 for key,value in physical_totals.items())
            expected = original[n, method, train, choice['steps'], seed]
            for key, value in metrics.items():
                if isinstance(value, str):
                    assert expected[key] == value, key
                else:
                    assert abs(float(expected[key]) - float(value)) < 1e-8, key
            starts = counts['preventive_starts'] + counts['corrective_starts']
            row = dict(tags, **counts, trace_ticks=trace_ticks,
                       **{key: trace_totals[key] if trace_ticks else None for key in TRACE_FIELDS},
                       policy_service_ticks=policy_service, policy_failed_waiting_ticks=policy_failed_waiting,
                       policy_unavailable_ticks=policy_unavailable, extended_cost=metrics['extended_cost'],
                       availability=metrics['availability'], pending=metrics['terminal_pending'],
                       policy_failures=metrics['policy_failures'],
                       mean_start_duration=service_sum / max(1, starts),
                       mean_start_restoration=restoration_sum / max(1, starts),
                       mean_start_age=age_sum / max(1, starts), checkpoint_sha256=x.sha(path) if path is not None else None)
            journals['episodes'].add(row)
            episodes.append(row)
        assert len(episodes) == len(jobs) and replays == 60 * len(jobs)
        assert {(r['n'],r['method'],r['train_seed'],r['development_seed']) for r in episodes} == set(jobs)
        aggregate = {}
        for n, method in sorted({(r['n'], r['method']) for r in episodes}):
            panel = [r for r in episodes if r['n'] == n and r['method'] == method]
            keys = [*counts, 'extended_cost', 'availability', 'pending', 'policy_failures',
                    'mean_start_duration', 'mean_start_restoration', 'mean_start_age']
            keys += ['policy_service_ticks','policy_failed_waiting_ticks','policy_unavailable_ticks','trace_ticks']
            aggregate[f'N{n}/{method}'] = {k: float(np.mean([r[k] for r in panel])) for k in keys}
            aggregate[f'N{n}/{method}'].update({key:float(np.mean([r[key] for r in panel])) if panel[0]['trace_ticks'] else None for key in TRACE_FIELDS})
        x.write_json(out / 'aggregate.json', aggregate)
        assert x.sha(source/'manifest.json') == result['source_manifest_sha256']
        for name,binding in manifest['output_bindings'].items():
            assert x.sha(source/name) == binding['sha256'] and (source/name).stat().st_size == binding['bytes']
        assert x.sha(Path(__file__)) == result['script_sha256'] and x.sha(plan) == result['plan_sha256']
        for path in Path(x.__file__).parent.glob('*.py'):
            assert x.sha(path) == input_seals['source_snapshot/'+path.name]
        assert x.sha(x.PLAN) == input_seals['protocol.md'] and x.sha(x.REGISTRY) == input_seals['seed_registry.json']
        for name,digest in json.loads((source/'seed_audit.json').read_text())['historical_registry_hashes'].items():
            assert x.sha(x.ROOT/'configs'/name) == digest
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
