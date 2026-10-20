"""DEV-only first-divergence one-action and four-tick planner intervention."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, FrameworkEnv, catalogue, configs
from ht_pdm_fjsp.maintenance_framework_v7_campaign import V7RuleController
from ht_pdm_fjsp.maintenance_framework_v12_campaign import load_checkpoint, legal_present, policy_action

ROOT = Path(__file__).resolve().parent.parent
V12 = ROOT / 'research_result/framework_v12_dev_20261020T125625Z'
PLAN = ROOT / 'research_result/v12_sparse_planner_intervention_plan_vi.md'
VERSION = 'v12_sparse_planner_intervention_dev_v1'
TARGET = ('interchangeable_scale1', 'reciprocal_scale2')
TRAIN = tuple(range(71500000, 71500005))
DEV = tuple(range(71510000, 71510016))
ACTIONS = catalogue()
METRICS = ('objective', 'maintenance_cost', 'unavailability_cost', 'failure_cost',
           'tail_cost', 'H_availability', 'H_failures', 'queued_cm_ticks',
           'queued_pm_ticks', 'H_pending', 'end_pending', 'service_starts',
           'requests_created', 'requests_completed')
SOURCES = tuple(ROOT / path for path in (
    'research_result/diagnose_v12_sparse_planner.py',
    'research_result/v12_sparse_planner_intervention_plan_vi.md',
    'src/ht_pdm_fjsp/maintenance_framework_v1.py',
    'src/ht_pdm_fjsp/maintenance_framework_v7_campaign.py',
    'src/ht_pdm_fjsp/maintenance_framework_v12_campaign.py',
    'src/ht_pdm_fjsp/maintenance_requests.py',
    'src/ht_pdm_fjsp/maintenance_request_policies.py',
    'src/ht_pdm_fjsp/maintenance_dispatch.py',
))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def csv_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def add_row(path, value):
    path = Path(path)
    first = not path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(value))
        if first:
            writer.writeheader()
        writer.writerow(value)


def category(actor_action, planner_action):
    actor_edges, planner_edges = ACTIONS[actor_action], ACTIONS[planner_action]
    if actor_edges == planner_edges:
        return 'match'
    if not actor_edges:
        return 'actor_wait_planner_service'
    if not planner_edges:
        return 'actor_service_planner_wait'
    if len(actor_edges) < len(planner_edges):
        return 'actor_lower_cardinality'
    if len(actor_edges) > len(planner_edges):
        return 'actor_higher_cardinality'
    if {m for m, _ in actor_edges} != {m for m, _ in planner_edges}:
        return 'different_machine_set'
    return 'different_technician_assignment'


def forced_actions(first, planner_action, label):
    if label not in ('actor', 'one', 'wait4'):
        raise ValueError(label)
    if first is None or label == 'actor':
        return {}
    if label == 'one' or planner_action != 0:
        return {first: planner_action}
    return {tick: 0 for tick in range(first, min(HORIZON, first + 4))}


def load_inputs():
    manifest = json.loads((V12 / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETED' or manifest['profile'] != 'dev':
        raise RuntimeError('V12 source incomplete')
    selection = json.loads((V12 / 'selection.json').read_text())
    if sha(V12 / 'selection.json') != manifest['selection_sha256']:
        raise RuntimeError('V12 selection changed')
    models = {}; hashes = {}
    for train in TRAIN:
        selected = selection['reliability_credit'][str(train)]
        path = V12 / selected['checkpoint']
        if sha(path) != selected['sha256']:
            raise RuntimeError('V12 checkpoint changed')
        models[train] = load_checkpoint(path)
        hashes[str(train)] = selected['sha256']
    references = {(r['case'], r['controller'], r['train_seed'], int(r['seed'])): r
                  for r in csv_rows(V12 / 'episodes.csv')
                  if r['case'] in TARGET and r['controller'] in ('reliability_credit', 'rollout')}
    expected = len(TARGET) * len(DEV) * (len(TRAIN) + 1)
    if len(references) != expected:
        raise RuntimeError('V12 reference episode grid incomplete')
    return manifest, models, hashes, references


def episode(case, physical, actor, forecast_seed, deadline, forced=None):
    cfg = configs()[case]
    env = FrameworkEnv(cfg)
    env.reset(seed=physical)
    planner = V7RuleController('rollout', forecast_seed, deadline) if actor is None else None
    forced = forced or {}
    actions = []; states = []; forecast_counts = []
    for tick in range(HORIZON):
        deadline()
        state = env.inner.state
        states.append(asdict(state))
        submissions, _, legal, observation = legal_present(cfg, state, tick)
        if actor is None:
            supplied, matching = planner.choose(cfg, state, tick, case=case, seed=physical)
            if supplied != submissions:
                raise AssertionError('planner requester mismatch')
            choice = ACTIONS.index(matching)
        else:
            choice = forced.get(tick, policy_action(actor, observation, legal))
        if choice not in legal:
            raise AssertionError('illegal intervention')
        actions.append(choice)
        forecast_counts.append(planner.forecast_transitions if planner is not None else 0)
        _, _, done, _, _ = env.step(choice)
    if not done:
        raise AssertionError('unterminated episode')
    return actions, states, env.episode_metrics(), forecast_counts


def summarize(records):
    result = {}
    for case in TARGET:
        for train in TRAIN:
            subset = [r for r in records if r['case'] == case and r['train_seed'] == train]
            if not subset:
                continue
            key = f'{case}/{train}'
            means = {name: float(np.mean([r[f'{name}_cost'] for r in subset]))
                     for name in ('actor', 'one', 'wait4', 'rollout')}
            gap = means['actor'] - means['rollout']
            result[key] = dict(pairs=len(subset), mean_cost=means, actor_rollout_gap=gap,
                one_recovery=(means['actor'] - means['one']) / gap if gap > 0 else None,
                wait4_recovery=(means['actor'] - means['wait4']) / gap if gap > 0 else None,
                first_tick_mean=float(np.mean([r['first_tick'] for r in subset if r['first_tick'] >= 0]))
                    if any(r['first_tick'] >= 0 for r in subset) else None,
                divergence_categories=dict(Counter(r['category'] for r in subset)),
                planner_wait_divergences=sum(r['planner_action'] == 0 and r['first_tick'] >= 0
                                             for r in subset),
                oracle_forecast_transitions=sum(r['oracle_forecast_transitions'] for r in subset))
    if len(result) == len(TARGET) * len(TRAIN):
        overall = {}
        for case in TARGET:
            cell = [r for r in records if r['case'] == case]
            means = {name: float(np.mean([r[f'{name}_cost'] for r in cell]))
                     for name in ('actor', 'one', 'wait4', 'rollout')}
            gap = means['actor'] - means['rollout']
            overall[case] = dict(pairs=len(cell), mean_cost=means,
                one_recovery=(means['actor']-means['one'])/gap if gap > 0 else None,
                wait4_recovery=(means['actor']-means['wait4'])/gap if gap > 0 else None)
        result['overall'] = overall
        one_ok = all(overall[case]['one_recovery'] is not None and
                     overall[case]['one_recovery'] >= .30 and
                     sum(result[f'{case}/{train}']['one_recovery'] is not None and
                         result[f'{case}/{train}']['one_recovery'] >= .30
                         for train in TRAIN) >= 4 for case in TARGET)
        reciprocal = overall['reciprocal_scale2']
        extra = (reciprocal['wait4_recovery'] - reciprocal['one_recovery']
                 if reciprocal['wait4_recovery'] is not None and
                    reciprocal['one_recovery'] is not None else None)
        interchangeable = overall['interchangeable_scale1']['mean_cost']
        result['gates'] = dict(one_action_supported=bool(one_ok),
            wait4_supported=bool(extra is not None and extra >= .20 and
                interchangeable['wait4']-interchangeable['one'] <= 5))
    return result


def run(profile, output):
    if profile not in ('smoke', 'dev'):
        raise ValueError(profile)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    start = time.monotonic(); limit = 600 if profile == 'smoke' else 1800
    def deadline():
        if time.monotonic() - start >= limit:
            raise TimeoutError('sparse planner diagnostic timeout')
    rel = [str(p.relative_to(ROOT)) for p in SOURCES]
    dirty = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *rel], cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', *rel], cwd=ROOT, text=True).splitlines()
    if profile == 'dev' and (dirty or untracked):
        raise RuntimeError('uncommitted diagnostic source')
    source, models, hashes, references = load_inputs()
    trains = TRAIN if profile == 'dev' else TRAIN[:1]
    physicals = DEV if profile == 'dev' else DEV[:1]
    forecast_seed = source['registry']['forecast_seed']
    manifest = dict(status='RUNNING', phase='preflight', profile=profile, version=VERSION,
        started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        invocation=[sys.executable, *sys.argv], runtime=dict(python=platform.python_version(),
        platform=platform.platform()), timeout_seconds=limit,
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
        v12_manifest_sha256=sha(V12 / 'manifest.json'),
        v12_selection_sha256=sha(V12 / 'selection.json'),
        v12_episodes_sha256=sha(V12 / 'episodes.csv'), selected_checkpoint_hashes=hashes,
        forecast_seed=forecast_seed, cases=list(TARGET), training_seeds=list(trains),
        physical_seeds=list(physicals))
    save(output / 'manifest.json', manifest)
    shutil.copy2(PLAN, output / 'protocol.md')
    try:
        rollouts = {}
        manifest['phase'] = 'rollout_replay'; save(output / 'manifest.json', manifest)
        for case in tqdm(TARGET, desc='Unique rollout trajectories', unit='case'):
            for physical in physicals:
                actions, states, metrics, counts = episode(case, physical, None, forecast_seed, deadline)
                original = references[(case, 'rollout', '', physical)]
                for metric in METRICS:
                    if not np.isclose(float(metrics[metric]), float(original[metric]), rtol=0, atol=1e-8):
                        raise AssertionError(f'rollout source {case}/{physical}/{metric}')
                rollouts[(case, physical)] = (actions, states, metrics, counts)
                add_row(output / 'rollout_episodes.partial.csv', dict(case=case, physical_seed=physical,
                    forecast_transitions=counts[-1], **{k: metrics[k] for k in METRICS}))
                for tick, action in enumerate(actions):
                    add_row(output / 'rollout_actions.partial.csv', dict(case=case,
                        physical_seed=physical, tick=tick, action=action,
                        cumulative_forecast_transitions=counts[tick]))
        manifest['phase'] = 'paired_intervention'; save(output / 'manifest.json', manifest)
        pair_records = []
        grid = [(case, train, physical) for case in TARGET for train in trains for physical in physicals]
        for case, train, physical in tqdm(grid, desc='V12 planner intervention', unit='pair'):
            deadline(); actor = models[train]
            rollout_actions, rollout_states, rollout_metrics, rollout_counts = rollouts[(case, physical)]
            actor_actions, actor_states, actor_metrics, _ = episode(case, physical, actor,
                forecast_seed, deadline)
            original = references[(case, 'reliability_credit', str(train), physical)]
            for metric in METRICS:
                if not np.isclose(float(actor_metrics[metric]), float(original[metric]), rtol=0, atol=1e-8):
                    raise AssertionError(f'V12 actor source {case}/{train}/{physical}/{metric}')
            first = next((tick for tick, (a, b) in enumerate(zip(actor_actions, rollout_actions))
                          if a != b), None)
            if first is not None and any(actor_states[tick] != rollout_states[tick]
                                         for tick in range(first + 1)):
                raise AssertionError('first-divergence state mismatch')
            planner_action = rollout_actions[first] if first is not None else -1
            actor_action = actor_actions[first] if first is not None else -1
            outcomes = {'actor': (actor_actions, actor_metrics)}
            for label in ('one', 'wait4'):
                forced = forced_actions(first, planner_action, label)
                actions, _, metrics, _ = episode(case, physical, actor, forecast_seed, deadline, forced)
                outcomes[label] = (actions, metrics)
            for label, (actions, metrics) in outcomes.items():
                add_row(output / 'episodes.partial.csv', dict(case=case, train_seed=train,
                    physical_seed=physical, variant=label, first_tick=-1 if first is None else first,
                    **{k: metrics[k] for k in METRICS}))
                for tick, action in enumerate(actions):
                    add_row(output / 'actions.partial.csv', dict(case=case, train_seed=train,
                        physical_seed=physical, variant=label, tick=tick, action=action,
                        forced=int(tick in forced_actions(first, planner_action, label))))
            row = dict(case=case, train_seed=train, physical_seed=physical,
                first_tick=-1 if first is None else first,
                category=category(actor_action, planner_action) if first is not None else 'match',
                actor_action=actor_action, planner_action=planner_action,
                oracle_forecast_transitions=rollout_counts[first] if first is not None else rollout_counts[-1],
                actor_cost=actor_metrics['objective'],
                one_cost=outcomes['one'][1]['objective'],
                wait4_cost=outcomes['wait4'][1]['objective'],
                rollout_cost=rollout_metrics['objective'])
            pair_records.append(row); add_row(output / 'divergences.partial.csv', row)
        for name in ('rollout_episodes', 'rollout_actions', 'episodes', 'actions', 'divergences'):
            shutil.copy2(output / f'{name}.partial.csv', output / f'{name}.csv')
        save(output / 'summary.json', summarize(pair_records))
        if manifest['source_hashes'] != {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}:
            raise RuntimeError('source changed during diagnostic')
        manifest.update(status='COMPLETED', phase='done', finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - start,
            counts=dict(unique_rollout_episodes=len(TARGET)*len(physicals), pairs=len(grid),
                        actor_intervention_episodes=3*len(grid), actor_intervention_decisions=3*len(grid)*HORIZON),
            artifact_hashes={p.name: sha(p) for p in output.iterdir()
                             if p.is_file() and p.name != 'manifest.json'})
        save(output / 'manifest.json', manifest)
        return output
    except Exception as exc:
        manifest.update(status='INCOMPLETE' if isinstance(exc, TimeoutError) else 'FAILED',
            phase='stopped', error=repr(exc), elapsed_seconds=time.monotonic() - start)
        save(output / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('smoke', 'dev'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.profile, args.output_dir))
