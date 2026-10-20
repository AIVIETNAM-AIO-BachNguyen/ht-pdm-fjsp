"""Fixed K=0/1/2/4 oracle planner-correction curve on V12 opened DEV."""
from __future__ import annotations

import argparse
import csv
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

from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, FrameworkEnv, configs
from ht_pdm_fjsp.maintenance_framework_v7_campaign import V7RuleController

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'research_result'))
import diagnose_v12_sparse_planner as previous

V12 = previous.V12
PRIOR = ROOT / 'research_result/v12_sparse_planner_dev_20261020T135214Z'
PLAN = ROOT / 'research_result/v12_planner_budget_curve_plan_vi.md'
VERSION = 'v12_planner_budget_curve_dev_v1'
TARGET = previous.TARGET
TRAIN = previous.TRAIN
DEV = previous.DEV
ACTIONS = previous.ACTIONS
METRICS = previous.METRICS
BUDGETS = (0, 1, 2, 4)
BOOTSTRAP = 71590000
SOURCES = tuple(dict.fromkeys((*previous.SOURCES,
    ROOT / 'research_result/diagnose_v12_planner_budget.py', PLAN)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_row(path, item):
    path = Path(path); first = not path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(item))
        if first: writer.writeheader()
        writer.writerow(item)


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def corrected_choice(actor_action, planner_action, used, budget):
    if used >= budget:
        return actor_action, used, False
    if planner_action == actor_action:
        return actor_action, used, False
    return planner_action, used + 1, True


def replay(case, physical, actor, forecast_seed, budget, deadline):
    if budget not in BUDGETS:
        raise ValueError(budget)
    cfg = configs()[case]; env = FrameworkEnv(cfg); env.reset(seed=physical)
    planner = V7RuleController('rollout', forecast_seed, deadline)
    decisions = []; corrections = 0; queries = 0; actor_seconds = 0.; planner_seconds = 0.
    for tick in range(HORIZON):
        deadline(); state = env.inner.state
        start = time.perf_counter()
        requests, _, legal, observation = previous.legal_present(cfg, state, tick)
        actor_action = previous.policy_action(actor, observation, legal)
        actor_seconds += time.perf_counter() - start
        planner_action = -1; queried = 0; corrected = 0; choice = actor_action
        if corrections < budget:
            start = time.perf_counter()
            supplied, matching = planner.choose(cfg, state, tick, case=case, seed=physical)
            planner_seconds += time.perf_counter() - start
            if supplied != requests:
                raise AssertionError('requester mismatch')
            planner_action = ACTIONS.index(matching)
            queried = 1; queries += 1
            choice, corrections, changed = corrected_choice(actor_action, planner_action,
                                                             corrections, budget)
            corrected = int(changed)
        if choice not in legal:
            raise AssertionError('illegal correction')
        decisions.append(dict(tick=tick, actor_action=actor_action, planner_action=planner_action,
                              action=choice, queried=queried, corrected=corrected,
                              cumulative_queries=queries, cumulative_corrections=corrections,
                              cumulative_forecast_transitions=planner.forecast_transitions))
        env.step(choice)
    return env.episode_metrics(), decisions, dict(planner_calls=queries, corrections=corrections,
        forecast_transitions=planner.forecast_transitions, actor_selection_seconds=actor_seconds,
        planner_selection_seconds=planner_seconds,
        total_selection_seconds=actor_seconds+planner_seconds)


def summarize(records, source_rollout):
    trains = tuple(sorted({r['train_seed'] for r in records}))
    physicals = tuple(sorted({r['physical_seed'] for r in records}))
    result = {'by_case': {}, 'gates': {}}
    for case in TARGET:
        result['by_case'][case] = {}
        for budget in BUDGETS:
            subset = [r for r in records if r['case'] == case and r['budget'] == budget]
            cost = float(np.mean([r['objective'] for r in subset]))
            rollout = float(np.mean([source_rollout[(case, r['physical_seed'])]['objective']
                                     for r in subset]))
            actor = float(np.mean([r['objective'] for r in records if r['case'] == case and r['budget'] == 0]))
            gap = actor-rollout
            by_seed = {str(train): float(np.mean([r['objective'] for r in subset
                       if r['train_seed'] == train])) for train in trains}
            rollout_by_seed = {str(train): float(np.mean([source_rollout[(case, r['physical_seed'])]['objective']
                       for r in subset if r['train_seed'] == train])) for train in trains}
            paired = np.asarray([[next(r['objective'] for r in subset if r['train_seed'] == train and
                                   r['physical_seed'] == physical) - source_rollout[(case, physical)]['objective']
                                   for physical in physicals] for train in trains])
            rng = np.random.default_rng(BOOTSTRAP + TARGET.index(case)*10 + budget)
            ti = rng.integers(0, len(trains), size=(20000, len(trains)))
            pi = rng.integers(0, len(physicals), size=(20000, len(physicals)))
            ci = np.quantile(paired[ti[:, :, None], pi[:, None, :]].mean(axis=(1,2)), [.025,.975]).tolist()
            full_rollout_seconds = float(np.mean([source_rollout[(case,r['physical_seed'])]['selection_seconds']
                                                  for r in subset]))
            result['by_case'][case][str(budget)] = dict(episodes=len(subset), mean_cost=cost,
                rollout_cost=rollout, actor_cost=actor, cost_gap_to_rollout=cost-rollout,
                gap_recovery=(actor-cost)/gap if gap > 0 else None,
                cost_gap_ci95=ci, training_seed_cost=by_seed,
                training_seed_rollout_cost=rollout_by_seed,
                mean_planner_calls=float(np.mean([r['planner_calls'] for r in subset])),
                mean_corrections=float(np.mean([r['corrections'] for r in subset])),
                mean_forecast_transitions=float(np.mean([r['forecast_transitions'] for r in subset])),
                mean_total_selection_seconds=float(np.mean([r['total_selection_seconds'] for r in subset])),
                full_rollout_selection_seconds=full_rollout_seconds,
                mean_metrics={metric:float(np.mean([r[metric] for r in subset])) for metric in METRICS})
    if len(trains) != len(TRAIN) or len(physicals) != len(DEV):
        return result
    for budget in BUDGETS[1:]:
        cost_ok = all((result['by_case'][case][str(budget)]['mean_cost'] <=
                       1.01*result['by_case'][case][str(budget)]['rollout_cost'] and
                       sum(result['by_case'][case][str(budget)]['training_seed_cost'][str(seed)] <=
                           1.01*result['by_case'][case][str(budget)]['training_seed_rollout_cost'][str(seed)]
                           for seed in TRAIN) >= 4) for case in TARGET)
        runtime_ok = all(result['by_case'][case][str(budget)]['mean_total_selection_seconds'] <=
                         .1*result['by_case'][case][str(budget)]['full_rollout_selection_seconds']
                         for case in TARGET)
        result['gates'][str(budget)] = dict(cost_headroom=bool(cost_ok),
                                            direct_runtime_headroom=bool(runtime_ok))
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
        if time.monotonic()-start >= limit:
            raise TimeoutError('planner budget timeout')
    rel = [str(p.relative_to(ROOT)) for p in SOURCES]
    dirty = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *rel], cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', *rel], cwd=ROOT, text=True).splitlines()
    if profile == 'dev' and (dirty or untracked):
        raise RuntimeError('uncommitted diagnostic source')
    source_manifest, models, hashes, refs = previous.load_inputs()
    prior_manifest = json.loads((PRIOR / 'manifest.json').read_text())
    if prior_manifest['status'] != 'COMPLETED':
        raise RuntimeError('prior intervention incomplete')
    prior_episodes = {(r['case'], int(r['train_seed']), int(r['physical_seed']), r['variant']): r
                      for r in previous.csv_rows(PRIOR / 'episodes.csv')}
    trains = TRAIN if profile == 'dev' else TRAIN[:1]
    physicals = DEV if profile == 'dev' else DEV[:1]
    forecast_seed = source_manifest['registry']['forecast_seed']
    manifest = dict(status='RUNNING', phase='preflight', profile=profile, version=VERSION,
        started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        invocation=[sys.executable, *sys.argv], runtime=dict(python=platform.python_version(),
        platform=platform.platform()), timeout_seconds=limit, source_hashes={str(p.relative_to(ROOT)):sha(p)
        for p in SOURCES}, v12_manifest_sha256=sha(V12 / 'manifest.json'),
        v12_selection_sha256=sha(V12 / 'selection.json'),
        v12_episodes_sha256=sha(V12 / 'episodes.csv'),
        prior_manifest_sha256=sha(PRIOR / 'manifest.json'),
        prior_episodes_sha256=sha(PRIOR / 'episodes.csv'), selected_checkpoint_hashes=hashes,
        forecast_seed=forecast_seed, bootstrap_seed=BOOTSTRAP, budgets=list(BUDGETS),
        cases=list(TARGET), training_seeds=list(trains), physical_seeds=list(physicals))
    save(output / 'manifest.json', manifest)
    shutil.copy2(PLAN, output / 'protocol.md')
    try:
        manifest['phase'] = 'budget_replays'; save(output / 'manifest.json', manifest)
        records = []
        grid = [(case, train, physical) for case in TARGET for train in trains for physical in physicals]
        for case, train, physical in tqdm(grid, desc='Budgeted V12 planner corrections', unit='pair'):
            deadline()
            for budget in BUDGETS:
                metrics, decisions, compute = replay(case, physical, models[train], forecast_seed,
                                                     budget, deadline)
                if budget in (0,1):
                    prior = prior_episodes[(case, train, physical, 'actor' if budget == 0 else 'one')]
                    for metric in METRICS:
                        if not np.isclose(float(metrics[metric]), float(prior[metric]), rtol=0, atol=1e-8):
                            raise AssertionError(f'prior K{budget} mismatch {case}/{train}/{physical}/{metric}')
                if budget == 0:
                    original = refs[(case, 'reliability_credit', str(train), physical)]
                    for metric in METRICS:
                        if not np.isclose(float(metrics[metric]), float(original[metric]), rtol=0, atol=1e-8):
                            raise AssertionError('V12 selected actor mismatch')
                record = dict(case=case, train_seed=train, physical_seed=physical, budget=budget,
                              **compute, **{key: metrics[key] for key in METRICS})
                records.append(record); add_row(output / 'episodes.partial.csv', record)
                for item in decisions:
                    add_row(output / 'actions.partial.csv', dict(case=case, train_seed=train,
                        physical_seed=physical, budget=budget, **item))
        for name in ('episodes', 'actions'):
            shutil.copy2(output / f'{name}.partial.csv', output / f'{name}.csv')
        source_rollout = {(case, physical): {key:float(refs[(case, 'rollout', '', physical)][key])
                          for key in (*METRICS, 'selection_seconds')}
                          for case in TARGET for physical in physicals}
        save(output / 'summary.json', summarize(records, source_rollout))
        if manifest['source_hashes'] != {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}:
            raise RuntimeError('source changed during diagnostic')
        manifest.update(status='COMPLETED', phase='done', finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-start,
            counts=dict(pairs=len(grid), episodes=len(records), decisions=len(records)*HORIZON),
            artifact_hashes={p.name:sha(p) for p in output.iterdir()
                             if p.is_file() and p.name != 'manifest.json'})
        save(output / 'manifest.json', manifest)
        return output
    except Exception as exc:
        manifest.update(status='INCOMPLETE' if isinstance(exc, TimeoutError) else 'FAILED',
            phase='stopped', error=repr(exc), elapsed_seconds=time.monotonic()-start)
        save(output / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('smoke', 'dev'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.profile, args.output_dir))
