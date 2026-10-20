"""Independent source/action/metric audit of V12 sparse-planner intervention."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON, FrameworkEnv, catalogue, configs, vector_observation
from ht_pdm_fjsp.maintenance_framework_v6_policy import EdgeSharingActor
from ht_pdm_fjsp.maintenance_framework_v7_campaign import V7RuleController
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests

ROOT = Path(__file__).resolve().parent.parent
V12 = ROOT / 'research_result/framework_v12_dev_20261020T125625Z'
ACTIONS = catalogue()
CASES = ('interchangeable_scale1', 'reciprocal_scale2')
TRAINS = range(71500000, 71500005)
PHYSICALS = range(71510000, 71510016)
METRICS = ('objective', 'maintenance_cost', 'unavailability_cost', 'failure_cost',
           'tail_cost', 'H_availability', 'H_failures', 'queued_cm_ticks',
           'queued_pm_ticks', 'H_pending', 'end_pending', 'service_starts',
           'requests_created', 'requests_completed')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def close(a, b, label):
    if not np.isclose(float(a), float(b), atol=1e-7, rtol=0):
        raise AssertionError(f'{label}: {a} != {b}')


def actor_at(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    actor = EdgeSharingActor('shared_edge')
    actor.load_state_dict(payload['state_dict'])
    actor.eval()
    return actor


def choice(actor, cfg, state, tick):
    requests = request_action(cfg, state, tick, 'threshold')
    admitted = submit_requests(cfg, state, tick, requests)
    legal = sorted(ACTIONS.index(edges) for edges in legal_matchings(cfg, admitted))
    observation = vector_observation(cfg, admitted, tick)
    with torch.no_grad():
        logits = actor(torch.tensor(observation).reshape(1, 56))[0]
        masked = torch.full_like(logits, -1e9)
        masked[legal] = logits[legal]
        action = int(np.argmax(torch.softmax(masked, dim=0).numpy()))
    return action, legal


def replay(case, physical, actor, planner_seed, override=None):
    cfg = configs()[case]
    env = FrameworkEnv(cfg)
    env.reset(seed=physical)
    planner = V7RuleController('rollout', planner_seed) if actor is None else None
    actions = []; states = []; counts = []
    override = override or {}
    for tick in range(HORIZON):
        states.append(env.inner.state)
        if actor is None:
            requests, matching = planner.choose(cfg, env.inner.state, tick, case=case, seed=physical)
            assert requests == request_action(cfg, env.inner.state, tick, 'threshold')
            action = ACTIONS.index(matching)
        else:
            actor_action, legal = choice(actor, cfg, env.inner.state, tick)
            action = override[tick] if tick in override else actor_action
            assert action in legal
        actions.append(action)
        counts.append(planner.forecast_transitions if planner is not None else 0)
        env.step(action)
    return actions, states, env.episode_metrics(), counts


def class_of(a, b):
    left, right = ACTIONS[a], ACTIONS[b]
    if not left and right:return 'actor_wait_planner_service'
    if left and not right:return 'actor_service_planner_wait'
    if len(left) < len(right):return 'actor_lower_cardinality'
    if len(left) > len(right):return 'actor_higher_cardinality'
    if {m for m, _ in left} != {m for m, _ in right}:return 'different_machine_set'
    return 'different_technician_assignment'


def audit(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED' and manifest['profile'] == 'dev'
    assert manifest['counts'] == {'unique_rollout_episodes': 32, 'pairs': 160,
                                  'actor_intervention_episodes': 480,
                                  'actor_intervention_decisions': 23040}
    assert manifest['v12_manifest_sha256'] == sha(V12 / 'manifest.json')
    assert manifest['v12_selection_sha256'] == sha(V12 / 'selection.json')
    assert manifest['v12_episodes_sha256'] == sha(V12 / 'episodes.csv')
    for name, digest in manifest['source_hashes'].items():
        assert sha(ROOT / name) == digest
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(run / name) == digest
    selection = json.loads((V12 / 'selection.json').read_text())
    models = {}
    for train in TRAINS:
        item = selection['reliability_credit'][str(train)]
        path = V12 / item['checkpoint']
        assert sha(path) == item['sha256'] == manifest['selected_checkpoint_hashes'][str(train)]
        models[train] = actor_at(path)
    originals = {(r['case'], r['controller'], r['train_seed'], int(r['seed'])): r
                 for r in rows(V12 / 'episodes.csv')
                 if r['case'] in CASES and r['controller'] in ('rollout', 'reliability_credit')}
    rollout_actions = {(r['case'], int(r['physical_seed']), int(r['tick'])): r
                       for r in rows(run / 'rollout_actions.csv')}
    rollout_episodes = {(r['case'], int(r['physical_seed'])): r
                        for r in rows(run / 'rollout_episodes.csv')}
    actions = {(r['case'], int(r['train_seed']), int(r['physical_seed']),
                r['variant'], int(r['tick'])): r for r in rows(run / 'actions.csv')}
    episodes = {(r['case'], int(r['train_seed']), int(r['physical_seed']), r['variant']): r
                for r in rows(run / 'episodes.csv')}
    divergence = {(r['case'], int(r['train_seed']), int(r['physical_seed'])): r
                  for r in rows(run / 'divergences.csv')}
    assert len(rollout_actions) == 32 * HORIZON and len(rollout_episodes) == 32
    assert len(actions) == 23040 and len(episodes) == 480 and len(divergence) == 160
    plan_paths = {}
    for case in tqdm(CASES, desc='Audit rollout oracle trajectories', unit='case'):
        for physical in PHYSICALS:
            chosen, states, metrics, counts = replay(case, physical, None, manifest['forecast_seed'])
            original = originals[(case, 'rollout', '', physical)]
            recorded = rollout_episodes[(case, physical)]
            for metric in METRICS:
                close(metrics[metric], original[metric], 'V12 rollout '+metric)
                close(metrics[metric], recorded[metric], 'diagnostic rollout '+metric)
            assert int(recorded['forecast_transitions']) == counts[-1]
            for tick, action in enumerate(chosen):
                row = rollout_actions[(case, physical, tick)]
                assert int(row['action']) == action
                assert int(row['cumulative_forecast_transitions']) == counts[tick]
            plan_paths[(case, physical)] = chosen, states, metrics, counts
    raw_records = []
    for case in tqdm(CASES, desc='Audit actor and forced decisions', unit='case'):
        for train in TRAINS:
            for physical in PHYSICALS:
                key = case, train, physical
                plan_actions, plan_states, plan_metrics, counts = plan_paths[(case, physical)]
                actor_actions, actor_states, actor_metrics, _ = replay(case, physical,
                    models[train], manifest['forecast_seed'])
                original = originals[(case, 'reliability_credit', str(train), physical)]
                for metric in METRICS:
                    close(actor_metrics[metric], original[metric], 'V12 actor '+metric)
                first = next((t for t, (a, b) in enumerate(zip(actor_actions, plan_actions))
                              if a != b), None)
                assert all(actor_states[t] == plan_states[t]
                           for t in range((first + 1) if first is not None else HORIZON))
                div = divergence[key]
                assert int(div['first_tick']) == (-1 if first is None else first)
                a = actor_actions[first] if first is not None else -1
                b = plan_actions[first] if first is not None else -1
                assert int(div['actor_action']) == a and int(div['planner_action']) == b
                assert div['category'] == ('match' if first is None else class_of(a, b))
                expected_forecasts = counts[first] if first is not None else counts[-1]
                assert int(div['oracle_forecast_transitions']) == expected_forecasts
                interventions = {}
                for label in ('actor', 'one', 'wait4'):
                    if first is None or label == 'actor':
                        override = {}
                    elif label == 'one' or b != 0:
                        override = {first: b}
                    else:
                        override = {t: 0 for t in range(first, min(HORIZON, first+4))}
                    if label == 'actor':
                        chosen, metrics = actor_actions, actor_metrics
                    else:
                        chosen, _, metrics, _ = replay(case, physical, models[train],
                            manifest['forecast_seed'], override)
                    interventions[label] = metrics
                    record = episodes[(*key, label)]
                    assert int(record['first_tick']) == (-1 if first is None else first)
                    for metric in METRICS:
                        close(metrics[metric], record[metric], f'{label} '+metric)
                    for tick, action in enumerate(chosen):
                        row = actions[(*key, label, tick)]
                        assert int(row['action']) == action
                        assert int(row['forced']) == int(tick in override)
                for label in ('actor', 'one', 'wait4'):
                    close(interventions[label]['objective'], div[f'{label}_cost'], label+' cost')
                close(plan_metrics['objective'], div['rollout_cost'], 'rollout cost')
                raw_records.append(div)
    summary = json.loads((run / 'summary.json').read_text())
    for case in CASES:
        all_rows = [r for r in raw_records if r['case'] == case]
        overall = summary['overall'][case]
        means = {name: float(np.mean([float(r[f'{name}_cost']) for r in all_rows]))
                 for name in ('actor', 'one', 'wait4', 'rollout')}
        assert overall['pairs'] == len(all_rows) == 80
        for name in means:
            close(means[name], overall['mean_cost'][name], 'overall cost')
        gap = means['actor'] - means['rollout']
        for label in ('one', 'wait4'):
            if gap > 0:close((means['actor']-means[label])/gap, overall[f'{label}_recovery'], 'recovery')
            else:assert overall[f'{label}_recovery'] is None
        for train in TRAINS:
            cell = [r for r in all_rows if int(r['train_seed']) == train]
            item = summary[f'{case}/{train}']
            assert item['pairs'] == len(cell) == 16
            assert item['divergence_categories'] == dict(Counter(r['category'] for r in cell))
            assert item['planner_wait_divergences'] == sum(int(r['planner_action']) == 0 and
                int(r['first_tick']) >= 0 for r in cell)
            assert item['oracle_forecast_transitions'] == sum(int(r['oracle_forecast_transitions']) for r in cell)
            group_means = {name: float(np.mean([float(r[f'{name}_cost']) for r in cell]))
                           for name in means}
            for name in means:close(group_means[name], item['mean_cost'][name], 'seed cost')
            group_gap = group_means['actor']-group_means['rollout']
            for label in ('one', 'wait4'):
                if group_gap > 0:close((group_means['actor']-group_means[label])/group_gap,
                                     item[f'{label}_recovery'], 'seed recovery')
                else:assert item[f'{label}_recovery'] is None
            ticks = [int(r['first_tick']) for r in cell if int(r['first_tick']) >= 0]
            if ticks:close(np.mean(ticks), item['first_tick_mean'], 'first tick')
            else:assert item['first_tick_mean'] is None
    one = all(summary['overall'][case]['one_recovery'] is not None and
              summary['overall'][case]['one_recovery'] >= .30 and
              sum(summary[f'{case}/{train}']['one_recovery'] is not None and
                  summary[f'{case}/{train}']['one_recovery'] >= .30 for train in TRAINS) >= 4
              for case in CASES)
    reciprocal = summary['overall']['reciprocal_scale2']
    extra = reciprocal['wait4_recovery'] - reciprocal['one_recovery'] if (
        reciprocal['wait4_recovery'] is not None and reciprocal['one_recovery'] is not None) else None
    interchange = summary['overall']['interchangeable_scale1']['mean_cost']
    wait4 = extra is not None and extra >= .20 and interchange['wait4']-interchange['one'] <= 5
    assert summary['gates'] == {'one_action_supported': bool(one), 'wait4_supported': bool(wait4)}
    return {'status': 'PASS', 'run': str(run.relative_to(ROOT)), 'unique_rollout_episodes': 32,
            'paired_interventions': 160, 'actor_intervention_episodes': 480,
            'actor_intervention_decisions': 23040, 'source_hashes_verified': len(manifest['source_hashes']),
            'artifact_hashes_verified': len(manifest['artifact_hashes'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = audit(args.run.resolve())
    (output / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(output)
