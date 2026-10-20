"""Independent replay and aggregation audit for V12 correction budget curve."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

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
PRIOR = ROOT / 'research_result/v12_sparse_planner_dev_20261020T135214Z'
CASES = ('interchangeable_scale1', 'reciprocal_scale2')
TRAIN = tuple(range(71500000, 71500005))
DEV = tuple(range(71510000, 71510016))
K = (0, 1, 2, 4)
ACTIONS = catalogue()
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


def actor_choice(actor, cfg, state, tick):
    submissions = request_action(cfg, state, tick, 'threshold')
    admitted = submit_requests(cfg, state, tick, submissions)
    legal = sorted(ACTIONS.index(edges) for edges in legal_matchings(cfg, admitted))
    observation = vector_observation(cfg, admitted, tick)
    with torch.no_grad():
        logits = actor(torch.tensor(observation).reshape(1,56))[0]
        masked = torch.full_like(logits, -1e9)
        masked[legal] = logits[legal]
        choice = int(np.argmax(torch.softmax(masked, dim=0).numpy()))
    return choice, legal, submissions


def replay(case, physical, actor, forecast_seed, budget, recorded):
    cfg = configs()[case]; env = FrameworkEnv(cfg); env.reset(seed=physical)
    planner = V7RuleController('rollout', forecast_seed)
    calls = corrections = 0
    for tick in range(HORIZON):
        state = env.inner.state
        candidate, legal, submissions = actor_choice(actor, cfg, state, tick)
        queried = corrected = 0; planner_action = -1; chosen = candidate
        if corrections < budget:
            supplied, matching = planner.choose(cfg, state, tick, case=case, seed=physical)
            assert supplied == submissions
            planner_action = ACTIONS.index(matching)
            queried = 1; calls += 1
            if planner_action != candidate:
                chosen = planner_action; corrections += 1; corrected = 1
        assert chosen in legal
        row = recorded[tick]
        for key, value in (('actor_action', candidate), ('planner_action', planner_action),
                           ('action', chosen), ('queried', queried), ('corrected', corrected),
                           ('cumulative_queries', calls), ('cumulative_corrections', corrections),
                           ('cumulative_forecast_transitions', planner.forecast_transitions)):
            assert int(row[key]) == int(value), (case, physical, budget, tick, key)
        env.step(chosen)
    return env.episode_metrics(), calls, corrections, planner.forecast_transitions


def audit(run):
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED' and manifest['profile'] == 'dev'
    assert manifest['budgets'] == list(K) and manifest['cases'] == list(CASES)
    assert manifest['training_seeds'] == list(TRAIN) and manifest['physical_seeds'] == list(DEV)
    assert manifest['counts'] == {'pairs': 160, 'episodes': 640, 'decisions': 30720}
    for path, key in ((V12/'manifest.json','v12_manifest_sha256'),
                      (V12/'selection.json','v12_selection_sha256'),
                      (V12/'episodes.csv','v12_episodes_sha256'),
                      (PRIOR/'manifest.json','prior_manifest_sha256'),
                      (PRIOR/'episodes.csv','prior_episodes_sha256')):
        assert sha(path) == manifest[key]
    for name, digest in manifest['source_hashes'].items():
        assert sha(ROOT/name) == digest
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(run/name) == digest
    selection = json.loads((V12/'selection.json').read_text())
    actors = {}
    for train in TRAIN:
        item = selection['reliability_credit'][str(train)]
        path = V12/item['checkpoint']
        assert sha(path) == item['sha256'] == manifest['selected_checkpoint_hashes'][str(train)]
        actors[train] = actor_at(path)
    source = {(r['case'],r['controller'],r['train_seed'],int(r['seed'])):r
              for r in rows(V12/'episodes.csv') if r['case'] in CASES and
              r['controller'] in ('reliability_credit','rollout')}
    prior = {(r['case'],int(r['train_seed']),int(r['physical_seed']),r['variant']):r
             for r in rows(PRIOR/'episodes.csv')}
    episodes = {(r['case'],int(r['train_seed']),int(r['physical_seed']),int(r['budget'])):r
                for r in rows(run/'episodes.csv')}
    decisions = {(r['case'],int(r['train_seed']),int(r['physical_seed']),int(r['budget']),int(r['tick'])):r
                 for r in rows(run/'actions.csv')}
    assert len(episodes) == 640 and len(decisions) == 30720
    for case in tqdm(CASES, desc='Audit correction budget cases', unit='case'):
        for train in TRAIN:
            for physical in DEV:
                for budget in K:
                    key = case,train,physical,budget
                    trace = [decisions[(*key,tick)] for tick in range(HORIZON)]
                    metrics, calls, corrections, transitions = replay(case,physical,actors[train],
                        manifest['forecast_seed'],budget,trace)
                    record = episodes[key]
                    assert int(record['planner_calls']) == calls
                    assert int(record['corrections']) == corrections
                    assert int(record['forecast_transitions']) == transitions
                    assert 0 <= corrections <= budget and 0 <= calls <= HORIZON
                    close(float(record['actor_selection_seconds'])+
                          float(record['planner_selection_seconds']),
                          record['total_selection_seconds'],'selection decomposition')
                    for metric in METRICS:
                        close(metrics[metric],record[metric],metric)
                    if budget in (0,1):
                        previous = prior[(case,train,physical,'actor' if budget == 0 else 'one')]
                        for metric in METRICS:close(metrics[metric],previous[metric],'prior K0/K1 '+metric)
                    if budget == 0:
                        original = source[(case,'reliability_credit',str(train),physical)]
                        for metric in METRICS:close(metrics[metric],original[metric],'V12 actor '+metric)
    summary = json.loads((run/'summary.json').read_text())
    assert manifest['bootstrap_seed'] == 71590000
    for case in CASES:
        for budget in K:
            cell = [episodes[(case,train,physical,budget)] for train in TRAIN for physical in DEV]
            item = summary['by_case'][case][str(budget)]
            assert item['episodes'] == len(cell) == 80
            means = {metric:float(np.mean([float(r[metric]) for r in cell])) for metric in METRICS}
            cost = means['objective']
            rollout = float(np.mean([float(source[(case,'rollout','',physical)]['objective'])
                                     for train in TRAIN for physical in DEV]))
            actor_cost = float(np.mean([float(episodes[(case,train,physical,0)]['objective'])
                                       for train in TRAIN for physical in DEV]))
            for key,value in (('mean_cost',cost),('rollout_cost',rollout),
                              ('actor_cost',actor_cost),('cost_gap_to_rollout',cost-rollout)):
                close(item[key],value,key)
            gap = actor_cost-rollout
            if gap > 0:close(item['gap_recovery'],(actor_cost-cost)/gap,'recovery')
            else:assert item['gap_recovery'] is None
            for metric,value in means.items():close(item['mean_metrics'][metric],value,'mean '+metric)
            for key,column in (('mean_planner_calls','planner_calls'),
                               ('mean_corrections','corrections'),
                               ('mean_forecast_transitions','forecast_transitions'),
                               ('mean_total_selection_seconds','total_selection_seconds')):
                close(item[key],np.mean([float(r[column]) for r in cell]),key)
            close(item['full_rollout_selection_seconds'],np.mean([
                float(source[(case,'rollout','',physical)]['selection_seconds'])
                for train in TRAIN for physical in DEV]),'rollout speed')
            paired = np.asarray([[float(episodes[(case,train,physical,budget)]['objective'])-
                        float(source[(case,'rollout','',physical)]['objective'])
                        for physical in DEV] for train in TRAIN])
            rng = np.random.default_rng(71590000+CASES.index(case)*10+budget)
            ti = rng.integers(0,5,size=(20000,5)); pi = rng.integers(0,16,size=(20000,16))
            ci = np.quantile(paired[ti[:,:,None],pi[:,None,:]].mean(axis=(1,2)),[.025,.975])
            for a,b in zip(ci,item['cost_gap_ci95']):close(a,b,'bootstrap')
            for train in TRAIN:
                subset = [episodes[(case,train,physical,budget)] for physical in DEV]
                close(item['training_seed_cost'][str(train)],
                      np.mean([float(r['objective']) for r in subset]),'seed cost')
                close(item['training_seed_rollout_cost'][str(train)],
                      np.mean([float(source[(case,'rollout','',physical)]['objective'])
                               for physical in DEV]),'seed rollout cost')
    for budget in K[1:]:
        cost_ok = all((summary['by_case'][case][str(budget)]['mean_cost'] <=
                       1.01*summary['by_case'][case][str(budget)]['rollout_cost'] and
                       sum(summary['by_case'][case][str(budget)]['training_seed_cost'][str(train)] <=
                           1.01*summary['by_case'][case][str(budget)]['training_seed_rollout_cost'][str(train)]
                           for train in TRAIN) >= 4) for case in CASES)
        time_ok = all(summary['by_case'][case][str(budget)]['mean_total_selection_seconds'] <=
                      .1*summary['by_case'][case][str(budget)]['full_rollout_selection_seconds']
                      for case in CASES)
        assert summary['gates'][str(budget)] == dict(cost_headroom=bool(cost_ok),
            direct_runtime_headroom=bool(time_ok))
    return dict(status='PASS',run=str(run.relative_to(ROOT)),pairs=160,episodes=640,decisions=30720,
        source_hashes_verified=len(manifest['source_hashes']),
        artifact_hashes_verified=len(manifest['artifact_hashes']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(); output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    result = audit(args.run.resolve())
    (output/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(output)
