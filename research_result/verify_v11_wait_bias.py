"""Independent audit of the V11 WAIT diagnostic's stored Q and policy actions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, HORIZON, FrameworkEnv, catalogue, configs, vector_observation
from ht_pdm_fjsp.maintenance_framework_v6_policy import EdgeSharingActor
from ht_pdm_fjsp.maintenance_request_policies import request_action, risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests

ROOT = Path(__file__).resolve().parent.parent
V11 = ROOT / 'research_result/framework_v11_dev_20261020T115151Z'
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


def check_close(a, b, label):
    if not np.isclose(float(a), float(b), rtol=0, atol=1e-7):
        raise AssertionError(f'{label}: {a} != {b}')


def actor_at(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    actor = EdgeSharingActor('shared_edge')
    actor.load_state_dict(payload['state_dict'])
    actor.eval()
    return actor


def action_at(actor, observation, legal):
    with torch.no_grad():
        logits = actor(torch.tensor(observation).reshape(1, 56))[0]
        masked = torch.full_like(logits, -1e9)
        masked[legal] = logits[legal]
        probabilities = torch.softmax(masked, dim=0).numpy()
        return int(np.argmax(probabilities))


def audit(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED' and manifest['profile'] == 'dev'
    assert manifest['counts'] == {'q_states': 7680, 'episodes': 1280, 'decisions': 61440}
    assert manifest['v11_manifest_sha256'] == sha(V11 / 'manifest.json')
    for name in ('selection', 'development', 'episodes', 'q'):
        target = {'selection': 'selection.json', 'development': 'development.csv',
                  'episodes': 'episodes.csv', 'q': 'counterfactual_states.csv'}[name]
        assert manifest[f'v11_{name}_sha256'] == sha(V11 / target)
    for name, digest in manifest['source_hashes'].items():
        assert sha(ROOT / name) == digest
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(run / name) == digest

    selections = json.loads((V11 / 'selection.json').read_text())
    q_source = rows(V11 / 'counterfactual_states.csv')
    q_audit = rows(run / 'q_wait.csv')
    expected_q = []
    for item in q_source:
        legal = set(json.loads(item['legal']))
        self_q = {int(k): float(v) for k, v in json.loads(item['scores_self']).items()}
        ref_q = {int(k): float(v) for k, v in json.loads(item['scores_reference']).items()}
        assert legal == set(self_q) == set(ref_q)
        service = [index for index in legal if ACTIONS[index]]
        if 0 not in legal or not service:
            continue
        self_delta = self_q[0] - min(self_q[index] for index in service)
        ref_delta = ref_q[0] - min(ref_q[index] for index in service)
        expected_q.append((item, len(legal), self_delta, ref_delta))
    assert len(expected_q) == len(q_audit) == 7680
    for (source, count, ds, dr), actual in tqdm(zip(expected_q, q_audit), total=len(q_audit), desc='Audit stored Q', unit='state'):
        for key in ('arm', 'train_seed', 'iteration', 'case', 'physical_seed', 'tick', 'source_action'):
            assert source[key] == actual[key]
        assert int(actual['legal_count']) == count
        assert int(actual['clock_bin']) == min(int(source['tick']) // 16, 2)
        for key, value in (('q_self_wait_delta', ds), ('q_ref_wait_delta', dr),
                           ('q_preference_difference', dr - ds)):
            check_close(actual[key], value, key)
        for key, value in (('self_favors_wait', ds < 0), ('ref_favors_wait', dr < 0),
                           ('self_favors_wait_ge5', ds <= -5), ('ref_favors_wait_ge5', dr <= -5)):
            assert int(actual[key]) == int(value)

    actions = rows(run / 'actions.csv')
    episodes = rows(run / 'episodes.csv')
    assert len(actions) == 61440 and len(episodes) == 1280
    development = {(x['case'], x['controller'], int(x['train_seed']),
                    int(x['checkpoint_iteration']), int(x['seed'])): x
                   for x in rows(V11 / 'development.csv')}
    originals = {(x['case'], x['controller'], int(x['train_seed']), int(x['seed'])): x
                 for x in rows(V11 / 'episodes.csv') if x['controller'] in ('reference_credit', 'self_credit')}
    actors = {}
    for arm in ('reference_credit', 'self_credit'):
        for train in range(70500000, 70500005):
            selected = selections[arm][str(train)]
            for label, path in (('initial', V11 / 'checkpoints' / arm / str(train) / '0.pt'),
                                ('selected', V11 / selected['checkpoint'])):
                assert sha(path) == manifest['checkpoint_hashes'][f'{arm}/{train}/{label}']
                if label == 'selected':
                    assert sha(path) == selected['sha256']
                actors[(arm, train, label)] = actor_at(path)
    for train in range(70500000, 70500005):
        left = actors[('reference_credit', train, 'initial')].state_dict()
        right = actors[('self_credit', train, 'initial')].state_dict()
        assert left.keys() == right.keys()
        assert all(torch.equal(left[key], right[key]) for key in left)

    for episode_index, record in enumerate(tqdm(episodes, desc='Audit episode actions', unit='episode')):
        case, arm, train, label = (record['case'], record['arm'], int(record['train_seed']),
                                    record['checkpoint_label'])
        seed = int(record['physical_seed'])
        selected_iteration = selections[arm][str(train)]['iteration']
        iteration = 0 if label == 'initial' else selected_iteration
        assert case in CASES and int(record['checkpoint_iteration']) == iteration
        cfg = configs()[case]
        env = FrameworkEnv(cfg)
        env.reset(seed=seed)
        legal_count = waits = starts = 0
        for tick in range(HORIZON):
            actual = actions[episode_index * HORIZON + tick]
            assert (actual['case'], actual['arm'], int(actual['train_seed']),
                    actual['checkpoint_label'], int(actual['checkpoint_iteration']),
                    int(actual['physical_seed']), int(actual['tick'])) == \
                   (case, arm, train, label, iteration, seed, tick)
            state = env.inner.state
            submissions = request_action(cfg, state, tick, 'threshold')
            admitted = submit_requests(cfg, state, tick, submissions)
            legal = sorted(ACTIONS.index(edges) for edges in legal_matchings(cfg, admitted))
            observation = vector_observation(cfg, admitted, tick)
            chosen = action_at(actors[(arm, train, label)], observation, legal)
            risk = ACTIONS.index(risk_matching(cfg, admitted, tick))
            service = any(ACTIONS[index] for index in legal)
            queued_cm = sum(req is not None and req.started_at is None and req.kind == 'corrective'
                            for req in admitted.requests)
            for key, value in (('legal_service', service), ('chosen_action', chosen),
                               ('risk_action', risk), ('legal_count', len(legal)),
                               ('chosen_cardinality', len(ACTIONS[chosen])), ('queued_cm', queued_cm)):
                assert int(actual[key]) == int(value), (episode_index, tick, key)
            legal_count += service
            waits += chosen == 0 and service
            starts += len(ACTIONS[chosen])
            env.step(chosen)
        metrics = env.episode_metrics()
        for key, value in (('legal_service_opportunities', legal_count),
                           ('wait_with_legal_service', waits), ('H_service_starts', starts)):
            check_close(record[key], value, key)
        source = development[(case, arm, train, iteration, seed)]
        for key in METRICS:
            check_close(record[key], metrics[key], key)
            check_close(source[key], metrics[key], 'V11 development ' + key)
            if label == 'selected':
                check_close(originals[(case, arm, train, seed)][key], metrics[key],
                            'V11 selected ' + key)
    summary = json.loads((run / 'summary.json').read_text())
    for arm in ('reference_credit', 'self_credit'):
        for train in range(70500000, 70500005):
            cell = [x for x in q_audit if x['arm'] == arm and int(x['train_seed']) == train]
            reported = summary['q'][f'{arm}/{train}']
            assert reported['states'] == len(cell) == 768
            for field in ('self_favors_wait', 'ref_favors_wait', 'self_favors_wait_ge5', 'ref_favors_wait_ge5'):
                assert reported[field] == sum(int(x[field]) for x in cell)
            check_close(reported['mean_q_preference_difference'],
                        np.mean([float(x['q_preference_difference']) for x in cell]), 'summary Q mean')
            for label in ('initial', 'selected'):
                for case in CASES:
                    subset = [x for x in episodes if x['case'] == case and x['arm'] == arm and
                              int(x['train_seed']) == train and x['checkpoint_label'] == label]
                    aggregate = summary['episodes'][f'{case}/{arm}/{train}/{label}']
                    assert aggregate['episodes'] == len(subset) == 16
                    for key in ('legal_service_opportunities', 'wait_with_legal_service'):
                        assert aggregate[key] == sum(int(x[key]) for x in subset)
                    for key, original in (('mean_H_service_starts', 'H_service_starts'),
                                          ('mean_queued_cm_ticks', 'queued_cm_ticks'),
                                          ('mean_objective', 'objective')):
                        check_close(aggregate[key], np.mean([float(x[original]) for x in subset]),
                                    'summary ' + key)
    return {'status': 'PASS', 'run': str(run.relative_to(ROOT)), 'q_states': len(q_audit),
            'episodes': len(episodes), 'decisions': len(actions),
            'source_hashes_verified': len(manifest['source_hashes']),
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
