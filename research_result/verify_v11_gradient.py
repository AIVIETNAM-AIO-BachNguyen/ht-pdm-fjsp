"""Independent full audit of V11 stored-Q policy-gradient diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, catalogue, configs, vector_observation
from ht_pdm_fjsp.maintenance_framework_v6_policy import EdgeSharingActor
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings, state_from_dict, submit_requests

ROOT = Path(__file__).resolve().parent.parent
V11 = ROOT / 'research_result/framework_v11_dev_20261020T115151Z'
ACTIONS = catalogue()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def close(a, b, label):
    if not np.isclose(float(a), float(b), rtol=0, atol=1e-6):
        raise AssertionError(f'{label}: {a} != {b}')


def audit(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED' and manifest['profile'] == 'dev'
    assert manifest['counts'] == {'source_states': 7680, 'service_legal_states': 7680}
    assert manifest['v11_manifest_sha256'] == sha(V11 / 'manifest.json')
    assert manifest['v11_selection_sha256'] == sha(V11 / 'selection.json')
    assert manifest['v11_q_sha256'] == sha(V11 / 'counterfactual_states.csv')
    for name, digest in manifest['source_hashes'].items():
        assert sha(ROOT / name) == digest
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(run / name) == digest
    source = rows(V11 / 'counterfactual_states.csv')
    result = rows(run / 'gradient_rows.csv')
    assert len(source) == len(result) == 7680
    actors = {}
    for key, digest in manifest['checkpoint_hashes'].items():
        arm, train, iteration = key.split('/')
        path = V11 / 'checkpoints' / arm / train / f'{int(iteration)-1}.pt'
        assert sha(path) == digest
        payload = torch.load(path, map_location='cpu', weights_only=True)
        actor = EdgeSharingActor('shared_edge')
        actor.load_state_dict(payload['state_dict'])
        actor.eval()
        actors[(arm, int(train), int(iteration))] = actor
    assert len(actors) == 60
    for original, item in tqdm(zip(source, result), total=len(result), desc='Audit V11 Q gradient', unit='state'):
        for key in ('arm', 'train_seed', 'iteration', 'case', 'physical_seed', 'tick', 'source_action'):
            assert original[key] == item[key]
        arm = item['arm']; train = int(item['train_seed']); iteration = int(item['iteration'])
        case = item['case']; tick = int(item['tick']); cfg = configs()[case]
        state = state_from_dict(json.loads(original['original_state']))
        admitted = submit_requests(cfg, state, tick, request_action(cfg, state, tick, 'threshold'))
        allowed = set(legal_matchings(cfg, admitted))
        legal = [i for i, action in enumerate(ACTIONS) if action in allowed]
        assert legal == json.loads(original['legal']) and int(item['legal_count']) == len(legal)
        assert 0 in legal and any(ACTIONS[i] for i in legal)
        with torch.no_grad():
            obs = vector_observation(cfg, admitted, tick)
            logits = actors[(arm, train, iteration)](torch.tensor(obs).reshape(1, 56))[0]
            logits[[i for i in range(len(ACTIONS)) if i not in legal]] = -1e9
            p = torch.softmax(logits, dim=0).numpy()
        close(item['p_wait'], p[0], 'p_wait')
        service = [i for i in legal if ACTIONS[i]]
        for continuation, key in (('self', 'scores_self'), ('reference', 'scores_reference')):
            q = {int(k): float(v) for k, v in json.loads(original[key]).items()}
            assert set(q) == set(legal)
            best = min(legal, key=lambda i: q[i])
            assert int(item[f'best_{continuation}']) == best
            close(item[f'q_wait_delta_{continuation}'], q[0] - min(q[i] for i in service), 'wait delta')
            close(item[f'q_spread_{continuation}'], max(q.values()) - min(q.values()), 'spread')
            expected_q = sum(float(p[i]) * q[i] for i in legal)
            advantages = {i: (expected_q - q[i]) / 10 for i in legal}
            mean_adv = sum(float(p[i]) * advantages[i] for i in legal)
            expected_gradient = float(p[0]) * (advantages[0] - mean_adv)
            close(item[f'gradient_{continuation}'], expected_gradient, 'gradient')
    summary = json.loads((run / 'summary.json').read_text())
    assert len(summary) == 60
    for arm in ('reference_credit', 'self_credit'):
        for train in range(70500000, 70500005):
            for iteration in range(1, 7):
                key = f'{arm}/{train}/{iteration}'
                cell = [x for x in result if x['arm'] == arm and int(x['train_seed']) == train
                        and int(x['iteration']) == iteration]
                aggregate = summary[key]
                assert aggregate['states'] == len(cell) == 128
                for field, condition in (('service_gradient_self', lambda x: float(x['gradient_self']) < 0),
                                         ('service_gradient_reference', lambda x: float(x['gradient_reference']) < 0),
                                         ('self_opt_wait', lambda x: int(x['best_self']) == 0),
                                         ('reference_opt_wait', lambda x: int(x['best_reference']) == 0),
                                         ('source_wait', lambda x: int(x['source_action']) == 0)):
                    assert aggregate[field] == sum(condition(x) for x in cell)
                for mean_field, row_field in (('mean_p_wait', 'p_wait'),
                    ('mean_gradient_self', 'gradient_self'), ('mean_gradient_reference', 'gradient_reference'),
                    ('mean_q_spread_self', 'q_spread_self'),
                    ('mean_q_spread_reference', 'q_spread_reference')):
                    close(aggregate[mean_field], np.mean([float(x[row_field]) for x in cell]), mean_field)
    for train in range(70500000, 70500005):
        left = actors[('reference_credit', train, 1)].state_dict()
        right = actors[('self_credit', train, 1)].state_dict()
        assert all(torch.equal(left[k], right[k]) for k in left)
        left_rows = [x for x in source if x['arm'] == 'reference_credit' and
                     int(x['train_seed']) == train and int(x['iteration']) == 1]
        right_rows = [x for x in source if x['arm'] == 'self_credit' and
                      int(x['train_seed']) == train and int(x['iteration']) == 1]
        assert [(x['case'], x['physical_seed'], x['tick'], x['scores_self'], x['scores_reference'])
                for x in left_rows] == [(x['case'], x['physical_seed'], x['tick'], x['scores_self'], x['scores_reference'])
                                      for x in right_rows]
    return dict(status='PASS', run=str(run.relative_to(ROOT)), states=len(result),
                checkpoint_hashes_verified=len(actors), initial_pairs_verified=5,
                source_hashes_verified=len(manifest['source_hashes']),
                artifact_hashes_verified=len(manifest['artifact_hashes']))


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
