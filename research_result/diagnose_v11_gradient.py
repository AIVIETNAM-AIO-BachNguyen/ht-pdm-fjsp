"""Fixed pass over immutable V11 counterfactual banks and old-actor gradients."""
from __future__ import annotations

import argparse
from collections import defaultdict
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
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, catalogue, configs, vector_observation
from ht_pdm_fjsp.maintenance_framework_v6_policy import EdgeSharingActor
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings, state_from_dict, submit_requests

ROOT = Path(__file__).resolve().parent.parent
V11 = ROOT / 'research_result/framework_v11_dev_20261020T115151Z'
PLAN = ROOT / 'research_result/v11_gradient_diagnostic_plan_vi.md'
ACTIONS = catalogue()
ARMS = ('reference_credit', 'self_credit')
TRAIN = range(70500000, 70500005)
SOURCES = [ROOT / p for p in (
    'research_result/diagnose_v11_gradient.py',
    'research_result/v11_gradient_diagnostic_plan_vi.md',
    'src/ht_pdm_fjsp/maintenance_framework_v1.py',
    'src/ht_pdm_fjsp/maintenance_framework_v6_policy.py',
    'src/ht_pdm_fjsp/maintenance_requests.py',
    'src/ht_pdm_fjsp/maintenance_request_policies.py',
)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def wait_gradient(probabilities, scores, legal):
    """Exact initial logit-WAIT direction for maximizing expected (baseline-Q)/10."""
    if 0 not in legal or not any(ACTIONS[i] for i in legal):
        return None
    p = np.asarray(probabilities, dtype=np.float64)
    q = np.asarray(scores, dtype=np.float64)
    baseline = float(np.dot(p, q))
    advantage = (baseline - q) / 10.0
    return float(p[0] * (advantage[0] - np.dot(p, advantage)))


def masked_probabilities(actor, observation, legal):
    with torch.no_grad():
        logits = actor(torch.as_tensor(observation).reshape(1, 56))[0]
        masked = torch.full_like(logits, -1e9)
        masked[list(legal)] = logits[list(legal)]
        return torch.softmax(masked, dim=0).numpy()


def checkpoint(actor_path):
    payload = torch.load(actor_path, map_location='cpu', weights_only=True)
    actor = EdgeSharingActor('shared_edge')
    actor.load_state_dict(payload['state_dict'])
    actor.eval()
    return actor


def summarize(rows):
    result = {}
    for arm in ARMS:
        for seed in TRAIN:
            for iteration in range(1, 7):
                subset = [r for r in rows if r['arm'] == arm and r['train_seed'] == seed
                          and r['iteration'] == iteration]
                if not subset:
                    continue
                key = f'{arm}/{seed}/{iteration}'
                result[key] = dict(states=len(subset), mean_p_wait=float(np.mean([r['p_wait'] for r in subset])),
                    mean_gradient_self=float(np.mean([r['gradient_self'] for r in subset])),
                    mean_gradient_reference=float(np.mean([r['gradient_reference'] for r in subset])),
                    service_gradient_self=sum(r['gradient_self'] < 0 for r in subset),
                    service_gradient_reference=sum(r['gradient_reference'] < 0 for r in subset),
                    self_opt_wait=sum(r['best_self'] == 0 for r in subset),
                    reference_opt_wait=sum(r['best_reference'] == 0 for r in subset),
                    source_wait=sum(r['source_action'] == 0 for r in subset),
                    mean_q_spread_self=float(np.mean([r['q_spread_self'] for r in subset])),
                    mean_q_spread_reference=float(np.mean([r['q_spread_reference'] for r in subset])))
    return result


def run(profile, output):
    if profile not in ('smoke', 'dev'):
        raise ValueError(profile)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    start = time.monotonic()
    limit = 180 if profile == 'smoke' else 600
    def deadline():
        if time.monotonic() - start > limit:
            raise TimeoutError('gradient diagnostic timeout')
    source_rel = [str(p.relative_to(ROOT)) for p in SOURCES]
    dirty = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *source_rel], cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', *source_rel], cwd=ROOT, text=True).splitlines()
    if profile == 'dev' and (dirty or untracked):
        raise RuntimeError('uncommitted diagnostic source')
    prior = json.loads((V11 / 'manifest.json').read_text())
    assert prior['status'] == 'COMPLETED' and prior['profile'] == 'dev'
    assert sha(V11 / 'selection.json') == prior['selection_sha256']
    source = read_csv(V11 / 'counterfactual_states.csv')
    assert len(source) == len(ARMS) * len(TRAIN) * 6 * len(CASES) * 32
    if profile == 'smoke':
        source = [r for r in source if int(r['train_seed']) == min(TRAIN) and int(r['iteration']) == 1]
    models = {}
    hashes = {}
    for row in source:
        key = (row['arm'], int(row['train_seed']), int(row['iteration']))
        if key not in models:
            path = V11 / 'checkpoints' / key[0] / str(key[1]) / f'{key[2]-1}.pt'
            hashes['/'.join(map(str, key))] = sha(path)
            models[key] = checkpoint(path)
    manifest = dict(status='RUNNING', phase='preflight', profile=profile,
        started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        invocation=[sys.executable, *sys.argv], runtime=dict(python=platform.python_version(),
        platform=platform.platform()), timeout_seconds=limit,
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
        v11_manifest_sha256=sha(V11 / 'manifest.json'),
        v11_q_sha256=sha(V11 / 'counterfactual_states.csv'),
        v11_selection_sha256=sha(V11 / 'selection.json'), checkpoint_hashes=hashes,
        cases=list(CASES), training_seeds=list(TRAIN if profile == 'dev' else TRAIN[:1]),
        iterations=list(range(1, 7) if profile == 'dev' else (1,)))
    save(output / 'manifest.json', manifest)
    shutil.copy2(PLAN, output / 'protocol.md')
    try:
        manifest['phase'] = 'scan'
        save(output / 'manifest.json', manifest)
        rows = []
        for row in tqdm(source, desc='V11 stored-Q gradient scan', unit='state'):
            deadline()
            arm, seed, iteration = row['arm'], int(row['train_seed']), int(row['iteration'])
            case, tick = row['case'], int(row['tick'])
            state = state_from_dict(json.loads(row['original_state']))
            cfg = configs()[case]
            admitted = submit_requests(cfg, state, tick, request_action(cfg, state, tick, 'threshold'))
            allowed = set(legal_matchings(cfg, admitted))
            legal = tuple(index for index, action in enumerate(ACTIONS) if action in allowed)
            assert legal == tuple(json.loads(row['legal']))
            self_q = {int(k): float(v) for k, v in json.loads(row['scores_self']).items()}
            ref_q = {int(k): float(v) for k, v in json.loads(row['scores_reference']).items()}
            assert set(legal) == set(self_q) == set(ref_q)
            if 0 not in legal or not any(ACTIONS[i] for i in legal):
                continue
            observation = vector_observation(cfg, admitted, tick)
            p = masked_probabilities(models[(arm, seed, iteration)], observation, legal)
            q_self = np.zeros(len(ACTIONS)); q_ref = np.zeros(len(ACTIONS))
            for index in legal:
                q_self[index] = self_q[index]
                q_ref[index] = ref_q[index]
            service = [index for index in legal if ACTIONS[index]]
            item = dict(arm=arm, train_seed=seed, iteration=iteration, case=case,
                physical_seed=int(row['physical_seed']), tick=tick,
                source_action=int(row['source_action']), legal_count=len(legal),
                p_wait=float(p[0]), best_self=min(legal, key=lambda i: self_q[i]),
                best_reference=min(legal, key=lambda i: ref_q[i]),
                q_wait_delta_self=self_q[0]-min(self_q[i] for i in service),
                q_wait_delta_reference=ref_q[0]-min(ref_q[i] for i in service),
                gradient_self=wait_gradient(p,q_self,legal),
                gradient_reference=wait_gradient(p,q_ref,legal),
                q_spread_self=max(self_q.values())-min(self_q.values()),
                q_spread_reference=max(ref_q.values())-min(ref_q.values()))
            rows.append(item)
        with (output / 'gradient_rows.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        save(output / 'summary.json', summarize(rows))
        if manifest['source_hashes'] != {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}:
            raise RuntimeError('source changed during scan')
        manifest.update(status='COMPLETED', phase='done', elapsed_seconds=time.monotonic()-start,
            finished_utc=datetime.now(timezone.utc).isoformat(), counts=dict(source_states=len(source),
            service_legal_states=len(rows)), artifact_hashes={p.name: sha(p) for p in output.iterdir()
            if p.is_file() and p.name != 'manifest.json'})
        save(output / 'manifest.json', manifest)
        return output
    except Exception as exc:
        manifest.update(status='INCOMPLETE' if isinstance(exc, TimeoutError) else 'FAILED',
            error=repr(exc), phase='stopped')
        save(output / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['smoke', 'dev'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.profile, args.output_dir))
