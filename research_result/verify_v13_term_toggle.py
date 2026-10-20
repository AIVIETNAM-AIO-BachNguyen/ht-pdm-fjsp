"""Independent row/action/aggregation audit for the V13 term-toggle diagnostic."""
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
from ht_pdm_fjsp.maintenance_framework_v13_policy import OpportunityCostActor
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings, state_from_dict, submit_requests

ROOT = Path(__file__).resolve().parent.parent
ACTIONS = catalogue()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def close(left, right, label: str) -> None:
    if not np.isclose(float(left), float(right), rtol=0, atol=1e-8):
        raise AssertionError(f'{label}: {left} != {right}')


def category(a: int, b: int) -> str:
    if a == b:
        return 'same'
    if not ACTIONS[a] or not ACTIONS[b]:
        return 'wait'
    if sorted(m for m, _ in ACTIONS[a]) != sorted(m for m, _ in ACTIONS[b]):
        return 'machine_set'
    return 'assignment'


def row_key(row: dict[str, str]) -> tuple:
    return (int(row['train_seed']), int(row['iteration']), row['case'],
            int(row['physical_seed']), int(row['tick']), row['rank'])


def audit(run: Path, diagnostic: Path) -> dict:
    manifest = json.loads((diagnostic / 'manifest.json').read_text())
    source = json.loads((run / 'manifest.json').read_text())
    profile = manifest['profile']
    assert profile in ('dev', 'smoke')
    assert manifest['status'] == source['status'] == 'COMPLETED'
    assert manifest['source_manifest_sha256'] == sha(run / 'manifest.json')
    assert manifest['source_q_sha256'] == sha(run / 'counterfactual_states.csv')
    assert manifest['source_selection_sha256'] == sha(run / 'selection.json')
    assert manifest['script_sha256'] == sha(ROOT / 'research_result/run_v13_term_toggle.py')
    assert manifest['protocol_sha256'] == sha(ROOT / 'research_result/v13_component_intervention_protocol_vi.md')
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(diagnostic / name) == digest
    for name, digest in source['artifact_hashes'].items():
        assert sha(run / name) == digest
    assert source['profile'] == profile
    changed_sources = {name for name, digest in source['source_hashes'].items()
                       if sha(ROOT / name) != digest}
    expected_smoke_docs = {'research_result/framework_v13_dev_protocol_vi.md',
                           'research_result/method_v13_proposal_vi.md'}
    assert changed_sources == (expected_smoke_docs if profile == 'smoke' else set())
    assert manifest['smoke_document_hash_exceptions'] == sorted(changed_sources)
    raw = csv_rows(run / 'counterfactual_states.csv')
    raw_comp = {row_key(row): row for row in raw if row['arm'] == 'competition'}
    assert len(raw_comp) * 2 == len(raw)
    output = csv_rows(diagnostic / 'rows.csv')
    assert len(output) == len(raw_comp) == int(json.loads((diagnostic / 'summary.json').read_text())['rows'])
    assert {row_key(row) for row in output} == set(raw_comp)
    if profile == 'dev':
        seeds, updates, states, draws = tuple(range(72500000, 72500005)), 6, 64, 20000
    else:
        seeds, updates, states, draws = (72520000,), 1, 3, 100
    assert len(output) == len(seeds) * updates * len(CASES) * states
    assert manifest['bootstrap_draws'] == draws
    cube = np.empty((len(seeds), updates, len(CASES), states))
    model_cache = {}
    decisions = []
    for si, seed in enumerate(seeds):
        for update in range(1, updates + 1):
            model_path = run / 'checkpoints' / 'competition' / str(seed) / f'{update - 1}.pt'
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            assert checkpoint['metadata']['arm'] == 'competition'
            model = OpportunityCostActor(competition=True)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            model_cache[(seed, update)] = model
            for ci, case in enumerate(CASES):
                cell = sorted((row for row in output if int(row['train_seed']) == seed and
                               int(row['iteration']) == update and row['case'] == case),
                              key=lambda row: row['rank'])
                assert len(cell) == states
                for ri, record in enumerate(tqdm(cell, desc=f'Independent term audit {seed}/{update}/{case}',
                                                  unit='state', leave=False)):
                    raw_row = raw_comp[row_key(record)]
                    cfg = configs()[case]
                    origin = state_from_dict(json.loads(raw_row['original_state']))
                    tick = int(record['tick'])
                    submitted = request_action(cfg, origin, tick, 'threshold')
                    admitted = submit_requests(cfg, origin, tick, submitted)
                    available = {ACTIONS.index(action) for action in legal_matchings(cfg, admitted)}
                    legal = [int(value) for value in json.loads(raw_row['legal'])]
                    assert set(legal) == available
                    obs = vector_observation(cfg, admitted, tick)
                    with torch.no_grad():
                        logits, penalty, demand = model.components(torch.as_tensor(obs).reshape(1, 56))
                    duration = torch.as_tensor(obs[40:56]).reshape(1, 4, 2, 2)[..., 1].clamp(min=0)
                    occupation = torch.einsum('bmj,amj->baj', duration, model.incidence)
                    exposed = torch.einsum('bmj,am->baj', demand, model.unserved)
                    assert torch.allclose(penalty, (occupation * exposed).sum(dim=2), atol=1e-6)
                    score_full = logits[0].numpy()
                    score_zero = (logits + penalty)[0].numpy()
                    chosen_full = sorted(legal, key=lambda a: (-float(score_full[a]), a))[0]
                    chosen_zero = sorted(legal, key=lambda a: (-float(score_zero[a]), a))[0]
                    assert int(record['full_action']) == chosen_full
                    assert int(record['zero_action']) == chosen_zero
                    q = {int(a): float(v) for a, v in json.loads(raw_row['scores']).items()}
                    assert set(q) == set(legal)
                    close(record['full_q'], q[chosen_full], 'full Q')
                    close(record['zero_q'], q[chosen_zero], 'zero Q')
                    delta = q[chosen_full] - q[chosen_zero]
                    close(record['delta'], delta, 'Q delta')
                    assert record['decision_class'] == category(chosen_full, chosen_zero)
                    assert int(record['full_cardinality']) == len(ACTIONS[chosen_full])
                    assert int(record['zero_cardinality']) == len(ACTIONS[chosen_zero])
                    cube[si, update - 1, ci, ri] = delta
                    decisions.append(record)
    summary = json.loads((diagnostic / 'summary.json').read_text())
    close(summary['mean_delta'], cube.mean(), 'mean delta')
    for si, seed in enumerate(seeds):
        close(summary['by_train_seed'][str(seed)], cube[si].mean(), 'seed mean')
    for ci, case in enumerate(CASES):
        close(summary['by_case'][case], cube[:, :, ci, :].mean(), 'case mean')
    for ui in range(updates):
        close(summary['by_iteration'][str(ui + 1)], cube[:, ui].mean(), 'update mean')
    assert summary['changed_actions'] == sum(row['decision_class'] != 'same' for row in decisions)
    for name in ('same', 'wait', 'machine_set', 'assignment'):
        assert summary['decision_classes'][name] == sum(row['decision_class'] == name for row in decisions)
    assert summary['improved'] == int((cube < 0).sum())
    assert summary['worsened'] == int((cube > 0).sum())
    assert summary['tied'] == int((cube == 0).sum())
    assert summary['material_abs_gap_ge_5'] == int((np.abs(cube) >= 5).sum())
    close(summary['mean_absolute_gap'], np.abs(cube).mean(), 'absolute gap')
    for count in (0, 1, 2):
        assert summary['full_cardinality'][str(count)] == sum(int(row['full_cardinality']) == count for row in decisions)
        assert summary['zero_cardinality'][str(count)] == sum(int(row['zero_cardinality']) == count for row in decisions)
    rng = np.random.default_rng(int(source['registry']['bootstrap_seed']))
    draws_out = np.empty(draws)
    for left in range(0, draws, 1000):
        right = min(left + 1000, draws)
        size = right - left
        boot_seeds = rng.integers(0, len(seeds), size=(size, len(seeds)))
        boot_states = rng.integers(0, states, size=(size, len(seeds), updates, len(CASES), states))
        draws_out[left:right] = np.take_along_axis(cube[boot_seeds], boot_states, axis=-1).mean(axis=(1, 2, 3, 4))
    for actual, expected in zip(summary['hierarchical_ci95'], np.quantile(draws_out, [.025, .975])):
        close(actual, expected, 'bootstrap CI')
    seed_means = cube.mean(axis=(1, 2, 3))
    assert summary['directional_signal'] == bool(cube.mean() <= -5 and
                                                 sum(seed_means < 0) >= 4 and
                                                 summary['hierarchical_ci95'][1] < 0)
    return dict(status='PASS', source_run=str(run), diagnostic=str(diagnostic),
                rows=len(output), checkpoints=len(model_cache),
                checks=['input and diagnostic hashes', 'complete training-state grid',
                        'current-state legal mask and term formula',
                        'all actor choices and Q deltas', 'all summary metrics and bootstrap'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result = audit(args.run_dir.resolve(), args.diagnostic_dir.resolve())
    (output / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
