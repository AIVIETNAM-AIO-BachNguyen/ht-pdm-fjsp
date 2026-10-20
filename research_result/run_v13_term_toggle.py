"""Audit-friendly inference toggle on V13's saved training Q banks."""
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

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, catalogue, configs, vector_observation
from ht_pdm_fjsp.maintenance_framework_v13_policy import OpportunityCostActor
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import state_from_dict, submit_requests

ROOT = Path(__file__).resolve().parent.parent
ACTIONS = catalogue()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def actor_at(path: Path) -> OpportunityCostActor:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['metadata']['arm'] == 'competition'
    actor = OpportunityCostActor(competition=True)
    actor.load_state_dict(payload['state_dict'])
    actor.eval()
    return actor


def classify(first: int, second: int) -> str:
    if first == second:
        return 'same'
    if not ACTIONS[first] or not ACTIONS[second]:
        return 'wait'
    if {m for m, _ in ACTIONS[first]} != {m for m, _ in ACTIONS[second]}:
        return 'machine_set'
    return 'assignment'


def observation(row: dict[str, str]) -> np.ndarray:
    cfg = configs()[row['case']]
    state = state_from_dict(json.loads(row['original_state']))
    tick = int(row['tick'])
    submitted = request_action(cfg, state, tick, 'threshold')
    admitted = submit_requests(cfg, state, tick, submitted)
    return vector_observation(cfg, admitted, tick)


def paired_bootstrap(cube: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    """Resample training seeds, then selected states within every case/update cell."""
    rng = np.random.default_rng(seed)
    sample_means = np.empty(draws)
    nseeds, updates, cases, states = cube.shape
    for start in tqdm(range(0, draws, 1000), desc='Hierarchical bootstrap', unit='chunk', leave=False):
        stop = min(start + 1000, draws)
        count = stop - start
        seed_indices = rng.integers(0, nseeds, size=(count, nseeds))
        source = cube[seed_indices]
        state_indices = rng.integers(0, states, size=(count, nseeds, updates, cases, states))
        sample_means[start:stop] = np.take_along_axis(source, state_indices, axis=-1).mean(axis=(1, 2, 3, 4))
    return tuple(float(value) for value in np.quantile(sample_means, [.025, .975]))


def analyze(run: Path, profile: str, out: Path) -> None:
    start = time.monotonic()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED' and manifest['profile'] == profile
    changed_sources = {name for name, digest in manifest['source_hashes'].items()
                       if sha(ROOT / name) != digest}
    # Smoke preceded the final pre-full wording correction of these two documents.
    smoke_docs = {'research_result/framework_v13_dev_protocol_vi.md',
                  'research_result/method_v13_proposal_vi.md'}
    assert changed_sources == (smoke_docs if profile == 'smoke' else set())
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(run / name) == digest
    all_rows = read_csv(run / 'counterfactual_states.csv')
    data = [row for row in all_rows if row['arm'] == 'competition']
    assert len(all_rows) == 2 * len(data)
    if profile == 'dev':
        seeds = tuple(range(72500000, 72500005))
        updates, states, draws = 6, 64, 20000
    else:
        seeds = (72520000,)
        updates, states, draws = 1, 3, 100
    expected = len(seeds) * updates * len(CASES) * states
    assert len(data) == expected
    groups: dict[tuple[int, int, str], list[dict[str, str]]] = {}
    for row in data:
        key = (int(row['train_seed']), int(row['iteration']), row['case'])
        groups.setdefault(key, []).append(row)
    assert len(groups) == len(seeds) * updates * len(CASES)
    assert all(len(cell) == states for cell in groups.values())
    assert set(groups) == {(seed, iteration, case) for seed in seeds
                           for iteration in range(1, updates + 1) for case in CASES}
    cube = np.empty((len(seeds), updates, len(CASES), states))
    output = []
    cache: dict[tuple[int, int], OpportunityCostActor] = {}
    for si, seed in enumerate(seeds):
        for iteration in range(1, updates + 1):
            key = (seed, iteration)
            cache[key] = actor_at(run / 'checkpoints' / 'competition' / str(seed) / f'{iteration - 1}.pt')
            actor = cache[key]
            for ci, case in enumerate(CASES):
                cell = sorted(groups[(seed, iteration, case)], key=lambda row: row['rank'])
                for ri, row in enumerate(tqdm(cell, desc=f'Term toggle {seed}/{iteration}/{case}',
                                               unit='state', leave=False)):
                    obs = torch.as_tensor(observation(row)).reshape(1, 56)
                    legal = [int(value) for value in json.loads(row['legal'])]
                    assert legal == sorted(set(legal)) and legal
                    with torch.no_grad():
                        logits, penalty, _ = actor.components(obs)
                    full = logits[0].numpy()
                    zero = (logits + penalty)[0].numpy()
                    full_action = max(legal, key=lambda action: (float(full[action]), -action))
                    zero_action = max(legal, key=lambda action: (float(zero[action]), -action))
                    scores = {int(action): float(value) for action, value in
                              json.loads(row['scores']).items()}
                    assert set(legal) == set(scores)
                    full_q, zero_q = scores[full_action], scores[zero_action]
                    delta = full_q - zero_q
                    cube[si, iteration - 1, ci, ri] = delta
                    output.append(dict(train_seed=seed, iteration=iteration, case=case,
                                       physical_seed=int(row['physical_seed']), tick=int(row['tick']),
                                       rank=row['rank'], full_action=full_action,
                                       zero_action=zero_action, full_q=full_q,
                                       zero_q=zero_q, delta=delta,
                                       decision_class=classify(full_action, zero_action),
                                       full_cardinality=len(ACTIONS[full_action]),
                                       zero_cardinality=len(ACTIONS[zero_action])))
    assert len(output) == expected and np.isfinite(cube).all()
    ci = paired_bootstrap(cube, draws, int(manifest['registry']['bootstrap_seed']))
    means = cube.mean(axis=(1, 2, 3))
    changed = [row for row in output if row['decision_class'] != 'same']
    summary = dict(status='COMPLETED', profile=profile, rows=expected,
                   mean_delta=float(cube.mean()), hierarchical_ci95=ci,
                   by_train_seed={str(seed): float(means[i]) for i, seed in enumerate(seeds)},
                   by_case={case: float(cube[:, :, i, :].mean()) for i, case in enumerate(CASES)},
                   by_iteration={str(i + 1): float(cube[:, i, :, :].mean())
                                 for i in range(updates)},
                   changed_actions=len(changed),
                   decision_classes={name: sum(row['decision_class'] == name for row in output)
                                     for name in ('same', 'wait', 'machine_set', 'assignment')},
                   improved=sum(row['delta'] < 0 for row in output),
                   worsened=sum(row['delta'] > 0 for row in output),
                   tied=sum(row['delta'] == 0 for row in output),
                   material_abs_gap_ge_5=sum(abs(row['delta']) >= 5 for row in output),
                   mean_absolute_gap=float(np.abs(cube).mean()),
                   full_cardinality={str(i): sum(row['full_cardinality'] == i for row in output)
                                     for i in (0, 1, 2)},
                   zero_cardinality={str(i): sum(row['zero_cardinality'] == i for row in output)
                                     for i in (0, 1, 2)},
                   directional_signal=bool(cube.mean() <= -5 and sum(means < 0) >= 4 and ci[1] < 0))
    with (out / 'rows.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    provenance = dict(status='COMPLETED', source_run=str(run), source_manifest_sha256=sha(run / 'manifest.json'),
                      source_q_sha256=sha(run / 'counterfactual_states.csv'),
                      source_selection_sha256=sha(run / 'selection.json'),
                      script_sha256=sha(Path(__file__)),
                      protocol_sha256=sha(ROOT / 'research_result/v13_component_intervention_protocol_vi.md'),
                      smoke_document_hash_exceptions=sorted(changed_sources), profile=profile,
                      bootstrap_draws=draws, wall_seconds=time.monotonic() - start,
                      artifact_hashes={name: sha(out / name) for name in ('rows.csv', 'summary.json')})
    (out / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile', choices=('smoke', 'dev'), default='dev')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    analyze(args.run_dir.resolve(), args.profile, out)


if __name__ == '__main__':
    main()
