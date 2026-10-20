"""Prelocked V14 same-weight joint-versus-count identity diagnostic."""
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
from ht_pdm_fjsp.maintenance_framework_v14_policy import OpportunityCostActor
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
    actor = OpportunityCostActor(mode='competition')
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


def analyze(run: Path, profile: str, out: Path, main_audit: Path | None) -> None:
    start = time.monotonic()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED' and manifest['profile'] == profile
    if profile == 'dev':
        assert main_audit is not None
        audit = json.loads((main_audit / 'independent_audit.json').read_text())
        assert audit['status'] == 'PASS' and Path(audit['run']).resolve() == run
    else:
        assert main_audit is None
    assert all(sha(ROOT / name) == digest for name, digest in manifest['source_hashes'].items())
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(run / name) == digest
    all_rows = read_csv(run / 'counterfactual_states.csv')
    data = [row for row in all_rows if row['arm'] == 'competition']
    assert len(all_rows) == 2 * len(data)
    if profile == 'dev':
        seeds = tuple(range(72600000, 72600005))
        updates, states, draws = 6, 64, 20000
    else:
        seeds = (72620000,)
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
    checkpoint_hashes = {}
    trace = {(int(row['train_seed']), int(row['iteration'])): row['checkpoint_sha256']
             for row in read_csv(run / 'training_trace.csv') if row['arm'] == 'competition'}
    starts = json.loads((run / 'initial_actor_hashes.json').read_text())
    cache: dict[tuple[int, int], tuple[OpportunityCostActor, OpportunityCostActor]] = {}
    for si, seed in enumerate(seeds):
        for iteration in range(1, updates + 1):
            key = (seed, iteration)
            checkpoint = run / 'checkpoints' / 'competition' / str(seed) / f'{iteration - 1}.pt'
            if iteration > 1:
                assert sha(checkpoint) == trace[(seed, iteration - 1)]
            joint = actor_at(checkpoint)
            if iteration == 1:
                digest = hashlib.sha256()
                for name, value in sorted(joint.state_dict().items()):
                    digest.update(name.encode())
                    digest.update(value.detach().cpu().numpy().tobytes())
                assert digest.hexdigest() == starts[str(seed)]
            count = OpportunityCostActor(mode='count_calibrated')
            count.load_state_dict(joint.state_dict())
            count.eval()
            cache[key] = joint, count
            checkpoint_hashes[f'{seed}/{iteration-1}'] = sha(checkpoint)
            for ci, case in enumerate(CASES):
                cell = sorted(groups[(seed, iteration, case)], key=lambda row: row['rank'])
                for ri, row in enumerate(tqdm(cell, desc=f'Identity toggle {seed}/{iteration}/{case}',
                                               unit='state', leave=False)):
                    obs = torch.as_tensor(observation(row)).reshape(1, 56)
                    legal = [int(value) for value in json.loads(row['legal'])]
                    assert legal == sorted(set(legal)) and legal
                    if time.monotonic() - start > 1200:
                        raise TimeoutError('V14 identity diagnostic exceeded 20 minutes')
                    joint, count = cache[key]
                    with torch.no_grad():
                        joint_logits, joint_penalty, joint_demand = joint.components(obs)
                        count_logits, count_penalty, count_demand = count.components(obs)
                    assert torch.equal(joint_demand, count_demand)
                    assert torch.equal(joint_logits[:, ACTIONS.index(())],
                                       count_logits[:, ACTIONS.index(())])
                    singles = [action for action in legal if len(ACTIONS[action]) == 1]
                    assert torch.allclose(joint_logits[:, singles], count_logits[:, singles], atol=1e-6)
                    joint_scores = joint_logits[0].numpy()
                    count_scores = count_logits[0].numpy()
                    joint_action = max(legal, key=lambda action: (float(joint_scores[action]), -action))
                    count_action = max(legal, key=lambda action: (float(count_scores[action]), -action))
                    if joint_action != count_action:
                        assert len(ACTIONS[joint_action]) == 2 or len(ACTIONS[count_action]) == 2
                    eligible_double = [action for action in legal if len(ACTIONS[action]) == 2]
                    effect = (joint_penalty - count_penalty)[0].numpy()
                    opportunity = (len(eligible_double) >= 2 and
                                   np.ptp(effect[eligible_double]) > 1e-6)
                    scores = {int(action): float(value) for action, value in
                              json.loads(row['scores']).items()}
                    assert set(legal) == set(scores)
                    joint_q, count_q = scores[joint_action], scores[count_action]
                    delta = joint_q - count_q
                    cube[si, iteration - 1, ci, ri] = delta
                    output.append(dict(train_seed=seed, iteration=iteration, case=case,
                                       physical_seed=int(row['physical_seed']), tick=int(row['tick']),
                                       rank=row['rank'], joint_action=joint_action,
                                       count_action=count_action, joint_q=joint_q,
                                       count_q=count_q, delta=delta,
                                       decision_class=classify(joint_action, count_action),
                                       joint_cardinality=len(ACTIONS[joint_action]),
                                       count_cardinality=len(ACTIONS[count_action]),
                                       opportunity=bool(opportunity)))
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
                   changed_fraction=len(changed) / expected,
                   decision_classes={name: sum(row['decision_class'] == name for row in output)
                                     for name in ('same', 'wait', 'machine_set', 'assignment')},
                   improved=sum(row['delta'] < 0 for row in output),
                   worsened=sum(row['delta'] > 0 for row in output),
                   tied=sum(row['delta'] == 0 for row in output),
                   material_abs_gap_ge_5=sum(abs(row['delta']) >= 5 for row in output),
                   mean_absolute_gap=float(np.abs(cube).mean()),
                   changed_mean_delta=float(np.mean([row['delta'] for row in changed])) if changed else 0.,
                   changed_mean_absolute_gap=float(np.mean([abs(row['delta']) for row in changed])) if changed else 0.,
                   changed_improved=sum(row['delta'] < 0 for row in changed),
                   changed_worsened=sum(row['delta'] > 0 for row in changed),
                   changed_tied=sum(row['delta'] == 0 for row in changed),
                   opportunity_rows=sum(row['opportunity'] for row in output),
                   opportunity_fraction=sum(row['opportunity'] for row in output) / expected,
                   opportunity_changed=sum(row['opportunity'] for row in changed),
                   joint_cardinality={str(i): sum(row['joint_cardinality'] == i for row in output)
                                      for i in (0, 1, 2)},
                   count_cardinality={str(i): sum(row['count_cardinality'] == i for row in output)
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
                      protocol_sha256=sha(ROOT / 'research_result/v14_identity_toggle_protocol_vi.md'),
                      checkpoint_hashes=checkpoint_hashes, profile=profile,
                      main_audit_dir=str(main_audit) if main_audit else None,
                      main_audit_sha256=sha(main_audit / 'independent_audit.json') if main_audit else None,
                      bootstrap_draws=draws, wall_seconds=time.monotonic() - start,
                      artifact_hashes={name: sha(out / name) for name in ('rows.csv', 'summary.json')})
    (out / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile', choices=('smoke', 'dev'), default='dev')
    parser.add_argument('--main-audit-dir', type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    analyze(args.run_dir.resolve(), args.profile, out,
            args.main_audit_dir.resolve() if args.main_audit_dir else None)


if __name__ == '__main__':
    main()
