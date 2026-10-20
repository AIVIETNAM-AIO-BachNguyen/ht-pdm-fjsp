"""Independent row/action/aggregation audit for the V14 identity-toggle diagnostic."""
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
from ht_pdm_fjsp.maintenance_framework_v14_policy import OpportunityCostActor
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
    assert manifest['script_sha256'] == sha(ROOT / 'research_result/run_v14_identity_toggle.py')
    assert manifest['protocol_sha256'] == sha(ROOT / 'research_result/v14_identity_toggle_protocol_vi.md')
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(diagnostic / name) == digest
    for name, digest in source['artifact_hashes'].items():
        assert sha(run / name) == digest
    assert source['profile'] == profile
    changed_sources = {name for name, digest in source['source_hashes'].items()
                       if sha(ROOT / name) != digest}
    assert not changed_sources
    assert Path(manifest['source_run']).resolve() == run
    if profile == 'dev':
        main_audit = Path(manifest['main_audit_dir'])
        assert manifest['main_audit_sha256'] == sha(main_audit / 'independent_audit.json')
        report = json.loads((main_audit / 'independent_audit.json').read_text())
        assert report['status'] == 'PASS' and Path(report['run']).resolve() == run
    else:
        assert manifest['main_audit_dir'] is None and manifest['main_audit_sha256'] is None
    raw = csv_rows(run / 'counterfactual_states.csv')
    raw_comp = {row_key(row): row for row in raw if row['arm'] == 'competition'}
    assert len(raw_comp) * 2 == len(raw)
    output = csv_rows(diagnostic / 'rows.csv')
    assert len(output) == len(raw_comp) == int(json.loads((diagnostic / 'summary.json').read_text())['rows'])
    assert {row_key(row) for row in output} == set(raw_comp)
    if profile == 'dev':
        seeds, updates, states, draws = tuple(range(72600000, 72600005)), 6, 64, 20000
    else:
        seeds, updates, states, draws = (72620000,), 1, 3, 100
    assert len(output) == len(seeds) * updates * len(CASES) * states
    assert manifest['bootstrap_draws'] == draws
    cube = np.empty((len(seeds), updates, len(CASES), states))
    model_cache = {}
    decisions = []
    trace = {(int(row['train_seed']), int(row['iteration'])): row['checkpoint_sha256']
             for row in csv_rows(run / 'training_trace.csv') if row['arm'] == 'competition'}
    starts = json.loads((run / 'initial_actor_hashes.json').read_text())
    torch.set_num_threads(1)
    for si, seed in enumerate(seeds):
        for update in range(1, updates + 1):
            model_path = run / 'checkpoints' / 'competition' / str(seed) / f'{update - 1}.pt'
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            assert checkpoint['metadata']['arm'] == 'competition'
            assert manifest['checkpoint_hashes'][f'{seed}/{update-1}'] == sha(model_path)
            if update > 1:
                assert sha(model_path) == trace[(seed, update - 1)]
            joint = OpportunityCostActor(mode='competition')
            joint.load_state_dict(checkpoint['state_dict'])
            joint.eval()
            if update == 1:
                digest = hashlib.sha256()
                for name, value in sorted(joint.state_dict().items()):
                    digest.update(name.encode())
                    digest.update(value.detach().cpu().numpy().tobytes())
                assert digest.hexdigest() == starts[str(seed)]
            count = OpportunityCostActor(mode='count_calibrated')
            count.load_state_dict(joint.state_dict())
            count.eval()
            model_cache[(seed, update)] = (joint, count)
            for ci, case in enumerate(CASES):
                cell = sorted((row for row in output if int(row['train_seed']) == seed and
                               int(row['iteration']) == update and row['case'] == case),
                              key=lambda row: row['rank'])
                assert len(cell) == states
                for ri, record in enumerate(tqdm(cell, desc=f'Independent identity audit {seed}/{update}/{case}',
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
                        joint_logits, joint_penalty, joint_demand = joint.components(torch.as_tensor(obs).reshape(1, 56))
                        count_logits, count_penalty, count_demand = count.components(torch.as_tensor(obs).reshape(1, 56))
                    assert torch.equal(joint_demand, count_demand)
                    duration = torch.as_tensor(obs[40:56]).reshape(1, 4, 2, 2)[..., 1].clamp(min=0)
                    occupancy = torch.einsum('bmj,amj->baj', duration, joint.incidence)
                    exposed = torch.einsum('bmj,am->baj', joint_demand, joint.unserved)
                    assert torch.allclose(joint_penalty, (occupancy * exposed).sum(dim=2), atol=1e-6)
                    cardinality = (1. - joint.unserved).sum(dim=1)
                    factor = (4. - cardinality) / 3.
                    other_demand = joint_demand.sum(dim=1, keepdim=True) - joint_demand
                    expected_count = torch.einsum('bmj,amj->ba', duration * other_demand, joint.incidence) * factor[None, :]
                    assert torch.allclose(count_penalty, expected_count, atol=1e-6)
                    assert torch.equal(joint_logits[:, ACTIONS.index(())], count_logits[:, ACTIONS.index(())])
                    singles = [action for action in legal if len(ACTIONS[action]) == 1]
                    assert torch.allclose(joint_logits[:, singles], count_logits[:, singles], atol=1e-6)
                    joint_scores = joint_logits[0].numpy()
                    count_scores = count_logits[0].numpy()
                    chosen_joint = sorted(legal, key=lambda a: (-float(joint_scores[a]), a))[0]
                    chosen_count = sorted(legal, key=lambda a: (-float(count_scores[a]), a))[0]
                    assert int(record['joint_action']) == chosen_joint
                    assert int(record['count_action']) == chosen_count
                    if chosen_joint != chosen_count:
                        assert len(ACTIONS[chosen_joint]) == 2 or len(ACTIONS[chosen_count]) == 2
                    eligible_double = [action for action in legal if len(ACTIONS[action]) == 2]
                    effect = (joint_penalty - count_penalty)[0].numpy()
                    opportunity = len(eligible_double) >= 2 and np.ptp(effect[eligible_double]) > 1e-6
                    assert record['opportunity'] == str(bool(opportunity))
                    q = {int(a): float(v) for a, v in json.loads(raw_row['scores']).items()}
                    assert set(q) == set(legal)
                    close(record['joint_q'], q[chosen_joint], 'joint Q')
                    close(record['count_q'], q[chosen_count], 'count Q')
                    delta = q[chosen_joint] - q[chosen_count]
                    close(record['delta'], delta, 'Q delta')
                    assert record['decision_class'] == category(chosen_joint, chosen_count)
                    assert int(record['joint_cardinality']) == len(ACTIONS[chosen_joint])
                    assert int(record['count_cardinality']) == len(ACTIONS[chosen_count])
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
    close(summary['changed_fraction'], summary['changed_actions'] / len(decisions), 'changed fraction')
    for name in ('same', 'wait', 'machine_set', 'assignment'):
        assert summary['decision_classes'][name] == sum(row['decision_class'] == name for row in decisions)
    assert summary['improved'] == int((cube < 0).sum())
    assert summary['worsened'] == int((cube > 0).sum())
    assert summary['tied'] == int((cube == 0).sum())
    assert summary['material_abs_gap_ge_5'] == int((np.abs(cube) >= 5).sum())
    close(summary['mean_absolute_gap'], np.abs(cube).mean(), 'absolute gap')
    changed = [row for row in decisions if row['decision_class'] != 'same']
    close(summary['changed_mean_delta'],
          np.mean([float(row['delta']) for row in changed]) if changed else 0.,
          'changed subset mean')
    close(summary['changed_mean_absolute_gap'],
          np.mean([abs(float(row['delta'])) for row in changed]) if changed else 0.,
          'changed subset absolute gap')
    assert summary['changed_improved'] == sum(float(row['delta']) < 0 for row in changed)
    assert summary['changed_worsened'] == sum(float(row['delta']) > 0 for row in changed)
    assert summary['changed_tied'] == sum(float(row['delta']) == 0 for row in changed)
    assert summary['opportunity_rows'] == sum(row['opportunity'] == 'True' for row in decisions)
    assert summary['opportunity_changed'] == sum(row['opportunity'] == 'True' for row in changed)
    close(summary['opportunity_fraction'], summary['opportunity_rows'] / len(decisions),
          'opportunity fraction')
    for count in (0, 1, 2):
        assert summary['joint_cardinality'][str(count)] == sum(int(row['joint_cardinality']) == count for row in decisions)
        assert summary['count_cardinality'][str(count)] == sum(int(row['count_cardinality']) == count for row in decisions)
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
                        'current-state legal mask and both term formulas',
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
