"""Independent full-episode/action/event audit of the V13 exposure diagnostic."""
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

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, HORIZON, FrameworkEnv, catalogue, configs, vector_observation
from ht_pdm_fjsp.maintenance_framework_v13_policy import OpportunityCostActor
from ht_pdm_fjsp.maintenance_framework_v7_campaign import V7RuleController
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests

ROOT = Path(__file__).resolve().parent.parent
ACTIONS = catalogue()
DOC_EXCEPTIONS = {'research_result/framework_v13_dev_protocol_vi.md',
                  'research_result/method_v13_proposal_vi.md'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def close(left, right, label: str) -> None:
    if not np.isclose(float(left), float(right), rtol=0, atol=1e-8):
        raise AssertionError(f'{label}: {left} != {right}')


def episode(case: str, seed: int, train: str, controller, actor,
            deadline: float) -> tuple[dict, list[dict]]:
    cfg = configs()[case]
    env = FrameworkEnv(cfg)
    env.reset(seed=seed)
    records = []
    counts = dict(H_risk_exposed=0, H_risk_feasible_exposed=0,
                  H_risk_capacity_exposed=0, H_risk_served=0,
                  H_upgrades=0, H_new_corrective=0, H_pm_starts=0,
                  H_cm_starts=0, H_pm_start_age_sum=0, H_pm_start_wait_sum=0,
                  H_upgrade_age_sum=0, H_upgrade_wait_sum=0,
                  H_pm_start_count=0, H_upgrade_count=0)
    failure_events = []
    for tick in range(HORIZON):
        if time.monotonic() >= deadline:
            raise TimeoutError('V13 independent exposure audit exceeded 20 minutes')
        before = env.inner.state
        submitted = request_action(cfg, before, tick, 'threshold')
        admitted = submit_requests(cfg, before, tick, submitted)
        legal = legal_matchings(cfg, admitted)
        if actor is not None:
            observation = vector_observation(cfg, admitted, tick)
            with torch.no_grad():
                logits = actor(torch.as_tensor(observation).reshape(1, 56))[0].numpy()
            masked = torch.full((len(ACTIONS),), -1e9, dtype=torch.float32)
            legal_indices = [ACTIONS.index(action) for action in legal]
            masked[legal_indices] = torch.as_tensor(logits)[legal_indices]
            choice = int(np.argmax(torch.softmax(masked, dim=0).numpy()))
            matching = ACTIONS[choice]
            name = 'competition'
        else:
            selections, matching = controller.choose(cfg, before, tick, case=case, seed=seed)
            assert selections == submitted
            name = 'rollout'
        assert matching in legal
        served = {m for m, _ in matching}
        for m, request in enumerate(admitted.requests):
            if request is None or request.started_at is not None or request.kind != 'preventive':
                continue
            if admitted.physical.failed[m] or admitted.physical.ages[m] + 1 < cfg.failure_age:
                continue
            feasible = any(any(mm == m for mm, _ in action) for action in legal)
            if m in served:
                event = 'risk_served'
                counts['H_risk_served'] += 1
            elif feasible:
                event = 'risk_exposed_feasible'
                counts['H_risk_exposed'] += 1
                counts['H_risk_feasible_exposed'] += 1
            else:
                event = 'risk_exposed_capacity'
                counts['H_risk_exposed'] += 1
                counts['H_risk_capacity_exposed'] += 1
            records.append(dict(case=case, seed=seed, train_seed=train,
                                controller=name, event=event, at=tick,
                                machine=m, kind='preventive', age=int(admitted.physical.ages[m]),
                                wait=tick - int(request.created_at)))
        _, _, done, _, info = env.step(ACTIONS.index(matching))
        after = env.inner.state
        for m in range(cfg.machines):
            if not before.physical.failed[m] and after.physical.failed[m]:
                failure_events.append([m, tick + 1])
        for life in info['lifecycle']:
            event = life['event']
            at = int(life['at'])
            m = int(life['machine'])
            kind = life['kind']
            age = int(before.physical.ages[m] if event in ('created', 'started')
                      else after.physical.ages[m])
            wait = at - int(life['created_at'])
            if event == 'started':
                if kind == 'preventive':
                    counts['H_pm_starts'] += 1
                    counts['H_pm_start_age_sum'] += age
                    counts['H_pm_start_wait_sum'] += wait
                    counts['H_pm_start_count'] += 1
                else:
                    counts['H_cm_starts'] += 1
            elif event == 'upgraded':
                counts['H_upgrades'] += 1
                counts['H_upgrade_age_sum'] += age
                counts['H_upgrade_wait_sum'] += wait
                counts['H_upgrade_count'] += 1
            elif event == 'created' and kind == 'corrective':
                counts['H_new_corrective'] += 1
            records.append(dict(case=case, seed=seed, train_seed=train,
                                controller=name, event=event, at=at, machine=m,
                                kind=kind, age=age, wait=wait))
    assert done
    measured = env.episode_metrics()
    assert counts['H_pm_starts'] + counts['H_cm_starts'] == int(env.cutoff['service_starts'])
    assert counts['H_upgrades'] == int(env.cutoff['requests_upgraded'])
    return dict(case=case, seed=seed, train_seed=train, controller=name,
                **{'H_failure_residual': int(measured['H_failures']) -
                   counts['H_upgrades'] - counts['H_new_corrective']},
                **counts, first_failure_tick=min((at for _, at in failure_events), default=''),
                failure_events=json.dumps(failure_events, separators=(',', ':')),
                H_failures=float(measured['H_failures']),
                H_service_starts=counts['H_pm_starts'] + counts['H_cm_starts'],
                H_availability=float(measured['H_availability']),
                objective=float(measured['objective']),
                failure_cost=float(measured['failure_cost']),
                service_starts=float(measured['service_starts']),
                requests_upgraded=int(measured['requests_upgraded'])), records


def audit(source: Path, diagnostic: Path) -> dict:
    started = time.monotonic()
    deadline = started + 1200
    manifest = json.loads((diagnostic / 'manifest.json').read_text())
    original = json.loads((source / 'manifest.json').read_text())
    profile = manifest['profile']
    assert profile in ('smoke', 'dev')
    assert original['status'] == manifest['status'] == 'COMPLETED'
    assert original['profile'] == profile
    assert Path(manifest['source_run']).resolve() == source
    assert manifest['source_manifest_sha256'] == sha(source / 'manifest.json')
    assert manifest['source_episode_sha256'] == sha(source / 'episodes.csv')
    assert manifest['source_selection_sha256'] == sha(source / 'selection.json')
    assert manifest['script_sha256'] == sha(ROOT / 'research_result/run_v13_failure_events.py')
    assert manifest['protocol_sha256'] == sha(ROOT / 'research_result/v13_failure_exposure_protocol_v2_vi.md')
    changed = {name for name, digest in original['source_hashes'].items()
               if sha(ROOT / name) != digest}
    assert changed == (DOC_EXCEPTIONS if profile == 'smoke' else set())
    assert manifest['source_document_hash_exceptions'] == sorted(changed)
    for name, digest in original['artifact_hashes'].items():
        assert sha(source / name) == digest
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(diagnostic / name) == digest
    selected = json.loads((source / 'selection.json').read_text())
    seeds = tuple(range(72500000, 72500005)) if profile == 'dev' else (72520000,)
    physicals = tuple(range(72510000, 72510016)) if profile == 'dev' else (72520000, 72520001)
    models = {}
    for seed in seeds:
        path = selected['competition'][str(seed)]['checkpoint']
        assert manifest['checkpoint_hashes'][str(seed)] == sha(source / path)
        payload = torch.load(source / path, map_location='cpu', weights_only=True)
        assert payload['metadata']['arm'] == 'competition'
        model = OpportunityCostActor(competition=True)
        model.load_state_dict(payload['state_dict'])
        model.eval()
        models[seed] = model
    reported = {(r['case'], r['train_seed'], int(r['seed']), r['controller']): r
                for r in rows(diagnostic / 'rows.csv')}
    original_rows = {(r['case'], r['train_seed'], int(r['seed']), r['controller']): r
                     for r in rows(source / 'episodes.csv') if r['controller'] in ('competition', 'rollout')}
    event_rows = rows(diagnostic / 'events.csv')
    offset = 0
    assert len(reported) == len(CASES) * len(physicals) * (len(seeds) + 1)
    torch.set_num_threads(1)
    grid = [(case, seed, physical) for case in CASES for seed in seeds for physical in physicals]
    grid += [(case, None, physical) for case in CASES for physical in physicals]
    for case, seed, physical in tqdm(grid, desc='Independent V13 event audit', unit='episode'):
        train = str(seed) if seed is not None else ''
        controller = None if seed is not None else V7RuleController(
            'rollout', int(original['registry']['forecast_seed']),
            lambda: None if time.monotonic() < deadline else
            (_ for _ in ()).throw(TimeoutError('V13 event audit timeout')))
        expected, lifecycle = episode(case, physical, train, controller,
                                      models[seed] if seed is not None else None, deadline)
        key = (case, train, physical, 'competition' if seed is not None else 'rollout')
        actual = reported[key]
        src = original_rows[key]
        for field, value in expected.items():
            if field in ('case', 'train_seed', 'controller', 'failure_events'):
                assert actual[field] == str(value), (key, field, actual[field], value)
            elif field == 'first_failure_tick' and value == '':
                assert actual[field] == ''
            else:
                close(actual[field], value, field)
        for field in ('objective', 'failure_cost', 'H_failures', 'H_availability',
                      'service_starts', 'requests_upgraded'):
            close(expected[field], src[field], 'source ' + field)
        for item in lifecycle:
            row = event_rows[offset]
            offset += 1
            for field, value in item.items():
                if field in ('case', 'train_seed', 'controller', 'event', 'kind'):
                    assert row[field] == str(value)
                else:
                    close(row[field], value, 'event ' + field)
    assert offset == len(event_rows)
    summary = json.loads((diagnostic / 'summary.json').read_text())
    assert summary['episodes'] == len(reported)
    assert summary['lifecycle_events'] == len(event_rows)
    fields = ('H_risk_exposed', 'H_risk_feasible_exposed', 'H_risk_capacity_exposed',
              'H_risk_served', 'H_upgrades', 'H_new_corrective', 'H_failure_residual',
              'H_pm_starts', 'H_cm_starts', 'H_failures', 'H_service_starts',
              'H_availability', 'objective', 'failure_cost')
    for case in CASES:
        cand = [reported[(case, str(seed), p, 'competition')] for seed in seeds for p in physicals]
        roll = [reported[(case, '', p, 'rollout')] for p in physicals]
        for field in fields:
            c_mean = np.mean([float(row[field]) for row in cand])
            r_mean = np.mean([float(row[field]) for row in roll])
            close(summary['by_case'][case]['means']['competition'][field], c_mean, field)
            close(summary['by_case'][case]['means']['rollout'][field], r_mean, field)
            close(summary['by_case'][case]['candidate_minus_rollout'][field], c_mean-r_mean, field)
        for controller, group in (('competition', cand), ('rollout', roll)):
            exposure = sum(float(row['H_risk_exposed']) for row in group)
            feasible = sum(float(row['H_risk_feasible_exposed']) for row in group)
            expected_fraction = feasible / exposure if exposure else None
            reported_fraction = summary['by_case'][case]['feasible_fraction_of_exposure'][controller]
            if expected_fraction is None:
                assert reported_fraction is None
            else:
                close(reported_fraction, expected_fraction, 'feasible exposure fraction')
            for name, sum_key, count_key in (
                ('pm_start_age_mean', 'H_pm_start_age_sum', 'H_pm_start_count'),
                ('pm_start_wait_mean', 'H_pm_start_wait_sum', 'H_pm_start_count'),
                ('upgrade_age_mean', 'H_upgrade_age_sum', 'H_upgrade_count'),
                ('upgrade_wait_mean', 'H_upgrade_wait_sum', 'H_upgrade_count')):
                expected_mean = sum(float(r[sum_key]) for r in group) / max(1, sum(int(r[count_key]) for r in group))
                close(summary['by_case'][case][name][controller], expected_mean, name)
        if case in ('interchangeable_scale1', 'reciprocal_scale2'):
            cube = np.array([[float(reported[(case, str(seed), p, 'competition')]['H_risk_exposed'])-
                              float(reported[(case, '', p, 'rollout')]['H_risk_exposed'])
                              for p in physicals] for seed in seeds])
            target = summary['targets'][case]
            close(target['exposure_delta'], cube.mean(), 'exposure mean')
            for i, seed in enumerate(seeds):
                close(target['by_train_seed'][str(seed)], cube[i].mean(), 'seed mean')
            draws = int(manifest['bootstrap_draws'])
            rng = np.random.default_rng(int(manifest['bootstrap_seed']))
            values = np.empty(draws)
            for left in range(0, draws, 1000):
                right = min(left + 1000, draws)
                train_ix = rng.integers(0, len(seeds), size=(right-left, len(seeds)))
                phys_ix = rng.integers(0, len(physicals), size=(right-left, len(seeds), len(physicals)))
                values[left:right] = np.take_along_axis(cube[train_ix], phys_ix, axis=-1).mean(axis=(1,2))
            for got, expected_ci in zip(target['hierarchical_ci95'], np.quantile(values,[.025,.975])):
                close(got, expected_ci, 'bootstrap CI')
            assert target['excess_exposure_gate'] == bool(cube.mean() >= 3 and
                 np.quantile(values,.025) > 0 and sum(cube.mean(axis=1)>0) >= (4 if profile=='dev' else 1))
    assert summary['joint_gate'] == all(x['excess_exposure_gate'] for x in summary['targets'].values())
    return dict(status='PASS', source_run=str(source), diagnostic=str(diagnostic),
                episodes=len(reported), event_rows=len(event_rows), checkpoints=len(models),
                checks=['source, checkpoint and output hashes', 'all actor/planner actions and H48 events',
                        'source H64 episode metrics', 'all case summaries and paired bootstrap'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    result = audit(args.run_dir.resolve(), args.diagnostic_dir.resolve())
    (out / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
