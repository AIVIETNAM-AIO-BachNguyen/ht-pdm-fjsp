"""Prelocked lifecycle-event replay of audited V13 DEV actor and rollout episodes."""
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

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, HORIZON, FrameworkEnv, configs
from ht_pdm_fjsp.maintenance_framework_v13_campaign import V13Controller, V7RuleController
from ht_pdm_fjsp.maintenance_framework_v13_policy import OpportunityCostActor
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / 'research_result/v13_failure_exposure_protocol_v2_vi.md'
SMOKE_DOC_EXCEPTIONS = {'research_result/framework_v13_dev_protocol_vi.md',
                        'research_result/method_v13_proposal_vi.md'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_actor(run: Path, checkpoint: str) -> OpportunityCostActor:
    payload = torch.load(run / checkpoint, map_location='cpu', weights_only=True)
    assert payload['metadata']['arm'] == 'competition'
    actor = OpportunityCostActor(competition=True)
    actor.load_state_dict(payload['state_dict'])
    actor.eval()
    return actor


def replay(case: str, seed: int, controller, train_seed: str, deadline: float) -> tuple[dict, list[dict]]:
    cfg = configs()[case]
    env = FrameworkEnv(cfg)
    env.reset(seed=seed)
    events: list[dict] = []
    pm_starts = cm_starts = upgrades = new_corrective = 0
    pm_age = pm_wait = upgrade_age = upgrade_wait = 0
    pm_count = upgrade_count = 0
    risk_exposed = risk_feasible_exposed = risk_capacity_exposed = risk_served = 0
    failures: list[list[int]] = []
    for tick in range(HORIZON):
        if time.monotonic() >= deadline:
            raise TimeoutError('V13 failure-event replay exceeded 20 minutes')
        before = env.inner.state
        submissions, matching = controller.choose(cfg, before, tick, case=case, seed=seed)
        assert submissions == request_action(cfg, before, tick, 'threshold')
        admitted = submit_requests(cfg, before, tick, submissions)
        available = legal_matchings(cfg, admitted)
        assert matching in available
        served = {m for m, _ in matching}
        for m, request in enumerate(admitted.requests):
            if (request is None or request.started_at is not None or
                    request.kind != 'preventive' or admitted.physical.failed[m] or
                    admitted.physical.ages[m] + 1 < cfg.failure_age):
                continue
            feasible = any(any(machine == m for machine, _ in action) for action in available)
            if m in served:
                risk_served += 1
                risk_event = 'risk_served'
            else:
                risk_exposed += 1
                if feasible:
                    risk_feasible_exposed += 1
                else:
                    risk_capacity_exposed += 1
                risk_event = 'risk_exposed_feasible' if feasible else 'risk_exposed_capacity'
            events.append(dict(case=case, seed=seed, train_seed=train_seed,
                               controller=controller.name, event=risk_event, at=tick,
                               machine=m, kind='preventive', age=int(admitted.physical.ages[m]),
                               wait=tick - int(request.created_at)))
        _, _, terminated, _, info = env.step(env.actions.index(matching))
        after = env.inner.state
        for m, (left, right) in enumerate(zip(before.physical.failed, after.physical.failed)):
            if not left and right:
                failures.append([m, tick + 1])
        for entry in info['lifecycle']:
            event = entry['event']
            at = int(entry['at'])
            machine = int(entry['machine'])
            kind = entry['kind']
            if event in ('created', 'started'):
                age = int(before.physical.ages[machine])
            else:
                age = int(after.physical.ages[machine])
            wait = at - int(entry['created_at'])
            if event == 'started' and kind == 'preventive':
                pm_starts += 1
                pm_age += age
                pm_wait += wait
                pm_count += 1
            elif event == 'started' and kind == 'corrective':
                cm_starts += 1
            elif event == 'upgraded':
                upgrades += 1
                upgrade_age += age
                upgrade_wait += wait
                upgrade_count += 1
            elif event == 'created' and kind == 'corrective':
                new_corrective += 1
            events.append(dict(case=case, seed=seed, train_seed=train_seed,
                               controller=controller.name, event=event, at=at,
                               machine=machine, kind=kind, age=age, wait=wait))
    assert terminated
    metrics = env.episode_metrics()
    assert pm_starts + cm_starts == int(env.cutoff['service_starts'])
    assert upgrades == int(env.cutoff['requests_upgraded'])
    # All lifecycle rows above are from H48; the environment subsequently runs its fixed tail.
    row = dict(case=case, seed=seed, train_seed=train_seed, controller=controller.name,
               H_upgrades=upgrades, H_new_corrective=new_corrective,
               H_failure_residual=int(metrics['H_failures']) - upgrades - new_corrective,
               H_pm_starts=pm_starts, H_cm_starts=cm_starts,
               H_risk_exposed=risk_exposed,
               H_risk_feasible_exposed=risk_feasible_exposed,
               H_risk_capacity_exposed=risk_capacity_exposed,
               H_risk_served=risk_served,
               H_pm_start_age_sum=pm_age, H_pm_start_wait_sum=pm_wait,
               H_upgrade_age_sum=upgrade_age, H_upgrade_wait_sum=upgrade_wait,
               H_pm_start_count=pm_count, H_upgrade_count=upgrade_count,
               first_failure_tick=min((at for _, at in failures), default=''),
               failure_events=json.dumps(failures, separators=(',', ':')),
               H_failures=float(metrics['H_failures']),
               H_service_starts=pm_starts + cm_starts,
               H_availability=float(metrics['H_availability']),
               objective=float(metrics['objective']),
               failure_cost=float(metrics['failure_cost']),
               service_starts=float(metrics['service_starts']),
               requests_upgraded=int(metrics['requests_upgraded']))
    return row, events


def bootstrap(cube: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    n_train, n_physical = cube.shape
    for start in tqdm(range(0, draws, 1000), desc='Event bootstrap', unit='chunk', leave=False):
        stop = min(start + 1000, draws)
        count = stop - start
        train = rng.integers(0, n_train, size=(count, n_train))
        physical = rng.integers(0, n_physical, size=(count, n_train, n_physical))
        out[start:stop] = np.take_along_axis(cube[train], physical, axis=-1).mean(axis=(1, 2))
    return tuple(float(x) for x in np.quantile(out, [.025, .975]))


def summarize(rows: list[dict], profile: str) -> dict:
    seeds = tuple(range(72500000, 72500005)) if profile == 'dev' else (72520000,)
    physicals = tuple(range(72510000, 72510016)) if profile == 'dev' else tuple(range(72520000, 72520002))
    by_key = {(r['case'], r['train_seed'], int(r['seed']), r['controller']): r for r in rows}
    targets = ('interchangeable_scale1', 'reciprocal_scale2')
    result = dict(status='COMPLETED', profile=profile, episodes=len(rows),
                  lifecycle_events=None, by_case={}, targets={})
    fields = ('H_risk_exposed', 'H_risk_feasible_exposed', 'H_risk_capacity_exposed',
              'H_risk_served', 'H_upgrades', 'H_new_corrective', 'H_failure_residual', 'H_pm_starts',
              'H_cm_starts', 'H_failures', 'H_service_starts', 'H_availability',
              'objective', 'failure_cost')
    for case in CASES:
        cand = [by_key[(case, str(seed), p, 'competition')] for seed in seeds for p in physicals]
        roll = [by_key[(case, '', p, 'rollout')] for p in physicals]
        means = {name: {field: float(np.mean([float(r[field]) for r in group]))
                        for field in fields} for name, group in
                 (('competition', cand), ('rollout', roll))}
        deltas = {field: means['competition'][field] - means['rollout'][field] for field in fields}
        result['by_case'][case] = dict(means=means, candidate_minus_rollout=deltas,
                                       feasible_fraction_of_exposure={name: float(
                                           means[name]['H_risk_feasible_exposed'] /
                                           means[name]['H_risk_exposed'])
                                           if means[name]['H_risk_exposed'] > 0 else None
                                           for name in ('competition', 'rollout')},
                                       pm_start_age_mean={name: float(sum(float(r['H_pm_start_age_sum']) for r in group) /
                                                   max(1, sum(int(r['H_pm_start_count']) for r in group)))
                                                          for name, group in (('competition', cand), ('rollout', roll))},
                                       pm_start_wait_mean={name: float(sum(float(r['H_pm_start_wait_sum']) for r in group) /
                                                   max(1, sum(int(r['H_pm_start_count']) for r in group)))
                                                           for name, group in (('competition', cand), ('rollout', roll))},
                                       upgrade_age_mean={name: float(sum(float(r['H_upgrade_age_sum']) for r in group) /
                                                   max(1, sum(int(r['H_upgrade_count']) for r in group)))
                                                         for name, group in (('competition', cand), ('rollout', roll))},
                                       upgrade_wait_mean={name: float(sum(float(r['H_upgrade_wait_sum']) for r in group) /
                                                   max(1, sum(int(r['H_upgrade_count']) for r in group)))
                                                          for name, group in (('competition', cand), ('rollout', roll))})
        if case in targets:
            cube = np.array([[float(by_key[(case, str(seed), p, 'competition')]['H_risk_exposed']) -
                              float(by_key[(case, '', p, 'rollout')]['H_risk_exposed'])
                              for p in physicals] for seed in seeds])
            seed_means = cube.mean(axis=1)
            excess_failures = deltas['H_failures']
            ci = bootstrap(cube, 20000 if profile == 'dev' else 100, 72540001)
            result['targets'][case] = dict(exposure_delta=float(cube.mean()),
                                            by_train_seed={str(seed): float(seed_means[i])
                                                           for i, seed in enumerate(seeds)},
                                            hierarchical_ci95=ci,
                                            excess_exposure_gate=bool(cube.mean() >= 3 and ci[0] > 0 and
                                                                      sum(seed_means > 0) >=
                                                                      (4 if profile == 'dev' else 1)))
    result['joint_gate'] = all(x['excess_exposure_gate'] for x in result['targets'].values())
    return result


def analyze(run: Path, out: Path, profile: str) -> None:
    start = time.monotonic()
    deadline = start + (1200 if profile == 'dev' else 300)
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED' and manifest['profile'] == profile
    changed = {name for name, digest in manifest['source_hashes'].items()
               if sha(ROOT / name) != digest}
    assert changed == (SMOKE_DOC_EXCEPTIONS if profile == 'smoke' else set())
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(run / name) == digest
    source = read_rows(run / 'episodes.csv')
    seeds = tuple(range(72500000, 72500005)) if profile == 'dev' else (72520000,)
    physicals = tuple(range(72510000, 72510016)) if profile == 'dev' else tuple(range(72520000, 72520002))
    selection = json.loads((run / 'selection.json').read_text())
    actors = {}
    checkpoint_hashes = {}
    for seed in seeds:
        path = selection['competition'][str(seed)]['checkpoint']
        checkpoint_hashes[str(seed)] = sha(run / path)
        actors[seed] = selected_actor(run, path)
    selected = {(r['case'], r['train_seed'], int(r['seed']), r['controller']): r
                for r in source if r['controller'] in ('competition', 'rollout')}
    rows: list[dict] = []
    events: list[dict] = []
    torch.set_num_threads(1)
    grid = [(case, seed, physical) for case in CASES for seed in seeds for physical in physicals]
    for case, seed, physical in tqdm(grid, desc='V13 event replay', unit='candidate'):
        expected = selected[(case, str(seed), physical, 'competition')]
        actual, lifecycle = replay(case, physical, V13Controller(actors[seed], 'competition'),
                                   str(seed), deadline)
        for field, source_field in (('objective', 'objective'), ('failure_cost', 'failure_cost'),
                                    ('H_failures', 'H_failures'), ('H_availability', 'H_availability'),
                                    ('service_starts', 'service_starts'),
                                    ('requests_upgraded', 'requests_upgraded')):
            assert np.isclose(float(actual[field]), float(expected[source_field]), rtol=0, atol=1e-7)
        rows.append(actual)
        events.extend(lifecycle)
    for case in tqdm(CASES, desc='V13 rollout event replay', unit='case'):
        for physical in physicals:
            expected = selected[(case, '', physical, 'rollout')]
            controller = V7RuleController('rollout', int(manifest['registry']['forecast_seed']),
                                          lambda: None if time.monotonic() < deadline else
                                          (_ for _ in ()).throw(TimeoutError('event replay timeout')))
            actual, lifecycle = replay(case, physical, controller, '', deadline)
            for field in ('objective', 'failure_cost', 'H_failures', 'H_availability',
                          'service_starts', 'requests_upgraded'):
                assert np.isclose(float(actual[field]), float(expected[field]), rtol=0, atol=1e-7)
            rows.append(actual)
            events.extend(lifecycle)
    assert len(rows) == len(CASES) * len(physicals) * (len(seeds) + 1)
    summary = summarize(rows, profile)
    summary['lifecycle_events'] = len(events)
    write_rows(out / 'rows.csv', rows)
    write_rows(out / 'events.csv', events)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    provenance = dict(status='COMPLETED', profile=profile, source_run=str(run),
                      source_manifest_sha256=sha(run / 'manifest.json'),
                      source_episode_sha256=sha(run / 'episodes.csv'),
                      source_selection_sha256=sha(run / 'selection.json'),
                      checkpoint_hashes=checkpoint_hashes,
                      script_sha256=sha(Path(__file__)), protocol_sha256=sha(PROTOCOL),
                      source_document_hash_exceptions=sorted(changed),
                      bootstrap_seed=72540001,
                      bootstrap_draws=20000 if profile == 'dev' else 100,
                      wall_seconds=time.monotonic() - start,
                      artifact_hashes={name: sha(out / name) for name in
                                       ('rows.csv', 'events.csv', 'summary.json')})
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
    analyze(args.run_dir.resolve(), out, args.profile)


if __name__ == '__main__':
    main()
