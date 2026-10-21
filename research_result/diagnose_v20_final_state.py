"""Opened-DEV selected V20 divergence, full-matching Q, and gate diagnostic."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import CASES, HORIZON, FrameworkEnv, configs
from ht_pdm_fjsp.maintenance_framework_v14_campaign import (
    load_checkpoint as load_v14, masked_probabilities as v14_probabilities,
    policy_action as v14_action,
)
from ht_pdm_fjsp.maintenance_framework_v19_campaign import episode_seed as v19_episode_seed
from ht_pdm_fjsp.maintenance_framework_v20_campaign import (
    ACTIONS, V14, V7, episode_seed as v20_episode_seed, legal_present,
    load_checkpoint as load_v20, masked_probabilities as v20_probabilities,
    policy_action as v20_action,
)
from ht_pdm_fjsp.maintenance_framework_v14_campaign import episode_seed as v14_episode_seed
from ht_pdm_fjsp.maintenance_framework_v7_campaign import V7RuleController
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
from ht_pdm_fjsp.maintenance_request_policies import risk_matching
from ht_pdm_fjsp.maintenance_requests import state_from_dict, transition

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'research_result/framework_v20_dev_20261021T125049Z'
SOURCE_AUDIT = ROOT / 'research_result/framework_v20_audit_20261021T152700Z/independent_audit.json'
BASELINE_AUDIT = ROOT / 'research_result/framework_v20_baseline_audit_20261021T152701Z/baseline_replay_audit.json'
PLAN = ROOT / 'research_result/v20_final_state_diagnostic_plan_vi.md'
TRAIN = tuple(range(73700000, 73700005))
V14_TRAIN = tuple(range(72600000, 72600005))
PHYSICAL = tuple(range(73711000, 73711016))
OTHERS = ('generic_q', 'count_calibrated', 'rollout')
ANCHOR = 74070000
BOOTSTRAP = 74070001
METRICS = (
    'objective', 'maintenance_cost', 'unavailability_cost', 'failure_cost',
    'tail_cost', 'H_availability', 'H_failures', 'queued_cm_ticks',
    'queued_pm_ticks', 'H_pending', 'end_pending', 'service_starts',
    'requests_created', 'requests_completed', 'legal_service_ticks',
    'legal_service_waits', 'legal_service_wait_fraction',
)
DELTA_METRICS = (
    ('cost', 'objective'), ('failures', 'H_failures'),
    ('pm_queue', 'queued_pm_ticks'), ('cm_queue', 'queued_cm_ticks'),
    ('starts', 'service_starts'), ('availability', 'H_availability'),
)
CATEGORIES = (
    'same', 'v20_service_other_wait', 'v20_wait_other_service',
    'v20_higher_cardinality', 'v20_lower_cardinality', 'machine_set', 'assignment',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def read(path):
    with Path(path).open(newline='') as file:
        return list(csv.DictReader(file))


def write(path, rows):
    if not rows:
        raise ValueError(f'empty artifact {path}')
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close_metrics(actual, recorded):
    for key in METRICS:
        if not np.isclose(float(actual[key]), float(recorded[key]), rtol=0, atol=1e-8):
            raise AssertionError((key, actual[key], recorded[key]))


def check_seed_roles():
    used = set()
    for registry in (ROOT / 'configs').glob('*seed_registry.json'):
        used |= registered_integers(json.loads(registry.read_text()))
    for function, start, updates in (
        (v20_episode_seed, 73700000, 8),
        (v19_episode_seed, 73400000, 6),
        (v14_episode_seed, 72600000, 6),
    ):
        for seed in range(start, start + 5):
            for iteration in range(1, updates + 1):
                for case in CASES:
                    for index in range(16):
                        used.add(function(seed, iteration, case, index))
    assert ANCHOR != BOOTSTRAP and not {ANCHOR, BOOTSTRAP} & used
    return len(used)


def classify(left, right):
    a, b = ACTIONS[left], ACTIONS[right]
    if a == b:
        return 'same'
    if a and not b:
        return 'v20_service_other_wait'
    if b and not a:
        return 'v20_wait_other_service'
    if len(a) != len(b):
        return 'v20_higher_cardinality' if len(a) > len(b) else 'v20_lower_cardinality'
    return 'machine_set' if {m for m, _ in a} != {m for m, _ in b} else 'assignment'


def choose(arm, model, config, state, tick, case, physical, legal, obs, forecast_anchor, deadline):
    if arm in ('activation_margin', 'generic_q'):
        return v20_action(model, obs, legal)
    if arm == 'count_calibrated':
        return v14_action(model, obs, legal)
    if arm == 'rollout':
        controller = V7RuleController('rollout', forecast_anchor, deadline)
        submissions, matching = controller.choose(config, state, tick, case=case, seed=physical)
        assert matching in ACTIONS
        return ACTIONS.index(matching), submissions
    raise ValueError(arm)


def episode(case, physical, arm, model, forecast_anchor, deadline):
    config = configs()[case]
    env = FrameworkEnv(config)
    env.reset(seed=physical)
    actions, states = [], []
    legal_ticks = legal_waits = 0
    for tick in range(HORIZON):
        deadline()
        state = env.inner.state
        submissions, _, legal, obs = legal_present(config, state, tick)
        result = choose(arm, model, config, state, tick, case, physical, legal, obs, forecast_anchor, deadline)
        action = result if isinstance(result, int) else result[0]
        if isinstance(result, tuple):
            assert result[1] == submissions
        assert action in legal
        if any(ACTIONS[index] for index in legal):
            legal_ticks += 1
            legal_waits += int(not ACTIONS[action])
        actions.append(action)
        states.append(asdict(state))
        env.step(action)
    metrics = env.episode_metrics()
    metrics.update(legal_service_ticks=legal_ticks, legal_service_waits=legal_waits,
                   legal_service_wait_fraction=(legal_waits / legal_ticks if legal_ticks else 0.0))
    return actions, states, metrics


def scenario_bank(config, case, seed, physical, tick, state, count):
    identity = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    bank_seed = role_seed(ANCHOR, f'v20_final_state:{case}:{seed}:{physical}:{tick}:{identity}')
    rng = random.Random(bank_seed)
    tapes = []
    for _ in range(count):
        events = [[rng.random() < config.failure_probability for _ in range(config.machines)]
                  for _ in range(tick, config.horizon)]
        uniforms = [rng.random() for _ in range(tick, HORIZON)]
        tapes.append(dict(events=events, uniforms=uniforms))
    return bank_seed, tapes


def root_cost(config, state, tick, root, tape, arm, model, case, physical, anchor, deadline):
    short = total = 0.0
    steps = 0
    for offset, current_tick in enumerate(range(tick, config.horizon)):
        deadline()
        submissions, admitted, legal, obs = legal_present(config, state, current_tick)
        if offset == 0:
            action = root
        elif current_tick >= HORIZON or arm == 'risk':
            action = ACTIONS.index(risk_matching(config, admitted, current_tick))
        else:
            result = choose(arm, model, config, state, current_tick, case, physical,
                            legal, obs, anchor, deadline)
            action = result if isinstance(result, int) else result[0]
            if isinstance(result, tuple):
                assert result[1] == submissions
        assert action in legal
        state, cost, _ = transition(config, state, current_tick, submissions,
                                    ACTIONS[action], tape['events'][offset])
        total += cost
        steps += 1
        if offset < 6:
            short += cost
    return [short, total], steps


def bootstrap(cube):
    rng = np.random.default_rng(BOOTSTRAP)
    draws = np.empty(20000)
    for start in range(0, 20000, 1000):
        seed_indices = rng.integers(0, 5, size=(1000, 5))
        physical_indices = rng.integers(0, 16, size=(1000, 5, 16))
        draws[start:start + 1000] = np.take_along_axis(
            cube[seed_indices], physical_indices, axis=2).mean((1, 2))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def run(profile, output_dir):
    if profile not in ('smoke', 'dev'):
        raise ValueError(profile)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    limit = 600 if profile == 'smoke' else 2700

    def deadline():
        if time.monotonic() - started > limit:
            raise TimeoutError('V20 final-state diagnostic hard deadline')

    manifest = dict(status='RUNNING', profile=profile,
                    started_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'manifest.json', manifest)
    try:
        if profile == 'dev':
            paths = ['research_result/diagnose_v20_final_state.py',
                     'research_result/verify_v20_final_state.py',
                     'research_result/v20_final_state_diagnostic_plan_vi.md']
            changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *paths],
                                              cwd=ROOT, text=True).splitlines()
            untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard',
                                                 '--', *paths], cwd=ROOT, text=True).splitlines()
            if changed or untracked:
                raise RuntimeError(f'uncommitted diagnostic source: {changed + untracked}')
        source = json.loads((SOURCE / 'manifest.json').read_text())
        primary_audit = json.loads(SOURCE_AUDIT.read_text())
        baseline_audit = json.loads(BASELINE_AUDIT.read_text())
        assert source['status'] == 'COMPLETED'
        assert primary_audit['status'] == baseline_audit['status'] == 'PASS'
        assert Path(primary_audit['run']).resolve() == Path(baseline_audit['run']).resolve() == SOURCE
        for name, digest in source['artifact_hashes'].items():
            assert sha(SOURCE / name) == digest
        used_seed_count = check_seed_roles()
        train = TRAIN if profile == 'dev' else TRAIN[3:4]
        physicals = PHYSICAL if profile == 'dev' else PHYSICAL[:1]
        cases = CASES if profile == 'dev' else ('interchangeable_scale1', 'reciprocal_scale3')
        count = 16 if profile == 'dev' else 1
        selection = json.loads((SOURCE / 'selection.json').read_text())
        v14_selection = json.loads((V14 / 'selection.json').read_text())
        models, checkpoint_hashes = {}, {}
        for arm in ('activation_margin', 'generic_q'):
            for seed in train:
                item = selection[arm][str(seed)]
                checkpoint = SOURCE / item['checkpoint']
                assert sha(checkpoint) == item['sha256']
                models[(arm, seed)] = load_v20(checkpoint)
                checkpoint_hashes[f'{arm}/{seed}'] = item['sha256']
        for seed in train:
            v14_seed = V14_TRAIN[seed - TRAIN[0]]
            item = v14_selection['count_calibrated'][str(v14_seed)]
            checkpoint = V14 / item['checkpoint']
            assert sha(checkpoint) == item['sha256']
            models[('count_calibrated', seed)] = load_v14(checkpoint)
            checkpoint_hashes[f'count_calibrated/{v14_seed}'] = item['sha256']
        source_rows = {(r['controller'], r['case'], r['train_seed'], int(r['seed'])): r
                       for r in read(SOURCE / 'episodes.csv')}
        forecast_anchor = source['registry']['forecast_seed']
        traces, episode_rows = {}, []
        specs = [('rollout', case, '', physical, None)
                 for case in cases for physical in physicals]
        specs += [(arm, case, str(seed if arm != 'count_calibrated' else V14_TRAIN[seed - TRAIN[0]]),
                   physical, models[(arm, seed)])
                  for arm in ('activation_margin', 'generic_q', 'count_calibrated')
                  for case in cases for seed in train for physical in physicals]
        for arm, case, seed, physical, model in tqdm(specs, desc='V20 final-state episode replay', unit='episode'):
            deadline()
            actions, states, metrics = episode(case, physical, arm, model, forecast_anchor, deadline)
            close_metrics(metrics, source_rows[(arm, case, seed, physical)])
            traces[(arm, case, seed, physical)] = (actions, states, metrics)
            episode_rows.append(dict(controller=arm, case=case, train_seed=seed,
                                     physical_seed=physical, **{name: metrics[name] for name in METRICS}))
        write(output / 'episodes.csv', episode_rows)
        with (output / 'traces.jsonl').open('w') as file:
            for (arm, case, seed, physical), (actions, states, _) in traces.items():
                file.write(json.dumps(dict(controller=arm, case=case, train_seed=seed,
                                           physical_seed=physical, actions=actions, states=states),
                                      sort_keys=True) + '\n')
        pair_rows, q_rows = [], []
        transitions = 0
        grid = [(case, seed, physical, other)
                for case in cases for seed in train for physical in physicals for other in OTHERS]
        for case, seed, physical, other in tqdm(grid, desc='V20 final-state first divergence/Q', unit='pair'):
            deadline()
            left, left_states, left_metrics = traces[('activation_margin', case, str(seed), physical)]
            other_seed = str(V14_TRAIN[seed - TRAIN[0]]) if other == 'count_calibrated' else ('' if other == 'rollout' else str(seed))
            right, right_states, right_metrics = traces[(other, case, other_seed, physical)]
            first = next((tick for tick, (a, b) in enumerate(zip(left, right)) if a != b), None)
            selected_q = (physical in PHYSICAL[:4] or
                          (case == 'reciprocal_scale3' and seed == 73700003))
            if profile == 'smoke':
                selected_q = True
            row = dict(case=case, train_seed=seed, physical_seed=physical, other=other,
                       other_train_seed=other_seed, first_tick=-1 if first is None else first,
                       first_state='' if first is None else json.dumps(left_states[first], sort_keys=True),
                       category='same' if first is None else classify(left[first], right[first]),
                       v20_action=-1 if first is None else left[first],
                       other_action=-1 if first is None else right[first],
                       q_selected=int(selected_q), forecast_seed=0, forecast_transitions=0,
                       v20_action_probability=None, other_action_probability_v20=None,
                       wait_probability=None, v20_logit_margin=None,
                       other_own_action_probability=None,
                       own_q4_margin=None, own_q16_margin=None, risk_q4_margin=None,
                       risk_q16_margin=None, own_q4_regret=None, own_q16_regret=None,
                       own_q16_service_regret=None, own_q16_pair_delta=0.0,
                       risk_q16_pair_delta=0.0, comparator_own_q16_pair_delta=None,
                       **{f'actual_{label}_delta': left_metrics[metric] - right_metrics[metric]
                          for label, metric in DELTA_METRICS})
            if first is not None:
                assert left_states[first] == right_states[first]
                assert all(left[t] == right[t] and left_states[t] == right_states[t]
                           for t in range(first))
                config = configs()[case]
                state = state_from_dict(left_states[first])
                _, _, legal, obs = legal_present(config, state, first)
                model = models[('activation_margin', seed)]
                probabilities = v20_probabilities(model, obs, legal)
                with torch.no_grad():
                    logits = model(torch.as_tensor(obs).reshape(1, 56))[0].cpu().numpy()
                row.update(v20_action_probability=float(probabilities[left[first]]),
                           other_action_probability_v20=float(probabilities[right[first]]),
                           wait_probability=float(probabilities[0]),
                           v20_logit_margin=float(logits[left[first]] - logits[right[first]]))
                if other == 'generic_q':
                    op = v20_probabilities(models[(other, seed)], obs, legal)
                    row['other_own_action_probability'] = float(op[right[first]])
                elif other == 'count_calibrated':
                    op = v14_probabilities(models[(other, seed)], obs, legal)
                    row['other_own_action_probability'] = float(op[right[first]])
                else:
                    row['other_own_action_probability'] = 1.0
                if selected_q:
                    bank_seed, tapes = scenario_bank(config, case, seed, physical,
                                                     first, left_states[first], count)
                    scores = {mode: {str(root): [] for root in legal} for mode in ('own', 'risk')}
                    comparator_scores = {str(root): [] for root in (left[first], right[first])}
                    local_steps = 0
                    for tape in tqdm(tapes, desc=f'Q {case}/{other}', unit='scenario', leave=False):
                        for mode in ('own', 'risk'):
                            for root in legal:
                                value, steps = root_cost(config, state, first, root, tape,
                                                         'activation_margin' if mode == 'own' else 'risk',
                                                         model if mode == 'own' else None,
                                                         case, physical, forecast_anchor, deadline)
                                scores[mode][str(root)].append(value)
                                local_steps += steps
                        for root in (left[first], right[first]):
                            value, steps = root_cost(config, state, first, root, tape,
                                                     other, models.get((other, seed)), case,
                                                     physical, forecast_anchor, deadline)
                            comparator_scores[str(root)].append(value)
                            local_steps += steps
                    def q(mode, root, n):
                        return float(np.mean([value[1] for value in scores[mode][str(root)][:n]]))
                    def margin(mode, n):
                        return q(mode, 0, n) - min(q(mode, root, n) for root in legal if root != 0)
                    if not any(root != 0 for root in legal):
                        raise AssertionError('first divergence without legal service')
                    row.update(own_q4_margin=margin('own', min(4, count)),
                               own_q16_margin=margin('own', count),
                               risk_q4_margin=margin('risk', min(4, count)),
                               risk_q16_margin=margin('risk', count),
                               own_q4_regret=q('own', left[first], min(4, count)) -
                                   min(q('own', root, min(4, count)) for root in legal),
                               own_q16_regret=q('own', left[first], count) -
                                   min(q('own', root, count) for root in legal),
                               own_q16_service_regret=(q('own', left[first], count) -
                                   min(q('own', root, count) for root in legal if root != 0))
                                   if left[first] != 0 else None,
                               own_q16_pair_delta=q('own', left[first], count) - q('own', right[first], count),
                               risk_q16_pair_delta=q('risk', left[first], count) - q('risk', right[first], count),
                               comparator_own_q16_pair_delta=float(np.mean(
                                   [value[1] for value in comparator_scores[str(left[first])]]) -
                                   np.mean([value[1] for value in comparator_scores[str(right[first])]])),
                               forecast_seed=bank_seed, forecast_transitions=local_steps)
                    q_rows.append(dict(case=case, train_seed=seed, physical_seed=physical,
                                       other=other, tick=first, original_state=left_states[first],
                                       legal=list(legal), v20_action=left[first], other_action=right[first],
                                       forecast_seed=bank_seed, tapes=tapes, scores=scores,
                                       comparator_scores=comparator_scores,
                                       forecast_transitions=local_steps))
                    transitions += local_steps
            pair_rows.append(row)
        write(output / 'pairs.csv', pair_rows)
        with (output / 'q_banks.jsonl').open('w') as file:
            for bank in q_rows:
                file.write(json.dumps(bank, sort_keys=True) + '\n')
        summary = summarize(pair_rows, profile, cases, train, physicals, len(q_rows), transitions)
        save(output / 'summary.json', summary)
        artifact_names = ('episodes.csv', 'traces.jsonl', 'pairs.csv', 'q_banks.jsonl', 'summary.json')
        manifest.update(status='COMPLETED', finished_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic() - started,
                        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                           text=True).strip(),
                        source_run=str(SOURCE), source_manifest_sha256=sha(SOURCE / 'manifest.json'),
                        source_audit_sha256=sha(SOURCE_AUDIT), baseline_audit_sha256=sha(BASELINE_AUDIT),
                        source_episodes_sha256=sha(SOURCE / 'episodes.csv'),
                        source_selection_sha256=sha(SOURCE / 'selection.json'),
                        v14_selection_sha256=sha(V14 / 'selection.json'),
                        plan_sha256=sha(PLAN), runner_sha256=sha(Path(__file__)),
                        verifier_sha256=sha(ROOT / 'research_result/verify_v20_final_state.py'),
                        checkpoint_hashes=checkpoint_hashes,
                        forecast_anchor=ANCHOR, bootstrap_seed=BOOTSTRAP,
                        scenario_count=count, historical_role_count=used_seed_count,
                        counts=dict(episodes=len(episode_rows), pairs=len(pair_rows),
                                    q_banks=len(q_rows), forecast_transitions=transitions),
                        artifact_hashes={name: sha(output / name) for name in artifact_names})
        save(output / 'manifest.json', manifest)
        return output
    except Exception as exc:
        manifest.update(status='INCOMPLETE' if isinstance(exc, TimeoutError) else 'FAILED',
                        error=f'{type(exc).__name__}: {exc}',
                        elapsed_seconds=time.monotonic() - started)
        save(output / 'manifest.json', manifest)
        raise


def summarize(rows, profile, cases, train, physicals, q_count, transitions):
    result = dict(status='COMPLETED', profile=profile, episodes_per_seed=len(physicals),
                  pairs=len(rows), q_banks=q_count, forecast_transitions=transitions,
                  comparisons={}, flags={})
    for other in OTHERS:
        for case in cases:
            cell = [row for row in rows if row['case'] == case and row['other'] == other]
            label = f'{case}/{other}'
            item = dict(pairs=len(cell), categories={category: sum(row['category'] == category
                         for row in cell) for category in CATEGORIES},
                        mean_actual_cost_delta=float(np.mean([row['actual_cost_delta'] for row in cell])),
                        mean_actual_failure_delta=float(np.mean([row['actual_failures_delta'] for row in cell])),
                        mean_actual_pm_queue_delta=float(np.mean([row['actual_pm_queue_delta'] for row in cell])),
                        mean_actual_cm_queue_delta=float(np.mean([row['actual_cm_queue_delta'] for row in cell])),
                        mean_actual_start_delta=float(np.mean([row['actual_starts_delta'] for row in cell])),
                        q_selected=sum(row['q_selected'] for row in cell),
                        q_evaluated=sum(row['own_q16_margin'] is not None for row in cell),
                        mean_q_pair_delta_unconditional=float(np.mean(
                            [row['own_q16_pair_delta'] for row in cell if row['q_selected']])),
                        mean_q_pair_delta_conditional=(float(np.mean(
                            [row['own_q16_pair_delta'] for row in cell if row['own_q16_margin'] is not None]))
                            if any(row['own_q16_margin'] is not None for row in cell) else None))
            result['comparisons'][label] = item
    if profile == 'dev':
        for other in OTHERS:
            cube = np.asarray([[[next(row['actual_cost_delta'] for row in rows
                                      if row['case'] == case and row['other'] == other
                                      and row['train_seed'] == seed and row['physical_seed'] == physical)
                                 for physical in physicals] for seed in train] for case in cases])
            result['comparisons'][f'aggregate/{other}'] = dict(
                mean_actual_cost_delta=float(cube.mean()),
                descriptive_ci95=bootstrap(cube.mean(axis=0)) if other == 'generic_q' else None,
                training_seed_means={str(seed): float(cube[:, i, :].mean())
                                     for i, seed in enumerate(train)})
        qrows = [row for row in rows if row['own_q16_margin'] is not None]
        waiting = [row for row in qrows if row['case'] == 'reciprocal_scale3'
                   and row['train_seed'] == 73700003 and row['v20_action'] == 0]
        mismatch = sum(row['own_q16_margin'] >= 5 for row in waiting)
        flips = sum((row['own_q4_margin'] > 0) != (row['own_q16_margin'] > 0) for row in qrows)
        service = [row for row in qrows if row['v20_action'] != 0 and row['other_action'] != 0]
        gap = sum(row['own_q16_service_regret'] >= 5 for row in service)
        result['flags'] = dict(
            gate_mismatch=None if not waiting else mismatch / len(waiting) >= 0.5,
            gate_mismatch_numerator=mismatch, gate_mismatch_denominator=len(waiting),
            label_instability=None if not qrows else flips / len(qrows) >= 0.25,
            label_instability_numerator=flips, label_instability_denominator=len(qrows),
            service_matching_gap=None if not service else gap / len(service) >= 0.25,
            service_matching_gap_numerator=gap, service_matching_gap_denominator=len(service))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('smoke', 'dev'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(dict(status='COMPLETED', output_dir=str(run(args.profile, args.output_dir)))))


if __name__ == '__main__':
    main()
