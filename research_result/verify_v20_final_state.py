"""Independent exact replay of V20 opened-DEV final-state diagnostic."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import CASES, HORIZON, FrameworkEnv, catalogue, configs
from ht_pdm_fjsp.maintenance_framework_v14_campaign import episode_seed as v14_episode_seed
from ht_pdm_fjsp.maintenance_framework_v19_campaign import episode_seed as v19_episode_seed
from ht_pdm_fjsp.maintenance_framework_v20_campaign import episode_seed as v20_episode_seed
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
from ht_pdm_fjsp.maintenance_framework_v14_campaign import (
    load_checkpoint as load_v14, masked_probabilities as v14_probabilities,
    policy_action as v14_action,
)
from ht_pdm_fjsp.maintenance_framework_v20_campaign import (
    V14, legal_present, load_checkpoint as load_v20,
    masked_probabilities as v20_probabilities, policy_action as v20_action,
)
from ht_pdm_fjsp.maintenance_framework_v7_campaign import V7RuleController
from ht_pdm_fjsp.maintenance_request_policies import risk_matching
from ht_pdm_fjsp.maintenance_requests import state_from_dict, transition

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'research_result/framework_v20_dev_20261021T125049Z'
SOURCE_AUDIT = ROOT / 'research_result/framework_v20_audit_20261021T152700Z/independent_audit.json'
BASELINE_AUDIT = ROOT / 'research_result/framework_v20_baseline_audit_20261021T152701Z/baseline_replay_audit.json'
PLAN = ROOT / 'research_result/v20_final_state_diagnostic_plan_vi.md'
ACTIONS = catalogue()
TRAIN = tuple(range(73700000, 73700005))
V14_TRAIN = tuple(range(72600000, 72600005))
PHYSICAL = tuple(range(73711000, 73711016))
OTHERS = ('generic_q', 'count_calibrated', 'rollout')
METRICS = (
    'objective', 'maintenance_cost', 'unavailability_cost', 'failure_cost',
    'tail_cost', 'H_availability', 'H_failures', 'queued_cm_ticks',
    'queued_pm_ticks', 'H_pending', 'end_pending', 'service_starts',
    'requests_created', 'requests_completed', 'legal_service_ticks',
    'legal_service_waits', 'legal_service_wait_fraction',
)
DELTAS = (
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


def read(path):
    with Path(path).open(newline='') as file:
        return list(csv.DictReader(file))


def close(actual, expected, label):
    if not np.isclose(float(actual), float(expected), rtol=0, atol=1e-8):
        raise AssertionError((label, actual, expected))


def policy(arm, actor, config, state, tick, case, physical, legal, obs, anchor, deadline):
    if arm in ('activation_margin', 'generic_q'):
        return v20_action(actor, obs, legal)
    if arm == 'count_calibrated':
        return v14_action(actor, obs, legal)
    assert arm == 'rollout'
    controller = V7RuleController('rollout', anchor, deadline)
    expected_submissions, matching = controller.choose(config, state, tick,
                                                        case=case, seed=physical)
    return ACTIONS.index(matching), expected_submissions


def replay_episode(config, case, physical, arm, actor, anchor, deadline):
    env = FrameworkEnv(config)
    env.reset(seed=physical)
    sequence, states = [], []
    opportunities = waits = 0
    for tick in range(HORIZON):
        deadline()
        state = env.inner.state
        submitted, _, legal, obs = legal_present(config, state, tick)
        result = policy(arm, actor, config, state, tick, case, physical,
                        legal, obs, anchor, deadline)
        action = result if isinstance(result, int) else result[0]
        if isinstance(result, tuple):
            assert result[1] == submitted
        assert action in legal
        opportunities += int(any(ACTIONS[i] for i in legal))
        waits += int(any(ACTIONS[i] for i in legal) and not ACTIONS[action])
        states.append(asdict(state))
        sequence.append(action)
        env.step(action)
    metrics = env.episode_metrics()
    metrics['legal_service_ticks'] = opportunities
    metrics['legal_service_waits'] = waits
    metrics['legal_service_wait_fraction'] = waits / opportunities if opportunities else 0.0
    return sequence, states, metrics


def classify(a, b):
    left, right = ACTIONS[a], ACTIONS[b]
    if left == right:
        return 'same'
    if left and not right:
        return 'v20_service_other_wait'
    if right and not left:
        return 'v20_wait_other_service'
    if len(left) > len(right):
        return 'v20_higher_cardinality'
    if len(left) < len(right):
        return 'v20_lower_cardinality'
    return 'machine_set' if {m for m, _ in left} != {m for m, _ in right} else 'assignment'


def tapes_for(config, case, seed, physical, tick, state, count):
    digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    bank_seed = role_seed(74070000, f'v20_final_state:{case}:{seed}:{physical}:{tick}:{digest}')
    generator = random.Random(bank_seed)
    tapes = []
    for _ in range(count):
        tape = dict(
            events=[[generator.random() < config.failure_probability
                     for _ in range(config.machines)]
                    for _ in range(tick, config.horizon)],
            uniforms=[generator.random() for _ in range(tick, HORIZON)],
        )
        tapes.append(tape)
    return bank_seed, tapes


def q_path(config, start_state, start_tick, forced_action, tape, continuation,
           actor, case, physical, forecast_anchor, deadline):
    current = start_state
    total = short = 0.0
    steps = 0
    for offset, tick in enumerate(range(start_tick, config.horizon)):
        deadline()
        submissions, admitted, legal, observation = legal_present(config, current, tick)
        if offset == 0:
            action = forced_action
        elif tick >= HORIZON or continuation == 'risk':
            action = ACTIONS.index(risk_matching(config, admitted, tick))
        else:
            result = policy(continuation, actor, config, current, tick, case, physical,
                            legal, observation, forecast_anchor, deadline)
            action = result if isinstance(result, int) else result[0]
            if isinstance(result, tuple):
                assert result[1] == submissions
        assert action in legal
        current, cost, _ = transition(config, current, tick, submissions,
                                      ACTIONS[action], tape['events'][offset])
        total += cost
        if offset < 6:
            short += cost
        steps += 1
    return [short, total], steps


def bootstrap(cube):
    generator = np.random.default_rng(74070001)
    results = np.empty(20000)
    for begin in range(0, 20000, 1000):
        selected_seeds = generator.integers(0, 5, size=(1000, 5))
        selected_physical = generator.integers(0, 16, size=(1000, 5, 16))
        results[begin:begin + 1000] = np.take_along_axis(
            cube[selected_seeds], selected_physical, axis=2).mean((1, 2))
    return np.quantile(results, [0.025, 0.975]).tolist()


def audit(run):
    start = time.monotonic()
    torch.set_num_threads(1)
    def deadline():
        if time.monotonic() - start > 2700:
            raise TimeoutError('V20 final-state independent audit hard deadline')
    manifest = json.loads((run / 'manifest.json').read_text())
    profile = manifest['profile']
    assert manifest['status'] == 'COMPLETED' and profile in ('smoke', 'dev')
    assert manifest['source_run'] == str(SOURCE)
    assert sha(SOURCE / 'manifest.json') == manifest['source_manifest_sha256']
    assert sha(SOURCE / 'selection.json') == manifest['source_selection_sha256']
    assert sha(SOURCE / 'episodes.csv') == manifest['source_episodes_sha256']
    assert sha(SOURCE_AUDIT) == manifest['source_audit_sha256']
    assert sha(BASELINE_AUDIT) == manifest['baseline_audit_sha256']
    assert sha(V14 / 'selection.json') == manifest['v14_selection_sha256']
    assert sha(PLAN) == manifest['plan_sha256']
    assert sha(ROOT / 'research_result/diagnose_v20_final_state.py') == manifest['runner_sha256']
    assert sha(Path(__file__)) == manifest['verifier_sha256']
    for name, digest in manifest['artifact_hashes'].items():
        assert sha(run / name) == digest
    assert json.loads(SOURCE_AUDIT.read_text())['status'] == 'PASS'
    assert json.loads(BASELINE_AUDIT.read_text())['status'] == 'PASS'
    source = json.loads((SOURCE / 'manifest.json').read_text())
    assert source['status'] == 'COMPLETED'
    assert manifest['forecast_anchor'] == 74070000
    assert manifest['bootstrap_seed'] == 74070001
    used_roles = set()
    for registry in (ROOT / 'configs').glob('*seed_registry.json'):
        used_roles |= registered_integers(json.loads(registry.read_text()))
    for function, first_seed, updates in ((v20_episode_seed, 73700000, 8),
                                          (v19_episode_seed, 73400000, 6),
                                          (v14_episode_seed, 72600000, 6)):
        for training_seed in range(first_seed, first_seed + 5):
            for iteration in range(1, updates + 1):
                for case in CASES:
                    for index in range(16):
                        used_roles.add(function(training_seed, iteration, case, index))
    assert manifest['historical_role_count'] == len(used_roles)
    assert not {74070000, 74070001} & used_roles
    for name, digest in source['artifact_hashes'].items():
        assert sha(SOURCE / name) == digest
    train = TRAIN if profile == 'dev' else TRAIN[3:4]
    physicals = PHYSICAL if profile == 'dev' else PHYSICAL[:1]
    cases = CASES if profile == 'dev' else ('interchangeable_scale1', 'reciprocal_scale3')
    n_scenarios = 16 if profile == 'dev' else 1
    assert manifest['scenario_count'] == n_scenarios
    frozen = json.loads((SOURCE / 'selection.json').read_text())
    v14_frozen = json.loads((V14 / 'selection.json').read_text())
    actors = {}
    for seed in train:
        for arm in ('activation_margin', 'generic_q'):
            item = frozen[arm][str(seed)]
            assert sha(SOURCE / item['checkpoint']) == item['sha256'] == manifest['checkpoint_hashes'][f'{arm}/{seed}']
            actors[(arm, seed)] = load_v20(SOURCE / item['checkpoint'])
        old_seed = V14_TRAIN[seed - TRAIN[0]]
        item = v14_frozen['count_calibrated'][str(old_seed)]
        assert sha(V14 / item['checkpoint']) == item['sha256'] == manifest['checkpoint_hashes'][f'count_calibrated/{old_seed}']
        actors[('count_calibrated', seed)] = load_v14(V14 / item['checkpoint'])
    source_episodes = {(row['controller'], row['case'], row['train_seed'], int(row['seed'])): row
                       for row in read(SOURCE / 'episodes.csv')}
    recorded_episodes = {(row['controller'], row['case'], row['train_seed'],
                          int(row['physical_seed'])): row for row in read(run / 'episodes.csv')}
    recorded_traces = {(row['controller'], row['case'], row['train_seed'],
                        row['physical_seed']): row
                       for row in map(json.loads, (run / 'traces.jsonl').open())}
    specs = [('rollout', case, '', physical, None)
             for case in cases for physical in physicals]
    specs.extend((arm, case, str(seed if arm != 'count_calibrated'
                                  else V14_TRAIN[seed - TRAIN[0]]),
                  physical, actors[(arm, seed)])
                 for arm in ('activation_margin', 'generic_q', 'count_calibrated')
                 for case in cases for seed in train for physical in physicals)
    assert len(specs) == len(recorded_episodes) == len(recorded_traces) == manifest['counts']['episodes']
    traces = {}
    for arm, case, seed, physical, actor in tqdm(specs, desc='Independent V20 final episode replay', unit='episode'):
        deadline()
        sequence, states, metrics = replay_episode(configs()[case], case, physical,
                                                   arm, actor, source['registry']['forecast_seed'], deadline)
        key = (arm, case, seed, physical)
        report = recorded_traces[key]
        assert sequence == report['actions'] and json.loads(json.dumps(states)) == report['states']
        for metric in METRICS:
            close(metrics[metric], recorded_episodes[key][metric], f'recorded {metric}')
            close(metrics[metric], source_episodes[key][metric], f'source {metric}')
        traces[key] = (sequence, states, metrics)
    pairs = {(row['case'], int(row['train_seed']), int(row['physical_seed']), row['other']): row
             for row in read(run / 'pairs.csv')}
    banks = {(row['case'], int(row['train_seed']), int(row['physical_seed']), row['other']): row
             for row in map(json.loads, (run / 'q_banks.jsonl').open())}
    expected_pairs = len(cases) * len(train) * len(physicals) * len(OTHERS)
    assert len(pairs) == expected_pairs == manifest['counts']['pairs']
    assert len(banks) == manifest['counts']['q_banks']
    transitions = 0
    replayed_bank_keys = set()
    for case in cases:
        for seed in train:
            for physical in physicals:
                lseq, lstates, lm = traces[('activation_margin', case, str(seed), physical)]
                for other in OTHERS:
                    deadline()
                    key = (case, seed, physical, other)
                    row = pairs[key]
                    other_seed = (str(V14_TRAIN[seed - TRAIN[0]]) if other == 'count_calibrated'
                                  else '' if other == 'rollout' else str(seed))
                    rseq, rstates, rm = traces[(other, case, other_seed, physical)]
                    first = next((tick for tick, (a, b) in enumerate(zip(lseq, rseq)) if a != b), None)
                    assert row['other_train_seed'] == other_seed
                    assert int(row['first_tick']) == (-1 if first is None else first)
                    assert row['category'] == ('same' if first is None else classify(lseq[first], rseq[first]))
                    assert int(row['v20_action']) == (-1 if first is None else lseq[first])
                    assert int(row['other_action']) == (-1 if first is None else rseq[first])
                    assert row['first_state'] == ('' if first is None else json.dumps(lstates[first], sort_keys=True))
                    for label, metric in DELTAS:
                        close(row[f'actual_{label}_delta'], lm[metric] - rm[metric], f'actual {label}')
                    selected = physical in PHYSICAL[:4] or (case == 'reciprocal_scale3' and seed == 73700003)
                    if profile == 'smoke':
                        selected = True
                    assert int(row['q_selected']) == int(selected)
                    if first is None:
                        assert key not in banks
                        assert int(row['forecast_transitions']) == 0
                        continue
                    assert lstates[first] == rstates[first]
                    assert all(lseq[t] == rseq[t] and lstates[t] == rstates[t] for t in range(first))
                    config = configs()[case]
                    state = state_from_dict(lstates[first])
                    _, _, legal, obs = legal_present(config, state, first)
                    p = v20_probabilities(actors[('activation_margin', seed)], obs, legal)
                    with torch.no_grad():
                        logits = actors[('activation_margin', seed)](torch.as_tensor(obs).reshape(1, 56))[0].cpu().numpy()
                    close(row['v20_action_probability'], p[lseq[first]], 'v20 probability')
                    close(row['other_action_probability_v20'], p[rseq[first]], 'other probability under v20')
                    close(row['wait_probability'], p[0], 'WAIT probability')
                    close(row['v20_logit_margin'], logits[lseq[first]] - logits[rseq[first]], 'logit margin')
                    if other == 'generic_q':
                        op = v20_probabilities(actors[(other, seed)], obs, legal)
                        own_probability = op[rseq[first]]
                    elif other == 'count_calibrated':
                        op = v14_probabilities(actors[(other, seed)], obs, legal)
                        own_probability = op[rseq[first]]
                    else:
                        own_probability = 1.0
                    close(row['other_own_action_probability'], own_probability, 'other own probability')
                    if not selected:
                        assert key not in banks and int(row['forecast_transitions']) == 0
                        continue
                    bank = banks[key]
                    replayed_bank_keys.add(key)
                    assert bank['case'] == case and int(bank['train_seed']) == seed
                    assert int(bank['physical_seed']) == physical and bank['other'] == other
                    assert bank['original_state'] == json.loads(json.dumps(lstates[first]))
                    assert bank['legal'] == list(legal)
                    assert int(bank['tick']) == first
                    assert int(bank['v20_action']) == lseq[first]
                    assert int(bank['other_action']) == rseq[first]
                    bank_seed, tapes = tapes_for(config, case, seed, physical, first,
                                                 lstates[first], n_scenarios)
                    assert int(row['forecast_seed']) == bank_seed == int(bank['forecast_seed'])
                    assert tapes == bank['tapes']
                    local_steps = 0
                    scores = {'own': {str(root): [] for root in legal},
                              'risk': {str(root): [] for root in legal}}
                    other_scores = {str(root): [] for root in (lseq[first], rseq[first])}
                    for tape in tapes:
                        for mode in ('own', 'risk'):
                            for root in legal:
                                got, steps = q_path(config, state, first, root, tape,
                                                    'activation_margin' if mode == 'own' else 'risk',
                                                    actors[('activation_margin', seed)] if mode == 'own' else None,
                                                    case, physical, source['registry']['forecast_seed'], deadline)
                                for a, b in zip(got, bank['scores'][mode][str(root)][len(scores[mode][str(root)])]):
                                    close(a, b, f'full {mode} Q')
                                scores[mode][str(root)].append(got)
                                local_steps += steps
                        for root in (lseq[first], rseq[first]):
                            got, steps = q_path(config, state, first, root, tape, other,
                                                actors.get((other, seed)), case, physical,
                                                source['registry']['forecast_seed'], deadline)
                            for a, b in zip(got, bank['comparator_scores'][str(root)][len(other_scores[str(root)])]):
                                close(a, b, 'comparator Q')
                            other_scores[str(root)].append(got)
                            local_steps += steps
                    assert local_steps == int(row['forecast_transitions']) == int(bank['forecast_transitions'])
                    transitions += local_steps
                    def q(mode, root, n):
                        return float(np.mean([score[1] for score in scores[mode][str(root)][:n]]))
                    def margin(mode, n):
                        return q(mode, 0, n) - min(q(mode, root, n) for root in legal if root != 0)
                    four = min(4, n_scenarios)
                    for field, want in (
                        ('own_q4_margin', margin('own', four)),
                        ('own_q16_margin', margin('own', n_scenarios)),
                        ('risk_q4_margin', margin('risk', four)),
                        ('risk_q16_margin', margin('risk', n_scenarios)),
                        ('own_q4_regret', q('own', lseq[first], four) -
                         min(q('own', action, four) for action in legal)),
                        ('own_q16_regret', q('own', lseq[first], n_scenarios) -
                         min(q('own', action, n_scenarios) for action in legal)),
                        ('own_q16_pair_delta', q('own', lseq[first], n_scenarios) -
                         q('own', rseq[first], n_scenarios)),
                        ('risk_q16_pair_delta', q('risk', lseq[first], n_scenarios) -
                         q('risk', rseq[first], n_scenarios)),
                        ('comparator_own_q16_pair_delta',
                         float(np.mean([value[1] for value in other_scores[str(lseq[first])]]) -
                               np.mean([value[1] for value in other_scores[str(rseq[first])]]))),
                    ):
                        close(row[field], want, field)
                    if lseq[first] != 0:
                        want = q('own', lseq[first], n_scenarios) - min(
                            q('own', action, n_scenarios) for action in legal if action != 0)
                        close(row['own_q16_service_regret'], want, 'service regret')
                    else:
                        assert row['own_q16_service_regret'] == ''
    assert replayed_bank_keys == set(banks)
    assert transitions == manifest['counts']['forecast_transitions']
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['status'] == 'COMPLETED' and summary['q_banks'] == len(banks)
    assert summary['forecast_transitions'] == transitions
    assert summary['pairs'] == len(pairs)
    for other in OTHERS:
        for case in cases:
            cell = [pairs[(case, seed, physical, other)] for seed in train for physical in physicals]
            item = summary['comparisons'][f'{case}/{other}']
            assert item['pairs'] == len(cell)
            assert item['categories'] == {c: sum(row['category'] == c for row in cell) for c in CATEGORIES}
            for key, field in (
                ('mean_actual_cost_delta', 'actual_cost_delta'),
                ('mean_actual_failure_delta', 'actual_failures_delta'),
                ('mean_actual_pm_queue_delta', 'actual_pm_queue_delta'),
                ('mean_actual_cm_queue_delta', 'actual_cm_queue_delta'),
                ('mean_actual_start_delta', 'actual_starts_delta'),
            ):
                close(item[key], np.mean([float(row[field]) for row in cell]), key)
            selected = [row for row in cell if int(row['q_selected'])]
            evaluated = [row for row in selected if row['own_q16_margin'] != '']
            assert item['q_selected'] == len(selected) and item['q_evaluated'] == len(evaluated)
            close(item['mean_q_pair_delta_unconditional'],
                  np.mean([float(row['own_q16_pair_delta']) for row in selected]), 'unconditional Q')
            if evaluated:
                close(item['mean_q_pair_delta_conditional'],
                      np.mean([float(row['own_q16_pair_delta']) for row in evaluated]), 'conditional Q')
            else:
                assert item['mean_q_pair_delta_conditional'] is None
    if profile == 'dev':
        for other in OTHERS:
            cube = np.asarray([[[float(pairs[(case, seed, physical, other)]['actual_cost_delta'])
                                 for physical in physicals] for seed in train] for case in cases])
            item = summary['comparisons'][f'aggregate/{other}']
            close(item['mean_actual_cost_delta'], cube.mean(), 'aggregate cost')
            for index, seed in enumerate(train):
                close(item['training_seed_means'][str(seed)], cube[:, index, :].mean(), 'seed cost')
            if other == 'generic_q':
                for got, want in zip(item['descriptive_ci95'], bootstrap(cube.mean(axis=0))):
                    close(got, want, 'bootstrap')
            else:
                assert item['descriptive_ci95'] is None
        evaluated = [row for row in pairs.values() if row['own_q16_margin'] != '']
        waits = [row for row in evaluated if row['case'] == 'reciprocal_scale3'
                 and int(row['train_seed']) == 73700003 and int(row['v20_action']) == 0]
        mismatches = sum(float(row['own_q16_margin']) >= 5 for row in waits)
        flips = sum((float(row['own_q4_margin']) > 0) !=
                    (float(row['own_q16_margin']) > 0) for row in evaluated)
        service = [row for row in evaluated if int(row['v20_action']) != 0
                   and int(row['other_action']) != 0]
        gaps = sum(float(row['own_q16_service_regret']) >= 5 for row in service)
        flags = summary['flags']
        assert flags == dict(
            gate_mismatch=None if not waits else mismatches / len(waits) >= 0.5,
            gate_mismatch_numerator=mismatches, gate_mismatch_denominator=len(waits),
            label_instability=None if not evaluated else flips / len(evaluated) >= 0.25,
            label_instability_numerator=flips, label_instability_denominator=len(evaluated),
            service_matching_gap=None if not service else gaps / len(service) >= 0.25,
            service_matching_gap_numerator=gaps, service_matching_gap_denominator=len(service))
    return dict(status='PASS', run=str(run), profile=profile,
                episode_replays=len(specs), first_divergence_pairs=len(pairs),
                full_q_banks=len(banks), forecast_transitions=transitions,
                wall_seconds=time.monotonic() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    try:
        report = audit(args.run_dir.resolve())
    except Exception as error:
        (output / 'independent_audit.json').write_text(json.dumps(
            dict(status='FAIL', error=f'{type(error).__name__}: {error}'), indent=2) + '\n')
        raise
    (output / 'independent_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
