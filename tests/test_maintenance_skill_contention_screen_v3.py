"""Assignment obligations, cancellation controls and five sealed gates."""

from dataclasses import replace
import hashlib
import io
import json

import pytest

from ht_pdm_fjsp import maintenance_skill_contention_screen_v3 as s
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import transition


def idle(ages=(3, 3, 3, 3)):
    return DispatchState(ages, (False,) * 4, (0,) * 4, (0, 0), (-1, -1))


def metrics(c, state, selected=None):
    return s.state_metrics(c, state, s.choose(c, state, 'deadline_matching') if selected is None else selected)


def test_grid_is_48_distinct_candidates_five_controls_and_no_dense_mask():
    cfg = s.settings('smoke'); specs = s.candidates(cfg)
    assert len(specs) == 53 and sum(r['selectable'] for r in specs) == 48
    assert len({json.dumps(r['config'], sort_keys=True) for r in specs}) == 53
    for spec in specs:
        c = s.decode_config(spec); c.validate()
        assert c.horizon == 60 and spec['eligible_edges'] < 8
        assert max(d for row in c.service_time for d in row) <= cfg['tail_ticks']
        if spec['selectable']:
            assert all(c.service_time[m][0] < c.service_time[m][1] for m in range(4)
                if all(c.service_time[m]))
    with pytest.raises(ValueError): s.settings('unknown')
    with pytest.raises(ValueError): s.physical_config('balanced2', 'uniform_control', cfg)
    with pytest.raises(ValueError): s.physical_config('overlap3', 'unknown', cfg)


def test_same_machine_subset_has_exact_twelve_unit_committed_cost_span():
    c = s.physical_config('overlap3', 'fast1_slow352', s.settings('smoke'))
    state = idle((0, 3, 3, 0)); m = metrics(c, state)
    assert c.service_time == ((1, 0), (1, 3), (1, 5), (1, 2))
    assert m['skill_contention_rate'] == m['cost_sensitive_contention_rate'] == 1
    assert m['selected_cost_sensitive_contention_rate'] == 1
    assert m['committed_cost_span_max'] == m['selected_committed_cost_span'] == 12
    spans = s.committed_cost_spans(c, s.action_space(c, state)[2])
    assert spans[(1, 2)] == 12


def test_committed_cost_span_matches_actual_in_service_ticks_through_completion():
    c = s.physical_config('overlap3', 'fast1_slow352', s.settings('smoke'))
    state = idle((0, 3, 3, 0))
    obligations = []
    for action in (((1, 0), (2, 1)), ((1, 1), (2, 0))):
        following, _, info = transition(c, state, action, (False,) * 4)
        cost = info['unavailability_cost']
        while any(following.remaining):
            following, _, info = transition(c, following, (), (False,) * 4)
            cost += info['unavailability_cost']
        obligations.append(cost)
    assert obligations == [36, 24]
    assert max(obligations) - min(obligations) == metrics(c, state)['committed_cost_span_max']


def test_uniform_v2_winner_has_true_competition_but_no_aggregate_assignment_span():
    c = s.physical_config('overlap3', 'uniform_control', s.settings('smoke'))
    m = metrics(c, idle((0, 3, 3, 0)))
    assert m['skill_contention_rate'] == m['contested_choice_rate'] == 1
    assert m['cost_sensitive_contention_rate'] == m['selected_cost_sensitive_contention_rate'] == 0
    assert m['committed_cost_span_max'] == 0


def test_unequal_machine_rows_can_cancel_exactly_and_must_not_count():
    c = s.physical_config('overlap3', 'cancellation_control', s.settings('smoke'))
    assert c.service_time[1] != c.service_time[2]
    m = metrics(c, idle((0, 3, 3, 0)))
    assert m['skill_contention_rate'] == m['contested_choice_rate'] == 1
    assert m['cost_sensitive_contention_rate'] == m['selected_cost_sensitive_contention_rate'] == 0
    assert m['committed_cost_span_max'] == 0


def test_selected_gate_excludes_underfilled_or_forced_unique_subsets():
    c = s.physical_config('overlap3', 'fast1_slow352', s.settings('smoke'))
    m = metrics(c, idle((0, 3, 3, 0)), ((1, 0),))
    assert m['cost_sensitive_contention_rate'] == 1
    assert m['selected_cost_sensitive_contention_rate'] == 0
    m = metrics(c, idle((3, 3, 3, 0)), ((0, 0), (1, 1)))
    assert m['cost_sensitive_contention_rate'] == 1
    assert m['selected_cost_sensitive_contention_rate'] == 0
    assert m['selected_committed_cost_span'] == 0


def test_one_machine_duration_choice_without_competition_does_not_count():
    c = s.physical_config('overlap3', 'fast1_slow352', s.settings('smoke'))
    m = metrics(c, idle((0, 3, 0, 0)))
    assert m['committed_cost_span_max'] == 12
    assert m['skill_contention_rate'] == m['cost_sensitive_contention_rate'] == 0
    m = metrics(c, idle((0, 0, 0, 0)))
    assert m['committed_cost_span_max'] == m['cost_sensitive_contention_rate'] == 0


def test_pure_capacity_and_unavailable_worker_are_not_strict_competition():
    c = s.physical_config('overlap3', 'fast1_slow352', s.settings('smoke'))
    state = replace(idle((0, 3, 3, 0)), remaining=(1, 0), assigned=(0, -1))
    m = metrics(c, state)
    assert m['capacity_pressure_rate'] == 1
    assert m['skill_contention_rate'] == m['cost_sensitive_contention_rate'] == 0


def test_equal_and_incomparable_tradeoff_controls_keep_ties_and_cannot_pass():
    cfg = s.settings('smoke')
    for pattern in ('equal_control', 'tradeoff_control'):
        c = s.physical_config('overlap3', pattern, cfg)
        m = metrics(c, idle((0, 3, 3, 0)))
        assert m['legacy_proposal_conflict_rate'] == 1
        assert m['skill_contention_rate'] == m['cost_sensitive_contention_rate'] == 0


def test_cost_metrics_invariant_to_worker_labels_and_scale_with_price():
    cfg = s.settings('smoke'); state = idle((0, 3, 3, 0))
    fields = ('skill_contention_rate', 'cost_sensitive_contention_rate',
              'selected_cost_sensitive_contention_rate', 'committed_cost_span_max', 'selected_committed_cost_span')
    for pattern in s.PATTERNS:
        c = s.physical_config('overlap3', pattern, cfg)
        swapped = replace(c, service_time=tuple(tuple(reversed(r)) for r in c.service_time),
            restoration=tuple(tuple(reversed(r)) for r in c.restoration))
        a, b = metrics(c, state), metrics(swapped, state)
        assert {k: a[k] for k in fields} == {k: b[k] for k in fields}
        doubled = metrics(replace(c, unavailable_cost=c.unavailable_cost * 2), state)
        assert doubled['cost_sensitive_contention_rate'] == a['cost_sensitive_contention_rate']
        assert doubled['committed_cost_span_max'] == 2 * a['committed_cost_span_max']


def synthetic_rows(cfg, specs, phase):
    return [dict(phase=phase, candidate=spec['candidate'], policy=policy, seed=seed, physical_seed=0,
        skill_contention_rate=.4, cost_sensitive_contention_rate=.2, selected_cost_sensitive_contention_rate=.1,
        availability=.5, terminal_pending=.25, extended_cost=100.)
        for spec in specs for policy in s.POLICIES for seed in cfg[f'{phase}_seeds']]


def test_ranking_ignores_realized_cost_and_excludes_controls():
    cfg = s.settings('smoke'); specs = s.candidates(cfg); rows = synthetic_rows(cfg, specs, 'screening')
    chosen = 'overlap3__fast1_slow352'
    for row in rows:
        if row['candidate'] in s.CONTROLS: row.update(skill_contention_rate=1., cost_sensitive_contention_rate=1.)
        if row['candidate'] == chosen: row.update(skill_contention_rate=.6, extended_cost=1000.)
        elif row['candidate'] == 'overlap3__fast1_slow235': row['extended_cost'] = -1000.
    _, choice = s.rank_screen(cfg, specs, rows)
    assert choice['winner'] == chosen
    assert set(s.CONTROLS).issubset(choice['confirmation_candidates'])
    assert len(choice['confirmation_candidates']) == 7 and not choice['confirmation_outcomes_opened']


def test_no_eligible_when_rules_never_serve_consequential_subsets():
    cfg = s.settings('smoke'); specs = s.candidates(cfg); rows = synthetic_rows(cfg, specs, 'screening')
    for row in rows: row['selected_cost_sensitive_contention_rate'] = 0.
    _, choice = s.rank_screen(cfg, specs, rows)
    assert choice['status'] == 'NO_ELIGIBLE_SETTING' and choice['winner'] is None
    assert choice['confirmation_candidates'] == list(s.CONTROLS)
    assert s.confirm(cfg, choice, [])['status'] == 'NOT_APPLICABLE'


def test_five_99_percent_gates_require_both_rules_and_every_bound():
    cfg = s.settings('smoke'); specs = s.candidates(cfg)[:1]
    rows = synthetic_rows(cfg, specs, 'confirmation'); selection = dict(winner=specs[0]['candidate'])
    result = s.confirm(cfg, selection, rows)
    assert result['status'] == 'CONFIRMED' and result['confidence'] == .99
    assert len(result['metrics']) == 5
    for row in rows:
        if row['policy'] == 'priority_greedy': row['selected_cost_sensitive_contention_rate'] = .01
    assert s.confirm(cfg, selection, rows)['status'] == 'NOT_CONFIRMED'
    for row in rows: row.update(selected_cost_sensitive_contention_rate=.1, terminal_pending=2.)
    assert not s.confirm(cfg, selection, rows)['metrics']['terminal_pending']['pass_gate']


def test_seed_panels_new_separate_and_locked_against_v1_v2():
    cfg = s.settings('smoke'); audit = s.seed_audit(cfg)
    assert audit['status'] == 'PASS' and audit['prior_derived_checked'] == 404
    with pytest.raises(ValueError, match='locked'): s.seed_audit({**cfg, 'confidence': .95})
    assert set(cfg['screening_seeds']).isdisjoint(s.settings('full')['screening_seeds'])


def test_paired_shocks_costs_and_consequence_metric_tampering_detected(tmp_path):
    cfg = s.settings('smoke'); specs = s.candidates(cfg)[:1]; streams = [io.StringIO(), io.StringIO()]
    for policy, stream in zip(s.POLICIES, streams):
        row = s.one_episode(cfg, specs[0], policy, cfg['screening_seeds'][0], 'screening', stream)
        assert row['policy_cost'] + row['tail_cost'] == row['extended_cost']
        assert row['extended_invalid_assignments'] == 0
    records = [[json.loads(line) for line in stream.getvalue().splitlines()] for stream in streams]
    assert records[0][0]['state'] == records[1][0]['state']
    assert [r['events'] for r in records[0]] == [r['events'] for r in records[1]]
    log = tmp_path / 'audit.jsonl'; log.write_text(''.join(stream.getvalue() for stream in streams))
    assert s.replay_audit(log, specs, cfg)['replayed_transitions'] == 120
    records[0][0]['metrics']['committed_cost_span_max'] = 999
    log.write_text('\n'.join(json.dumps(r) for group in records for r in group) + '\n')
    with pytest.raises(RuntimeError, match='metric'): s.replay_audit(log, specs, cfg)


def test_all_controls_and_nested_consequence_rates_are_audited():
    row = dict(candidate=s.CONTROLS[0], skill_contention_rate=1., contested_choice_rate=1.,
        cost_sensitive_contention_rate=0., selected_cost_sensitive_contention_rate=0.)
    assert s.control_audit([row])['status'] == 'PASS'
    with pytest.raises(RuntimeError, match='negative control'):
        s.control_audit([{**row, 'cost_sensitive_contention_rate': .1}])
    with pytest.raises(RuntimeError, match='nested'):
        s.control_audit([{**row, 'candidate': 'other', 'selected_cost_sensitive_contention_rate': .1}])


def test_complete_smoke_grid_hashes_snapshot_seal_and_overwrite_refusal(tmp_path):
    output = tmp_path / 'run'; manifest = s.run('smoke', output)
    assert manifest['status'] == 'COMPLETED' and manifest['audit'] == 'PASS'
    assert manifest['actual_counts']['screening'] == 424
    assert manifest['actual_counts']['confirmation'] in (40, 48, 56)
    assert manifest['actual_counts'] == manifest['expected_counts']
    assert manifest['confirmation_outcomes_opened'] and not manifest['full_confirmation_outcomes_opened']
    assert manifest['metric_version'] == s.METRIC_VERSION
    assert hashlib.sha256((output / 'selection.json').read_bytes()).hexdigest() == manifest['selection_sha256']
    for module in (s.base, s.previous):
        assert (output / 'source_snapshot' / module.SEED_PATH.name).read_bytes() == module.SEED_PATH.read_bytes()
    for path, data in manifest['files'].items():
        p = output / path
        assert p.stat().st_size == data['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == data['sha256']
    audit = json.loads((output / 'audit.json').read_text())
    assert audit['metrics_replayed'] and audit['control_audit']['status'] == 'PASS'
    before = (output / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError): s.run('smoke', output)
    assert (output / 'manifest.json').read_bytes() == before
    with pytest.raises(ValueError): s.run('smoke', tmp_path / 'gpu', device='cuda')
