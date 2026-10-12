"""Physical competition, tie invariance, joint choices and sealed screening."""

from dataclasses import replace
import hashlib
import io
import json

import pytest

from ht_pdm_fjsp import maintenance_skill_contention_screen_v2 as s
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_matching


def idle(ages=(3, 3, 3, 3)):
    return DispatchState(ages, (False,) * 4, (0,) * 4, (0, 0), (-1, -1))


def metrics(c, state):
    return s.state_metrics(c, state, s.choose(c, state, "deadline_matching"))


def test_locked_grid_has_72_candidates_three_controls_and_no_dense_masks():
    cfg = s.settings("smoke"); specs = s.candidates(cfg)
    assert len(specs) == 75 and sum(r["selectable"] for r in specs) == 72
    assert len({json.dumps(r["config"], sort_keys=True) for r in specs}) == 75
    for row in specs:
        c = s.decode_config(row); c.validate()
        assert c.horizon == 60 and c.machines == 4 and c.technicians == 2
        assert row["eligible_edges"] < 8
    with pytest.raises(ValueError): s.settings("unknown")
    with pytest.raises(ValueError): s.physical_config("balanced2", "equal_control", cfg)
    with pytest.raises(ValueError): s.physical_config("overlap3", "unknown", cfg)


def test_equal_score_collision_and_physical_tradeoff_are_not_strict_skill_competition():
    cfg = s.settings("smoke"); state = idle((0, 3, 3, 0))
    for pattern in ("equal_control", "tradeoff_control"):
        c = s.physical_config("overlap3", pattern, cfg); m = metrics(c, state)
        assert m["legacy_proposal_conflict_rate"] == 1 and m["legacy_tie_rate"] == 1
        assert m["skill_contention_rate"] == m["contested_choice_rate"] == 0
        assert m["feasible_matching_cardinality"] == m["preferred_matching_cardinality"] == 2
        assert m["sensitive_choice_rate"] == int(pattern == "tradeoff_control")


def test_strict_dominated_service_required_and_joint_same_machine_choices_present():
    c = s.physical_config("overlap3", "common_1_3", s.settings("smoke"))
    m = metrics(c, idle((0, 3, 3, 0)))
    assert m["skill_contention_rate"] == m["contested_choice_rate"] == 1
    assert m["feasible_matching_cardinality"] == 2 and m["preferred_matching_cardinality"] == 1
    assert m["preferred_deficit"] == 1
    assert m["capacity_pressure_rate"] == m["skill_deficit_rate"] == 0


def test_forced_unique_assignment_is_not_joint_contested_choice():
    c = s.physical_config("overlap3", "common_1_3", s.settings("smoke"))
    m = metrics(c, idle((3, 3, 0, 0)))
    assert m["skill_contention_rate"] == 1
    assert m["sensitive_choice_rate"] == m["contested_choice_rate"] == 0


def test_pure_capacity_pressure_and_compatibility_deficit_are_separate():
    cfg = s.settings("smoke")
    c = s.physical_config("overlap3", "paired_1_3", cfg)
    m = metrics(c, idle())
    assert m["capacity_pressure_rate"] == 1
    assert m["skill_contention_rate"] == m["contested_choice_rate"] == 0
    c = s.physical_config("bottleneck1", "common_1_3", cfg)
    state = replace(idle((3, 3, 3, 0)), remaining=(1, 0), assigned=(0, -1))
    m = metrics(c, state)
    assert m["skill_deficit_rate"] == 1
    assert m["skill_contention_rate"] == m["contested_choice_rate"] == 0


def test_busy_technician_cannot_create_comparison_against_unavailable_service():
    c = s.physical_config("overlap3", "common_1_3", s.settings("smoke"))
    state = replace(idle((0, 3, 3, 0)), remaining=(1, 0), assigned=(0, -1))
    m = metrics(c, state)
    assert m["capacity_pressure_rate"] == 1 and m["feasible_matching_cardinality"] == 1
    assert m["skill_contention_rate"] == 0


def test_restoration_dominance_and_incomparable_tradeoff_use_physical_values():
    c = s.physical_config("overlap3", "common_2_4", s.settings("smoke"))
    state = idle((0, 3, 3, 0))
    # Equal duration but T0 restores strictly more on every compatible edge.
    c = replace(c, service_time=((2, 0), (2, 2), (2, 2), (2, 2)), restoration=((1., .5),) * 4)
    assert metrics(c, state)["skill_contention_rate"] == 1
    # Faster T0 restores less: neither worker physically dominates.
    c = replace(c, service_time=((1, 0), (1, 2), (1, 2), (1, 2)))
    c = replace(c, restoration=((.5, 1.),) * 4)
    assert metrics(c, state)["skill_contention_rate"] == 0


def test_new_metrics_are_invariant_to_worker_index_permutation():
    cfg = s.settings("smoke"); state = idle((0, 3, 3, 0))
    fields = ("skill_contention_rate", "contested_choice_rate", "preferred_deficit",
              "feasible_matching_cardinality", "preferred_matching_cardinality", "sensitive_choice_rate")
    for pattern in (*s.PATTERNS, "equal_control", "tradeoff_control"):
        c = s.physical_config("overlap3", pattern, cfg)
        swapped = replace(c, service_time=tuple(tuple(reversed(r)) for r in c.service_time),
            restoration=tuple(tuple(reversed(r)) for r in c.restoration))
        a, b = metrics(c, state), metrics(swapped, state)
        assert {k: a[k] for k in fields} == {k: b[k] for k in fields}


def test_empty_state_and_all_controls_cannot_masquerade_as_skill_competition():
    cfg = s.settings("smoke")
    for row in s.candidates(cfg):
        c = s.decode_config(row); m = metrics(c, idle((0, 0, 0, 0)))
        assert m["skill_contention_rate"] == m["contested_choice_rate"] == 0
        assert m["feasible_matching_cardinality"] == 0
        if not row["selectable"]:
            m = metrics(c, idle())
            assert m["skill_contention_rate"] == m["contested_choice_rate"] == 0


def synthetic_rows(cfg, specs, phase):
    return [dict(phase=phase, candidate=spec["candidate"], policy=policy, seed=seed, physical_seed=0,
        skill_contention_rate=.4, contested_choice_rate=.2, availability=.5, terminal_pending=.25, extended_cost=100.)
        for spec in specs for policy in s.POLICIES for seed in cfg[f"{phase}_seeds"]]


def test_ranking_uses_primary_then_joint_not_cost_or_controls():
    cfg = s.settings("smoke"); specs = s.candidates(cfg); rows = synthetic_rows(cfg, specs, "screening")
    for row in rows:
        if row["candidate"] in s.CONTROLS: row.update(skill_contention_rate=1., contested_choice_rate=1.)
        if row["candidate"] == "overlap3__common_1_2": row.update(skill_contention_rate=.6, contested_choice_rate=.1, extended_cost=1000.)
        if row["candidate"] == "overlap3__common_1_3": row.update(skill_contention_rate=.5, contested_choice_rate=.9, extended_cost=0.)
    _, choice = s.rank_screen(cfg, specs, rows)
    assert choice["winner"] == "overlap3__common_1_2"
    assert choice["runner_up"] == "overlap3__common_1_3"
    assert set(choice["confirmation_candidates"]) == {choice["winner"], choice["runner_up"], *s.CONTROLS}
    assert not choice["confirmation_outcomes_opened"]


def test_no_eligible_candidate_confirms_controls_without_forcing_winner():
    cfg = s.settings("smoke"); specs = s.candidates(cfg); rows = synthetic_rows(cfg, specs, "screening")
    for row in rows: row["contested_choice_rate"] = .01
    _, choice = s.rank_screen(cfg, specs, rows)
    assert choice["status"] == "NO_ELIGIBLE_SETTING" and choice["winner"] is None
    assert choice["confirmation_candidates"] == list(s.CONTROLS)
    assert s.confirm(cfg, choice, rows)["status"] == "NOT_APPLICABLE"


def test_confirmation_keeps_both_rules_and_requires_every_registered_gate():
    cfg = s.settings("smoke"); specs = s.candidates(cfg)[:1]
    rows = synthetic_rows(cfg, specs, "confirmation"); selection = dict(winner=specs[0]["candidate"])
    assert s.confirm(cfg, selection, rows)["status"] == "CONFIRMED"
    for row in rows:
        if row["policy"] == "priority_greedy": row["contested_choice_rate"] = .01
    assert s.confirm(cfg, selection, rows)["status"] == "NOT_CONFIRMED"
    for row in rows: row.update(contested_choice_rate=.2, terminal_pending=2.)
    assert not s.confirm(cfg, selection, rows)["metrics"]["terminal_pending"]["pass_gate"]


def test_seeds_are_new_locked_and_separate_from_v1():
    cfg = s.settings("smoke"); audit = s.seed_audit(cfg)
    assert audit["status"] == "PASS" and audit["v1_derived_checked"] == 202
    own = set(cfg["screening_seeds"] + cfg["confirmation_seeds"] + s.settings("full")["screening_seeds"] + s.settings("full")["confirmation_seeds"])
    old = s.base.settings("full")
    assert not own.intersection(old["screening_seeds"] + old["confirmation_seeds"])
    with pytest.raises(ValueError, match="locked"): s.seed_audit({**cfg, "horizon": 12})


def test_paired_episode_replay_costs_and_metric_tampering(tmp_path):
    cfg = s.settings("smoke"); specs = s.candidates(cfg)[:1]; streams = [io.StringIO(), io.StringIO()]
    for policy, stream in zip(s.POLICIES, streams):
        row = s.one_episode(cfg, specs[0], policy, cfg["screening_seeds"][0], "screening", stream)
        assert row["policy_cost"] + row["tail_cost"] == row["extended_cost"]
        assert row["extended_invalid_assignments"] == 0
        assert 0 <= row["contested_choice_rate"] <= row["skill_contention_rate"] <= 1
    records = [[json.loads(line) for line in stream.getvalue().splitlines()] for stream in streams]
    assert records[0][0]["state"] == records[1][0]["state"]
    assert [r["events"] for r in records[0]] == [r["events"] for r in records[1]]
    log = tmp_path / "audit.jsonl"; log.write_text("".join(stream.getvalue() for stream in streams))
    assert s.replay_audit(log, specs, cfg)["replayed_transitions"] == 120
    records[0][0]["metrics"]["skill_contention_rate"] = 99
    log.write_text("\n".join(json.dumps(r) for group in records for r in group) + "\n")
    with pytest.raises(RuntimeError, match="metric"): s.replay_audit(log, specs, cfg)


def test_control_audit_catches_false_positive_and_joint_inconsistency():
    row = dict(candidate=s.CONTROLS[0], skill_contention_rate=0., contested_choice_rate=0., sensitive_choice_rate=0.)
    assert s.control_audit([row])["status"] == "PASS"
    with pytest.raises(RuntimeError, match="negative control"): s.control_audit([{**row, "skill_contention_rate": .1}])
    with pytest.raises(RuntimeError, match="subset"): s.control_audit([{**row, "candidate": "other", "contested_choice_rate": .1}])


def test_complete_smoke_artifact_grid_hashes_controls_and_overwrite_refusal(tmp_path):
    output = tmp_path / "run"; manifest = s.run("smoke", output)
    assert manifest["status"] == "COMPLETED" and manifest["audit"] == "PASS"
    assert manifest["actual_counts"]["screening"] == 600
    assert manifest["expected_counts"] == manifest["actual_counts"]
    assert manifest["actual_counts"]["confirmation"] in (24, 32, 40)
    assert manifest["metric_version"] == s.METRIC_VERSION
    assert manifest["confirmation_outcomes_opened"]
    assert not manifest["full_confirmation_outcomes_opened"]
    assert hashlib.sha256((output / "selection.json").read_bytes()).hexdigest() == manifest["selection_sha256"]
    for path, data in manifest["files"].items():
        p = output / path
        assert p.stat().st_size == data["bytes"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == data["sha256"]
    audit = json.loads((output / "audit.json").read_text())
    assert audit["control_audit"]["status"] == "PASS" and audit["metrics_replayed"]
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError): s.run("smoke", output)
    assert (output / "manifest.json").read_bytes() == before
    with pytest.raises(ValueError): s.run("smoke", tmp_path / "gpu", device="cuda")
