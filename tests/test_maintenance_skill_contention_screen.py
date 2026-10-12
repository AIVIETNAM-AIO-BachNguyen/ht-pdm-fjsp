"""Meaningful skill pressure, fixed setting selection and audit contracts."""

from dataclasses import replace
import hashlib
import io
import json

import pytest

from ht_pdm_fjsp import maintenance_skill_contention_screen as s
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_matching
from ht_pdm_fjsp.maintenance_heterogeneous_model import reference
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv


def idle(ages=(3, 3, 3, 3)):
    return DispatchState(ages, (False,) * 4, (0,) * 4, (0, 0), (-1, -1))


def test_fixed_grid_has_25_distinct_configs_and_no_dense_setting():
    cfg = s.settings("smoke")
    specs = s.candidates(cfg)
    assert len(specs) == 25 and sum(r["selectable"] for r in specs) == 20
    assert len({json.dumps(r["config"], sort_keys=True) for r in specs}) == 25
    for row in specs:
        c = s.decode_config(row); c.validate()
        assert c.machines == 4 and c.technicians == 2 and c.horizon == 60
        assert row["eligible_edges"] < 8
        assert any(d > 0 for d in c.service_time[0])
    with pytest.raises(ValueError): s.settings("unknown")
    with pytest.raises(ValueError): s.physical_config("balanced2", "equal_control", cfg)


def test_skill_deficit_is_distinct_from_total_capacity_pressure():
    cfg = s.settings("smoke")
    c = s.physical_config("bottleneck1", "common_2_4", cfg)
    state = idle()
    m = s.state_metrics(c, state, s.choose(c, state, "deadline_matching"))
    assert m["capacity_pressure_rate"] == 1 and m["skill_deficit_rate"] == 0
    # T0 is busy on M0; T1 is free but cannot serve urgent M1 or M2.
    state = replace(state, ages=(3, 3, 3, 0), remaining=(1, 0), assigned=(0, -1))
    m = s.state_metrics(c, state, ())
    assert m["skill_deficit_rate"] == 1 and m["conflict_rate"] == 0


def test_proposed_conflicts_never_become_executed_collisions():
    cfg = s.settings("smoke")
    c = s.physical_config("overlap3", "common_2_4", cfg)
    state = replace(idle((3, 3, 0, 0)), failed=(False, True, False, False))
    global_action = s.choose(c, state, "deadline_matching")
    greedy_action = s.choose(c, state, "priority_greedy")
    assert len(global_action) == 2 and len(greedy_action) == 1
    assert s.state_metrics(c, state, global_action)["collision_excess"] == 1
    for action in (global_action, greedy_action): validate_matching(c, state, action)


def test_physical_assignment_choices_exclude_label_swaps_and_empty_actions():
    cfg = s.settings("smoke")
    state = idle((0, 3, 3, 0))
    equal = s.physical_config("overlap3", "equal_control", cfg)
    action = s.choose(equal, state, "deadline_matching")
    assert s.state_metrics(equal, state, action)["sensitive_choice_rate"] == 0
    asymmetric = s.physical_config("overlap3", "paired_2_8", cfg)
    action = s.choose(asymmetric, state, "deadline_matching")
    m = s.state_metrics(asymmetric, state, action)
    assert m["sensitive_choice_rate"] == 1 and m["selected_sensitive_rate"] == 1
    assert s.state_metrics(asymmetric, idle((0, 0, 0, 0)), ())["sensitive_choice_rate"] == 0
    for pattern in s.PATTERNS:
        c = s.physical_config("partitioned_control", pattern, cfg)
        assert s.state_metrics(c, idle(), s.choose(c, idle(), "deadline_matching"))["sensitive_choice_rate"] == 0


def test_deadline_policy_agrees_with_existing_reference_on_reachable_states():
    cfg = s.settings("smoke")
    for spec in s.candidates(cfg):
        c = s.decode_config(spec); env = WaitingEnv(c); env.reset(81)
        for tick in range(12):
            action = s.choose(c, env.state, "deadline_matching")
            assert action == reference(c, env.state, c.horizon - tick, "deadline_matching")
            env.step(action)
    with pytest.raises(ValueError): s.choose(c, idle(), "unknown")


def synthetic_rows(cfg, specs, phase, metrics=None):
    rows = []
    for spec in specs:
        for policy in s.POLICIES:
            for seed in cfg[f"{phase}_seeds"]:
                rows.append(dict(phase=phase, candidate=spec["candidate"], policy=policy,
                    seed=seed, physical_seed=0, conflict_rate=.4, sensitive_choice_rate=.2,
                    availability=.5, terminal_pending=.25, extended_cost=100., **(metrics or {})))
    return rows


def test_selection_ignores_control_and_cost_and_breaks_ties_reproducibly():
    cfg = s.settings("smoke"); specs = s.candidates(cfg)
    rows = synthetic_rows(cfg, specs, "screening")
    for row in rows:
        if row["candidate"] == "overlap3__equal_control":
            row.update(conflict_rate=1., sensitive_choice_rate=1., extended_cost=0.)
        elif row["candidate"] == "overlap3__paired_2_8": row["extended_cost"] = -1000.
    ranking, choice = s.rank_screen(cfg, specs, rows)
    expected = min(r["candidate"] for r in specs if r["selectable"])
    assert choice["winner"] == expected
    assert choice["strongest_contention"] == "overlap3__equal_control"
    assert len(choice["confirmation_candidates"]) == 4
    assert not choice["confirmation_outcomes_opened"]
    assert ranking[0]["eligible"]


def test_no_eligible_candidate_produces_no_forced_confirmation():
    cfg = s.settings("smoke"); specs = s.candidates(cfg)
    rows = synthetic_rows(cfg, specs, "screening")
    for row in rows: row["sensitive_choice_rate"] = 0.
    _, choice = s.rank_screen(cfg, specs, rows)
    assert choice["status"] == "NO_ELIGIBLE_SETTING"
    assert not choice["confirmation_candidates"]
    assert s.confirm(cfg, choice, [])["status"] == "NOT_APPLICABLE"


def test_confirmation_requires_all_gates_under_both_policies():
    cfg = s.settings("smoke"); specs = s.candidates(cfg)
    winner = specs[0]["candidate"]; choice = dict(winner=winner)
    rows = synthetic_rows(cfg, specs[:1], "confirmation")
    result = s.confirm(cfg, choice, rows)
    assert result["status"] == "CONFIRMED"
    for row in rows:
        if row["policy"] == "priority_greedy": row["conflict_rate"] = .05
    assert s.confirm(cfg, choice, rows)["status"] == "NOT_CONFIRMED"
    for row in rows: row.update(conflict_rate=.4, terminal_pending=2.)
    assert not s.confirm(cfg, choice, rows)["metrics"]["terminal_pending"]["pass_gate"]


def test_raw_derived_seed_panels_are_separate_and_settings_are_locked():
    cfg = s.settings("smoke"); full = s.settings("full")
    assert s.seed_audit(cfg)["status"] == "PASS"
    panels = [set(c[f"{phase}_seeds"]) for c in (cfg, full) for phase in ("screening", "confirmation")]
    assert all(not a & b for i, a in enumerate(panels) for b in panels[i + 1:])
    changed = {**cfg, "horizon": 12}
    with pytest.raises(ValueError, match="locked"): s.seed_audit(changed)


def test_episode_pairing_cost_and_replay_detect_tampering(tmp_path):
    cfg = s.settings("smoke"); specs = s.candidates(cfg)[:1]
    streams = [io.StringIO(), io.StringIO()]
    for policy, stream in zip(s.POLICIES, streams):
        row = s.one_episode(cfg, specs[0], policy, cfg["screening_seeds"][0], "screening", stream)
        assert row["policy_cost"] + row["tail_cost"] == row["extended_cost"]
        assert 0 <= row["availability"] <= 1 and 0 <= row["technician_utilization"] <= 1
    records = [[json.loads(line) for line in stream.getvalue().splitlines()] for stream in streams]
    assert records[0][0]["state"] == records[1][0]["state"]
    assert [r["events"] for r in records[0]] == [r["events"] for r in records[1]]
    log = tmp_path / "audit.jsonl"
    log.write_text("".join(stream.getvalue() for stream in streams))
    assert s.replay_audit(log, specs, cfg)["replayed_transitions"] == 120
    records[0][0]["following"]["ages"][0] = 99
    log.write_text("\n".join(json.dumps(r) for group in records for r in group) + "\n")
    with pytest.raises(RuntimeError, match="transition"): s.replay_audit(log, specs, cfg)


def test_complete_smoke_artifact_grid_seal_hashes_and_overwrite_refusal(tmp_path):
    output = tmp_path / "run"
    manifest = s.run("smoke", output)
    assert manifest["status"] == "COMPLETED" and manifest["audit"] == "PASS"
    assert manifest["actual_counts"]["screening"] == 200
    assert manifest["expected_counts"] == manifest["actual_counts"]
    selection = output / "selection.json"
    assert hashlib.sha256(selection.read_bytes()).hexdigest() == manifest["selection_sha256"]
    for path, data in manifest["files"].items():
        p = output / path
        assert p.stat().st_size == data["bytes"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == data["sha256"]
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError): s.run("smoke", output)
    assert (output / "manifest.json").read_bytes() == before
    with pytest.raises(ValueError): s.run("smoke", tmp_path / "gpu", device="cuda")
