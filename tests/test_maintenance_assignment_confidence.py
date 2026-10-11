"""Empirical guard, source isolation, coverage, held-out seeds and paired budgets."""
from dataclasses import asdict, replace
import itertools
import json
import numpy as np
import pytest
import torch

from ht_pdm_fjsp import maintenance_assignment_confidence_benchmark as b
from ht_pdm_fjsp.maintenance_assignment_confidence import counts, run
from ht_pdm_fjsp.maintenance_assignment_confidence_data import probe_anchor
from ht_pdm_fjsp.maintenance_assignment_confidence_policy import choose_assignment, calibrate, ConfidenceController
from ht_pdm_fjsp.maintenance_assignment_confidence_statistics import summarize
from ht_pdm_fjsp.maintenance_assignment_confidence_training import draw_indices
from ht_pdm_fjsp.maintenance_assignment_policy import AssignmentPolicy
from ht_pdm_fjsp.maintenance_assignment_value_policy import RelativeValueHead
from ht_pdm_fjsp.maintenance_autoregressive_policy import feature_batch, STOP
from ht_pdm_fjsp.maintenance_dispatch import validate_state, validate_matching, DispatchState
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_waiting import transition, WaitingConfig


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def calibration(enabled=True, margin=.2):
    return dict(enabled=enabled, margin=margin if enabled else None)


def test_override_requires_gain_beyond_empirical_margin_and_disagreement():
    # Lower score means lower predicted cost; ID2 (WAIT/other set) is irrelevant.
    scores = np.array([[0., -.3, -100.], [0., -.3, -100.], [0., -.3, -100.]])
    assert choose_assignment(scores, 0, [0, 1], True, calibration())[0] == 1
    assert choose_assignment(scores, 0, [0, 1], True, calibration(margin=.4))[0] == 0
    assert choose_assignment(scores, 0, [0, 1], True, calibration(False))[0] == 0
    assert choose_assignment(scores, 0, [0, 1], False, calibration(False))[0] == 1
    scores[:, 1] = [-.9, -.3, .3]
    assert choose_assignment(scores, 0, [0, 1], True, calibration(margin=0.))[0] == 0


def test_guard_ties_keep_original_and_gain_sign_is_original_minus_alternative():
    scores = np.array([[-.5, 0.], [-.5, 0.], [-.5, 0.]])
    selected, audit = choose_assignment(scores, 1, [0, 1], False, calibration())
    assert selected == 0 and audit["predicted_gains"] == [.5, 0.]
    selected, _ = choose_assignment(np.zeros((3, 2)), 1, [0, 1], False, calibration())
    assert selected == 1
    selected, _ = choose_assignment(scores, 1, [1], True, calibration())
    assert selected == 1
    with pytest.raises(ValueError):
        choose_assignment(np.full((3, 2), np.nan), 1, [0, 1], False, calibration())


def test_calibration_accounts_for_mc_uncertainty_and_ignores_other_machine_classes():
    class Scores:
        def __call__(self, features):
            return torch.tensor([[0., -1., -5., -8.]]).repeat(len(features["context"]), 1)
    data = dict(context=torch.zeros(2, 96), raw_pairs=torch.zeros(2, 4, 2, 13),
        pairs=torch.tensor([[0, 0, 1], [1, 2, 3]]), original_ids=torch.tensor([0, 0]),
        targets=torch.tensor([-.2, -3.]), mc_se=torch.tensor([.1, .1]),
        conditions=torch.tensor([0, 0]), origins=torch.tensor([1, 1]), ticks=torch.tensor([18, 30]))
    cfg = b.settings("smoke")
    margins, rows = calibrate([Scores()] * 3, data, cfg)
    assert margins["dense"]["samples"] == 1 and margins["dense"]["enabled"]
    assert margins["dense"]["margin"] == pytest.approx(1. - (.2 - 1.96 * .1))
    assert rows[0]["observed_gain"] == pytest.approx(.2)
    assert margins["skill_mask"]["margin"] is None and not margins["skill_mask"]["enabled"]
    margins, _ = calibrate([Scores()] * 3, data, {**cfg, "calibration_minimum": 10})
    assert not margins["dense"]["enabled"]


def test_matched_draw_budget_changes_coverage_and_never_uses_probe_for_control():
    data = dict(conditions=torch.tensor([0, 1, 0, 1, 0, 1]), origins=torch.tensor([0, 0, 1, 1, 1, 1]),
        ticks=torch.tensor([0, 0, 6, 6, 30, 30]))
    natural = draw_indices(data, "natural", 64, np.random.default_rng(3))
    broad = draw_indices(data, "broad", 64, np.random.default_rng(3))
    assert len(natural) == len(broad) == 64
    assert data["origins"][natural].sum() == 0 and data["origins"][broad].sum() == 32
    assert set(data["ticks"][broad].tolist()) == {0, 6, 30}
    assert torch.equal(broad, draw_indices(data, "broad", 64, np.random.default_rng(3)))
    with pytest.raises(RuntimeError):
        draw_indices({k: v[:2] for k, v in data.items()}, "broad", 64, np.random.default_rng(3))


@pytest.mark.parametrize("condition", b.CONDITIONS)
def test_generated_probes_follow_real_legal_transitions_at_late_clock(condition):
    c = b.make_config(23100010, condition)
    seen = set()
    for cohort, tick in itertools.product(range(23100010, 23100030), [6, 18, 30, 42]):
        identity = dict(condition=condition, cohort=cohort, shock=23100020, model_seed=18100000, tick=tick, origin="probe")
        s = probe_anchor(c, identity); seen.add(max(s.remaining))
        assert probe_anchor(c, identity) == s
        for _ in range(2):
            validate_state(c, s)
            s, _, _ = transition(c, s, (), (False,) * 4)
        validate_state(c, s)
    assert {0, 1, 2}.issubset(seen)


def test_controller_changes_workers_only_and_calibration_condition_uses_structural_edges(monkeypatch):
    c = WaitingConfig(4, 2, 60, 4, .4, ((2, 5), (3, 2), (5, 4), (2, 3)),
        ((.5, 1.), (.75, .5), (.25, 1.), (1., .25)), waiting_price=12.)
    s = DispatchState((1, 2, 3, 0), (False,) * 4, (0,) * 4, (0,) * 2, (-1,) * 2)
    base = AssignmentPolicy(); base.requires_grad_(False)
    monkeypatch.setattr(base, "greedy_tokens", lambda data: torch.tensor([[0, 3, STOP]]).repeat(len(data["machines"]), 1))
    class Scores:
        def eval(self): return self
        def __call__(self, data):
            values = torch.zeros(len(data["context"]), 21)
            values[:, CATALOGUE.index(((0, 1), (1, 0)))] = -1
            values[:, 0] = -100
            return values
    margins = {"dense": calibration(margin=.2), "skill_mask": calibration(False)}
    ctrl = ConfidenceController(base, [Scores()] * 3, "broad_gated", margins, b.settings("smoke"))
    assert ctrl.select(c, s, 48, 48)[0] == ((0, 1), (1, 0))
    assert ctrl.last_decode_audit[0]["condition"] == "dense"
    # Corrective edge must remain exact, not merely preserve its machine.
    failed = replace(s, failed=(True, False, False, False))
    assert ctrl.select(c, failed, 48, 48)[0] == ((0, 0), (1, 1))
    batch = feature_batch([c, c], [s, s], [48, 48], 48)
    action, _, _ = ctrl.sample(batch, deterministic=True)
    assert ctrl.decode(c, s, action[0]) == ctrl.select(c, s, 48, 48)[0]
    with pytest.raises(ValueError): ctrl.sample(batch, deterministic=False)


def test_fresh_configs_rng_and_fixed_budgets():
    _, records, audit = b.metadata()
    signatures = [r["physical_signature"] for rows in records.values() for r in rows]
    assert len(signatures) == len(set(signatures)) == 218 and not audit["outcomes_used"]
    assert b.seed_audit(b.settings("smoke"))["status"] == "PASS"
    cfg = b.settings("full"); budget = counts(cfg)
    assert budget["test_episodes"] == 2880 and budget["trained_heads"] == 18
    assert budget["train_states"] == 3072 and budget["optimizer_steps"] == 11520
    assert budget["diagnostic_states"] == 384
    identity = dict(condition="dense", cohort=22200000, shock=22300000, model_seed=18100000, tick=18, origin="probe")
    phases = ("train", "development", "calibration", "ranking_selection", "ranking_evaluation")
    assert len({b.future_seed(cfg, identity, p) for p in phases}) == 5
    assert b.future_seed(cfg, identity, "train") != b.future_seed(cfg, {**identity, "origin": "natural"}, "train")


def test_primary_grid_sign_and_missing_episode_rejected():
    cfg = b.settings("smoke"); rows = []
    metrics = ("extended_objective", "base_objective", "tail_objective", "extended_failure_cost", "extended_waiting_cost",
        "extended_maintenance_cost", "extended_unavailability_cost", "preventive_starts", "corrective_starts",
        "wait_with_pm_feasible", "pm_assignment_alternative_ticks", "assignment_override_ticks", "guard_rejected_ticks", "batch_amortized_policy_ms_mean")
    for condition, name in itertools.product(cfg["conditions"], b.NAMES):
        cost = 90. if name == "broad_gated" else 100.
        rows.append(dict(condition=condition, cohort=cfg["test_cohorts"][0], shock=cfg["test_shocks"][0],
            controller=name, train_seed=18100000, horizon=48, **{k: cost for k in metrics}))
    summary, _ = summarize(rows, cfg)
    assert len(summary["primary_contrasts"]) == 4
    assert all(r["relative_gain"] == pytest.approx(.1) for r in summary["primary_contrasts"].values())
    with pytest.raises(RuntimeError): summarize(rows[:-1], cfg)


def test_run_preserves_existing_artifacts_and_missing_source_leaves_failed_manifest(tmp_path):
    prior = tmp_path / "prior"; prior.mkdir(); (prior / "keep").write_text("intact")
    with pytest.raises(ValueError): run(prior, tmp_path / "missing")
    assert (prior / "keep").read_text() == "intact"
    output = tmp_path / "new"
    with pytest.raises(FileNotFoundError): run(output, tmp_path / "missing")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "FAILED" and not manifest["full_test_outcomes_opened"]
