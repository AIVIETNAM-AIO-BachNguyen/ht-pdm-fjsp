"""Scientific-contract tests for current-decoder same-state counterfactuals."""
import csv
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ht_pdm_fjsp import maintenance_ar_decision_diagnostic as e
from ht_pdm_fjsp import maintenance_ar_diagnostic_forks as f
from ht_pdm_fjsp import maintenance_autoregressive_benchmark as b
from ht_pdm_fjsp import maintenance_autoregressive_pilot as p
from ht_pdm_fjsp.maintenance_autoregressive_policy import CONTEXT, STOP, feature_batch
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, eligible_edges, validate_matching
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays, feasible, membership
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def probe(condition="dense"):
    config = b.make_config(17500000, condition, 60)
    state = b.initial_state(config, 17500000)
    model = p.model_for(b.settings("full"), 16100000, CONTEXT)
    return config, state, model


@pytest.mark.parametrize("condition", ("dense", "skill_mask"))
def test_candidate_menu_complete_cm_preserving_and_partial_deferral(condition):
    config, initial, _ = probe(condition)
    for state in (initial, replace(initial, failed=(True, False, False, False))):
        for original in feasible(config, state):
            roots, groups, no_pm = f.candidates(config, state, original)
            cm = tuple(pair for pair in original if state.failed[pair[0]])
            pm = tuple(pair for pair in original if not state.failed[pair[0]])
            machines = {m for m, _ in pm}
            expected = {a for a in feasible(config, state)
                        if tuple(pair for pair in a if state.failed[pair[0]]) == cm
                        and {m for m, _ in a if not state.failed[m]} <= machines}
            assert set(roots) == expected and roots[0] == original and roots[no_pm] == cm
            assert len(roots) <= 7
            for i, action in enumerate(roots):
                validate_matching(config, state, action)
                assert (i in groups["timing"]) == all(state.failed[m] or (m, j) in pm for m, j in action)
                assert (i in groups["assignment"]) == ({m for m, _ in action if not state.failed[m]} == machines)
                assert (i in groups["best_single"]) == (i in groups["timing"] or i in groups["assignment"])
    if condition == "dense":
        roots, groups, _ = f.candidates(config, initial, ((0, 0), (1, 1)))
        assert len(roots) == 7
        assert roots.index(((0, 1),)) in groups["joint"]
        assert roots.index(((0, 1),)) not in groups["best_single"]


@pytest.mark.parametrize("condition", ("dense", "skill_mask"))
def test_vector_features_and_greedy_actions_match_scalar_in_busy_states(condition):
    config, state, model = probe(condition)
    env = WaitingEnv(config)
    env.reset(876, state)
    states = []
    for tick in range(48):
        states.append(env.state)
        env.step(model.select(config, env.state, 48-tick, 48)[0])
    data = {key: np.concatenate([arrays(s)[key] for s in states]) for key in arrays(state)}
    fast = f.ar_batch_features(config, data, 17)
    scalar = feature_batch([config]*len(states), states, [17]*len(states), 48, CONTEXT)
    assert fast.keys() == scalar.keys()
    for key in fast:
        torch.testing.assert_close(fast[key], scalar[key], rtol=0, atol=0)
    selected, audit = f.rl_batch(config, data, 17, model)
    for i, current in enumerate(states):
        action = model.select(config, current, 17, 48)[0]
        np.testing.assert_array_equal(selected[i], membership(4, 2)[CATALOGUE.index(action)].astype(bool))
    assert audit["decision_rows"] == 48
    for parameter in model.actor.parameters():
        parameter.data.zero_()
    selected, audit = f.rl_batch(config, data, 17, model)
    assert audit["near_tie_scalar_rows"] == sum(any(any(row) for row in eligible_edges(config, s)) for s in states)
    for i, current in enumerate(states):
        action = model.select(config, current, 17, 48)[0]
        np.testing.assert_array_equal(selected[i], membership(4, 2)[CATALOGUE.index(action)].astype(bool))


@pytest.mark.parametrize("tick", (0, 42, 47))
def test_original_clock_pairing_components_boundaries_and_scalar_replay(tick):
    config, state, model = probe()
    original = model.select(config, state, 48-tick, 48)[0]
    roots, _, _ = f.candidates(config, state, original)
    before = e.parameters_sha(model)
    value = f.fork_roots(config, state, roots, model, 891, 2, tick, record_trace=True)
    assert value["transitions"] == len(roots)*2*(60-tick)
    assert value["counts"]["decision_rows"] == len(roots)*2*(47-tick)
    assert value["counts"]["tail_rows"] == len(roots)*2*12
    np.testing.assert_array_equal(value["costs"], value["base"]+value["tail"])
    np.testing.assert_array_equal(value["costs"], sum(value["components"].values()))
    assert f.scalar_replay(config, state, value, model, tick)
    for i, root in enumerate(roots):
        alone = f.fork_roots(config, state, (root,), model, 891, 2, tick)
        np.testing.assert_array_equal(value["costs"][i], alone["costs"][0])
        np.testing.assert_array_equal(value["shocks"], alone["shocks"])
    assert before == e.parameters_sha(model)


@pytest.mark.parametrize("condition", ("dense", "skill_mask"))
def test_unordered_matching_probability_and_prefix_stop_scores(condition):
    config, state, model = probe(condition)
    before = e.parameters_sha(model)
    original, info, masses, sequences, prefixes = f.policy_scores(config, state, 48, model)
    assert model.algorithm == CONTEXT and e.parameters_sha(model) == before
    assert original == model.select(config, state, 48, 48)[0]
    assert set(masses) == set(feasible(config, state))
    assert sum(masses.values()) == pytest.approx(1, abs=2e-6)
    assert len(sequences) == sum(2 if len(action) == 2 else 1 for action in feasible(config, state))
    for action, mass in masses.items():
        group = [row for row in sequences if row["action"] == action]
        assert mass == pytest.approx(sum(row["probability"] for row in group), abs=1e-8)
        for row in group:
            trace = torch.tensor([row["trace"]], dtype=torch.long)
            lp, _, _ = model.evaluate(feature_batch([config], [state], [48], 48, CONTEXT), trace)
            assert row["log_probability"] == pytest.approx(lp.item(), abs=2e-6)
    assert prefixes[0]["context_total_variation"] == 0
    assert prefixes[0]["context_l2"] == 0
    assert len(prefixes) == len([token for token in info["token_trace"] if token >= 0])
    assert prefixes[-1]["token"] == STOP
    for row in prefixes:
        assert sum(row["probabilities"]) == pytest.approx(1, abs=1e-6)
        assert all(prob == 0 for prob, mask in zip(row["probabilities"], row["live_mask"], strict=True) if not mask)


def test_selection_is_independent_of_evaluation_and_negative_gains_are_retained():
    selection = np.array([[10, 12], [8, 9], [8, 9]], dtype=float)
    assert f.choose(selection, (0, 1, 2)) == 1
    assert f.choose(np.ones((3, 2)), (0, 1, 2)) == 0
    selected = f.choose(selection, (0, 1, 2))
    evaluation = np.array([[1, 2], [9, 10], [0, 0]], dtype=float)
    gain, se = f.paired_gain(evaluation, 0, selected)
    assert gain == -8 and se == 0
    for bad in ((1, 2), (0, 9), (0, 0)):
        with pytest.raises(ValueError):
            f.choose(selection, bad)
    with pytest.raises(ValueError):
        f.choose(np.array([[np.nan, 1]]), (0,))


def test_fresh_seed_metadata_and_fixed_budget_contract():
    cfg = e.settings("full")
    assert e.counts(cfg) == dict(models=3, trajectories=144, trajectory_ticks=6912, states=720,
        training_steps=0, max_forecast_transitions=19740672)
    assert e.seed_audit(cfg)["status"] == "PASS"
    with pytest.raises(ValueError):
        e.seed_audit({**cfg, "evaluation_seed": cfg["selection_seed"]})
    panels, _, audit = e.metadata()
    assert audit["outcomes_used"] is False
    blocked = set(b.historical_pool())
    old_panels, _, training = b.metadata()
    for values in old_panels.values():
        blocked.update(e.boundary.signature(c) for c in values.values())
    for values in training.values():
        blocked.update(e.boundary.signature(row[0]) for row in values.values())
    seen = set()
    for mode, group in panels.items():
        for (cohort, cond), config in group.items():
            signature = e.boundary.signature(config)
            assert signature not in blocked | seen
            seen.add(signature)
            state = b.initial_state(config, cohort)
            focal = b.scenario(config, cohort)["focal_machine"]
            assert sum(eligible_edges(config, state)[focal]) == 2
    derived = []
    for mode in ("full", "smoke"):
        cfg = e.settings(mode)
        for values in itertools.product(cfg["conditions"], cfg["cohorts"], cfg["physical_shocks"], cfg["model_seeds"], cfg["landmarks"]):
            identity = dict(zip(e.KEYS, values, strict=True))
            derived.extend(e.forecast_seed(cfg, identity, phase) for phase in e.PHASES)
    assert len(derived) == len(set(derived))


def summary_rows(cfg, exposure=1):
    rows = []
    for values in itertools.product(cfg["conditions"], cfg["cohorts"], cfg["physical_shocks"], cfg["model_seeds"], cfg["landmarks"]):
        row = dict(zip(e.KEYS, values, strict=True))
        for contrast in f.CONTRASTS:
            row.update({f"{contrast}_exposure": exposure, f"{contrast}_gain": 2. if exposure else 0., f"{contrast}_selected": 1 if exposure else 0})
        row.update(joint_vs_best_single_gain=0., defer_all_gain=2., joint_selected_both_changed=0)
        rows.append(row)
    return rows


def test_bootstrap_resamples_exposure_and_detects_incomplete_grid():
    cfg = e.settings("full")
    summary, cells = e.summarize(summary_rows(cfg), cfg)
    assert len(cells) == 2*3*3*12
    for result in summary["primary_contrasts"].values():
        assert result["estimate"] == 2
        assert result["confidence_interval"] == [2, 2]
        assert result["confidence"] == 1-.05/6
        assert result["status"] == "POSITIVE_CORRECTION_GAIN"
    empty, _ = e.summarize(summary_rows(cfg, 0), cfg)
    for result in empty["primary_contrasts"].values():
        assert result["estimate"] is None and result["confidence_interval"] is None
        assert result["status"] == "INSUFFICIENT"
    rows = summary_rows(cfg)
    with pytest.raises(RuntimeError):
        e.summarize(rows[:-1], cfg)
    with pytest.raises(RuntimeError):
        e.summarize(rows + [rows[0]], cfg)
    rows[0]["timing_gain"] = float("nan")
    with pytest.raises(ValueError):
        e.summarize(rows, cfg)


@pytest.fixture
def fake_source(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    cfg = b.settings("full")
    selections, models, artifacts = {}, {}, {}
    config, state, _ = probe()
    for seed in cfg["train_seeds"]:
        model = p.model_for(cfg, seed, CONTEXT)
        path = source/f"{CONTEXT}/seed_{seed}/model.pt"
        p.save_checkpoint(path, model, cfg, seed, 10, 30720, (config, state))
        rel = str(path.relative_to(source))
        records = [dict(update=0, learned_steps=0, mean_extended_cost=100), dict(update=10, learned_steps=30720, mean_extended_cost=90)]
        selections[f"{CONTEXT}:{seed}"] = dict(update=10, learned_steps=30720, selected_checkpoint=rel, candidates=records)
        models[str(seed)] = dict(path=rel, sha256=e.sha(path), update=10, learned_steps=30720)
        artifacts[rel] = dict(sha256=e.sha(path), bytes=path.stat().st_size)
    prior_selection = source/"checkpoint_selection.json"
    prior_selection.write_text(json.dumps(dict(locked_before_test=True, selections=selections)))
    artifacts[prior_selection.name] = dict(sha256=e.sha(prior_selection), bytes=prior_selection.stat().st_size)
    manifest = dict(status="COMPLETED", audit="PASS", dirty=False, commit="test", settings=cfg, artifacts=artifacts)
    (source/"manifest.json").write_text(json.dumps(manifest))
    contract = dict(protocol=e.VERSION, source_protocol=p.VERSION, source_commit="test", source_run="test",
        models=models, files={name: e.sha(source/name) for name in ("manifest.json", "checkpoint_selection.json")})
    return source, contract


def test_source_pin_roundtrip_and_tamper_rejection(fake_source):
    source, contract = fake_source
    models, _, audit = e.verify_source(source, e.settings("smoke"), contract)
    assert set(models) == {16100000} and audit["status"] == "PASS"
    model_file = source/contract["models"]["16100000"]["path"]
    model_file.write_bytes(model_file.read_bytes()+b"tampered")
    with pytest.raises(ValueError, match="artifact mismatch"):
        e.verify_source(source, e.settings("smoke"), contract)


def test_smoke_artifact_schema_forecast_replay_and_no_new_full_outcomes(fake_source, tmp_path, monkeypatch):
    source, contract = fake_source
    original = e.verify_source
    monkeypatch.setattr(e, "verify_source", lambda path, cfg: original(path, cfg, contract))
    output = tmp_path/"smoke"
    result = e.run(output, source, "smoke")
    assert result["status"] == "COMPLETED" and result["audit"] == "PASS"
    assert result["new_full_outcomes_opened"] is False
    assert result["actual_counts"]["states"] == 4 and result["actual_counts"]["training_steps"] == 0
    rows = list(csv.DictReader((output/"states.csv").open()))
    panels, _, _ = e.metadata()
    for row in rows:
        identity = {key: row[key] if key == "condition" else int(row[key]) for key in e.KEYS}
        config = panels["smoke"][identity["cohort"], identity["condition"]]
        state = DispatchState(**{key: tuple(values) for key, values in json.loads(row["state"]).items()})
        roots = tuple(tuple(tuple(pair) for pair in action) for action in json.loads(row["roots"]))
        model, _ = p.load_checkpoint(output/"frozen_models"/f"seed_{identity['model_seed']}.pt")
        with np.load(output/row["forecast_file"], allow_pickle=False) as data:
            for phase in e.PHASES:
                value = f.fork_roots(config, state, roots, model, int(row[f"{phase}_seed"]), 2, identity["tick"])
                for key in ("costs", "base", "tail", "shocks", "first"):
                    np.testing.assert_array_equal(data[f"{phase}_{key}"], value[key])
            for contrast in (*f.CONTRASTS, "best_single"):
                eligible = tuple(json.loads(row[f"{contrast}_candidates"]))
                chosen = f.choose(data["selection_costs"], eligible)
                assert chosen == int(row[f"{contrast}_selected"])
                gain, se = f.paired_gain(data["evaluation_costs"], 0, chosen)
                assert gain == float(row[f"{contrast}_gain"]) and se == float(row[f"{contrast}_mc_se"])
    for name, record in result["artifacts"].items():
        assert e.sha(output/name) == record["sha256"]
    with pytest.raises(ValueError, match="new/empty"):
        e.run(output, source)
