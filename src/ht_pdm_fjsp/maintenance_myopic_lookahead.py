"""Fresh-panel comparison on the fixed exploratory overlap3/fast1/slow325 setting."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_skill_contention_screen_v3 as screen
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays, batch_tick, reference_batch
from ht_pdm_fjsp.maintenance_opportunity_cost import Journal, write_csv, write_json
from ht_pdm_fjsp.maintenance_waiting import ENV_VERSION, WaitingEnv, transition

VERSION = "maintenance_myopic_lookahead_v1"
ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = ROOT / "configs" / f"{VERSION}_seed_registry.json"
PLAN_PATH = ROOT / "docs" / f"{VERSION}_plan.md"
CANDIDATE = "overlap3__fast1_slow325"
CONTROLLERS = ("myopic_matching", "joint_lookahead", "assignment_lookahead")
LANDMARKS = (0, 8, 16, 24, 32, 40, 47)
SOURCES = ("maintenance_myopic_lookahead.py", "maintenance_skill_contention_screen_v3.py",
    "maintenance_skill_contention_screen_v2.py", "maintenance_skill_contention_screen.py",
    "maintenance_dispatch.py", "maintenance_waiting.py", "maintenance_heterogeneous_model.py",
    "maintenance_opportunity_cost.py")


def settings(profile):
    if profile not in ("smoke", "full"):
        raise ValueError("unknown profile")
    raw = json.loads(SEED_PATH.read_text())["panels"][profile]
    return dict(profile=profile, horizon=48, tail_ticks=12, lookahead_ticks=12,
        scenarios=128 if profile == "full" else 8,
        evaluation_seeds=list(range(raw["evaluation_start"], raw["evaluation_start"] + raw["evaluation_count"])),
        analysis_seed=raw["analysis_seed"], forecast_seed=raw["forecast_seed"],
        bootstrap_samples=20000 if profile == "full" else 200,
        confidence=.99, practical_margin=.01, availability_margin=.01, pending_margin=.10,
        landmarks=list(LANDMARKS), controllers=list(CONTROLLERS), candidate=CANDIDATE,
        max_seconds=7200 if profile == "full" else 600, audit_seeds=2 if profile == "full" else 4,
        train_seeds=[], development_seeds=[], training="NOT_APPLICABLE", tail_policy="myopic_matching")


def make_config(cfg):
    return screen.physical_config("overlap3", "fast1_slow325", cfg)


def physical_seed(seed):
    return role_seed(seed, f"{VERSION}:physical")


def forecast_seed(cfg, seed, tick, phase="selection"):
    if phase not in ("selection", "holdout") or tick not in range(cfg["horizon"]):
        raise ValueError("invalid forecast role")
    return role_seed(cfg["forecast_seed"], f"{VERSION}:{phase}:{seed}:{tick}")


def integers(v):
    if isinstance(v, dict):
        return set().union(set(), *(integers(x) for x in v.values()))
    if isinstance(v, list):
        return set().union(set(), *(integers(x) for x in v))
    return {v} if isinstance(v, int) and not isinstance(v, bool) else set()


def seed_audit(cfg):
    groups, derived, historical, hashes = [], [], set(), {}
    for path in sorted((ROOT / "configs").glob("*seed_registry.json")):
        if path == SEED_PATH:
            continue
        historical |= integers(json.loads(path.read_text()))
        hashes[path.name] = screen.sha(path)
    for profile in ("full", "smoke"):
        c = settings(profile)
        groups += [set(c["evaluation_seeds"]), {c["analysis_seed"]}, {c["forecast_seed"]}]
        derived += [physical_seed(seed) for seed in c["evaluation_seeds"]]
        derived.append(role_seed(c["analysis_seed"], f"{VERSION}:bootstrap"))
        derived += [forecast_seed(c, seed, tick) for seed, tick in itertools.product(c["evaluation_seeds"], range(c["horizon"]))]
        derived += [forecast_seed(c, seed, tick, "holdout") for seed, tick in itertools.product(c["evaluation_seeds"], c["landmarks"])]
    prior = set()
    for module in (screen.base, screen.previous, screen):
        for profile in ("full", "smoke"):
            old = module.settings(profile)
            prior.update(role_seed(seed, f"{module.VERSION}:physical") for phase in ("screening", "confirmation") for seed in old[f"{phase}_seeds"])
            prior.add(role_seed(old["analysis_seed"], f"{module.VERSION}:bootstrap"))
    own = set.union(*groups)
    if any(a & b for a, b in itertools.combinations(groups, 2)) or own & (historical | prior):
        raise ValueError("raw seed overlap")
    if len(set(derived)) != len(derived) or set(derived) & (own | historical | prior):
        raise ValueError("derived seed overlap")
    if cfg != settings(cfg["profile"]):
        raise ValueError("settings must match locked profile")
    return dict(status="PASS", raw_unique=len(own), derived_unique=len(derived),
        prior_derived_checked=len(prior), historical_registry_hashes=hashes,
        intentional_pairing="physical tape across all methods; selection scenarios across roots and planning methods")


def myopic(c, state):
    return screen.choose(c, state, "deadline_matching")


def roots(c, state):
    baseline = myopic(c, state)
    return (baseline, *(a for a in screen.action_space(c, state)[2] if a != baseline))


def eligible_indices(actions, assignment_only=False):
    subset = tuple(m for m, _ in actions[0])
    return tuple(i for i, action in enumerate(actions)
        if not assignment_only or tuple(m for m, _ in action) == subset)


def minimum(means, eligible):
    if not eligible or eligible[0] != 0 or not np.isfinite(means).all():
        raise ValueError("baseline must be included in finite root scores")
    lowest = min(means[i] for i in eligible)
    return next(i for i in eligible if means[i] <= lowest + 1e-10)


def forecast_values(c, state, actions, horizon, scenarios, seed, check_timeout=None):
    """Independent forecast draws; current state only, no access to a real env."""
    if horizon < 1 or scenarios < 2 or not actions:
        raise ValueError("positive forecast horizon and at least two scenarios required")
    for action in actions:
        validate_matching(c, state, action)
    b = len(actions)
    states = arrays(state, b * scenarios)
    start = np.zeros((b, c.machines, c.technicians), dtype=bool)
    for i, action in enumerate(actions):
        for m, j in action:
            start[i, m, j] = True
    chosen = np.repeat(start, scenarios, axis=0)
    events = np.random.default_rng(seed).random((horizon, scenarios, c.machines)) < c.failure_probability
    total = np.zeros(b * scenarios)
    for tick in range(horizon):
        if check_timeout is not None:
            check_timeout()
        if tick:
            chosen = reference_batch(c, states, horizon - tick, "deadline_matching")
        states, costs = batch_tick(c, states, chosen, np.tile(events[tick], (b, 1)))
        total += costs
    samples = total.reshape(b, scenarios)
    if not np.isfinite(samples).all():
        raise RuntimeError("non-finite forecast cost")
    return dict(samples=samples, means=samples.mean(1), mc_se=samples.std(1, ddof=1) / np.sqrt(scenarios),
        transitions=b * scenarios * horizon)


def select(c, state, controller, cfg, seed, tick, check_timeout=None):
    if controller not in CONTROLLERS:
        raise ValueError("unknown controller")
    actions = roots(c, state)
    allowed = eligible_indices(actions, controller == "assignment_lookahead")
    diagnostic = dict(planned=False, candidate_count=1, selection_seed=None,
        selected_mean=None, myopic_mean=None, forecast_transitions=0)
    if controller == "myopic_matching" or tick >= cfg["horizon"]:
        return actions[0], diagnostic
    pool = tuple(actions[i] for i in allowed)
    diagnostic["candidate_count"] = len(pool)
    if len(pool) == 1:
        return pool[0], diagnostic
    bank = forecast_seed(cfg, seed, tick)
    values = forecast_values(c, state, pool, cfg["lookahead_ticks"], cfg["scenarios"], bank, check_timeout)
    picked = minimum(values["means"], tuple(range(len(pool))))
    diagnostic.update(planned=True, selection_seed=bank,
        selected_mean=float(values["means"][picked]), myopic_mean=float(values["means"][0]),
        forecast_transitions=values["transitions"])
    return pool[picked], diagnostic


def compact(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def change_metrics(baseline, chosen):
    same_subset = tuple(m for m, _ in baseline) == tuple(m for m, _ in chosen)
    return dict(disagreement_rate=int(baseline != chosen), subset_change_rate=int(not same_subset),
        assignment_only_change_rate=int(baseline != chosen and same_subset),
        cardinality_change_rate=int(len(baseline) != len(chosen)),
        wait_change_rate=int(bool(baseline) != bool(chosen)))


def episode(cfg, controller, seed, journal=None, check_timeout=None):
    c = make_config(cfg)
    env = WaitingEnv(c); env.reset(physical_seed(seed))
    counts, interior, saved = defaultdict(float), defaultdict(float), {}
    cutoff, planning_seconds, spent = None, 0., 0
    for tick in range(c.horizon):
        if check_timeout is not None:
            check_timeout()
        state = env.state
        if controller == "myopic_matching" and tick in cfg["landmarks"]:
            saved[tick] = state
        before = time.perf_counter()
        action, diag = select(c, state, controller, cfg, seed, tick, check_timeout)
        planning_seconds += time.perf_counter() - before
        spent += diag["forecast_transitions"]
        validate_matching(c, state, action)
        metrics = None
        if tick < cfg["horizon"]:
            metrics = dict(**screen.state_metrics(c, state, action), **change_metrics(myopic(c, state), action))
            for k, v in metrics.items():
                counts[k] += v
        events = env.events[tick]
        _, _, _, info = env.step(action)
        if tick < cfg["horizon"]:
            for k, v in info.items():
                interior[k] += v
        if tick + 1 == cfg["horizon"]:
            cutoff = dict(terminal_pending=screen.base.pending(c, env.state), terminal_failed=sum(env.state.failed))
        if journal is not None:
            journal.add(dict(controller=controller, seed=seed, physical_seed=physical_seed(seed), tick=tick,
                state=compact(asdict(state)), action=compact(action), events=compact(events),
                metrics=compact(metrics), following=compact(asdict(env.state)), info=compact(info), diagnostic=compact(diag)))
    if cutoff is None:
        raise RuntimeError("missing H cutoff")
    row = dict(controller=controller, seed=seed, physical_seed=physical_seed(seed),
        **{k: v / cfg["horizon"] for k, v in counts.items()},
        availability=1 - interior["unavailable_ticks"] / (c.machines * cfg["horizon"]),
        technician_utilization=interior["service_ticks"] / (c.technicians * cfg["horizon"]),
        **cutoff, tail_pending=screen.base.pending(c, env.state), tail_failed=sum(env.state.failed),
        policy_cost=interior["objective"], tail_cost=env.metrics["objective"] - interior["objective"],
        extended_cost=env.metrics["objective"], policy_failures=interior["failures"],
        policy_overdue_ticks=interior["overdue_waiting_ticks"], planning_seconds=planning_seconds,
        forecast_transitions=spent, **{f"extended_{k}": env.metrics[k] for k in
            ("maintenance_cost", "unavailability_cost", "failure_cost", "waiting_cost", "invalid_assignments")})
    if not np.isclose(sum(row[f"extended_{k}"] for k in ("maintenance_cost", "unavailability_cost", "failure_cost", "waiting_cost")),
            row["extended_cost"], atol=1e-8, rtol=0):
        raise RuntimeError("component cost mismatch")
    return row, saved


def landmark(cfg, state, seed, tick, root_journal=None, check_timeout=None):
    c = make_config(cfg); actions = roots(c, state)
    selection_seed = forecast_seed(cfg, seed, tick)
    holdout_seed = forecast_seed(cfg, seed, tick, "holdout")
    if selection_seed == holdout_seed:
        raise RuntimeError("selection and holdout bank collision")
    selection = forecast_values(c, state, actions, cfg["lookahead_ticks"], cfg["scenarios"], selection_seed, check_timeout)
    heldout = forecast_values(c, state, actions, cfg["lookahead_ticks"], cfg["scenarios"], holdout_seed, check_timeout)
    row = dict(seed=seed, tick=tick, state=compact(asdict(state)), roots=len(actions),
        selection_seed=selection_seed, holdout_seed=holdout_seed,
        forecast_transitions=selection["transitions"] + heldout["transitions"])
    for name, restricted in (("joint", False), ("assignment", True)):
        chosen = minimum(selection["means"], eligible_indices(actions, restricted))
        delta = heldout["samples"][0] - heldout["samples"][chosen]
        row.update({f"{name}_root": chosen, f"{name}_action": compact(actions[chosen]),
            f"{name}_heldout_gain": float(delta.mean()), f"{name}_heldout_gain_se": float(delta.std(ddof=1) / np.sqrt(cfg["scenarios"]))})
        row.update({f"{name}_{k}": v for k, v in change_metrics(actions[0], actions[chosen]).items()})
    if root_journal is not None:
        for i, action in enumerate(actions):
            root_journal.add(dict(seed=seed, tick=tick, root=i, action=compact(action),
                same_myopic_subset=int(i in eligible_indices(actions, True)),
                selected_joint=int(i == row["joint_root"]), selected_assignment=int(i == row["assignment_root"]),
                selection_mean=float(selection["means"][i]), selection_mc_se=float(selection["mc_se"][i]),
                holdout_mean=float(heldout["means"][i]), holdout_mc_se=float(heldout["mc_se"][i]),
                selection_seed=selection_seed, holdout_seed=holdout_seed))
    return row


def summarize(cfg, rows, landmarks):
    expected = set(itertools.product(CONTROLLERS, cfg["evaluation_seeds"]))
    lookup = {(r["controller"], r["seed"]): r for r in rows}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise RuntimeError("incomplete or duplicate episode grid")
    grid = set(itertools.product(cfg["evaluation_seeds"], cfg["landmarks"]))
    local = {(r["seed"], r["tick"]): r for r in landmarks}
    if len(landmarks) != len(grid) or set(local) != grid:
        raise RuntimeError("incomplete or duplicate landmark grid")
    n = len(cfg["evaluation_seeds"])
    draws = np.random.default_rng(role_seed(cfg["analysis_seed"], f"{VERSION}:bootstrap")).integers(n, size=(cfg["bootstrap_samples"], n))
    fields = ("extended_cost", "availability", "terminal_pending")
    values = np.asarray([[[lookup[p, seed][k] for k in fields] for p in CONTROLLERS] for seed in cfg["evaluation_seeds"]])
    boot, point = values[draws].mean(1), values.mean(0)
    if (boot[:, :, 0] <= 0).any() or (point[:, 0] <= 0).any():
        raise RuntimeError("positive cost denominators required")
    alpha = (1 - cfg["confidence"]) / 2
    result = {}
    for key, improved, base in (("joint_vs_myopic", 1, 0), ("assignment_vs_myopic", 2, 0), ("joint_vs_assignment", 1, 2)):
        dist = 1 - boot[:, improved, 0] / boot[:, base, 0]
        result[key] = dict(estimate=float(1 - point[improved, 0] / point[base, 0]),
            ci=np.quantile(dist, [alpha, 1-alpha]).tolist(), confidence=cfg["confidence"])
    for key, column in (("availability_joint_minus_myopic", 1), ("pending_joint_minus_myopic", 2)):
        result[key] = dict(estimate=float(point[1, column] - point[0, column]),
            ci=np.quantile(boot[:, 1, column] - boot[:, 0, column], [alpha, 1-alpha]).tolist(), confidence=cfg["confidence"])
    gates = dict(practical_cost=result["joint_vs_myopic"]["ci"][0] >= cfg["practical_margin"],
        availability=result["availability_joint_minus_myopic"]["ci"][0] >= -cfg["availability_margin"],
        pending=result["pending_joint_minus_myopic"]["ci"][1] <= cfg["pending_margin"])
    means = []
    for controller in CONTROLLERS:
        group = [r for r in rows if r["controller"] == controller]
        means.append(dict(controller=controller, **{k: float(np.mean([r[k] for r in group])) for k in group[0] if k not in ("controller", "seed", "physical_seed")}))
    mechanisms = {}
    for name in ("joint", "assignment"):
        seed_values = np.asarray([np.mean([local[seed, tick][f"{name}_heldout_gain"] for tick in cfg["landmarks"]]) for seed in cfg["evaluation_seeds"]])
        dist = seed_values[draws].mean(1)
        mechanisms[name] = dict(mean_holdout_12_tick_gain_per_landmark=float(seed_values.mean()),
            exploratory_ci95=np.quantile(dist, [.025, .975]).tolist(),
            changed_roots=sum(r[f"{name}_disagreement_rate"] for r in landmarks),
            total_landmarks=len(landmarks), scope="independent forecast bank, myopic states, fixed landmarks; not episode savings")
    return dict(candidate=CANDIDATE, primary="joint_vs_myopic", comparisons=result, gates=gates,
        scientific_status=("JOINT_HEADROOM_CONFIRMED" if all(gates.values()) else "NOT_CONFIRMED") if cfg["profile"] == "full" else "ENGINEERING_ONLY",
        means=means, mechanism_diagnostics=mechanisms,
        interpretation="fixed exploratory setting; known-model rollout headroom, not RL learnability or v3 setting confirmation")


def close_enough(actual, expected):
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise RuntimeError("replay schema mismatch")
        for key in actual:
            close_enough(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise RuntimeError("replay sequence mismatch")
        for a, b in zip(actual, expected):
            close_enough(a, b)
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if not np.isclose(actual, expected, atol=1e-8, rtol=0):
            raise RuntimeError("replay numeric mismatch")
    elif actual != expected:
        raise RuntimeError("replay value mismatch")


def replay_audit(path, cfg, episodes, landmarks, root_rows, check_timeout=None):
    import csv

    c = make_config(cfg)
    envs, accumulated, cutoff, pairing, common_states = {}, {}, {}, {}, {}
    measured = defaultdict(lambda: defaultdict(float))
    spent_by_episode = defaultdict(int)
    regens = 0
    lookup = {(r["controller"], r["seed"]): r for r in episodes}
    seen = set()
    first = set(cfg["evaluation_seeds"][:cfg["audit_seeds"]])
    with path.open() as stream:
        for raw in csv.DictReader(stream):
            if check_timeout is not None:
                check_timeout()
            controller, seed, tick = raw["controller"], int(raw["seed"]), int(raw["tick"])
            key = (controller, seed)
            if (controller, seed, tick) in seen or key not in lookup or tick not in range(c.horizon):
                raise RuntimeError("invalid decision grid")
            seen.add((controller, seed, tick))
            if tick == 0:
                envs[key] = WaitingEnv(c); envs[key].reset(physical_seed(seed)); accumulated[key] = defaultdict(float)
            env = envs[key]
            state = screen.base.unpack_state(json.loads(raw["state"]))
            action = tuple(map(tuple, json.loads(raw["action"])))
            events, info, diag = json.loads(raw["events"]), json.loads(raw["info"]), json.loads(raw["diagnostic"])
            if env.time != tick or env.state != state or int(raw["physical_seed"]) != physical_seed(seed) or tuple(events) != env.events[tick]:
                raise RuntimeError("state/event/seed replay mismatch")
            pair = pairing.setdefault((seed, tick), (raw["events"], raw["state"] if tick == 0 else None))
            if pair != (raw["events"], raw["state"] if tick == 0 else None):
                raise RuntimeError("paired physical tape mismatch")
            if controller == "myopic_matching" and tick in cfg["landmarks"]:
                common_states[seed, tick] = raw["state"]
            candidates = roots(c, state)
            if controller == "assignment_lookahead":
                candidates = tuple(candidates[i] for i in eligible_indices(candidates, True))
            planning = controller != "myopic_matching" and tick < cfg["horizon"] and len(candidates) > 1
            expected_count = len(candidates) if controller != "myopic_matching" and tick < cfg["horizon"] else 1
            expected_spent = expected_count * cfg["scenarios"] * cfg["lookahead_ticks"] if planning else 0
            identity = dict(planned=planning, candidate_count=expected_count,
                selection_seed=forecast_seed(cfg, seed, tick) if planning else None,
                forecast_transitions=expected_spent)
            close_enough({k: diag[k] for k in identity}, identity)
            if planning and (not np.isfinite([diag['selected_mean'], diag['myopic_mean']]).all()
                    or diag['selected_mean'] > diag['myopic_mean'] + 1e-10):
                raise RuntimeError("invalid chosen forecast score")
            if not planning and (diag['selected_mean'] is not None or diag['myopic_mean'] is not None):
                raise RuntimeError("unplanned decision has forecast scores")
            spent_by_episode[key] += expected_spent
            if seed in first or controller == "myopic_matching" or tick >= cfg["horizon"]:
                expected_action, expected_diag = select(c, state, controller, cfg, seed, tick, check_timeout)
                if expected_action != action:
                    raise RuntimeError("replayed planner action mismatch")
                close_enough(diag, expected_diag); regens += expected_diag["forecast_transitions"]
            else:
                allowed = roots(c, state)
                if controller == "assignment_lookahead":
                    allowed = tuple(allowed[i] for i in eligible_indices(allowed, True))
                if action not in allowed:
                    raise RuntimeError("root restriction mismatch")
            metrics = dict(**screen.state_metrics(c, state, action), **change_metrics(myopic(c, state), action)) if tick < cfg["horizon"] else None
            close_enough(json.loads(raw["metrics"]), metrics)
            following, cost, expected_info = transition(c, state, action, events)
            if following != screen.base.unpack_state(json.loads(raw["following"])) or cost != info["objective"]:
                raise RuntimeError("following/cost mismatch")
            close_enough(info, expected_info)
            if tick < cfg["horizon"]:
                for k, v in metrics.items():
                    measured[key][k] += v
                for k, v in info.items():
                    accumulated[key][k] += v
            env.step(action)
            if tick + 1 == cfg["horizon"]:
                cutoff[key] = dict(terminal_pending=screen.base.pending(c, env.state), terminal_failed=sum(env.state.failed))
    if seen != set(itertools.product(CONTROLLERS, cfg["evaluation_seeds"], range(c.horizon))):
        raise RuntimeError("incomplete decision grid")
    for key, env in envs.items():
        row, interior = lookup[key], accumulated[key]
        for k, v in measured[key].items():
            close_enough(row[k], v / cfg["horizon"])
        expected = dict(**cutoff[key], extended_cost=env.metrics["objective"], policy_cost=interior["objective"],
            forecast_transitions=spent_by_episode[key],
            tail_cost=env.metrics["objective"] - interior["objective"],
            availability=1 - interior["unavailable_ticks"] / (c.machines * cfg["horizon"]),
            technician_utilization=interior["service_ticks"] / (c.technicians * cfg["horizon"]),
            policy_failures=interior["failures"], policy_overdue_ticks=interior["overdue_waiting_ticks"],
            tail_pending=screen.base.pending(c, env.state), tail_failed=sum(env.state.failed),
            **{f"extended_{k}": env.metrics[k] for k in ("maintenance_cost", "unavailability_cost", "failure_cost", "waiting_cost", "invalid_assignments")})
        close_enough({k: row[k] for k in expected}, expected)
    root_grid = {(int(r["seed"]), int(r["tick"]), int(r["root"])) for r in root_rows}
    expected_roots = {(r["seed"], r["tick"], i) for r in landmarks for i in range(r["roots"])}
    if len(root_rows) != len(expected_roots) or root_grid != expected_roots:
        raise RuntimeError("incomplete root grid")
    root_lookup = {(int(r["seed"]), int(r["tick"]), int(r["root"])): r for r in root_rows}
    audited_landmarks = 0
    for row in tqdm(landmarks, desc="Audit landmark forecasts", unit="state", leave=False):
        if row['state'] != common_states.get((row['seed'], row['tick'])):
            raise RuntimeError("landmark is not the recorded common myopic state")
        state = screen.base.unpack_state(json.loads(row["state"]))
        actions = roots(c, state)
        if row['roots'] != len(actions) or row['selection_seed'] != forecast_seed(cfg, row['seed'], row['tick']) or row['holdout_seed'] != forecast_seed(cfg, row['seed'], row['tick'], 'holdout'):
            raise RuntimeError("landmark identity mismatch")
        values = []
        for i, action in enumerate(actions):
            stored = root_lookup[row['seed'], row['tick'], i]
            identity = dict(action=compact(action), same_myopic_subset=int(i in eligible_indices(actions, True)),
                selected_joint=int(i == row['joint_root']), selected_assignment=int(i == row['assignment_root']),
                selection_seed=row['selection_seed'], holdout_seed=row['holdout_seed'])
            if any(str(stored[k]) != str(v) for k, v in identity.items()):
                raise RuntimeError("landmark root identity mismatch")
            numbers = [float(stored[k]) for k in ('selection_mean', 'holdout_mean', 'selection_mc_se', 'holdout_mc_se')]
            if not np.isfinite(numbers).all() or any(v < 0 for v in numbers):
                raise RuntimeError("invalid landmark cost/SE")
            values.append(numbers)
        for name, restricted in (('joint', False), ('assignment', True)):
            picked = minimum(np.asarray(values)[:, 0], eligible_indices(actions, restricted))
            if row[f'{name}_root'] != picked or row[f'{name}_action'] != compact(actions[picked]):
                raise RuntimeError("holdout improperly influenced root selection")
            close_enough(row[f'{name}_heldout_gain'], values[0][1] - values[picked][1])
            close_enough({f'{name}_{k}': row[f'{name}_{k}'] for k in change_metrics(actions[0], actions[picked])},
                {f'{name}_{k}': v for k,v in change_metrics(actions[0], actions[picked]).items()})
        if row["seed"] in first:
            captured = []
            class Capture:
                def add(self, r): captured.append(r)
            expected = landmark(cfg, state, row["seed"], row["tick"], Capture(), check_timeout)
            close_enough(row, expected)
            for r in captured:
                stored = root_lookup[r["seed"], r["tick"], r["root"]]
                close_enough(r, {k: stored[k] if isinstance(v, str) else type(v)(stored[k]) for k, v in r.items()})
            regens += expected["forecast_transitions"]; audited_landmarks += 1
    return dict(status="PASS", replayed_physical_transitions=len(seen), replayed_episodes=len(envs),
        paired_tapes=True, metrics_replayed=True, forecast_audited_seeds=len(first),
        audited_landmarks=audited_landmarks, root_rows=len(root_rows), replay_forecast_transitions=regens)


def run(profile, output_dir, device="cpu"):
    if device != "cpu":
        raise ValueError("CPU-only non-learning experiment")
    cfg = settings(profile); c = make_config(cfg); seeds = seed_audit(cfg)
    possible = [a for a in screen.base.catalogue(4, 2) if all(c.service_time[m][j] > 0 for m, j in a)]
    if len(possible) != 17:
        raise RuntimeError("fixed mask root catalogue changed")
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("refusing to overwrite a previous run")
    git = screen.git_metadata()
    if profile == "full" and git["dirty"]:
        raise RuntimeError("full run requires clean committed source")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def check_timeout():
        if time.monotonic() - started > cfg["max_seconds"]:
            raise TimeoutError("locked experiment timeout")
    manifest = dict(protocol=VERSION, candidate=CANDIDATE, environment_version=ENV_VERSION,
        status="RUNNING", profile=profile, git=git, started_utc=datetime.now(timezone.utc).isoformat(),
        full_evaluation_outcomes_opened=False, runtime=dict(python=platform.python_version(),
            numpy=np.__version__, platform=platform.platform(), device=device))
    write_json(output / "manifest.json", manifest)
    try:
        write_json(output / "resolved_config.json", dict(settings=cfg, config=asdict(c)))
        shutil.copy2(PLAN_PATH, output / "protocol.md"); shutil.copy2(SEED_PATH, output / "seed_registry.json")
        snapshot = output / "source_snapshot"; snapshot.mkdir()
        for name in SOURCES:
            shutil.copy2(Path(__file__).parent / name, snapshot / name)
        for module in (screen.base, screen.previous, screen):
            shutil.copy2(module.SEED_PATH, snapshot / module.SEED_PATH.name)
        seals = {str(p.relative_to(output)): screen.sha(p) for p in [output / "protocol.md", output / "seed_registry.json", *snapshot.iterdir()]}
        manifest.update(source_seal=seals, full_evaluation_outcomes_opened=profile == "full")
        write_json(output / "manifest.json", manifest)
        rows, saved = [], {}
        journal, partial = Journal(output / "decisions.csv"), Journal(output / "episodes.partial.csv")
        try:
            for seed, controller in tqdm(list(itertools.product(cfg["evaluation_seeds"], CONTROLLERS)), desc="Evaluate paired episodes", unit="episode"):
                check_timeout()
                row, states = episode(cfg, controller, seed, journal, check_timeout)
                rows.append(row); partial.add(row)
                if controller == "myopic_matching":
                    saved.update({(seed, tick): state for tick, state in states.items()})
        finally:
            journal.close(); partial.close()
        write_csv(output / "episodes.csv", rows)
        landmarks, root_journal, local_journal = [], Journal(output / "landmark_roots.csv"), Journal(output / "landmarks.csv")
        try:
            for (seed, tick), state in tqdm(saved.items(), desc="Independent-bank common-state diagnostics", unit="state"):
                row = landmark(cfg, state, seed, tick, root_journal, check_timeout)
                landmarks.append(row); local_journal.add(row)
        finally:
            root_journal.close(); local_journal.close()
        import csv
        with (output / "landmark_roots.csv").open() as stream:
            root_rows = list(csv.DictReader(stream))
        audit = replay_audit(output / "decisions.csv", cfg, rows, landmarks, root_rows, check_timeout)
        spent = sum(r["forecast_transitions"] for r in rows + landmarks)
        cap = (2*len(cfg["evaluation_seeds"])*cfg["horizon"] + 2*len(cfg["evaluation_seeds"])*len(cfg["landmarks"])) * 21 * cfg["scenarios"] * cfg["lookahead_ticks"]
        if spent > cap or any(r["extended_invalid_assignments"] for r in rows):
            raise RuntimeError("forecast cap or feasibility failure")
        for name, digest in seals.items():
            if screen.sha(output / name) != digest:
                raise RuntimeError("source/protocol seal changed")
        for name in SOURCES:
            if screen.sha(Path(__file__).parent / name) != seals[f"source_snapshot/{name}"]:
                raise RuntimeError("live source changed")
        for original, copied in ((PLAN_PATH, "protocol.md"), (SEED_PATH, "seed_registry.json"),
                *((module.SEED_PATH, f"source_snapshot/{module.SEED_PATH.name}") for module in (screen.base, screen.previous, screen))):
            if screen.sha(original) != seals[copied]:
                raise RuntimeError("live protocol/registry changed")
        expected = dict(episodes=len(CONTROLLERS)*len(cfg["evaluation_seeds"]), physical_transitions=len(CONTROLLERS)*len(cfg["evaluation_seeds"])*60,
            landmarks=len(cfg["evaluation_seeds"])*len(cfg["landmarks"]))
        actual = dict(episodes=len(rows), physical_transitions=audit["replayed_physical_transitions"], landmarks=len(landmarks))
        if actual != expected:
            raise RuntimeError("incomplete counts")
        audit.update(seed_audit=seeds, actual_counts=actual, expected_counts=expected, forecast_transitions=spent,
            forecast_cap=cap, invalid_assignments=0, source_seal_unchanged=True)
        write_json(output / "audit.json", audit)
        summary = summarize(cfg, rows, landmarks)
        write_json(output / "summary.json", summary)
        check_timeout()
        files = {str(p.relative_to(output)): dict(sha256=screen.sha(p), bytes=p.stat().st_size)
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        manifest.update(status="COMPLETED", audit="PASS", elapsed_seconds=time.monotonic()-started,
            expected_counts=expected, actual_counts=actual, forecast_transitions=spent, forecast_cap=cap,
            scientific_status=summary["scientific_status"], files=files, artifact_bytes=sum(r["bytes"] for r in files.values()))
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="FAILED", elapsed_seconds=time.monotonic()-started, error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.profile, args.output_dir, args.device)
    print(json.dumps({k: result[k] for k in ("status", "audit", "actual_counts", "scientific_status", "elapsed_seconds", "artifact_bytes")}, indent=2))


if __name__ == "__main__":
    main()
