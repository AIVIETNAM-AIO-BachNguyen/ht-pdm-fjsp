"""Boundary follow-up: nested horizons and baseline continuation, fixed exploratory setting."""

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
from ht_pdm_fjsp import maintenance_myopic_lookahead as previous
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays, batch_tick, reference_batch
from ht_pdm_fjsp.maintenance_opportunity_cost import Journal, write_csv, write_json
from ht_pdm_fjsp.maintenance_waiting import ENV_VERSION, WaitingEnv, transition

VERSION = "maintenance_lookahead_boundary_v1"
ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = ROOT / "configs" / f"{VERSION}_seed_registry.json"
PLAN_PATH = ROOT / "docs" / f"{VERSION}_plan.md"
CANDIDATE = "overlap3__fast1_slow325"
PLANNERS = ("joint_h12", "joint_h24", "joint_endpoint", "assignment_h12", "assignment_h24", "assignment_endpoint")
CONTROLLERS = ("myopic_matching", *PLANNERS)
LANDMARKS = (0, 8, 16, 24, 32, 40, 47)
SOURCES = ("maintenance_lookahead_boundary.py", "maintenance_myopic_lookahead.py", "maintenance_skill_contention_screen_v3.py",
    "maintenance_skill_contention_screen_v2.py", "maintenance_skill_contention_screen.py",
    "maintenance_dispatch.py", "maintenance_waiting.py", "maintenance_heterogeneous_model.py",
    "maintenance_opportunity_cost.py")


def settings(profile):
    if profile not in ("smoke", "full"):
        raise ValueError("unknown profile")
    raw = json.loads(SEED_PATH.read_text())["panels"][profile]
    return dict(profile=profile, horizon=48, tail_ticks=12,
        scenarios=128 if profile == "full" else 8,
        evaluation_seeds=list(range(raw["evaluation_start"], raw["evaluation_start"] + raw["evaluation_count"])),
        analysis_seed=raw["analysis_seed"], forecast_seed=raw["forecast_seed"],
        bootstrap_samples=20000 if profile == "full" else 200,
        confidence=1-.05/12, practical_margin=.01, availability_margin=.01, pending_margin=.10,
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
    for profile in ("full", "smoke"):
        old=previous.settings(profile)
        prior.update(previous.physical_seed(seed) for seed in old["evaluation_seeds"])
        prior.add(role_seed(old["analysis_seed"],f"{previous.VERSION}:bootstrap"))
        prior.update(previous.forecast_seed(old,seed,tick) for seed,tick in itertools.product(old["evaluation_seeds"],range(old["horizon"])))
        prior.update(previous.forecast_seed(old,seed,tick,"holdout") for seed,tick in itertools.product(old["evaluation_seeds"],old["landmarks"]))
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


def checkpoint_values(c, state, actions, checkpoints, scenarios, seed, check_timeout=None):
    """One CRN tape, myopic continuation and nested prefix sums; no real future."""
    checkpoints = tuple(sorted(set(checkpoints)))
    if not checkpoints or checkpoints[0] < 1 or scenarios < 2 or not actions:
        raise ValueError("positive checkpoints and at least two scenarios required")
    for action in actions:
        validate_matching(c, state, action)
    b = len(actions); horizon = checkpoints[-1]
    states = arrays(state, b * scenarios)
    start = np.zeros((b, c.machines, c.technicians), dtype=bool)
    for i, action in enumerate(actions):
        for m, j in action:
            start[i, m, j] = True
    chosen = np.repeat(start, scenarios, axis=0)
    events = np.random.default_rng(seed).random((horizon, scenarios, c.machines)) < c.failure_probability
    total = np.zeros(b * scenarios); values = {}
    for tick in range(horizon):
        if check_timeout is not None:
            check_timeout()
        if tick:
            chosen = reference_batch(c, states, horizon - tick, "deadline_matching")
        states, costs = batch_tick(c, states, chosen, np.tile(events[tick], (b, 1)))
        total += costs
        if tick + 1 in checkpoints:
            samples = total.reshape(b, scenarios).copy()
            if not np.isfinite(samples).all():
                raise RuntimeError("non-finite forecast cost")
            values[tick + 1] = dict(samples=samples, means=samples.mean(1),
                mc_se=samples.std(1, ddof=1) / np.sqrt(scenarios))
    return dict(values=values, transitions=b * scenarios * horizon)


def score_ticks(cfg, controller, tick):
    if controller not in PLANNERS or tick not in range(cfg["horizon"]):
        raise ValueError("invalid planning controller/tick")
    remaining = cfg["horizon"] + cfg["tail_ticks"] - tick
    return 12 if controller.endswith("h12") else min(24, remaining) if controller.endswith("h24") else remaining


def forecast_cap(cfg):
    episode_ticks = 2 * sum(12 + min(24, 60-t) + 60-t for t in range(cfg["horizon"]))
    landmark_ticks = 2 * sum(60-t for t in cfg["landmarks"])
    return len(cfg["evaluation_seeds"]) * 21 * cfg["scenarios"] * (episode_ticks + landmark_ticks)


def select(c, state, controller, cfg, seed, tick, check_timeout=None):
    if controller not in CONTROLLERS or tick not in range(c.horizon):
        raise ValueError("unknown controller or invalid tick")
    actions = roots(c, state)
    diagnostic = dict(planned=False, candidate_count=1, selection_seed=None,
        score_ticks=None, selected_mean=None, myopic_mean=None,
        selected_prefix12=None, myopic_prefix12=None,
        selected_terminal=None, myopic_terminal=None, forecast_transitions=0)
    if controller == "myopic_matching" or tick >= cfg["horizon"]:
        return actions[0], diagnostic
    pool = tuple(actions[i] for i in eligible_indices(actions, controller.startswith("assignment_")))
    diagnostic["candidate_count"] = len(pool)
    if len(pool) == 1:
        return pool[0], diagnostic
    horizon = score_ticks(cfg, controller, tick)
    bank = forecast_seed(cfg, seed, tick)
    forecast = checkpoint_values(c, state, pool, (12, horizon), cfg["scenarios"], bank, check_timeout)
    values = forecast["values"][horizon]
    picked = minimum(values["means"], tuple(range(len(pool))))
    prefix = forecast["values"][12]["means"]
    # For h24 the residual is an in-window extension, not an extra value head.
    diagnostic.update(planned=True, selection_seed=bank, score_ticks=horizon,
        selected_mean=float(values["means"][picked]), myopic_mean=float(values["means"][0]),
        selected_prefix12=float(prefix[picked]), myopic_prefix12=float(prefix[0]),
        selected_terminal=float(values["means"][picked]-prefix[picked]),
        myopic_terminal=float(values["means"][0]-prefix[0]), forecast_transitions=forecast["transitions"])
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
    remaining = c.horizon - tick
    checkpoints = (12, min(24, remaining), remaining)
    selection = checkpoint_values(c, state, actions, checkpoints, cfg["scenarios"], selection_seed, check_timeout)
    heldout = checkpoint_values(c, state, actions, checkpoints, cfg["scenarios"], holdout_seed, check_timeout)
    row = dict(seed=seed, tick=tick, state=compact(asdict(state)), roots=len(actions),
        selection_seed=selection_seed, holdout_seed=holdout_seed,
        forecast_transitions=selection["transitions"] + heldout["transitions"])
    for name in PLANNERS:
        horizon = score_ticks(cfg, name, tick)
        chosen = minimum(selection["values"][horizon]["means"], eligible_indices(actions, name.startswith("assignment_")))
        own = heldout["values"][horizon]["samples"]
        endpoint = heldout["values"][remaining]["samples"]
        delta, endpoint_delta = own[0]-own[chosen], endpoint[0]-endpoint[chosen]
        row.update({f"{name}_root": chosen, f"{name}_action": compact(actions[chosen]),
            f"{name}_heldout_gain": float(delta.mean()),
            f"{name}_heldout_gain_se": float(delta.std(ddof=1)/np.sqrt(cfg["scenarios"])),
            f"{name}_endpoint_gain": float(endpoint_delta.mean()),
            f"{name}_endpoint_gain_se": float(endpoint_delta.std(ddof=1)/np.sqrt(cfg["scenarios"]))})
        row.update({f"{name}_{k}": v for k,v in change_metrics(actions[0],actions[chosen]).items()})
    if root_journal is not None:
        for i, action in enumerate(actions):
            record = dict(seed=seed,tick=tick,root=i,action=compact(action),
                same_myopic_subset=int(i in eligible_indices(actions,True)),
                selection_seed=selection_seed,holdout_seed=holdout_seed)
            record.update({f"selected_{name}":int(i==row[f"{name}_root"]) for name in PLANNERS})
            for label, horizon in zip(("h12","h24","endpoint"),checkpoints):
                for phase, bank in (("selection",selection),("holdout",heldout)):
                    values = bank["values"][horizon]
                    record[f"{phase}_{label}_mean"] = float(values["means"][i])
                    record[f"{phase}_{label}_mc_se"] = float(values["mc_se"][i])
            root_journal.add(record)
    return row


def summarize(cfg, rows, landmarks):
    expected = set(itertools.product(CONTROLLERS,cfg["evaluation_seeds"]))
    lookup = {(r["controller"],r["seed"]):r for r in rows}
    if len(rows)!=len(expected) or set(lookup)!=expected:
        raise RuntimeError("incomplete or duplicate episode grid")
    grid = set(itertools.product(cfg["evaluation_seeds"],cfg["landmarks"]))
    local = {(r["seed"],r["tick"]):r for r in landmarks}
    if len(landmarks)!=len(grid) or set(local)!=grid:
        raise RuntimeError("incomplete or duplicate landmark grid")
    n=len(cfg["evaluation_seeds"])
    draws=np.random.default_rng(role_seed(cfg["analysis_seed"],f"{VERSION}:bootstrap")).integers(n,size=(cfg["bootstrap_samples"],n))
    fields=("extended_cost","availability","terminal_pending")
    values=np.asarray([[[lookup[p,seed][k] for k in fields] for p in CONTROLLERS] for seed in cfg["evaluation_seeds"]])
    boot,point=values[draws].mean(1),values.mean(0)
    if (boot[:,:,0]<=0).any() or (point[:,0]<=0).any():
        raise RuntimeError("positive cost denominators required")
    alpha=(1-cfg["confidence"])/2; result={}
    pairs=[(f"{name}_vs_myopic",name,"myopic_matching") for name in PLANNERS]
    pairs += [(f"{scope}_endpoint_vs_h12",f"{scope}_endpoint",f"{scope}_h12") for scope in ("joint","assignment")]
    for key, improved, base in pairs:
        i,b=CONTROLLERS.index(improved),CONTROLLERS.index(base)
        dist=1-boot[:,i,0]/boot[:,b,0]
        result[key]=dict(estimate=float(1-point[i,0]/point[b,0]),ci=np.quantile(dist,[alpha,1-alpha]).tolist(),confidence=cfg["confidence"])
    gates={}
    for scope in ("joint","assignment"):
        i=CONTROLLERS.index(f"{scope}_endpoint")
        for label,column in (("availability",1),("pending",2)):
            key=f"{label}_{scope}_endpoint_minus_myopic"
            result[key]=dict(estimate=float(point[i,column]-point[0,column]),
                ci=np.quantile(boot[:,i,column]-boot[:,0,column],[alpha,1-alpha]).tolist(),confidence=cfg["confidence"])
        gates[scope]=dict(practical_cost=result[f"{scope}_endpoint_vs_myopic"]["ci"][0]>=cfg["practical_margin"],
            availability=result[f"availability_{scope}_endpoint_minus_myopic"]["ci"][0]>=-cfg["availability_margin"],
            pending=result[f"pending_{scope}_endpoint_minus_myopic"]["ci"][1]<=cfg["pending_margin"])
    if len(result)!=12:
        raise RuntimeError("locked twelve-comparison family changed")
    means=[]
    for controller in CONTROLLERS:
        group=[r for r in rows if r["controller"]==controller]
        means.append(dict(controller=controller,**{k:float(np.mean([r[k] for r in group])) for k in group[0] if k not in ("controller","seed","physical_seed")}))
    mechanisms={}
    for name in PLANNERS:
        seed_values=np.asarray([np.mean([local[seed,tick][f"{name}_endpoint_gain"] for tick in cfg["landmarks"]]) for seed in cfg["evaluation_seeds"]])
        mechanisms[name]=dict(mean_holdout_endpoint_gain_per_landmark=float(seed_values.mean()),
            exploratory_ci95=np.quantile(seed_values[draws].mean(1),[.025,.975]).tolist(),
            changed_roots=sum(r[f"{name}_disagreement_rate"] for r in landmarks),total_landmarks=len(landmarks),
            scope="independent bank, common myopic states; root continuation value, not episode savings")
    return dict(candidate=CANDIDATE,primary="assignment_endpoint_vs_myopic",comparisons=result,gates=gates,
        scientific_status=("ASSIGNMENT_HEADROOM_CONFIRMED" if all(gates["assignment"].values()) else "NOT_CONFIRMED") if cfg["profile"]=="full" else "ENGINEERING_ONLY",
        secondary_joint_headroom=all(gates["joint"].values()),
        secondary_assignment_boundary_repair=result["assignment_endpoint_vs_h12"]["ci"][0]>0,
        means=means,mechanism_diagnostics=mechanisms,
        interpretation="known-model remaining-horizon myopic rollout; exploratory setting, not RL learnability")


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
            if controller.startswith("assignment_"):
                candidates = tuple(candidates[i] for i in eligible_indices(candidates, True))
            planning = controller != "myopic_matching" and tick < cfg["horizon"] and len(candidates) > 1
            expected_count = len(candidates) if controller != "myopic_matching" and tick < cfg["horizon"] else 1
            expected_spent = expected_count * cfg["scenarios"] * score_ticks(cfg, controller, tick) if planning else 0
            identity = dict(planned=planning, candidate_count=expected_count,
                selection_seed=forecast_seed(cfg, seed, tick) if planning else None,
                score_ticks=score_ticks(cfg, controller, tick) if planning else None, forecast_transitions=expected_spent)
            close_enough({k: diag[k] for k in identity}, identity)
            if planning and (not np.isfinite([diag['selected_mean'], diag['myopic_mean']]).all()
                    or diag['selected_mean'] > diag['myopic_mean'] + 1e-10):
                raise RuntimeError("invalid chosen forecast score")
            if planning:
                for prefix in ("selected", "myopic"):
                    close_enough(diag[f"{prefix}_mean"], diag[f"{prefix}_prefix12"] + diag[f"{prefix}_terminal"])
                    if min(diag[f"{prefix}_prefix12"], diag[f"{prefix}_terminal"]) < 0:
                        raise RuntimeError("negative forecast component")
            if not planning and any(diag[k] is not None for k in ("selected_mean", "myopic_mean", "selected_prefix12", "myopic_prefix12", "selected_terminal", "myopic_terminal")):
                raise RuntimeError("unplanned decision has forecast scores")
            spent_by_episode[key] += expected_spent
            if seed in first or controller == "myopic_matching" or tick >= cfg["horizon"]:
                expected_action, expected_diag = select(c, state, controller, cfg, seed, tick, check_timeout)
                if expected_action != action:
                    raise RuntimeError("replayed planner action mismatch")
                close_enough(diag, expected_diag); regens += expected_diag["forecast_transitions"]
            else:
                allowed = roots(c, state)
                if controller.startswith("assignment_"):
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
    root_grid={(int(r["seed"]),int(r["tick"]),int(r["root"])) for r in root_rows}
    expected_roots={(r["seed"],r["tick"],i) for r in landmarks for i in range(r["roots"])}
    if len(root_rows)!=len(expected_roots) or root_grid!=expected_roots:
        raise RuntimeError("incomplete root grid")
    root_lookup={(int(r["seed"]),int(r["tick"]),int(r["root"])):r for r in root_rows}
    audited_landmarks=0
    for row in tqdm(landmarks,desc="Audit landmark forecasts",unit="state",leave=False):
        if row['state']!=common_states.get((row['seed'],row['tick'])):
            raise RuntimeError("landmark is not the recorded common myopic state")
        state=screen.base.unpack_state(json.loads(row["state"])); actions=roots(c,state)
        if row['roots']!=len(actions) or row['selection_seed']!=forecast_seed(cfg,row['seed'],row['tick']) or row['holdout_seed']!=forecast_seed(cfg,row['seed'],row['tick'],'holdout'):
            raise RuntimeError("landmark identity mismatch")
        stored_rows=[root_lookup[row['seed'],row['tick'],i] for i in range(len(actions))]
        for i,(action,stored) in enumerate(zip(actions,stored_rows)):
            identity=dict(action=compact(action),same_myopic_subset=int(i in eligible_indices(actions,True)),
                selection_seed=row['selection_seed'],holdout_seed=row['holdout_seed'])
            identity.update({f"selected_{name}":int(i==row[f"{name}_root"]) for name in PLANNERS})
            if any(str(stored[k])!=str(v) for k,v in identity.items()):
                raise RuntimeError("landmark root identity mismatch")
            numbers=[float(stored[f"{phase}_{label}_{field}"]) for phase in ('selection','holdout') for label in ('h12','h24','endpoint') for field in ('mean','mc_se')]
            if not np.isfinite(numbers).all() or any(v<0 for v in numbers):
                raise RuntimeError("invalid landmark cost/SE")
            for phase in ('selection','holdout'):
                if not float(stored[f'{phase}_h12_mean'])<=float(stored[f'{phase}_h24_mean'])<=float(stored[f'{phase}_endpoint_mean']):
                    raise RuntimeError("nonmonotone nested forecast costs")
        for name in PLANNERS:
            label=name.split('_',1)[1]
            means=np.asarray([float(r[f'selection_{label}_mean']) for r in stored_rows])
            picked=minimum(means,eligible_indices(actions,name.startswith('assignment_')))
            if row[f'{name}_root']!=picked or row[f'{name}_action']!=compact(actions[picked]):
                raise RuntimeError("holdout improperly influenced root selection")
            for field,key in (('heldout_gain',label),('endpoint_gain','endpoint')):
                close_enough(row[f'{name}_{field}'],float(stored_rows[0][f'holdout_{key}_mean'])-float(stored_rows[picked][f'holdout_{key}_mean']))
            changes=change_metrics(actions[0],actions[picked])
            close_enough({k:row[f'{name}_{k}'] for k in changes},changes)
        if row["seed"] in first:
            captured=[]
            class Capture:
                def add(self,r): captured.append(r)
            expected=landmark(cfg,state,row['seed'],row['tick'],Capture(),check_timeout)
            close_enough(row,expected)
            for r in captured:
                stored=root_lookup[r['seed'],r['tick'],r['root']]
                close_enough(r,{k:stored[k] if isinstance(v,str) else type(v)(stored[k]) for k,v in r.items()})
            regens+=expected['forecast_transitions']; audited_landmarks+=1
    return dict(status="PASS",replayed_physical_transitions=len(seen),replayed_episodes=len(envs),
        paired_tapes=True,metrics_replayed=True,forecast_audited_seeds=len(first),
        audited_landmarks=audited_landmarks,root_rows=len(root_rows),replay_forecast_transitions=regens)


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
        for module in (screen.base, screen.previous, screen, previous):
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
        cap = forecast_cap(cfg)
        if spent > cap or any(r["extended_invalid_assignments"] for r in rows):
            raise RuntimeError("forecast cap or feasibility failure")
        for name, digest in seals.items():
            if screen.sha(output / name) != digest:
                raise RuntimeError("source/protocol seal changed")
        for name in SOURCES:
            if screen.sha(Path(__file__).parent / name) != seals[f"source_snapshot/{name}"]:
                raise RuntimeError("live source changed")
        for original, copied in ((PLAN_PATH, "protocol.md"), (SEED_PATH, "seed_registry.json"),
                *((module.SEED_PATH, f"source_snapshot/{module.SEED_PATH.name}") for module in (screen.base, screen.previous, screen, previous))):
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
