"""Prospective coverage/gating factorial; metadata construction runs no outcomes."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random

from ht_pdm_fjsp import maintenance_assignment_value_benchmark as previous
from ht_pdm_fjsp import maintenance_boundary_sensitivity as boundary
from ht_pdm_fjsp import maintenance_joint_rl as historical
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed, validate_state
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig

VERSION = "maintenance_assignment_confidence_v1"
ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("dense", "skill_mask")
ARMS = ("natural", "broad")
NAMES = ("frozen_ppo", "natural_ungated", "natural_gated", "broad_ungated", "broad_gated")


def settings(profile):
    if profile not in ("full", "smoke"):
        raise ValueError("unknown profile")
    registry = json.loads((ROOT / f"configs/{VERSION}_seed_registry.json").read_text())
    if registry["protocol"] != VERSION:
        raise ValueError("seed protocol")
    full = profile == "full"
    return dict(profile=profile, **registry["panels"][profile],
        model_seeds=[18100000, 18100001, 18100002] if full else [18100000],
        conditions=list(CONDITIONS), horizon=48, tail_ticks=12, hidden=32, reward_scale=20.,
        landmarks=[0, 12, 24, 36], probe_landmarks=[6, 18, 30, 42], ensemble_members=3,
        train_scenarios=16 if full else 2, development_scenarios=32 if full else 2,
        calibration_scenarios=32 if full else 2, selection_scenarios=32 if full else 2,
        evaluation_scenarios=64 if full else 2, epochs=80 if full else 2,
        checkpoints=[20, 40, 60, 80] if full else [1, 2], draws_per_epoch=1024 if full else 64,
        minibatch=128 if full else 32, learning_rate=.001, gradient_clip=1.,
        gate_quantile=.9, gate_sd_weight=1., gate_mc_weight=1.96,
        calibration_minimum=10 if full else 1, tie_tolerance=1e-5,
        diagnostic_profiles=8 if full else 1, evaluation_batch=32 if full else 2,
        bootstrap_samples=20000 if full else 200, primary_confidence=.9875,
        cost_margin=.01, torch_threads=1, audit_profiles=4 if full else 1, audit_shocks=1)


def roles(seed):
    rng = random.Random(role_seed(seed, f"{VERSION}:roles"))
    machines, techs = list(range(4)), list(range(2))
    rng.shuffle(machines); rng.shuffle(techs)
    return machines, techs


def make_config(seed, condition, attempt=0):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    rng = random.Random(role_seed(seed, f"{VERSION}:configuration:{attempt}"))
    duration = [[rng.randint(2, 6) for _ in range(2)] for _ in range(4)]
    effect = [[rng.choice((.25, .5, .75, 1.)) for _ in range(2)] for _ in range(4)]
    machines, techs = roles(seed)
    if condition == "skill_mask":
        duration[machines[1]][techs[1]] = 0
        duration[machines[3]][techs[0]] = 0
    c = WaitingConfig(4, 2, 60, rng.choice((3, 4, 5)), rng.choice((.25, .45, .6)),
        tuple(map(tuple, duration)), tuple(map(tuple, effect)), waiting_price=12.)
    c.validate()
    return c


def initial_state(c, seed):
    machines, _ = roles(seed)
    rng = random.Random(role_seed(seed, f"{VERSION}:initial_scenario"))
    case = rng.randrange(4)
    ages = [rng.randrange(c.failure_age) for _ in range(4)]
    ages[machines[0]] = c.failure_age - 1 if case & 1 else 0
    ages[machines[1]] = c.failure_age - 1 if case & 2 else 0
    s = DispatchState(tuple(ages), (False,) * 4, (0,) * 4, (0,) * 2, (-1,) * 2)
    validate_state(c, s)
    return s


def scenario(c, seed):
    s = initial_state(c, seed); machines, techs = roles(seed)
    return dict(focal_machine=machines[0], competing_machine=machines[1],
        scarce_technician=techs[0], focal_at_risk=s.ages[machines[0]] + 1 >= c.failure_age,
        competitor_at_risk=s.ages[machines[1]] + 1 >= c.failure_age, initial_state=asdict(s))


def physical_seed(cohort, shock):
    return role_seed(shock, f"{VERSION}:physical:{cohort}")


def probe_seed(identity):
    # Pair probe anchors and prefix shocks across source replicas.
    fields = ("condition", "cohort", "shock", "tick")
    return role_seed(identity["cohort"], f"{VERSION}:probe:" + ":".join(str(identity[k]) for k in fields))


def future_seed(cfg, identity, phase):
    key = f"{phase}_future_seed" if phase in ("train", "development", "calibration") else f"{phase}_seed"
    fields = ("condition", "cohort", "shock", "model_seed", "tick", "origin")
    return role_seed(cfg[key], f"{VERSION}:{phase}:" + ":".join(str(identity[k]) for k in fields))


@lru_cache(maxsize=1)
def metadata():
    blocked = set(previous.previous.historical_pool())
    for package in (previous.previous, previous):
        panels, _, *rest = package.metadata()
        for p in panels.values():
            blocked.update(boundary.signature(c) for c in p.values())
        if package is previous.previous:
            for p in rest[0].values():
                blocked.update(boundary.signature(r[0]) for r in p.values())
    excluded = len(blocked); panels = {}; records = {}
    for mode in ("full", "smoke"):
        cfg = settings(mode); panels[mode] = {}; records[mode] = []
        for split in ("train", "development", "calibration", "test"):
            for cohort in cfg[f"{split}_cohorts"]:
                for attempt in range(10000):
                    generated = {cond: make_config(cohort, cond, attempt) for cond in CONDITIONS}
                    sigs = {boundary.signature(c) for c in generated.values()}
                    if len(sigs) == 2 and not sigs & blocked:
                        break
                else:
                    raise RuntimeError("fresh configuration pool exhausted")
                blocked.update(sigs)
                for cond, c in generated.items():
                    panels[mode][split, cohort, cond] = c
                    records[mode].append(dict(split=split, cohort=cohort, condition=cond,
                        generator_attempt=attempt, physical_signature=boundary.signature(c), config=asdict(c)))
    return panels, records, dict(status="PASS", excluded_signatures=excluded, outcomes_used=False,
        full_smoke_train_dev_calibration_test_disjoint=True, fresh_signatures=sum(map(len, records.values())))


def seed_audit(cfg):
    own = ROOT / f"configs/{VERSION}_seed_registry.json"
    raw_old = historical.integers(historical.exclusions()["historical_raw_panels"])
    hashes = {}
    for path in sorted((ROOT / "configs").glob("*seed_registry.json")):
        if path != own:
            raw_old |= historical.integers(json.loads(path.read_text()))
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = set(); derived = []; _, records, _ = metadata()
    for mode in ("full", "smoke"):
        c = settings(mode)
        for value in json.loads(own.read_text())["panels"][mode].values():
            group = historical.integers(value)
            if group & (raw | raw_old):
                raise ValueError("raw seed collision")
            raw |= group
        configs = {(r["cohort"], r["generator_attempt"]) for r in records[mode]}
        derived += [role_seed(cohort, f"{VERSION}:configuration:{attempt}") for cohort, attempt in sorted(configs)]
        for seed in c["train_seeds"]:
            for member in range(c["ensemble_members"]):
                derived.append(role_seed(seed, f"{VERSION}:initialization:{member}"))
                derived += [role_seed(seed, f"{VERSION}:draws:{member}:{epoch}") for epoch in range(1, c["epochs"] + 1)]
        for split in ("train", "development", "calibration", "test"):
            cohorts = c[f"{split}_cohorts"]
            for cohort in cohorts:
                derived += [role_seed(cohort, f"{VERSION}:{r}") for r in ("roles", "initial_scenario")]
                derived += [physical_seed(cohort, shock) for shock in c[f"{split}_shocks"]]
            used = cohorts if split != "test" else cohorts[:c["diagnostic_profiles"]]
            shocks = c[f"{split}_shocks"] if split != "test" else c["test_shocks"][:1]
            for cond, cohort, shock, seed in itertools.product(CONDITIONS, used, shocks, c["model_seeds"]):
                for origin, ticks in (("natural", c["landmarks"]), ("probe", c["probe_landmarks"])):
                    for tick in ticks:
                        identity = dict(condition=cond, cohort=cohort, shock=shock, model_seed=seed, tick=tick, origin=origin)
                        if origin == "probe" and seed == c["model_seeds"][0]:
                            derived.append(probe_seed(identity))
                        phases = [split] if split != "test" else ["ranking_selection", "ranking_evaluation"]
                        derived += [future_seed(c, identity, p) for p in phases]
        derived.append(role_seed(c["analysis_seed"], f"{VERSION}:crossed_bootstrap"))
    old_roles = previous.historical_roles()
    # Include previous assignment-value configuration, optimization and forecast roles.
    _, old_records, _ = previous.metadata()
    for mode in ("full", "smoke"):
        c = previous.settings(mode)
        for r in old_records[mode]:
            old_roles.add(role_seed(r["cohort"], f"{previous.VERSION}:configuration:{r['generator_attempt']}"))
        for seed in c["train_seeds"]:
            old_roles.update(role_seed(seed, f"{previous.VERSION}:{r}") for r in ("initialization", "minibatches", "shuffle"))
        for split in ("train", "development", "test"):
            for cohort in c[f"{split}_cohorts"]:
                old_roles.update(role_seed(cohort, f"{previous.VERSION}:{r}") for r in ("roles", "initial_scenario"))
                old_roles.update(previous.physical_seed(cohort, shock) for shock in c[f"{split}_shocks"])
            ticks = c["landmarks"] if split != "test" else sorted(set(c["ranking_landmarks"] + c["timing_landmarks"]))
            shocks = c[f"{split}_shocks"] if split != "test" else c["test_shocks"][:1]
            for cond, cohort, shock, seed, tick in itertools.product(CONDITIONS, c[f"{split}_cohorts"], shocks, c["model_seeds"], ticks):
                identity = dict(condition=cond, cohort=cohort, shock=shock, model_seed=seed, tick=tick)
                phases = [split] if split != "test" else []
                if split == "test" and tick in c["ranking_landmarks"]:
                    phases += ["ranking_selection", "ranking_evaluation"]
                if split == "test" and cohort in c["test_cohorts"][:c["timing_profiles"]] and tick in c["timing_landmarks"]:
                    phases += ["timing_selection", "timing_evaluation"]
                old_roles.update(previous.future_seed(c, identity, p) for p in phases)
        old_roles.add(role_seed(c["analysis_seed"], f"{previous.VERSION}:crossed_bootstrap"))
    if cfg != settings(cfg["profile"]) or len(derived) != len(set(derived)) or set(derived) & (raw | raw_old | old_roles) or raw & old_roles:
        raise ValueError("RNG/settings collision")
    return dict(status="PASS", unique_derived_roles=len(derived), historical_derived_roles=len(old_roles),
        historical_registry_hashes=hashes, own_registry_sha256=hashlib.sha256(own.read_bytes()).hexdigest(),
        intentional_pairing="physical tapes across controllers/sources; roots within phase; initial ensemble weights across data arms")
