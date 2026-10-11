"""Fresh natural-opportunity panels; constructing metadata runs no outcomes."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random

from ht_pdm_fjsp import maintenance_assignment_confidence_benchmark as previous
from ht_pdm_fjsp import maintenance_boundary_sensitivity as boundary
from ht_pdm_fjsp import maintenance_joint_rl as historical
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed, validate_state
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig

VERSION = "maintenance_assignment_opportunity_v1"
ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("dense", "skill_mask")
ARMS = ("landmark_low", "opportunity_low", "opportunity_high")
NAMES = ("frozen_ppo", *(f"{a}_{s}" for a in ARMS for s in ("ungated", "gated")))


def settings(profile):
    if profile not in ("full", "smoke"):
        raise ValueError("known profile required")
    registry = json.loads((ROOT / f"configs/{VERSION}_seed_registry.json").read_text())
    if registry["protocol"] != VERSION:
        raise ValueError("seed protocol")
    full = profile == "full"
    return dict(profile=profile, **registry["panels"][profile],
        model_seeds=[18100000, 18100001, 18100002] if full else [18100000],
        conditions=list(CONDITIONS), horizon=48, tail_ticks=12, hidden=32, reward_scale=20.,
        landmarks=[0, 12, 24, 36], opportunity_ticks=list(range(48)), ensemble_members=3,
        train_scenarios=128 if full else 4, low_scenarios=16 if full else 2,
        development_scenarios=128 if full else 4, calibration_scenarios=128 if full else 4,
        selection_scenarios=32 if full else 2, evaluation_scenarios=64 if full else 2,
        epochs=80 if full else 2, checkpoints=[20, 40, 60, 80] if full else [1, 2],
        draws_per_epoch=1024 if full else 64, minibatch=128 if full else 32,
        learning_rate=.001, gradient_clip=1., gate_quantile=.9, gate_sd_weight=1., gate_mc_weight=1.96,
        calibration_minimum=40 if full else 1, calibration_profiles=16 if full else 1,
        calibration_late=10 if full else 0, tie_tolerance=1e-5,
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
    return dict(focal_machine=machines[0], competing_machine=machines[1], scarce_technician=techs[0],
        focal_at_risk=s.ages[machines[0]] + 1 >= c.failure_age,
        competitor_at_risk=s.ages[machines[1]] + 1 >= c.failure_age, initial_state=asdict(s))


def physical_seed(cohort, shock):
    return role_seed(shock, f"{VERSION}:physical:{cohort}")


def future_seed(cfg, identity, phase):
    key = f"{phase}_future_seed" if phase in ("train", "development", "calibration") else f"{phase}_seed"
    fields = ("condition", "cohort", "shock", "model_seed", "tick", "origin")
    return role_seed(cfg[key], f"{VERSION}:{phase}:" + ":".join(str(identity[k]) for k in fields))


@lru_cache(maxsize=1)
def metadata():
    scoring = previous.previous.previous
    blocked = set(scoring.historical_pool())
    for package in (scoring, previous.previous, previous):
        panels, _, *rest = package.metadata()
        for p in panels.values():
            blocked.update(boundary.signature(c) for c in p.values())
        if package is scoring:
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


def derived_roles(package, mode):
    c = package.settings(mode); version = package.VERSION
    _, records, _ = package.metadata(); result = []
    result += [role_seed(cohort, f"{version}:configuration:{attempt}")
        for cohort, attempt in sorted({(r["cohort"], r["generator_attempt"]) for r in records[mode]})]
    for seed in c["train_seeds"]:
        if package is previous.previous:
            result += [role_seed(seed, f"{version}:{r}") for r in ("initialization", "minibatches", "shuffle")]
        else:
            for member in range(c["ensemble_members"]):
                result.append(role_seed(seed, f"{version}:initialization:{member}"))
                result += [role_seed(seed, f"{version}:draws:{member}:{epoch}") for epoch in range(1, c["epochs"] + 1)]
    for split in ("train", "development", "calibration", "test"):
        if f"{split}_cohorts" not in c:
            continue
        cohorts = c[f"{split}_cohorts"]
        for cohort in cohorts:
            result += [role_seed(cohort, f"{version}:{r}") for r in ("roles", "initial_scenario")]
            result += [package.physical_seed(cohort, shock) for shock in c[f"{split}_shocks"]]
        if package is previous.previous:
            used = cohorts
            ticks = c["landmarks"] if split != "test" else sorted(set(c["ranking_landmarks"] + c["timing_landmarks"]))
            origins = ((None, ticks),)
        else:
            used = cohorts if split != "test" else cohorts[:c["diagnostic_profiles"]]
            origins = (("natural", c["opportunity_ticks"]),) if package is sys_module() else (
                ("natural", c["landmarks"]), ("probe", c["probe_landmarks"]))
        shocks = c[f"{split}_shocks"] if split != "test" else c["test_shocks"][:1]
        for cond, cohort, shock, seed in itertools.product(CONDITIONS, used, shocks, c["model_seeds"]):
            for origin, ticks in origins:
                for tick in ticks:
                    identity = dict(condition=cond, cohort=cohort, shock=shock, model_seed=seed, tick=tick)
                    if origin is not None:
                        identity["origin"] = origin
                    if origin == "probe" and seed == c["model_seeds"][0]:
                        result.append(package.probe_seed(identity))
                    phases = [split] if split != "test" else ["ranking_selection", "ranking_evaluation"]
                    if package is previous.previous and split == "test":
                        phases = []
                        if tick in c["ranking_landmarks"]:
                            phases += ["ranking_selection", "ranking_evaluation"]
                        if cohort in c["test_cohorts"][:c["timing_profiles"]] and tick in c["timing_landmarks"]:
                            phases += ["timing_selection", "timing_evaluation"]
                    result += [package.future_seed(c, identity, phase) for phase in phases]
    result.append(role_seed(c["analysis_seed"], f"{version}:crossed_bootstrap"))
    return result


def sys_module():
    import sys
    return sys.modules[__name__]


def seed_audit(cfg):
    own = ROOT / f"configs/{VERSION}_seed_registry.json"
    raw_old = historical.integers(historical.exclusions()["historical_raw_panels"]); hashes = {}
    for path in sorted((ROOT / "configs").glob("*seed_registry.json")):
        if path != own:
            raw_old |= historical.integers(json.loads(path.read_text()))
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = set(); derived = []; old_roles = previous.previous.historical_roles()
    for mode in ("full", "smoke"):
        for value in json.loads(own.read_text())["panels"][mode].values():
            group = historical.integers(value)
            if group & (raw | raw_old):
                raise ValueError("raw seed collision")
            raw |= group
        derived += derived_roles(sys_module(), mode)
        for package in (previous.previous, previous):
            old_roles.update(derived_roles(package, mode))
    if cfg != settings(cfg["profile"]) or len(derived) != len(set(derived)) or set(derived) & (raw | raw_old | old_roles) or raw & old_roles:
        raise ValueError("RNG/settings collision")
    return dict(status="PASS", unique_derived_roles=len(derived), historical_derived_roles=len(old_roles),
        historical_registry_hashes=hashes, own_registry_sha256=hashlib.sha256(own.read_bytes()).hexdigest(),
        intentional_pairing="same physical tapes across controllers/sources; roots and nested low/high futures; initial weights; opportunity low/high draws")
