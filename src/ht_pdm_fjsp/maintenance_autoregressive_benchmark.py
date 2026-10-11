"""Outcome-independent PM choice benchmark and sealed metadata-only splits."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_joint_rl as old
from ht_pdm_fjsp import maintenance_timing_diagnostic as historical
from ht_pdm_fjsp import maintenance_decision_headroom as headroom
from ht_pdm_fjsp.maintenance_dispatch import DispatchState,role_seed,validate_state
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig
from ht_pdm_fjsp.maintenance_autoregressive_policy import ARMS

VERSION="maintenance_autoregressive_pilot_v1"
ROOT=Path(__file__).resolve().parents[2]
CONDITIONS=("dense","skill_mask")
RAW_KEYS=("train_seeds","development_cohorts","development_shocks","test_cohorts","test_shocks","analysis_seed")


def settings(profile):
    if profile not in ("full","smoke"):raise ValueError("unknown profile")
    full=profile=="full"
    return dict(profile=profile,horizon=48 if full else 4,tail_ticks=12 if full else 2,
        train_seeds=list(range(16100000,16100003)) if full else [16700000],
        development_cohorts=list(range(16200000,16200006)) if full else [16700020],
        development_shocks=[16300000,16300001] if full else [16700030],
        test_cohorts=list(range(16400000,16400024)) if full else [16700040],
        test_shocks=list(range(16500000,16500004)) if full else [16700050,16700051],
        analysis_seed=16600000 if full else 16700060,bootstrap_samples=20000 if full else 200,
        hidden=32 if full else 16,vector_envs=32 if full else 4,
        rollout=3072 if full else 32,env_steps=122880 if full else 64,
        minibatch=256 if full else 16,epochs=4 if full else 2,
        checkpoint_updates=[10,20,30,40] if full else [1,2],
        learning_rate=3e-4,clip=.2,entropy_weight=.01,value_weight=.5,gradient_clip=.5,
        reward_scale=20.,discount=1.,torch_threads=1,evaluation_batch=32 if full else 4,
        conditions=list(CONDITIONS),arms=list(ARMS),rules=list(old.prior.RULES),
        primary_confidence=.9875,cost_margin=.01,audit_profiles=2,audit_shocks=2,
        tail_rule="deadline_matching",purpose="pilot, not confirmatory replication")


def roles(seed):
    rng=random.Random(role_seed(seed,f"{VERSION}:roles"))
    machines=list(range(4));techs=list(range(2));rng.shuffle(machines);rng.shuffle(techs)
    return machines,techs


def make_config(seed,condition,horizon,attempt=0):
    if condition not in CONDITIONS:raise ValueError(condition)
    rng=random.Random(role_seed(seed,f"{VERSION}:configuration:{attempt}"))
    durations=[[rng.randint(2,6) for _ in range(2)] for _ in range(4)]
    effects=[[rng.choice((.25,.5,.75,1.)) for _ in range(2)] for _ in range(4)]
    machines,techs=roles(seed)
    if condition=="skill_mask":
        durations[machines[1]][techs[1]]=0
        durations[machines[3]][techs[0]]=0
    c=WaitingConfig(4,2,horizon,rng.choice((3,4,5)),rng.choice((.25,.45,.6)),
        tuple(map(tuple,durations)),tuple(map(tuple,effects)),waiting_price=12.)
    c.validate();return c


def initial_state(c,seed):
    machines,_=roles(seed)
    rng=random.Random(role_seed(seed,f"{VERSION}:initial_scenario"))
    case=rng.randrange(4);ages=[rng.randrange(c.failure_age) for _ in range(4)]
    ages[machines[0]]=c.failure_age-1 if case&1 else 0
    ages[machines[1]]=c.failure_age-1 if case&2 else 0
    state=DispatchState(tuple(ages),(False,)*4,(0,)*4,(0,)*2,(-1,)*2)
    validate_state(c,state);return state


def scenario(c,seed):
    s=initial_state(c,seed);m,j=roles(seed)
    return dict(focal_machine=m[0],competing_machine=m[1],scarce_technician=j[0],
        focal_at_risk=s.ages[m[0]]+1>=c.failure_age,
        competitor_at_risk=s.ages[m[1]]+1>=c.failure_age,
        initial_state=asdict(s))


@lru_cache(maxsize=1)
def historical_pool():
    blocked=set(historical.exclusion_pool())
    for mode in ("full","smoke"):
        old_panels,_=historical.build_panels(historical.settings(mode))
        blocked.update(old.boundary.signature(c) for c in old_panels.values())
        old_panels,_,_=headroom.build_panels(mode)
        blocked.update(old.boundary.signature(c) for c in old_panels.values())
    return frozenset(blocked)


@lru_cache(maxsize=1)
def metadata():
    """Resolve both protocols without running any environment or policy."""
    blocked=set(historical_pool());panels={};rows={};training={}
    for mode in ("full","smoke"):
        cfg=settings(mode);panels[mode]={};rows[mode]=[]
        for split in ("development","test"):
            for cohort in cfg[f"{split}_cohorts"]:
                for attempt in range(10000):
                    generated={cond:make_config(cohort,cond,cfg["horizon"],attempt) for cond in CONDITIONS}
                    sigs={old.boundary.signature(c) for c in generated.values()}
                    if len(sigs)==2 and not sigs&blocked:break
                else:raise RuntimeError("panel physical pool exhausted")
                blocked.update(sigs)
                for cond,c in generated.items():
                    panels[mode][split,cohort,cond]=c
                    rows[mode].append(dict(split=split,cohort=cohort,condition=cond,generator_attempt=attempt,
                        config=asdict(c),**scenario(c,cohort)))
    # Training excludes all panels and other training signatures. Reuse across
    # arms is intentional: their sampler looks up the exact same immutable spec.
    for mode in ("full","smoke"):
        cfg=settings(mode);training[mode]={}
        for seed in tqdm(cfg["train_seeds"],desc=f"Resolve {mode} training metadata",leave=False):
            for index in range(cfg["env_steps"]//cfg["horizon"]):
                cs=role_seed(seed,f"{VERSION}:training_configuration:{index}")
                es=role_seed(seed,f"{VERSION}:training_environment:{index}")
                condition=CONDITIONS[index%2]
                for attempt in range(10000):
                    c=make_config(cs,condition,cfg["horizon"]+cfg["tail_ticks"],attempt)
                    sig=old.boundary.signature(c)
                    if sig not in blocked:break
                else:raise RuntimeError("training physical pool exhausted")
                blocked.add(sig);training[mode][seed,index]=(c,condition,cs,es,attempt)
    return panels,rows,training


def build_panels(cfg):
    if cfg!=settings(cfg["profile"]):raise ValueError("settings changed")
    p,r,_=metadata();return p[cfg["profile"]],r[cfg["profile"]]


def exclusions():
    return dict(method="metadata-only historical reconstruction and canonical physical signature exclusion",
        excluded_historical_signatures=len(historical_pool()),outcomes_used=False,
        smoke_full_and_train_development_test_disjoint=True)


def seed_audit(cfg):
    own_path=ROOT/f"configs/{VERSION}_seed_registry.json"
    own=json.loads(own_path.read_text())["panels"]
    historical_raw=old.integers(old.exclusions()["historical_raw_panels"]);hashes={}
    for p in sorted((ROOT/"configs").glob("*seed_registry.json")):
        if p!=own_path:
            historical_raw|=old.integers(json.loads(p.read_text()))
            hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    fresh=set();derived=[]
    for mode in ("full","smoke"):
        panel=settings(mode)
        if own[mode]!={k:panel[k] for k in RAW_KEYS}:raise ValueError("seed registry mismatch")
        for key in RAW_KEYS:
            group=old.integers(panel[key])
            if group&(fresh|historical_raw):raise ValueError("raw seed collision")
            fresh|=group
        for seed in panel["train_seeds"]:
            derived.extend(role_seed(seed,f"{VERSION}:{role}") for role in ("initialization","actions","minibatches"))
            for index in range(panel["env_steps"]//panel["horizon"]):
                cs=role_seed(seed,f"{VERSION}:training_configuration:{index}")
                es=role_seed(seed,f"{VERSION}:training_environment:{index}")
                derived.extend((cs,es,role_seed(cs,f"{VERSION}:roles"),role_seed(cs,f"{VERSION}:initial_scenario")))
        for split in ("development","test"):
            for cohort in panel[f"{split}_cohorts"]:
                derived.extend((role_seed(cohort,f"{VERSION}:roles"),role_seed(cohort,f"{VERSION}:initial_scenario")))
                derived.extend(role_seed(s,f"{VERSION}:physical:{cohort}") for s in panel[f"{split}_shocks"])
        derived.append(role_seed(panel["analysis_seed"],f"{VERSION}:crossed_bootstrap"))
    if len(derived)!=len(set(derived)) or set(derived)&(fresh|historical_raw):raise ValueError("derived seed collision")
    if cfg!=settings(cfg["profile"]):raise ValueError("settings changed")
    return dict(status="PASS",historical_registry_hashes=hashes,derived_unique_rng_roles=len(derived),
        own_registry_sha256=hashlib.sha256(own_path.read_bytes()).hexdigest(),
        intentional_pairing="initial weights/train tapes across arms; physical shock tapes across arms/conditions"),fresh|historical_raw


def training_sampler(cfg,panels,forbidden):
    if panels!=build_panels(cfg)[0]:raise ValueError("panel mismatch")
    _,_,specs=metadata()
    def specification(seed,index):
        spec=specs[cfg["profile"]][seed,index]
        if spec[2]==spec[3] or spec[2] in forbidden or spec[3] in forbidden:raise ValueError("training seed overlap")
        return spec
    return specification
