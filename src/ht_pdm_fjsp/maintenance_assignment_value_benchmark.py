"""Fresh metadata-only configurations for frozen-policy assignment value learning."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random
from ht_pdm_fjsp import maintenance_assignment_benchmark as previous
from ht_pdm_fjsp import maintenance_joint_rl as historical
from ht_pdm_fjsp import maintenance_boundary_sensitivity as boundary
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed, validate_state
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig
VERSION="maintenance_assignment_value_v1"
ROOT=Path(__file__).resolve().parents[2]
CONDITIONS=("dense","skill_mask")

def settings(profile):
    if profile not in ("full","smoke"): raise ValueError("unknown profile")
    registry=json.loads((ROOT/f"configs/{VERSION}_seed_registry.json").read_text())
    if registry["protocol"]!=VERSION: raise ValueError("seed protocol")
    full=profile=="full"
    return dict(profile=profile,**registry["panels"][profile],model_seeds=[18100000,18100001,18100002] if full else [18100000],
        conditions=list(CONDITIONS),horizon=48,tail_ticks=12,hidden=32,reward_scale=20.,
        landmarks=[0,12,24,36] if full else [0,42],ranking_landmarks=[0,24] if full else [0,42],
        timing_landmarks=[0,6,18,30,42] if full else [0,42],timing_profiles=4 if full else 1,
        train_scenarios=16 if full else 2,development_scenarios=32 if full else 2,
        selection_scenarios=32 if full else 2,evaluation_scenarios=64 if full else 2,
        epochs=80 if full else 2,checkpoints=[20,40,60,80] if full else [1,2],
        minibatch=128,learning_rate=.001,gradient_clip=1.,evaluation_batch=32 if full else 2,
        bootstrap_samples=20000 if full else 200,primary_confidence=.9875,cost_margin=.01,
        torch_threads=1,audit_profiles=4 if full else 1,audit_shocks=1)

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



def physical_seed(cohort,shock):
    return role_seed(shock,f"{VERSION}:physical:{cohort}")

def future_seed(cfg,identity,phase):
    key={"train":"train_future_seed","development":"development_future_seed",
         "ranking_selection":"ranking_selection_seed","ranking_evaluation":"ranking_evaluation_seed",
         "timing_selection":"timing_selection_seed","timing_evaluation":"timing_evaluation_seed"}[phase]
    name=":".join(str(identity[k]) for k in ("condition","cohort","shock","model_seed","tick"))
    # Timing tapes are shared by PPO/value controllers at the same baseline state.
    return role_seed(cfg[key],f"{VERSION}:{phase}:{name}")

@lru_cache(maxsize=1)
def metadata():
    blocked=set(previous.historical_pool())
    p,_,t=previous.metadata()
    for values in p.values():blocked.update(boundary.signature(c) for c in values.values())
    for values in t.values():blocked.update(boundary.signature(r[0]) for r in values.values())
    excluded=len(blocked);panels={};registry={}
    for mode in ("full","smoke"):
        cfg=settings(mode);panels[mode]={};registry[mode]=[]
        for split in ("train","development","test"):
            for cohort in cfg[f"{split}_cohorts"]:
                for attempt in range(10000):
                    generated={cond:make_config(cohort,cond,60,attempt) for cond in CONDITIONS}
                    sigs={boundary.signature(c) for c in generated.values()}
                    if len(sigs)==2 and not sigs&blocked:break
                else:raise RuntimeError("fresh physical pool exhausted")
                blocked.update(sigs)
                for cond,c in generated.items():
                    panels[mode][split,cohort,cond]=c
                    registry[mode].append(dict(split=split,cohort=cohort,condition=cond,generator_attempt=attempt,
                        physical_signature=boundary.signature(c),config=asdict(c),**scenario(c,cohort)))
    return panels,registry,dict(status="PASS",excluded_signatures=excluded,outcomes_used=False,
        full_smoke_train_dev_test_disjoint=True,fresh_signatures=sum(len(v) for v in registry.values()))


def historical_roles():
    from ht_pdm_fjsp import maintenance_autoregressive_benchmark as ar
    from ht_pdm_fjsp import maintenance_ar_decision_diagnostic as diag
    from ht_pdm_fjsp import maintenance_assignment_diagnostic as ad
    roles=set()
    for b in (ar,previous):
        _,records,training=b.metadata()
        for mode in ("full","smoke"):
            c=b.settings(mode)
            for r in records[mode]:roles.add(role_seed(r["cohort"],f"{b.VERSION}:configuration:{r['generator_attempt']}"))
            for seed in c["train_seeds"]:
                roles.update(role_seed(seed,f"{b.VERSION}:{r}") for r in ("initialization","actions","minibatches"))
            for _,_,cs,es,attempt in training[mode].values():
                roles.add(role_seed(cs,f"{b.VERSION}:configuration:{attempt}"))
                roles.update((cs,es,role_seed(cs,f"{b.VERSION}:roles"),role_seed(cs,f"{b.VERSION}:initial_scenario")))
            for split in ("development","test"):
                for cohort in c[f"{split}_cohorts"]:
                    roles.update((role_seed(cohort,f"{b.VERSION}:roles"),role_seed(cohort,f"{b.VERSION}:initial_scenario")))
                    roles.update(role_seed(s,f"{b.VERSION}:physical:{cohort}") for s in c[f"{split}_shocks"])
            roles.update(role_seed(c["analysis_seed"],f"{b.VERSION}:{r}") for r in ("crossed_bootstrap","diagnostic_bootstrap"))
            if b is previous:
                for cond,cohort,shock,seed,tick in itertools.product(CONDITIONS,c["test_cohorts"][:c["audit_profiles"]],c["test_shocks"][:c["audit_shocks"]],c["train_seeds"],c["diagnostic_landmarks"]):
                    identity=dict(condition=cond,cohort=cohort,shock=shock,model_seed=seed,tick=tick)
                    roles.update(ad.forecast_seed(c,identity,phase) for phase in ("selection","evaluation"))
    for mode in ("full","smoke"):
        c=diag.settings(mode)
        for cohort in c["cohorts"]:
            roles.update((role_seed(cohort,f"{ar.VERSION}:roles"),role_seed(cohort,f"{ar.VERSION}:initial_scenario")))
            roles.update(diag.physical_seed(cohort,s) for s in c["physical_shocks"])
        for values in itertools.product(c["conditions"],c["cohorts"],c["physical_shocks"],c["model_seeds"],c["landmarks"]):
            identity=dict(zip(diag.KEYS,values,strict=True))
            roles.update(diag.forecast_seed(c,identity,phase) for phase in diag.PHASES)
        roles.add(role_seed(c["analysis_seed"],f"{diag.VERSION}:crossed_bootstrap"))
    return roles


def seed_audit(cfg):
    own=ROOT/f"configs/{VERSION}_seed_registry.json"
    raw_old=historical.integers(historical.exclusions()["historical_raw_panels"]);hashes={}
    for p in sorted((ROOT/"configs").glob("*seed_registry.json")):
        if p!=own:
            raw_old|=historical.integers(json.loads(p.read_text()))
            hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    fresh=set();derived=[];old_roles=historical_roles()
    registry=json.loads(own.read_text())["panels"]
    _,physical_records,_=metadata()
    for mode in ("full","smoke"):
        c=settings(mode)
        config_roles={(r["cohort"],r["generator_attempt"]) for r in physical_records[mode]}
        derived.extend(role_seed(cohort,f"{VERSION}:configuration:{attempt}") for cohort,attempt in sorted(config_roles))
        for value in registry[mode].values():
            group=historical.integers(value)
            if group&(fresh|raw_old|old_roles):raise ValueError("raw seed collision")
            fresh|=group
        for seed in c["train_seeds"]:
            derived.extend(role_seed(seed,f"{VERSION}:{r}") for r in ("initialization","minibatches","shuffle"))
        for split in ("train","development","test"):
            for cohort in c[f"{split}_cohorts"]:
                derived.extend((role_seed(cohort,f"{VERSION}:roles"),role_seed(cohort,f"{VERSION}:initial_scenario")))
                derived.extend(physical_seed(cohort,s) for s in c[f"{split}_shocks"])
            shocks=c[f"{split}_shocks"] if split!="test" else c["test_shocks"][:1]
            landmarks=c["landmarks"] if split!="test" else sorted(set(c["ranking_landmarks"]+c["timing_landmarks"]))
            for cond,cohort,shock,seed,tick in itertools.product(CONDITIONS,c[f"{split}_cohorts"],shocks,c["model_seeds"],landmarks):
                identity=dict(condition=cond,cohort=cohort,shock=shock,model_seed=seed,tick=tick)
                phases=[split] if split!="test" else (["ranking_selection","ranking_evaluation"] if tick in c["ranking_landmarks"] else [])
                if split=="test" and cohort in c["test_cohorts"][:c["timing_profiles"]] and tick in c["timing_landmarks"]:
                    phases += ["timing_selection","timing_evaluation"]
                derived.extend(future_seed(c,identity,phase) for phase in phases)
        derived.append(role_seed(c["analysis_seed"],f"{VERSION}:crossed_bootstrap"))
    if cfg!=settings(cfg["profile"]) or len(derived)!=len(set(derived)) or set(derived)&(fresh|raw_old|old_roles):
        raise ValueError("derived RNG/settings collision")
    return dict(status="PASS",unique_derived_roles=len(derived),historical_derived_roles=len(old_roles),
        historical_registry_hashes=hashes,own_registry_sha256=hashlib.sha256(own.read_bytes()).hexdigest(),
        intentional_pairing="physical across sources/controllers/conditions, roots within phase; truth/shuffle init and minibatches; independent forecasting phases")
