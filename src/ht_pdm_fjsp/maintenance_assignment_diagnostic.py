"""Separate one-step PM timing/assignment checks on common baseline states."""
import csv
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_opportunity_cost as prior
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed
from ht_pdm_fjsp.maintenance_autoregressive_policy import feature_batch
from ht_pdm_fjsp.maintenance_assignment_policy import ARMS, controller
from ht_pdm_fjsp.maintenance_assignment_forks import candidates, choose, paired_gain, policy_scores, fork_roots, scalar_replay
from ht_pdm_fjsp.maintenance_ar_decision_diagnostic import save_forecasts
from ht_pdm_fjsp.maintenance_timing_statistics import crossed_ratio

VERSION="maintenance_assignment_scoring_v1"
PHASES=("selection","evaluation")


def forecast_seed(cfg, identity, phase):
    if phase not in PHASES:
        raise ValueError("unknown phase")
    name=":".join(str(identity[k]) for k in ("condition","cohort","shock","model_seed","tick"))
    return role_seed(cfg[f"diagnostic_{phase}_seed"],f"{VERSION}:forecast:{phase}:{name}")


def parameters_sha(model):
    return hashlib.sha256(b"".join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def summarize(rows, cfg):
    profiles=cfg["test_cohorts"][:cfg["audit_profiles"]]
    shocks=cfg["test_shocks"][:cfg["audit_shocks"]]
    seeds=cfg["train_seeds"];ticks=cfg["diagnostic_landmarks"]
    index={(r["arm"],r["condition"],r["cohort"],r["shock"],r["model_seed"],r["tick"]):r for r in rows}
    expected=set(itertools.product(ARMS,cfg["conditions"],profiles,shocks,seeds,ticks))
    if len(index)!=len(rows) or set(index)!=expected:
        raise RuntimeError("incomplete common-state diagnostic grid")
    rng=np.random.default_rng(role_seed(cfg["analysis_seed"],f"{VERSION}:diagnostic_bootstrap"))
    rd=rng.integers(len(seeds),size=(cfg["bootstrap_samples"],len(seeds)))
    pd=rng.integers(len(profiles),size=(cfg["bootstrap_samples"],len(profiles)))
    results={};strata=[]
    for arm,cond,contrast in itertools.product(ARMS,cfg["conditions"],("timing","assignment","mass_decoder")):
        num=np.zeros((len(seeds),len(profiles)));den=np.zeros_like(num)
        for i,seed in enumerate(seeds):
            for j,cohort in enumerate(profiles):
                group=[index[arm,cond,cohort,shock,seed,t] for shock,t in itertools.product(shocks,ticks)]
                num[i,j]=sum(r[f"{contrast}_gain"] for r in group)
                den[i,j]=sum(r[f"{contrast}_exposure"] for r in group)
        draws=crossed_ratio(num,den,rd,pd);finite=draws[np.isfinite(draws)]
        empty=float(1-len(finite)/len(draws))
        ci=np.quantile(finite,[.025,.975]).tolist() if den.sum() and empty<=.05 else None
        results[f"{arm}:{cond}:{contrast}"]=dict(gain=float(num.sum()/den.sum()) if den.sum() else None,
            confidence=.95,confidence_interval=ci,exposure=int(den.sum()),total_states=int(num.size*len(shocks)*len(ticks)),
            empty_bootstrap_fraction=empty,exploratory=True,
            status="ENGINEERING_ONLY" if cfg["profile"]=="smoke" else "DESCRIPTIVE" if ci else "INSUFFICIENT",
            replica_gains={str(s):float(num[i].sum()/den[i].sum()) if den[i].sum() else None for i,s in enumerate(seeds)})
        for t in ticks:
            group=[r for r in rows if r["arm"]==arm and r["condition"]==cond and r["tick"]==t and r[f"{contrast}_exposure"]]
            strata.append(dict(arm=arm,condition=cond,contrast=contrast,tick=t,exposure=len(group),
                gain=float(np.mean([r[f"{contrast}_gain"] for r in group])) if group else None))
    return dict(contrasts=results,strata=strata,
        scope="same baseline-visited states, own scorer's greedy root and continuation; independent selection/evaluation tapes",
        uncertainty="exploratory95% crossed replica/profile intervals conditional on MC tapes; root paired MC SE separate",
        interpretation="timing and assignment residual gaps have different eligibility denominators; no oracle training or optimality claim")


def diagnose(output, panels, cfg, selections, loader):
    output=Path(output);h=cfg["horizon"];tail=cfg["tail_ticks"]
    models={};before={}
    for arm,seed in itertools.product(ARMS,cfg["train_seeds"]):
        model,_=loader(output/selections[f"{arm}:{seed}"]["selected_checkpoint"])
        models[arm,seed]=model;before[arm,seed]=parameters_sha(model)
    common=[]
    with (output/"decisions.csv").open() as stream:
        for row in csv.DictReader(stream):
            if (row["controller"]==controller(ARMS[0],"greedy") and row["phase"]=="base"
                and int(row["time"]) in cfg["diagnostic_landmarks"]):
                state=DispatchState(**{k:tuple(v) for k,v in json.loads(row["state"]).items()})
                common.append((dict(condition=row["condition"],cohort=int(row["cohort"]),shock=int(row["shock"]),
                    model_seed=int(row["train_seed"]),tick=int(row["time"])),state))
    expected=len(cfg["conditions"])*min(cfg["audit_profiles"],len(cfg["test_cohorts"]))*min(cfg["audit_shocks"],len(cfg["test_shocks"]))*len(cfg["train_seeds"])*len(cfg["diagnostic_landmarks"])
    if len(common)!=expected:
        raise RuntimeError("common-state count mismatch")
    directory=output/"diagnostic_forecasts";directory.mkdir()
    journals={name:prior.Journal(output/f"diagnostic_{name}.csv") for name in ("states.partial","roots","audits")}
    rows=[];transitions=0;root_count=0
    try:
        for position,(identity,state) in enumerate(tqdm(common,desc="Separate timing/assignment diagnostic",unit="state")):
            c=replace(panels["test",identity["cohort"],identity["condition"]],horizon=h+tail)
            for arm in ARMS:
                model=models[arm,identity["model_seed"]]
                original,info,masses,_,prefixes=policy_scores(c,state,h-identity["tick"],model)
                roots,groups,no_pm=candidates(c,state,original)
                batch=feature_batch([c],[state],[h-identity["tick"]],h)
                greedy=model.greedy_tokens(batch)
                tokens,decode_audit=model.conditional_decode(batch,greedy)
                mass_action=model.decode(c,state,tokens[0]);mass_index=roots.index(mass_action)
                if mass_index not in groups["assignment"]:
                    raise RuntimeError("decoder changed machines/CM")
                selection=fork_roots(c,state,roots,model,forecast_seed(cfg,identity,"selection"),cfg["diagnostic_selection_scenarios"],identity["tick"],record_trace=cfg["profile"]=="smoke")
                selected={k:choose(selection["costs"],groups[k]) for k in ("timing","assignment")}
                selected["mass_decoder"]=mass_index
                evaluation=fork_roots(c,state,roots,model,forecast_seed(cfg,identity,"evaluation"),cfg["diagnostic_evaluation_scenarios"],identity["tick"],record_trace=cfg["profile"]=="smoke")
                row=dict(**identity,arm=arm,state=prior.compact(asdict(state)),original=prior.compact(original),
                    roots=prior.compact(roots),unique_roots=len(roots),no_pm_index=no_pm,
                    token_trace=prior.compact(info["token_trace"]),policy_value=info["value"],
                    original_mass=float(masses[original]),matching_masses=prior.compact([float(masses[r]) for r in roots]),
                    mass_decode=prior.compact(decode_audit[0]),first_stop_probability=prefixes[0]["stop_probability"],
                    selection_seed=forecast_seed(cfg,identity,"selection"),evaluation_seed=forecast_seed(cfg,identity,"evaluation"),
                    remaining_learned=h-identity["tick"],remaining_physical=h+tail-identity["tick"])
                for contrast,index in selected.items():
                    gain,se=paired_gain(evaluation["costs"],0,index)
                    exposure=int(len(groups["assignment" if contrast=="mass_decoder" else contrast])>1)
                    row.update({f"{contrast}_exposure":exposure,f"{contrast}_selected":index,
                        f"{contrast}_gain":gain,f"{contrast}_mc_se":se})
                row["defer_all_gain"],row["defer_all_mc_se"]=paired_gain(evaluation["costs"],0,no_pm)
                journals["states.partial"].add(row);rows.append(row)
                save_forecasts(directory/f"{position:04d}_{arm}.npz",identity,(selection,evaluation))
                root_count+=len(roots)
                for phase,value in zip(PHASES,(selection,evaluation),strict=True):
                    if cfg["profile"]=="smoke":scalar_replay(c,state,value,model,identity["tick"])
                    transitions+=value["transitions"]
                    journals["audits"].add(dict(**identity,arm=arm,phase=phase,transitions=value["transitions"],
                        **value["counts"],feasibility="PASS",scalar_replay="PASS" if cfg["profile"]=="smoke" else "NOT_RUN"))
                    for i,root in enumerate(roots):
                        gain,se=paired_gain(value["costs"],0,i)
                        journals["roots"].add(dict(**identity,arm=arm,phase=phase,root=i,action=prior.compact(root),
                            mass=float(masses[root]),cost=float(value["costs"][i].mean()),gain=gain,paired_mc_se=se,
                            base=float(value["base"][i].mean()),tail=float(value["tail"][i].mean()),
                            **{k:float(v[i].mean()) for k,v in value["components"].items()}))
    finally:
        for journal in journals.values():journal.close()
    if any(parameters_sha(model)!=before[key] for key,model in models.items()):
        raise RuntimeError("diagnostic changed frozen parameters")
    summary=summarize(rows,cfg)
    summary["audit"]=dict(status="PASS",common_states=len(common),model_states=len(rows),unique_roots=root_count,
        forecast_transitions=transitions,parameters_unchanged=True,original_clock_preserved=True,
        timing_and_assignment_separate=True,full_outcomes_opened=cfg["profile"]=="full")
    prior.write_csv(output/"diagnostic_states.csv",rows)
    prior.write_json(output/"diagnostic_summary.json",summary)
    return summary
