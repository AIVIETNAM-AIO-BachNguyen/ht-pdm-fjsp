"""Closed-loop H48 evaluation with physical replay and PM assignment logging."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_opportunity_cost as prior
from ht_pdm_fjsp import maintenance_boundary_sensitivity as boundary
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching, eligible_edges
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, ENV_VERSION
from ht_pdm_fjsp.maintenance_heterogeneous_model import feasible, fixed_candidates, arrays, batch_tick
from ht_pdm_fjsp.maintenance_solver_diagnostics import Requests
from ht_pdm_fjsp.maintenance_autoregressive_policy import CATALOGUE_ARM, FEATURE_CONTRACT as OLD_CONTRACT, feature_batch
FEATURE_CONTRACT = {**OLD_CONTRACT, "scoring": "parameter-matched pair/resource interaction", "token_tie_tolerance": 1e-5, "assignment_mass_tolerance": 1e-6}
from ht_pdm_fjsp.maintenance_assignment_value_benchmark import initial_state, scenario, VERSION
ROOT=Path(__file__).resolve().parents[2]
CONDITIONS=("dense","skill_mask")


class CountingJournal(prior.Journal):
    def __init__(self, path):
        super().__init__(path)
        self.rows = 0

    def add(self, row):
        super().add(row)
        self.rows += 1


def evaluate(panels,cfg,split,name,seed=-1,model=None,update=-1,rule=None,journals=None,time_totals=None):
    h,tail=cfg["horizon"],cfg["tail_ticks"]
    grid=list(itertools.product(CONDITIONS,cfg[f"{split}_cohorts"],cfg[f"{split}_shocks"]))
    rows=[];size=cfg["evaluation_batch"]
    for begin in tqdm(range(0,len(grid),size),desc=f"{split} {name} seed {seed} update {update}",unit="batch",leave=False):
        identities=[dict(condition=cond,cohort=cohort,shock=shock,horizon=h,controller=name,train_seed=seed,checkpoint_update=update)
                    for cond,cohort,shock in grid[begin:begin+size]]
        configs=[replace(panels[split,r["cohort"],r["condition"]],horizon=h+tail) for r in identities]
        envs=[WaitingEnv(c) for c in configs]
        env_seeds=[role_seed(r["shock"],f"{VERSION}:physical:{r['cohort']}") for r in identities]
        for env,es,identity in zip(envs,env_seeds,identities,strict=True):env.reset(es,initial_state(env.config,identity["cohort"]))
        base_track=[Requests(env.state) for env in envs];all_track=[Requests(env.state) for env in envs]
        interior=[defaultdict(float) for _ in envs];behavior=[defaultdict(int) for _ in envs]
        latency=[[] for _ in envs];cut=[None]*len(envs);base_metrics=[None]*len(envs);cut_state=[None]*len(envs)
        for t in range(h+tail):
            states=[env.state for env in envs];phase="base" if t<h else "tail"
            start=time.perf_counter_ns()
            if t<h and model is not None:
                action,prob,value=model.sample(feature_batch(configs,states,[h-t]*len(envs),h,model.algorithm),deterministic=True)
                pairs=[model.decode(c,state,a) for c,state,a in zip(configs,states,action,strict=True)]
                diagnostics=[dict(action_trace=prior.compact(a.tolist()),log_probability=float(p),value=float(v),
                    assignment_decode=model.last_decode_audit[i] if model.last_decode_audit else None) for i,(a,p,v) in enumerate(zip(action,prob,value,strict=True))]
            else:
                rule_names = ["deadline_matching" if t >= h else
                    (rule[r["condition"]] if isinstance(rule, dict) else rule) for r in identities]
                pairs=[prior.rule_action(c,state,h+tail-t if t>=h else h-t,selected)
                       for c,state,selected in zip(configs,states,rule_names,strict=True)]
                diagnostics=[dict(rule=selected) for selected in rule_names]
            amortized=(time.perf_counter_ns()-start)/1e6/len(envs)
            for i,(c,env,state,pair,identity,diag) in enumerate(zip(configs,envs,states,pairs,identities,diagnostics,strict=True)):
                validate_matching(c,state,pair)
                if cfg["profile"]=="smoke" and t<h and model is not None:
                    if model.select(c,state,h-t,h)[0]!=pair:raise RuntimeError("smoke scalar/batch policy mismatch")
                if t<h:
                    latency[i].append(amortized)
                    candidates=feasible(c,state)
                    behavior[i]["wait_with_choice"]+=not pair and len(candidates)>1
                    behavior[i]["nonfixed_assignments"]+=pair not in fixed_candidates(c,state)
                    behavior[i]["assignment_alternatives"]+=sum(tuple(m for m,j in a)==tuple(m for m,j in pair) for a in candidates)>1
                    behavior[i]["preventive_starts"]+=sum(not state.failed[m] for m,j in pair)
                    behavior[i]["corrective_starts"]+=sum(state.failed[m] for m,j in pair)
                    legal=eligible_edges(c,state)
                    pm_rows=[m for m in range(4) if not state.failed[m] and any(legal[m])]
                    behavior[i]["pm_feasible_ticks"]+=bool(pm_rows)
                    behavior[i]["wait_with_pm_feasible"]+=not pair and bool(pm_rows)
                    behavior[i]["pm_two_worker_opportunity_ticks"]+=any(sum(legal[m])==2 for m in pm_rows)
                    used_m={m for m,j in pair};used_j={j for m,j in pair}
                    extension=any(not state.failed[m] and m not in used_m and j not in used_j and legal[m][j]
                        for m in range(4) for j in range(2))
                    behavior[i]["stop_with_pm_extension_feasible"]+=extension
                    fixed_cm={(m,j) for m,j in pair if state.failed[m]}
                    same_set=[a for a in candidates if {m for m,j in a}==used_m and fixed_cm.issubset(a)]
                    alternatives=any({(m,j) for m,j in a if not state.failed[m]}!=
                                     {(m,j) for m,j in pair if not state.failed[m]} for a in same_set)
                    behavior[i]["pm_assignment_alternative_ticks"]+=alternatives
                    behavior[i]["pm_starts_with_two_workers"]+=sum(not state.failed[m] and sum(legal[m])==2 for m,j in pair)
                    behavior[i]["cm_only_assignment_alternative_ticks"]+=bool(pair) and all(state.failed[m] for m,j in pair) and sum({m for m,j in a}==used_m for a in candidates)>1
                    if journals is not None and split=="test":
                        for m,j in pair:
                            if not state.failed[m]:
                                journals["pm_choices"].add({**identity,"time":t,"machine":m,"technician":j,
                                    "age":state.ages[m],"at_risk":state.ages[m]+1>=c.failure_age,
                                    "initial_eligible_technicians":prior.compact([jj for jj in range(2) if legal[m][jj]]),
                                    "service_times":prior.compact(c.service_time[m]),"restorations":prior.compact(c.restoration[m]),
                                    "pm_assignment_alternative_same_machine_set_fixed_cm":any((m,j) not in a for a in same_set),
                                    "action":prior.compact(pair),"controller_diagnostics":prior.compact(diag)})
                _,_,done,info=env.step(pair)
                if done!=(t==h+tail-1):raise RuntimeError("evaluation boundary mismatch")
                selected=np.zeros((1,4,2),bool)
                for m,j in pair:selected[0,m,j]=True
                following,cost=batch_tick(c,arrays(state),selected,np.asarray([env.events[t]]))
                if not all(np.array_equal(following[k][0],getattr(env.state,k)) for k in following) or abs(cost[0]-info["objective"])>1e-8:
                    raise RuntimeError("evaluation scalar/batch mismatch")
                all_track[i].step(t,state,pair,env.state)
                if t<h:base_track[i].step(t,state,pair,env.state)
                if h//4<=t<h-h//4:
                    for k,v in info.items():interior[i][k]+=v
                    for k,v in boundary.snapshot(state).items():interior[i][k]+=v
                if time_totals is not None:
                    total=time_totals[identity["condition"],name,seed,t,phase];total["episodes"]+=1
                    for k,v in info.items():total[k]+=v
                    for k,v in boundary.snapshot(state).items():total[f"before_{k}"]+=v
                if journals is not None and split=="test" and identity["cohort"] in cfg["test_cohorts"][:cfg["audit_profiles"]] and identity["shock"] in cfg["test_shocks"][:cfg["audit_shocks"]]:
                    journals["decisions"].add({**identity,"time":t,"phase":phase,"env_seed":env_seeds[i],
                        "state":prior.compact(asdict(state)),"action":prior.compact(pair),"following":prior.compact(asdict(env.state)),
                        "physical_metrics":prior.compact(info),"controller_diagnostics":prior.compact(diag)})
                if t+1==h:
                    base_metrics[i]=env.metrics.copy();cut_state[i]=boundary.snapshot(env.state)
                    cut[i]=[{**r,"overdue":r["wait"]>c.waiting_limit} for r in base_track[i].finish(h)]
        for i,(c,env,identity) in enumerate(zip(configs,envs,identities,strict=True)):
            extended=[{**r,"overdue":r["wait"]>c.waiting_limit} for r in all_track[i].finish(h+tail)]
            later={(r["machine"],r["request"]):r for r in extended};carry=[]
            for r in cut[i]:
                if r["censored"]:
                    end=later[r["machine"],r["request"]]
                    carry.append(dict(machine=r["machine"],request=r["request"],arrival=r["arrival"],wait_at_cutoff=r["wait"],
                        started_in_tail=end["served"],wait_at_start_or_end=end["wait"],censored_after_tail=end["censored"]))
            window=h-2*(h//4)
            row={**identity,**{k:v for k,v in scenario(c,identity["cohort"]).items() if k!="initial_state"},**{f"base_{k}":v for k,v in base_metrics[i].items()},
                **{f"tail_{k}":v-base_metrics[i][k] for k,v in env.metrics.items()},
                **{f"extended_{k}":v for k,v in env.metrics.items()},
                **{f"cutoff_{k}":v for k,v in cut_state[i].items()},
                **{f"post_tail_{k}":v for k,v in boundary.snapshot(env.state).items()},
                **{f"cutoff_{k}":v for k,v in boundary.request_metrics(cut[i]).items()},
                **{f"extended_{k}":v for k,v in boundary.request_metrics(extended).items()},
                "extended_cost_per_machine_tick":env.metrics["objective"]/(4*(h+tail)),
                "interior_cost_per_machine_tick":interior[i]["objective"]/(4*window),
                "interior_pending_mean":interior[i]["pending"]/window,"interior_failed_mean":interior[i]["failed"]/window,
                "interior_overdue_per_tick":interior[i]["overdue_waiting_ticks"]/window,
                **{k:behavior[i][k] for k in ("wait_with_choice","nonfixed_assignments","assignment_alternatives","preventive_starts","corrective_starts","pm_feasible_ticks","wait_with_pm_feasible",
                    "pm_two_worker_opportunity_ticks","stop_with_pm_extension_feasible","pm_assignment_alternative_ticks",
                    "pm_starts_with_two_workers","cm_only_assignment_alternative_ticks")},
                "batch_amortized_policy_ms_mean":float(np.mean(latency[i])),"evaluation_batch_size":len(envs),
                "carryover_requests":len(carry),"carryover_started_in_tail":sum(r["started_in_tail"] for r in carry)}
            rows.append(row)
            if journals is not None:
                journals["development_episodes" if split=="development" else "episodes.partial"].add(row)
                if split=="test":
                    for scope,records in (("cutoff",cut[i]),("extended",extended)):
                        for r in records:journals["requests"].add({**identity,"scope":scope,**r})
                    for r in carry:journals["carryover"].add({**identity,**r})
                    journals["coordination"].add({**identity,"base_decisions":h,"tail_decisions":tail,
                        "invalid_actions":0,"scalar_batch_replay":"PASS","matching_feasibility":"PASS","token_padding_audit":"PASS" if model is not None and model.algorithm!=CATALOGUE_ARM else "NOT_APPLICABLE"})
    return rows
