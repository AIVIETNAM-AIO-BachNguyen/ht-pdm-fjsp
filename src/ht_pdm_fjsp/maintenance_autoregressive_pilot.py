"""Train paired catalogue/pair-STOP RL architectures on a PM choice benchmark."""
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
from ht_pdm_fjsp.maintenance_autoregressive_policy import CataloguePolicy, AutoregressivePolicy, ARMS, CATALOGUE_ARM, CONTEXT, FROZEN, FEATURE_CONTRACT, feature_batch
from ht_pdm_fjsp.maintenance_autoregressive_training import fit
from ht_pdm_fjsp.maintenance_autoregressive_statistics import summarize
from ht_pdm_fjsp.maintenance_autoregressive_benchmark import (settings, build_panels, seed_audit, training_sampler, exclusions, initial_state, scenario)

VERSION="maintenance_autoregressive_pilot_v1"
ROOT=Path(__file__).resolve().parents[2]
CONDITIONS=("dense","skill_mask")


def counts(cfg):
    n=len(ARMS)*len(cfg["train_seeds"]);h=cfg["horizon"];tail=cfg["tail_ticks"]
    dev=len(CONDITIONS)*len(cfg["development_cohorts"])*len(cfg["development_shocks"])
    test=len(CONDITIONS)*len(cfg["test_cohorts"])*len(cfg["test_shocks"])
    episodes=(n+1)*test
    return dict(models=n,training_learned_steps=n*cfg["env_steps"],
        training_physical_steps=n*(cfg["env_steps"]//h)*(h+tail),training_episodes=n*cfg["env_steps"]//h,
        optimizer_steps=n*(cfg["env_steps"]//cfg["rollout"])*cfg["epochs"]*cfg["rollout"]//cfg["minibatch"],
        development_episodes=dev*(len(prior.RULES)+n*(len(cfg["checkpoint_updates"])+1)),
        test_episodes=episodes,test_base_decisions=episodes*h,test_tail_decisions=episodes*tail,
        test_decisions=episodes*(h+tail),
        logged_test_decisions=(n+1)*len(CONDITIONS)*min(cfg["audit_profiles"],len(cfg["test_cohorts"]))*min(cfg["audit_shocks"],len(cfg["test_shocks"]))* (h+tail))


def model_for(cfg,seed,algorithm):
    torch.manual_seed(role_seed(seed,f"{VERSION}:initialization"))
    return (CataloguePolicy(cfg["hidden"]) if algorithm==CATALOGUE_ARM else AutoregressivePolicy(algorithm,cfg["hidden"])).eval()


def load_checkpoint(path):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload["protocol"]!=VERSION or payload["feature_contract"]!=FEATURE_CONTRACT or payload["environment_version"]!=ENV_VERSION:
        raise ValueError("checkpoint contract mismatch")
    model=model_for(payload["settings"],payload["train_seed"],payload["algorithm"])
    model.load_state_dict(payload["state_dict"]);return model.eval(),payload


def save_checkpoint(path,model,cfg,seed,update,steps,probe):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(protocol=VERSION,feature_contract=FEATURE_CONTRACT,environment_version=ENV_VERSION,
        algorithm=model.algorithm,train_seed=seed,settings=cfg,update=update,learned_steps=steps,state_dict=model.state_dict()),path)
    restored,payload=load_checkpoint(path)
    if any(not torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()):raise RuntimeError("checkpoint parameter mismatch")
    c,state=probe
    if model.select(c,state,cfg["horizon"],cfg["horizon"])!=restored.select(c,state,cfg["horizon"],cfg["horizon"]):
        raise RuntimeError("checkpoint action/probability/value mismatch")
    return restored,payload


def choose_checkpoint(records):
    trained=[r for r in records if r["update"]>0]
    if not trained:raise ValueError("trained checkpoints required")
    return min(trained,key=lambda r:(r["mean_extended_cost"],r["update"]))


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
                diagnostics=[dict(action_trace=prior.compact(a.tolist()),log_probability=float(p),value=float(v)) for a,p,v in zip(action,prob,value,strict=True)]
            else:
                rule_names = ["deadline_matching" if t >= h else
                    (rule[r["condition"]] if isinstance(rule, dict) else rule) for r in identities]
                pairs=[prior.rule_action(c,state,h+tail-t if t>=h else h-t,selected)
                       for c,state,selected in zip(configs,states,rule_names,strict=True)]
                diagnostics=[dict(rule=selected) for selected in rule_names]
            amortized=(time.perf_counter_ns()-start)/1e6/len(envs)
            for i,(c,env,state,pair,identity,diag) in enumerate(zip(configs,envs,states,pairs,identities,diagnostics,strict=True)):
                validate_matching(c,state,pair)
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


def run(output,profile="smoke",device="cpu"):
    if device!="cpu":raise ValueError("CPU only")
    run_start=time.perf_counter()
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError("new/empty directory required")
    output.mkdir(parents=True,exist_ok=True);cfg=settings(profile);torch.set_num_threads(1)
    now=lambda:datetime.now(timezone.utc).isoformat()
    git=lambda *args:subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
    manifest=dict(protocol=VERSION,status="RUNNING",started=now(),commit=git("rev-parse","HEAD"),dirty=bool(git("status","--porcelain")),
        settings=cfg,expected_counts=counts(cfg),device=device,python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),
        platform=platform.platform(),logical_cpus=os.cpu_count(),environment_version=ENV_VERSION,full_test_outcomes_opened=False,
        thread_environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})
    prior.write_json(output/"manifest.json",manifest);journals={}
    try:
        if profile=="full" and manifest["dirty"]:raise ValueError("full experiment requires clean committed code")
        if profile=="full" and any(v!="1" for v in manifest["thread_environment"].values()):raise ValueError("full requires OMP/OPENBLAS/MKL_NUM_THREADS=1")
        audit,forbidden=seed_audit(cfg);manifest["seed_audit"]=audit
        panels,registry=build_panels(cfg);specification=training_sampler(cfg,panels,forbidden)
        prior.write_json(output/"resolved_config.json",dict(settings=cfg,physical_profiles=registry,feature_contract=FEATURE_CONTRACT))
        prior.write_json(output/"seed_registry.json",dict(raw_panels={k:v for k,v in cfg.items() if 'seed' in k or 'cohort' in k or 'shock' in k},exclusions=exclusions()))
        shutil.copyfile(ROOT/f"docs/{VERSION}_plan.md",output/"plan.md")
        source=output/"source_snapshot";source.mkdir()
        for path in sorted((ROOT/"src/ht_pdm_fjsp").glob("*.py")):
            shutil.copyfile(path,source/path.name)
        for filename in ("pyproject.toml","uv.lock"):shutil.copyfile(ROOT/filename,source/filename)
        for path in (ROOT/"configs").glob("*.json"):shutil.copyfile(path,source/path.name)
        for name in ("training_progress","training_episodes","development_episodes","episodes.partial","decisions","requests","carryover","coordination","pm_choices"):
            journals[name]=CountingJournal(output/f"{name}.csv")
        print(json.dumps(dict(protocol=VERSION,budgets=counts(cfg),training_steps_per_model=cfg["env_steps"],physical_tail_ticks=cfg["tail_ticks"]),indent=2),flush=True)
        development=[]
        for rule in tqdm(prior.RULES,desc="Select strong rule on development"):
            development.extend(evaluate(panels,cfg,"development",rule,rule=rule,journals=journals))
        selected_rules={cond:min(prior.RULES,key=lambda name:np.mean([r["extended_objective"] for r in development if r["condition"]==cond and r["controller"]==name])) for cond in CONDITIONS}
        selections={};coverage=[]
        probe_c=replace(panels["development",cfg["development_cohorts"][0],CONDITIONS[0]],horizon=cfg["horizon"]+cfg["tail_ticks"])
        probe_env=WaitingEnv(probe_c);probe_env.reset(cfg["development_shocks"][0],initial_state(probe_c,cfg["development_cohorts"][0]));probe=(probe_c,probe_env.state)
        for seed,algorithm in tqdm(list(itertools.product(cfg["train_seeds"],ARMS)),desc="Paired training replicas"):
            model=model_for(cfg,seed,algorithm)
            initial_sha=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            base_dir=output/algorithm/f"seed_{seed}";records=[]
            def checkpoint(current,update,steps):
                path=base_dir/"checkpoints"/f"update_{update:04d}.pt"
                restored,_=save_checkpoint(path,current,cfg,seed,update,steps,probe)
                dev=evaluate(panels,cfg,"development",algorithm,seed,restored,update,journals=journals)
                development.extend(dev)
                records.append(dict(update=update,learned_steps=steps,checkpoint=str(path.relative_to(output)),
                    mean_extended_cost=float(np.mean([r["extended_objective"] for r in dev]))))
            checkpoint(model,0,0)
            trained=fit(model,cfg,seed,specification,journals,checkpoint)
            chosen=choose_checkpoint(records);destination=base_dir/"model.pt"
            shutil.copyfile(output/chosen["checkpoint"],destination);load_checkpoint(destination)
            selections[f"{algorithm}:{seed}"]={**chosen,"selected_checkpoint":str(destination.relative_to(output)),"candidates":records}
            coverage.append({**trained,"initial_parameters_sha256":initial_sha,"parameter_count":sum(p.numel() for p in model.parameters())})
            prior.write_json(output/"training_coverage.json",coverage)
            prior.write_json(output/"checkpoint_selection.json",dict(locked_before_test=True,selections=selections,selected_rules=selected_rules))
        if len({r['parameter_count'] for r in coverage})!=1 or any(len({r['initial_parameters_sha256'] for r in coverage if r['train_seed']==seed})!=1 for seed in cfg['train_seeds']):
            raise RuntimeError("paired architecture/initialization audit")
        prior.write_json(output/"development_selection.json",dict(locked_before_test=True,selected_rules=selected_rules))
        manifest["full_test_outcomes_opened"]=profile=="full";prior.write_json(output/"manifest.json",manifest)
        time_totals=defaultdict(lambda:defaultdict(float));rows=[]
        rows.extend(evaluate(panels,cfg,"test","best_rule",rule=selected_rules,journals=journals,time_totals=time_totals))
        for seed,algorithm in tqdm(list(itertools.product(cfg["train_seeds"],ARMS)),desc="Sealed paired multi-seed evaluation"):
            chosen=selections[f"{algorithm}:{seed}"];model,payload=load_checkpoint(output/chosen["selected_checkpoint"])
            rows.extend(evaluate(panels,cfg,"test",algorithm,seed,model,payload["update"],journals=journals,time_totals=time_totals))
        summary,differences=summarize(rows,cfg)
        prior.write_json(output/"summary.json",summary);prior.write_csv(output/"episodes.csv",rows)
        prior.write_csv(output/"profile_differences.csv",differences)
        time_rows=[dict(condition=cond,controller=name,train_seed=seed,time=t,phase=phase,episodes=int(v['episodes']),
            **{k:x/v['episodes'] for k,x in v.items() if k!='episodes'}) for (cond,name,seed,t,phase),v in sorted(time_totals.items())]
        prior.write_csv(output/"time_profiles.csv",time_rows)
        actual=dict(models=len(coverage),training_learned_steps=sum(r['learned_steps'] for r in coverage),
            training_physical_steps=sum(r['physical_steps'] for r in coverage),training_episodes=sum(r['episodes'] for r in coverage),
            optimizer_steps=sum(r['optimizer_steps'] for r in coverage),development_episodes=len(development),test_episodes=len(rows),
            test_base_decisions=len(rows)*cfg['horizon'],test_tail_decisions=len(rows)*cfg['tail_ticks'],
            test_decisions=len(rows)*(cfg['horizon']+cfg['tail_ticks']),logged_test_decisions=journals['decisions'].rows)
        if actual!=counts(cfg):raise RuntimeError("budget/count audit")
        manifest.update(status="COMPLETED",audit="PASS",actual_counts=actual,
            pm_choice_rows=journals["pm_choices"].rows,physical_split_audit=exclusions(),
            elapsed_seconds=time.perf_counter()-run_start)
    except BaseException as error:
        manifest.update(status="FAILED",error=f"{type(error).__name__}: {error}");raise
    finally:
        for journal in journals.values():journal.close()
        manifest['finished']=now()
        manifest['artifacts']={str(p.relative_to(output)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
                               for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        prior.write_json(output/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('smoke','full'),default='smoke')
    parser.add_argument('--device',choices=('cpu',),default='cpu')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.output_dir,args.profile,args.device)


if __name__=='__main__':main()
