"""One-step RL assignment-value improvement with separate timing diagnostics."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
import itertools
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_assignment_value_benchmark import ROOT,VERSION,settings,metadata,seed_audit
from ht_pdm_fjsp.maintenance_assignment_value_source import verify_source,sha,parameters_sha
from ht_pdm_fjsp.maintenance_assignment_value_policy import ValueController,CONTRACT
from ht_pdm_fjsp.maintenance_assignment_value_data import visits,make_dataset
from ht_pdm_fjsp.maintenance_assignment_value_training import fit
from ht_pdm_fjsp.maintenance_assignment_value_evaluation import CountingJournal,evaluate
from ht_pdm_fjsp.maintenance_assignment_value_diagnostic import diagnose
from ht_pdm_fjsp.maintenance_assignment_value_statistics import summarize,diagnostic_summary,NAMES
from ht_pdm_fjsp.maintenance_opportunity_cost import write_json,write_csv
from ht_pdm_fjsp.maintenance_waiting import ENV_VERSION


def counts(cfg):
    n=len(cfg["model_seeds"]);c=len(cfg["conditions"])
    traj=lambda split:n*c*len(cfg[f"{split}_cohorts"])*len(cfg[f"{split}_shocks"])
    episodes=traj("test")*3
    tr,dev=traj("train"),traj("development")
    rank_traj=n*c*len(cfg["test_cohorts"]);timing_traj=n*c*min(cfg["timing_profiles"],len(cfg["test_cohorts"]))
    upper=sum(traj(split)*21*cfg[f"{split}_scenarios"]*sum(60-t for t in cfg["landmarks"]) for split in ("train","development"))
    upper+=rank_traj*21*(cfg["selection_scenarios"]+cfg["evaluation_scenarios"])*sum(60-t for t in cfg["ranking_landmarks"])
    upper+=timing_traj*2*4*(cfg["selection_scenarios"]+cfg["evaluation_scenarios"])*sum(60-t for t in cfg["timing_landmarks"])
    return dict(source_models=n,learned_heads=2*n,train_trajectories=tr,development_trajectories=dev,
        train_states=tr*len(cfg["landmarks"]),development_states=dev*len(cfg["landmarks"]),
        test_source_trajectories=rank_traj,source_decisions=48*(tr+dev+rank_traj),
        ranking_states=rank_traj*len(cfg["ranking_landmarks"]),timing_common_states=timing_traj*len(cfg["timing_landmarks"]),
        test_episodes=episodes,test_base_decisions=episodes*48,test_tail_decisions=episodes*12,
        logged_test_decisions=n*c*3*min(len(cfg["test_cohorts"]),cfg["audit_profiles"])*min(len(cfg["test_shocks"]),cfg["audit_shocks"])*60,
        max_forecast_transitions=upper)


def run(output,source_dir,profile="smoke",device="cpu"):
    if device!="cpu":raise ValueError("CPU only")
    output=Path(output);start=time.perf_counter();cfg=settings(profile);torch.set_num_threads(1)
    if output.exists() and any(output.iterdir()):raise ValueError("new empty timestamped directory required; resume unsupported")
    output.mkdir(parents=True,exist_ok=True)
    git=lambda *a:subprocess.check_output(["git",*a],cwd=ROOT,text=True).strip()
    now=lambda:datetime.now(timezone.utc).isoformat()
    m=dict(protocol=VERSION,status="RUNNING",started=now(),commit=git("rev-parse","HEAD"),dirty=bool(git("status","--porcelain")),
        settings=cfg,expected_counts=counts(cfg),environment_version=ENV_VERSION,feature_contract=CONTRACT,
        device=device,platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),
        thread_environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
        full_test_outcomes_opened=False,source_outcomes_previously_opened=True)
    journals={};stats=defaultdict(int);write_json(output/"manifest.json",m)
    try:
        if profile=="full" and m["dirty"]:raise ValueError("full requires clean committed code")
        if profile=="full" and any(v!="1" for v in m["thread_environment"].values()):raise ValueError("full requires OMP/OPENBLAS/MKL_NUM_THREADS=1")
        models,contract,audit=verify_source(source_dir,cfg);m["source_audit"]=audit
        before={seed:parameters_sha(model) for seed,model in models.items()}
        m["seed_audit"]=seed_audit(cfg)
        allpanels,registry,physical=metadata();panels=allpanels[profile];m["physical_split_audit"]=physical
        write_json(output/"source_contract.json",contract)
        write_json(output/"resolved_config.json",dict(settings=cfg,physical_profiles=registry[profile],feature_contract=CONTRACT))
        shutil.copyfile(ROOT/f"configs/{VERSION}_seed_registry.json",output/"seed_registry.json")
        shutil.copyfile(ROOT/f"docs/{VERSION}_plan.md",output/"plan.md")
        snapshot=output/"source_snapshot";snapshot.mkdir()
        for p in sorted((ROOT/"src/ht_pdm_fjsp").glob("*.py")):shutil.copyfile(p,snapshot/p.name)
        for p in sorted((ROOT/"configs").glob("*.json")):shutil.copyfile(p,snapshot/p.name)
        for name in ("pyproject.toml","uv.lock"):shutil.copyfile(ROOT/name,snapshot/name)
        names=("source_trajectories","label_states","label_pairs","training_progress","episodes.partial","decisions","pm_choices","requests","carryover","coordination",
            "ranking_predictions","ranking_states","ranking_pairs","timing_predictions","timing_states")
        journals={name:CountingJournal(output/f"{name}.csv") for name in names}
        print(json.dumps(dict(protocol=VERSION,settings=cfg,budgets=counts(cfg)),indent=2),flush=True)
        source_states={};data={}
        for split in ("train","development"):
            source_states[split]=visits(panels,cfg,split,models,journals["source_trajectories"])
            data[split]=make_dataset(output,cfg,split,source_states[split],models,journals["label_states"],journals["label_pairs"],stats)
        selections={};controllers={};coverage=[]
        for source_seed,train_seed in tqdm(list(zip(cfg["model_seeds"],cfg["train_seeds"],strict=True)),desc="Matched multi-source value training"):
            heads,chosen=fit(output,cfg,train_seed,source_seed,contract["models"][str(source_seed)]["sha256"],data["train"][source_seed],data["development"][source_seed],journals["training_progress"],stats)
            controllers[source_seed]={name:ValueController(models[source_seed],head,name) for name,head in heads.items()}
            selections[str(source_seed)]=chosen
            expected_updates=2*cfg["epochs"]*((len(data["train"][source_seed]["pairs"])+cfg["minibatch"]-1)//cfg["minibatch"])
            coverage.append(dict(source_seed=source_seed,train_seed=train_seed,train_labels=len(data["train"][source_seed]["pairs"]),development_labels=len(data["development"][source_seed]["pairs"]),
                train_states=len(data["train"][source_seed]["context"]),development_states=len(data["development"][source_seed]["context"]),
                source_parameters=sum(v.numel() for v in models[source_seed].parameters()),extra_parameters=sum(v.numel() for v in heads["true_value"].parameters()),
                matched_optimizer_steps=expected_updates,source_parameters_sha256=before[source_seed]))
        if stats["optimizer_steps"]!=sum(r["matched_optimizer_steps"] for r in coverage):raise RuntimeError("optimizer budget mismatch")
        write_json(output/"training_coverage.json",coverage)
        write_json(output/"checkpoint_selection.json",dict(locked_before_test=True,selections=selections,criterion="minimum true dev pair MSE; earliest epoch tie"))
        m["full_test_outcomes_opened"]=profile=="full";write_json(output/"manifest.json",m)
        rows=[];totals=defaultdict(lambda:defaultdict(float))
        for seed,name in tqdm(list(itertools.product(cfg["model_seeds"],NAMES)),desc="Sealed closed-loop multi-seed evaluation"):
            ctrl=models[seed] if name=="frozen_ppo" else controllers[seed][name]
            epoch=contract["models"][str(seed)]["update"] if name=="frozen_ppo" else selections[str(seed)][name]["selected"]["epoch"]
            rows.extend(evaluate(panels,cfg,"test",name,seed,ctrl,epoch,journals=journals,time_totals=totals))
        # Only first test shock; baseline visits supply identical root states to all interventions.
        landmarks=sorted(set(cfg["ranking_landmarks"]+cfg["timing_landmarks"]))
        records=visits(panels,cfg,"test",models,journals["source_trajectories"],landmarks=landmarks,shocks=cfg["test_shocks"][:1])
        ranking,pairs,timing=diagnose(output,cfg,records,models,controllers,journals,stats)
        summary,cells=summarize(rows,cfg);diagnostic,dcells=diagnostic_summary(ranking,pairs,timing,cfg)
        write_json(output/"summary.json",summary);write_json(output/"diagnostic_summary.json",diagnostic)
        write_csv(output/"episodes.csv",rows);write_csv(output/"profile_differences.csv",cells);write_csv(output/"diagnostic_profile_cells.csv",dcells)
        time_rows=[dict(condition=cond,controller=name,model_seed=seed,time=t,phase=phase,episodes=int(v["episodes"]),**{k:x/v["episodes"] for k,x in v.items() if k!="episodes"}) for (cond,name,seed,t,phase),v in sorted(totals.items())]
        write_csv(output/"time_profiles.csv",time_rows)
        if before!={seed:parameters_sha(model) for seed,model in models.items()}:raise RuntimeError("frozen PPO mutated")
        exp=counts(cfg)
        actual=dict(source_models=len(models),learned_heads=sum(len(c) for c in controllers.values()),
            train_trajectories=sum(r["train_states"] for r in coverage)//len(cfg["landmarks"]),development_trajectories=sum(r["development_states"] for r in coverage)//len(cfg["landmarks"]),
            train_states=len(source_states["train"]),development_states=len(source_states["development"]),ranking_states=len(ranking),timing_common_states=len(timing)//2,
            test_source_trajectories=len(records)//len(landmarks),source_decisions=journals["source_trajectories"].rows,
            test_episodes=len(rows),test_base_decisions=len(rows)*48,test_tail_decisions=len(rows)*12,logged_test_decisions=journals["decisions"].rows)
        if actual!={k:v for k,v in exp.items() if k!="max_forecast_transitions"} or stats["forecast_transitions"]>exp["max_forecast_transitions"]:raise RuntimeError("budget/count mismatch")
        if journals["label_states"].rows!=actual["train_states"]+actual["development_states"] or journals["label_pairs"].rows!=sum(r["train_labels"]+r["development_labels"] for r in coverage):raise RuntimeError("label table count mismatch")
        if profile=="smoke" and stats["scalar_replayed_futures"]!=stats["forecast_futures"]:raise RuntimeError("smoke forecasts not fully scalar replayed")
        m.update(status="COMPLETED",audit="PASS",actual_counts=actual,effective_compute=dict(stats),
            frozen_ppo_audit="PASS",pm_choice_rows=journals["pm_choices"].rows,elapsed_seconds=time.perf_counter()-start)
    except BaseException as error:
        m.update(status="FAILED",error=f"{type(error).__name__}: {error}");raise
    finally:
        for j in journals.values():j.close()
        m["finished"]=now()
        m["artifacts"]={p.relative_to(output).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.rglob("*")) if p.is_file() and p.name!="manifest.json"}
        write_json(output/"manifest.json",m)
    print(json.dumps(dict(status=m["status"],audit=m["audit"],output_dir=str(output),elapsed_seconds=m["elapsed_seconds"])),flush=True)
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--profile",choices=("full","smoke"),default="smoke")
    p.add_argument("--device",choices=("cpu",),default="cpu");p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--source-dir",type=Path,required=True)
    a=p.parse_args();run(a.output_dir,a.source_dir,a.profile,a.device)

if __name__=="__main__":main()
