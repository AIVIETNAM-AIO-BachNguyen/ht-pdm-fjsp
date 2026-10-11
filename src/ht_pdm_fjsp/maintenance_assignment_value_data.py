"""Source-visited states, all CM-fixed PM subsets, and paired simulator labels."""
from dataclasses import asdict
import itertools
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv
from ht_pdm_fjsp.maintenance_heterogeneous_model import feasible
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_autoregressive_policy import feature_batch,CONTEXT
from ht_pdm_fjsp.maintenance_assignment_value_policy import inputs
from ht_pdm_fjsp.maintenance_assignment_value_benchmark import initial_state,physical_seed,future_seed
from ht_pdm_fjsp.maintenance_assignment_value_forks import fork_roots,scalar_replay
from ht_pdm_fjsp.maintenance_opportunity_cost import compact


def all_assignment_classes(config,state,original):
    cm=tuple(e for e in original if state.failed[e[0]])
    roots=tuple(dict.fromkeys([original,*[a for a in feasible(config,state) if tuple(e for e in a if state.failed[e[0]])==cm]]))
    groups={}
    for i,a in enumerate(roots):groups.setdefault(tuple(m for m,j in a if not state.failed[m]),[]).append(i)
    if not 1<=len(roots)<=21 or any(len(v)>2 for v in groups.values()):raise RuntimeError("N4/K2 PM-class contract")
    return roots,groups


def visits(panels,cfg,split,models,journal,landmarks=None,cohorts=None,shocks=None):
    landmarks=cfg["landmarks"] if landmarks is None else landmarks
    cohorts=cfg[f"{split}_cohorts"] if cohorts is None else cohorts
    shocks=cfg[f"{split}_shocks"] if shocks is None else shocks
    grid=list(itertools.product(cfg["conditions"],cohorts,shocks,cfg["model_seeds"]))
    records=[]
    for cond,cohort,shock,seed in tqdm(grid,desc=f"Source {split} trajectories",unit="trajectory"):
        config=panels[split,cohort,cond];model=models[seed]
        env=WaitingEnv(config);es=physical_seed(cohort,shock);env.reset(es,initial_state(config,cohort))
        for tick in range(cfg["horizon"]):
            state=env.state;original,info=model.select(config,state,48-tick,48)
            identity=dict(condition=cond,cohort=cohort,shock=shock,model_seed=seed,tick=tick)
            if tick in landmarks:records.append(dict(identity=identity,config=config,state=state,original=original))
            _,_,_,metrics=env.step(original)
            journal.add({**identity,"split":split,"env_seed":es,"state":compact(asdict(state)),"action":compact(original),
                "following":compact(asdict(env.state)),"metrics":compact(metrics),"shock_bits":compact(list(env.events[tick])),"diagnostics":compact(info)})
    expected=len(grid)*len(landmarks)
    if len(records)!=expected:raise RuntimeError("source-state count mismatch")
    return records


def save_forecast(path,record,roots,result,seed,phase):
    path.parent.mkdir(parents=True,exist_ok=True)
    arrays={k:result[k] for k in ("costs","base","tail","first","shocks")}
    for scope in ("components","cutoff","terminal"):
        arrays.update({f"{scope}_{k}":v for k,v in result[scope].items()})
    if result["trace"] is not None:arrays["trace"]=result["trace"]
    arrays["identity_json"]=np.array(compact(dict(**record["identity"],phase=phase,seed=seed,roots=roots,
        state=asdict(record["state"]),config=asdict(record["config"]),counts=result["counts"])))
    np.savez_compressed(path,**arrays)


def forecast(output,cfg,record,roots,model,phase,scenarios,stats,filename):
    seed=future_seed(cfg,record["identity"],phase)
    r=fork_roots(record["config"],record["state"],roots,model,seed,scenarios,record["identity"]["tick"],record_trace=cfg["profile"]=="smoke")
    if cfg["profile"]=="smoke":
        scalar_replay(record["config"],record["state"],r,model,record["identity"]["tick"])
        stats["scalar_replayed_futures"]+=len(roots)*scenarios
    stats[f"{phase}_transitions"]+=r["transitions"]
    stats[f"{phase}_states"]+=1
    stats["forecast_transitions"]+=r["transitions"];stats["forecast_roots"]+=len(roots)
    stats["forecast_futures"]+=len(roots)*scenarios
    for k,v in r["counts"].items():stats[k]+=v
    path=output/"forecasts"/phase/f"{filename}.npz"
    save_forecast(path,record,roots,r,seed,phase)
    return r,path.relative_to(output).as_posix()


def make_dataset(output,cfg,split,records,models,state_journal,pair_journal,stats):
    by_seed={}
    for seed in cfg["model_seeds"]:
        selected=[r for r in records if r["identity"]["model_seed"]==seed]
        configs=[r["config"] for r in selected];states=[r["state"] for r in selected];remaining=[48-r["identity"]["tick"] for r in selected]
        data=inputs(feature_batch(configs,states,remaining,48,CONTEXT),models[seed]);pairs=[];labels=[];ses=[];conditions=[]
        for si,record in enumerate(tqdm(selected,desc=f"Label {split} PPO {seed}",unit="state")):
            roots,groups=all_assignment_classes(record["config"],record["state"],record["original"])
            r,path=forecast(output,cfg,record,roots,models[seed],split,cfg[f"{split}_scenarios"],stats,f"{seed}_{si}")
            state_journal.add({**record["identity"],"split":split,"roots":compact(roots),"classes":compact([dict(pm_machines=k,roots=v) for k,v in groups.items()]),
                "state":compact(asdict(record["state"])),"original":compact(record["original"]),"forecast":path,
                "classes_with_choice":sum(len(g)>1 for g in groups.values()),"singleton_classes":sum(len(g)==1 for g in groups.values())})
            for machines,group in groups.items():
                if len(group)==1:continue
                left,right=sorted(group,key=lambda i:CATALOGUE.index(roots[i]))
                delta=r["costs"][right]-r["costs"][left]
                target=float(delta.mean()/cfg["reward_scale"]);se=float(delta.std(ddof=1)/np.sqrt(len(delta))/cfg["reward_scale"])
                pairs.append([si,CATALOGUE.index(roots[left]),CATALOGUE.index(roots[right])]);labels.append(target);ses.append(se)
                conditions.append(cfg["conditions"].index(record["identity"]["condition"]))
                pair_journal.add({**record["identity"],"split":split,"state_index":si,"pm_machines":compact(machines),"left":compact(roots[left]),"right":compact(roots[right]),
                    "target_normalized":target,"paired_mc_se_normalized":se,"forecast":path})
        if not pairs or set(conditions)!=set(range(len(cfg["conditions"]))):raise RuntimeError("no assignment training/development labels in a condition; no resampling allowed")
        dataset={**data,"pairs":torch.tensor(pairs,dtype=torch.long),"targets":torch.tensor(labels,dtype=torch.float32),
            "mc_se":torch.tensor(ses,dtype=torch.float32),"conditions":torch.tensor(conditions,dtype=torch.long)}
        if not all(torch.isfinite(v).all() for v in dataset.values()):raise RuntimeError("nonfinite dataset")
        path=output/"datasets"/f"{split}_{seed}.pt";path.parent.mkdir(exist_ok=True);torch.save(dataset,path)
        restored=torch.load(path,weights_only=True)
        if any(not torch.equal(v,restored[k]) for k,v in dataset.items()):raise RuntimeError("dataset roundtrip")
        by_seed[seed]=dataset
    return by_seed
