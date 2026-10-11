"""Held-out assignment ranking under PPO; separate own-policy timing forks."""
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_autoregressive_policy import feature_batch,CONTEXT
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_assignment_value_policy import inputs
from ht_pdm_fjsp.maintenance_assignment_value_data import all_assignment_classes,forecast
from ht_pdm_fjsp.maintenance_assignment_value_forks import candidates,choose,paired_gain
from ht_pdm_fjsp.maintenance_opportunity_cost import compact


@torch.no_grad()
def diagnose(output,cfg,records,models,controllers,journals,stats):
    ranking=[];pairs=[];timing=[];rank_count=0;timing_count=0
    for si,record in enumerate(tqdm(records,desc="Separate assignment / timing diagnostics",unit="state")):
        identity=record["identity"];seed=identity["model_seed"];tick=identity["tick"]
        c,state,original=record["config"],record["state"],record["original"]
        base=models[seed];cs=controllers[seed]
        if tick in cfg["ranking_landmarks"]:
            roots,groups=all_assignment_classes(c,state,original)
            data=inputs(feature_batch([c],[state],[48-tick],48,CONTEXT),base)
            scores={name:ctrl.head(data)[0].numpy() for name,ctrl in cs.items()}
            actions={name:ctrl.select(c,state,48-tick,48)[0] for name,ctrl in cs.items()}
            original_group=next(g for g in groups.values() if 0 in g)
            chosen={"frozen_ppo":0,**{name:roots.index(a) for name,a in actions.items()}}
            if any(i not in original_group for i in chosen.values()):raise RuntimeError("assignment changed source machine set/CM")
            journals["ranking_predictions"].add({**identity,"roots":compact(roots),"chosen":compact(chosen),
                **{f"{name}_relative_scores":compact([float(values[CATALOGUE.index(a)]) for a in roots]) for name,values in scores.items()},
                "recorded_before_forecasts":True,"continuation":"frozen PPO"})
            selection,p1=forecast(output,cfg,record,roots,base,"ranking_selection",cfg["selection_scenarios"],stats,str(si))
            evaluation,p2=forecast(output,cfg,record,roots,base,"ranking_evaluation",cfg["evaluation_scenarios"],stats,str(si))
            reference=choose(selection["costs"],original_group)
            row={**identity,"exposure":int(len(original_group)>1),"roots":compact(roots),"original_class":compact(original_group),
                "reference_root":reference,"selection_forecast":p1,"evaluation_forecast":p2}
            for name,index in {**chosen,"simulation_reference":reference}.items():
                gain,se=paired_gain(evaluation["costs"],0,index)
                row.update({f"{name}_chosen":index,f"{name}_gain":gain,f"{name}_mc_se":se})
            ranking.append(row);journals["ranking_states"].add(row)
            for machines,group in groups.items():
                if len(group)==1:continue
                left,right=sorted(group,key=lambda i:CATALOGUE.index(roots[i]))
                delta=evaluation["costs"][right]-evaluation["costs"][left]
                measured=float(delta.mean()/cfg["reward_scale"]);mcse=float(delta.std(ddof=1)/np.sqrt(len(delta))/cfg["reward_scale"])
                pr={**identity,"pm_machines":compact(machines),"left":compact(roots[left]),"right":compact(roots[right]),
                    "original_class":0 in group,"measured_delta":measured,"paired_mc_se":mcse,"evaluation_forecast":p2}
                for name,values in scores.items():
                    pred=float(values[CATALOGUE.index(roots[right])]-values[CATALOGUE.index(roots[left])])
                    pr.update({f"{name}_predicted_delta":pred,f"{name}_squared_error":(pred-measured)**2})
                pairs.append(pr);journals["ranking_pairs"].add(pr)
            rank_count+=1
        if identity["cohort"] in cfg["test_cohorts"][:cfg["timing_profiles"]] and tick in cfg["timing_landmarks"]:
            for name,ctrl in (("frozen_ppo",base),("true_value",cs["true_value"])):
                action=ctrl.select(c,state,48-tick,48)[0]
                allroots,groups,_=candidates(c,state,action)
                roots=tuple(allroots[i] for i in groups["timing"])
                cm=tuple(e for e in action if state.failed[e[0]]);no_pm=roots.index(cm)
                journals["timing_predictions"].add({**identity,"controller":name,"roots":compact(roots),"original":compact(action),
                    "recorded_before_forecasts":True,"continuation":name})
                selection,p1=forecast(output,cfg,record,roots,ctrl,"timing_selection",cfg["selection_scenarios"],stats,f"{si}_{name}")
                evaluation,p2=forecast(output,cfg,record,roots,ctrl,"timing_evaluation",cfg["evaluation_scenarios"],stats,f"{si}_{name}")
                selected=choose(selection["costs"],tuple(range(len(roots))))
                gain,se=paired_gain(evaluation["costs"],0,selected);defer,dse=paired_gain(evaluation["costs"],0,no_pm)
                row={**identity,"controller":name,"exposure":int(len(roots)>1),"roots":compact(roots),"selected_root":selected,
                    "correction_gain":gain,"correction_mc_se":se,"no_pm_root":no_pm,"no_pm_gain":defer,"no_pm_mc_se":dse,
                    "original_mean_cost":float(evaluation["costs"][0].mean()),"selected_mean_cost":float(evaluation["costs"][selected].mean()),
                    "selection_forecast":p1,"evaluation_forecast":p2,"continuation":name}
                timing.append(row);journals["timing_states"].add(row)
            timing_count+=1
    expected_rank=len(cfg["conditions"])*len(cfg["test_cohorts"])*len(cfg["model_seeds"])*len(cfg["ranking_landmarks"])
    expected_timing=len(cfg["conditions"])*min(cfg["timing_profiles"],len(cfg["test_cohorts"]))*len(cfg["model_seeds"])*len(cfg["timing_landmarks"])
    if rank_count!=expected_rank or timing_count!=expected_timing:raise RuntimeError("diagnostic state grid mismatch")
    return ranking,pairs,timing
