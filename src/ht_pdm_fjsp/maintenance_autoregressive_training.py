"""Paired complete-episode PPO: policy decisions before H, shared rule after H."""
from dataclasses import asdict
import hashlib
import json
import math
import time

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv
from ht_pdm_fjsp.maintenance_opportunity_cost import rule_action
from ht_pdm_fjsp.maintenance_autoregressive_policy import feature_batch
from ht_pdm_fjsp.maintenance_autoregressive_benchmark import initial_state

NAMESPACE="maintenance_autoregressive_pilot_v1"


def complete_targets(rewards,learned_horizon):
    if rewards.ndim!=2 or not 0<learned_horizon<=len(rewards) or not torch.isfinite(rewards).all():
        raise ValueError("complete episode reward matrix")
    return rewards.flip(0).cumsum(0).flip(0)[:learned_horizon]


def fit(model,cfg,seed,specification,journals,on_checkpoint):
    h,tail,b=cfg["horizon"],cfg["tail_ticks"],cfg["vector_envs"]
    if cfg["env_steps"]%cfg["rollout"] or cfg["rollout"]%(b*h) or cfg["rollout"]%cfg["minibatch"]:
        raise ValueError("exact PPO rollout budget required")
    actions=torch.Generator().manual_seed(role_seed(seed,f"{NAMESPACE}:actions"))
    minibatches=torch.Generator().manual_seed(role_seed(seed,f"{NAMESPACE}:minibatches"))
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    episodes=steps=physical=optimizer_steps=0
    mix=dict(dense=0,skill_mask=0)
    start=time.perf_counter()
    for update in tqdm(range(1,cfg["env_steps"]//cfg["rollout"]+1),desc=f"Train {model.algorithm} seed {seed}",unit="update",leave=False):
        features=[];action_rows=[];prob_rows=[];value_rows=[];targets=[];costs=[]
        model.eval()
        for _ in range(cfg["rollout"]//(b*h)):
            specs=[specification(seed,episodes+i) for i in range(b)]
            configs=[s[0] for s in specs];envs=[WaitingEnv(c) for c in configs]
            for env,spec in zip(envs,specs,strict=True):env.reset(spec[3],initial_state(env.config,spec[2]))
            rewards=[]
            for t in range(h+tail):
                if t<h:
                    batch=feature_batch(configs,[env.state for env in envs],[h-t]*b,h,model.algorithm)
                    action,prob,value=model.sample(batch,generator=actions)
                    selected=[model.decode(c,env.state,a) for c,env,a in zip(configs,envs,action,strict=True)]
                    features.append(batch);action_rows.append(action);prob_rows.append(prob);value_rows.append(value)
                    steps+=b
                else:selected=[rule_action(c,env.state,h+tail-t,"deadline_matching") for c,env in zip(configs,envs,strict=True)]
                tick=[]
                for env,pairs in zip(envs,selected,strict=True):
                    _,reward,done,_=env.step(pairs)
                    if done!=(t==h+tail-1):raise RuntimeError("physical episode boundary mismatch")
                    tick.append(reward/cfg["reward_scale"])
                rewards.append(tick);physical+=b
            reward_matrix=torch.tensor(rewards,dtype=torch.float32)
            returns=complete_targets(reward_matrix,h)
            targets.append(returns.flatten())
            for i,(env,spec) in enumerate(zip(envs,specs,strict=True)):
                c,condition,config_seed,env_seed,attempt=spec
                if abs(float(returns[0,i])+env.metrics["objective"]/cfg["reward_scale"])>1e-4:
                    raise RuntimeError("tail return reconciliation")
                episodes+=1;mix[condition]+=1;costs.append(env.metrics["objective"])
                journals["training_episodes"].add(dict(algorithm=model.algorithm,train_seed=seed,episode=episodes,
                    rollout_update=update,config_seed=config_seed,environment_seed=env_seed,generator_attempt=attempt,
                    condition=condition,config_sha256=hashlib.sha256(json.dumps(asdict(c),sort_keys=True).encode()).hexdigest(),
                    learned_steps=steps,physical_steps=physical,**env.metrics))
        batch={k:torch.cat([r[k] for r in features]) for k in features[0]}
        action=torch.cat(action_rows);oldprob=torch.cat(prob_rows);oldvalue=torch.cat(value_rows);target=torch.cat(targets)
        if len(target)!=cfg["rollout"]:raise RuntimeError("rollout population mismatch")
        advantage=target-oldvalue
        advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
        with torch.no_grad():checked,_,values=model.evaluate(batch,action)
        if not torch.allclose(checked,oldprob,atol=5e-5,rtol=1e-5) or not torch.allclose(values,oldvalue,atol=1e-5,rtol=1e-5):
            raise RuntimeError("sampled/reevaluated likelihood/value mismatch")
        model.train();statistics=[]
        for _ in range(cfg["epochs"]):
            order=torch.randperm(cfg["rollout"],generator=minibatches)
            for begin in range(0,cfg["rollout"],cfg["minibatch"]):
                idx=order[begin:begin+cfg["minibatch"]];mini={k:v[idx] for k,v in batch.items()}
                prob,entropy,value=model.evaluate(mini,action[idx]);ratio=(prob-oldprob[idx]).exp()
                policy_loss=-torch.minimum(ratio*advantage[idx],ratio.clamp(1-cfg["clip"],1+cfg["clip"])*advantage[idx]).mean()
                value_loss=(value-target[idx]).square().mean();ent=entropy.mean()
                loss=policy_loss+cfg["value_weight"]*value_loss-cfg["entropy_weight"]*ent
                if not torch.isfinite(loss):raise RuntimeError("nonfinite PPO loss")
                optimizer.zero_grad();loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip"],error_if_nonfinite=True)
                optimizer.step();optimizer_steps+=1
                statistics.append(dict(loss=float(loss.detach()),policy_loss=float(policy_loss.detach()),
                    value_loss=float(value_loss.detach()),entropy=float(ent.detach()),gradient_norm=float(norm),
                    clipped_fraction=float((abs(ratio.detach()-1)>cfg["clip"]).float().mean())))
        stats={k:sum(r[k] for r in statistics)/len(statistics) for k in statistics[0]}
        if not all(math.isfinite(x) for x in stats.values()):raise RuntimeError("nonfinite training stats")
        journals["training_progress"].add(dict(algorithm=model.algorithm,train_seed=seed,rollout_update=update,
            learned_steps=steps,physical_steps=physical,episodes=episodes,optimizer_steps=optimizer_steps,
            mean_extended_episode_cost=sum(costs)/len(costs),elapsed_seconds=time.perf_counter()-start,**stats))
        if update in cfg["checkpoint_updates"]:on_checkpoint(model.eval(),update,steps)
    changed=any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    expected=(cfg["env_steps"]//cfg["rollout"])*cfg["epochs"]*cfg["rollout"]//cfg["minibatch"]
    if steps!=cfg["env_steps"] or physical!=episodes*(h+tail) or optimizer_steps!=expected or not changed:
        raise RuntimeError("training budget/parameter-change audit")
    return dict(algorithm=model.algorithm,train_seed=seed,learned_steps=steps,physical_steps=physical,
        episodes=episodes,optimizer_steps=optimizer_steps,mixture_counts=mix,parameters_changed=changed,
        complete_tail_returns=True,probability_audit=True,teacher_labels=0,reserved_training_transitions=0,
        elapsed_seconds=time.perf_counter()-start)
