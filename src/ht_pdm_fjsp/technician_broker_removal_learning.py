"""Complete-episode PPO-derived learner with immutable behavior baseline cache."""
import hashlib,time
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_balanced_algorithms as base
from ht_pdm_fjsp.technician_broker_removal_policy import CORE,baseline_cache,prepare_actor_advantage,actor_objective,critic_loss


def cache_digest(cache):
    if cache is None:return None
    h=hashlib.sha256()
    for name in sorted(cache):
        t=cache[name];h.update(name.encode());h.update(t.detach().numpy().tobytes())
    return h.hexdigest()


def fit(model,cfg,seed,make_env,journals,callback,guard):
    torch.set_num_threads(1);workers=cfg['vector_envs'];total=cfg['env_steps'];rollout=cfg['rollout']
    if total%rollout or rollout%(48*workers) or any(s%rollout or s<=0 or s>total for s in cfg['checkpoints']):raise ValueError('complete rollout/episode/checkpoint alignment')
    action_rng=torch.Generator().manual_seed(base.role_seed(seed,cfg['rng_namespace']+'actions'));mini_rng=torch.Generator().manual_seed(base.role_seed(seed,cfg['rng_namespace']+'minibatches'));optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);envs=[make_env(seed,i) for i in range(workers)];obs=[e.reset() for e in envs];buffer=[];steps=completed=physical=opts=updates=0;episode_reward=np.zeros(workers);episode_steps=np.zeros(workers,int);initial=base.parameter_digest(model);maximum_replay=0.;emitted=[];started=time.monotonic()
    def emit(name,row):journals[name].add(row)
    bar=tqdm(total=total,desc=f'Train {model.algorithm}/{seed}',unit='step',mininterval=2)
    try:
        while steps<total:
            guard();batch=base.tensors(obs);action,log,value=model.sample(batch,action_rng,epsilon=0.);rewards=[];dones=[];following=[]
            for i,env in enumerate(envs):
                submitted=model.environment_action(action[i]) if hasattr(model,'environment_action') else int(action[i]) if model.algorithm=='ppo' else action[i].numpy().copy();o,r,done,info=env.step(submitted);rewards.append(r);dones.append(done);following.append(o);episode_reward[i]+=r;episode_steps[i]+=1
                if done:
                    if episode_steps[i]!=48:raise RuntimeError('H48 decision clock')
                    metrics=info['episode_metrics']
                    if metrics['physical_ticks']!=60 or metrics['invalid_assignments'] or abs(20*episode_reward[i]+metrics['extended_cost'])>1e-5:raise RuntimeError('training physical/reward reconciliation')
                    emit('training_episodes',dict(metrics,algorithm=model.algorithm,train_seed=seed,worker=i,learned_steps=steps+workers,episode_reward=float(episode_reward[i])));completed+=1;physical+=60;episode_reward[i]=0;episode_steps[i]=0
            buffer.append((batch,action,log,value,torch.tensor(rewards,dtype=torch.float32),torch.tensor(dones,dtype=torch.bool)));steps+=workers;bar.update(workers);obs=[env.reset() if dones[i] and steps<total else following[i] for i,env in enumerate(envs)]
            if steps%rollout:continue
            flat={k:torch.cat([row[0][k] for row in buffer]) for k in batch};actions=torch.cat([r[1] for r in buffer]);old_log=torch.cat([r[2] for r in buffer]);old_value=torch.cat([r[3] for r in buffer]);returns=base.mc_returns(torch.stack([r[4] for r in buffer]),torch.stack([r[5] for r in buffer])).flatten()
            with torch.no_grad():
                check_log,_,check_value=model.evaluate(flat,actions);replay=base.behavior_replay_audit(check_log,old_log,check_value,old_value,cfg)
            maximum_replay=max(maximum_replay,replay['behavior_normalized_max_error']);before_cache_parameters=base.parameter_digest(model);cache=baseline_cache(model,flat,actions,old_value);seal=cache_digest(cache);advantage,scale=prepare_actor_advantage(returns,old_value,cache,model.algorithm)
            if before_cache_parameters!=base.parameter_digest(model) or advantage.requires_grad or (cache is not None and any(t.requires_grad for t in cache.values())):raise RuntimeError('behavior baseline must be detached before optimizer')
            before_actor=[p.detach().clone() for p in model.actor.parameters()];critic_parameters=[p for name,p in model.named_parameters() if not name.startswith('actor.')];before_critic=[p.detach().clone() for p in critic_parameters];losses=[];norms=[]
            for epoch in range(cfg['epochs']):
                permutation=torch.randperm(len(returns),generator=mini_rng)
                for ids in permutation.split(cfg['minibatch']):
                    guard();mini={k:v[ids] for k,v in flat.items()};current_log,entropy,current_value=model.evaluate(mini,actions[ids]);payload=None if cache is None else dict(contexts=cache['contexts'][ids]);objective=actor_objective(current_log,old_log[ids],advantage[ids],model.algorithm,cfg['clip'],getattr(model,'k',2));value_loss=critic_loss(model,mini,returns[ids],current_value,payload,cfg['value_loss_normalizer']);loss=-objective-cfg['entropy_weight']*entropy.mean()+cfg['value_weight']*value_loss
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite PPO-derived loss')
                    optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                    if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
                    optimizer.step();opts+=1;losses.append(float(loss.detach()));norms.append(float(norm))
            if cache_digest(cache)!=seal:raise RuntimeError('baseline cache mutated/recomputed during PPO epochs')
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('nonfinite parameters')
            actor_change=float(torch.sqrt(sum((p.detach()-q).square().sum() for p,q in zip(model.actor.parameters(),before_actor))));critic_change=float(torch.sqrt(sum((p.detach()-q).square().sum() for p,q in zip(critic_parameters,before_critic))));updates+=1
            diagnostics={} if cache is None else dict(baseline_cache_sha256=seal,cache_phase='before_current_batch_optimizer',baseline_context_sensitivity_over_sd=float(cache['context_sensitivity']/scale),baseline_delta_over_sd=float((cache['baseline']-old_value[:,None]).abs().mean()/scale),fallback_mass=float(cache['fallback_mass']),weight_normalization_error=float(cache['weight_error']),removal_residual_variance=float((returns[:,None]-cache['baseline']).var(unbiased=False)),state_residual_variance=float((returns-old_value).var(unbiased=False)))
            emit('training_progress',dict(algorithm=model.algorithm,train_seed=seed,learned_steps=steps,physical_ticks=physical,training_episodes=completed,optimizer_steps=opts,update_index=updates,loss=float(np.mean(losses)),preclip_global_gradient_norm=float(np.mean(norms)),actor_update_norm=actor_change,critic_update_norm=critic_change,group_updates_scope='rollout net displacement',parameter_sha256=base.parameter_digest(model),**replay,**diagnostics));buffer.clear()
            if steps in cfg['checkpoints']:callback(model,steps);emitted.append(steps)
        expected_opts=total//rollout*cfg['epochs']*(rollout//cfg['minibatch'])
        if buffer or episode_steps.any() or emitted!=cfg['checkpoints'] or opts!=expected_opts or completed!=total//48 or physical!=total//48*60 or initial==base.parameter_digest(model):raise RuntimeError('incomplete training/budget/update contract')
        return dict(algorithm=model.algorithm,train_seed=seed,from_scratch=True,pretraining_steps=0,learned_steps=steps,training_episodes=completed,physical_ticks=physical,optimizer_steps=opts,updates=updates,checkpoints=emitted,changed=True,parameter_sha256=base.parameter_digest(model),behavior_normalized_max_error=maximum_replay,behavior_baselines_cached_before_current_batch_fit=True,elapsed_seconds=time.monotonic()-started)
    finally:bar.close()
