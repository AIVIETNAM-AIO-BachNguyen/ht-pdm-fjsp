"""Feedforward masked PPO/MAPPO and Double-Q IQL/QMIX for the fixed setting.

Decentralized policies propose actions; the environment owns resource arbitration.
Training uses complete 48-decision episodes, including terminal common-tail cost.
There is no recurrent state, future-event input, or evaluation RNG mutation.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

VERSION = "maintenance_balanced_algorithm_benchmark_v1"
ALGORITHMS = ("ppo", "mappo", "iql", "qmix")


def role_seed(seed, role):
    return int.from_bytes(hashlib.sha256(f"{seed}:{VERSION}:{role}".encode()).digest()[:8], "big")


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs))


def tensors(observations):
    if isinstance(observations, dict):
        return {k: torch.as_tensor(v, dtype=torch.bool if k.endswith("mask") else torch.float32)
            for k, v in observations.items()}
    return tensors({key: np.stack([o[key] for o in observations]) for key in observations[0]})


def masked(logits, mask):
    if logits.shape != mask.shape or not mask.any(-1).all():
        raise ValueError("nonempty legal action mask with matching logits required")
    if not torch.isfinite(logits).all():
        raise RuntimeError("nonfinite policy logits/Q values")
    return logits.masked_fill(~mask, -1e9)


class Mixer(nn.Module):
    """State-conditioned monotone QMIX mixer with nonnegative hyperweights."""
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.w1 = nn.Linear(71, 4 * hidden)
        self.b1 = nn.Linear(71, hidden)
        self.w2 = nn.Linear(71, hidden)
        self.bias = mlp(71, hidden, 1)

    def forward(self, agent_values, state):
        count = len(state)
        w1 = self.w1(state).abs().reshape(count, 4, self.hidden)
        w2 = self.w2(state).abs().reshape(count, self.hidden, 1)
        hidden = torch.nn.functional.elu(torch.bmm(agent_values[:, None], w1)
            + self.b1(state)[:, None])
        return torch.bmm(hidden, w2).reshape(count) + self.bias(state).reshape(count)


class Model(nn.Module):
    def __init__(self, algorithm, hidden=64, mixer_hidden=32):
        super().__init__()
        if algorithm not in ALGORITHMS or min(hidden, mixer_hidden) < 1:
            raise ValueError("unknown algorithm or invalid network width")
        self.algorithm = algorithm
        self.hidden = hidden
        self.mixer_hidden = mixer_hidden
        if algorithm == "ppo":
            self.actor = mlp(71, hidden, 21)
        elif algorithm == "mappo":
            self.actor = mlp(33, hidden, 3)
        elif algorithm == "iql":
            self.agents = nn.ModuleList(mlp(33, hidden, 3) for _ in range(4))
        else:
            self.agents = nn.ModuleList([mlp(33, hidden, 3)])
            self.mixer = Mixer(mixer_hidden)
        if algorithm in ("ppo", "mappo"):
            self.critic = mlp(71, hidden, 1)

    def q_values(self, batch):
        if self.algorithm == "iql":
            return torch.stack([net(batch["local"][:, i]) for i, net in enumerate(self.agents)], dim=1)
        if self.algorithm == "qmix":
            return self.agents[0](batch["local"])
        raise ValueError("Q values require value learner")

    def policy(self, batch):
        if self.algorithm == "ppo":
            return masked(self.actor(batch["state"]), batch["joint_mask"])
        if self.algorithm == "mappo":
            return masked(self.actor(batch["local"]), batch["local_mask"])
        raise ValueError("policy logits require actor learner")

    def evaluate(self, batch, actions):
        logits = self.policy(batch)
        mask = batch["joint_mask"] if self.algorithm == "ppo" else batch["local_mask"]
        if actions.shape != logits.shape[:-1] or not mask.gather(-1, actions[..., None]).all():
            raise ValueError("illegal policy action or action dimensions")
        log = torch.log_softmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -(probabilities * log).sum(-1)
        return log.gather(-1, actions[..., None]).squeeze(-1), entropy, self.critic(batch["state"]).squeeze(-1)

    @torch.no_grad()
    def sample(self, batch, generator=None, deterministic=False, epsilon=0):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon outside [0,1]")
        if self.algorithm in ("ppo", "mappo"):
            logits = self.policy(batch)
            if deterministic:
                action = logits.argmax(-1)
            else:
                if generator is None:
                    raise ValueError("explicit action RNG required")
                shape = logits.shape[:-1]
                action = torch.multinomial(torch.softmax(logits, -1).reshape(-1, logits.shape[-1]),
                    1, generator=generator).reshape(shape)
            log, _, value = self.evaluate(batch, action)
            return action, log, value
        mask = batch["local_mask"]
        q = masked(self.q_values(batch), mask)
        action = q.argmax(-1)
        if not deterministic and epsilon:
            if generator is None:
                raise ValueError("explicit action RNG required")
            explore = torch.rand(action.shape, generator=generator) < epsilon
            random_actions = torch.multinomial(mask.float().reshape(-1, 3), 1,
                generator=generator).reshape(action.shape)
            action = torch.where(explore, random_actions, action)
        return action, None, None

    @torch.no_grad()
    def select(self, observation):
        batch = tensors({key: np.asarray(value)[None] for key, value in observation.items()})
        action, _, _ = self.sample(batch, deterministic=True)
        result = action[0].cpu().numpy()
        return int(result) if self.algorithm == "ppo" else result


def make_model(algorithm, hidden=64, mixer_hidden=32):
    return Model(algorithm, hidden, mixer_hidden)


def parameter_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def mc_returns(reward, done, gamma=1.):
    if reward.shape != done.shape or not done[-1].all():
        raise ValueError("complete terminal episodes required for MC returns")
    result = torch.zeros_like(reward)
    future = torch.zeros(reward.shape[1])
    for tick in reversed(range(len(reward))):
        future = reward[tick] + gamma * future * (~done[tick])
        result[tick] = future
    return result


class Replay:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("positive replay capacity required")
        self.capacity = capacity; self.count = 0; self.cursor = 0; self.storage = None

    def add(self, batch, action, reward, following, done):
        data = {**{"obs_" + key: value.numpy() for key, value in batch.items()},
            **{"next_" + key: value.numpy() for key, value in following.items()},
            "action": action.numpy(), "reward": np.asarray(reward, dtype=np.float32),
            "done": np.asarray(done, dtype=bool)}
        if self.storage is None:
            self.storage = {key: np.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
                for key, value in data.items()}
        for index in range(len(action)):
            for key, value in data.items():
                self.storage[key][self.cursor] = value[index]
            self.cursor = (self.cursor + 1) % self.capacity
            self.count = min(self.capacity, self.count + 1)

    def sample(self, size, generator):
        if self.count < size:
            raise ValueError("insufficient replay population")
        indices = generator.integers(self.count, size=size)
        return {key: torch.from_numpy(value[indices]) for key, value in self.storage.items()}


def td_loss(model, target, data, gamma):
    observation = {key[4:]: value for key, value in data.items() if key.startswith("obs_")}
    following = {key[5:]: value for key, value in data.items() if key.startswith("next_")}
    if not observation["local_mask"].gather(-1, data["action"][..., None]).all():
        raise ValueError("illegal replay action")
    chosen = model.q_values(observation).gather(-1, data["action"][..., None]).squeeze(-1)
    with torch.no_grad():
        legal = following["local_mask"].clone()
        legal[data["done"]] = True  # Terminal masks may be all-false; never bootstrap them.
        next_action = masked(model.q_values(following), legal).argmax(-1)
        next_q = target.q_values(following).gather(-1, next_action[..., None]).squeeze(-1)
        if model.algorithm == "qmix":
            future = target.mixer(next_q, following["state"])
            expected = data["reward"] + gamma * torch.where(data["done"], 0., future)
        else:
            expected = data["reward"][:, None] + gamma * torch.where(data["done"][:, None], 0., next_q)
    predicted = model.mixer(chosen, observation["state"]) if model.algorithm == "qmix" else chosen
    return torch.nn.functional.smooth_l1_loss(predicted, expected), predicted.detach(), expected


def fit(model, cfg, seed, make_env, journals=None, on_checkpoint=None, timeout=None):
    """Train on exact step budget; callback(model, steps, audit) after updates.

    Environment factory initializes streams. ``reset()`` advances the next whole
    episode; ``step`` returns obs/reward/done/info. Terminal info must expose
    ``episode_metrics`` or ``finalepisode_metrics``. Journal mapping values have
    ``add(dict)``. Rewards already include scaling and tail accounting.
    """
    journals = journals or {}
    total = int(cfg["env_steps"]); workers = int(cfg["vector_envs"])
    checkpoints = list(cfg["checkpoints"])
    if total <= 0 or workers < 1 or total % (48 * workers) or cfg["gamma"] != 1.:
        raise ValueError("whole 48-step episodes and gamma1 required")
    if checkpoints != sorted(set(checkpoints)) or any(s <= 0 or s > total or s % workers for s in checkpoints):
        raise ValueError("sorted distinct aligned checkpoint steps required")
    actor = model.algorithm in ("ppo", "mappo")
    rollout = int(cfg["rollout"])
    if min(rollout, cfg["epochs"], cfg["minibatch"], cfg["learning_rate"], cfg["gradient_clip"]) <= 0:
        raise ValueError("positive update settings required")
    if actor and (rollout % (48 * workers) or total % rollout or any(s % rollout for s in checkpoints)):
        raise ValueError("actor rollouts/checkpoints must contain complete episodes")
    if not actor and (min(cfg["learning_starts"], cfg["batch_size"], cfg["train_frequency"], cfg["target_interval"]) <= 0
        or cfg["replay_capacity"] < cfg["batch_size"]):
        raise ValueError("positive replay/update settings and sufficient capacity required")
    if not actor and (any(s < max(cfg["learning_starts"], cfg["batch_size"])
        or s % cfg["train_frequency"] for s in checkpoints)
        or cfg["train_frequency"] % workers or cfg["target_interval"] % workers):
        raise ValueError("value update/checkpoint alignment")
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    action_rng = torch.Generator().manual_seed(role_seed(seed, "actions"))
    minibatch_rng = torch.Generator().manual_seed(role_seed(seed, "minibatches"))
    replay_rng = np.random.default_rng(role_seed(seed, "replay"))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    initial = parameter_digest(model)
    target = copy.deepcopy(model) if not actor else None
    replay = Replay(int(cfg.get("replay_capacity", 1))) if not actor else None
    envs = [make_env(seed, worker) for worker in range(workers)]
    observations = [env.reset() for env in envs]
    episode_rewards = np.zeros(workers); episode_steps = np.zeros(workers, dtype=int)
    completed = physical_ticks = optimizer_steps = steps = update_index = 0
    actor_buffer = []; emitted = []; behavior_error = 0.; model.train()

    def emit(name, record):
        if name in journals:
            journals[name].add(record)

    def guard():
        if timeout:
            timeout()

    def optimize(loss):
        nonlocal optimizer_steps
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        optimizer.zero_grad(); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite training gradient")
        optimizer.step(); optimizer_steps += 1
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("nonfinite updated parameters")
        return float(loss.detach())

    progress = tqdm(total=total, desc=f"Train {model.algorithm} seed {seed}", unit="step")
    try:
        while steps < total:
            guard()
            batch = tensors(observations)
            fraction = min(1., steps / max(1., total * cfg.get("epsilon_fraction", .5)))
            epsilon = cfg.get("epsilon_start", 1.) + fraction * (cfg.get("epsilon_end", .05) - cfg.get("epsilon_start", 1.))
            action, log, value = model.sample(batch, action_rng, epsilon=epsilon)
            next_observations = []; rewards = []; dones = []
            for worker, env in enumerate(envs):
                submitted = int(action[worker]) if model.algorithm == "ppo" else action[worker].numpy().copy()
                nxt, reward, done, info = env.step(submitted)
                if not math.isfinite(reward):
                    raise RuntimeError("nonfinite environment reward")
                next_observations.append(nxt); rewards.append(float(reward)); dones.append(bool(done))
                episode_rewards[worker] += reward; episode_steps[worker] += 1
                if done:
                    if episode_steps[worker] != 48:
                        raise RuntimeError("unexpected physical episode decision count")
                    metrics = info.get("episode_metrics", info.get("finalepisode_metrics"))
                    if not isinstance(metrics, dict):
                        raise RuntimeError("terminal episode metrics required")
                    if metrics.get("physical_ticks", 60) != 60:
                        raise RuntimeError("common physical tail missing")
                    emit("training_episodes", dict(metrics, algorithm=model.algorithm,
                        train_seed=seed, worker=worker, learned_steps=steps + workers,
                        learned_episode_steps=48, episode_reward=float(episode_rewards[worker]), physical_ticks=60))
                    completed += 1; physical_ticks += 60
                    episode_rewards[worker] = 0.; episode_steps[worker] = 0
            reward_tensor = torch.tensor(rewards); done_tensor = torch.tensor(dones)
            if actor:
                actor_buffer.append((batch, action, log, value, reward_tensor, done_tensor))
            else:
                replay.add(batch, action, rewards, tensors(next_observations), dones)
            observations = [env.reset() if dones[i] and steps + workers < total
                else next_observations[i] for i, env in enumerate(envs)]
            steps += workers; progress.update(workers)
            losses = []; updated = False
            if actor and steps % rollout == 0:
                flat = {key: torch.cat([row[0][key] for row in actor_buffer]) for key in batch}
                actions_ = torch.cat([row[1] for row in actor_buffer])
                old_log = torch.cat([row[2] for row in actor_buffer])
                old_value = torch.cat([row[3] for row in actor_buffer])
                returns = mc_returns(torch.stack([row[4] for row in actor_buffer]),
                    torch.stack([row[5] for row in actor_buffer])).reshape(-1)
                with torch.no_grad():
                    check_log, _, check_value = model.evaluate(flat, actions_)
                    error = max(float((check_log - old_log).abs().max()), float((check_value - old_value).abs().max()))
                    behavior_error = max(behavior_error, error)
                    if error > 2e-6:
                        raise RuntimeError("sampled/re-evaluated behavior policy mismatch")
                advantage = returns - old_value
                advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
                for epoch in range(int(cfg["epochs"])):
                    permutation = torch.randperm(len(returns), generator=minibatch_rng)
                    for indices in permutation.split(int(cfg["minibatch"])):
                        guard()
                        mini = {key: val[indices] for key, val in flat.items()}
                        current_log, entropy, current_value = model.evaluate(mini, actions_[indices])
                        ratio = torch.exp(current_log - old_log[indices])
                        adv = advantage[indices] if model.algorithm == "ppo" else advantage[indices, None]
                        objective = torch.minimum(ratio * adv,
                            ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * adv).mean()
                        loss = (-objective - cfg["entropy_weight"] * entropy.mean()
                            + cfg["value_weight"] * torch.nn.functional.mse_loss(current_value, returns[indices]))
                        losses.append(optimize(loss))
                actor_buffer.clear(); updated = True
            elif not actor and steps >= cfg["learning_starts"] and steps % cfg["train_frequency"] == 0 and replay.count >= cfg["batch_size"]:
                data = replay.sample(int(cfg["batch_size"]), replay_rng)
                loss, _, expected = td_loss(model, target, data, cfg["gamma"])
                if not torch.isfinite(expected).all():
                    raise RuntimeError("nonfinite Double-Q target")
                losses.append(optimize(loss)); updated = True
                if steps % cfg["target_interval"] == 0:
                    target.load_state_dict(model.state_dict())
            if updated:
                update_index += 1
                digest = parameter_digest(model)
                record = dict(algorithm=model.algorithm, train_seed=seed, learned_steps=steps,
                    physical_ticks=physical_ticks + int(episode_steps.sum()), training_episodes=completed,
                    optimizer_steps=optimizer_steps, update_index=update_index, loss=float(np.mean(losses)),
                    epsilon=0. if actor else epsilon, parameter_sha256=digest, behavior_max_error=behavior_error)
                emit("training_progress", record)
                if steps in checkpoints:
                    if on_checkpoint:
                        on_checkpoint(model, steps, record.copy())
                    emitted.append(steps)
        require_changed = parameter_digest(model) != initial
        if actor_buffer or episode_steps.any() or emitted != checkpoints or not optimizer_steps or not require_changed:
            raise RuntimeError("incomplete training/checkpoints or unchanged parameters")
        expected_updates = ((total // rollout) * int(cfg["epochs"]) * math.ceil(rollout / cfg["minibatch"])
            if actor else sum(s >= max(cfg["learning_starts"], cfg["batch_size"])
                and s % cfg["train_frequency"] == 0 for s in range(workers, total + 1, workers)))
        if optimizer_steps != expected_updates:
            raise RuntimeError("optimizer update accounting mismatch")
        return dict(algorithm=model.algorithm, train_seed=seed, learned_steps=steps,
            physical_ticks=physical_ticks, training_episodes=completed, optimizer_steps=optimizer_steps,
            updates=update_index, checkpoints=emitted, changed=True, behavior_max_error=behavior_error,
            parameter_sha256=parameter_digest(model), partial_episode_steps=int(episode_steps.sum()))
    finally:
        progress.close()
