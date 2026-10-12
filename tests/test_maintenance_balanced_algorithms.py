"""Masks, actor locality, monotone mixing, exact returns and learner updates."""
import copy
import numpy as np
import pytest
import torch

from ht_pdm_fjsp import maintenance_balanced_algorithms as a


def observation(batch=3):
    rng = np.random.default_rng(817)
    mask = np.ones((batch, 4, 3), bool); mask[:, 0, 1:] = False
    joint = np.zeros((batch, 21), bool); joint[:, [0, 2, 7]] = True
    return dict(state=torch.tensor(rng.normal(size=(batch, 71)), dtype=torch.float32),
        local=torch.tensor(rng.normal(size=(batch, 4, 33)), dtype=torch.float32),
        local_mask=torch.tensor(mask), joint_mask=torch.tensor(joint))


@pytest.mark.parametrize("algorithm", a.ALGORITHMS)
def test_actions_are_masked_reproducible_and_deterministic_selection_preserves_rng(algorithm):
    model = a.make_model(algorithm, hidden=8, mixer_hidden=4)
    obs = observation(); first = torch.Generator().manual_seed(51); second = torch.Generator().manual_seed(51)
    action, log, value = model.sample(obs, first, epsilon=1.)
    assert torch.equal(action, model.sample(obs, second, epsilon=1.)[0])
    mask = obs["joint_mask"] if algorithm == "ppo" else obs["local_mask"]
    assert mask.gather(-1, action[..., None]).all()
    if algorithm in ("ppo", "mappo"):
        evaluated, entropy, evaluated_value = model.evaluate(obs, action)
        torch.testing.assert_close(evaluated, log)
        torch.testing.assert_close(evaluated_value, value)
        assert torch.isfinite(entropy).all() and (entropy >= 0).all()
    raw = {k: v[0].numpy() for k, v in obs.items()}
    rng_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    selected = model.select(raw)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert np.array_equal(selected, model.select(raw))


def test_decentralized_actors_ignore_central_state_and_peer_agent_features():
    obs = observation(); changed = copy.deepcopy(obs)
    changed["state"] += 100; changed["local"][:, 1:] += 20
    mappo = a.make_model("mappo", hidden=8)
    torch.testing.assert_close(mappo.policy(obs)[:, 0], mappo.policy(changed)[:, 0])
    for algorithm in ("iql", "qmix"):
        model = a.make_model(algorithm, hidden=8)
        torch.testing.assert_close(model.q_values(obs)[:, 0], model.q_values(changed)[:, 0])
    assert len(a.make_model("iql", hidden=8).agents) == 4
    assert len(a.make_model("qmix", hidden=8).agents) == 1


def test_qmix_mixer_is_monotone_and_state_conditioned():
    mixer = a.Mixer(8); state = observation()["state"]
    q = torch.randn(3, 4, requires_grad=True)
    output = mixer(q, state)
    derivative, = torch.autograd.grad(output.sum(), q)
    assert (derivative >= -1e-7).all()
    assert (mixer(q + 1, state) >= output - 1e-7).all()
    assert not torch.allclose(mixer(q, state), mixer(q, state + 1))


def test_exact_complete_episode_mc_return_includes_tail_and_resets_at_done():
    rewards = torch.tensor([[1., 2.], [4., 8.], [3., 7.], [5., 6.]])
    done = torch.tensor([[False, False], [True, True], [False, False], [True, True]])
    expected = torch.tensor([[5., 10.], [4., 8.], [8., 13.], [5., 6.]])
    torch.testing.assert_close(a.mc_returns(rewards, done), expected)
    with pytest.raises(ValueError, match="complete"):
        a.mc_returns(rewards, torch.zeros_like(done))


class ConstantQ(torch.nn.Module):
    def __init__(self, algorithm, values):
        super().__init__(); self.algorithm = algorithm
        self.values = torch.tensor(values, dtype=torch.float32)

    def q_values(self, batch):
        return self.values.expand(len(batch["state"]), 4, 3)

    def mixer(self, values, state):
        return values.sum(-1)


@pytest.mark.parametrize("algorithm", ("iql", "qmix"))
def test_double_q_online_selection_target_values_and_terminal_mask(algorithm):
    obs = observation(2); nxt = copy.deepcopy(obs)
    nxt["local_mask"][:] = True; nxt["local_mask"][1] = False
    data = {**{"obs_" + k: v for k, v in obs.items()}, **{"next_" + k: v for k, v in nxt.items()},
        "action": torch.zeros((2, 4), dtype=torch.long), "reward": torch.tensor([2., -3.]),
        "done": torch.tensor([False, True])}
    # Online argmax1, target argmax2: bootstrap must use target value at action1.
    online = ConstantQ(algorithm, [0., 9., 4.]); target = ConstantQ(algorithm, [0., 3., 99.])
    loss, _, expected = a.td_loss(online, target, data, 1.)
    assert torch.isfinite(loss)
    if algorithm == "qmix":
        torch.testing.assert_close(expected, torch.tensor([14., -3.]))
    else:
        torch.testing.assert_close(expected, torch.tensor([[5.] * 4, [-3.] * 4]))


class TinyWholeEpisode:
    def __init__(self, seed, worker):
        self.seed = seed; self.worker = worker; self.index = -1

    def reset(self):
        self.index += 1; self.tick = 0
        return self.obs()

    def obs(self):
        return {key: value[0].numpy().copy() for key, value in observation(1).items()}

    def step(self, action):
        self.tick += 1
        done = self.tick == 48
        reward = -1. - (.5 if np.any(action) else 0.) - (12 if done else 0.)
        return self.obs(), reward, done, dict(episode_metrics=dict(
            physical_seed=self.seed + self.worker * 100 + self.index,
            episode_index=self.index, stream_seed=self.seed + self.worker, physical_ticks=60))


def config():
    return dict(env_steps=384, vector_envs=2, rollout=192, epochs=2, minibatch=48,
        checkpoints=[192, 384], gamma=1., learning_rate=3e-4, clip=.2,
        entropy_weight=.01, value_weight=.5, gradient_clip=.5, torch_threads=1,
        replay_capacity=512, learning_starts=96, batch_size=32, train_frequency=2,
        target_interval=96, epsilon_start=1., epsilon_end=.05, epsilon_fraction=.5)


class Sink:
    def __init__(self):
        self.rows = []

    def add(self, row):
        self.rows.append(row)


@pytest.mark.parametrize("algorithm", a.ALGORITHMS)
def test_complete_unit_training_updates_checkpoints_journals_and_determinism(algorithm):
    with torch.random.fork_rng():
        torch.manual_seed(187); model = a.make_model(algorithm, hidden=8, mixer_hidden=4)
    clone = copy.deepcopy(model)
    sinks = {"training_episodes": Sink(), "training_progress": Sink()}; checkpoints = []
    def checkpoint(m, steps, audit):
        checkpoints.append((steps, audit["optimizer_steps"], a.parameter_digest(m)))
    result = a.fit(model, config(), 631, TinyWholeEpisode, sinks, checkpoint)
    other = a.fit(clone, config(), 631, TinyWholeEpisode)
    assert result == other and a.parameter_digest(model) == a.parameter_digest(clone)
    assert result["learned_steps"] == 384 and result["training_episodes"] == 8
    assert result["physical_ticks"] == 480 and result["partial_episode_steps"] == 0
    assert [row[0] for row in checkpoints] == [192, 384]
    assert all(row[1] > 0 for row in checkpoints) and result["changed"]
    assert len(sinks["training_episodes"].rows) == 8 and sinks["training_progress"].rows
    assert all(row["learned_episode_steps"] == 48 and row["physical_ticks"] == 60
        for row in sinks["training_episodes"].rows)
    if algorithm in ("ppo", "mappo"):
        assert result["optimizer_steps"] == 16 and result["behavior_max_error"] < 2e-6
    else:
        assert result["optimizer_steps"] == 145


def test_invalid_masks_and_budget_are_rejected():
    model = a.make_model("ppo", hidden=8); obs = observation()
    obs["joint_mask"][:] = False
    with pytest.raises(ValueError, match="mask"):
        model.sample(obs, deterministic=True)
    with pytest.raises(ValueError, match="whole"):
        a.fit(model, {**config(), "env_steps": 385}, 1, TinyWholeEpisode)
    with pytest.raises(ValueError, match="rollouts"):
        a.fit(model, {**config(), "rollout": 128}, 1, TinyWholeEpisode)
