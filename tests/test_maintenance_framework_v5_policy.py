"""Post-decision actor arithmetic, matching structure, symmetry and PPO IO."""
from __future__ import annotations

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from ht_pdm_fjsp.maintenance_framework_v1 import FrameworkEnv, catalogue
from ht_pdm_fjsp.maintenance_framework_v5_policy import (
    ACTIONS, PostdecisionCapacityActor, PostdecisionCapacityPolicy,
)


def probe_observation():
    value = np.zeros(56, dtype=np.float32)
    value[:2] = [.25, .75]
    machine = value[2:34].reshape(4, 8)
    machine[:, 0] = 3 / 8
    machine[:3, 2] = 1  # three queued requests
    worker = value[34:40].reshape(2, 3)
    worker[:, 0] = 1
    edge = value[40:56].reshape(4, 2, 2)
    edge[..., 0] = 1
    edge[..., 1] = np.asarray([[2, 6], [6, 2], [3, 4], [6, 6]]) / 12
    return value


def permute(value, machine_order, worker_order):
    result = value.copy()
    result[2:34] = value[2:34].reshape(4, 8)[machine_order].reshape(-1)
    workers = value[34:40].reshape(2, 3)[worker_order].copy()
    for worker in workers:
        old_machine = round(float(worker[2]) * 5) - 1
        worker[2] = 0 if old_machine < 0 else (machine_order.index(old_machine) + 1) / 5
    result[34:40] = workers.reshape(-1)
    result[40:56] = value[40:56].reshape(4, 2, 2)[machine_order][:, worker_order].reshape(-1)
    return result


def test_incidence_wait_and_exact_catalogue():
    actor = PostdecisionCapacityActor("postdecision")
    assert ACTIONS == catalogue(4, 2)
    assert ACTIONS[0] == () and actor.incidence.shape == (21, 8)
    for row, matching in zip(actor.incidence, ACTIONS):
        expected = torch.zeros(4, 2)
        for m, j in matching:
            expected[m, j] = 1
        assert torch.equal(row.reshape(4, 2), expected)


def test_residual_wait_is_action_conditioned_and_uses_real_ticks():
    actor = PostdecisionCapacityActor("postdecision")
    for parameter in actor.potential_network.parameters():
        torch.nn.init.zeros_(parameter)
    observation = torch.tensor(probe_observation()[None])
    _, machine, worker, edge = actor.split_observation(observation)
    penalty = actor.postdecision_penalty(machine, worker, edge)[0]
    one = ACTIONS.index(((0, 0),))
    two = ACTIONS.index(((2, 1),))
    both = ACTIONS.index(((0, 0), (2, 1)))
    assert penalty[0] == 0
    assert penalty[one] == 0 and penalty[two] == 0
    expected = torch.nn.functional.softplus(actor.log_opportunity_scale) * np.log(2) * 2
    torch.testing.assert_close(penalty[both], expected, atol=1e-6, rtol=1e-6)
    # The full matching blocks both workers, while either assignment alone
    # leaves another immediately available for the unserved request.
    assert penalty[both] > penalty[one] + penalty[two]


def test_structured_arms_share_initial_parameters_and_postdecision_gradient():
    torch.manual_seed(17)
    edge = PostdecisionCapacityActor("edge_only")
    torch.manual_seed(17)
    additive = PostdecisionCapacityActor("additive_queue")
    torch.manual_seed(17)
    post = PostdecisionCapacityActor("postdecision")
    assert all(torch.equal(edge.state_dict()[name], additive.state_dict()[name])
               and torch.equal(edge.state_dict()[name], post.state_dict()[name])
               for name in edge.state_dict())
    logits = post(torch.tensor(probe_observation()[None]))
    logits[0, ACTIONS.index(((0, 0), (2, 1)))].backward()
    assert post.log_opportunity_scale.grad is not None
    assert post.log_opportunity_scale.grad.abs() > 0


def test_actor_logits_remap_under_machine_worker_permutation():
    machine_order, worker_order = [2, 0, 3, 1], [1, 0]
    original = probe_observation()
    original[34:40].reshape(2, 3)[0] = [0, 3/12, 2/5]  # busy on old machine 1
    relabeled = permute(original, machine_order, worker_order)
    assert np.isclose(relabeled[34:40].reshape(2, 3)[1, 2], 4/5)
    for variant in ("edge_only", "additive_queue", "postdecision"):
        torch.manual_seed(22)
        actor = PostdecisionCapacityActor(variant)
        a = actor(torch.tensor(original[None]))[0]
        b = actor(torch.tensor(relabeled[None]))[0]
        for old_index, matching in enumerate(ACTIONS):
            new_matching = tuple(sorted((machine_order.index(m), worker_order.index(j))
                                        for m, j in matching))
            new_index = ACTIONS.index(new_matching)
            torch.testing.assert_close(a[old_index], b[new_index], atol=1e-6, rtol=1e-6)


def test_scratch_ppo_learns_reload_and_respects_mask(tmp_path):
    env = FrameworkEnv()
    model = MaskablePPO(PostdecisionCapacityPolicy, env,
                        policy_kwargs={"variant": "postdecision"},
                        n_steps=32, batch_size=32, n_epochs=1,
                        device="cpu", seed=66000000, verbose=0)
    before = [p.detach().clone() for p in model.policy.action_net.parameters()]
    model.learn(total_timesteps=32)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.policy.action_net.parameters()))
    path = tmp_path / "v5.zip"
    model.save(path)
    loaded = MaskablePPO.load(path, device="cpu")
    obs, _ = env.reset(seed=66500000)
    mask = env.action_masks()
    a = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
    b = int(loaded.predict(obs, action_masks=mask, deterministic=True)[0])
    assert a == b and mask[a]


def test_additive_queue_exactly_reproduces_v4_queue_actor():
    from ht_pdm_fjsp.maintenance_framework_v4_policy import CapacityOpportunityActor
    torch.manual_seed(31)
    old = CapacityOpportunityActor("current_queue")
    torch.manual_seed(31)
    new = PostdecisionCapacityActor("additive_queue")
    obs = torch.tensor(probe_observation()[None])
    torch.testing.assert_close(old(obs), new(obs), atol=1e-7, rtol=1e-7)
