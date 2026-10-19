"""Method-v4 actor invariants: matching structure, temporal gate, and PPO IO."""
from __future__ import annotations

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from ht_pdm_fjsp.maintenance_framework_v1 import FrameworkEnv, catalogue
from ht_pdm_fjsp.maintenance_framework_v4_policy import (
    ACTIONS, CapacityOpportunityActor, CapacityOpportunityPolicy,
)


def probe_observation():
    value = np.zeros((56,), dtype=np.float32)
    value[:2] = [0.25, 0.75]
    machine = value[2:34].reshape(4, 8)
    machine[0] = [3 / 8, 0, 1, 0, 0.05, 0, 0, 0]
    machine[1] = [2 / 8, 0, 0, 0, 0, 0, 0, 0]
    machine[2] = [6 / 8, 1, 1, 1, 0.2, 0, 0, 0.1]
    machine[3] = [1 / 8, 0, 0, 0, 0, 0, 0, 0]
    worker = value[34:40].reshape(2, 3)
    worker[:] = [[1, 0, 0.2], [1, 0, 0.4]]
    edge = value[40:56].reshape(4, 2, 2)
    edge[..., 0] = [[1, 1], [1, 0], [1, 1], [0, 1]]
    edge[..., 1] = [[2/12, 6/12], [3/12, 0], [6/12, 2/12], [0, 4/12]]
    return value


def permute(value, machine_order, worker_order):
    result = value.copy()
    result[2:34] = value[2:34].reshape(4, 8)[machine_order].reshape(-1)
    result[34:40] = value[34:40].reshape(2, 3)[worker_order].reshape(-1)
    result[40:56] = value[40:56].reshape(4, 2, 2)[machine_order][:, worker_order].reshape(-1)
    return result


def test_action_incidence_includes_wait_and_exact_matching_catalogue():
    actor = CapacityOpportunityActor("temporal")
    assert ACTIONS == catalogue(4, 2)
    assert actor.incidence.shape == (21, 8)
    for row, matching in zip(actor.incidence, ACTIONS):
        expected = torch.zeros(4, 2)
        for m, j in matching:
            expected[m, j] = 1
        assert torch.equal(row.reshape(4, 2), expected)


def test_actor_logits_remap_under_machine_and_worker_permutations():
    machine_order = [2, 0, 3, 1]  # new position -> original machine
    worker_order = [1, 0]
    original = probe_observation()
    relabeled = permute(original, machine_order, worker_order)
    for variant in ("edge_only", "current_queue", "temporal"):
        torch.manual_seed(10)
        actor = CapacityOpportunityActor(variant)
        one = actor(torch.tensor(original[None]))
        two = actor(torch.tensor(relabeled[None]))
        for old_index, matching in enumerate(ACTIONS):
            new_matching = tuple(sorted(
                (machine_order.index(m), worker_order.index(j)) for m, j in matching
            ))
            new_index = ACTIONS.index(new_matching)
            torch.testing.assert_close(one[0, old_index], two[0, new_index], atol=1e-6, rtol=1e-6)


def test_temporal_gate_detects_only_near_future_competitor():
    observation = torch.tensor(probe_observation()[None])
    machine = observation[:, 2:34].reshape(1, 4, 8)
    edge = observation[:, 40:56].reshape(1, 4, 2, 2)
    torch.manual_seed(11)
    current = CapacityOpportunityActor("current_queue")
    temporal = CapacityOpportunityActor("temporal")
    temporal.load_state_dict(current.state_dict())
    before = current.opportunity_context(machine, edge)
    after = temporal.opportunity_context(machine, edge)
    # Machine 1 has no request, age2, and only technician 0 can serve it.
    assert after[0, 0, 0] > before[0, 0, 0]
    assert after[0, 0, 1] == before[0, 0, 1]
    lower = observation.clone()
    lower[:, 2 + 8] = 1 / 8  # machine 1 age falls below near-request gate
    low_machine = lower[:, 2:34].reshape(1, 4, 8)
    torch.testing.assert_close(
        temporal.opportunity_context(low_machine, edge)[0, 0],
        current.opportunity_context(low_machine, edge)[0, 0],
    )


def test_scratch_ppo_policy_learns_and_reload_preserves_masked_choice(tmp_path):
    env = FrameworkEnv()
    model = MaskablePPO(
        CapacityOpportunityPolicy, env, policy_kwargs={"variant": "temporal"},
        n_steps=32, batch_size=32, n_epochs=1, device="cpu", seed=65000000,
        verbose=0,
    )
    obs, _ = env.reset(seed=65500000)
    mask = env.action_masks()
    first = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
    assert mask[first]
    before = [value.detach().clone() for value in model.policy.action_net.parameters()]
    model.learn(total_timesteps=32)
    assert any(not torch.equal(a, b) for a, b in
               zip(before, model.policy.action_net.parameters()))
    path = tmp_path / "method_v4.zip"
    model.save(path)
    loaded = MaskablePPO.load(path, device="cpu")
    after = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
    again = int(loaded.predict(obs, action_masks=mask, deterministic=True)[0])
    assert after == again and mask[again]
