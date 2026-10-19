"""Edge-head sharing structure, matched parameter budget and PPO IO."""
from __future__ import annotations

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from ht_pdm_fjsp.maintenance_framework_v1 import FrameworkEnv, catalogue
from ht_pdm_fjsp.maintenance_framework_v6_policy import (
    ACTIONS, EdgeSharingActor, EdgeSharingPolicy,
)
from ht_pdm_fjsp.maintenance_framework_v6_campaign import hierarchical, seed_registry


def probe():
    value = np.zeros(56, dtype=np.float32)
    value[:2] = [.35, .65]
    machines = value[2:34].reshape(4, 8)
    machines[:, 0] = [2/8, 3/8, 5/8, 7/8]
    machines[:, 2] = [0, 1, 1, 1]
    machines[2, 3] = 1
    machines[:, 4] = [0, .1, .2, .3]
    workers = value[34:40].reshape(2, 3)
    workers[:] = [[1, 0, 0], [0, 3/12, 2/5]]
    edge = value[40:56].reshape(4, 2, 2)
    edge[..., 0] = [[1, 1], [1, 1], [1, 0], [0, 1]]
    edge[..., 1] = np.array([[2, 6], [6, 2], [3, 0], [0, 8]]) / 12
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


def test_catalogue_incidence_and_matched_edge_head_parameter_count():
    shared = EdgeSharingActor("shared_edge")
    unshared = EdgeSharingActor("unshared_edge")
    assert ACTIONS == catalogue(4, 2) and ACTIONS[0] == ()
    assert shared.incidence.shape == unshared.incidence.shape == (21, 8)
    assert sum(p.numel() for p in unshared.parameters()) - sum(p.numel() for p in shared.parameters()) == 7
    for index, matching in enumerate(ACTIONS):
        expected = torch.zeros(4, 2)
        for machine, worker in matching:
            expected[machine, worker] = 1
        assert torch.equal(shared.incidence[index].reshape(4, 2), expected)


def test_shared_actor_equivariance_and_unshared_label_dependence():
    order, workers = [2, 0, 3, 1], [1, 0]
    original = probe()
    relabeled = permute(original, order, workers)
    torch.manual_seed(12)
    shared = EdgeSharingActor("shared_edge")
    torch.manual_seed(12)
    unshared = EdgeSharingActor("unshared_edge")
    # Common encoders start with identical weights under the paired seed.
    for name in shared.machine_encoder.state_dict():
        assert torch.equal(shared.machine_encoder.state_dict()[name],
                           unshared.machine_encoder.state_dict()[name])
    a = shared(torch.tensor(original[None]))[0]
    b = shared(torch.tensor(relabeled[None]))[0]
    c = unshared(torch.tensor(original[None]))[0]
    d = unshared(torch.tensor(relabeled[None]))[0]
    differences = []
    for old_index, matching in enumerate(ACTIONS):
        new_matching = tuple(sorted((order.index(m), workers.index(j)) for m, j in matching))
        new_index = ACTIONS.index(new_matching)
        torch.testing.assert_close(a[old_index], b[new_index], atol=1e-6, rtol=1e-6)
        differences.append(abs(float((c[old_index] - d[new_index]).detach())))
    assert max(differences) > 1e-3


def test_shared_matches_v5_edge_only_to_roundoff():
    from ht_pdm_fjsp.maintenance_framework_v5_policy import PostdecisionCapacityActor
    torch.manual_seed(19)
    old = PostdecisionCapacityActor("edge_only")
    torch.manual_seed(19)
    new = EdgeSharingActor("shared_edge")
    torch.testing.assert_close(old(torch.tensor(probe()[None])),
                               new(torch.tensor(probe()[None])), atol=1e-7, rtol=1e-7)


def test_seed_splits_and_hierarchical_bootstrap_constant():
    registry, groups = seed_registry()
    assert registry["campaign_version"] == "maintenance_framework_v6_edge_sharing"
    assert [len(group) for group in groups] == [5, 16, 64, 2, 1, 1]
    assert hierarchical(np.full((5, 64), -2.0), registry["bootstrap_seed"], "probe") == [-2., -2.]


def test_ppo_gradient_save_reload_and_mask(tmp_path):
    env = FrameworkEnv()
    model = MaskablePPO(EdgeSharingPolicy, env,
                        policy_kwargs={"variant": "shared_edge"},
                        n_steps=32, batch_size=32, n_epochs=1,
                        device="cpu", seed=67000000, verbose=0)
    before = [param.detach().clone() for param in model.policy.action_net.parameters()]
    model.learn(total_timesteps=32)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.policy.action_net.parameters()))
    path = tmp_path / "shared.zip"
    model.save(path)
    again = MaskablePPO.load(path, device="cpu")
    obs, _ = env.reset(seed=67500000)
    mask = env.action_masks()
    first = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
    second = int(again.predict(obs, action_masks=mask, deterministic=True)[0])
    assert first == second and mask[first]
