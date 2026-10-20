"""Current-state technician allocator, paired initialization and PPO IO."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, FrameworkEnv, catalogue, configs
from ht_pdm_fjsp.maintenance_framework_v7_campaign import (
    PairedMixtureEnv, V7LearnedController, hierarchical, model_for_arm, seed_registry,
)
from ht_pdm_fjsp.maintenance_framework_v7_policy import canonical_mask


def test_min_duration_per_machine_subset_and_ties():
    actions = catalogue()
    config = SimpleNamespace(service_time=((2, 6), (6, 2), (3, 3), (7, 8)))
    legal = np.zeros(len(actions), dtype=bool)
    for matching in ((), ((0, 0),), ((0, 1),), ((1, 0),), ((1, 1),),
                     ((0, 0), (1, 1)), ((0, 1), (1, 0)), ((2, 0),), ((2, 1),)):
        legal[actions.index(matching)] = True
    kept = canonical_mask(config, actions, legal)
    expected = {(), ((0, 0),), ((1, 1),), ((0, 0), (1, 1)), ((2, 0),), ((2, 1),)}
    assert {actions[i] for i, selected in enumerate(kept) if selected} == expected
    assert np.array_equal(canonical_mask(config, actions, kept), kept)
    assert np.all(~kept | legal)


def test_training_mask_uses_present_legal_actions_only():
    for case in CASES:
        env = FrameworkEnv(configs()[case])
        env.reset(seed=68500000)
        for _ in range(6):
            legal = env.action_masks()
            kept = canonical_mask(env.config, env.actions, legal)
            assert kept[0] and np.all(~kept | legal)
            for index, matching in enumerate(env.actions):
                if not legal[index]:
                    continue
                machines = {machine for machine, _ in matching}
                same_set = [other for j, other in enumerate(env.actions)
                            if legal[j] and {machine for machine, _ in other} == machines]
                duration = sum(env.config.service_time[m][w] for m, w in matching)
                minimum = min(sum(env.config.service_time[m][w] for m, w in other)
                              for other in same_set)
                assert bool(kept[index]) == (duration == minimum)
            action = int(np.flatnonzero(kept)[-1])
            env.step(action)


def test_paired_initial_weights_and_seed_splits():
    registry, groups = seed_registry()
    assert registry["campaign_version"] == "maintenance_framework_v7_canonical_assignment"
    assert [len(group) for group in groups] == [5, 16, 64, 2, 1, 1]
    assert hierarchical(np.full((5, 64), -2.0), registry["bootstrap_seed"], "probe") == [-2., -2.]
    initial = {}
    for arm in ("flat_free", "flat_canonical", "shared_free", "shared_canonical"):
        env = PairedMixtureEnv(68500000, canonical=arm.endswith("_canonical"))
        model = model_for_arm(arm, env, 68000000)
        initial[arm] = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
        env.reset(seed=68500000)
        assert env.action_masks()[0]
    for left, right in (("flat_free", "flat_canonical"),
                        ("shared_free", "shared_canonical")):
        assert initial[left].keys() == initial[right].keys()
        assert all(torch.equal(initial[left][key], initial[right][key]) for key in initial[left])


def test_canonical_policy_gradient_save_reload(tmp_path):
    env = PairedMixtureEnv(68500000, canonical=True)
    model = MaskablePPO(
        "MlpPolicy", env, n_steps=32, batch_size=32, n_epochs=1,
        device="cpu", seed=68000000, verbose=0,
    )
    before = [parameter.detach().clone() for parameter in model.policy.parameters()]
    model.learn(total_timesteps=32)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.policy.parameters()))
    path = tmp_path / "canonical.zip"
    model.save(path)
    reloaded = MaskablePPO.load(path, device="cpu")
    obs, _ = env.reset(seed=68500000)
    mask = env.action_masks()
    left = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
    right = int(reloaded.predict(obs, action_masks=mask, deterministic=True)[0])
    assert left == right and mask[left]
    controller = V7LearnedController(reloaded, canonical=True)
    _, matching = controller.choose(env.config, env.inner.state, 0,
                                    case=CASES[0], seed=68500000)
    assert matching in env.actions
