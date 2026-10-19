import numpy as np
import torch

from ht_pdm_fjsp.maintenance_framework_v1 import FrameworkEnv, configs
from ht_pdm_fjsp.maintenance_framework_v1_campaign import model_new
from ht_pdm_fjsp.maintenance_framework_v3_campaign import (
    V3RuleController, create_teacher_bank, policy_digest, pretrain_actor,
    seed_registry,
)


class Counter:
    def __init__(self):
        self.count = 0

    def update(self, amount):
        self.count += amount


def test_fresh_seed_registry_and_teacher_future_privacy():
    registry, groups = seed_registry()
    flat = [seed for group in groups for seed in group]
    assert len(flat) == len(set(flat))
    assert groups[0] == [64000000, 64000001, 64000002]
    assert groups[4][0] == 64600000 and groups[4][-1] == 64600095
    config = configs()["reciprocal_scale2"]
    env = FrameworkEnv(config)
    env.reset(seed=64500000)
    teacher = V3RuleController("rollout", registry["forecast_seed"], "bank")
    first = teacher.choose(config, env.inner.state, 0, case="reciprocal_scale2", seed=64500000)
    env.inner._events = tuple(tuple(not x for x in row) for row in env.inner._events)
    second = teacher.choose(config, env.inner.state, 0, case="reciprocal_scale2", seed=64500000)
    assert first == second


def test_teacher_labels_legal_and_masked_bc_changes_policy(tmp_path):
    torch.set_num_threads(1)
    config = configs()["reciprocal_scale2"]
    bar = Counter()
    bank, meta = create_teacher_bank(tmp_path, 64000000, [64600000], config,
                                     64300000, lambda: None, bar)
    data = np.load(bank)
    assert bar.count == 1 and meta["labels"] == 48
    assert data["observations"].shape == (48, 56)
    assert data["masks"].shape == (48, 21)
    assert all(data["masks"][np.arange(48), data["actions"]])
    first = model_new(FrameworkEnv(config), 64000000)
    second = model_new(FrameworkEnv(config), 64000000)
    before = policy_digest(first)
    assert before == policy_digest(second)
    info = pretrain_actor(first, bank, 64000000, 2, tmp_path / "bc.csv", lambda: None)
    assert info["optimizer_updates"] == 2
    assert policy_digest(first) != before
