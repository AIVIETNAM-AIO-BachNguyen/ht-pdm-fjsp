import numpy as np

from ht_pdm_fjsp.maintenance_framework_v1 import CASES
from ht_pdm_fjsp.maintenance_framework_v2_campaign import MixtureEnv, seed_registry
from ht_pdm_fjsp.maintenance_requests import legal_matchings


def test_mixture_episode_sampling_reproducible_and_masks_exact():
    def sample():
        env = MixtureEnv()
        selected = []
        for index in range(24):
            observation, _ = env.reset(seed=63000000 if index == 0 else None)
            selected.append(next(name for name in CASES
                                 if env.config == env.case_configs[name]))
            assert observation.shape == (56,)
            _, admitted = env._admitted()
            assert {action for action, legal in zip(env.actions, env.action_masks()) if legal} == (
                set(legal_matchings(env.config, admitted)))
        assert set(selected) == set(CASES)
        assert sum(env.case_counts.values()) == 24
        return selected
    assert sample() == sample()


def test_v2_seed_splits_are_disjoint_from_historical_registries():
    registry, groups = seed_registry()
    assert registry["campaign_version"] == "maintenance_framework_v2_mixture"
    flat = [seed for group in groups for seed in group]
    assert len(flat) == len(set(flat))
    assert groups[0] == [63000000, 63000001, 63000002]
    assert groups[2][0] == 63200000 and groups[2][-1] == 63200063
