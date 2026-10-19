import torch

from ht_pdm_fjsp.passive_technician_marl import PassiveTechnicianEnv
from ht_pdm_fjsp.passive_technician_scale_transfer import (
    ScalableSetQMix,
    _profile,
    evaluation_cells,
)


def test_scale_transfer_protocol_and_panels_are_locked() -> None:
    train, development, budgets = _profile("full")
    assert train == tuple(range(31, 36))
    assert development == tuple(range(501, 551))
    assert budgets == (5_000, 10_000, 20_000)
    assert not set(train + development).intersection(range(601, 701))


def test_sparse_scale_cells_have_feasible_edges_and_dynamic_dimensions() -> None:
    for seed in (501, 502, 503):
        cells = evaluation_cells(seed)
        for name, config in cells.items():
            env = PassiveTechnicianEnv(config, seed=seed)
            observations = env.reset()
            assert len(observations) == config.machines
            assert len(observations[0]) == 3 + 6 * config.technicians
            assert all(any(mask[1:]) for mask in env.action_masks())
            if name.endswith("sparse"):
                assert any(
                    any(config.service_time[machine][tech] == float("inf") for tech in range(config.technicians))
                    for machine in range(config.machines)
                )


def test_set_qmix_models_transfer_across_machine_and_technician_counts() -> None:
    for mode in ("pooled_set_qmix", "relational_set_qmix"):
        model = ScalableSetQMix(mode, hidden=16)
        for config in evaluation_cells(501).values():
            local = torch.randn(2, config.machines, config.observation_dim)
            actions = torch.zeros(2, config.machines, dtype=torch.long)
            assert model.agent_q(local).shape == (2, config.machines, config.technicians + 1)
            assert model.total_q(local, actions).shape == (2,)
