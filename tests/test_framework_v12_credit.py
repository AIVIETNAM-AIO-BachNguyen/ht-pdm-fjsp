"""V12 paired-scenario credit and registry checks."""
import numpy as np
import torch

from ht_pdm_fjsp.maintenance_framework_v12_campaign import (
    ARMS, VERSION, reliability_advantages, seed_registry,
)


def test_registry_is_fresh_and_arms_are_distinct():
    registry, (train, dev, smoke) = seed_registry()
    assert registry['campaign_version'] == VERSION
    assert len(train) == 5 and len(dev) == 16 and len(smoke) == 2
    assert len(set((*train, *dev, *smoke))) == 23
    assert ARMS == ('reliability_credit', 'plain_credit')


def test_plain_and_reliable_agree_on_certain_cost_gap():
    p = torch.tensor([[.5, .5]])
    samples = torch.tensor([[[20., 0.], [20., 0.], [20., 0.], [20., 0.]]])
    mask = torch.tensor([[True, True]])
    proposed, weights = reliability_advantages(p, samples, mask, 'reliability_credit')
    control, control_weights = reliability_advantages(p, samples, mask, 'plain_credit')
    assert torch.allclose(proposed, control)
    assert torch.allclose(weights, control_weights)
    assert torch.allclose(proposed, torch.tensor([[-1., 1.]]))


def test_uncertain_paired_gap_is_attenuated_without_masking_action():
    p = torch.tensor([[.5, .5, 0.]])
    samples = torch.tensor([[[20., 0., 0.], [0., 0., 0.],
                             [0., 0., 0.], [0., 0., 0.]]])
    mask = torch.tensor([[True, True, False]])
    proposed, weights = reliability_advantages(p, samples, mask, 'reliability_credit')
    control, _ = reliability_advantages(p, samples, mask, 'plain_credit')
    assert np.isclose(float(weights[0, 0]), .5)
    assert np.isclose(float(weights[0, 1]), .5)
    assert torch.allclose(proposed[:, :2], control[:, :2] * .5)
    assert float(proposed[0, 2]) == 0
