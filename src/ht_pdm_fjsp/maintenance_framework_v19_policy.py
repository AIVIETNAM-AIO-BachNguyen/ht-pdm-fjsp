"""V19 present-information temporal matching-deficit features and full-matching actor."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ht_pdm_fjsp.maintenance_framework_v1 import catalogue

ACTIONS = catalogue()
OFFSETS = np.arange(1, 10, dtype=np.int16)
INCIDENCE = np.zeros((len(ACTIONS), 4, 2), dtype=bool)
for index, matching in enumerate(ACTIONS):
    for machine, technician in matching:
        INCIDENCE[index, machine, technician] = True
SERVED = INCIDENCE.any(axis=2)


def matching_profiles(observation: np.ndarray, mode: str) -> np.ndarray:
    """Return [batch, actions, 18] forecast demand/shortage features.

    Demand is an event-free present-state projection. It never reads a future
    simulator event tape or a learned continuation. The snapshot arm differs
    only by repeating the action-specific shortage at offset one.
    """
    if mode not in ('temporal_deficit', 'snapshot_deficit'):
        raise ValueError(mode)
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != 56:
        raise ValueError('expected [batch,56] current observation')
    batch = len(obs)
    machine = obs[:, 2:34].reshape(batch, 4, 8)
    worker = obs[:, 34:40].reshape(batch, 2, 3)
    edges = obs[:, 40:56].reshape(batch, 4, 2, 2)
    age = np.rint(machine[..., 0] * 8).astype(np.int16)
    queued = machine[..., 2] > .5
    occupied = machine[..., 5] > .5
    compatibility = edges[..., 0] > .5
    duration = np.rint(edges[..., 1] * 12).astype(np.int16)
    remaining = np.rint(worker[..., 1] * 12).astype(np.int16)
    requested = (~occupied[:, None, :]) & (
        queued[:, None, :] | (age[:, None, :] + OFFSETS[None, :, None] >= 3))
    demanded = requested[:, None, :, :] & (~SERVED[None, :, None, :])
    demand_count = demanded.sum(axis=-1).astype(np.int16)
    selected_duration = np.einsum('amj,bmj->baj', INCIDENCE.astype(np.int16), duration)
    release = np.maximum(remaining[:, None, :], selected_duration)
    free = release[:, :, None, :] <= OFFSETS[None, None, :, None]
    eligible = (demanded[..., :, None]
                & compatibility[:, None, None, :, :]
                & free[..., None, :])
    count0 = eligible[..., 0].sum(axis=-1).astype(np.int16)
    count1 = eligible[..., 1].sum(axis=-1).astype(np.int16)
    overlap = (eligible[..., 0] & eligible[..., 1]).sum(axis=-1).astype(np.int16)
    cardinality = ((count0 + count1) > 0).astype(np.int16)
    cardinality += (count0 * count1 > overlap).astype(np.int16)
    deficit = demand_count - cardinality
    if mode == 'snapshot_deficit':
        deficit = np.broadcast_to(deficit[:, :, :1], deficit.shape)
    return np.concatenate((demand_count / 4., deficit / 4.), axis=-1).astype(np.float32)


class TemporalMatchingActor(nn.Module):
    """Same active network for both arms; temporal deficit is the only input ablation."""

    def __init__(self, mode: str):
        super().__init__()
        if mode not in ('temporal_deficit', 'snapshot_deficit'):
            raise ValueError(mode)
        self.mode = mode
        incidence = torch.as_tensor(INCIDENCE, dtype=torch.float32)
        self.register_buffer('incidence', incidence.flatten(1))
        self.register_buffer('served', incidence.sum(dim=2).clamp(max=1))
        self.net = nn.Sequential(nn.Linear(92, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def feature_tensor(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != 56:
            raise ValueError('expected [batch,56] current observation')
        batch = observation.shape[0]
        rem = observation[:, [35, 38]]
        comp = observation[:, 40:56].reshape(batch, 4, 2, 2)[..., 0] > 0
        earliest = rem[:, None, :].expand(-1, 4, -1).masked_fill(~comp, float('inf')).min(dim=2).values
        if not torch.isfinite(earliest).all():
            raise ValueError('machine without compatible technician')
        release = torch.cat((rem, earliest.clamp(min=1/12)), dim=1)
        profiles = matching_profiles(observation.detach().cpu().numpy(), self.mode)
        profile_tensor = torch.as_tensor(profiles, dtype=observation.dtype, device=observation.device)
        features = torch.cat((
            observation[:, None, :].expand(-1, len(ACTIONS), -1),
            self.incidence[None, :, :].expand(batch, -1, -1),
            self.served[None, :, :].expand(batch, -1, -1),
            release[:, None, :].expand(-1, len(ACTIONS), -1),
            profile_tensor), dim=2)
        assert features.shape == (batch, len(ACTIONS), 92)
        return features

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return -self.net(self.feature_tensor(observation)).squeeze(-1)
