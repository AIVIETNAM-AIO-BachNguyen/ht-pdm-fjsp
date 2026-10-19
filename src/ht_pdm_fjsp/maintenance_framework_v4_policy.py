"""Capacity-opportunity matching actor for the fixed N4/K2 maintenance adapter.

The actor sees only the current 56-feature observation. Feasibility remains
with the environment's legal mask. No forecast or realized event tape is used.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from ht_pdm_fjsp.maintenance_framework_v1 import catalogue

VARIANTS = ("edge_only", "current_queue", "temporal")
MACHINES = 4
TECHNICIANS = 2
OBSERVATION_DIM = 56
ACTIONS = catalogue(MACHINES, TECHNICIANS)


def action_incidence() -> torch.Tensor:
    incidence = torch.zeros((len(ACTIONS), MACHINES * TECHNICIANS), dtype=torch.float32)
    for a, matching in enumerate(ACTIONS):
        for m, j in matching:
            incidence[a, m * TECHNICIANS + j] = 1
    return incidence


class CapacityOpportunityActor(nn.Module):
    """Shared edge score plus a duration-weighted price for competing demand."""

    def __init__(self, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        self.variant = variant
        self.machine_encoder = nn.Sequential(nn.Linear(8, 24), nn.Tanh())
        # Worker identity/assigned-machine index is deliberately excluded.
        self.worker_encoder = nn.Sequential(nn.Linear(2, 12), nn.Tanh())
        self.edge_network = nn.Sequential(
            nn.Linear(24 + 12 + 2 + 2, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.potential_network = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1),
        )
        self.log_opportunity_scale = nn.Parameter(torch.tensor(-2.0))
        self.register_buffer("incidence", action_incidence())

    @staticmethod
    def split_observation(observation: torch.Tensor):
        if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM:
            raise ValueError(f"expected [batch,{OBSERVATION_DIM}] observation")
        batch = observation.shape[0]
        global_features = observation[:, :2]
        machine = observation[:, 2:34].reshape(batch, MACHINES, 8)
        worker = observation[:, 34:40].reshape(batch, TECHNICIANS, 3)
        edge = observation[:, 40:56].reshape(batch, MACHINES, TECHNICIANS, 2)
        return global_features, machine, worker, edge

    def opportunity_context(self, machine: torch.Tensor, edge: torch.Tensor):
        """Potential of other queued/near-request machines eligible for each worker."""
        batch = machine.shape[0]
        potential = F.softplus(self.potential_network(machine).squeeze(-1))
        queued = machine[..., 2] > 0.5
        active = machine[..., 5] > 0.5
        if self.variant == "temporal":
            # Current age 2 reaches requester threshold 3 at the next tick.
            near = ((machine[..., 0] >= (2.0 / 8.0 - 1e-6))
                    & (machine[..., 0] < (3.0 / 8.0 - 1e-6))
                    & ~queued & ~active & (machine[..., 1] < 0.5))
            demand = queued.to(machine.dtype) + 0.5 * near.to(machine.dtype)
        elif self.variant == "current_queue":
            demand = queued.to(machine.dtype)
        else:
            demand = torch.zeros((batch, MACHINES), dtype=machine.dtype, device=machine.device)
        eligible = edge[..., 0]
        alternatives = eligible.sum(dim=2).clamp_min(1)
        weighted = potential[..., None] * demand[..., None] * eligible / alternatives[..., None]
        # Exclude the target request: using j for m does not displace m.
        return weighted.sum(dim=1, keepdim=True) - weighted

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        global_features, machine, worker, edge = self.split_observation(observation)
        batch = observation.shape[0]
        machine_embed = self.machine_encoder(machine)
        # worker[2] encodes assigned machine ID, which is absent from actor.
        worker_embed = self.worker_encoder(worker[..., :2])
        m = machine_embed[:, :, None, :].expand(batch, MACHINES, TECHNICIANS, -1)
        w = worker_embed[:, None, :, :].expand(batch, MACHINES, TECHNICIANS, -1)
        g = global_features[:, None, None, :].expand(batch, MACHINES, TECHNICIANS, -1)
        inputs = torch.cat((m, w, edge, g), dim=-1)
        base_score = self.edge_network(inputs).squeeze(-1)
        if self.variant != "edge_only":
            competition = self.opportunity_context(machine, edge)
            duration = edge[..., 1]
            price = F.softplus(self.log_opportunity_scale) * duration * competition
            edge_score = base_score - price
        else:
            edge_score = base_score
        return edge_score.reshape(batch, -1) @ self.incidence.T


class CapacityOpportunityPolicy(MaskableActorCriticPolicy):
    """Maskable PPO policy with a structured actor and standard value critic."""

    def __init__(self, *args, variant: str = "temporal", **kwargs):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        kwargs.setdefault("net_arch", dict(pi=[], vf=[64, 64]))
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = CapacityOpportunityActor(self.variant)
        # The default optimizer was created before replacing the action head.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs,
        )
