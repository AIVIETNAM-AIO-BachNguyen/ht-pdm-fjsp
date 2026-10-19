"""Post-decision capacity matching actor for the fixed N4/K2 maintenance adapter.

The actor sees only the current 56-feature observation. Feasibility remains
with the environment's legal mask. No forecast or realized event tape is used.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from ht_pdm_fjsp.maintenance_framework_v1 import catalogue

VARIANTS = ("edge_only", "additive_queue", "postdecision")
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


class PostdecisionCapacityActor(nn.Module):
    """Shared edge score plus an action-conditioned residual wait penalty."""

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
        self.register_buffer("served", self.incidence.reshape(len(ACTIONS), MACHINES, TECHNICIANS).sum(dim=2))

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
        if self.variant == "additive_queue":
            demand = queued.to(machine.dtype)
        else:
            demand = torch.zeros((batch, MACHINES), dtype=machine.dtype, device=machine.device)
        eligible = edge[..., 0]
        alternatives = eligible.sum(dim=2).clamp_min(1)
        weighted = potential[..., None] * demand[..., None] * eligible / alternatives[..., None]
        # Exclude the target request: using j for m does not displace m.
        return weighted.sum(dim=1, keepdim=True) - weighted

    def postdecision_penalty(self, machine: torch.Tensor, worker: torch.Tensor,
                             edge: torch.Tensor) -> torch.Tensor:
        """Extra earliest-start delay of still-queued requests, in physical ticks."""
        batch = machine.shape[0]
        incidence = self.incidence.reshape(len(ACTIONS), MACHINES, TECHNICIANS)
        # Each action assigns at most one machine to each technician.
        assigned_ticks = torch.einsum("amj,bmj->baj", incidence, edge[..., 1] * 12.0)
        remaining_ticks = worker[..., 1] * 12.0
        after = torch.maximum(remaining_ticks[:, None, :], assigned_ticks)
        eligible = edge[..., 0] > 0.5
        before_start = torch.where(
            eligible, remaining_ticks[:, None, :], torch.full_like(edge[..., 0], 1e6)
        ).amin(dim=-1)
        after_start = torch.where(
            eligible[:, None, :, :], after[:, :, None, :],
            torch.full((batch, len(ACTIONS), MACHINES, TECHNICIANS), 1e6,
                       dtype=edge.dtype, device=edge.device),
        ).amin(dim=-1)
        extra_wait = (after_start - before_start[:, None, :]).clamp_min(0)
        queued = (machine[..., 2] > 0.5) & (machine[..., 5] < 0.5)
        unserved = queued[:, None, :] & (self.served[None, :, :] < 0.5)
        potential = F.softplus(self.potential_network(machine).squeeze(-1))
        return F.softplus(self.log_opportunity_scale) * (
            extra_wait * potential[:, None, :] * unserved.to(edge.dtype)
        ).sum(dim=-1)

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
        logits = base_score.reshape(batch, -1) @ self.incidence.T
        if self.variant == "additive_queue":
            competition = self.opportunity_context(machine, edge)
            duration = edge[..., 1]
            price = F.softplus(self.log_opportunity_scale) * duration * competition
            logits = logits - (price.reshape(batch, -1) @ self.incidence.T)
        elif self.variant == "postdecision":
            logits = logits - self.postdecision_penalty(machine, worker, edge)
        return logits


class PostdecisionCapacityPolicy(MaskableActorCriticPolicy):
    """Maskable PPO policy with a structured actor and standard value critic."""

    def __init__(self, *args, variant: str = "postdecision", **kwargs):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        kwargs.setdefault("net_arch", dict(pi=[], vf=[64, 64]))
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = PostdecisionCapacityActor(self.variant)
        # The default optimizer was created before replacing the action head.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs,
        )
