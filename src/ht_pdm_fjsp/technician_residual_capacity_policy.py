"""One fixed residual-capacity representation; ordinary canonical joint PPO."""
import torch
from ht_pdm_fjsp.maintenance_balanced_algorithms import mlp, tensors
from ht_pdm_fjsp.technician_completion_policy import initial_calendar
from ht_pdm_fjsp.technician_marginal_matching_policy import MarginalMatchingPolicy

CORE_ARMS = ('residual_capacity_ppo', 'no_capacity_ppo', 'initial_capacity_ppo',
             'permuted_capacity_ppo', 'independent_capacity_ppo')
HORIZONS = (1, 2, 4)


def capacity_summaries(batch, reserved, technician):
    """Return true and static B,N,3 fluid summaries using current inputs only."""
    machines, technicians, edges = (batch[k] for k in ('machines', 'technicians', 'edges'))
    b, n, k = edges.shape[:3]
    urgent = (machines[..., 1] > .5) | (
        machines[..., 0]*8 >= batch['global_features'][:, None, 1]*8-1)
    initial_u = urgent & (machines[..., 3] < .5)
    duration = torch.round(edges[..., 1]*12)
    remaining = torch.round(technicians[..., 1]*12)
    horizons = torch.tensor(HORIZONS, dtype=edges.dtype, device=edges.device)
    feasible = ((edges[..., :1] > .5) &
                ((duration+remaining[:, None])[..., None] <= horizons))
    later = torch.arange(k, device=edges.device)[None] > technician[:, None]

    def summary(unserved, roles):
        support = feasible & unserved[:, :, None, None] & roles[:, None, :, None]
        demand = support.sum(1, keepdim=True).clamp_min(1)
        return (support.to(edges.dtype)/demand).sum(2)

    true = summary(initial_u & ~reserved, later)
    static = summary(initial_u, torch.ones((b, k), dtype=torch.bool, device=edges.device))
    return true, static


class ResidualCapacityPolicy(MarginalMatchingPolicy):
    def __init__(self, algorithm, hidden=27):
        if algorithm not in CORE_ARMS:
            raise ValueError('residual capacity core arm')
        super().__init__('forward_order_ppo', hidden)
        self.algorithm = algorithm
        self.capacity_encoder = mlp(3, hidden, hidden)
        torch.nn.init.zeros_(self.capacity_encoder[-1].weight)
        torch.nn.init.zeros_(self.capacity_encoder[-1].bias)

    def conditionals(self, batch, base, calendar, planned, technician, position, reserved):
        if self.algorithm == 'independent_capacity_ppo':
            calendar, planned = initial_calendar(batch)
            reserved = torch.zeros_like(reserved)
        true, static = capacity_summaries(batch, reserved, technician)
        permuted = torch.roll(true, shifts=1, dims=1)
        message = (torch.zeros_like(true) if self.algorithm == 'no_capacity_ppo'
                   else static if self.algorithm == 'initial_capacity_ppo'
                   else permuted if self.algorithm == 'permuted_capacity_ppo' else true)
        capacity_edge = self.capacity_encoder(message)[:, :, None, :]
        return super().conditionals(batch, base+capacity_edge, calendar, planned,
                                    technician, position, reserved)

    def evaluate(self, batch, actions):
        b, _, k = batch['edges'].shape[:3]
        if (actions.shape != (b, k+1) or actions.dtype != torch.long
                or (actions[:, -1] != 0).any()):
            raise ValueError('canonical proposal and fixed zero auxiliary token')
        raw = self.raw_pairs(batch)
        _, log, entropy = self.component(batch, self.encoder(raw), actions[:, -1],
                                         recorded=actions[:, :-1])
        return log, entropy.mean(1), self.physical_value(batch, raw)

    def greedy(self, batch, base=None):
        base = self.encoder(self.raw_pairs(batch)) if base is None else base
        order = torch.zeros(len(base), dtype=torch.long, device=base.device)
        matching = self.component(batch, base, order, deterministic=True)[0]
        return torch.cat([matching, order[:, None]], 1)

    @torch.no_grad()
    def sample(self, batch, generator=None, deterministic=False, epsilon=0):
        if epsilon != 0 or (not deterministic and generator is None):
            raise ValueError('explicit RNG and zero epsilon required')
        raw = self.raw_pairs(batch)
        order = torch.zeros(len(raw), dtype=torch.long, device=raw.device)
        matching, log, _ = self.component(batch, self.encoder(raw), order,
                                          generator=generator, deterministic=deterministic)
        return torch.cat([matching, order[:, None]], 1), log, self.physical_value(batch, raw)

    @torch.no_grad()
    def select(self, obs):
        return self.greedy(tensors([obs]))[0].numpy().copy()
