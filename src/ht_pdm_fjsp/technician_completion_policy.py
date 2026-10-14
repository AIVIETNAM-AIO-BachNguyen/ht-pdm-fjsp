"""Completion-aware prefix actor/critic; current inputs and own reservations only."""
import numpy as np
import torch
from torch import nn
from ht_pdm_fjsp.maintenance_balanced_algorithms import mlp, masked, tensors
from ht_pdm_fjsp.technician_completion_env import GRAPH_ARMS


def initial_calendar(batch):
    machines, edges = batch['machines'], batch['edges']
    planned = edges[..., 3]
    restoration = (edges[..., 2] * planned).sum(2)
    completion_age = torch.floor(machines[..., 0] * 8 * (1 - restoration)) / 8
    completion_age = torch.where(machines[..., 3] > .5, completion_age, machines[..., 0])
    return torch.stack([machines[..., 3], machines[..., 4], completion_age], -1), planned.clone()


def calendar_embeddings(encoder, calendar, planned):
    """Exact binary-flag evaluation using two machine encodings per stage."""
    raw = calendar[:, :, None, :].expand(-1, -1, 2, -1)
    flag = torch.tensor([0., 1.], dtype=raw.dtype, device=raw.device)
    flag = flag[None, None, :, None].expand(*raw.shape[:-1], 1)
    pair = encoder(torch.cat([raw, flag], -1))
    return pair[:, :, 0, None, :] + planned[..., None] * (pair[:, :, 1, None, :] - pair[:, :, 0, None, :])


class CompletionPolicy(nn.Module):
    is_actor = True
    joint_policy = False

    def __init__(self, algorithm, hidden=27):
        super().__init__()
        if algorithm not in GRAPH_ARMS:
            raise ValueError('completion graph arm')
        self.algorithm = algorithm
        self.hidden = hidden
        self.allocation_learning_mode = ('prefix' if algorithm == 'completion_credit_ppo'
            else 'state' if algorithm == 'state_credit_ppo' else 'joint')
        self.encoder = mlp(22, hidden, hidden)
        self.value_encoder = mlp(22, hidden, hidden)
        self.calendar_encoder = mlp(4, hidden, hidden)
        self.value_calendar_encoder = mlp(4, hidden, hidden)
        self.bid = mlp(4 * hidden, hidden, 1)
        self.wait = mlp(2 * hidden + 1, hidden, 1)
        self.value_context = mlp(2 * hidden + 2, hidden, hidden)
        self.value_head = mlp(hidden + 10, hidden, 1)

    def gradient_groups(self):
        return {label: [p for name, p in self.named_parameters()
            if name.startswith(('value_encoder.', 'value_calendar_encoder.', 'value_context.', 'value_head.')) == (label == 'critic')]
            for label in ('actor', 'critic')}

    def encode_static(self, batch, compute_values=True):
        m, t, e, g = (batch[key] for key in ('machines', 'technicians', 'edges', 'global_features'))
        _, n, k, _ = e.shape
        raw = torch.cat([m[:, :, None].expand(-1, -1, k, -1),
                         t[:, None].expand(-1, n, -1, -1), e,
                         g[:, None, None].expand(-1, n, k, -1)], -1)
        return self.encoder(raw), self.value_encoder(raw) if compute_values else None

    def trace(self, batch, recorded=None, generator=None, deterministic=False, compute_values=True):
        base, value_base = self.encode_static(batch, compute_values)
        b, n, k, _ = base.shape
        if recorded is not None and (recorded.shape != (b, k) or recorded.dtype != torch.long):
            raise ValueError('recorded canonical trace shape/type')
        calendar, planned = initial_calendar(batch)
        original_calendar, original_planned = calendar.clone(), planned.clone()
        reserved = torch.zeros((b, n), dtype=torch.bool, device=base.device)
        actions, logs, entropies, values = [], [], [], []
        for j in range(k):
            independent = self.algorithm == 'independent_ppo'
            actor_calendar = original_calendar if independent else calendar
            actor_planned = original_planned if independent else planned
            edge = base + calendar_embeddings(self.calendar_encoder, actor_calendar, actor_planned)
            global_pool, machine_pool, tech_pool = edge.mean((1, 2)), edge.mean(2), edge.mean(1)
            context = torch.cat([edge[:, :, j], machine_pool,
                tech_pool[:, j, None].expand(-1, n, -1), global_pool[:, None].expand(-1, n, -1)], -1)
            bids = self.bid(context).squeeze(-1)
            stage = torch.full((b, 1), j / k, dtype=base.dtype, device=base.device)
            wait = self.wait(torch.cat([tech_pool[:, j], global_pool, stage], -1)).squeeze(-1)
            legal = batch['pair_mask'][:, :, j] & (torch.ones_like(reserved) if independent else ~reserved)
            support = torch.cat([torch.ones((b, 1), dtype=torch.bool, device=base.device), legal], -1)
            log = torch.log_softmax(masked(torch.cat([wait[:, None], bids], -1), support), -1)
            # Values precede this choice, so its action cannot leak into its baseline.
            if compute_values:
                value_edge = torch.tanh(value_base + calendar_embeddings(self.value_calendar_encoder, calendar, planned))
                tech_value, global_value = value_edge.mean(1), value_edge.mean((1, 2))
                positions = torch.arange(k, dtype=base.dtype, device=base.device)[None, :, None].expand(b, -1, -1)
                roles = torch.cat([positions / k, (positions >= j).to(base.dtype)], -1)
                value_pool = self.value_context(torch.cat([tech_value,
                    global_value[:, None].expand(-1, k, -1), roles], -1)).mean(1)
                value = self.value_head(torch.cat([value_pool,
                    batch['global_features'], stage], -1)).squeeze(-1) * (n / 4)
            else:
                value = torch.zeros(b, dtype=base.dtype, device=base.device)
            chosen = ((log.argmax(-1) if deterministic else
                torch.multinomial(log.exp(), 1, generator=generator).squeeze(-1))
                if recorded is None else recorded[:, j])
            if ((chosen < 0) | (chosen > n)).any() or not support.gather(1, chosen[:, None]).all():
                raise ValueError('masked/repeated canonical trace')
            actions.append(chosen)
            logs.append(log.gather(1, chosen[:, None]).squeeze(-1))
            entropies.append(-(log.exp() * log).sum(1))
            values.append(value)
            rows = torch.arange(b, device=base.device)[chosen > 0]
            mi = chosen[rows] - 1
            calendar = calendar.clone()
            calendar[rows, mi, 0] = 1.
            calendar[rows, mi, 1] = batch['edges'][rows, mi, j, 1]
            calendar[rows, mi, 2] = torch.floor(batch['machines'][rows, mi, 0] * 8 *
                (1 - batch['edges'][rows, mi, j, 2])) / 8
            planned = planned.clone()
            planned[rows, mi, j] = 1.
            reserved = reserved.clone()
            reserved[rows, mi] = True
        return tuple(torch.stack(parts, 1) for parts in (actions, logs, entropies, values))

    def evaluate(self, batch, actions):
        _, log, entropy, value = self.trace(batch, recorded=actions)
        return log, entropy, value

    @torch.no_grad()
    def sample(self, batch, generator=None, deterministic=False, epsilon=0):
        if not deterministic and generator is None:
            raise ValueError('explicit action RNG')
        action, log, _, value = self.trace(batch, generator=generator, deterministic=deterministic)
        return action, log, value

    def environment_action(self, action):
        return action.numpy().copy()

    @torch.no_grad()
    def select(self, obs):
        return self.trace(tensors({key: np.asarray(value)[None] for key, value in obs.items()}),
                          deterministic=True, compute_values=False)[0][0].numpy().copy()
