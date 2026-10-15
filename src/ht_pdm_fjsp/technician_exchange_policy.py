"""Exact latent joint-trace PPO policy over bounded bilateral exchanges."""
import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot

from ht_pdm_fjsp.maintenance_balanced_algorithms import mlp, masked, tensors
from ht_pdm_fjsp.technician_completion_policy import initial_calendar, calendar_embeddings
from ht_pdm_fjsp.technician_exchange_env import CORE_ARMS, value_unit


def intent_view(matching, technician, independent):
    if not independent:
        return matching
    view = torch.zeros_like(matching)
    view[:, technician] = matching[:, technician]
    return view


def provisional_calendar(batch, matching):
    n = batch['machines'].shape[1]
    calendar, planned = initial_calendar(batch)
    intent = one_hot(matching, n+1)[..., 1:].transpose(1, 2).to(calendar.dtype)
    active = intent.any(2)
    duration = (batch['edges'][..., 1]*intent).sum(2)
    restoration = (batch['edges'][..., 2]*intent).sum(2)
    completion = torch.floor(batch['machines'][..., 0]*8*(1-restoration))/8
    return torch.stack([torch.where(active, 1., calendar[..., 0]),
        torch.where(active, duration, calendar[..., 1]),
        torch.where(active, completion, calendar[..., 2])], -1), planned+intent, intent


def move_support(batch, matching, technician, repair, arm):
    b, n, k = batch['edges'].shape[:3]
    view = intent_view(matching, technician, arm == 'independent_ppo')
    intent = one_hot(view, n+1)[..., 1:].transpose(1, 2).bool()
    has_owner, owners = intent.any(2), intent.long().argmax(2)
    own = matching[:, technician]
    legal = batch['pair_mask'][:, :, technician].clone()
    if not repair:
        legal &= ~has_owner
        drop = torch.zeros(b, dtype=torch.bool, device=matching.device)
    else:
        legal &= (torch.arange(n, device=matching.device)[None]+1 != own[:, None])
        if arm == 'exchange_ppo':
            rows = torch.arange(b, device=matching.device)[:, None]
            reciprocal = batch['pair_mask'][rows, (own-1).clamp(min=0)[:, None], owners]
            legal &= ~has_owner | (own[:, None] == 0) | reciprocal
        elif arm != 'independent_ppo':
            legal &= ~has_owner
        drop = own > 0
        if arm == 'construct_only_ppo':
            legal[:] = False
            drop[:] = False
    return torch.cat([torch.ones((b, 1), dtype=torch.bool, device=matching.device), legal, drop[:, None]], 1)


def apply_move(matching, action, technician, repair, arm, n):
    result = matching.clone()
    if not repair:
        result[:, technician] = action
        return result
    target = (action > 0) & (action <= n)
    if arm == 'exchange_ppo':
        claimed = (matching == action[:, None]) & target[:, None]
        rows = torch.where(claimed.any(1))[0]
        owner = claimed.long().argmax(1)
        result[rows, owner[rows]] = matching[rows, technician]
    result[target, technician] = action[target]
    result[action == n+1, technician] = 0
    return result


class ExchangePolicy(nn.Module):
    is_actor = True
    joint_policy = True
    allocation_learning_mode = 'joint'

    def __init__(self, algorithm, hidden=27):
        super().__init__()
        if algorithm not in CORE_ARMS:
            raise ValueError('exchange core arm')
        self.algorithm, self.hidden = algorithm, hidden
        self.encoder = mlp(22, hidden, hidden)
        self.calendar_encoder = mlp(4, hidden, hidden)
        self.move_head = mlp(5*hidden+2, hidden, 1)
        self.control_head = mlp(2*hidden+2, hidden, 2)
        self.value_encoder = mlp(22, hidden, hidden)
        self.value_calendar_encoder = mlp(4, hidden, hidden)
        self.value_context = mlp(2*hidden+2, hidden, hidden)
        self.value_head = mlp(hidden+9, hidden, 1)

    def gradient_groups(self):
        return {label: [p for name, p in self.named_parameters()
            if name.startswith(('value_encoder.', 'value_calendar_encoder.', 'value_context.', 'value_head.')) == (label == 'critic')]
            for label in ('actor', 'critic')}

    def encode_static(self, batch, compute_values):
        m, t, e, g = (batch[key] for key in ('machines', 'technicians', 'edges', 'global_features'))
        b, n, k = e.shape[:3]
        raw = torch.cat([m[:, :, None].expand(-1, -1, k, -1),
            t[:, None].expand(-1, n, -1, -1), e, g[:, None, None].expand(-1, n, k, -1)], -1)
        base = self.encoder(raw)
        if not compute_values:
            return base, torch.zeros(b, dtype=base.dtype, device=base.device)
        calendar, planned = initial_calendar(batch)
        edge = torch.tanh(self.value_encoder(raw)+calendar_embeddings(self.value_calendar_encoder, calendar, planned))
        tech, graph = edge.mean(1), edge.mean((1, 2))
        position = torch.arange(k, dtype=base.dtype, device=base.device)[None, :, None].expand(b, -1, -1)/k
        roles = torch.cat([position, t[..., 0, None]], -1)
        context = self.value_context(torch.cat([tech, graph[:, None].expand(-1, k, -1), roles], -1)).mean(1)
        value = self.value_head(torch.cat([context, g], -1)).squeeze(-1)*value_unit(n)
        return base, value

    def logits(self, batch, base, matching, technician, repair):
        b, n, k, h = base.shape
        view = intent_view(matching, technician, self.algorithm == 'independent_ppo')
        calendar, planned, intent = provisional_calendar(batch, view)
        edge = base+calendar_embeddings(self.calendar_encoder, calendar, planned)
        owners, occupied = intent.argmax(2), intent.any(2)
        rows = torch.arange(b, device=base.device)
        own = matching[:, technician]
        own_row = edge[rows, (own-1).clamp(min=0)]
        own_edge = own_row[:, technician]*(own > 0)[:, None]
        holder = edge.gather(2, owners[..., None, None].expand(-1, -1, 1, h)).squeeze(2)*occupied[..., None]
        reciprocal = own_row.gather(1, owners[..., None].expand(-1, -1, h))*occupied[..., None]*(own > 0)[:, None, None]
        graph = edge.mean((1, 2))
        phase = torch.tensor([float(repair), technician/k], dtype=base.dtype, device=base.device)[None].expand(b, -1)
        context = torch.cat([edge[:, :, technician], own_edge[:, None].expand(-1, n, -1), holder,
            reciprocal, graph[:, None].expand(-1, n, -1), phase[:, None].expand(-1, n, -1)], -1)
        moves = self.move_head(context).squeeze(-1)
        controls = self.control_head(torch.cat([own_edge, graph, phase], -1))
        return torch.cat([controls[:, :1], moves, controls[:, 1:]], -1)

    def trace(self, batch, recorded=None, generator=None, deterministic=False, compute_values=True):
        base, value = self.encode_static(batch, compute_values)
        b, n, k = base.shape[:3]
        if recorded is not None and (recorded.shape != (b, 2*k) or recorded.dtype != torch.long):
            raise ValueError('recorded exchange trace shape/type')
        matching = torch.zeros((b, k), dtype=torch.long, device=base.device)
        actions, logs, entropies = [], [], []
        for slot in range(2*k):
            j, repair = slot % k, slot >= k
            logits = self.logits(batch, base, matching, j, repair)
            support = move_support(batch, matching, j, repair, self.algorithm)
            log = torch.log_softmax(masked(logits, support), -1)
            action = (recorded[:, slot] if recorded is not None else log.argmax(1) if deterministic
                else torch.multinomial(log.exp(), 1, generator=generator).squeeze(1))
            if ((action < 0) | (action > n+1)).any() or not support.gather(1, action[:, None]).all():
                raise ValueError('masked exchange trace action')
            actions.append(action)
            logs.append(log.gather(1, action[:, None]).squeeze(1))
            entropies.append(-(log.exp()*log).sum(1))
            matching = apply_move(matching, action, j, repair, self.algorithm, n)
        return (torch.stack(actions, 1), torch.stack(logs, 1), torch.stack(entropies, 1),
                value[:, None].expand(-1, 2*k))

    def evaluate(self, batch, actions):
        _, log, entropy, value = self.trace(batch, recorded=actions)
        return log, entropy, value

    @torch.no_grad()
    def sample(self, batch, generator=None, deterministic=False, epsilon=0):
        if not deterministic and generator is None:
            raise ValueError('explicit action RNG')
        action, log, _, value = self.trace(batch, generator=generator, deterministic=deterministic)
        return action, log, value

    @torch.no_grad()
    def select(self, observation):
        return self.trace(tensors([observation]), deterministic=True, compute_values=False)[0][0].numpy().copy()

    def environment_action(self, action):
        return action.numpy().copy()

    def return_diagnostics(self, returns, old_values):
        unit = value_unit(old_values.shape[1])  # 2K=N in the registered settings.
        return dict(physical_value_unit=unit, return_mean=float(returns.mean()),
            return_std=float(returns.std(unbiased=False)), value_mean=float(old_values.mean()),
            normalized_return_mean=float(returns.mean()/unit), normalized_return_std=float(returns.std(unbiased=False)/unit),
            normalized_value_mean=float(old_values.mean()/unit), normalized_value_std=float(old_values.std(unbiased=False)/unit),
            normalized_prefix_baseline_spread=float(old_values.std(1, unbiased=False).mean()/unit))
