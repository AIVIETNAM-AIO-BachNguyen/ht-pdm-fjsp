"""Two canonical orders; exact physical-matching or sampled-order PPO scores."""
import math
import numpy as np
import torch
from torch import nn

from ht_pdm_fjsp.maintenance_balanced_algorithms import mlp, masked, tensors
from ht_pdm_fjsp.technician_completion_policy import initial_calendar, calendar_embeddings
from ht_pdm_fjsp.technician_marginal_matching_env import CORE_ARMS, value_unit


class MarginalMatchingPolicy(nn.Module):
    is_actor = True
    joint_policy = True

    def __init__(self, algorithm, hidden=27):
        super().__init__()
        if algorithm not in CORE_ARMS:
            raise ValueError('marginal matching core arm')
        self.algorithm, self.hidden = algorithm, hidden
        self.encoder = mlp(22, hidden, hidden)
        self.calendar_encoder = mlp(4, hidden, hidden)
        self.bid = mlp(4*hidden, hidden, 1)
        self.wait = mlp(2*hidden+1, hidden, 1)
        self.value_encoder = mlp(22, hidden, hidden)
        self.value_calendar_encoder = mlp(4, hidden, hidden)
        self.value_context = mlp(2*hidden+1, hidden, hidden)
        self.value_head = mlp(hidden+9, hidden, 1)

    def gradient_groups(self):
        return {label: [p for name, p in self.named_parameters()
                        if name.startswith('value_') == (label == 'critic')]
                for label in ('actor', 'critic')}

    @staticmethod
    def raw_pairs(batch):
        m, t, e, g = (batch[key] for key in ('machines', 'technicians', 'edges', 'global_features'))
        b, n, k = e.shape[:3]
        return torch.cat([m[:, :, None].expand(-1, -1, k, -1),
                          t[:, None].expand(-1, n, -1, -1), e,
                          g[:, None, None].expand(-1, n, k, -1)], -1)

    def physical_value(self, batch, raw=None):
        raw = self.raw_pairs(batch) if raw is None else raw
        calendar, planned = initial_calendar(batch)
        edge = torch.tanh(self.value_encoder(raw)
                          + calendar_embeddings(self.value_calendar_encoder, calendar, planned))
        tech, graph = edge.mean(1), edge.mean((1, 2))
        free = batch['technicians'][..., :1]
        context = self.value_context(torch.cat([tech, graph[:, None].expand_as(tech), free], -1)).mean(1)
        return self.value_head(torch.cat([context, batch['global_features']], -1)).squeeze(-1)*value_unit(raw.shape[1])

    def conditionals(self, batch, base, calendar, planned, technician, position, reserved):
        b, n, k, _ = base.shape
        rows = torch.arange(b, device=base.device)
        edge = base + calendar_embeddings(self.calendar_encoder, calendar, planned)
        machine, tech, graph = edge.mean(2), edge.mean(1), edge.mean((1, 2))
        own = tech[rows, technician]
        pair = edge[rows[:, None], torch.arange(n, device=base.device)[None], technician[:, None]]
        context = torch.cat([pair, machine, own[:, None].expand(-1, n, -1),
                             graph[:, None].expand(-1, n, -1)], -1)
        bids = self.bid(context).squeeze(-1)
        stage = torch.full((b, 1), position/k, dtype=base.dtype, device=base.device)
        wait = self.wait(torch.cat([own, graph, stage], -1))
        legal = batch['pair_mask'][rows[:, None], torch.arange(n, device=base.device)[None], technician[:, None]] & ~reserved
        support = torch.cat([torch.ones((b, 1), dtype=torch.bool, device=base.device), legal], -1)
        return torch.log_softmax(masked(torch.cat([wait, bids], -1), support), -1), support

    def component(self, batch, base, order, recorded=None, generator=None, deterministic=False):
        b, n, k, _ = base.shape
        rows = torch.arange(b, device=base.device)
        if order.shape != (b,) or order.dtype != torch.long or ((order < 0) | (order > 1)).any():
            raise ValueError('order vector')
        if recorded is not None and (recorded.shape != (b, k) or recorded.dtype != torch.long):
            raise ValueError('physical matching shape/type')
        calendar, planned = initial_calendar(batch)
        original_calendar, original_planned = calendar, planned
        reserved = torch.zeros((b, n), dtype=torch.bool, device=base.device)
        action = torch.zeros((b, k), dtype=torch.long, device=base.device)
        logs, entropies = [], []
        for position in range(k):
            technician = torch.where(order == 0, position, k-1-position)
            no_prefix = self.algorithm == 'no_prefix_ppo'
            log, support = self.conditionals(batch, base,
                original_calendar if no_prefix else calendar,
                original_planned if no_prefix else planned, technician, position, reserved)
            if recorded is not None:
                chosen = recorded[rows, technician]
            elif deterministic:
                chosen = log.argmax(-1)
            else:
                if generator is None:
                    raise ValueError('explicit action RNG')
                chosen = torch.multinomial(log.exp(), 1, generator=generator).squeeze(-1)
            if ((chosen < 0) | (chosen > n)).any() or not support.gather(1, chosen[:, None]).all():
                raise ValueError('masked or duplicate canonical matching')
            action[rows, technician] = chosen
            logs.append(log.gather(1, chosen[:, None]).squeeze(-1))
            entropies.append(-(log.exp()*log).sum(-1))
            selected = rows[chosen > 0]
            machine = chosen[selected]-1
            tech = technician[selected]
            calendar, planned, reserved = calendar.clone(), planned.clone(), reserved.clone()
            calendar[selected, machine, 0] = 1.
            calendar[selected, machine, 1] = batch['edges'][selected, machine, tech, 1]
            calendar[selected, machine, 2] = torch.floor(batch['machines'][selected, machine, 0]*8
                * (1-batch['edges'][selected, machine, tech, 2]))/8
            planned[selected, machine, tech] = 1.
            reserved[selected, machine] = True
        return action, torch.stack(logs, 1).sum(1), torch.stack(entropies, 1)

    def components(self, batch, matching, base=None):
        base = self.encoder(self.raw_pairs(batch)) if base is None else base
        b = len(base)
        result = [self.component(batch, base, torch.full((b,), o, dtype=torch.long, device=base.device),
                                 recorded=matching) for o in (0, 1)]
        return torch.stack([r[1] for r in result], 1), torch.stack([r[2] for r in result], 1).mean((1, 2))

    def physical_log(self, component_logs):
        if self.algorithm == 'forward_order_ppo':
            return component_logs[:, 0]
        if self.algorithm == 'reverse_order_ppo':
            return component_logs[:, 1]
        return torch.logsumexp(component_logs, 1)-math.log(2)

    def learning_log(self, component_logs, order):
        if self.algorithm == 'latent_order_ppo':
            return component_logs.gather(1, order[:, None]).squeeze(1)-math.log(2)
        return self.physical_log(component_logs)

    def evaluate(self, batch, actions):
        b, _, k = batch['edges'].shape[:3]
        if actions.shape != (b, k+1) or actions.dtype != torch.long:
            raise ValueError('matching and order shape/type')
        order = actions[:, -1]
        if ((order < 0) | (order > 1)).any():
            raise ValueError('invalid recorded order')
        forced = 0 if self.algorithm == 'forward_order_ppo' else 1 if self.algorithm == 'reverse_order_ppo' else None
        if forced is not None and (order != forced).any():
            raise ValueError('recorded order conflicts with fixed-order prior')
        raw = self.raw_pairs(batch)
        logs, entropy = self.components(batch, actions[:, :-1], self.encoder(raw))
        return self.learning_log(logs, order), entropy, self.physical_value(batch, raw)

    def greedy(self, batch, base=None):
        base = self.encoder(self.raw_pairs(batch)) if base is None else base
        b = len(base)
        plans = [self.component(batch, base, torch.full((b,), o, dtype=torch.long, device=base.device),
                                deterministic=True)[0] for o in (0, 1)]
        scores = [self.physical_log(self.components(batch, plan, base)[0]) for plan in plans]
        chosen = (scores[1] > scores[0]).long()
        matching = torch.stack(plans, 1)[torch.arange(b, device=base.device), chosen]
        if self.algorithm == 'forward_order_ppo':
            chosen = torch.zeros_like(chosen)
        elif self.algorithm == 'reverse_order_ppo':
            chosen = torch.ones_like(chosen)
        return torch.cat([matching, chosen[:, None]], 1)

    @torch.no_grad()
    def sample(self, batch, generator=None, deterministic=False, epsilon=0):
        if epsilon != 0 or (not deterministic and generator is None):
            raise ValueError('explicit action RNG and zero epsilon required')
        raw = self.raw_pairs(batch)
        base = self.encoder(raw)
        b = len(base)
        if deterministic:
            actions = self.greedy(batch, base)
        else:
            # Every arm consumes the same uniform-order draw, even fixed priors.
            order = torch.multinomial(torch.full((b, 2), .5, device=base.device), 1,
                                      generator=generator).squeeze(1)
            if self.algorithm == 'forward_order_ppo':
                order = torch.zeros_like(order)
            elif self.algorithm == 'reverse_order_ppo':
                order = torch.ones_like(order)
            matching = self.component(batch, base, order, generator=generator)[0]
            actions = torch.cat([matching, order[:, None]], 1)
        logs, _ = self.components(batch, actions[:, :-1], base)
        return actions, self.learning_log(logs, actions[:, -1]), self.physical_value(batch, raw)

    @staticmethod
    def environment_action(action):
        return action.numpy().copy()

    @torch.no_grad()
    def select(self, obs):
        return self.greedy(tensors([obs]))[0].numpy().copy()
