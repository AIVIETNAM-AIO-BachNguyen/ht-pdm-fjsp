"""Bounded provisional matching exchange before unchanged physical starts."""
from dataclasses import asdict
import hashlib
import json
import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_skill_contention_screen import quality
from ht_pdm_fjsp.technician_completion_v2_env import ResourceEnv as BaseEnv, value_unit, scaled_config

VERSION = 'technician_exchange_ppo_v2'
CORE_ARMS = ('exchange_ppo', 'unrestricted_exchange_ppo', 'inverse_mask_ppo', 'self_repair_ppo', 'construct_only_ppo', 'independent_ppo')
BILATERAL_ARMS = CORE_ARMS[:3]
GRAPH_ARMS = (*CORE_ARMS, 'completion_reference_ppo')
COUNTERS = ('exchanges', 'revisions', 'transfers', 'drops', 'construction_proposals', 'net_plan_changes', 'repair_cancellations')


def decode_trace(mask, trace, arm):
    """Independent scalar broker; token0 is WAIT then KEEP, tokenN+1 DROP."""
    if arm not in CORE_ARMS:
        raise ValueError('exchange arm')
    mask, trace = np.asarray(mask, dtype=bool), np.asarray(trace)
    n, k = mask.shape
    if trace.shape != (2*k,) or trace.dtype.kind not in 'iu' or ((trace < 0) | (trace > n+1)).any():
        raise ValueError('exchange trace shape/range')
    plan = np.zeros(k, dtype=np.int64)
    counts = dict.fromkeys(COUNTERS, 0)
    previous_owner = -1
    construction = None
    for slot, token in enumerate(trace):
        j, repair = slot % k, slot >= k
        token, old = int(token), int(plan[j])
        last_owner, previous_owner = previous_owner, -1
        if slot == k: construction = plan.copy()
        if not repair:
            if token == n+1 or (token and (not mask[token-1, j] or
                    (arm != 'independent_ppo' and token in plan))):
                raise ValueError('illegal construction')
            plan[j] = token
            counts['construction_proposals'] += int(token > 0)
            continue
        if token == 0:
            continue
        if arm == 'construct_only_ppo':
            raise ValueError('construction control cannot repair')
        if token == n+1:
            if old == 0:
                raise ValueError('cannot drop empty proposal')
            plan[j] = 0
            counts['drops'] += 1
            counts['revisions'] += 1
            continue
        if token == old or not mask[token-1, j]:
            raise ValueError('illegal repair target')
        owners = np.flatnonzero(plan == token)
        if arm != 'independent_ppo' and len(owners):
            owner = int(owners[0])
            if arm not in BILATERAL_ARMS or (old and not mask[old-1, owner]):
                raise ValueError('cross exchange or reciprocal edge forbidden')
            if arm == 'exchange_ppo' and owner <= j:
                raise ValueError('cannot revise a committed technician')
            if arm == 'inverse_mask_ppo' and last_owner == j and owner == j-1:
                raise ValueError('immediate inverse exchange forbidden')
            previous_owner = owner
            plan[owner] = old
            counts['exchanges'] += 1
            counts['transfers'] += int(old == 0)
        plan[j] = token
        counts['revisions'] += 1
    counts['net_plan_changes'] = int(not np.array_equal(plan, construction))
    counts['repair_cancellations'] = int(counts['revisions'] > 0 and not counts['net_plan_changes'])
    if arm == 'exchange_ppo' and counts['repair_cancellations']:
        raise RuntimeError('ordered repair cancellation invariant')
    return plan, counts


class ResourceEnv(BaseEnv):
    def reset(self):
        self.episode_index += 1
        namespace = f'{VERSION}:N{self.config.machines}'
        role = (f'{namespace}:training_physical:worker:{self.worker}:episode:{self.episode_index}'
                if self.training else f'{namespace}:physical')
        self.physical_seed = role_seed(self.seed, role)
        self.env.reset(self.physical_seed)
        self.active = True
        self.broker_totals = dict.fromkeys((*COUNTERS, 'conflicts', 'rejected_proposals'), 0)
        payload = json.dumps(dict(seed=self.physical_seed, state=asdict(self.env.state), events=self.env.events), sort_keys=True)
        self.population = hashlib.sha256(payload.encode()).hexdigest()
        return self.observe()

    def resolve(self, submitted):
        if self.method not in GRAPH_ARMS:
            pairs, info = super().resolve(submitted)
            return pairs, dict(info, **dict.fromkeys(COUNTERS, 0))
        n, k = self.config.machines, self.config.technicians
        if self.method == 'completion_reference_ppo':
            original = np.asarray(submitted)
            if original.shape != (k,) or original.dtype.kind not in 'iu':
                raise ValueError('reference trace shape/type')
            submitted = np.concatenate([original, np.zeros(k, dtype=np.int64)])
            arm = 'construct_only_ppo'
        else:
            arm = self.method
        mask = self.observe()['pair_mask']
        plan, counters = decode_trace(mask, submitted, arm)
        requests = [(int(token)-1, j) for j, token in enumerate(plan) if token]
        conflicts = rejected = 0
        if self.method == 'independent_ppo':
            pairs = []
            for m in range(n):
                group = [p for p in requests if p[0] == m]
                if group:
                    conflicts += int(len(group) > 1)
                    rejected += len(group)-1
                    pairs.append(max(group, key=lambda p: (quality(self.config, *p), -p[1])))
        else:
            pairs = requests
        pairs = tuple(sorted(pairs))
        validate_matching(self.config, self.env.state, pairs)
        return pairs, dict(conflicts=conflicts, rejected_proposals=rejected,
                           executed_starts=len(pairs), **counters)

    def step(self, submitted):
        obs, reward, done, info = super().step(submitted)
        for key in self.broker_totals:
            self.broker_totals[key] += info['coordination'][key]
        if done:
            info['episode_metrics'].update({'broker_'+k: v for k, v in self.broker_totals.items()})
        return obs, reward, done, info
