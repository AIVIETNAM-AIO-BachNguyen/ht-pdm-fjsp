"""Canonical physical matchings with an auxiliary, physically ignored order."""
from dataclasses import asdict
import hashlib
import json
import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.technician_completion_v2_env import ResourceEnv as BaseEnv, scaled_config, value_unit

VERSION = 'technician_marginal_matching_ppo_v1'
CORE_ARMS = ('marginal_matching_ppo', 'latent_order_ppo', 'forward_order_ppo',
             'reverse_order_ppo', 'no_prefix_ppo')
GRAPH_ARMS = CORE_ARMS
COUNTERS = ('order_forward', 'order_reverse')


def decode_matching(mask, submitted):
    mask = np.asarray(mask, dtype=bool)
    n, k = mask.shape
    action = np.asarray(submitted)
    if (action.shape != (k+1,) or action.dtype.kind not in 'iu'
            or action[-1] not in (0, 1) or ((action[:-1] < 0) | (action[:-1] > n)).any()):
        raise ValueError('matching vector/order shape, type or range')
    matching = action[:-1]
    selected = matching[matching > 0]
    if len(set(selected.tolist())) != len(selected):
        raise ValueError('duplicate reserved machine')
    pairs = tuple(sorted((int(token)-1, j) for j, token in enumerate(matching) if token))
    if any(not mask[m, j] for m, j in pairs):
        raise ValueError('masked matching edge')
    return pairs


class ResourceEnv(BaseEnv):
    def __init__(self, seed, n, method, worker=0, training=False, journal=None):
        if n != 4:
            raise ValueError('small-first experiment is N4/K2 only')
        super().__init__(seed, n, method, worker, training, journal)

    def reset(self):
        self.episode_index += 1
        role = (f'{VERSION}:N4:training_physical:worker:{self.worker}:episode:{self.episode_index}'
                if self.training else f'{VERSION}:N4:physical')
        self.physical_seed = role_seed(self.seed, role)
        self.env.reset(self.physical_seed)
        self.active = True
        self.broker_totals = dict.fromkeys((*COUNTERS, 'conflicts', 'rejected_proposals'), 0)
        payload = json.dumps(dict(seed=self.physical_seed, state=asdict(self.env.state),
                                  events=self.env.events), sort_keys=True)
        self.population = hashlib.sha256(payload.encode()).hexdigest()
        return self.observe()

    def resolve(self, submitted):
        if self.method not in CORE_ARMS:
            pairs, info = super().resolve(submitted)
            return pairs, dict(info, **dict.fromkeys(COUNTERS, 0))
        pairs = decode_matching(self.observe()['pair_mask'], submitted)
        validate_matching(self.config, self.env.state, pairs)
        order = int(np.asarray(submitted)[-1])
        return pairs, dict(conflicts=0, rejected_proposals=0, executed_starts=len(pairs),
                           order_forward=int(order == 0), order_reverse=int(order == 1))

    def step(self, submitted):
        obs, reward, done, info = super().step(submitted)
        for key in self.broker_totals:
            self.broker_totals[key] += info['coordination'][key]
        if done:
            info['episode_metrics'].update({'broker_'+key: value
                                            for key, value in self.broker_totals.items()})
        return obs, reward, done, info
