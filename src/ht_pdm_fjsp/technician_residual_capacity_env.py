"""Fresh N4/K2 physics tapes and explicit independent-proposal arbitration."""
from dataclasses import asdict
import hashlib
import json
import numpy as np
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_skill_contention_screen import quality
from ht_pdm_fjsp.technician_completion_v2_env import ResourceEnv as BaseEnv, scaled_config, value_unit
from ht_pdm_fjsp.technician_marginal_matching_env import decode_matching
from ht_pdm_fjsp.technician_residual_capacity_policy import CORE_ARMS

VERSION = 'technician_residual_service_capacity_ppo_v1'
GRAPH_ARMS = CORE_ARMS
COUNTERS = ('order_forward', 'order_reverse')


class ResourceEnv(BaseEnv):
    def __init__(self, seed, n, method, worker=0, training=False, journal=None):
        if n != 4:
            raise ValueError('N4/K2 first; no larger runs registered')
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
        mask = self.observe()['pair_mask']
        tokens = np.asarray(submitted)
        if (tokens.shape != (3,) or tokens.dtype.kind not in 'iu' or tokens[-1] != 0
                or ((tokens[:-1] < 0) | (tokens[:-1] > 4)).any()):
            raise ValueError('canonical tokens and fixed zero auxiliary token')
        conflicts = rejected = 0
        if self.method == 'independent_capacity_ppo':
            requests = [(int(token)-1, j) for j, token in enumerate(tokens[:-1]) if token]
            if any(not mask[m, j] for m, j in requests):
                raise ValueError('masked independent proposal')
            pairs = []
            for m in range(4):
                group = [p for p in requests if p[0] == m]
                if group:
                    conflicts += len(group) > 1
                    rejected += len(group)-1
                    pairs.append(max(group, key=lambda p: (quality(self.config, *p), -p[1])))
            pairs = tuple(sorted(pairs))
        else:
            pairs = decode_matching(mask, tokens)
        validate_matching(self.config, self.env.state, pairs)
        return pairs, dict(conflicts=int(conflicts), rejected_proposals=rejected,
            executed_starts=len(pairs), order_forward=1, order_reverse=0)

    def step(self, submitted):
        obs, reward, done, info = super().step(submitted)
        for key in self.broker_totals:
            self.broker_totals[key] += info['coordination'][key]
        if done:
            info['episode_metrics'].update({'broker_'+key: value
                for key, value in self.broker_totals.items()})
        return obs, reward, done, info
