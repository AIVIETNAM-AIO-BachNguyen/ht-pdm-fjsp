"""Canonical per-technician reservations on unchanged maintenance physics."""
from dataclasses import asdict
import hashlib
import json
import numpy as np
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_skill_contention_screen import quality
from ht_pdm_fjsp.technician_reservation_env import ResourceEnv as BaseEnv, scaled_config, deadline

VERSION = 'technician_completion_credit_ppo_v2'
GRAPH_ARMS = ('completion_credit_ppo', 'state_credit_ppo', 'joint_credit_ppo', 'independent_ppo', 'no_calendar_ppo')


def value_unit(n):
    config = scaled_config(n)
    return config.unavailable_cost * n * 48 / 20


class ResourceEnv(BaseEnv):
    def __init__(self, seed, n, method, worker=0, training=False, journal=None):
        super().__init__(seed, n, 'ppo' if method == 'normalized_ppo' else method,
                         worker, training, journal)
        self.method = method

    def reset(self):
        self.episode_index += 1
        namespace = f'{VERSION}:N{self.config.machines}'
        role = (f'{namespace}:training_physical:worker:{self.worker}:episode:{self.episode_index}'
                if self.training else f'{namespace}:physical')
        self.physical_seed = role_seed(self.seed, role)
        self.env.reset(self.physical_seed)
        self.active = True
        payload = json.dumps(dict(seed=self.physical_seed, state=asdict(self.env.state),
                                  events=self.env.events), sort_keys=True)
        self.population = hashlib.sha256(payload.encode()).hexdigest()
        return self.observe()

    def observe(self):
        obs = super().observe()
        if self.method == 'full_mappo':
            n = self.config.machines
            obs['full_local'] = np.concatenate([np.tile(obs['state'], (n, 1)),
                                                np.eye(n, dtype=np.float32)], axis=1)
        return obs

    def resolve(self, submitted):
        if self.method == 'normalized_ppo':
            if not isinstance(submitted, (int, np.integer)) or not 0 <= submitted < len(self.actions):
                raise ValueError('bad catalogue action')
            pairs = self.actions[int(submitted)]
            mask = self.observe()['pair_mask']
            if any(not mask[m, j] for m, j in pairs):
                raise ValueError('masked catalogue action')
            validate_matching(self.config, self.env.state, pairs)
            return pairs, dict(conflicts=0, rejected_proposals=0, executed_starts=len(pairs))
        if self.method not in GRAPH_ARMS:
            return super().resolve(submitted)
        n, k = self.config.machines, self.config.technicians
        tokens = np.asarray(submitted)
        if tokens.shape != (k,) or tokens.dtype.kind not in 'iu' or ((tokens < 0) | (tokens > n)).any():
            raise ValueError('canonical technician token shape/range')
        legal = self.observe()['pair_mask']
        requests = [(int(token) - 1, j) for j, token in enumerate(tokens) if token]
        if any(not legal[m, j] for m, j in requests):
            raise ValueError('masked technician assignment')
        conflicts = rejected = 0
        if self.method == 'independent_ppo':
            pairs = []
            for m in range(n):
                group = [p for p in requests if p[0] == m]
                if group:
                    conflicts += len(group) > 1
                    rejected += len(group) - 1
                    pairs.append(max(group, key=lambda p: (quality(self.config, *p), -p[1])))
        else:
            if len({m for m, _ in requests}) != len(requests):
                raise ValueError('duplicate reserved machine')
            pairs = requests
        pairs = tuple(sorted(pairs))
        validate_matching(self.config, self.env.state, pairs)
        return pairs, dict(conflicts=int(conflicts), rejected_proposals=rejected,
                          executed_starts=len(pairs))
