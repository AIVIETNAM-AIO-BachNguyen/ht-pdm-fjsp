"""Persistent-request adapter for the preregistered diverse maintenance panel."""

from dataclasses import asdict
import hashlib
import json

import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import eligible_edges, role_seed, validate_matching
from ht_pdm_fjsp.maintenance_shared_technician_factorial import (
    dispatch, proposals, terminal_liability,
)
from ht_pdm_fjsp.maintenance_skill_contention_screen import quality
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, observation


VERSION = 'technician_diverse_maintenance_v1'
FAMILIES = ('skill_bottleneck', 'long_service')
SCALES = (8, 16)
GRAPH_ARMS = ('reservation_ppo', 'mask_only_ppo', 'independent_ppo')
HORIZON, TAIL = 48, 12


def family_config(family, n):
    if n not in SCALES or family not in FAMILIES:
        raise ValueError('unregistered diverse configuration')
    k = 3*n//8 if family == 'skill_bottleneck' else n//2
    durations = []
    for m in range(n):
        row = []
        for j in range(k):
            if family == 'skill_bottleneck':
                row.append(1 if m % 3 in (j % 3, (j+1) % 3) else 0)
            else:
                row.append((2 if j % 4 == m % 4 else 4)
                    if j % 4 in (m % 4, (m+1) % 4) else 0)
        durations.append(tuple(row))
    c = WaitingConfig(n, k, HORIZON+TAIL, 4, .45, tuple(durations),
        tuple(tuple(.75 for _ in range(k)) for _ in range(n)),
        max_age=8, preventive_cost=1., corrective_cost=2.,
        unavailable_cost=6., failure_cost=15., waiting_limit=4,
        waiting_price=12.)
    c.validate()
    return c


def graph_observation(c, state, tick, pending, since):
    obs = observation(c, state, tick)
    machines = obs['machines'].copy()
    for m in range(c.machines):
        machines[m, 2] = (tick-since[m]+1)/12 if m in pending else 0.
    obs['machines'] = machines
    edges = obs['edges'].copy()
    edges[edges[..., 0] < .5] = 0
    obs['edges'] = edges
    obs['pair_mask'] = np.asarray(eligible_edges(c, state), dtype=bool) & np.asarray(
        [m in pending for m in range(c.machines)], dtype=bool)[:, None]
    obs['state'] = np.concatenate((machines.ravel(), obs['technicians'].ravel(),
        edges.ravel(), obs['global_features'])).astype(np.float32)
    workers = obs['technicians'].copy(); workers[:, 2] = 0
    obs['local'] = np.concatenate((machines,
        np.tile(workers.ravel(), (c.machines, 1)),
        edges.reshape(c.machines, -1),
        np.tile(obs['global_features'], (c.machines, 1)),
        np.eye(c.machines, dtype=np.float32)), axis=1).astype(np.float32)
    obs['local_mask'] = np.concatenate((np.ones((c.machines, 1), bool),
        obs['pair_mask']), axis=1)
    return obs


class DiverseEnv:
    def __init__(self, seed, n, family, method, worker=0, training=False,
                 journal=None):
        self.config = family_config(family, n)
        self.env = WaitingEnv(self.config)
        if method not in (*GRAPH_ARMS, 'fcfs', 'risk'):
            raise ValueError(method)
        self.seed = int(seed)
        self.family = family
        self.method = method
        self.worker = int(worker)
        self.training = bool(training)
        self.journal = journal
        self.episode_index = -1
        self.active = False

    def _prepare(self):
        c, state, tick = self.config, self.env.state, self.env.time
        if tick >= HORIZON+TAIL:
            return
        active = {m for m in state.assigned if m >= 0}
        new = (proposals(c, state, 'threshold') if tick < HORIZON else
            [m for m in range(c.machines) if state.failed[m] and m not in active])
        for m in new:
            if m not in self.pending and m not in active:
                self.pending.add(m)
                self.since[m] = tick
                self.requests += 1

    def observe(self):
        return graph_observation(self.config, self.env.state, self.env.time,
            self.pending, self.since)

    def reset(self):
        self.episode_index += 1
        namespace = f'{VERSION}:{self.family}:N{self.config.machines}'
        role = (f'{namespace}:training_physical:worker:{self.worker}:episode:{self.episode_index}'
            if self.training else f'{namespace}:physical')
        self.physical_seed = role_seed(self.seed, role)
        self.env.reset(self.physical_seed)
        self.pending, self.since = set(), {}
        self.requests = self.starts = self.completions = 0
        self.physical_cost_accumulated = 0.
        self.active = True
        payload = json.dumps(dict(seed=self.physical_seed,
            state=asdict(self.env.state), events=self.env.events), sort_keys=True)
        self.population = hashlib.sha256(payload.encode()).hexdigest()
        self._prepare()
        return self.observe()

    def resolve(self, submitted):
        c, state = self.config, self.env.state
        if self.method in ('fcfs', 'risk'):
            pairs = dispatch(c, state, self.pending, self.since,
                self.method, self.env.time)
            return pairs, dict(conflicts=0, rejected_proposals=0,
                executed_starts=len(pairs))
        proposal = np.asarray(submitted)
        if (proposal.shape != (c.technicians,) or proposal.dtype.kind not in 'iu'
                or ((proposal < 0) | (proposal > c.machines)).any()):
            raise ValueError('bad technician proposals')
        legal = self.observe()['pair_mask']
        requests = [(int(m)-1, j) for j, m in enumerate(proposal) if m]
        if any(not legal[m, j] for m, j in requests):
            raise ValueError('masked or unrequested proposal')
        if self.method != 'independent_ppo' and len({m for m, _ in requests}) != len(requests):
            raise ValueError('coordinated policy submitted duplicate machine')
        pairs = []
        conflicts = rejected = 0
        for m in range(c.machines):
            group = [p for p in requests if p[0] == m]
            if group:
                conflicts += int(len(group) > 1)
                rejected += len(group)-1
                pairs.append(max(group, key=lambda p: (quality(c, *p), -p[1])))
        pairs = tuple(sorted(pairs))
        validate_matching(c, state, pairs)
        return pairs, dict(conflicts=conflicts,
            rejected_proposals=rejected, executed_starts=len(pairs))

    def _physical_step(self, pairs, phase):
        c, state, tick = self.config, self.env.state, self.env.time
        pairs = tuple(sorted(pairs))
        validate_matching(c, state, pairs)
        if any(m not in self.pending for m, _ in pairs):
            raise RuntimeError('assignment outside persistent request pool')
        active_before = {m for m in state.assigned if m >= 0} | {m for m, _ in pairs}
        for m, _ in pairs:
            self.pending.remove(m)
            del self.since[m]
            self.starts += 1
        _, _, _, info = self.env.step(pairs)
        self.physical_cost_accumulated += info['objective']
        active_after = {m for m in self.env.state.assigned if m >= 0}
        self.completions += len(active_before-active_after)
        for m, failed in enumerate(self.env.state.failed):
            if failed and m not in self.pending and m not in active_after:
                self.pending.add(m)
                self.since[m] = tick+1
                self.requests += 1
        if self.requests != len(self.pending)+len(active_after)+self.completions:
            raise RuntimeError('request conservation')
        if self.journal is not None:
            self.journal.add(dict(seed=self.seed, physical_seed=self.physical_seed,
                family=self.family, tick=tick, phase=phase,
                state=json.dumps(asdict(state)), action=json.dumps(pairs),
                events=json.dumps(self.env.events[tick]),
                following=json.dumps(asdict(self.env.state)),
                cost=info['objective'], pending_after=len(self.pending),
                active_after=len(active_after)))
        self._prepare()
        return info['objective']

    def step(self, submitted):
        if not self.active or self.env.time >= HORIZON:
            raise RuntimeError('reset before policy step')
        pairs, coordination = self.resolve(submitted)
        cost = self._physical_step(pairs, 'policy')
        done = self.env.time == HORIZON
        result = dict(coordination=coordination)
        if done:
            c = self.config
            cutoff = dict(self.env.metrics)
            terminal_pending = len(self.pending)
            terminal_failed = sum(self.env.state.failed)
            for _ in range(TAIL):
                pairs = dispatch(c, self.env.state, self.pending,
                    self.since, 'fcfs', self.env.time)
                cost += self._physical_step(pairs, 'tail')
            liability = terminal_liability(c, self.env.state, self.pending)
            cost += liability
            metrics = dict(self.env.metrics,
                terminal_pending=terminal_pending,
                terminal_failed=terminal_failed,
                tail_pending=len(self.pending),
                tail_failed=sum(self.env.state.failed),
                availability=1-cutoff['unavailable_ticks']/(c.machines*HORIZON),
                technician_utilization=cutoff['service_ticks']/(c.technicians*HORIZON),
                policy_cost=cutoff['objective'],
                tail_cost=self.env.metrics['objective']-cutoff['objective'],
                terminal_liability=liability,
                extended_cost=self.env.metrics['objective']+liability,
                policy_failures=cutoff['failures'],
                policy_overdue_ticks=cutoff['overdue_waiting_ticks'],
                requests=self.requests, starts=self.starts,
                completions=self.completions,
                active_end=sum(m >= 0 for m in self.env.state.assigned),
                physical_ticks=HORIZON+TAIL,
                physical_seed=self.physical_seed,
                population_sha256=self.population,
                episode_index=self.episode_index)
            if abs(self.physical_cost_accumulated-self.env.metrics['objective']) > 1e-8:
                raise RuntimeError('physical cost accumulation')
            if self.requests != self.completions+len(self.pending)+metrics['active_end']:
                raise RuntimeError('terminal request reconciliation')
            if abs(metrics['extended_cost'] - sum(metrics[k] for k in
                ('maintenance_cost','unavailability_cost','failure_cost',
                 'waiting_cost','terminal_liability'))) > 1e-8:
                raise RuntimeError('terminal cost reconciliation')
            result['episode_metrics'] = metrics
            self.active = False
        return self.observe(), -cost/20, done, result
