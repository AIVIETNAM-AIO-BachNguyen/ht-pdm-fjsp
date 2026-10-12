"""Masked Gym joint-action probe with a common physical continuation tail."""
from dataclasses import asdict, replace
import json

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ht_pdm_fjsp import maintenance_skill_contention_screen as screen
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, observation

POLICY_HORIZON = 48
TAIL_TICKS = 12
REWARD_SCALE = 20.0


class JointLearningEnv(gym.Env):
    """48 learned decisions followed by 12 fixed deadline-matching ticks.

    The final reward includes the complete tail cost, preserving the H+L
    undiscounted objective. Training episodes use worker/episode-separated
    physical streams; explicit evaluation raw seeds preserve policy pairing.
    Observations and masks depend exclusively on configuration/current state.
    """
    metadata = {"render_modes": []}

    def __init__(self, config, namespace='maintenance_joint_learning_mac_v1',
                 worker=0, training=False, journal=None):
        super().__init__()
        config.validate()
        if (config.machines, config.technicians, config.horizon) != (4, 2, 60):
            raise ValueError('joint learning probe requires N4/K2/H48+L12')
        if not isinstance(namespace, str) or not namespace or not isinstance(worker, int) or worker < 0:
            raise ValueError('nonempty namespace and nonnegative worker required')
        self.config = config
        self.observation_config = replace(config, horizon=POLICY_HORIZON)
        self.namespace, self.worker, self.training = namespace, worker, bool(training)
        self.journal = journal
        self.env = WaitingEnv(config)
        self.actions = catalogue(4, 2)
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(0., np.finfo(np.float32).max,
            shape=(71,), dtype=np.float32)
        self.policy_time = 0
        self.physical_seed = None
        self.raw_seed = None
        self.stream_seed = None
        self.episode_index = None
        self._next_episode = 0
        self._active = False
        self.episode_metrics = None

    def _observation(self):
        current = observation(self.observation_config, self.env.state, self.policy_time)
        edges = current['edges'].copy()
        edges *= (edges[:, :, :1] > 0)
        result = np.concatenate([current['machines'].ravel(), current['technicians'].ravel(),
            edges.ravel(), current['global_features'].ravel()]).astype(np.float32)
        if not self.observation_space.contains(result):
            raise RuntimeError('nonfinite or out-of-contract observation')
        return result

    def action_masks(self):
        legal = set(screen.action_space(self.config, self.env.state)[2])
        return np.asarray([action in legal for action in self.actions], dtype=bool)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('reset options are not supported')
        if seed is not None and (not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError('nonnegative integer seed required')
        if self.training:
            if seed is not None:
                self.stream_seed = int(seed)
                self._next_episode = 0
            if self.stream_seed is None:
                raise ValueError('training requires an initial explicit stream seed')
            self.episode_index = self._next_episode
            self._next_episode += 1
            self.raw_seed = None
            self.physical_seed = role_seed(self.stream_seed,
                f'{self.namespace}:training_physical:worker:{self.worker}:episode:{self.episode_index}')
        else:
            if seed is None:
                raise ValueError('evaluation requires an explicit raw seed')
            self.raw_seed = int(seed)
            self.stream_seed = None
            self.episode_index = 0
            self.physical_seed = role_seed(self.raw_seed, f'{self.namespace}:physical')
        super().reset(seed=int(seed) if seed is not None else None)
        self.env.reset(self.physical_seed)
        self.policy_time = 0
        self._active = True
        self.episode_metrics = None
        return self._observation(), self._seed_info()

    def _seed_info(self):
        return dict(physical_seed=self.physical_seed, raw_seed=self.raw_seed,
            stream_seed=self.stream_seed, episode_index=self.episode_index, worker=self.worker)

    def _physical_step(self, action, phase):
        tick, state = self.env.time, self.env.state
        validate_matching(self.config, state, action)
        if action not in screen.action_space(self.config, state)[2]:
            raise ValueError('masked physical action')
        events = self.env.events[tick]
        _, _, _, info = self.env.step(action)
        if self.journal is not None:
            def compact(value):
                return json.dumps(value, separators=(',', ':'), allow_nan=False)
            self.journal.add(dict(**self._seed_info(), training=self.training, phase=phase,
                tick=tick, policy_time=self.policy_time, state=compact(asdict(state)), action=compact(action),
                events=compact(events), following=compact(asdict(self.env.state)), cost=float(info['objective']), info=compact(info)))
        return info

    def step(self, action):
        if not self._active:
            raise RuntimeError('reset required before stepping a finished episode')
        if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError('invalid discrete action')
        action = int(action)
        if not self.action_masks()[action]:
            raise ValueError('masked action')
        info = self._physical_step(self.actions[action], 'policy')
        step_cost = float(info['objective'])
        self.policy_time += 1
        terminated = self.policy_time == POLICY_HORIZON
        if terminated:
            cutoff_metrics = dict(self.env.metrics)
            cutoff_pending = screen.pending(self.config, self.env.state)
            cutoff_failed = sum(self.env.state.failed)
            for _ in range(TAIL_TICKS):
                tail = screen.choose(self.config, self.env.state, 'deadline_matching')
                tail_info = self._physical_step(tail, 'tail')
                step_cost += float(tail_info['objective'])
            metrics = dict(self.env.metrics)
            metrics.update(**self._seed_info(), availability=1-cutoff_metrics['unavailable_ticks']/(4*POLICY_HORIZON),
                technician_utilization=cutoff_metrics['service_ticks']/(2*POLICY_HORIZON),
                terminal_pending=cutoff_pending, terminal_failed=cutoff_failed,
                tail_pending=screen.pending(self.config, self.env.state), tail_failed=sum(self.env.state.failed),
                policy_cost=cutoff_metrics['objective'], tail_cost=metrics['objective']-cutoff_metrics['objective'],
                extended_cost=metrics['objective'], physical_ticks=self.env.time,
                policy_failures=cutoff_metrics['failures'],
                policy_overdue_ticks=cutoff_metrics['overdue_waiting_ticks'])
            self.episode_metrics = metrics
            self._active = False
            info = {**info, 'episode_metrics': dict(metrics)}
        info = {**info, **self._seed_info(), 'step_cost': step_cost, 'physical_ticks': self.env.time}
        return self._observation(), -step_cost/REWARD_SCALE, terminated, False, info
