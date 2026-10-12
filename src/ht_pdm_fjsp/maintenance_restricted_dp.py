"""Economic isolated-machine DP surrogate with the benchmark's urgent actions.

Future resource availability is independent for each machine, so this is a
cheap surrogate, not a joint optimal value. Shared current occupancy is obeyed
by the matching decoder. All future DP starts also require failure or an age
of at least failure_age - 1. Failed waiting is capped at waiting_limit: at and
above that coordinate, every further unserved tick incurs the same overdue
price, while service completion resets waiting to zero.
"""

from functools import lru_cache
import hashlib
import math

import numpy as np

from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays, membership

NAMES = ("isolated_h6", "isolated_endpoint")


class RestrictedDP:
    def __init__(self, config):
        config.validate()
        self.config = config
        c = config
        shape = (c.horizon + 1, c.machines, c.max_age + 1, 2, c.waiting_limit + 1)
        self.values = np.zeros(shape, dtype=np.float64)
        self.gains = np.full((*shape, c.technicians), -np.inf, dtype=np.float64)
        for h in range(1, c.horizon + 1):
            for m in range(c.machines):
                for age in range(c.max_age + 1):
                    for failed in (False, True):
                        for wait in range(c.waiting_limit + 1):
                            idle = self._wait(h, m, age, failed, wait)
                            best = idle
                            if failed or age >= c.failure_age - 1:
                                for j, duration in enumerate(c.service_time[m]):
                                    if duration:
                                        start = (c.corrective_cost if failed else c.preventive_cost)
                                        start += self._busy(h, m, age, duration, j)
                                        self.gains[h, m, age, int(failed), wait, j] = idle - start
                                        best = min(best, start)
                            self.values[h, m, age, int(failed), wait] = best
        self.values.flags.writeable = False
        self.gains.flags.writeable = False

    def _wait(self, h, m, age, failed, wait):
        c = self.config
        if failed:
            return (c.unavailable_cost + c.waiting_price * (wait + 1 > c.waiting_limit)
                + self.values[h - 1, m, age, 1, min(wait + 1, c.waiting_limit)])
        following_age = min(c.max_age, age + 1)
        probability = c.failure_probability if following_age >= c.failure_age else 0.0
        return (probability * (c.failure_cost + self.values[h - 1, m, following_age, 1, 0])
            + (1 - probability) * self.values[h - 1, m, following_age, 0, 0])

    def _busy(self, h, m, age, remaining, worker):
        c = self.config
        cost = c.unavailable_cost * min(h, remaining)
        if remaining < h:
            restored = math.floor(age * (1 - c.restoration[m][worker]))
            cost += self.values[h - remaining, m, restored, 0, 0]
        return float(cost)

    def value(self, h, m, age, failed=False, wait=0, remaining=0, worker=-1):
        """Expected optimal isolated continuation; ongoing service has no start fee."""
        self._horizon(h, allow_zero=True)
        c = self.config
        if not 0 <= m < c.machines or not 0 <= age <= c.max_age or wait < 0:
            raise ValueError("invalid machine state")
        if remaining:
            if remaining < 0 or not 0 <= worker < c.technicians or not c.service_time[m][worker]:
                raise ValueError("invalid ongoing service")
            return self._busy(h, m, age, remaining, worker)
        return float(self.values[h, m, age, int(failed), min(wait, c.waiting_limit)])

    def _horizon(self, h, allow_zero=False):
        if not isinstance(h, (int, np.integer)) or not (0 if allow_zero else 1) <= h <= self.config.horizon:
            raise ValueError("remaining horizon outside precomputed tables")

    def batch(self, states, h, name):
        """Decode legal partial matchings, including WAIT, for a vector of states."""
        self._horizon(h)
        if name not in NAMES:
            raise ValueError(name)
        c = self.config
        horizon = min(h, 6) if name == "isolated_h6" else h
        age = np.asarray(states["ages"])
        failed = np.asarray(states["failed"])
        wait = np.asarray(states["pending_wait"])
        remaining = np.asarray(states["remaining"])
        assigned = np.asarray(states["assigned"])
        b, n = age.shape
        if n != c.machines or failed.shape != age.shape or wait.shape != age.shape or remaining.shape != (b, c.technicians) or assigned.shape != remaining.shape:
            raise ValueError("state array dimensions")
        if np.any((age < 0) | (age > c.max_age)) or np.any(wait < 0):
            raise ValueError("invalid age or waiting coordinate")
        occupied = np.zeros((b, n), dtype=bool)
        for j in range(c.technicians):
            machine = assigned[:, j]
            used = machine >= 0
            occupied[np.arange(b)[used], machine[used]] = True
        valid = ((~occupied)[:, :, None] & (remaining == 0)[:, None, :]
            & (np.asarray(c.service_time) > 0)[None]
            & (failed | (age >= c.failure_age - 1))[:, :, None])
        weights = self.gains[horizon, np.arange(n)[None], age, failed.astype(np.int64),
            np.minimum(wait, c.waiting_limit)]
        valid &= np.isfinite(weights)
        members = membership(n, c.technicians)
        scores = np.einsum("amj,bmj->ba", members, np.where(valid, weights, 0.0), optimize=True)
        illegal = np.einsum("amj,bmj->ba", members, ~valid, optimize=True) > 0
        indices = np.where(illegal, -np.inf, scores).argmax(axis=1)
        return members[indices].astype(bool)

    def select(self, state, h, name):
        chosen = self.batch(arrays(state), h, name)[0]
        return tuple((int(m), int(j)) for m, j in zip(*np.nonzero(chosen)))

    def fingerprints(self):
        return {"values": hashlib.sha256(self.values.tobytes()).hexdigest(),
            "gains": hashlib.sha256(self.gains.tobytes()).hexdigest()}


@lru_cache(maxsize=32)
def tables(c):
    return RestrictedDP(c)


def batch(c, states_arrays, h, name):
    return tables(c).batch(states_arrays, h, name)


def select(c, state, h, name):
    return tables(c).select(state, h, name)
