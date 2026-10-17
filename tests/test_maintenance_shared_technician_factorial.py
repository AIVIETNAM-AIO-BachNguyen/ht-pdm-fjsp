import json
import math

import pytest

from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_shared_technician_factorial import (
    H, TAIL, audit_rows, config, dispatch, episode, panel, proposals,
    seed_audit, terminal_liability,
)
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv


def test_persistent_request_proposals_respect_occupancy_and_capacity():
    c = config("scarce_overlap")
    s = DispatchState((1, 2, 3, 3), (False, False, True, False),
                      (0, 0, 2, 0), (3, 0), (0, -1))
    assert proposals(c, s, "reactive") == [2]
    assert proposals(c, s, "threshold") == [2, 3]
    assert proposals(c, s, "capacity_aware") == [2, 3]
    busy = DispatchState((1, 2, 3, 3), (False, False, True, False),
                         (0, 0, 2, 0), (3, 2), (0, 3))
    assert proposals(c, busy, "capacity_aware") == [1, 2]


def test_skill_capacity_and_service_completion():
    c = config("scarce_overlap")
    s = DispatchState((3, 3, 3, 3), (False,) * 4, (0,) * 4,
                      (0, 0), (-1, -1))
    pairs = dispatch(c, s, {0, 1, 2}, {0: 0, 1: 1, 2: 2}, "fcfs", 0)
    assert pairs == ((0, 0), (1, 1))
    env = WaitingEnv(c)
    env.reset(3, s)
    env.step(pairs)
    assert env.state.assigned == (-1, 1)
    assert env.state.remaining == (0, 2)
    assert env.state.ages[0] == 0
    assert env.state.ages[1] == 3  # no restoration until completion
    with pytest.raises(ValueError):
        env.step(((2, 1),))  # committed resource cannot be reassigned
    env.step(())
    env.step(())
    assert env.state.assigned == (-1, -1)
    assert env.state.ages[1] == 0


def test_healthy_pending_machine_degrades_and_failed_demand_persists():
    c = config("partitioned")
    s = DispatchState((3, 1, 1, 1), (False,) * 4, (0,) * 4,
                      (2, 0), (1, -1))
    env = WaitingEnv(c)
    env.reset(8, s)
    before = env.state.ages[0]
    env.step(())
    assert env.state.ages[0] == before + 1
    # The wrapper's request pool is independent of the physical state; failing
    # cannot make an already pending request ineligible for later dispatch.
    pending = {0}
    if env.state.failed[0]:
        assert 0 in pending
    assert dispatch(c, env.state, pending, {0: 0}, "fcfs", 1) == ()


def test_terminal_liability_covers_pending_and_committed_jobs():
    c = config("scarce_overlap")
    s = DispatchState((3, 3, 3, 3), (True, False, False, False),
                      (2, 0, 0, 0), (2, 0), (2, -1))
    assert terminal_liability(c, s, {0, 1}) == (2 + 6 + 12) + (1 + 6) + 12


def test_episode_trace_conservation_and_cost():
    class Trace:
        rows = []
        def writerow(self, row):
            self.rows.append(row)
    trace = Trace()
    r = episode("scarce_overlap", "capacity_aware", "lookahead", 91000000, trace)
    assert r["ticks"] == H + TAIL
    assert r["requests"] == r["completions"] + r["pending_end"] + r["active_end"]
    assert len(trace.rows) == H + TAIL
    assert math.isclose(r["cost"], r["policy_cost"] + r["tail_cost"] + r["terminal_liability"])
    assert all(len(json.loads(x["assignment"])) <= 2 for x in trace.rows)
    assert r["invalid_assignments"] == 0


def test_panel_and_artifact_gate():
    assert seed_audit()["status"] == "PASS"
    assert len(panel("smoke")["smoke"]) == 2
    with pytest.raises(RuntimeError, match="incomplete"):
        audit_rows([], panel("smoke"))
