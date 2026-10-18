import math

import pytest

from ht_pdm_fjsp import maintenance_shared_technician_factorial as old
from ht_pdm_fjsp.maintenance_coupled_dispatch_followup import (
    VERSION, add_requests, candidate_actions, forecast_step, forecast_total,
    matrix_config, physical_seed, run_episode, scenario_seed, scenario_tapes,
    seed_audit, select_baseline,
)
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv


@pytest.mark.parametrize("config_id", ("hub_dev", "balanced_dev", "dense3_dev"))
@pytest.mark.parametrize("timing", ("threshold", "capacity_aware"))
@pytest.mark.parametrize("dispatcher", ("fcfs", "risk", "myopic", "isolated"))
def test_simple_episode_exactly_reuses_v1_mechanics(config_id,timing,dispatcher):
    seed=94000000
    actual=run_episode(config_id,timing,dispatcher,seed,"smoke","fcfs",2)
    expected=old.episode(config_id,timing,"lookahead" if dispatcher=="isolated" else dispatcher,
        seed,config_fn=matrix_config,version=VERSION)
    for key,value in expected.items():
        if key=="dispatcher":
            assert actual[key]==dispatcher
        elif isinstance(value,float):
            assert actual[key]==pytest.approx(value,abs=1e-8)
        else:
            assert actual[key]==value


def test_forecast_exact_replay_with_supplied_physical_tape():
    config_id="balanced_dev"
    seed=94000000
    c=matrix_config(config_id)
    env=WaitingEnv(c)
    env.reset(physical_seed(config_id,seed))
    pending,since=set(),{}
    add_requests(c,env.state,pending,since,"threshold",0)
    action=old.dispatch(c,env.state,pending,since,"fcfs",0)
    forecast=forecast_total(c,env.state,pending,since,"threshold","fcfs",0,
        action,env.events)
    actual=old.episode(config_id,"threshold","fcfs",seed,
        config_fn=matrix_config,version=VERSION)
    assert forecast==pytest.approx(actual["cost"],abs=1e-8)


def test_scenario_banks_and_candidate_legality():
    assert seed_audit()["status"]=="PASS"
    config_id="hub_dev";seed=94000000;tick=3
    assert len({physical_seed(config_id,seed),
        scenario_seed(config_id,seed,tick,0,"decision"),
        scenario_seed(config_id,seed,tick,0,"diagnostic")})==3
    c=matrix_config(config_id)
    tapes=scenario_tapes(c,config_id,seed,tick,"decision",4)
    assert len(tapes)==4 and all(len(x)==60-tick for x in tapes)
    env=WaitingEnv(c);env.reset(physical_seed(config_id,seed))
    state=env.state
    pending=set(range(c.machines));since={m:0 for m in pending}
    baseline,candidates=candidate_actions(c,state,pending,since,0,"fcfs")
    assert baseline in candidates and () in candidates and len(candidates)<=10
    for action in candidates:
        assert len({m for m,_ in action})==len(action)
        assert len({j for _,j in action})==len(action)
        assert all(c.service_time[m][j]>0 for m,j in action)


def test_coupled_episode_conserves_requests_and_has_finite_cost():
    row=run_episode("hub_dev","capacity_aware","coupled_rollout",94000000,
        "smoke","fcfs",2)
    assert row["ticks"]==60 and math.isfinite(row["cost"])
    assert row["requests"]==row["completions"]+row["pending_end"]+row["active_end"]
    assert row["invalid_assignments"]==0
    assert row["scenario_trajectories"]>0


def test_forecast_preserves_committed_work_and_pending_conversion():
    c=matrix_config("balanced_dev")
    state=DispatchState((3,4,3,3),(False,True,False,False),(0,2,0,0),
                        (1,0),(0,-1))
    following,pending,since,cost,info=forecast_step(c,state,{1},{1:0},
        ((1,1),),(False,False,False,False),0)
    assert following.assigned==(-1,1)
    assert following.remaining==(0,2)
    assert following.ages[0]==0
    assert following.failed[1]
    assert pending==set() and since=={}
    assert cost==info["objective"]
    assert info["corrective_jobs"]==1


def test_development_selection_uses_all_config_seed_timing_rows():
    rows=[]
    for config_id in ("a","b"):
        for seed in (1,2):
            for timing in ("threshold","capacity_aware"):
                for dispatcher,cost in (("fcfs",10.),("risk",10.),("myopic",11.)):
                    rows.append(dict(condition=config_id,seed=seed,timing=timing,
                                     dispatcher=dispatcher,cost=cost))
    selection=select_baseline(rows,("a","b"),(1,2))
    assert selection["selected"]=="fcfs"  # registered tie order
    with pytest.raises(RuntimeError,match="incomplete"):
        select_baseline(rows[:-1],("a","b"),(1,2))
