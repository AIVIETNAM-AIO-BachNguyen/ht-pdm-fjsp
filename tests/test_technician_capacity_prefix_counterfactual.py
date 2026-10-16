import copy
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_capacity_factorized_experiment as x
from ht_pdm_fjsp.technician_capacity_prefix_counterfactual import (
    force_first,shortest_first,finish,equal_observation,summarize,BRANCHES)
from ht_pdm_fjsp.technician_capacity_service_choice_diagnostic import core_trace


def model_and_obs():
    torch.set_num_threads(1);torch.manual_seed(21)
    return x.FactorizedCapacityPolicy('residual_capacity_ppo'),x.mechanism_states('engineering')[0]['observation']


def test_force_first_preserves_noop_weights_methods_and_recomputes_reservation():
    m,obs=model_and_obs();a,traces=core_trace(m,obs);before=x.learner.parameter_digest(m)
    assert np.array_equal(force_first(m,obs,int(a[0])),a)
    forced=force_first(m,obs,1)
    assert forced[0]==1 and forced[1]!=1
    assert 'conditionals' not in m.__dict__ and x.learner.parameter_digest(m)==before
    with pytest.raises(ValueError):force_first(m,obs,4)
    assert 'conditionals' not in m.__dict__
    with pytest.raises(ValueError):force_first(m,obs,-1)
    assert shortest_first(obs)==1


def test_hook_restores_existing_override_even_on_error():
    m,obs=model_and_obs();saved=m.conditionals;m.conditionals=saved
    force_first(m,obs,0);assert m.__dict__['conditionals'] is saved
    with pytest.raises(ValueError):force_first(m,obs,4)
    assert m.__dict__['conditionals'] is saved


def test_private_replay_exact_tail_and_source_unchanged():
    m,_=model_and_obs();env=x.ResourceEnv(230700001,4,'residual_capacity_ppo');obs=env.reset()
    before=copy.deepcopy(env.env.state);metrics=copy.deepcopy(env.env.metrics)
    saved=[];reference=copy.deepcopy(env)
    for tick in range(48):
        o=reference.observe();saved.append(o)
        _,_,done,info=reference.step(m.select(o))
    result=finish(env,m,m.select(obs),48,expected=saved)
    assert result==info['episode_metrics'] and result['physical_ticks']==60
    assert env.env.state==before and env.env.metrics==metrics and env.env.time==0
    broken=copy.deepcopy(saved);broken[0]['state'][0]+=1
    with pytest.raises(RuntimeError,match='observation'):finish(env,m,m.select(obs),48,expected=broken)


def test_busy_first_forces_wait_and_shortest_uses_legal_only():
    m,obs=model_and_obs();obs=copy.deepcopy(obs);obs['pair_mask'][:,0]=False
    assert shortest_first(obs)==0 and force_first(m,obs,0)[0]==0
    with pytest.raises(ValueError):force_first(m,obs,1)


def synthetic_rows():
    return [dict(train_seed=t,state_index=i,branch=b,original_duration=4,original_token=1,
                 token=0 if b=='wait' else 1,gain_per_machine=2 if b=='wait' else 0,
                 availability_delta=0,pending_delta_per_machine=0,failures_delta_per_machine=0)
            for t in range(3) for i in range(2) for b in BRANCHES]


def test_summary_all_replicas_health_and_missing_grid():
    rows=synthetic_rows();s=summarize(rows)
    assert s['deferral_explanation_supported'] and s['strata']['four_tick']['wait']['improved']==6
    for r in rows:
        if r['branch']=='wait' and r['train_seed']==0:r['gain_per_machine']=-1
    assert not summarize(rows)['engineering_checks']['every_replica_positive']
    rows=synthetic_rows()
    for r in rows:
        if r['branch']=='wait':r['availability_delta']=-.02
    assert not summarize(rows)['engineering_checks']['availability']
    with pytest.raises(ValueError):summarize(rows[:-1])
    with pytest.raises(ValueError):summarize(rows+[rows[0]])


def test_observation_keys_and_values_are_exact():
    _,obs=model_and_obs();equal_observation(obs,copy.deepcopy(obs))
    other=copy.deepcopy(obs);other.pop('state')
    with pytest.raises(RuntimeError):equal_observation(obs,other)
