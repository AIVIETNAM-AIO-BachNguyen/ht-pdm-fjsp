import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_typed_gibbs_optimization as a


def test_uniform_masked_and_only_wait_entropy_is_informative_only():
    eligible=np.array([[True,True,True],[True,False,False],[True,False,True]]);p=eligible/eligible.sum(-1)[:,None];r=a.policy_statistics(p,eligible);assert r['states']==3 and r['informative_states']==2 and r['mean_normalized_entropy']==pytest.approx(1.) and r['near_deterministic_fraction']==0 and r['ambiguous_mode_states']==2
    p=np.array([[1.,0,0],[1.,0,0],[0,0,1.]]);r=a.policy_statistics(p,eligible);assert r['mean_normalized_entropy']==0 and r['near_deterministic_fraction']==1 and r['ambiguous_mode_states']==0
    with pytest.raises(AssertionError):a.policy_statistics(np.ones((3,3)),eligible)


def test_fixed_state_tv_and_near_tie_mode_changes_are_not_inflated():
    before=np.array([[.5-1e-12,.5+1e-12,0],[.99,.01,0],[1,0,0]]);after=np.array([[.5+1e-12,.5-1e-12,0],[.01,.99,0],[1,0,0]]);eligible=np.array([[True,True,False],[True,True,False],[True,False,False]]);r=a.movement(before,after,eligible);assert r['unique_mode_comparisons']==1 and r['unique_mode_changes']==1 and r['late_mean_tv']==pytest.approx(.49,abs=1e-10)


def late_fixture():
    return [dict(method=m,train_seed=t,previous_cfdev_cost=100.,final_cfdev_cost=99.,late_cfdev_cost_gain=.01,informative_states=200,near_deterministic_fraction=.95,late_mean_tv=.02) for m in a.CORE for t in range(5)]


def test_bounded_budget_readout_and_collapse_have_locked_distinct_thresholds():
    rows=late_fixture();r=a.decisions(rows);assert r['permit_designing_one_2x_budget_followup'] and not r['training_launch_authorized_by_this_artifact'] and not any(r['possible_probability_collapse'].values())
    rows[0].update(late_cfdev_cost_gain=0,final_cfdev_cost=100.,late_mean_tv=.01,near_deterministic_fraction=.9);r=a.decisions(rows);assert r['gibbs_positive_late_replicas']==4 and r['possible_probability_collapse'][f'{a.CORE[0]}:0'] and r['permit_designing_one_2x_budget_followup']
    rows[1]['late_cfdev_cost_gain']=0;assert not a.decisions(rows)['permit_designing_one_2x_budget_followup']
    with pytest.raises(AssertionError):a.decisions(rows[1:])


def test_budget_readout_rejects_insufficient_movement_or_pooled_progress():
    rows=late_fixture()
    for r in rows:r['late_mean_tv']=.009999
    assert not a.decisions(rows)['permit_designing_one_2x_budget_followup']
    rows=late_fixture()
    for r in rows:r.update(final_cfdev_cost=99.9,late_cfdev_cost_gain=.001)
    assert not a.decisions(rows)['permit_designing_one_2x_budget_followup']
