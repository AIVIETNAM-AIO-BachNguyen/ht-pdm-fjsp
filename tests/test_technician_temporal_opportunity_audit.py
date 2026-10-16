import itertools,json
from collections import defaultdict
from dataclasses import asdict
import numpy as np
import pytest
from ht_pdm_fjsp import technician_temporal_opportunity_audit as d
from ht_pdm_fjsp import technician_broker_marginal_experiment as x
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import transition

EVENTS=list(itertools.product((False,True),repeat=4))
def probability(e):return .45**sum(e)*.55**(4-sum(e))
def fixtures():
    return [dict(ages=[3,1,3,1],failed=[False]*4,pending_wait=[0]*4,remaining=[0,0],assigned=[-1,-1]),
            dict(ages=[5,4,2,1],failed=[True,False,False,False],pending_wait=[5,0,0,0],remaining=[0,1],assigned=[-1,2]),
            dict(ages=[7,3,2,5],failed=[True,False,False,True],pending_wait=[4,0,0,7],remaining=[3,0],assigned=[3,-1]),
            dict(ages=[0]*4,failed=[False]*4,pending_wait=[0]*4,remaining=[0,0],assigned=[-1,-1])]


def independent_actions(s):
    result=[];busy={m for m in s['assigned'] if m>=0}
    for a,b in itertools.product(range(-1,4),repeat=2):
        if a==b and a>=0:continue
        pairs=tuple(sorted([(a,0)] if a>=0 else [])+sorted([(b,1)] if b>=0 else []));pairs=tuple(sorted(pairs))
        if all(s['remaining'][j]==0 and m not in busy and (s['failed'][m] or s['ages'][m]>=3) for m,j in pairs):result.append(pairs)
    return sorted(result)


def scalar_q(s,a,horizon):
    cfg=x.scaled_config(4);total=0.
    for e in EVENTS:
        following,cost,_=transition(cfg,DispatchState(**{k:tuple(v) for k,v in s.items()}),a,e);following={k:list(v) for k,v in asdict(following).items()}
        total+=probability(e)*(cost+(min(scalar_q(following,b,horizon-1) for b in independent_actions(following)) if horizon>1 else 0))
    return total


@pytest.mark.parametrize('s',fixtures())
def test_sparse_failure_outcomes_match_every_scalar_event_and_component(s):
    engine=d.Exact(d.helpers());assert engine.actions(s)==independent_actions(s)
    for a in engine.actions(s):
        groups=defaultdict(lambda:[0.,0.])
        for e in EVENTS:
            following,cost,parts=transition(x.scaled_config(4),DispatchState(**{k:tuple(v) for k,v in s.items()}),a,e);k=d.key(asdict(following));groups[k][0]+=probability(e);groups[k][1]+=probability(e)*cost
            assert parts['objective']==cost
        sparse={k:(p,p*c) for p,c,k in engine.outcomes(s,a)}
        assert set(sparse)==set(groups)
        for k in groups:np.testing.assert_allclose(sparse[k],groups[k],rtol=2e-14,atol=2e-13)
        assert sum(v[0] for v in sparse.values())==pytest.approx(1,abs=2e-14)


@pytest.mark.parametrize('s',fixtures()[:2])
@pytest.mark.parametrize('horizon',[1,2])
def test_bellman_minimum_against_independent_full_scalar_tree(s,horizon):
    engine=d.Exact(d.helpers());expected={a:scalar_q(s,a,horizon) for a in independent_actions(s)}
    for a,q in expected.items():assert engine.q(s,a,horizon)==pytest.approx(q,abs=2e-10,rel=2e-12)
    assert engine.value(d.key(s),horizon)==pytest.approx(min(expected.values()),abs=2e-10,rel=2e-12)


def test_actual_engineering_state_key_roundtrip_and_fully_forced_wait():
    env=x.ResourceEnv(290700001,4,x.CORE_ARMS[0]);env.reset();s={k:list(v) for k,v in asdict(env.env.state).items()};assert d.state(d.key(s))==s
    engine=d.Exact(d.helpers());s=fixtures()[-1];result=engine.evaluate(s,());assert result['isolated_tokens']==[0,0] and result['conditional_tokens']==[0,0] and result['legal_matchings']==1
    assert result['joint_q']>=0 and result['joint_q']==pytest.approx(result['source_q'])


def test_conditional_choice_uses_accepted_first_assignment_and_does_not_retry_it():
    class ExactFixture(d.Exact):
        def q(self,s,a,horizon):return {():10.,((0,0),):5.,((0,1),):4.,((0,0),(1,1)):1.}.get(a,11.)
    engine=ExactFixture(d.helpers());s=dict(ages=[3]*4,failed=[False]*4,pending_wait=[0]*4,remaining=[0,0],assigned=[-1,-1]);r=engine.evaluate(s,())
    assert r['isolated_tokens']==[1,1] and r['isolated_pairs']==((0,0),)
    assert r['conditional_tokens']==[1,2] and r['conditional_pairs']==((0,0),(1,1)) and r['conditional_q']==1


def test_full_h4_busy_completion_and_overdue_cost_has_known_exact_value():
    s=dict(ages=[4]*4,failed=[True]*4,pending_wait=[0,0,3,4],remaining=[4,4],assigned=[0,1])
    # Four unavailable machines for four ticks; unserved machines incur
    # overdue penalties on three and four ticks respectively. No new starts.
    engine=d.Exact(d.helpers());assert engine.value(d.key(s),4)==276.
    assert engine.q(s,(),4)==276.


def test_expansion_guard_and_locked_screen_rejection():
    engine=d.Exact(d.helpers(),limit=0)
    with pytest.raises(AssertionError,match='hard cap'):engine.value(d.key(fixtures()[0]),1)
    rows=[dict(origin=a,train_seed=t,seed=s,tick=tick,isolated_q=100.,conditional_q=99.,joint_q=98.,source_q=100.) for a in d.ORIGINS for t in d.TRAINS for s in d.SEEDS for tick in d.TICKS]
    assert d.summarize(rows)['conditional_q_mechanism_further_design_motivated'];json.dumps(d.summarize(rows),allow_nan=False)
    for r in rows:r['conditional_q']=99.6
    assert not d.summarize(rows)['conditional_q_mechanism_further_design_motivated']
