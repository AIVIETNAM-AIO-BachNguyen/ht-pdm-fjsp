import importlib.util,itertools
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_joint_clearing_experiment as x
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import transition

spec=importlib.util.spec_from_file_location('joint_verifier',Path(__file__).resolve().parents[1]/'scripts/verify_joint_clearing_development.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def test_top_two_bound_matches_all_unit_test_matchings():
    rng=np.random.default_rng(81)
    for _ in range(100):
        q=rng.integers(-3,20,(4,2));mask=rng.random((4,2))>.3;values=[]
        for a,b in itertools.product(range(-1,4),repeat=2):
            if a==b and a>=0:continue
            if (a>=0 and not mask[a,0]) or (b>=0 and not mask[b,1]):continue
            values.append((int(q[a,0]) if a>=0 else 0)+(int(q[b,1]) if b>=0 else 0))
        assert v.top_two_optimum(q,mask)==max(values)


def test_independent_initial_matches_all_actual_random_constructors():
    torch.set_num_threads(1);c=x.settings('engineering');t=c['train_seeds'][0]
    for a in x.ARMS:
        expected=v.independent_initial(a,t);actual=x.make_model(a,c,t).state_dict()
        assert list(expected)==list(actual) and all(torch.equal(expected[k],actual[k]) for k in expected)


def test_independent_observation_primitive_events_and_all_decoders():
    torch.set_num_threads(1);env=x.ResourceEnv(250700001,4,x.CORE_ARMS[0]);env.reset()
    state,events,population=v.initial_events(env.physical_seed)
    assert state=={k:list(a) for k,a in asdict(env.env.state).items()} and population==env.population
    assert events==list(map(list,env.env.events))
    rng=np.random.default_rng(8);current=env.env.state
    for tick in range(60):
        state={k:list(a) for k,a in asdict(current).items()};env.env.state=current;env.env.time=tick
        expected=v.observation(state,tick);actual=env.observe()
        assert all(np.array_equal(actual[k],expected[k]) for k in expected)
        mask=expected['pair_mask'];u=np.where(mask,rng.normal(size=(4,2)),0)
        for arm in x.CORE_ARMS:
            pairs,diag=x.reference.np.asarray([]),None
            from ht_pdm_fjsp.technician_joint_clearing_policy import decode_utility
            pairs,diag=decode_utility(u,mask,arm);v.verify_decoder(u,mask,arm,pairs,diag)
            following,metrics=v.primitive_step(state,pairs,events[tick]);reference,cost,components=transition(env.config,current,pairs,events[tick])
            assert following=={k:list(a) for k,a in asdict(reference).items()} and metrics==components and metrics['objective']==cost
            if arm==x.CORE_ARMS[0]:advance=reference
        current=advance


def test_overdue_charge_and_one_tick_failed_service_completion():
    state=dict(ages=[4,0,0,0],failed=[True,False,False,False],pending_wait=[4,0,0,0],remaining=[0,0],assigned=[-1,-1])
    following,m=v.primitive_step(state,(),[False]*4)
    assert m['waiting_cost']==12 and m['objective']==24 and following['pending_wait'][0]==5
    following,m=v.primitive_step(state,((0,0),),[False]*4)
    assert m['waiting_cost']==0 and m['corrective_jobs']==1 and m['objective']==14
    assert not following['failed'][0] and following['ages'][0]==2 and following['pending_wait'][0]==0


@pytest.mark.parametrize('arm',x.ARMS)
def test_numpy_policy_forward_and_existing_baseline_interface(arm):
    torch.set_num_threads(1);c=x.settings('engineering');t=c['train_seeds'][0];model=x.make_model(arm,c,t)
    env=x.ResourceEnv(250700001,4,arm);env.env.state=DispatchState((3,4,5,6),(False,True,False,False),(0,2,0,0),(0,0),(-1,-1));env.env.time=7
    obs=env.observe();action=model.select(obs);state={k:list(a) for k,a in asdict(env.env.state).items()};independent=v.observation(state,7)
    weights={k:a.numpy().astype(float) for k,a in model.state_dict().items()}
    assert v.verify_actor([independent]*2,[action]*2,arm,weights)<1e-5
    pairs,diag=env.resolve(action)
    if arm in x.CORE_ARMS:v.verify_decoder(action,independent['pair_mask'],arm,pairs,diag)
    else:v.verify_baseline(int(action) if np.isscalar(action) else action.tolist(),independent['pair_mask'],arm,state,pairs)


def test_core_verifier_uses_no_catalogue_and_rejects_wrong_matching(monkeypatch):
    def forbidden():raise AssertionError('candidate verification enumerated matchings')
    monkeypatch.setattr(v,'baseline_catalogue',forbidden)
    from ht_pdm_fjsp.technician_joint_clearing_policy import decode_utility
    mask=np.ones((4,2),bool);u=np.array([[9.,8.],[8.,0.],[0,0],[0,0]])
    pairs,diag=decode_utility(u,mask,x.CORE_ARMS[0]);v.verify_decoder(u,mask,x.CORE_ARMS[0],pairs,diag)
    with pytest.raises(AssertionError):v.verify_decoder(u,mask,x.CORE_ARMS[0],((0,0),),diag)


def test_baseline_catalogue_matches_existing_token_order():
    from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue
    assert v.baseline_catalogue()==catalogue(4,2)


def test_independent_gate_family_and_boundary_logic():
    means={a:dict(normalized_cost=101 if a in (x.CORE_ARMS[3],'mappo','full_mappo') else 100,availability=.67,normalized_pending=0,normalized_failures=.1) for a in x.ARMS}
    points,checks=v.endpoint_gates(means)
    assert len(points)==13 and all(checks.values())
    means['mappo']['normalized_cost']=100;assert not v.endpoint_gates(means)[1]['gain_vs_mappo']
    means[x.CORE_ARMS[0]]['normalized_failures']=.4
    assert not v.endpoint_gates(means)[1]['normalized_failures_delta_vs_ppo']
