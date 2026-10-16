import importlib.util,itertools
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_raw_state_reservation_experiment as x
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import transition

spec=importlib.util.spec_from_file_location('raw_verifier',Path(__file__).resolve().parents[1]/'scripts/verify_raw_state_reservation_development.py')
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
    torch.set_num_threads(1);env=x.ResourceEnv(270700001,4,x.CORE_ARMS[0]);env.reset()
    state,events,population=v.initial_events(env.physical_seed)
    assert state=={k:list(a) for k,a in asdict(env.env.state).items()} and population==env.population
    assert events==list(map(list,env.env.events))
    rng=np.random.default_rng(8);current=env.env.state
    for tick in range(60):
        state={k:list(a) for k,a in asdict(current).items()};env.env.state=current;env.env.time=tick
        expected=v.observation(state,tick);actual=env.observe()
        assert all(np.array_equal(actual[k],expected[k]) for k in expected)
        mask=expected['pair_mask']
        for arm in x.CORE_ARMS:
            from ht_pdm_fjsp.technician_raw_state_reservation_policy import decode_tokens
            chosen=[];reserved=set()
            for j in range(2):
                candidates=[0]+[m+1 for m in range(4) if mask[m,j] and (arm==x.CORE_ARMS[3] or m not in reserved)]
                token=int(rng.choice(candidates));chosen.append(token)
                if token:reserved.add(token-1)
            chosen=np.array(chosen);pairs,diag=decode_tokens(chosen,mask,arm,expected['edges']);v.verify_decoder(chosen,mask,arm,pairs,diag)
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
    env=x.ResourceEnv(270700001,4,arm);env.env.state=DispatchState((3,4,5,6),(False,True,False,False),(0,2,0,0),(0,0),(-1,-1));env.env.time=7
    obs=env.observe();action=model.select(obs);state={k:list(a) for k,a in asdict(env.env.state).items()};independent=v.observation(state,7)
    weights={k:a.numpy().astype(float) for k,a in model.state_dict().items()}
    assert v.verify_actor([independent]*2,[action]*2,arm,weights)<1e-5
    pairs,diag=env.resolve(action)
    if arm in x.CORE_ARMS:v.verify_decoder(action,independent['pair_mask'],arm,pairs,diag)
    else:v.verify_baseline(int(action) if np.isscalar(action) else action.tolist(),independent['pair_mask'],arm,state,pairs)


def test_core_verifier_uses_no_catalogue_and_rejects_wrong_matching(monkeypatch):
    def forbidden():raise AssertionError('candidate verification enumerated matchings')
    monkeypatch.setattr(v,'baseline_catalogue',forbidden)
    from ht_pdm_fjsp.technician_raw_state_reservation_policy import decode_tokens
    mask=np.ones((4,2),bool);tokens=np.array([1,2]);edges=np.zeros((4,2,4));edges[:,:,1]=v.D/12;edges[:,:,2]=v.R
    pairs,diag=decode_tokens(tokens,mask,x.CORE_ARMS[0],edges);v.verify_decoder(tokens,mask,x.CORE_ARMS[0],pairs,diag)
    with pytest.raises(AssertionError):v.verify_decoder(tokens,mask,x.CORE_ARMS[0],((0,0),),diag)
    tokens=np.array([1,1]);pairs,diag=decode_tokens(tokens,mask,x.CORE_ARMS[3],edges)
    assert pairs==((0,0),);v.verify_decoder(tokens,mask,x.CORE_ARMS[3],pairs,diag)
    with pytest.raises(AssertionError):v.verify_decoder(tokens,mask,x.CORE_ARMS[0],pairs,diag)


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


def test_frozen_policy_uses_actual_execution_conditioning_and_order():
    torch.set_num_threads(1);c=x.settings('engineering');t=c['train_seeds'][0];model=x.make_model(x.CORE_ARMS[0],c,t)
    env=x.ResourceEnv(270700001,4,x.CORE_ARMS[0]);env.env.state=DispatchState((3,4,5,6),(False,True,False,False),(0,2,0,0),(0,0),(-1,-1));env.env.time=7
    obs=env.observe();weights={k:a.numpy().astype(float) for k,a in model.state_dict().items()}
    for arm in x.CORE_ARMS:
        model.algorithm=arm;action=model.select(obs)
        assert v.verify_actor([obs],[action],arm,weights)<1e-5


def test_timing_raw_samples_recompute_and_reject_tampering(tmp_path):
    import csv,json
    samples=[dict(train_seed=t,repeat=r,state_index=i,milliseconds=.5+i/100) for t in v.TRAINS for r in range(3) for i in range(48)]
    with (tmp_path/'isolated_timing_samples.csv').open('w') as f:
        writer=csv.DictWriter(f,list(samples[0]));writer.writeheader();writer.writerows(samples)
    choices={};bindings=[]
    for t in v.TRAINS:
        path=tmp_path/f'{t}.npz';np.savez(path,state=np.zeros((48,71)))
        meta=dict(states=48,seed=v.DEV[0],path=path.name,sha256=v.sha(path))
        choices[f'{v.CORE[0]}:{t}']=dict(steps=122880,sha256='abc',observations=meta)
        bindings.append(dict(train_seed=t,checkpoint_steps=122880,checkpoint_sha256='abc',**meta))
    (tmp_path/'checkpoint_choices.json').write_text(json.dumps(choices));times=[r['milliseconds'] for r in samples]
    latency=dict(samples=432,torch_threads=1,training_workers_active=False,mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),p99_ms=float(np.percentile(times,99)),choices_sha256=v.sha(tmp_path/'checkpoint_choices.json'),observation_bindings=bindings)
    (tmp_path/'latency.json').write_text(json.dumps(latency));assert v.verify_primary_timing(tmp_path,choices)==latency
    latency['p95_ms']+=.1;(tmp_path/'latency.json').write_text(json.dumps(latency))
    with pytest.raises(AssertionError):v.verify_primary_timing(tmp_path,choices)
