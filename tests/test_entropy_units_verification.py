import json,sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import entropy_units_reference as ref
import verify_entropy_units_development as v
import typed_gibbs_reference as base
from ht_pdm_fjsp import technician_typed_gibbs_entropy_units_experiment as e


@pytest.mark.parametrize('arm',v.ARMS)
def test_independent_all_ten_constructor_tensors(arm):
    torch.set_num_threads(1);t=400700000;actual=e.make_model(arm,e.settings('engineering'),t).state_dict();expected=v.independent_initial(arm,t,v.helpers());assert list(actual)==list(expected);assert all(torch.equal(actual[k],expected[k]) for k in actual)


@pytest.mark.parametrize('arm',ref.CORE)
def test_alias_density_and_physical_entropy_gradient_finite_difference(arm):
    torch.set_num_threads(1);h=v.helpers();o=h.observation(dict(ages=[3]*4,failed=[False]*4,pending_wait=[0]*4,remaining=[0]*2,assigned=[-1]*2),0);model=e.make_model(arm,e.settings('engineering'),400700000).double()
    with torch.no_grad():model.actor[-1].weight.copy_(torch.arange(128).reshape(2,64)/300-.2)
    batch={k:t.double() if t.is_floating_point() else t for k,t in e.learner.tensors([o]).items()};w={k:t.detach().numpy() for k,t in model.state_dict().items()};u,legal,canonical,r=ref.physical_reference([o],w,arm,h);action=model.trace(batch,deterministic=True)[0];lp,entropy,_=model.evaluate(batch,action);idx=(canonical==action.numpy()[0]).all(-1).argmax();scale=1 if arm in ref.CORE[:3] else 4
    np.testing.assert_allclose(lp.detach().numpy(),r['logp'][:,idx],atol=1e-10,rtol=0);np.testing.assert_allclose(entropy.detach().numpy()*scale,r['entropy'],atol=1e-10,rtol=0)
    utility=model.actor(model.inputs(batch)[0]);utility.retain_grad()
    # Differentiate the actual actor bias, and independently perturb equivalent utilities.
    grad=torch.autograd.grad(entropy.sum(),model.actor[-1].bias)[0].numpy();eps=1e-5
    for t in range(2):
        plus=u.copy();minus=u.copy();plus[:,:,t]+=eps;minus[:,:,t]-=eps
        numeric=(base.distribution(plus,legal,ref.BASE[arm],canonical)['entropy']-base.distribution(minus,legal,ref.BASE[arm],canonical)['entropy'])/(2*eps*scale)
        np.testing.assert_allclose(grad[t],numeric[0],atol=1e-8,rtol=0)
    assert ref.BASE[arm]==base.CORE[ref.CORE.index(arm)%3]


def test_fresh_independent_historical_roles(tmp_path):
    audit=e.seed_audit();(tmp_path/'seed_registry.json').write_text(e.REGISTRY.read_text());(tmp_path/'seed_audit.json').write_text(json.dumps(audit));r=v.seed_contract(tmp_path,v.helpers());assert r['raw']==365 and r['derived']==82397 and r['independent_historical_derived']=={'technician_capacity_order_ppo_v1':82424,'technician_typed_gibbs_ppo_v1':82397}


def means():return {a:dict(normalized_cost=99. if a==v.CORE[0] else 100.,availability=.67,normalized_pending=.04,normalized_failures=.4) for a in (*v.ARMS,'deadline')}


@pytest.mark.parametrize('target',v.ARMS[1:])
def test_independent_nine_cost_rejections_and_strict_zero(target):
    m=means();m[target]['normalized_cost']=99.4 if target==v.CORE[1] else 99. if target in ('mappo','full_mappo') else 97.;p,c=v.endpoint_gates(m,v.helpers());assert not c['gain_vs_'+target] and all(x for k,x in c.items() if k!='gain_vs_'+target)


@pytest.mark.parametrize('target,metric,value',[(a,m,x) for a in v.REFS for m,x in [('availability',.6901),('normalized_pending',.0149),('normalized_failures',.1499)]])
def test_independent_six_health_rejections(target,metric,value):
    m=means();m[target][metric]=value;_,c=v.endpoint_gates(m,v.helpers());assert not c[metric+'_delta_vs_'+target] and sum(c.values())==14


def test_secondary_readout_fixed_window_boundary_and_missing_grid():
    rows=[dict(method=a,train_seed=t,checkpoint_steps=s,mean_normalized_entropy=.02 if a==v.CORE[0] else 0.,near_deterministic_fraction=.9 if a==v.CORE[0] else 1.) for a in v.CORE for t in v.TRAINS for s in v.STEPS];r=v.secondary_readout(rows,v.helpers());assert r['delayed_saturation_descriptively_supported'] and r==e.entropy_readout(rows,e.settings())
    for row in rows:
        if row['method']==v.CORE[0] and row['checkpoint_steps']==122880:row['mean_normalized_entropy']=.01999
    assert not v.secondary_readout(rows,v.helpers())['delayed_saturation_descriptively_supported']
    with pytest.raises(AssertionError,match='complete90'):v.secondary_readout(rows[:-1],v.helpers())


def test_all_ten_manual_final_neural_actor_value_and_density(tmp_path):
    torch.set_num_threads(1);h=v.helpers();c=e.settings('engineering');t=c['train_seeds'][0];obs=[h.observation(dict(ages=[3]*4,failed=[False]*4,pending_wait=[0]*4,remaining=[0]*2,assigned=[-1]*2),0),h.observation(dict(ages=[4,5,6,7],failed=[False]*4,pending_wait=[0]*4,remaining=[1,0],assigned=[0,-1]),5)]
    choices={}
    for a in v.ARMS:
        model=e.make_model(a,c,t)
        with torch.no_grad():
            for parameter in model.actor.parameters():parameter.add_(.001)
        rich=[ref.enrich_observation(o,h,a) for o in obs];path=tmp_path/f'{a}.pt';meta=e.save_model(model,path,a,c,t,491520,rich[0]);meta['path']=path.name;choices[f'{a}:{t}']=meta
        actions=[model.select(o) for o in rich];data=[(o,u,model.describe(o,u) if a in v.CORE else {}) for o,u in zip(obs,actions)];r=ref.verify_neural(data,a,t,tmp_path,choices,c,h);assert r['states']==2 and r['value_max_error']<.0023041


def test_ninety_common_probe_cells_full_independent_replay_and_tamper(tmp_path,monkeypatch):
    torch.set_num_threads(1);c=e.settings('engineering');e.make_common_probe(tmp_path,c);monkeypatch.setattr(v,'DEV',tuple(c['development_seeds']));rows=[]
    with np.load(tmp_path/'entropy_probe.npz') as arrays:batch={k:torch.from_numpy(arrays[k].copy()) for k in arrays.files}
    for a in v.CORE:
        for t in v.TRAINS:
            for step in v.STEPS:
                model=e.make_model(a,c,t)
                with torch.no_grad():model.actor[-1].bias.copy_(torch.tensor([.7,-.4]))
                directory=tmp_path/f'N4/{a}/{t}/checkpoints';directory.mkdir(parents=True,exist_ok=True);e.save_model(model,directory/f'step_{step:09d}.pt',a,c,t,step,{k:value[0].numpy() for k,value in batch.items()});rows.append(dict(method=a,train_seed=t,checkpoint_steps=step,**model.entropy_probe(batch)))
    e.write_csv(tmp_path/'entropy_probe.csv',rows);e.write_json(tmp_path/'entropy_readout.json',v.secondary_readout(rows,v.helpers()));r=v.probe_audit(tmp_path,c,v.helpers(),lambda:None);assert r['cells']==90 and r['policy_state_evaluations']==4320 and r['unit_pair_invariants_checked']==90
    rows[0]['mean_normalized_entropy']+=.01;e.write_csv(tmp_path/'entropy_probe.csv',rows)
    with pytest.raises(AssertionError,match='probe metric'):v.probe_audit(tmp_path,c,v.helpers(),lambda:None)


def test_cardinalities_and_factorial_descriptors():
    assert len(v.ARMS)*len(v.TRAINS)==30 and 30*491520==14745600 and 30*10240==307200 and 30*7680==230400 and 30*5*32==4800 and (30*32+32)*60==59520
    m=means();assert v.factorial(m)['cost_difference_interaction_gibbs_minus_myopic']==-1;assert len(v.endpoint_gates(m,v.helpers())[0])==15


def test_original_failed_verdict_and_extra_guards_cannot_be_promoted():
    h=v.helpers();_,checks=v.endpoint_gates(means(),h);checks['gain_vs_'+v.CORE[1]]=False;per={str(t):.01 for t in v.TRAINS};extra=dict(every_replica_main_gain_positive=True,zero_invalid=True,isolated_p95_at_most_50ms=True);summary=dict(checks=checks,extra_checks=extra,development_screen_pass=False,goal_pass=False,test_opened=False);manifest=dict(development_screen_pass=False)
    assert not v.preserve_verdict(summary,manifest,checks,per,dict(p95_ms=3.),h)
    with pytest.raises(AssertionError,match='original verdict'):v.preserve_verdict(dict(summary,development_screen_pass=True),manifest,checks,per,dict(p95_ms=3.),h)
    with pytest.raises(AssertionError,match='original verdict'):v.preserve_verdict(summary,manifest,checks,dict(per,**{str(v.TRAINS[0]):0.}),dict(p95_ms=3.),h)
