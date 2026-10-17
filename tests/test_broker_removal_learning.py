import copy,csv,time
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_broker_removal_experiment as e
from ht_pdm_fjsp import technician_broker_removal_policy as p
from ht_pdm_fjsp.technician_broker_removal_learning import cache_digest


def test_all_capacity_fresh_initial_pairing_and_context_equivalence():
    torch.set_num_threads(1);c=e.settings('engineering');t=c['train_seeds'][0];models={a:e.make_model(a,c,t) for a in e.ARMS};assert len({e.learner.parameter_digest(models[a]) for a in e.CORE})==1
    assert all(sum(v.numel() for v in models[a].parameters())==29383 for a in e.CORE);assert sum(v.numel() for v in models['capacity_ppo'].parameters())==28375
    critics=[m.critic.net if isinstance(m.critic,e.old.reference.ValueUnits) else m.critic for m in models.values()];assert len({e.old.raw_tensor_digest(v) for v in critics})==1 and len({next(v.parameters()).data_ptr() for v in critics})==11
    obs=e.ResourceEnv(330700001,4,e.CORE[0]).reset();batch=e.learner.tensors([obs]*8)
    for a in (*e.CORE,'capacity_ppo'):
        arm_obs=e.ResourceEnv(330700001,4,a).reset();np.testing.assert_array_equal(arm_obs['state'],obs['state']);arm_batch=e.learner.tensors([arm_obs]*8)
        m=models[a];action,_,value=m.sample(arm_batch,torch.Generator().manual_seed(330700002));cache=p.baseline_cache(m,arm_batch,action,value);assert not any(v.requires_grad for v in cache.values());assert cache['weight_error']<1e-5;torch.testing.assert_close(cache['actual_values'],value[:,None].expand(8,2),atol=.0003,rtol=1e-5);torch.testing.assert_close(cache['mixed_values'],cache['actual_values'],atol=.0003,rtol=1e-5)
        assert next(m.removal.parameters()).data_ptr()!=next((m.critic.net if hasattr(m.critic,'net') else m.critic).parameters()).data_ptr()
    assert all(torch.equal(models['ppo'].actor.state_dict()[k],models['normalized_ppo'].actor.state_dict()[k]) and torch.equal(models['ppo'].actor.state_dict()[k],models['capacity_ppo'].actor.state_dict()[k]) for k in models['ppo'].actor.state_dict())


def test_behavior_cache_unchanged_after_critic_update_and_blind_channels_zero():
    c=e.settings('engineering');t=c['train_seeds'][0];m=e.make_model(e.CORE[0],c,t);obs=e.ResourceEnv(330700001,4,e.CORE[0]).reset();batch=e.learner.tensors([obs]*8);u,_,value=m.sample(batch,torch.Generator().manual_seed(330700002));cache=p.baseline_cache(m,batch,u,value);seal=cache_digest(cache);g=torch.arange(8,dtype=torch.float32)-10;adv,scale=p.prepare_actor_advantage(g,value,cache,m.algorithm)
    with torch.no_grad():
        for v in m.removal.parameters():v.add_(.1)
    assert cache_digest(cache)==seal and not adv.requires_grad
    adv2,_=p.prepare_actor_advantage(g,value,cache,m.algorithm);torch.testing.assert_close(adv,adv2,atol=0,rtol=0)
    blind=e.make_model('blind_removal_ppo',c,t);features=torch.randn(8,2,80);other=features.clone();other[:,:,71:79]+=100;torch.testing.assert_close(p.predict_removal(blind,features),p.predict_removal(blind,other),atol=0,rtol=0)


def test_factor_joint_behavior_gradient_scaling_and_fixed_normalization():
    z=torch.tensor([[.2,-.1],[.3,.4]],requires_grad=True);old=z.detach().clone();a=torch.tensor([[-1.,-1.],[1.,1.]])
    factor=p.actor_objective(z,old,a,'factor_state_ppo');joint=p.actor_objective(z.sum(1),old.sum(1),a[:,0],'joint_state_ppo');g=torch.autograd.grad(factor,z,retain_graph=True)[0];h=torch.autograd.grad(joint,z)[0];torch.testing.assert_close(g,h)
    current=torch.tensor([[np.log(1.5),np.log(.5)]],dtype=torch.float32);actual=p.actor_objective(current,torch.zeros_like(current),torch.tensor([[2.,-1.]]),'factor_state_ppo');assert actual.item()==pytest.approx((1.2*2-.8)/2)
    returns=torch.tensor([1.,4.,7.]);oldv=torch.tensor([.5,1.,1.5]);b=torch.tensor([[0.,1.],[1.,2.],[2.,3.]]);adv,scale=p.prepare_actor_advantage(returns,oldv,dict(baseline=b),'removal_rb_ppo');a0=returns-oldv;torch.testing.assert_close(adv,(returns[:,None]-b-a0.mean())/(a0.std(unbiased=False)+1e-8))


@pytest.fixture(scope='module')
def engineering_run(tmp_path_factory):
    torch.set_num_threads(1);root=tmp_path_factory.mktemp('removal_unit_integration');c=e.settings('engineering');c.update(env_steps=96,checkpoints=[96]);t=c['train_seeds'][0];results={a:e.fit_job((c,a,t,str(root),time.monotonic())) for a in e.ARMS};return root,c,results


@pytest.mark.parametrize('arm',e.ARMS)
def test_one_complete_engineering_rollout_roundtrip_shadow_updates_and_physics(engineering_run,arm):
    root,c,results=engineering_run;t=c['train_seeds'][0];initial=e.make_model(arm,c,t);result=results[arm];audit=result['audit'];assert audit['learned_steps']==96 and audit['training_episodes']==2 and audit['optimizer_steps']==2 and audit['behavior_baselines_cached_before_current_batch_fit']
    final=e.load_model(root/result['fixed']['path'],arm,c,t,96);assert result['fixed']['steps']==96 and e.learner.parameter_digest(final)!=e.learner.parameter_digest(initial)
    if hasattr(final,'removal'):assert any(not torch.equal(v,initial.removal.state_dict()[k]) for k,v in final.removal.state_dict().items())
    episode=e.evaluate(arm,t,330700001,final);assert episode['physical_ticks']==60 and episode['invalid_assignments']==0
    if arm in e.CORE:
        def fail(*args):raise AssertionError('critic must not run inselect')
        final.critic.forward=fail;final.removal.forward=fail;final.select(e.ResourceEnv(330700001,4,arm).reset())


def test_aggregate_budget_final_primitive_replay_and_isolated_timing(engineering_run):
    root,c,result=engineering_run;results=list(result.values());dev=[row for r in results for row in r['development']]
    for name in ('training_progress','training_episodes'):e.old.merge_journals(root,name,results)
    audit=e.audit_training(root,c,results,dev,lambda:None);assert audit['models']==11 and audit['training_episodes']==22
    choices={f"{r['method']}:{r['train_seed']}":r['fixed'] for r in results};e.write_json(root/'checkpoint_choices.json',choices);physical=e.Journal(root/'decisions.csv');policy=e.Journal(root/'coordination.csv');rows=[];t=c['train_seeds'][0]
    try:
        for a in e.ARMS:
            m=e.load_model(root/choices[f'{a}:{t}']['path'],a,c,t,96);rows.append(e.evaluate(a,t,330700001,m,policy,physical))
        rows.append(e.evaluate('deadline',-1,330700001,coordination=policy,decisions=physical))
    finally:physical.close();policy.close()
    e.write_csv(root/'selected_episodes.csv',rows);audit=e.audit_final(root,rows,c,lambda:None);assert audit['physical_transitions']==720 and audit['policy_decisions']==576
    timing=e.isolated_latency(root,c,choices);assert timing['samples']==144 and timing['p95_ms']>=0


@pytest.mark.parametrize('arm',['removal_rb_ppo','blind_removal_ppo'])
def test_second_rollout_uses_learned_critic_without_refitting_actor_baseline(tmp_path,arm):
    c=e.settings('engineering');c.update(env_steps=192,checkpoints=[96,192]);t=c['train_seeds'][0];r=e.fit_job((c,arm,t,str(tmp_path),time.monotonic()));assert r['audit']['training_episodes']==4 and r['audit']['optimizer_steps']==4
    with (tmp_path/'N4'/arm/str(t)/'training_progress.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==2 and all(row['cache_phase']=='before_current_batch_optimizer' for row in rows) and rows[0]['baseline_cache_sha256']!=rows[1]['baseline_cache_sha256']
    final=e.load_model(tmp_path/r['fixed']['path'],arm,c,t,192)
    if arm=='blind_removal_ppo':assert final.removal[0].weight[:,71:79].count_nonzero()==0 and float(rows[1]['baseline_context_sensitivity_over_sd'])==0
    else:assert final.removal[0].weight[:,71:79].count_nonzero()>0 and float(rows[1]['baseline_context_sensitivity_over_sd'])>0


def test_fresh_seed_budget_and19_gate_boundaries():
    audit=e.seed_audit();assert audit['status']=='PASS' and not audit['confirmation_states_constructed'];c=e.settings();assert len(e.ARMS)*len(c['train_seeds'])*c['env_steps']==16220160
    rows=[dict(method=a,train_seed=t,normalized_cost=99. if a==e.CORE[0] else 100.,availability=.67,normalized_pending=0.,normalized_failures=.1,invalid_assignments=0) for a in (*e.ARMS,'deadline') for t in c['train_seeds']];s=e.point_summary(rows,c,dict(p95_ms=.5));assert len(s['checks'])==19 and s['development_screen_pass']
    for r in rows:
        if r['method']==e.CORE[0]:r['normalized_cost']=99.6
    assert not e.point_summary(rows,c,dict(p95_ms=.5))['checks']['gain_vs_factor_state_ppo']
