import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_broker_learning_efficiency as d


def test_registered_budget_cadence_and_fresh_actual_roles():
    c=d.settings();assert c['env_steps']==245760 and c['checkpoints']==list(d.STEPS[1:]) and not c['test_seeds']
    assert c['training_processes']==8 and c['vector_envs']==16 and c['rollout']==3072
    assert len(c['train_seeds'])*8*c['env_steps']==5898240
    assert c['source_role_namespace']==d.x.VERSION and c['outer_protocol']==d.VERSION
    audit=d.seed_audit();assert audit['status']=='PASS' and not audit['confirmation_states_constructed']


def test_source_models_preserve_paired_scratch_actor_private_common_critics():
    torch.set_num_threads(1);c=d.settings('engineering');t=c['train_seeds'][0];models={a:d.x.make_model(a,c,t) for a in d.x.ARMS}
    assert len({d.x.learner.parameter_digest(models[a]) for a in d.x.CORE_ARMS})==1
    assert all(sum(p.numel() for p in models[a].parameters())==19974 for a in d.x.CORE_ARMS)
    critics=[m.critic.net if isinstance(m.critic,d.x.reference.ValueUnits) else m.critic for m in models.values()]
    assert len({d.x.raw_tensor_digest(m) for m in critics})==1
    assert len({next(m.parameters()).data_ptr() for m in critics})==8
    env=d.x.ResourceEnv(c['development_seeds'][0],4,d.x.CORE_ARMS[0]);env.reset();assert env.physical_seed==d.x.role_seed(c['development_seeds'][0],f'{d.x.VERSION}:N4:physical')


def synthetic_dev(c):
    return [dict(method=a,train_seed=t,seed=s,checkpoint_steps=step,normalized_cost=10+step/d.STEPS[-1]) for a in d.x.ARMS for t in c['train_seeds'] for s in c['development_seeds'] for step in d.STEPS]


def test_trapezoid_uses_true_step_spacing_complete_pair_grid_and_cost_units():
    c=d.settings('engineering');rows=synthetic_dev(c);result=d.auc_rows(rows,c)
    assert len(result)==8 and all(r['auc_cost_per_machine']==pytest.approx(10.5) and r['endpoint_cost_per_machine']==11 and r['nodes']==8 for r in result)
    with pytest.raises(AssertionError,match='complete AUC'):d.auc_rows(rows[:-1],c)
    with pytest.raises(AssertionError,match='unique registered'):d.auc_rows(rows+[rows[0]],c)


def test_fixed_final_choice_ignores_better_early_checkpoint(tmp_path):
    torch.set_num_threads(1);c=d.settings('engineering');c.update(env_steps=d.STEPS[-1]);a='ppo';t=c['train_seeds'][0];folder=tmp_path/'N4'/a/str(t)/'checkpoints';folder.mkdir(parents=True)
    model=d.x.make_model(a,c,t);probe=d.x.ResourceEnv(c['development_seeds'][0],4,a).reset()
    for step in (d.STEPS[1],d.STEPS[-1]):d.x.save_model(model,folder/f'step_{step:09d}.pt',a,c,t,step,probe)
    dev=[dict(method=a,train_seed=t,checkpoint_steps=step,extended_cost=cost) for step,cost in ((d.STEPS[1],1.),(d.STEPS[-1],100.))]
    choice=d.final_choice(tmp_path,a,t,c,dev);assert choice['steps']==245760 and choice['development_cost']==100 and '000245760' in choice['path']


def test_fourteen_gates_require_auc_endpoint_health_and_all_replicas():
    c=d.settings();candidate=d.x.CORE_ARMS[0];rows=[];auc=[]
    for a in (*d.x.ARMS,'deadline',*d.x.FROZEN):
        for t in c['train_seeds']:
            rows.append(dict(method=a,train_seed=t,normalized_cost=100.,availability=.67,normalized_pending=0.,normalized_failures=.1,invalid_assignments=0))
    for row in rows:
        if row['method']==candidate:row['normalized_cost']=99.5
        if row['method'] in d.x.ARMS:auc.append(dict(method=row['method'],train_seed=row['train_seed'],auc_cost_per_machine=99 if row['method']==candidate else 100.))
    result=d.summarize(rows,auc,dict(p95_ms=.5),c);assert len(result['checks'])==14 and result['development_screen_pass'];json.dumps(result,allow_nan=False)
    bad=copy.deepcopy(auc)
    for row in bad:
        if row['method']==candidate:row['auc_cost_per_machine']=99.6
    assert not d.summarize(rows,bad,dict(p95_ms=.5),c)['development_screen_pass']
    for row in rows:
        if row['method']==candidate:row['normalized_failures']=.36
    assert not d.summarize(rows,auc,dict(p95_ms=.5),c)['checks']['normalized_failures_delta_vs_ppo']
