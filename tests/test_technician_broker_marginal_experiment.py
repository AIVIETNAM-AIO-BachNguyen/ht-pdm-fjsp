import time
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_broker_marginal_experiment as x
from ht_pdm_fjsp.technician_broker_marginal_policy import CORE_ARMS


def test_private_critics_equal_fresh_initial_tensors_capacities_and_ppo_pair():
    c=x.settings('engineering');t=c['train_seeds'][0];models={a:x.make_model(a,c,t) for a in x.ARMS}
    assert len({x.learner.parameter_digest(models[a]) for a in CORE_ARMS})==1
    assert all(sum(p.numel() for p in models[a].parameters())==19974 for a in CORE_ARMS)
    critics=[m.critic.net if isinstance(m.critic,x.reference.ValueUnits) else m.critic for m in models.values()]
    assert len({x.raw_tensor_digest(m) for m in critics})==1
    assert len({next(m.parameters()).data_ptr() for m in critics})==8
    assert x.raw_tensor_digest(models['ppo'])==x.raw_tensor_digest(models['normalized_ppo'])
    assert sum(p.numel() for p in models[CORE_ARMS[0]].actor.parameters())==11141
    assert sum(p.numel() for p in critics[0].parameters())==8833

def test_end_to_end_fixed_correctness_budget_all_arms_audits_and_frozen_modes(tmp_path):
    """Disposable engineering fixture, not a separate research smoke experiment."""
    c=x.settings('engineering');c.update(env_steps=96,vector_envs=2,rollout=96,epochs=1,minibatch=48,checkpoints=[96])
    t=c['train_seeds'][0];start=time.monotonic();results=[x.fit_job((c,a,t,str(tmp_path),start)) for a in x.ARMS]
    choices={f"{r['method']}:{t}":r['choice'] for r in results};dev=[v for r in results for v in r['development']]
    x.write_json(tmp_path/'checkpoint_choices.json',choices)
    for key in ('training_episodes','training_progress'):x.merge_journals(tmp_path,key,results)
    assert x.audit_training(tmp_path,c,results,choices,dev,lambda:None)['training_episodes']==16
    rows=[];journals={key:x.Journal(tmp_path/f'{key}.csv') for key in ('coordination','decisions')}
    try:
        for label in (*x.ARMS,*x.FROZEN,'deadline'):
            a=CORE_ARMS[0] if label in x.FROZEN else label;choice=choices[f'{a}:{t}'] if a!='deadline' else None
            model=x.load_model(tmp_path/choice['path'],a,c,t,choice['steps']) if choice else None
            before=x.learner.parameter_digest(model) if model else None
            for s in c['development_seeds']:
                rows.append(x.evaluate(label,t if label!='deadline' else -1,s,model,journals['coordination'],journals['decisions']))
                if model:assert model.algorithm==a
            if model:assert before==x.learner.parameter_digest(model)
    finally:
        for j in journals.values():j.close()
    verified=x.audit_selected(tmp_path,c,rows,lambda:None);assert verified['physical_transitions']==1440
    latency=x.isolated_latency(tmp_path,c,choices,lambda:None);assert latency['samples']==144 and (tmp_path/'isolated_timing_samples.csv').exists()
    assert len(x.point_summary(rows,latency)['checks'])==13

def test_serial_and_process_training_identical_tensors_and_choices(tmp_path):
    c=x.settings('engineering');c.update(env_steps=96,vector_envs=2,rollout=96,epochs=1,minibatch=48,checkpoints=[96]);t=c['train_seeds'][0];start=time.monotonic()
    arms=(CORE_ARMS[0],'normalized_ppo');serial=[x.fit_job((c,a,t,str(tmp_path/'serial'),start)) for a in arms]
    with x.ProcessPoolExecutor(max_workers=2,mp_context=x.mp.get_context('spawn')) as pool:
        parallel=list(pool.map(x.fit_job,[(c,a,t,str(tmp_path/'parallel'),start) for a in arms]))
    for s,p in zip(serial,parallel):
        assert s['audit']['parameter_sha256']==p['audit']['parameter_sha256']
        assert s['choice']['steps']==p['choice']['steps'] and s['choice']['development_cost']==p['choice']['development_cost']

def test_new_seeds_budgets_gates_and_historical_critic_namespaces():
    audit=x.seed_audit();assert audit['status']=='PASS' and audit['raw_unique']==366 and audit['derived_unique']==164358
    c=x.settings();assert c['test_seeds']==[] and c['env_steps']==983040 and c['checkpoints']==list(range(122880,983041,122880))
    assert c['confidence']==1-(.05/12)/13
    for module in (x.raw_reference,x.budget_reference):
        p=module.settings('engineering');t=p['train_seeds'][0];v=module.VERSION
        assert x.learner.role_seed(t,f'{v}:initialization:critic') in x.derived_streams(p,v)
    rows=[dict(method=a,normalized_cost=101 if a in (CORE_ARMS[1],'mappo','full_mappo') else 100,availability=.67,normalized_pending=0,normalized_failures=.1,invalid_assignments=0) for a in (*x.ARMS,*x.FROZEN,'deadline')]
    s=x.point_summary(rows,dict(p95_ms=1));assert s['development_screen_pass'] and len(s['checks'])==13
    rows[1]['normalized_cost']=100;assert not x.point_summary(rows,dict(p95_ms=1))['checks']['gain_vs_'+CORE_ARMS[1]]
