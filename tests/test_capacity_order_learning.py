import csv,time
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_capacity_order_experiment as e


def test_private_equal_critics_and_core_initializations_and_role_grid():
    torch.set_num_threads(1);c=e.settings('engineering');models={a:e.make_model(a,c,c['train_seeds'][0]) for a in e.ARMS};assert len({e.learner.parameter_digest(models[a]) for a in e.CORE})==1
    critics=[m.critic.net if isinstance(m.critic,e.old.reference.ValueUnits) else m.critic for m in models.values()];assert len({e.old.raw_tensor_digest(v) for v in critics})==1 and len({next(v.parameters()).data_ptr() for v in critics})==11
    assert all(sum(p.numel() for p in models[a].parameters())==19908 for a in e.CORE);audit=e.seed_audit();assert audit['status']=='PASS' and audit['raw_unique']==365 and audit['derived_unique']==82424 and not audit['confirmation_states_constructed']


@pytest.fixture(scope='module')
def jobs(tmp_path_factory):
    torch.set_num_threads(1);root=tmp_path_factory.mktemp('capacity_order_integration');c=e.settings('engineering');c.update(env_steps=96,checkpoints=[96]);t=c['train_seeds'][0];results={a:e.fit_job((c,a,t,str(root),time.monotonic())) for a in e.ARMS};return root,c,results


@pytest.mark.parametrize('algorithm',e.ARMS)
def test_all_eleven_standard_joint_ppo_roundtrip_full_episode_updates(jobs,algorithm):
    root,c,results=jobs;r=results[algorithm];t=c['train_seeds'][0];audit=r['audit'];assert audit['learned_steps']==96 and audit['training_episodes']==2 and audit['optimizer_steps']==2 and audit['behavior_normalized_max_error']<=1
    model=e.load_model(root/r['fixed']['path'],algorithm,c,t,96);assert e.learner.parameter_digest(model)!=r['initial'];row=e.evaluate(algorithm,t,360700001,model);assert row['physical_ticks']==60 and row['invalid_assignments']==0


def test_merged_budgets_full_primitive_replay_gate_math_and_isolated_timing(jobs):
    root,c,mapping=jobs;results=list(mapping.values());dev=[row for r in results for row in r['development']]
    for name in ('training_progress','training_episodes'):e.old.merge_journals(root,name,results)
    audit=e.audit_training(root,c,results,dev,lambda:None);assert audit['models']==11 and audit['training_episodes']==22
    choices={f"{r['method']}:{r['train_seed']}":r['fixed'] for r in results};e.write_json(root/'checkpoint_choices.json',choices);policy=e.Journal(root/'coordination.csv');physical=e.Journal(root/'decisions.csv');rows=[];t=c['train_seeds'][0]
    try:
        for a in e.ARMS:rows.append(e.evaluate(a,t,360700001,e.load_model(root/choices[f'{a}:{t}']['path'],a,c,t,96),policy,physical))
        rows.append(e.evaluate('deadline',-1,360700001,coordination=policy,decisions=physical))
    finally:policy.close();physical.close()
    e.write_csv(root/'selected_episodes.csv',rows);replay=e.audit_final(root,rows,c,lambda:None);assert replay['physical_transitions']==720 and replay['policy_decisions']==576;timing=e.isolated_latency(root,c,choices);assert timing['samples']==144;summary=e.point_summary(rows,c,timing);assert len(summary['checks'])==16 and not summary['test_opened']


def test_sixteen_boundaries_and_three_replica_guard():
    c=e.settings();rows=[dict(method=a,train_seed=t,normalized_cost=99. if a==e.CORE[0] else 100.,availability=.67,normalized_pending=0.,normalized_failures=.1,invalid_assignments=0) for a in (*e.ARMS,'deadline') for t in c['train_seeds']];summary=e.point_summary(rows,c,dict(p95_ms=.1));assert all(summary['checks'].values()) and summary['development_screen_pass']
    for row in rows:
        if row['method']==e.CORE[0]:row['normalized_cost']=99.6
    summary=e.point_summary(rows,c,dict(p95_ms=.1));assert not summary['checks']['gain_vs_fixed_order_ppo'] and not summary['checks']['gain_vs_independent_machine_q_ppo']
