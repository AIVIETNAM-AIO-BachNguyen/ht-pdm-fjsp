import csv,time
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_typed_gibbs_entropy_units_experiment as e


def test_private_equal_initializations_capacities_and_fresh_role_grid():
    torch.set_num_threads(1);c=e.settings('engineering');t=c['train_seeds'][0];models={a:e.make_model(a,c,t) for a in e.ARMS};assert len({e.learner.parameter_digest(models[a]) for a in e.CORE})==1
    critics=[m.critic.net if isinstance(m.critic,e.old.reference.ValueUnits) else m.critic for m in models.values()];assert len({e.old.raw_tensor_digest(m) for m in critics})==1 and len({next(m.parameters()).data_ptr() for m in critics})==10
    assert all(sum(p.numel() for p in models[a].parameters())==18883 for a in e.CORE);audit=e.seed_audit();assert audit['status']=='PASS' and audit['raw_unique']==365 and audit['derived_unique']==82397 and not audit['confirmation_states_constructed']


@pytest.fixture(scope='module')
def jobs(tmp_path_factory):
    torch.set_num_threads(1);root=tmp_path_factory.mktemp('entropy_units_integration');c=e.settings('engineering');c.update(env_steps=96,checkpoints=[96]);t=c['train_seeds'][0];e.make_common_probe(root,c);results={a:e.fit_job((c,a,t,str(root),time.monotonic())) for a in e.ARMS};return root,c,results


@pytest.mark.parametrize('algorithm',e.ARMS)
def test_all_ten_standard_ppo_full_episode_updates_and_roundtrip(jobs,algorithm):
    root,c,results=jobs;r=results[algorithm];assert r['audit']['learned_steps']==96 and r['audit']['training_episodes']==2 and r['audit']['optimizer_steps']==2 and r['audit']['behavior_normalized_max_error']<=1
    model=e.load_model(root/r['fixed']['path'],algorithm,c,c['train_seeds'][0],96);assert e.learner.parameter_digest(model)!=r['initial'];row=e.evaluate(algorithm,c['train_seeds'][0],400700001,model);assert row['physical_ticks']==60 and row['invalid_assignments']==0


def test_probe_is_diagnostic_only_complete_grid_and_full_primitive_replay(jobs):
    root,c,mapping=jobs;results=list(mapping.values());dev=[row for r in results for row in r['development']]
    for name in ('training_progress','training_episodes'):e.old.merge_journals(root,name,results)
    audit=e.audit_training(root,c,results,dev,lambda:None);assert audit['models']==10 and audit['training_episodes']==20
    probe=e.read(root/'entropy_probe_binding.json');assert probe['states']==48 and probe['physical_ticks']==60 and not probe['training_labels'] and not probe['optimizer_consumes_probe'];before=probe['sha256'];choices={f"{r['method']}:{r['train_seed']}":r['fixed'] for r in results};e.write_json(root/'checkpoint_choices.json',choices)
    physical=e.Journal(root/'decisions.csv');policy=e.Journal(root/'coordination.csv');rows=[];t=c['train_seeds'][0]
    try:
        for a in e.ARMS:rows.append(e.evaluate(a,t,400700001,e.load_model(root/choices[f'{a}:{t}']['path'],a,c,t,96),policy,physical))
        rows.append(e.evaluate('deadline',-1,400700001,coordination=policy,decisions=physical))
    finally:physical.close();policy.close()
    replay=e.audit_final(root,rows,c,lambda:None);assert replay['physical_transitions']==660 and replay['policy_decisions']==528;latency=e.isolated_latency(root,c,choices);assert latency['samples']==144;summary=e.point_summary(rows,c,latency);assert len(summary['checks'])==15 and not summary['test_opened'];entropy=[row for r in results for row in r['entropy_probe']];readout=e.entropy_readout(entropy,c);assert len(entropy)==12 and not readout['selection_criterion'];assert e.sha(root/'entropy_probe.npz')==before


@pytest.mark.parametrize('target',e.ARMS[1:])
def test_each_cost_gate_independently_rejects(target):
    c=e.settings();rows=[dict(method=a,train_seed=t,normalized_cost=99. if a==e.CORE[0] else 100.,availability=.67,normalized_pending=.04,normalized_failures=.4,invalid_assignments=0) for a in (*e.ARMS,'deadline') for t in c['train_seeds']]
    for r in rows:
        if r['method']==target:r['normalized_cost']=99.4 if target==e.CORE[1] else 98. if target in ('mappo','full_mappo') else 97.
    s=e.point_summary(rows,c,dict(p95_ms=.1));assert not s['checks']['gain_vs_'+target] and all(v for k,v in s['checks'].items() if k!='gain_vs_'+target) and not s['development_screen_pass']


@pytest.mark.parametrize('target,metric,value',[('ppo','availability',.6901),('normalized_ppo','availability',.6901),('ppo','normalized_pending',.0149),('normalized_ppo','normalized_pending',.0149),('ppo','normalized_failures',.1499),('normalized_ppo','normalized_failures',.1499)])
def test_each_health_gate_independently_rejects(target,metric,value):
    c=e.settings();rows=[dict(method=a,train_seed=t,normalized_cost=99. if a==e.CORE[0] else 100.,availability=.67,normalized_pending=.04,normalized_failures=.4,invalid_assignments=0) for a in (*e.ARMS,'deadline') for t in c['train_seeds']]
    for r in rows:
        if r['method']==target:r[metric]=value
    s=e.point_summary(rows,c,dict(p95_ms=.1));key=metric+'_delta_vs_'+target;assert not s['checks'][key] and all(v for k,v in s['checks'].items() if k!=key)


def test_secondary_entropy_probe_thresholds_fixed_early_and_missing_grid():
    c=e.settings();rows=[dict(method=a,train_seed=t,checkpoint_steps=step,mean_normalized_entropy=.02 if a==e.CORE[0] else 0.,near_deterministic_fraction=.9 if a==e.CORE[0] else 1.) for a in e.CORE for t in c['train_seeds'] for step in (0,*c['checkpoints'])];r=e.entropy_readout(rows,c);assert r['mean_entropy_gap']==pytest.approx(.02) and r['early_checkpoint']==122880 and r['delayed_saturation_descriptively_supported']
    # Reduction is measured from floating data; move above threshold for a clean pass.
    for row in rows:
        if row['method']==e.CORE[0]:row['near_deterministic_fraction']=.89
    assert e.entropy_readout(rows,c)['delayed_saturation_descriptively_supported']
    for row in rows:
        if row['method']==e.CORE[0] and row['checkpoint_steps']==122880:row['mean_normalized_entropy']=.01999
    assert not e.entropy_readout(rows,c)['delayed_saturation_descriptively_supported']
    with pytest.raises(AssertionError,match='complete'):e.entropy_readout(rows[:-1],c)
