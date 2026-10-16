"""Runner correctness checks; temporary fixtures are not research pilot runs."""
import copy
import csv
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_marginal_matching_experiment as x


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def panel():
    return {4:{a:np.array([100. if a=='marginal_matching_ppo' else 110.,.7,0.,0.])
               for a in (*x.arms(4),'deadline')}}


def test_registered_budget_seed_audit_and_n4_only():
    audit=x.seed_audit()
    assert audit['status']=='PASS' and audit['raw_unique']==368
    assert audit['derived_unique']==82431
    c=x.settings('development')
    assert c['scales']==[4] and c['env_steps']==491520 and c['vector_envs']==16
    assert c['rollout']==3072 and c['epochs']==4 and c['minibatch']==256
    assert c['env_steps']//c['rollout']*c['epochs']*(c['rollout']//c['minibatch'])==7680
    assert len(x.schedule_jobs(c,'/unused',0.))==27 and not c['test_seeds']
    assert sum(c['env_steps'] for _ in x.schedule_jobs(c,'/unused',0.))==13271040
    assert c['max_seconds']==43200 and c['epsilon_start']==c['epsilon_end']==0
    cf=x.settings('confirmation')
    assert cf['confidence']==1-(.05/6)/14 and cf['bootstrap_samples']==200000
    assert len(cf['train_seeds'])==5 and len(cf['test_seeds'])==256
    with pytest.raises(ValueError):x.arms(8)
    with pytest.raises(ValueError):x.settings('pilot')


def test_factory_paired_capacity_and_original_baseline():
    c=x.settings('development');t=c['train_seeds'][0]
    models=[x.make_model(a,4,c,t) for a in x.CORE_ARMS]
    assert len({x.learner.parameter_digest(m) for m in models})==1
    assert {sum(p.numel() for p in m.parameters()) for m in models}=={18390}
    assert all(not hasattr(m,'allocation_learning_mode') for m in models)
    raw,norm=[x.make_model(a,4,c,t) for a in ('ppo','normalized_ppo')]
    assert all(torch.equal(a,b) for a,b in zip(raw.parameters(),norm.parameters()))
    with torch.random.fork_rng():
        torch.manual_seed(x.learner.role_seed(t,x.VERSION+':initialization:ppo'))
        original=x.learner.Model('ppo',c['baseline_hidden'])
    assert x.learner.parameter_digest(raw)==x.learner.parameter_digest(original)
    counts={a:sum(p.numel() for p in x.make_model(a,4,c,t).parameters()) for a in x.arms(4)}
    assert counts['ppo']==18966 and counts['mappo']==15364 and counts['full_mappo']==18052


def test_fourteen_gates_reject_mechanism_health_and_equal_mappo():
    points=x.estimates(panel(),[4]);assert len(points)==14 and all(x.point_checks(points).values())
    checks=x.guards({k:dict(lower=v,upper=v) for k,v in points.items()},[4],{'4:marginal_matching_ppo':dict(p95_ms=1.)})
    assert all(checks.values())
    for key,bad in [('N4_gain_vs_latent_order_ppo',.0049),('N4_gain_vs_no_prefix_ppo',0.),
                    ('N4_gain_vs_mappo',0.),('N4_gain_vs_full_mappo',0.),
                    ('N4_availability_difference_vs_ppo',-.0101),
                    ('N4_pending_per_machine_difference_vs_normalized_ppo',.0251),
                    ('N4_failures_per_machine_difference_vs_ppo',.2501)]:
        changed=dict(points);changed[key]=bad
        assert not x.point_checks(changed)[key]
    for a in ('ppo','normalized_ppo','forward_order_ppo','reverse_order_ppo'):
        points[f'N4_gain_vs_{a}']=-.01
        assert x.point_checks(points)[f'N4_gain_vs_{a}']


def synthetic_rows(c,development=False):
    rows=[]
    for a,t in [(a,t) for a in x.arms(4) for t in c['train_seeds']]+([] if development else [('deadline',-1)]):
        for step in ([0,*c['checkpoints']] if development else [None]):
            for seed in c['development_seeds'] if development else c['test_seeds']:
                cost=100. if a=='marginal_matching_ppo' else 110.
                r=dict(n=4,method=a,train_seed=t,seed=seed,normalized_cost=cost,extended_cost=4*cost,
                       availability=.7,normalized_pending=0.,normalized_failures=0.)
                if development:r.update(checkpoint_steps=step)
                rows.append(r)
    return rows


def test_crossed_bootstrap_and_engineering_cannot_claim_goal():
    c=dict(x.settings('smoke'),train_seeds=[1,2],test_seeds=[3,4])
    rows=synthetic_rows(c);lat={'4:marginal_matching_ppo':[1.,2.]}
    s=x.summarize(rows,c,lat)
    assert s['family_size']==14 and s['classification']=='ENGINEERING_ONLY'
    assert s['engineering_guard_pass'] and not s['goal_pass'] and len(s['leave_one_replica_out'])==2
    assert s['intervals']['N4_gain_vs_ppo']['lower']==pytest.approx(1-100/110)
    with pytest.raises(RuntimeError,match='exact test grid'):x.summarize(rows[:-1],c,lat)


def test_development_cp0_ties_latency_and_summary_grid():
    c=x.settings('development');rows=synthetic_rows(c,True)
    refs=[dict(n=4,normalized_cost=110.,availability=.7,normalized_pending=0.,normalized_failures=0.)]
    s=x.development_summary(rows,c,refs,{'4:marginal_matching_ppo':[1.,2.]})
    x.validate_summary_grid(s,c)
    assert s['development_screen_pass'] and set(s['selected_steps'].values())=={0}
    slow=x.development_summary(rows,c,refs,{'4:marginal_matching_ppo':[51.]})
    assert not slow['development_screen_pass'] and not slow['latency_pass']
    changed=copy.deepcopy(s);changed['estimates']['N4_failures_per_machine_difference_vs_ppo']=.3
    with pytest.raises(ValueError,match='thresholds'):x.validate_summary_grid(changed,c)
    changed=copy.deepcopy(s);changed['screen'].pop('N4_gain_vs_latent_order_ppo')
    with pytest.raises(ValueError,match='incomplete'):x.validate_summary_grid(changed,c)


@pytest.mark.parametrize('arm',x.arms(4))
def test_real_optimizer_checkpoint_and_whole_episode_accounting(tmp_path,arm):
    c=dict(x.settings('smoke'),env_steps=96,vector_envs=1,rollout=48,epochs=1,minibatch=48,
           checkpoints=[96],development_seeds=[200100000])
    t=c['train_seeds'][0]
    result=x.fit_job((c,4,arm,t,str(tmp_path),time.monotonic()))
    audit=result['audit']
    assert audit['learned_steps']==96 and audit['optimizer_steps']==2
    assert audit['training_episodes']==2 and audit['physical_ticks']==120 and audit['changed']
    assert audit['behavior_normalized_max_error']<=1 and len(result['devrows'])==2
    choice=result['choice'];model=x.load_model(tmp_path/choice['path'],4,c,t,arm,choice['steps'])
    obs=x.ResourceEnv(c['development_seeds'][0],4,arm).reset()
    x.ResourceEnv(c['development_seeds'][0],4,arm).reset()
    assert np.array_equal(model.select(obs),x.load_model(tmp_path/choice['path'],4,c,t,arm,choice['steps']).select(obs))
    with (tmp_path/'N4'/arm/str(t)/'training_episodes.csv').open() as stream:
        episodes=list(csv.DictReader(stream))
    assert all(float(r['invalid_assignments'])==0 and abs(float(r['episode_reward'])*20+float(r['extended_cost']))<1e-5 for r in episodes)
    if arm=='marginal_matching_ppo':
        choices={f'4:{arm}:{t}':choice};x.write_json(tmp_path/'checkpoint_choices.json',choices)
        latency=x.isolated_latency(tmp_path,c,choices,x.sha(tmp_path/'checkpoint_choices.json'))
        assert len(latency['4:marginal_matching_ppo'])==144
        report=json.loads((tmp_path/'latency.json').read_text())
        assert report['workers_active']==0 and report['contexts'][0]['states']==48


def test_serial_spawn_parameter_and_checkpoint_identity(tmp_path):
    c=dict(x.settings('smoke'),env_steps=96,vector_envs=1,rollout=48,epochs=1,minibatch=48,
           checkpoints=[96],development_seeds=[200100000])
    left,right=tmp_path/'serial',tmp_path/'spawn';left.mkdir();right.mkdir()
    job=(c,4,'marginal_matching_ppo',c['train_seeds'][0])
    a=x.fit_job((*job,str(left),time.monotonic()))
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:
        b=pool.submit(x.fit_job,(*job,str(right),time.monotonic())).result(timeout=60)
    assert a['initial']==b['initial'] and a['audit']['parameter_sha256']==b['audit']['parameter_sha256']
    assert a['choice']['sha256']==b['choice']['sha256']


def prerequisite_fixture(path):
    c=x.settings('development');rows=synthetic_rows(c,True)
    refs=[dict(n=4,normalized_cost=110.,availability=.7,normalized_pending=0.,normalized_failures=0.)]
    s=x.development_summary(rows,c,refs,{'4:marginal_matching_ppo':[1.]*432})
    s['selected_steps']={k:c['checkpoints'][0] for k in s['selected_steps']}
    audits=[dict(n=4,algorithm=a,train_seed=t,learned_steps=c['env_steps'],physical_ticks=c['env_steps']//48*60,
        training_episodes=c['env_steps']//48,optimizer_steps=7680,from_scratch=True,pretraining_steps=0,
        changed=True,behavior_normalized_max_error=0.) for a in x.arms(4) for t in c['train_seeds']]
    choices={k:dict(steps=v) for k,v in s['selected_steps'].items()}
    for name,data in [('resolved_config.json',c),('summary.json',s),('audit.json',dict(status='PASS')),
                      ('training_audit.json',audits),('checkpoint_choices.json',choices)]:
        x.write_json(path/name,data)
    x.write_json(path/'latency.json',dict(samples_ms={'4:marginal_matching_ppo':[1.]*432},choices_sha256=x.sha(path/'checkpoint_choices.json')))
    m=dict(protocol=x.VERSION,status='COMPLETED',audit_status='PASS',test_opened=False,
           source_seal={'source_snapshot/example.py':'frozen'},
           output_bindings={p.name:dict(sha256=x.sha(p)) for p in path.iterdir()})
    x.write_json(path/'manifest.json',m)
    return c,s,audits,m


def test_confirmation_prerequisite_bound_grid_budget_latency_and_source(tmp_path):
    c,s,audits,m=prerequisite_fixture(tmp_path)
    assert x.validate_prerequisite(tmp_path,x.settings('confirmation'),{'source_snapshot/example.py':'frozen'})['status']=='PASS'
    with pytest.raises(ValueError,match='source/protocol'):x.validate_prerequisite(tmp_path,x.settings('confirmation'),{'source_snapshot/example.py':'changed'})
    audits[0]['learned_steps']-=1;x.write_json(tmp_path/'training_audit.json',audits)
    with pytest.raises(ValueError,match='training artifact'):x.validate_prerequisite(tmp_path,x.settings('confirmation'),{})
    m['output_bindings']['training_audit.json']['sha256']=x.sha(tmp_path/'training_audit.json');x.write_json(tmp_path/'manifest.json',m)
    with pytest.raises(ValueError,match='budget grid'):x.validate_prerequisite(tmp_path,x.settings('confirmation'),{})


def test_no_missing_development_prerequisite():
    with pytest.raises(ValueError,match='confirmation needs'):x.validate_prerequisite(None,x.settings('confirmation'),{})


def test_independent_budget_tape_selection_and_sixty_tick_replay(tmp_path):
    """Exercise the final audit on temporary nine-arm logs, including physical replay."""
    c=dict(x.settings('smoke'),env_steps=48,vector_envs=1,rollout=48,epochs=1,minibatch=48,
           checkpoints=[48],development_seeds=[200100000],test_seeds=[200200000])
    t=c['train_seeds'][0];choices={};audits=[];devrows=[]
    journal=x.Journal(tmp_path/'training_episodes.csv')
    try:
        for arm in x.arms(4):
            r=x.fit_job((c,4,arm,t,str(tmp_path),time.monotonic()))
            choices[f'4:{arm}:{t}']=r['choice'];audits.append(r['audit']);devrows.extend(r['devrows'])
            with (tmp_path/'N4'/arm/str(t)/'training_episodes.csv').open() as stream:
                for row in csv.DictReader(stream):journal.add(row)
    finally:journal.close()
    coordination=x.Journal(tmp_path/'coordination.csv');decisions=x.Journal(tmp_path/'decisions.csv');rows=[]
    try:
        for arm in x.arms(4):
            choice=choices[f'4:{arm}:{t}'];model=x.load_model(tmp_path/choice['path'],4,c,t,arm,choice['steps'])
            rows.append(x.evaluate(arm,4,t,c['test_seeds'][0],model,coordination,decisions))
        rows.append(x.evaluate('deadline',4,-1,c['test_seeds'][0],None,coordination,decisions))
    finally:coordination.close();decisions.close()
    report=x.audit_run(tmp_path,c,devrows,rows,audits,choices)
    assert report['status']=='PASS' and report['test_physical_ticks_replayed']==600
    assert report['training_episodes']==9 and report['paired_training_tapes']==1
    changed=copy.deepcopy(choices);changed[f'4:marginal_matching_ppo:{t}']['development_cost']+=1
    with pytest.raises(RuntimeError,match='checkpoint selection'):x.audit_run(tmp_path,c,devrows,rows,audits,changed)
