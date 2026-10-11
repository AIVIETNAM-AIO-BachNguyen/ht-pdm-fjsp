"""Synthetic engineering fixtures verify physics and IO, not trained performance."""
import csv
from dataclasses import asdict,replace
import itertools
import json
import shutil
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import maintenance_policy_continuation as e
from ht_pdm_fjsp import maintenance_policy_forks as f
from ht_pdm_fjsp import maintenance_policy_statistics as stats
from ht_pdm_fjsp.maintenance_joint_policy import JOINT,FIXED,CATALOGUE,feature_batch
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays,membership


def model():return e.joint.model_for(e.joint.settings('full'),12100000,JOINT)


def test_counts_and_roles():
    full=e.counts(e.settings('full'));smoke=e.counts(e.settings('smoke'))
    assert (full['states'],full['max_forecast_transitions'],full['max_rl_continuation_decision_rows'])==(2560,23592960,7536640)
    assert (smoke['states'],smoke['max_forecast_transitions'],smoke['max_rl_continuation_decision_rows'])==(4,2304,320)
    assert full['training_steps']==full['new_physical_trajectory_ticks']==0
    assert e.seed_audit(e.settings('full'))['unique_derived_roles']==1284
    with pytest.raises(ValueError):e.seed_audit({**e.settings('full'),'scenarios':33})
    row=dict(cohort=13100000,shock=13200000,model_seed=12100000,origin=JOINT,tick=42)
    assert e.forecast_seed(e.settings('full'),row)!=e.forecast_seed(e.settings('smoke'),row)
    assert e.forecast_seed(e.settings('full'),row)!=e.previous.physical_seed(row['cohort'],row['shock'])


@pytest.mark.parametrize('condition',['dense','skill_mask'])
@pytest.mark.parametrize('remaining',[48,6,1])
def test_batch_scalar_features_masks_actions(condition,remaining):
    c=e.joint.make_config(13500000,condition,48);env=WaitingEnv(c);env.reset(13500001);states=[]
    for t in range(12):
        states.append(env.state);env.step(e.prior.rule_action(c,env.state,48-t,'deadline_matching'))
    batch={k:np.asarray([asdict(s)[k] for s in states]) for k in asdict(states[0])}
    expected=feature_batch([c]*len(states),states,[remaining]*len(states),48)
    for k,v in f.batch_features(c,batch,remaining).items():torch.testing.assert_close(v,expected[k],atol=0,rtol=0)
    policy=model();before=e.previous.parameters_sha(policy);chosen,counters=f.rl_batch(c,batch,remaining,policy)
    for i,s in enumerate(states):
        pairs,_=policy.select(c,s,remaining,48)
        np.testing.assert_array_equal(chosen[i],membership(4,2)[CATALOGUE.index(pairs)].astype(bool))
    assert counters['batched_decision_rows']==len(states) and e.previous.parameters_sha(policy)==before
    with pytest.raises(ValueError):f.batch_features(c,batch,0)
    policy.algorithm=FIXED
    with pytest.raises(ValueError):f.rl_batch(c,batch,remaining,policy)


def test_near_tie_scalar_fallback(monkeypatch):
    c=e.joint.make_config(13500000,'dense',48);s=WaitingEnv(c).state;policy=model()
    with torch.no_grad():
        for p in policy.actor.parameters():p.zero_()
    original=policy.select;calls=[]
    def spy(*args):calls.append(args[2:]);return original(*args)
    monkeypatch.setattr(policy,'select',spy)
    selected,counters=f.rl_batch(c,arrays(s,3),6,policy)
    assert not selected.any() and counters['near_tie_scalar_rows']==3 and calls==[(6,48)]*3


@pytest.mark.parametrize('condition',['dense','skill_mask'])
@pytest.mark.parametrize('tick',[6,42,47])
def test_fork_scalar_all_actions_costs_boundary_private_crn(condition,tick):
    c=e.joint.make_config(13500000,condition,48);env=WaitingEnv(c);env.reset(13500001)
    s=env.state;policy=model();root=policy.select(c,s,48-tick,48)[0];before=e.previous.parameters_sha(policy)
    ref=e.prior.rule_action(c,s,48-tick,'deadline_matching')
    v=f.fork_values(c,s,root,ref,policy,14300000,4,tick,record_trace=True)
    assert f.scalar_replay(c,s,v,policy,tick) and v['physical_ticks']==60-tick
    assert v['counters']['rl_continuation_transition_rows']==len(v['actions'])*4*(47-tick)
    assert v['counters']['tail_transition_rows']==2*len(v['actions'])*4*12
    np.testing.assert_array_equal(v['costs'],v['base']+v['tail']);np.testing.assert_array_equal(v['first'][0],v['first'][1])
    for a in range(2):np.testing.assert_array_equal(sum(v['component_costs'][a].values()),v['costs'][a])
    assert f.paired_delta(v,'rl','rl_root','rl_root')==(0.,0.)
    left=f.paired_delta(v,'rl','rl_root','wait');right=f.paired_delta(v,'rl','wait','rl_root')
    assert left[0]==-right[0] and left[1]==right[1]
    env.events=tuple((True,)*4 for _ in env.events)
    again=f.fork_values(c,s,root,ref,policy,14300000,4,tick)
    np.testing.assert_array_equal(v['costs'],again['costs'])
    assert env.state==s and e.previous.parameters_sha(policy)==before


def test_keep_cm_worker_dedup_forced_wait_and_repairs_cross_cutoff():
    c=replace(e.joint.make_config(13500000,'dense',48),service_time=((6,6),)*4);policy=model()
    s=DispatchState((4,3,1,0),(True,False,False,False),(2,0,0,0),(0,0),(-1,-1));root=((0,1),(1,0))
    actions,idx=f.root_actions(s,root,root)
    assert actions[idx[2]]==((0,1),) and idx[0]==idx[3] and len(actions)==3
    v=f.fork_values(c,s,root,root,policy,14300000,4,47,record_trace=True)
    assert f.scalar_replay(c,s,v,policy,47)
    np.testing.assert_array_equal(v['cutoff'][0]['remaining'][idx[0]],np.full((4,2),5))
    np.testing.assert_array_equal(v['cutoff'][0]['assigned'][idx[0]],np.tile([1,0],(4,1)))
    busy=replace(s,remaining=(6,6),assigned=(0,1))
    forced=f.fork_values(c,busy,(),(),policy,14300000,4,47,record_trace=True)
    assert forced['actions']==((),) and forced['label_indices']==(0,0,0,0) and f.scalar_replay(c,busy,forced,policy,47)
    np.testing.assert_array_equal(forced['cutoff'][0]['remaining'][0],np.full((4,2),5))
    for kwargs in [dict(scenarios=1,tick=47),dict(scenarios=4,tick=48),dict(scenarios=4,tick=47,tail=11)]:
        with pytest.raises(ValueError):f.fork_values(c,s,root,root,policy,1,**kwargs)


def statistic_fixture():
    cfg=dict(profile='full',conditions=['dense','skill_mask'],cohorts=[1,2],physical_shocks=[3],model_seeds=[4,5],
        origins=[JOINT,'best_rule'],landmarks=[6],analysis_seed=14200000,bootstrap_samples=400)
    rows=[dict(zip(e.KEYS,identity),pm_delta=0.,pm_exposure=0,now_delta=0.,now_exposure=0,busy_workers=0,
        rl_wait=1,reference_wait=1,at_risk_opportunities=2,young_opportunities=1,at_risk_pm_starts=0,young_pm_starts=0,
        structural_choice=1,rl_reference_delta=0.,deadline_reference_delta=0.,pm_deadline_delta=0.,pm_interaction_delta=0.,
        now_deadline_delta=0.,now_interaction_delta=0.) for identity in itertools.product(cfg['conditions'],cfg['cohorts'],
        cfg['physical_shocks'],cfg['model_seeds'],cfg['origins'],cfg['landmarks'])]
    return cfg,rows


def test_conditional_bootstrap_ratio_grid_and_empty_exposure():
    nums=np.array([[100.,0.],[20.,30.]]);dens=np.array([[10.,0.],[1.,3.]])
    actual=stats.crossed_ratio(nums,dens,np.array([[0,1],[0,0],[1,1]]),np.array([[0,1],[1,1],[0,0]]))
    assert actual[0]==150/14 and np.isnan(actual[1]) and actual[2]==20
    cfg,rows=statistic_fixture();summary,cells,_=stats.summarize(rows,cfg)
    assert len(summary['primary_contrasts'])==4 and len(cells)==16
    assert all(r['status']=='INSUFFICIENT' for r in summary['primary_contrasts'].values())
    for r in rows:r.update(now_exposure=1,now_delta=3.,now_deadline_delta=1.,now_interaction_delta=2.)
    summary,_,strata=stats.summarize(rows,cfg)
    assert summary['primary_contrasts']['dense:now']['estimate']==3.
    assert summary['primary_contrasts']['dense:now']['status']=='HIGHER_ROOT_COST' and strata[0]['now_interaction_mean']==2.
    cfg['profile']='smoke'
    assert all(v['status']=='ENGINEERING_ONLY' for v in stats.summarize(rows,cfg)[0]['primary_contrasts'].values())
    with pytest.raises(RuntimeError):stats.summarize(rows[:-1],cfg)
    with pytest.raises(RuntimeError):stats.summarize(rows+[rows[0]],cfg)
    rows[0]['now_exposure']=0
    with pytest.raises(ValueError):stats.summarize(rows,cfg)


@pytest.fixture
def source_fixture(tmp_path):
    source=tmp_path/'source';source.mkdir();cfg=e.previous.settings('full')
    origin=json.loads(json.dumps(e.contract()['joint_source_contract']));models={};hashes={}
    for seed in cfg['model_seeds']:
        p=model();path=source/'frozen_models'/f'seed_{seed}.pt';path.parent.mkdir(exist_ok=True);record=origin['models'][str(seed)]
        torch.save(dict(protocol=e.joint.VERSION,feature_contract=e.joint.FEATURE_CONTRACT,environment_version=e.joint.ENV_VERSION,
            algorithm=JOINT,train_seed=seed,settings=e.joint.settings('full'),update=record['update'],
            learned_steps=record['learned_steps'],state_dict=p.state_dict()),path)
        record['sha256']=e.sha(path);models[seed]=p;hashes[str(seed)]=e.previous.parameters_sha(p)
    profiles=[];cached={}
    for cohort,condition in itertools.product(cfg['cohorts'],cfg['conditions']):
        c=e.joint.make_config(cohort,condition,48);profiles.append(dict(cohort=cohort,condition=condition,config=asdict(c)));s=WaitingEnv(c).state
        for tick in cfg['landmarks']:
            rl,details=models[cfg['model_seeds'][0]].select(c,s,48-tick,48)
            ref=e.prior.rule_action(c,s,48-tick,origin['selected_rules'][condition]);cached[cohort,condition,tick]=(s,rl,details,ref)
    rows=[]
    for identity in itertools.product(cfg['conditions'],cfg['cohorts'],cfg['physical_shocks'],cfg['model_seeds'],cfg['origins'],cfg['landmarks']):
        row=dict(zip(e.KEYS,identity));s,rl,details,ref=cached[row['cohort'],row['condition'],row['tick']]
        rows.append(dict(**row,state=e.prior.compact(asdict(s)),rl_action=e.prior.compact(rl),rule_action=e.prior.compact(ref),**details,
            physical_seed=e.previous.physical_seed(row['cohort'],row['shock']),
            forecast_seed=e.previous.forecast_seed(cfg,row['cohort'],row['shock'],row['model_seed'],row['origin'],row['tick'])))
    e.prior.write_csv(source/'states.csv',rows);e.prior.write_json(source/'resolved_config.json',dict(settings=cfg,physical_profiles=profiles))
    e.prior.write_json(source/'source_contract.json',origin);snapshot=source/'source_snapshot';snapshot.mkdir()
    p=e.ROOT/'src/ht_pdm_fjsp/maintenance_waiting.py';shutil.copyfile(p,snapshot/p.name)
    m=dict(protocol=e.previous.VERSION,status='COMPLETED',audit='PASS',dirty=False,commit='engineering_fixture',settings=cfg,
        actual_counts=e.previous.counts(cfg),frozen_parameter_hashes=hashes,source_audit=dict(source_manifest_sha256=origin['manifest_sha256']))
    def repin():
        m['artifacts']={str(p.relative_to(source)):dict(sha256=e.sha(p),bytes=p.stat().st_size) for p in source.rglob('*') if p.is_file() and p.name!='manifest.json'}
        e.prior.write_json(source/'manifest.json',m)
        return dict(protocol=e.VERSION,state_source_run='synthetic_engineering',state_source_commit=m['commit'],
            state_source_manifest_sha256=e.sha(source/'manifest.json'),joint_source_contract=origin)
    return source,repin(),repin


def test_pinned_source_artifact_provenance_roots_grid_and_roles(source_fixture):
    source,locked,repin=source_fixture
    assert len(e.verify_source(source,locked)[2])==2560
    with pytest.raises(ValueError,match='manifest hash'):e.verify_source(source,{**locked,'state_source_manifest_sha256':'bad'})
    wrong=json.loads(json.dumps(locked));wrong['joint_source_contract']['models']['12100000']['update']+=1
    with pytest.raises(ValueError,match='provenance'):e.verify_source(source,wrong)
    rows=list(csv.DictReader((source/'states.csv').open()));original=rows[0]['rl_action'];rows[0]['rl_action']='[[3,1]]'
    e.prior.write_csv(source/'states.csv',rows)
    with pytest.raises(ValueError,match='artifact mismatch'):e.verify_source(source,locked)
    with pytest.raises(ValueError,match='policy root'):e.verify_source(source,repin())
    rows[0]['rl_action']=original;rows[0]['forecast_seed']='0';e.prior.write_csv(source/'states.csv',rows)
    with pytest.raises(ValueError,match='forecast role'):e.verify_source(source,repin())
    e.prior.write_csv(source/'states.csv',rows[1:])
    with pytest.raises(ValueError,match='incomplete'):e.verify_source(source,repin())


def test_preflight_dirty_threads_missing_input_nonempty_no_forecasts(tmp_path,monkeypatch):
    monkeypatch.setattr(e.platform,'platform',lambda:'engineering fixture')
    monkeypatch.setattr(e.subprocess,'check_output',lambda args,**kw:'dirty\n' if 'status' in args else 'commit\n');out=tmp_path/'dirty'
    with pytest.raises(ValueError,match='clean'):e.run(out,tmp_path/'absent','full')
    m=json.loads((out/'manifest.json').read_text());assert m['status']=='FAILED' and not m['new_full_forecast_outcomes_opened']
    with pytest.raises(ValueError,match='new/empty'):e.run(out,tmp_path/'absent')
    with pytest.raises(FileNotFoundError):e.run(tmp_path/'missing',tmp_path/'absent')
    monkeypatch.setattr(e.subprocess,'check_output',lambda args,**kw:'' if 'status' in args else 'commit\n');monkeypatch.setenv('OMP_NUM_THREADS','2')
    with pytest.raises(ValueError,match='NUM_THREADS'):e.run(tmp_path/'threads',tmp_path/'absent','full')


def test_smoke_io_cubes_roundtrip_repeatability_and_interruption(tmp_path,monkeypatch,source_fixture):
    source,locked,_=source_fixture;monkeypatch.setattr(e,'contract',lambda:locked)
    first=tmp_path/'first';second=tmp_path/'second';m=e.run(first,source)
    assert m['status']=='COMPLETED' and m['audit']=='PASS' and m['actual_counts']['states']==4 and not m['new_full_forecast_outcomes_opened']
    for name,record in m['artifacts'].items():assert e.sha(first/name)==record['sha256']
    e.run(second,source)
    for name in ['states.csv','root_values.csv','rollout_audits.csv','summary.json','profile_cells.csv','behavior_strata.csv']:
        assert (first/name).read_bytes()==(second/name).read_bytes()
    raw=np.load(first/'raw_forecast_costs.npz');assert raw['costs'].shape==(4,2,4,4)
    np.testing.assert_array_equal(raw['costs'],raw['base']+raw['tail']);boundaries=np.load(first/'raw_boundary_states.npz')
    assert boundaries['terminal_remaining'].shape==(4,2,4,4,2) and boundaries['cutoff_ages'].shape==(4,2,4,4,4)
    roots=list(csv.DictReader((first/'root_values.csv').open()));rows=list(csv.DictReader((first/'states.csv').open()))
    for i,row in enumerate(rows):
        for root in [r for r in roots if all(r[k]==row[k] for k in e.KEYS)]:
            for label in json.loads(root['labels']):np.testing.assert_array_equal(raw['costs'][i,f.ARMS.index(root['continuation']),f.LABELS.index(label)],json.loads(root['scenario_costs']))
    for seed in e.settings('smoke')['model_seeds']:assert e.sha(first/'frozen_models'/f'seed_{seed}.pt')==locked['joint_source_contract']['models'][str(seed)]['sha256']
    original=e.evaluate;calls=0
    def interrupted(*args):
        nonlocal calls
        calls+=1
        if calls==2:raise KeyboardInterrupt('engineering interruption')
        return original(*args)
    monkeypatch.setattr(e,'evaluate',interrupted);out=tmp_path/'interrupted'
    with pytest.raises(KeyboardInterrupt):e.run(out,source)
    m=json.loads((out/'manifest.json').read_text());assert m['status']=='FAILED' and not m['new_full_forecast_outcomes_opened']
    assert len(list(csv.DictReader((out/'states.partial.csv').open())))==1 and 'states.partial.csv' in m['artifacts'] and not (out/'states.csv').exists()
