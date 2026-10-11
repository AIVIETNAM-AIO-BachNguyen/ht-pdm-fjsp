import csv
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import itertools
import json

import numpy as np
import pytest

from ht_pdm_fjsp import maintenance_confirmation as e
from ht_pdm_fjsp import maintenance_boundary_sensitivity as b
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed
from ht_pdm_fjsp.maintenance_heterogeneous_model import fixed_candidates
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, transition
from ht_pdm_fjsp.maintenance_solver_diagnostics import Requests


def tiny():
    return WaitingConfig(2,2,15,2,.6,((3,2),(2,0)),((1.,1.),(1.,1.)),waiting_price=12.)


def test_full_counts_seeds_and_fresh_physical_panels():
    cfg=e.settings('full')
    assert e.counts(cfg)==dict(development_episodes=0,test_episodes=8192,
        base_decisions=393216,tail_decisions=98304,test_decisions=491520)
    assert e.seed_audit(cfg)['status']=='PASS'
    assert b.seed_audit(b.settings('full'))['status']=='PASS'
    assert e.prior.seed_audit(e.prior.settings('full'))['status']=='PASS'
    with pytest.raises(ValueError,match='registry'):
        e.seed_audit({**cfg,'analysis_seed':42})
    full,rows=e.build_panels(cfg)
    smoke,_=e.build_panels(e.settings('smoke'))
    signatures={b.signature(c) for c in full.values()}
    assert len(full)==len(signatures)==len(rows)==256
    historical={b.frozen(s) for s in e.exclusion_registry()['previous_physical_signatures']['skill_mask']}
    assert len(historical)==181 and not signatures & historical
    assert not signatures & {b.signature(c) for c in smoke.values()}
    assert e.build_panels(cfg)==(full,rows)
    baseline=json.loads((e.ROOT/f'configs/{e.VERSION}_frozen_baseline.json').read_text())
    assert baseline['baseline']==e.BASELINE=='deadline_matching'
    assert baseline['source_selection']['best_rule']==e.BASELINE
    assert baseline['source_commit']=='54d17f9ce9c195c59ec6745868ae9b3450c0b1c5'


def test_shortcut_selects_the_exact_sole_candidate_and_skips_forecasting(monkeypatch):
    c=replace(tiny(),service_time=((3,2),(2,2)))
    state=DispatchState((2,2),(True,False),(3,0),(3,2),(0,1))
    assert fixed_candidates(c,state)==((),)
    expected,diag=b.forecast_plan(c,state,4,3,81,8,fixed=True)
    assert expected==()
    monkeypatch.setattr(b,'forecast_plan',lambda *a,**k:pytest.fail('singleton forecast executed'))
    action,fast=e.forecast_plan(c,state,4,3,81,8)
    assert action==expected and fast['singleton_skipped']
    assert fast['forecast_ticks']==0 and fast['terminal_ticks']==0
    assert fast['selected_score'] is None and fast['candidates']==((),)


def test_nontrivial_forecast_unchanged_and_future_tapes_private():
    c=tiny();env=WaitingEnv(c);env.reset(71)
    expected,old=b.forecast_plan(c,env.state,4,3,81,16,fixed=True)
    action,diag=e.forecast_plan(c,env.state,4,3,81,16)
    assert action==expected and not diag.pop('singleton_skipped') and diag==old
    env.events=tuple((True,True) for _ in range(c.horizon))
    after,after_diag=e.forecast_plan(c,env.state,4,3,81,16)
    assert action==after and after_diag['means']==diag['means']
    for h,tail,scenarios in ((0,3,8),(4,-1,8),(4,3,1)):
        with pytest.raises(ValueError):
            e.forecast_plan(c,env.state,h,tail,81,scenarios)
    with pytest.raises(ValueError,match='fixed'):
        e.forecast_plan(c,env.state,4,3,81,16,fixed=False)


def test_full_size_forecast_equivalence_on_engineering_state_only():
    panels,_=e.build_panels(e.settings('smoke'))
    c=replace(next(iter(panels.values())),horizon=60)
    env=WaitingEnv(c);env.reset(11500000)
    old,diag=b.forecast_plan(c,env.state,48,12,11500001,128,fixed=True)
    action,actual=e.forecast_plan(c,env.state,48,12,11500001,128)
    assert action==old
    if not actual['singleton_skipped']:
        assert actual['means']==diag['means'] and actual['forecast_ticks']==60


def strip_latency(row):
    return {k:v for k,v in row.items() if 'latency' not in k}


def test_whole_trajectories_equal_with_shortcut_and_inherited_episode(monkeypatch):
    cfg=e.settings('smoke');panels,_=e.build_panels(cfg)
    # Same namespace is used only to check inherited physical semantics.
    monkeypatch.setattr(e,'VERSION',b.VERSION)
    for cohort,shock,controller in itertools.product(cfg['test_cohorts'],cfg['test_shocks'],e.CONTROLLERS):
        c=panels[cohort]
        args=(c,cohort,shock,e.CONDITION,cfg['horizon'],controller,e.BASELINE)
        optimized=e.episode(*args,cfg)
        unoptimized=e.episode(*args,{**cfg,'singleton_fast_path':False})
        inherited=b.episode(*args,cfg)
        assert strip_latency(optimized)==strip_latency(unoptimized)==strip_latency(inherited)


def synthetic(cfg,rule=100.,planner=90.,pending=.2):
    rows=[]
    for cohort,shock,name in itertools.product(cfg['test_cohorts'],cfg['test_shocks'],e.CONTROLLERS):
        rows.append(dict(condition=e.CONDITION,horizon=cfg['horizon'],cohort=cohort,shock=shock,controller=name,
            interior_cost_per_machine_tick=rule if name=='best_rule' else planner,
            interior_pending_mean=.2 if name=='best_rule' else pending,
            interior_failed_mean=.4,interior_overdue_per_tick=.01,optional_wait_mean=None))
    return rows


def test_joint_decision_strict_margins_and_engineering_labels():
    cfg=e.settings('smoke');cfg['profile']='full'
    result,differences=e.summary(synthetic(cfg),cfg)
    assert result['decision']=='GO' and result['advance_to_training']
    assert len(differences)==4*len(cfg['test_cohorts'])
    assert all(c['confidence']==.9875 for c in result['contrasts'].values())
    assert result['means']['best_rule']['optional_wait_mean'] is None
    failed,_=e.summary(synthetic(cfg,pending=.4),cfg)
    assert failed['decision']=='NO_GO' and not failed['advance_to_training']
    assert failed['contrasts']['pending_guard']['profile_wins']==0
    assert failed['contrasts']['pending_guard']['profile_losses']==2
    at_margin,_=e.summary(synthetic(cfg,planner=99.),cfg)
    assert at_margin['decision']=='INCONCLUSIVE'
    below,_=e.summary(synthetic(cfg,planner=99.5),cfg)
    assert below['decision']=='NO_GO'
    cfg['profile']='smoke'
    engineering,_=e.summary(synthetic(cfg),cfg)
    assert engineering['decision']=='ENGINEERING_ONLY' and not engineering['advance_to_training']
    assert all(c['status']=='ENGINEERING_ONLY' for c in engineering['contrasts'].values())
    assert e.decide({'a':{'status':'PASS'},'b':{'status':'INCONCLUSIVE'}})=='INCONCLUSIVE'
    assert e.decide({'a':{'status':'FAIL'},'b':{'status':'INCONCLUSIVE'}})=='NO_GO'


def test_profile_bootstrap_paired_ratio_and_malformed_grids():
    cfg=e.settings('smoke');cfg['bootstrap_samples']=1000
    rows=synthetic(cfg)
    for r in rows:
        first=r['cohort']==cfg['test_cohorts'][0]
        r['interior_cost_per_machine_tick']=(100 if first else 2) if r['controller']=='best_rule' else (90 if first else 1)
    result,_=e.summary(rows,cfg)
    got=result['contrasts']['cost_headroom']
    rng=np.random.default_rng(role_seed(cfg['analysis_seed'],f'{e.VERSION}:profile_bootstrap'))
    draws=rng.integers(2,size=(1000,2))
    a=np.array([100.,2.]);delta=np.array([10.,1.])
    samples=delta[draws].mean(1)/a[draws].mean(1)
    assert got['estimate']==pytest.approx(11/102)
    np.testing.assert_array_equal(got['confidence_interval'],np.quantile(samples,[.00625,.99375]))
    assert got['profiles']==2 and got['shocks_per_profile']==2
    for altered in (rows[:-1],rows+rows[:1],[{**r,'condition':'wrong'} for r in rows]):
        with pytest.raises(RuntimeError,match='grid'):
            e.summary(altered,cfg)
    for r in rows:r['interior_cost_per_machine_tick']=0.
    with pytest.raises(RuntimeError,match='positive'):
        e.summary(rows,cfg)


def read(out,name):
    return list(csv.DictReader((out/f'{name}.csv').open()))


def test_smoke_independent_physical_requests_work_counts_hashes_and_reproducibility(tmp_path):
    out=tmp_path/'smoke';manifest=e.run(out)
    assert manifest['status']=='COMPLETED' and manifest['audit']=='PASS'
    assert not manifest['full_test_outcomes_opened']
    assert manifest['actual_counts']==dict(development_episodes=0,test_episodes=8,base_decisions=96,tail_decisions=24,test_decisions=120)
    for name,metadata in manifest['artifacts'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==metadata['sha256']
    cfg=e.settings('smoke');panels,_=e.build_panels(cfg)
    key=lambda r:tuple(r[k] for k in ('condition','cohort','shock','horizon','controller'))
    grouped=defaultdict(list)
    for row in read(out,'decisions'):grouped[key(row)].append(row)
    recon_requests=[];recon_carry=[]
    for result in read(out,'episodes'):
        cond,cohort,shock,h,name=key(result);h=int(h)
        c=replace(panels[int(cohort)],horizon=h+cfg['tail_ticks'])
        env=WaitingEnv(c);env.reset(int(grouped[key(result)][0]['env_seed']))
        base,extended=Requests(env.state),Requests(env.state)
        totals=defaultdict(float)
        for row in grouped[key(result)]:
            t=int(row['time']);state=DispatchState(**{k:tuple(v) for k,v in json.loads(row['state']).items()})
            assert state==env.state
            action=tuple(map(tuple,json.loads(row['action'])))
            if t>=h or name=='best_rule':
                assert action==e.prior.rule_action(c,state,c.horizon-t if t>=h else h-t,e.BASELINE)
            else:assert action in fixed_candidates(c,state)
            following,cost,info=transition(c,state,action,env.events[t]);env.step(action)
            assert following==env.state and json.loads(json.dumps(asdict(following)))==json.loads(row['following'])
            assert info==json.loads(row['physical_metrics'])
            totals[row['phase']]+=cost;extended.step(t,state,action,following)
            if t<h:base.step(t,state,action,following)
            if t+1==h:cut=base.finish(h)
        end=extended.finish(c.horizon)
        identity={k:result[k] for k in ('condition','cohort','shock','horizon','controller')}
        for scope,records in (('cutoff',cut),('extended',end)):
            for r in records:recon_requests.append({**identity,'scope':scope,**{k:str(v) for k,v in r.items()},'overdue':str(r['wait']>c.waiting_limit)})
        for r in cut:
            if r['censored']:
                later=next(x for x in end if (x['machine'],x['request'])==(r['machine'],r['request']))
                recon_carry.append({**identity,**{k:str(v) for k,v in dict(machine=r['machine'],request=r['request'],arrival=r['arrival'],
                    wait_at_cutoff=r['wait'],started_in_tail=later['served'],wait_at_start_or_end=later['wait'],censored_after_tail=later['censored']).items()}})
        assert float(result['base_objective'])==totals['base']
        assert float(result['extended_objective'])==sum(totals.values())
    assert recon_requests==read(out,'requests') and recon_carry==read(out,'carryover')
    latency=read(out,'latency');assert any(r['singleton_skipped']=='True' for r in latency)
    for r in latency:
        count,scenarios,ticks=(int(r[k]) for k in ('candidate_count','forecast_scenarios','forecast_ticks'))
        assert int(r['forecast_transition_samples'])==count*scenarios*ticks
        if r['singleton_skipped']=='True':
            assert count==1 and ticks==scenarios==0 and r['predicted_action_cost']==''
            assert int(r['configured_forecast_ticks'])>0
    initial=defaultdict(list)
    for rows in grouped.values():initial[rows[0]['cohort'],rows[0]['shock']].append((rows[0]['env_seed'],rows[0]['state']))
    assert all(a[0]==a[1] for a in initial.values())
    assert all(r['invalid_actions']=='0' and r['fixed_allocation_audit']=='PASS' for r in read(out,'coordination'))
    assert json.loads((out/'summary.json').read_text())['decision']=='ENGINEERING_ONLY'
    second=tmp_path/'repeat';e.run(second)
    assert [strip_latency(r) for r in read(out,'episodes')]==[strip_latency(r) for r in read(second,'episodes')]
    for name in ('decisions','requests','carryover','coordination','time_profiles','profile_differences'):
        assert (out/f'{name}.csv').read_bytes()==(second/f'{name}.csv').read_bytes()
    with pytest.raises(ValueError,match='new/empty'):e.run(out)
    with pytest.raises(ValueError,match='CPU'):e.run(tmp_path/'cuda',device='cuda')


def test_interruption_and_full_preflight_do_not_open_outcomes(tmp_path,monkeypatch):
    original=e.episode
    def interrupt(*args,**kwargs):
        if args[5]=='fixed_tail':raise KeyboardInterrupt('injected interruption')
        return original(*args,**kwargs)
    monkeypatch.setattr(e,'episode',interrupt)
    out=tmp_path/'interrupted'
    with pytest.raises(KeyboardInterrupt):e.run(out)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='FAILED' and 'KeyboardInterrupt' in manifest['error']
    assert not manifest['full_test_outcomes_opened'] and len(read(out,'episodes.partial'))==1
    assert not (out/'episodes.csv').exists()
    monkeypatch.setattr(e.subprocess,'check_output',lambda args,**kw:'DIRTY' if args[1]=='status' else 'fake-commit')
    with pytest.raises(ValueError,match='clean'):e.run(tmp_path/'dirty','full')
    monkeypatch.setattr(e.subprocess,'check_output',lambda args,**kw:'' if args[1]=='status' else 'fake-commit')
    monkeypatch.delenv('OMP_NUM_THREADS',raising=False)
    with pytest.raises(ValueError,match='THREADS'):e.run(tmp_path/'threads','full')
    for name in ('dirty','threads'):
        m=json.loads((tmp_path/name/'manifest.json').read_text())
        assert m['status']=='FAILED' and not m['full_test_outcomes_opened']
