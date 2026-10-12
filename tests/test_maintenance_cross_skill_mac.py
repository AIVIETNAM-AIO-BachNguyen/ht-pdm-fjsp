"""Prospective allocation intervention, seed-cluster statistics and artifact gates."""
from dataclasses import replace
import itertools
import json
import numpy as np
import pytest
from ht_pdm_fjsp import maintenance_cross_skill_mac as s
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import transition


def idle(ages=(3,3,3,3)):
    return DispatchState(ages,(False,)*4,(0,)*4,(0,0),(-1,-1))


@pytest.mark.parametrize('base',s.CHEAP)
@pytest.mark.parametrize('p',(0.,.45,1.))
def test_reciprocal_forecast_matches_scalar_current_subset_intervention(base,p):
    cfg=s.settings('smoke');spec=s.core.specs('smoke')[0];c=replace(s.core.make_config(spec),failure_probability=p)
    state=idle();tick=44;pool=s.core.subset_pool(c,state,tick,base);seed=71;scenarios=4
    bank=s.f.forecast(c,state,pool,tick,scenarios,seed,base)
    shocks=np.random.default_rng(seed).random((60-tick,scenarios,4))<p
    expected=np.zeros_like(bank['samples'])
    for i,action in enumerate(pool):
        for sample in range(scenarios):
            x=state
            for offset in range(60-tick):
                a=action if offset==0 else s.f.cheap(c,x,tick+offset,base if tick+offset<48 else 'deadline_matching')
                x,cost,_=transition(c,x,a,shocks[offset,sample]);expected[i,sample]+=cost
    np.testing.assert_allclose(bank['samples'],expected,rtol=0,atol=1e-10)


def arithmetic(cfg,assignment=95,joint=90,local=2):
    spec=s.core.specs(cfg['profile'])[0]
    rows=[dict(candidate=spec['candidate'],method=m,seed=seed,phase='confirmation',physical_seed=s.physical_seed(seed),extended_cost=assignment if m=='assignment_rollout' else joint if m=='joint_rollout' else 100,
        policy_cost=assignment if m=='assignment_rollout' else joint if m=='joint_rollout' else 100,availability=.7,terminal_pending=0) for seed,m in itertools.product(cfg['confirmation_seeds'],s.METHODS)]
    labels=[dict(candidate=spec['candidate'],phase='confirmation',seed=seed,tick=t,choice=1,**{f'gain_{ref}':local for ref in s.REFERENCES}) for seed,t in itertools.product(cfg['confirmation_seeds'],s.LANDMARKS)]
    return spec,rows,labels


def test_exact_eighteen_gates_and_failures_do_not_fallback_to_joint_only():
    cfg=s.settings('full');spec,rows,local=arithmetic(cfg)
    result=s.summarize(cfg,spec,'isolated_endpoint',rows,local)
    assert result['scientific_status']=='CROSS_SKILL_SETTING_CONFIRMED' and len(result['gates'])==18
    assert all(c['confidence']==1-.05/18 for c in result['comparisons'].values())
    for assign,joint,gain in ((99.5,90,2),(95,99.5,2),(95,90,.49)):
        spec,r,l=arithmetic(cfg,assign,joint,gain)
        assert s.summarize(cfg,spec,'isolated_endpoint',r,l)['scientific_status']=='NOT_CONFIRMED'
    rows[0]['availability']=.1
    # Absolute assignment operational guard is explicit, not only joint health.
    for r in rows:
        if r['method']=='assignment_rollout':r['availability']=.1
    assert not s.summarize(cfg,spec,'isolated_endpoint',rows,local)['gates']['assignment_absolute_availability']
    cfg=s.settings('smoke');spec,r,l=arithmetic(cfg)
    assert s.summarize(cfg,spec,'deadline_matching',r,l)['scientific_status']=='ENGINEERING_ONLY'


def test_seed_clusters_not_landmark_pseudoreplication_and_all_zeros_retained():
    cfg=s.settings('smoke');spec,rows,local=arithmetic(cfg)
    for r in local:r['gain_baseline']=2 if r['seed']==cfg['confirmation_seeds'][0] else 0
    values,labels=s.matrices(rows,local,cfg['confirmation_seeds'],spec['candidate'])
    assert labels.shape==(2,4) and labels[:,0].tolist()==[2,0]
    result=s.summarize(cfg,spec,'deadline_matching',rows,local)
    assert result['comparisons']['local_baseline']['estimate']==1
    assert result['comparisons']['local_baseline']['ci'][0]==0
    with pytest.raises(RuntimeError,match='landmark grid'):s.matrices(rows,local[:-1],cfg['confirmation_seeds'],spec['candidate'])


def test_full_screen_cannot_promote_failed_best_or_controls_and_roles_are_fresh():
    cfg=s.settings('full');spec,rows,local=arithmetic(cfg)
    rows=[dict(candidate=spec['candidate'],method=m,seed=seed,extended_cost=90 if m in s.PLANNERS else 100,policy_cost=90 if m in s.PLANNERS else 100,availability=.7,terminal_pending=0) for seed,m in itertools.product(cfg['screening_seeds'],s.METHODS)]
    local=[dict(candidate=spec['candidate'],seed=seed,tick=t,choice=1,**{f'gain_{ref}':2 for ref in s.REFERENCES}) for seed,t in itertools.product(cfg['screening_seeds'],s.LANDMARKS)]
    choices={spec['candidate']:dict(base='deadline_matching')}
    assert s.screen_choice(cfg,[spec],choices,rows,local)['winner']==spec['candidate']
    for r in local:r['gain_endpoint_dp']=.74
    selection=s.screen_choice(cfg,[spec],choices,rows,local)
    assert selection['winner'] is None and selection['status']=='NO_SCREEN_CANDIDATE'
    for r in local:r['gain_endpoint_dp']=2
    assert s.screen_choice(cfg,[{**spec,'selectable':False}],choices,rows,local)['winner'] is None
    assert s.seed_audit(cfg)['status']=='PASS'
    raw=cfg['screening_seeds'][0]
    assert s.bank_seed(cfg,'screening',raw,0)!=s.bank_seed(cfg,'screening',raw,0,True)
    assert s.bank_seed(cfg,'screening',raw,0)!=s.bank_seed(cfg,'confirmation',raw,0)
    with pytest.raises(ValueError,match='locked'):s.seed_audit({**cfg,'scenarios':64})


def typed(path,strings):
    return [{k:v if k in strings else float(v) if '.' in v or 'e' in v.lower() else int(v) for k,v in r.items()} for r in s.records(path)]


def test_real_smoke_replay_bindings_and_root_provenance_tamper(tmp_path):
    out=tmp_path/'run';m=s.run('smoke',out)
    assert m['status']=='COMPLETED' and m['audit']=='PASS' and m['scientific_status']=='ENGINEERING_ONLY' and not m['confirmation_outcomes_opened']
    assert m['actual_counts']==dict(development=24,screening=40,confirmation=10,landmarks=70)
    for name,b in m['files'].items():assert s.f.old.screen.sha(out/name)==b['sha256'] and (out/name).stat().st_size==b['bytes']
    assert (out/'episodes.partial.csv').read_bytes()==(out/'episodes.csv').read_bytes()
    with pytest.raises(FileExistsError):s.run('smoke',out)
    rows=sum((typed(out/name,('phase','candidate','method')) for name in ('development_episodes.csv','screening_episodes.csv','episodes.csv')),[])
    labels=typed(out/'screening_landmarks.csv',('phase','candidate','subset'))+typed(out/'landmarks.csv',('phase','candidate','subset'))
    roots=list(s.records(out/'screening_roots.csv'))+list(s.records(out/'landmark_roots.csv'))
    choices=json.loads((out/'baseline_choices.json').read_text());specs=json.loads((out/'candidates.json').read_text());cfg=s.settings('smoke')
    decision=list(s.records(out/'decisions.csv'))
    target=next(r for r in decision if json.loads(r['diagnostic'])['planned'])
    diagnostic=json.loads(target['diagnostic']);diagnostic['selected_root']=-diagnostic['candidate_count']+diagnostic['selected_root']
    target['diagnostic']=json.dumps(diagnostic);s.write_csv(out/'tampered.csv',decision)
    with pytest.raises(RuntimeError,match='forecast diagnostic contract'):
        s.replay({**cfg,'audit_seeds':0},specs,choices,out/'tampered.csv',rows,labels,roots)
    roots[0]['reference_quality']=str(1-int(roots[0]['reference_quality']))
    with pytest.raises(RuntimeError,match='reference root flag'):s.replay(cfg,specs,choices,out/'decisions.csv',rows,labels,roots)
    rows[0]['physical_seed']+=1
    with pytest.raises(RuntimeError,match='seed provenance'):s.replay(cfg,specs,choices,out/'decisions.csv',rows,labels,roots)


def test_timeout_records_failure_and_full_must_be_clean(tmp_path,monkeypatch):
    original=s.settings;monkeypatch.setattr(s,'seed_audit',lambda cfg:{'status':'PASS','historical_registry_hashes':{}})
    monkeypatch.setattr(s,'settings',lambda profile:{**original(profile),'max_seconds':-1})
    with pytest.raises(TimeoutError):s.run('smoke',tmp_path/'timed')
    assert json.loads((tmp_path/'timed'/'manifest.json').read_text())['status']=='FAILED'
    monkeypatch.setattr(s.f.old.screen,'git_metadata',lambda:dict(dirty=True,commit='fake'))
    with pytest.raises(RuntimeError,match='clean committed'):s.run('full',tmp_path/'full')
