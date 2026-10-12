"""Balanced family, health-first ranking, locked inference and real replay."""
import itertools
import json
import numpy as np
import pytest
from ht_pdm_fjsp import maintenance_balanced_search_mac_v2 as s


def test_family_has_eight_balanced_profiles_and_nonselectable_controls():
    specs = s.specifications('full')
    assert len(specs)==10 and sum(x['selectable'] for x in specs)==8
    for spec in specs[:8]:
        c=s.core.make_config(spec)
        assert sum(a < b for a,b in c.service_time)==2
        assert sum(a > b for a,b in c.service_time)==2
        assert all(a>0 and b>0 for a,b in c.service_time)
        assert set(v for row in c.restoration for v in row)=={.5,1.}
        assert c.unavailable_cost in (9.,12.)
    assert not specs[-2]['selectable'] and specs[-1]['candidate']=='exchangeable'
    assert s.settings('full')['confidence']>s.engine.settings('full')['confidence']
    assert s.settings('full')['bootstrap_samples']==160000
    assert s.seed_audit(s.settings('full'))['status']=='PASS'
    with pytest.raises(ValueError,match='locked'):
        s.seed_audit({**s.settings('full'),'confidence':.95})


def ranking_data():
    cfg=s.settings('full');specs=s.specifications('smoke')[:2]
    rows=[];local=[];choices={x['candidate']:dict(base='deadline_matching') for x in specs}
    for i,spec in enumerate(specs):
        for seed,method in itertools.product(cfg['screening_seeds'],s.METHODS):
            cost=(80 if i==0 else 92) if method=='assignment_rollout' else (75 if i==0 else 88) if method=='joint_rollout' else 100
            av=.7+(.001 if i==0 else .03) if method in s.engine.PLANNERS else .7
            rows.append(dict(candidate=spec['candidate'],method=method,seed=seed,extended_cost=cost,policy_cost=cost,availability=av,terminal_pending=0))
        for seed,tick in itertools.product(cfg['screening_seeds'],s.LANDMARKS):
            local.append(dict(candidate=spec['candidate'],seed=seed,tick=tick,choice=1,**{f'gain_{r}':2 for r in s.engine.REFERENCES}))
    return cfg,specs,choices,rows,local


def test_health_stability_wins_over_larger_cost_gain_and_no_fallback():
    cfg,specs,choices,rows,local=ranking_data()
    selection=s.screen_choice(cfg,specs,choices,rows,local)
    assert selection['winner']==specs[1]['candidate']
    assert all(x['eligible'] for x in selection['scores'])
    assert selection['scores'][0]['score']>selection['scores'][1]['score']
    # Paired sample variability enters exactly once per raw seed.
    name=specs[0]['candidate'];seeds=cfg['screening_seeds']
    for r in rows:
        if r['candidate']==name and r['method'] in s.engine.PLANNERS:
            r['availability'] += .005 if seeds.index(r['seed'])%2 else -.005
    score=s.screen_choice(cfg,specs,choices,rows,local)['scores'][0]['health_stability_score']
    differences=np.array([.001+(.005 if i%2 else -.005) for i in range(32)])
    assert score==pytest.approx(differences.mean()-2*differences.std(ddof=1)/np.sqrt(32))
    for r in rows:
        if r['method'] in s.engine.PLANNERS:r['availability']=.69
    assert s.screen_choice(cfg,specs,choices,rows,local)['winner'] is None
    # A control cannot become eligible even with exceptional health/cost.
    cfg,specs,choices,rows,local=ranking_data()
    for x in specs:x['selectable']=False
    assert s.screen_choice(cfg,specs,choices,rows,local)['winner'] is None


def test_real_smoke_full_grids_replay_nullcontrol_source_binding(tmp_path):
    out=tmp_path/'smoke';m=s.run('smoke',out)
    assert m['status']=='COMPLETED' and m['audit']=='PASS'
    assert m['scientific_status']=='ENGINEERING_ONLY' and not m['confirmation_outcomes_opened']
    assert m['actual_counts']==dict(development=24,screening=40,confirmation=10,landmarks=70)
    summary=json.loads((out/'summary.json').read_text())
    assert len(summary['gates'])==18 and summary['family_alpha']==.0125
    assert all(x['confidence']==1-.0125/18 for x in summary['comparisons'].values())
    for name,data in m['files'].items():
        assert s.f.old.screen.sha(out/name)==data['sha256']
    assert (out/'episodes.csv').read_bytes()==(out/'episodes.partial.csv').read_bytes()
    with pytest.raises(FileExistsError):s.run('smoke',out)


@pytest.mark.parametrize('base',s.CHEAP)
@pytest.mark.parametrize('name',('balanced_a2_b4_u09','balanced_a2_b4_u12','balanced_a3_b5_u09','balanced_a3_b5_u12'))
def test_changed_downtime_price_and_heterogeneous_forecast_match_scalar(base,name):
    from ht_pdm_fjsp.maintenance_dispatch import DispatchState
    from ht_pdm_fjsp.maintenance_waiting import transition
    spec=next(x for x in s.specifications('full') if x['candidate']==name)
    c=s.core.make_config(spec)
    state=DispatchState((3,4,6,7),(False,True,True,False),(0,2,3,1),(0,0),(-1,-1))
    tick=44;scenarios=4;seed=71;pool=s.core.subset_pool(c,state,tick,base)
    forecast=s.f.forecast(c,state,pool,tick,scenarios,seed,base)
    shocks=np.random.default_rng(seed).random((60-tick,scenarios,4))<c.failure_probability
    expected=np.zeros_like(forecast['samples'])
    for i,action in enumerate(pool):
        for sample in range(scenarios):
            x=state
            for offset in range(60-tick):
                current=action if offset==0 else s.f.cheap(c,x,tick+offset,base if tick+offset<48 else 'deadline_matching')
                x,cost,_=transition(c,x,current,shocks[offset,sample]);expected[i,sample]+=cost
    np.testing.assert_allclose(forecast['samples'],expected,rtol=0,atol=1e-10)
