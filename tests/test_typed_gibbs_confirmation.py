import copy,csv,json,shutil,time
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_typed_gibbs_confirmation as c


def fixture_values(t=2,s=3):return np.arange(7*t*s*4,dtype=float).reshape(7,t,s,4)+10


def explicit_crossed(values,ti,si):
    return np.array([[[np.mean([values[m,t,s,f] for t in tr for s in sr]) for f in range(4)] for m in range(7)] for tr,sr in zip(ti,si)])


def test_crossed_count_weights_match_explicit_cartesian_and_preserve_pairing():
    v=fixture_values();ti=np.array([[0,0],[1,0],[1,1]]);si=np.array([[0,0,2],[1,2,1],[2,2,2]]);actual=c.crossed_means(v,ti,si);np.testing.assert_allclose(actual,explicit_crossed(v,ti,si),atol=1e-12,rtol=0)
    v[1]=v[0];effect=c.endpoint_effects(c.crossed_means(v,ti,si));assert (effect[:,0]==0).all()


def test_cost_endpoint_ratio_of_crossed_means_not_mean_episode_ratio():
    v=np.full((7,1,2,4),1.);v[0,0,:,0]=[1,100];v[1,0,:,0]=[1,200];point=c.endpoint_effects(v.mean((1,2))[None])[0,0];assert point==pytest.approx(1-101/201) and abs(point-.25)>.1


def test_bootstrap_actual_indices_regeneration_fixed_chunk_and_all_methods():
    v=fixture_values();effects,ti,si=c.bootstrap(v,127,draws=17,chunk=5,progress=False);rng=np.random.default_rng(127);et=[];es=[]
    for start in range(0,17,5):
        n=min(5,17-start);et.append(rng.integers(0,2,(n,2),dtype=np.int64));es.append(rng.integers(0,3,(n,3),dtype=np.int64))
    et=np.concatenate(et);es=np.concatenate(es);np.testing.assert_array_equal(ti,et);np.testing.assert_array_equal(si,es);assert ti.dtype==si.dtype==np.uint8
    np.testing.assert_allclose(effects,c.endpoint_effects(explicit_crossed(v,et,es)),atol=1e-12,rtol=0)
    again=c.bootstrap(v,127,draws=17,chunk=5,progress=False)
    for a,b in zip((effects,ti,si),again):np.testing.assert_array_equal(a,b)


def passing_points():return np.array([.01,.005,.005,.005,.005,.005,0.,0.,0.,0.,0.,0.])


@pytest.mark.parametrize('index',range(12))
def test_each_oriented_endpoint_bound_rejects_and_map_postive_is_strict(index):
    point=passing_points();effects=np.tile(point,(100,1));bad=[.0049,-.0101,-.0101,-.0101,0.,0.,-.0101,.0251,.2501,-.0101,.0251,.2501];effects[:,index]=bad[index];bounds=c.confidence_bounds(point,effects);assert len(bounds)==12
    assert not bounds[c.ENDPOINTS[index]]['passed'] and all(v['passed'] for key,v in bounds.items() if key!=c.ENDPOINTS[index]);assert bounds[c.ENDPOINTS[index]]['point']==point[index]


def test_exact_practical_health_boundaries_and_one_sided_alpha():
    point=np.array([.005,-.01,-.01,-.01,1e-9,1e-9,-.01,.025,.25,-.01,.025,.25]);bounds=c.confidence_bounds(point,np.tile(point,(100,1)));assert all(b['passed'] for b in bounds.values());assert c.ALPHA==pytest.approx(.05/56/12)
    assert all(b['one_sided_alpha']==c.ALPHA for b in bounds.values());assert [b['orientation'] for b in bounds.values()]==['lower']*7+['upper','upper','lower','upper','upper']
    assert not c.superiority_allowed(bounds,True)
    for a in c.e.REFERENCES:bounds['gain_vs_'+a]['bound']=.01
    assert c.superiority_allowed(bounds,True) and not c.superiority_allowed(bounds,False)
    bounds['gain_vs_ppo']['bound']=.009999;assert not c.superiority_allowed(bounds,True)


@pytest.fixture(scope='module')
def engineering(tmp_path_factory):
    torch.set_num_threads(1);root=tmp_path_factory.mktemp('typed_gibbs_cf_phase_integration');cfg=c.e.settings('engineering');cfg.update(env_steps=96,checkpoints=[96],test_seeds=[380700001]);t=cfg['train_seeds'][0]
    with pytest.MonkeyPatch.context() as patch:
        original=c.e.ResourceEnv
        def checked_env(seed,*args,**kwargs):
            assert seed in (380700000,380700001),'scientific CF/TEST physical seed in unit tests'
            return original(seed,*args,**kwargs)
        patch.setattr(c.e,'ResourceEnv',checked_env)
        results=[c.e.fit_job((cfg,a,t,str(root),time.monotonic())) for a in c.e.ARMS];dev=[row for r in results for row in r['development']]
        for name in ('training_progress','training_episodes'):c.e.old.merge_journals(root,name,results)
        audit=c.e.audit_training(root,cfg,results,dev,lambda:None);c.e.write_json(root/'training_grid_audit.json',audit);choices={f"{r['method']}:{t}":r['fixed'] for r in results};frozen=c.freeze_models(root,cfg,choices);gate=c.open_test_gate(root,cfg);tested=[c.test_job((cfg,a,t,str(root),time.monotonic())) for a in c.e.ARMS]+[c.test_job((cfg,'deadline',-1,str(root),time.monotonic()))]
    for name in ('coordination','decisions'):c.e.old.merge_journals(root,name,tested)
    rows=[row for item in tested for row in item['rows']];physical=c.e.audit_final(root,rows,dict(cfg,development_seeds=cfg['test_seeds']),lambda:None)
    return root,cfg,choices,frozen,gate,rows,physical


def test_all7_engineering_fit_freeze_test_journals_and_independent_primitive_clock(engineering):
    root,cfg,choices,frozen,gate,rows,physical=engineering;assert frozen['model_count']==7 and len(rows)==8 and physical['physical_transitions']==480 and physical['policy_decisions']==384;assert c.check_test_gate(root,cfg)==frozen;assert gate['test_action_updates']==0
    for a in c.e.ARMS:
        p=root/choices[f'{a}:380700000']['path'];m=c.e.load_model(p,a,cfg,380700000,96);assert c.e.learner.parameter_digest(m)==choices[f'{a}:380700000']['parameter_sha256']


def test_no_test_before_complete_seal_or_mutated_choices_or_models(engineering,tmp_path,monkeypatch):
    root,cfg,choices,*_=engineering
    def fail(*args,**kwargs):raise AssertionError('TEST constructed before gate')
    monkeypatch.setattr(c.e,'evaluate',fail)
    with pytest.raises(FileNotFoundError):c.test_job((cfg,c.e.CORE[0],380700000,str(tmp_path),time.monotonic()))
    with pytest.raises(AssertionError,match='complete fixed'):c.freeze_models(tmp_path,cfg,{key:value for key,value in choices.items() if key!=next(iter(choices))})
    copied=tmp_path/'mutated';shutil.copytree(root,copied);p=copied/'checkpoint_choices.json';p.write_text(p.read_text()+' ')
    with pytest.raises(AssertionError,match='binding'):c.test_job((cfg,c.e.CORE[0],380700000,str(copied),time.monotonic()))
    p.write_bytes((root/'checkpoint_choices.json').read_bytes());first=next(iter(choices.values()));model=copied/first['path'];model.write_bytes(model.read_bytes()+b'X')
    with pytest.raises(AssertionError,match='mutated'):c.test_job((cfg,c.e.CORE[0],380700000,str(copied),time.monotonic()))


def test_freeze_and_test_cannot_be_silently_reopened(engineering):
    root,cfg,choices,*_=engineering
    with pytest.raises(AssertionError,match='overwrite'):c.freeze_models(root,cfg,choices)
    with pytest.raises(AssertionError,match='reopening'):c.open_test_gate(root,cfg)
    with pytest.raises(AssertionError,match='overwrite'):c.test_job((cfg,c.e.CORE[0],380700000,str(root),time.monotonic()))


def test_actual_parent_and_fresh_confirmation_metadata_without_cf_states():
    assert c.parent_gate()['status']=='PASS';cfg=c.e.settings('confirmation');assert cfg['train_seeds']==list(range(380300000,380300005));assert cfg['test_seeds']==list(range(380500000,380500256));assert len(c.e.ARMS)*len(cfg['train_seeds'])==35;assert 35*491520==17203200 and 35*10240==358400 and 35*7680==268800 and 35*5*64==11200 and (35*256+256)*60==552960



def test_strict_zero_ci_does_not_claim_superiority_from_roundoff():
    point=passing_points();effects=np.tile(point,(100,1));effects[:,4:6]=1e-14;bounds=c.confidence_bounds(point,effects)
    assert not bounds['gain_vs_mappo']['passed'] and not bounds['gain_vs_full_mappo']['passed']
    assert bounds['gain_vs_full_mappo']['threshold']==0 and bounds['gain_vs_full_mappo']['numeric_margin']==1e-12
