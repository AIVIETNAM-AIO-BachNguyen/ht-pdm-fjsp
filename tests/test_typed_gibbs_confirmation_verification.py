import json,sys,time,shutil
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import typed_gibbs_confirmation_reference as ref
import verify_typed_gibbs_confirmation as verifier
from ht_pdm_fjsp import technician_typed_gibbs_confirmation as source


def test_explicit_advanced_cartesian_indexing_against_literal_nested_average():
    v=np.arange(7*2*3*4,dtype=float).reshape(7,2,3,4)+11;ti=np.array([[1,1],[1,0],[0,0]]);si=np.array([[2,2,0],[0,2,1],[1,0,1]])
    literal=np.array([[[np.mean([v[m,t,s,f] for t in tr for s in sr]) for f in range(4)] for m in range(7)] for tr,sr in zip(ti,si)])
    np.testing.assert_allclose(ref.cartesian_means(v,ti,si),literal,atol=1e-12,rtol=0);np.testing.assert_allclose(ref.cartesian_means(v,ti,si),source.crossed_means(v,ti,si),atol=1e-12,rtol=0)


def test_ratios_are_of_means_and_paired_same_method_has_zero_effect():
    v=np.full((7,1,2,4),1.);v[0,0,:,0]=[1,100];v[1,0,:,0]=[1,200];p=ref.effects(v.mean((1,2))[None]);assert p[0,0]==pytest.approx(1-101/201) and abs(p[0,0]-.25)>.1;v[1]=v[0];assert ref.effects(v.mean((1,2))[None])[0,0]==0


@pytest.mark.parametrize('index',range(12))
def test_independent_direction_thresholds_and_numerical_zero_ties(index):
    p=np.array([.01,.005,.005,.005,.005,.005,0.,0.,0.,0.,0.,0.]);fx=np.tile(p,(97,1));bad=[.0049,-.0101,-.0101,-.0101,1e-14,1e-14,-.0101,.0251,.2501,-.0101,.0251,.2501];fx[:,index]=bad[index];actual=ref.bounds(p,fx);assert not actual[ref.NAMES[index]]['passed'] and all(r['passed'] for k,r in actual.items() if k!=ref.NAMES[index]);assert actual==source.confidence_bounds(p,fx)


def test_independent_seed_namespace_draw_order_and_chunk_boundary():
    h=ref.helpers();v=np.arange(7*5*256*4,dtype=float).reshape(7,5,256,4)+10;seed=h.seed(127,f'maintenance_balanced_algorithm_benchmark_v1:{ref.FAMILY}:bootstrap');actual,ti,si=source.bootstrap(v,seed,draws=259,progress=False);independent=[]
    for start,t,s in ref.regenerate_indices(h,127,5,256,draws=259):
        np.testing.assert_array_equal(ti[start:start+len(t)],t);np.testing.assert_array_equal(si[start:start+len(t)],s);independent.extend(ref.effects(ref.cartesian_means(v,t,s)))
    np.testing.assert_allclose(actual,independent,atol=1e-10,rtol=0);assert si.max()==255


@pytest.fixture(scope='module')
def engineering(tmp_path_factory):
    torch.set_num_threads(1);root=tmp_path_factory.mktemp('independent_cf_engineering');c=source.e.settings('engineering');c.update(env_steps=96,checkpoints=[96],test_seeds=[380700001]);t=c['train_seeds'][0]
    with pytest.MonkeyPatch.context() as patch:
        original=source.e.ResourceEnv
        def only_engineering(seed,*args,**kwargs):
            assert seed in (380700000,380700001),'scientific physical panel opened by tests'
            return original(seed,*args,**kwargs)
        patch.setattr(source.e,'ResourceEnv',only_engineering);results=[source.e.fit_job((c,a,t,str(root),time.monotonic())) for a in ref.ARMS];dev=[row for r in results for row in r['development']]
        for name in ('training_progress','training_episodes'):source.e.old.merge_journals(root,name,results)
        source.e.write_csv(root/'development_episodes.csv',dev);source.e.write_json(root/'training_audit.json',[r['audit'] for r in results]);source.e.write_json(root/'parameter_counts.json',{a:sum(p.numel() for p in source.e.make_model(a,c,t).parameters()) for a in ref.ARMS});models={a:source.e.make_model(a,c,t) for a in ref.ARMS};source.e.write_json(root/'initial_parameter_hashes.json',{str(t):{a:source.e.learner.parameter_digest(m) for a,m in models.items()}});source.e.write_json(root/'initial_critic_tensor_hashes.json',{str(t):{a:source.e.old.raw_tensor_digest(m.critic.net if isinstance(m.critic,source.e.old.reference.ValueUnits) else m.critic) for a,m in models.items()}})
        audit=source.e.audit_training(root,c,results,dev,lambda:None);source.e.write_json(root/'training_grid_audit.json',audit);choices={f"{r['method']}:{t}":r['fixed'] for r in results};source.freeze_models(root,c,choices);latency=source.e.isolated_latency(root,c,choices);source.e.write_json(root/'latency.json',latency);source.open_test_gate(root,c);tested=[source.test_job((c,a,t,str(root),time.monotonic())) for a in ref.ARMS]+[source.test_job((c,'deadline',-1,str(root),time.monotonic()))]
    for name in ('coordination','decisions'):source.e.old.merge_journals(root,name,tested)
    source.e.write_csv(root/'selected_test_episodes.csv',[row for item in tested for row in item['rows']]);return root,c,choices


def test_all7_manual_tensors_full_episode_budgets_and_freeze(engineering):
    root,c,choices=engineering;h=ref.helpers();r=verifier.budgets(root,c,choices,h);assert r==dict(models=7,training_decisions=672,training_episodes=14,adam_updates=14,paired_training_tapes=2,development_nodes=14);assert verifier.freeze(root,c,choices,h)['model_count']==7;assert verifier.timing(root,c,choices,h)['samples']==144


def test_streamed_joins_full_primitive_replay_all7_numpy_actors_and_values(engineering):
    root,c,choices=engineering;rows,r=verifier.replay(root,c,choices,ref.helpers());assert len(rows)==8 and r['physical_transitions']==480 and r['policy_decisions']==384 and len(r['neural_checks'])==7
    assert all(x['states']==48 and x['value_max_error']<.0023041 for x in r['neural_checks'].values())


def test_model_freeze_mutation_is_rejected_independently(engineering,tmp_path):
    root,c,choices=engineering;copy=tmp_path/'mutated';shutil.copytree(root,copy);first=next(iter(choices.values()));p=copy/first['path'];p.write_bytes(p.read_bytes()+b'X')
    with pytest.raises(AssertionError,match='immutable'):verifier.freeze(copy,c,choices,ref.helpers())


def test_synthetic_full_array_statistical_audit_and_tamper_rejection(tmp_path):
    # Correct full dimensions but scalar fixtures ONLY: no simulator or physical seeds.
    cfg=dict(train_seeds=list(range(5)),test_seeds=list(range(256)),analysis_seed=127);v=np.zeros((7,5,256,4));base=np.arange(5*256).reshape(5,256)/1000
    for m in range(7):v[m,:,:,0]=100+base+m;v[m,:,:,1]=.65+m*.0001;v[m,:,:,2]=m*.0001;v[m,:,:,3]=.1+m*.001
    rows=[dict(method=a,train_seed=t,seed=s,**dict(zip(ref.FIELDS,v[m,t,s].tolist()))) for m,a in enumerate(ref.ARMS) for t in range(5) for s in range(256)]+[dict(method='deadline',train_seed=-1,seed=s,**dict(zip(ref.FIELDS,[110.,.65,0.,.1]))) for s in range(256)]
    np.save(tmp_path/'test_metric_array.npy',v);seed=ref.helpers().seed(127,f'maintenance_balanced_algorithm_benchmark_v1:{ref.FAMILY}:bootstrap');fx,ti,si=source.bootstrap(v,seed,draws=259,progress=False);np.savez_compressed(tmp_path/'bootstrap_indices.npz',train_indices=ti,seed_indices=si);np.savez_compressed(tmp_path/'bootstrap_endpoints.npz',endpoints=fx);points=source.endpoint_effects(v.mean((1,2))[None])[0];(tmp_path/'confidence_endpoints.json').write_text(json.dumps(source.confidence_bounds(points,fx)))
    r=verifier.statistical_audit(tmp_path,cfg,rows,ref.helpers(),draws=259);assert r['bootstrap']['all_endpoint_values_checked']==259*12 and r['bootstrap']['max_absolute_error']<1e-10
    si[0,0]=(int(si[0,0])+1)%256;np.savez_compressed(tmp_path/'bootstrap_indices.npz',train_indices=ti,seed_indices=si)
    with pytest.raises(AssertionError):verifier.statistical_audit(tmp_path,cfg,rows,ref.helpers(),draws=259)


def test_registered_confirmation_grid_cardinalities():
    assert len(verifier.TRAINS)==5 and len(verifier.DEV)==64 and len(verifier.TEST)==256;assert 7*5*491520==17203200 and 7*5*10240==358400 and 7*5*7680==268800 and 7*5*5*64==11200 and (7*5*256+256)*60==552960 and (7*5*256+256)*48==442368
