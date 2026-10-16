"""Frozen decoder diagnostic correctness; no research training or smoke run."""
import copy,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_marginal_matching_decoder_diagnostic as d


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


def test_registry_grid_and_all_diagnostic_inputs_are_new():
    c=d.settings();report=d.seed_audit();assert report['status']=='PASS' and report['raw_unique']==67
    assert report['derived_physical_unique']==66 and len(c['seeds'])==64 and c['training_decisions']==0
    assert len(d.jobspecs(c))==43 and len(d.jobspecs(c))*64==2752 and 2752*60==165120
    assert c['max_seconds']==1800 and not c['scientific_test_opened'] and not c['teacher_assistance']


@pytest.mark.parametrize('arm',d.x.CORE_ARMS)
def test_exact_decoder_full_support_and_probability_argmax(arm):
    c=d.x.settings('development');model=d.x.make_model(arm,4,c,200400000);env=d.x.ResourceEnv(210200000,4,arm);obs=env.reset()
    # Dense artificial support checks all21; normal physical states below cover masks.
    dense={k:v.copy() for k,v in obs.items()};dense['pair_mask'][:]=True
    for view in (dense,obs):
        standard,exact,r=d.decoder_choices(model,view);logs=np.array(json.loads(r['physical_log_probabilities']));tokens=np.array(json.loads(r['matching_tokens']))
        assert abs(np.exp(logs).sum()-1)<2e-5 and r['exact_index']==int(logs.argmax())
        assert np.array_equal(exact[:-1],tokens[r['exact_index']]) and np.array_equal(standard,model.select(view))
        assert r['exact_probability']+1e-10>=r['standard_probability']
        assert r['support']==(21 if view is dense else len(tokens))
    env.resolve(exact)


def checkpoint_fixture(tmp_path,arm):
    c=d.x.settings('development');t=c['train_seeds'][0];source=tmp_path/'source';source.mkdir();out=tmp_path/'out';out.mkdir()
    model=d.x.make_model(arm,4,c,t);obs=d.x.ResourceEnv(210200000,4,arm).reset();path=source/'zero.pt'
    meta=d.x.save_model(model,path,4,c,t,0,obs);meta.update(path='zero.pt',steps=0)
    cfg=dict(d.settings(),seeds=[210200000],train_seeds=[t])
    return cfg,source,c,meta,arm,t,'standard',out,time.monotonic()


@pytest.mark.parametrize('arm',d.x.arms(4))
def test_frozen_episode_physics_and_decoder_log_replay(tmp_path,arm):
    job=checkpoint_fixture(tmp_path,arm);result=d.evaluate_job(tuple(str(v) if isinstance(v,Path) else v for v in job))
    assert result['initial_parameter_sha256']==result['final_parameter_sha256']
    assert result['audit']['physical_ticks_replayed']==60 and result['rows'][0]['invalid_assignments']==0
    directory=job[-2]/'jobs'/arm/'standard'/str(job[5])
    assert d.audit_decoder_log(directory,arm,'standard',result['rows'])==(48 if arm in d.x.CORE_ARMS else 0)
    if arm in d.x.CORE_ARMS:
        exact=list(job);exact[6]='exact';result2=d.evaluate_job(tuple(str(v) if isinstance(v,Path) else v for v in exact))
        assert result2['initial_parameter_sha256']==result2['final_parameter_sha256']
        assert d.audit_decoder_log(job[-2]/'jobs'/arm/'exact'/str(job[5]),arm,'exact',result2['rows'])==48


def test_spawn_and_serial_frozen_outputs_match(tmp_path):
    job=checkpoint_fixture(tmp_path,'marginal_matching_ppo');job=tuple(str(v) if isinstance(v,Path) else v for v in job)
    serial=d.evaluate_job(job);right=tmp_path/'parallel';right.mkdir();parallel=(*job[:-2],str(right),time.monotonic())
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:other=pool.submit(d.evaluate_job,parallel).result(timeout=60)
    assert serial['rows']==other['rows'] and serial['initial_parameter_sha256']==other['initial_parameter_sha256']


def test_source_rejects_unfinished_or_passed_screen(tmp_path):
    d.x.write_json(tmp_path/'manifest.json',dict(protocol=d.x.VERSION,status='RUNNING',profile='development',audit_status='PASS',test_opened=False))
    d.x.write_json(tmp_path/'resolved_config.json',d.x.settings('development'));d.x.write_json(tmp_path/'summary.json',dict(development_screen_pass=False))
    with pytest.raises(ValueError,match='completed audited failed-screen'):d.validate_source(tmp_path)
    m=json.loads((tmp_path/'manifest.json').read_text());m['status']='COMPLETED';d.x.write_json(tmp_path/'manifest.json',m)
    d.x.write_json(tmp_path/'summary.json',dict(development_screen_pass=True))
    with pytest.raises(ValueError,match='completed audited failed-screen'):d.validate_source(tmp_path)


def test_duplicate_decoder_record_is_rejected(tmp_path):
    job=checkpoint_fixture(tmp_path,'marginal_matching_ppo');result=d.evaluate_job(tuple(str(v) if isinstance(v,Path) else v for v in job))
    directory=job[-2]/'jobs'/job[4]/'standard'/str(job[5]);path=directory/'decoder_choices.csv';lines=path.read_text().splitlines()
    with path.open('a') as stream:stream.write(lines[-1]+'\n')
    with pytest.raises(RuntimeError,match='duplicate decoder'):d.audit_decoder_log(directory,job[4],'standard',result['rows'])
