"""Frozen N4 typed-Gibbs confirmation: fresh scratch fits, seal, TEST once, crossed CI."""
import argparse,csv,json,multiprocessing as mp,platform,shutil,subprocess,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_typed_gibbs_experiment as e

ROOT=Path(__file__).resolve().parents[2]
VERSION='technician_typed_gibbs_confirmation_v1'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
ARTIFACTS=Path('/home/bachng/ht-pdm-fjsp/artifacts')
PARENT=ARTIFACTS/'technician_typed_gibbs_ppo_v1_development_20261017T115852309901Z'
VERIFICATION=ARTIFACTS/'technician_typed_gibbs_verification_v1_20261017T121421207679Z'
ALPHA=.05/(7*8)/12
FIELDS=('normalized_cost','availability','normalized_pending','normalized_failures')
COST_ENDPOINTS=tuple('gain_vs_'+a for a in e.ARMS[1:])
HEALTH_ENDPOINTS=tuple(f'{metric}_delta_vs_{a}' for a in e.REFERENCES for metric in FIELDS[1:])
ENDPOINTS=COST_ENDPOINTS+HEALTH_ENDPOINTS


def read(path):return json.loads(path.read_text())
def check(ok,label):
    if not ok:raise AssertionError(label)
def utc():return datetime.now(timezone.utc).isoformat()


def parent_gate():
    source=read(PARENT/'manifest.json');v=read(VERIFICATION/'verification.json');vm=read(VERIFICATION/'manifest.json');summary=read(PARENT/'summary.json')
    check(source['status']=='COMPLETED' and source['audit_status']=='PASS' and source['git']['commit']=='51ebbeca434e83077ff4533131af49fd56fcd5cb' and not source['test_opened'],'original committed DEV')
    check(vm['status']=='COMPLETED' and v['status']=='PASS' and v['source']==str(PARENT) and v['source_manifest_sha256']==e.sha(PARENT/'manifest.json') and v['original_development_pass'] and summary['development_screen_pass'] and all(summary['checks'].values()) and all(summary['extra_checks'].values()),'independently verified ALLPASS DEV')
    for name,meta in vm['outputs'].items():check(e.sha(VERIFICATION/name)==meta['sha256'],'parent verification binding')
    for name,digest in source['source_seal'].items():
        check(e.sha(PARENT/name)==digest,'parent source seal')
        if name.startswith('source_snapshot/'):check(e.sha(ROOT/'src/ht_pdm_fjsp'/Path(name).name)==digest,'frozen algorithm source unchanged')
    check(e.sha(e.REGISTRY)==e.sha(PARENT/'seed_registry.json') and e.sha(e.PLAN)==e.sha(PARENT/'protocol.md'),'original prospective family registry/protocol')
    return dict(status='PASS',source=str(PARENT),source_manifest_sha256=e.sha(PARENT/'manifest.json'),verification=str(VERIFICATION),verification_manifest_sha256=e.sha(VERIFICATION/'manifest.json'),frozen_algorithm_commit=source['git']['commit'])


def freeze_models(out,c,choices):
    check(not (out/'models_frozen.json').exists() and not (out/'checkpoint_choices.json').exists(),'model freeze overwrite refused')
    expected={f'{a}:{t}' for a in e.ARMS for t in c['train_seeds']};check(set(choices)==expected,'complete fixed-final model grid before TEST')
    for a in e.ARMS:
        for t in c['train_seeds']:
            meta=choices[f'{a}:{t}'];check(meta['steps']==c['env_steps'] and meta['path']==f"N4/{a}/{t}/checkpoints/step_{c['env_steps']:09d}.pt" and e.sha(out/meta['path'])==meta['sha256'],'fixed-final file binding')
            model=e.load_model(out/meta['path'],a,c,t,c['env_steps']);check(e.learner.parameter_digest(model)==meta['parameter_sha256'],'fixed-final parameter binding')
    e.write_json(out/'checkpoint_choices.json',choices)
    frozen=dict(frozen_utc=utc(),model_count=len(choices),fixed_steps=c['env_steps'],train_seeds=c['train_seeds'],choices_sha256=e.sha(out/'checkpoint_choices.json'),models={key:{field:meta[field] for field in ('path','sha256','parameter_sha256','steps')} for key,meta in choices.items()})
    e.write_json(out/'models_frozen.json',frozen);return frozen


def open_test_gate(out,c):
    check(not (out/'test_gate.json').exists(),'TEST reopening refused')
    frozen=read(out/'models_frozen.json');check(frozen['model_count']==len(e.ARMS)*len(c['train_seeds']) and frozen['choices_sha256']==e.sha(out/'checkpoint_choices.json'),'complete immutable freeze before TEST')
    check((out/'training_grid_audit.json').exists() and read(out/'training_grid_audit.json')['status']=='PASS','audited full learning grid before TEST')
    audit=read(out/'training_grid_audit.json');fits=len(e.ARMS)*len(c['train_seeds']);check(audit['models']==fits and audit['training_decisions']==fits*c['env_steps'] and audit['training_episodes']==fits*c['env_steps']//48 and audit['development_nodes']==fits*(1+len(c['checkpoints']))*len(c['development_seeds']) and audit['paired_training_tapes']==len(c['train_seeds'])*c['env_steps']//48,'complete audited training and CFDEV counts before TEST')
    gate=dict(opened_utc=utc(),models_frozen_sha256=e.sha(out/'models_frozen.json'),choices_sha256=e.sha(out/'checkpoint_choices.json'),training_grid_audit_sha256=e.sha(out/'training_grid_audit.json'),test_seeds=c['test_seeds'],test_action_updates=0)
    e.write_json(out/'test_gate.json',gate);check_test_gate(out,c);return gate


def check_test_gate(out,c):
    gate=read(out/'test_gate.json');frozen=read(out/'models_frozen.json');check(gate['models_frozen_sha256']==e.sha(out/'models_frozen.json') and gate['choices_sha256']==e.sha(out/'checkpoint_choices.json')==frozen['choices_sha256'] and gate['training_grid_audit_sha256']==e.sha(out/'training_grid_audit.json') and gate['test_seeds']==c['test_seeds'] and gate['test_action_updates']==0,'TEST gate/seal binding')
    expected={f'{a}:{t}' for a in e.ARMS for t in c['train_seeds']};check(set(frozen['models'])==expected and frozen['model_count']==len(expected) and frozen['fixed_steps']==c['env_steps'],'TEST complete model grid')
    for meta in frozen['models'].values():check(e.sha(out/meta['path'])==meta['sha256'],'TEST frozen model mutated')
    return frozen


def test_job(job):
    c,a,t,outname,start=job;torch.set_num_threads(1);out=Path(outname);frozen=check_test_gate(out,c);directory=out/'N4'/a/str(t);directory.mkdir(parents=True,exist_ok=True);journals={}
    for name in ('coordination','decisions'):
        check(not (directory/f'{name}.csv').exists(),'TEST journal overwrite refused');journals[name]=e.Journal(directory/f'{name}.csv')
    model=None if a=='deadline' else e.load_model(out/frozen['models'][f'{a}:{t}']['path'],a,c,t,c['env_steps']);rows=[]
    try:
        for s in tqdm(c['test_seeds'],desc=f'SEALED TEST {a}/{t}',leave=False):
            check(time.monotonic()-start<c['max_seconds'] and not (out/'abort.json').exists(),'TEST technical guard')
            rows.append(e.evaluate(a,t,s,model,journals['coordination'],journals['decisions']))
        return dict(method=a,train_seed=t,rows=rows)
    finally:
        for journal in journals.values():journal.close()


def crossed_means(values,train_indices,seed_indices):
    """Paired Cartesian row/column resampling, no mean of per-episode ratios."""
    values=np.asarray(values,dtype=np.float64);b,t=train_indices.shape;s=seed_indices.shape[1];check(values.shape[1:3]==(t,s),'crossed array dimensions')
    tc=np.zeros((b,t));sc=np.zeros((b,s));np.add.at(tc,(np.repeat(np.arange(b),t),train_indices.ravel()),1/t);np.add.at(sc,(np.repeat(np.arange(b),s),seed_indices.ravel()),1/s)
    temporary=np.einsum('bt,mtsf->bmsf',tc,values,optimize=True);return np.einsum('bmsf,bs->bmf',temporary,sc,optimize=True)


def endpoint_effects(means):
    check(means.ndim==3 and means.shape[1:]==(7,4) and np.isfinite(means).all() and (means[:,:,0]>0).all(),'finite positive-cost means')
    candidate=means[:,0];columns=[1-candidate[:,0]/means[:,i,0] for i in range(1,7)]
    columns += [candidate[:,metric]-means[:,e.ARMS.index(a),metric] for a in e.REFERENCES for metric in (1,2,3)]
    return np.stack(columns,-1)


def bootstrap(values,seed,draws=200000,chunk=256,progress=True):
    _,t,s,_=values.shape;check(0<t<=256 and 0<s<=256 and draws>0 and chunk>0,'stored bootstrap index contract');rng=np.random.default_rng(seed);check(type(rng.bit_generator).__name__=='PCG64','locked bootstrap PCG64');all_t=[];all_s=[];effects=[]
    for start in tqdm(range(0,draws,chunk),desc='Paired crossed bootstrap',disable=not progress):
        size=min(chunk,draws-start);ti=rng.integers(0,t,(size,t),dtype=np.int64);si=rng.integers(0,s,(size,s),dtype=np.int64);effects.append(endpoint_effects(crossed_means(values,ti,si)));all_t.append(ti.astype(np.uint8));all_s.append(si.astype(np.uint8))
    return np.concatenate(effects),np.concatenate(all_t),np.concatenate(all_s)


def confidence_bounds(point,effects,alpha=ALPHA):
    check(effects.ndim==2 and effects.shape[1]==12 and np.isfinite(effects).all(),'bootstrap endpoint matrix');q=np.quantile(effects,[alpha,1-alpha],axis=0,method='linear');result={}
    for i,name in enumerate(ENDPOINTS):
        lower=name.startswith('gain_') or name.startswith('availability_');bound=float(q[0 if lower else 1,i]);threshold=.005 if name==COST_ENDPOINTS[0] else 0. if name in ('gain_vs_mappo','gain_vs_full_mappo') else -.01 if lower else .025 if name.startswith('normalized_pending_') else .25
        strict=name in ('gain_vs_mappo','gain_vs_full_mappo');passed=bound>threshold if strict else bound>=threshold if lower else bound<=threshold
        result[name]=dict(point=float(point[i]),bound=bound,orientation='lower' if lower else 'upper',threshold=threshold,strict=strict,passed=bool(passed),one_sided_alpha=alpha)
    return result


def superiority_allowed(bounds,confirmation_pass):
    return bool(confirmation_pass and all(bounds['gain_vs_'+a]['bound']>=.01 for a in e.REFERENCES))


def analyze_test(out,c,rows,latency):
    expected={(a,t,s) for a in e.ARMS for t in c['train_seeds'] for s in c['test_seeds']}|{('deadline',-1,s) for s in c['test_seeds']};lookup={(r['method'],r['train_seed'],r['seed']):r for r in rows};check(len(rows)==len(expected) and set(lookup)==expected,'complete TEST panel')
    values=np.array([[[[lookup[a,t,s][field] for field in FIELDS] for s in c['test_seeds']] for t in c['train_seeds']] for a in e.ARMS],dtype=np.float64);check(np.isfinite(values).all(),'finite TEST data');np.save(out/'test_metric_array.npy',values)
    seed=e.learner.role_seed(c['analysis_seed'],e.VERSION+':bootstrap');effects,ti,si=bootstrap(values,seed);np.savez_compressed(out/'bootstrap_indices.npz',train_indices=ti,seed_indices=si);np.savez_compressed(out/'bootstrap_endpoints.npz',endpoints=effects);point=endpoint_effects(values.mean(axis=(1,2))[None])[0];bounds=confidence_bounds(point,effects);e.write_json(out/'confidence_endpoints.json',bounds)
    points=e.point_summary(rows,c,latency);checks={key:value['passed'] for key,value in bounds.items()};extra=points['extra_checks'];passed=all(checks.values()) and all(extra.values())
    return dict(classification='SEALED_TEST_CONFIRMATION_ONLY',means=points['means'],endpoints=dict(zip(ENDPOINTS,point.tolist())),confidence_endpoints=bounds,checks=checks,point_checks=points['checks'],extra_checks=extra,per_replica_main_gain=points['per_replica_main_gain'],latency=latency,confirmation_screen_pass=passed,independent_verification_pending=True,ppo_superiority_supported=superiority_allowed(bounds,passed),goal_pass=False,test_opened=True,teacher_assistance=False,bootstrap=dict(draws=200000,chunk=256,seed=seed,bit_generator='PCG64',paired=True,crossed=True,quantile_method='linear',sequential_round=7,family_alpha=.05/(7*8),one_sided_endpoint_alpha=ALPHA,training_replicas=len(c['train_seeds']),physical_seeds=len(c['test_seeds'])),limitation='Empirical paired crossed uncertainty with only five training replicas and sparse extreme tails; source verdict pending independent full audit. No scaling/generalization claim.')


def run(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(1);c=e.settings('confirmation');manifest=dict(protocol=VERSION,family=e.VERSION,status='RUNNING',phase='PREFLIGHT',test_opened=False,goal_pass=False,teacher_assistance=False,pretraining_steps=0,started_utc=utc(),git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())),runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu'))
    def save():e.write_json(out/'manifest.json',manifest)
    def guard():check(time.monotonic()-start<c['max_seconds'],'12hour confirmation hardstop')
    save()
    try:
        check(not manifest['git']['dirty'] and manifest['git']['commit']==subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=ROOT,text=True).strip(),'clean pushed code gate');check(not any('from multiprocessing.spawn import spawn_main' in r for r in subprocess.check_output(['ps','-eo','args'],text=True).splitlines()),'no active training workers');parents=parent_gate();e.write_json(out/'parent_bindings.json',parents);e.write_json(out/'seed_audit.json',e.seed_audit());check(c['train_seeds']==list(range(380300000,380300005)) and c['development_seeds']==list(range(380400000,380400064)) and c['test_seeds']==list(range(380500000,380500256)) and c['env_steps']==491520,'registered confirmation only')
        e.write_json(out/'resolved_config.json',c);e.write_json(out/'physics.json',asdict(e.scaled_config(4)));shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(e.REGISTRY,out/'seed_registry.json');(out/'source_snapshot').mkdir()
        for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,out/'source_snapshot'/path.name)
        seals={str(path.relative_to(out)):e.sha(path) for path in (out/'source_snapshot').iterdir()};seals.update({name:e.sha(out/name) for name in ('protocol.md','seed_registry.json','resolved_config.json','physics.json','parent_bindings.json')});manifest.update(phase='SCRATCH_LEARNING_AND_CFDEV',source_seal=seals);save()
        counts={a:sum(p.numel() for p in e.make_model(a,c,c['train_seeds'][0]).parameters()) for a in e.ARMS};check(all(counts[a]==18883 for a in e.CORE),'same core capacity');initial={};critics={}
        for t in c['train_seeds']:
            models={a:e.make_model(a,c,t) for a in e.ARMS};initial[str(t)]={a:e.learner.parameter_digest(m) for a,m in models.items()};critics[str(t)]={a:e.old.raw_tensor_digest(m.critic.net if isinstance(m.critic,e.old.reference.ValueUnits) else m.critic) for a,m in models.items()};check(len({initial[str(t)][a] for a in e.CORE})==1 and len(set(critics[str(t)].values()))==1,'matched private core and critic initialization')
        e.write_json(out/'parameter_counts.json',counts);e.write_json(out/'initial_parameter_hashes.json',initial);e.write_json(out/'initial_critic_tensor_hashes.json',critics);results=[]
        with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(e.fit_job,(c,a,t,str(out),start)) for a in e.ARMS for t in c['train_seeds']]
            try:
                for future in tqdm(as_completed(futures),total=35,desc='35FULL fresh confirmation fits'):guard();results.append(future.result());e.write_json(out/'fit_results.partial.json',[{k:v for k,v in r.items() if k!='development'} for r in results])
            except Exception as exc:
                e.write_json(out/'abort.json',dict(error=str(exc)))
                for f in futures:f.cancel()
                raise
        results.sort(key=lambda r:(e.ARMS.index(r['method']),r['train_seed']));dev=[row for r in results for row in r['development']];choices={f"{r['method']}:{r['train_seed']}":r['fixed'] for r in results};e.write_csv(out/'development_episodes.csv',dev);e.write_json(out/'training_audit.json',[r['audit'] for r in results]);e.write_json(out/'ancillary_checkpoint_choices.json',{f"{r['method']}:{r['train_seed']}":r['ancillary'] for r in results})
        for name in ('training_progress','training_episodes'):e.old.merge_journals(out,name,results)
        learning_audit=e.audit_training(out,c,results,dev,guard);e.write_json(out/'training_grid_audit.json',learning_audit);freeze_models(out,c,choices);latency=e.isolated_latency(out,c,choices);e.write_json(out/'latency.json',latency);open_test_gate(out,c);manifest.update(phase='SEALED_TEST_FIXED_MODELS',test_opened=True,model_seal_sha256=e.sha(out/'models_frozen.json'),test_gate_sha256=e.sha(out/'test_gate.json'));save();tested=[]
        with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(test_job,(c,a,t,str(out),start)) for a in e.ARMS for t in c['train_seeds']]+[pool.submit(test_job,(c,'deadline',-1,str(out),start))]
            try:
                for future in tqdm(as_completed(futures),total=36,desc='Frozen TEST method/replica jobs'):guard();tested.append(future.result());e.write_csv(out/'selected_test_episodes.partial.csv',[r for item in tested for r in item['rows']])
            except Exception as exc:
                e.write_json(out/'abort.json',dict(error=str(exc)))
                for f in futures:f.cancel()
                raise
        tested.sort(key=lambda r:(((*e.ARMS,'deadline').index(r['method'])),r['train_seed']));rows=[row for item in tested for row in item['rows']];e.write_csv(out/'selected_test_episodes.csv',rows)
        for name in ('coordination','decisions'):e.old.merge_journals(out,name,tested)
        check_test_gate(out,c);manifest.update(phase='PHYSICAL_AND_STATISTICAL_AUDIT');save();physical=e.audit_final(out,rows,dict(c,development_seeds=c['test_seeds']),guard);summary=analyze_test(out,c,rows,latency);e.write_json(out/'summary.json',summary);e.write_json(out/'audit.json',dict(status='PASS',learning=learning_audit,test_physics=physical,model_freeze_unchanged=True,independent_verification_pending=True))
        for name,digest in seals.items():check(e.sha(out/name)==digest and (not name.startswith('source_snapshot/') or e.sha(Path(__file__).parent/Path(name).name)==digest),'source seal unchanged')
        check(e.sha(PLAN)==seals['protocol.md'] and e.sha(e.REGISTRY)==seals['seed_registry.json'],'protocol registry unchanged');check_test_gate(out,c);manifest.update(status='COMPLETED',phase='PENDING_INDEPENDENT_CONFIRMATION_AUDIT',audit_status='PASS',confirmation_screen_pass=summary['confirmation_screen_pass'],elapsed_seconds=time.monotonic()-start,output_bindings={str(path.relative_to(out)):dict(sha256=e.sha(path),bytes=path.stat().st_size) for path in out.rglob('*') if path.is_file() and path.name!='manifest.json'});save();print(json.dumps(dict(output=str(out),status='COMPLETED',confirmation_screen_pass=summary['confirmation_screen_pass'],goal_pass=False)));return summary
    except Exception as exc:
        manifest.update(status='FAILED',phase='FAILED_PRESERVED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True);run(parser.parse_args().output_dir)
