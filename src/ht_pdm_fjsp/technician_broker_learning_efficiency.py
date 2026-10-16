"""Fresh random-init bounded learning-efficiency study of the frozen broker learner."""
import argparse,json,math,multiprocessing as mp,shutil,subprocess,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_broker_marginal_experiment as x

ROOT=Path(__file__).resolve().parents[2];VERSION='technician_broker_learning_efficiency_v1'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md';REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
STEPS=(0,15360,30720,61440,92160,122880,184320,245760)
SOURCE=Path('/home/bachng/ht-pdm-fjsp/artifacts/technician_broker_marginal_ppo_v1_development_20261016T142958102925918Z')


def check(ok,label):
    if not ok:raise AssertionError(label)
def read(p):return json.loads(p.read_text())


def settings(profile='development'):
    raw=read(REGISTRY)['panels'][profile];c=x.settings('engineering' if profile=='engineering' else 'development')
    c.update(profile=profile,train_seeds=raw['train_seeds'],development_seeds=list(range(raw['development_start'],raw['development_start']+raw['development_count'])),test_seeds=list(range(raw.get('test_start',0),raw.get('test_start',0)+raw.get('test_count',0))),analysis_seed=raw['analysis_seed'],scales=[4],outer_protocol=VERSION,source_role_namespace=x.VERSION)
    if profile!='engineering':c.update(env_steps=STEPS[-1],checkpoints=list(STEPS[1:]),confidence=1-(.05/20)/14)
    return c


def seed_audit():
    history=set();registries={};own_hash=x.sha(REGISTRY)
    for p in [*(ROOT/'configs').glob('*seed_registry.json'),*Path('/home/bachng/ht-pdm-fjsp-worktrees').glob('*/configs/*seed_registry.json')]:
        digest=x.sha(p)
        if digest==own_hash or digest in registries:continue
        registries[digest]=str(p);history|=x.registered_integers(read(p))
    groups=[];derived=[]
    for profile in ('development','confirmation','engineering'):
        c=settings(profile);groups.extend([set(c['train_seeds']),set(c['development_seeds']),set(c['test_seeds']),{c['analysis_seed']}]);derived+=x.derived_streams(c,x.VERSION);derived.append(x.learner.role_seed(c['analysis_seed'],VERSION+':bootstrap'))
    raw=set.union(*groups);known=set()
    for module in (x,x.reference,x.old_rsc,x.clearing_reference,x.raw_reference,x.budget_reference):
        for profile in ('development','confirmation','engineering'):known.update(x.derived_streams(module.settings(profile),module.VERSION))
    # All registered numeric historical raw seeds are additionally hashed into
    # this study's actual physical namespace; conservative across diagnostic families.
    known.update(x.role_seed(s,f'{x.VERSION}:N4:physical') for s in history)
    check(not any(a&b for i,a in enumerate(groups) for b in groups[i+1:]) and not raw&history and len(derived)==len(set(derived)) and not set(derived)&(raw|history|known),'fresh disjoint raw/actual-role seeds')
    return dict(status='PASS',raw_unique=len(raw),derived_unique=len(derived),historical_registries=registries,known_historical_derived_checked=len(known),actual_learning_namespace=x.VERSION,outer_bootstrap_namespace=VERSION,confirmation_states_constructed=False,limitation='Explicit six learning-family stream sets plus conservative historical raw-to-current-physical hashes; no claim of exhaustively reconstructing every ancient external RNG namespace.')


def auc_rows(dev,c):
    lookup={};expected={(a,t,s,step) for a in x.ARMS for t in c['train_seeds'] for s in c['development_seeds'] for step in STEPS}
    for r in dev:
        k=(r['method'],r['train_seed'],r['seed'],r['checkpoint_steps']);check(k not in lookup and k in expected,'unique registered AUC node');lookup[k]=r['normalized_cost']
    check(set(lookup)==expected,'complete AUC grid');rows=[]
    for t in c['train_seeds']:
        for s in c['development_seeds']:check(lookup[x.CORE_ARMS[0],t,s,0]==lookup[x.CORE_ARMS[1],t,s,0],'same physical initial candidate/control cost')
    for a in x.ARMS:
        for t in c['train_seeds']:
            for s in c['development_seeds']:
                values=[lookup[a,t,s,step] for step in STEPS];integral=math.fsum(.5*(values[i]+values[i+1])*(STEPS[i+1]-STEPS[i]) for i in range(len(STEPS)-1))
                rows.append(dict(method=a,train_seed=t,seed=s,auc_cost_per_machine=integral/STEPS[-1],endpoint_cost_per_machine=values[-1],nodes=len(STEPS)))
    return rows


def final_choice(out,a,t,c,dev):
    step=c['env_steps'];path=Path('N4')/a/str(t)/'checkpoints'/f'step_{step:09d}.pt';model=x.load_model(out/path,a,c,t,step)
    meta=dict(path=str(path),sha256=x.sha(out/path),parameter_sha256=x.learner.parameter_digest(model),steps=step,development_cost=float(np.mean([r['extended_cost'] for r in dev if r['method']==a and r['train_seed']==t and r['checkpoint_steps']==step])))
    if a==x.CORE_ARMS[0]:
        p=path.parent/f'observations_{step:09d}.npz';meta['observations']=dict(path=str(p),sha256=x.sha(out/p),states=48,seed=c['development_seeds'][0])
    return meta


def summarize(rows,auc,latency,c):
    names=(*x.ARMS,'deadline',*x.FROZEN);means={a:{k:float(np.mean([r[k] for r in rows if r['method']==a])) for k in ('normalized_cost','availability','normalized_pending','normalized_failures')} for a in names};candidate=x.CORE_ARMS[0];control=x.CORE_ARMS[1];p=means[candidate]
    auc_means={a:float(np.mean([r['auc_cost_per_machine'] for r in auc if r['method']==a])) for a in x.ARMS};points=dict(auc_gain_vs_broker_proposal_ppo=1-auc_means[candidate]/auc_means[control]);checks={k:v>=.005 for k,v in points.items()}
    for a in x.ARMS[1:]:
        name='final_gain_vs_'+a;points[name]=1-p['normalized_cost']/means[a]['normalized_cost'];checks[name]=points[name]>0 if a in ('mappo','full_mappo') else points[name]>=-.01
    for a in ('ppo','normalized_ppo'):
        for metric in ('availability','normalized_pending','normalized_failures'):
            name=metric+'_delta_vs_'+a;delta=p[metric]-means[a][metric];points[name]=delta;checks[name]=delta>=-.01 if metric=='availability' else delta<=.025 if metric=='normalized_pending' else delta<=.25
    check(len(checks)==14,'14endpoint family');per={}
    for t in c['train_seeds']:
        values={a:float(np.mean([r['auc_cost_per_machine'] for r in auc if r['method']==a and r['train_seed']==t])) for a in (candidate,control)};per[str(t)]=dict(candidate_auc=values[candidate],control_auc=values[control],gain=1-values[candidate]/values[control])
    invalid=sum(r['invalid_assignments'] for r in rows);extra=dict(every_replica_auc_gain_positive=all(r['gain']>0 for r in per.values()),zero_invalid=invalid==0,isolated_p95_at_most_50ms=latency['p95_ms']<=50)
    return dict(classification='FRESH_LEARNING_EFFICIENCY_DEV_ONLY',final_means=means,auc_means=auc_means,endpoints=points,checks=checks,extra_checks=extra,per_replica_auc=per,primary_fixed_steps=STEPS[-1],latency=latency,invalid_assignments=invalid,development_screen_pass=all(checks.values()) and all(extra.values()),goal_pass=False,test_opened=False,teacher_assistance=False,limitation='New bounded learning-speed question. Prior983040step final-quality failure unchanged. DEV points not CI; no scientific/large-N authorization without independent audit and separate confirmation implementation.')


def run(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();c=settings();torch.set_num_threads(1)
    manifest=dict(protocol=VERSION,source_learning_protocol=x.VERSION,source_learning_commit='dcef1f554bf54172aed7f61cfdefe8d8e16b70d9',source_manifest_sha256=x.sha(SOURCE/'manifest.json'),status='RUNNING',goal_pass=False,test_opened=False,teacher_assistance=False,git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())))
    x.write_json(out/'manifest.json',manifest)
    def guard():check(time.monotonic()-start<c['max_seconds'],'12hour study hardstop')
    def frozen_source():
        check(x.sha(SOURCE/'manifest.json')==manifest['source_manifest_sha256'],'learning source manifest unchanged')
        for name,digest in read(SOURCE/'manifest.json')['source_seal'].items():
            if name.startswith('source_snapshot/'):check(x.sha(Path(x.__file__).parent/Path(name).name)==digest,'frozen source learner '+name)
    try:
        check(not manifest['git']['dirty'] and c['test_seeds']==[],'clean DEV-only study');check(not any('from multiprocessing.spawn import spawn_main' in line for line in subprocess.check_output(['ps','-eo','args'],text=True).splitlines()),'no other training workers');frozen_source();x.write_json(out/'resolved_config.json',c);x.write_json(out/'seed_audit.json',seed_audit());x.write_json(out/'physics.json',asdict(x.scaled_config(4)));shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'seed_registry.json');(out/'source_snapshot').mkdir()
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,out/'source_snapshot'/p.name)
        manifest['source_seal']={str(p.relative_to(out)):x.sha(p) for p in (out/'source_snapshot').iterdir()};manifest['source_seal'].update({name:x.sha(out/name) for name in ('protocol.md','seed_registry.json','resolved_config.json','physics.json')});x.write_json(out/'manifest.json',manifest)
        counts={a:sum(p.numel() for p in x.make_model(a,c,c['train_seeds'][0]).parameters()) for a in x.ARMS};check(all(counts[a]==19974 for a in x.CORE_ARMS),'matched core capacities')
        initial={str(t):{a:x.learner.parameter_digest(x.make_model(a,c,t)) for a in x.ARMS} for t in c['train_seeds']};raw={str(t):{a:x.raw_tensor_digest(x.make_model(a,c,t)) for a in x.ARMS} for t in c['train_seeds']};critics={}
        for t in c['train_seeds']:
            critics[str(t)]={}
            for a in x.ARMS:
                model=x.make_model(a,c,t);critic=model.critic.net if isinstance(model.critic,x.reference.ValueUnits) else model.critic;critics[str(t)][a]=x.raw_tensor_digest(critic)
        check(all(len({r[a] for a in x.CORE_ARMS})==1 for r in initial.values()) and all(r['ppo']==r['normalized_ppo'] for r in raw.values()) and all(len(set(r.values()))==1 for r in critics.values()),'fresh paired actors/critics')
        for name,value in (('parameter_counts',counts),('initial_parameter_hashes',initial),('initial_raw_tensor_hashes',raw),('initial_critic_tensor_hashes',critics)):x.write_json(out/f'{name}.json',value)
        results=[];jobs=[(c,a,t,str(out),start) for a in x.ARMS for t in c['train_seeds']]
        with ProcessPoolExecutor(max_workers=c['training_processes'],mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(x.fit_job,job) for job in jobs]
            try:
                for future in tqdm(as_completed(futures),total=24,desc='24FULL fresh learning-efficiency fits'):
                    guard();results.append(future.result());x.write_json(out/'fit_results.partial.json',[{k:v for k,v in r.items() if k!='development'} for r in results])
            except Exception as exc:
                x.write_json(out/'abort.json',dict(error=str(exc)))
                for future in futures:future.cancel()
                raise
        results.sort(key=lambda r:(x.ARMS.index(r['method']),r['train_seed']));dev=[row for result in results for row in result['development']];ancillary={f"{r['method']}:{r['train_seed']}":r['choice'] for r in results};fixed={f'{a}:{t}':final_choice(out,a,t,c,dev) for a in x.ARMS for t in c['train_seeds']}
        x.write_json(out/'ancillary_checkpoint_choices.json',ancillary);x.write_json(out/'checkpoint_choices.json',fixed);x.write_json(out/'training_audit.json',[r['audit'] for r in results]);x.write_csv(out/'development_episodes.csv',dev)
        for name in ('training_progress','training_episodes'):x.merge_journals(out,name,results)
        training=x.audit_training(out,c,results,ancillary,dev,guard);auc=auc_rows(dev,c);x.write_csv(out/'auc_episodes.csv',auc)
        journals={name:x.Journal(out/f'{name}.csv') for name in ('coordination','decisions')};rows=[]
        try:
            for a in (*x.ARMS,*x.FROZEN):
                for t in c['train_seeds']:
                    weight_arm=x.CORE_ARMS[0] if a in x.FROZEN else a;choice=fixed[f'{weight_arm}:{t}'];model=x.load_model(out/choice['path'],weight_arm,c,t,choice['steps']);before=x.learner.parameter_digest(model)
                    for s in tqdm(c['development_seeds'],desc=f'Fixed-final {a}/{t}',leave=False):
                        guard();row=x.evaluate(a,t,s,model,journals['coordination'],journals['decisions']);rows.append(row)
                        if a in x.ARMS:
                            expected=next(r for r in dev if r['method']==a and r['train_seed']==t and r['seed']==s and r['checkpoint_steps']==STEPS[-1]);check(all(abs(row[k]-expected[k])<1e-8 for k in ('extended_cost','availability','terminal_pending','policy_failures')),'fixed final replay')
                    check(x.learner.parameter_digest(model)==before,'evaluation frozen weights')
            for s in tqdm(c['development_seeds'],desc='Deadline reference'):guard();rows.append(x.evaluate('deadline',-1,s,coordination=journals['coordination'],decisions=journals['decisions']))
        finally:
            for journal in journals.values():journal.close()
        x.write_csv(out/'selected_episodes.csv',rows);selected=x.audit_selected(out,c,rows,guard);latency=x.isolated_latency(out,c,fixed,guard);x.write_json(out/'latency.json',latency);summary=summarize(rows,auc,latency,c);x.write_json(out/'summary.json',summary);x.write_json(out/'audit.json',dict(status='PASS',training=training,final=selected,auc_rows=len(auc),primary_fixed_endpoint=True,ancillary_selection_used_for_primary=False))
        frozen_source()
        for name,digest in manifest['source_seal'].items():check(x.sha(out/name)==digest and (not name.startswith('source_snapshot/') or x.sha(Path(__file__).parent/Path(name).name)==digest),'study source unchanged')
        check(x.sha(PLAN)==manifest['source_seal']['protocol.md'] and x.sha(REGISTRY)==manifest['source_seal']['seed_registry.json'],'study protocol unchanged');manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,development_screen_pass=summary['development_screen_pass'],output_bindings={str(p.relative_to(out)):dict(sha256=x.sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'});x.write_json(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),status='PASS',development_screen_pass=summary['development_screen_pass'])));return summary
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);x.write_json(out/'manifest.json',manifest);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);args=p.parse_args();run(args.output_dir)
