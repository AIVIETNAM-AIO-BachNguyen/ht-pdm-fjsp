"""Prospective N4/K2 learned-continuation typed-Gibbs PPO FULL, all from scratch."""
import argparse,csv,hashlib,importlib.util,json,multiprocessing as mp,shutil,subprocess,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_broker_marginal_experiment as old
from ht_pdm_fjsp import technician_broker_learning_efficiency as efficiency
from ht_pdm_fjsp.technician_typed_gibbs_env import VERSION,ResourceEnv,scaled_config,value_unit,role_seed
from ht_pdm_fjsp.technician_typed_gibbs_policy import CORE,TypedGibbsPolicy,decode_tokens
from ht_pdm_fjsp.technician_broker_removal_learning import fit

ROOT=Path(__file__).resolve().parents[2];PLAN=ROOT/'docs'/f'{VERSION}_plan.md';REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
ARMS=(*CORE,'ppo','normalized_ppo','mappo','full_mappo');REFERENCES=('ppo','normalized_ppo')
sha=old.sha;write_json=old.write_json;write_csv=old.write_csv;Journal=old.Journal;learner=old.learner


def check(ok,label):
    if not ok:raise AssertionError(label)
def read(p):return json.loads(p.read_text())
def helpers():
    spec=importlib.util.spec_from_file_location('removal_primitive',ROOT/'scripts/verify_broker_marginal_development.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);return h


def settings(profile='development'):
    r=read(REGISTRY)['panels'][profile];c=old.settings('engineering' if profile=='engineering' else 'development');eng=profile=='engineering'
    c.update(profile=profile,train_seeds=r['train_seeds'],development_seeds=list(range(r['development_start'],r['development_start']+r['development_count'])),test_seeds=list(range(r.get('test_start',0),r.get('test_start',0)+r.get('test_count',0))),analysis_seed=r['analysis_seed'],env_steps=192 if eng else 491520,checkpoints=[96,192] if eng else [122880,245760,368640,491520],vector_envs=2 if eng else 16,rollout=96 if eng else 3072,epochs=1 if eng else 4,minibatch=48 if eng else 256,training_processes=1 if eng else 16,confidence=1-(.05/56)/12,scales=[4])
    return c


def rng_role(a):return 'graph' if a in CORE else 'ppo' if a in REFERENCES else a


def derived_streams(c):
    result=[learner.role_seed(c['analysis_seed'],VERSION+':bootstrap')]
    result += [role_seed(s,f'{VERSION}:N4:physical') for s in (*c['development_seeds'],*c['test_seeds'])]
    for t in c['train_seeds']:
        result += [learner.role_seed(t,f'{VERSION}:initialization:{r}') for r in ('graph','ppo','mappo','full_mappo','critic_state')]
        result += [learner.role_seed(t,f'{VERSION}:N4:{r}:{stream}') for r in ('graph','ppo','mappo','full_mappo') for stream in ('actions','minibatches')]
        result += [role_seed(t,f'{VERSION}:N4:training_physical:worker:{w}:episode:{e}') for w in range(c['vector_envs']) for e in range(c['env_steps']//(48*c['vector_envs']))]
    return result


def seed_audit():
    history=set();histories={};own=sha(REGISTRY)
    for p in [*(ROOT/'configs').glob('*seed_registry.json'),*Path('/home/bachng/ht-pdm-fjsp-worktrees').glob('*/configs/*seed_registry.json'),*Path('/home/bachng/ht-pdm-fjsp/configs').glob('*seed_registry.json')]:
        digest=sha(p)
        if digest==own or digest in histories:continue
        histories[digest]=str(p);history|=old.registered_integers(read(p))
    groups=[];streams=[]
    for profile in ('development','confirmation','engineering'):
        c=settings(profile);groups.extend([set(c['train_seeds']),set(c['development_seeds']),set(c['test_seeds']),{c['analysis_seed']}]);streams+=derived_streams(c)
    raw=set.union(*groups);known=set()
    for m in (old,old.reference,old.old_rsc,old.clearing_reference,old.raw_reference,old.budget_reference):
        for profile in ('development','confirmation','engineering'):known.update(old.derived_streams(m.settings(profile),m.VERSION))
    for profile in ('development','confirmation','engineering'):known.update(old.derived_streams(efficiency.settings(profile),old.VERSION))
    from ht_pdm_fjsp import technician_capacity_order_experiment as capacity
    for profile in ('development','confirmation','engineering'):known.update(capacity.derived_streams(capacity.settings(profile)))
    known.update(role_seed(s,f'{VERSION}:N4:physical') for s in history)
    check(not any(a&b for i,a in enumerate(groups) for b in groups[i+1:]) and not raw&history and len(streams)==len(set(streams)) and not set(streams)&(raw|history|known),'fresh disjoint raw/derived streams')
    return dict(status='PASS',raw_unique=len(raw),derived_unique=len(streams),historical_registries=histories,known_historical_derived_checked=len(known),confirmation_states_constructed=False,limitation='Known six learning families plus efficiency and capacity-order and conservative historical raw/current-physical hashes; unrelated ancient RNG namespaces not exhaustively reconstructed.')


def make_model(a,c,t):
    with torch.random.fork_rng():
        torch.manual_seed(learner.role_seed(t,f'{VERSION}:initialization:{rng_role(a)}'))
        model=TypedGibbsPolicy(a) if a in CORE else old.reference.NormalizedPPO(64) if a=='normalized_ppo' else old.reference.FullMAPPO(64) if a=='full_mappo' else learner.Model(a,64)
    with torch.random.fork_rng():torch.manual_seed(learner.role_seed(t,f'{VERSION}:initialization:critic_state'));critic=learner.mlp(71,64,1)
    model.critic=critic if a in CORE or a=='ppo' else old.reference.ValueUnits(critic,value_unit(4))
    return model


def load_model(path,a,c,t,steps):
    data=torch.load(path,map_location='cpu',weights_only=True);check(all(data[k]==v for k,v in dict(protocol=VERSION,algorithm=a,train_seed=t,steps=steps).items()),'checkpoint metadata');model=make_model(a,c,t);model.load_state_dict(data['state_dict']);model.eval();check(learner.parameter_digest(model)==data['parameter_sha256'],'checkpoint parameter digest');return model


def save_model(model,path,a,c,t,steps,probe):
    check(not path.exists(),'checkpoint overwrite refused');data=dict(protocol=VERSION,algorithm=a,train_seed=t,steps=steps,parameter_sha256=learner.parameter_digest(model),state_dict=model.state_dict());torch.save(data,path);roundtrip=load_model(path,a,c,t,steps);check(np.array_equal(model.select(probe),roundtrip.select(probe)),'checkpoint actor roundtrip');return dict(path=str(path),sha256=sha(path),parameter_sha256=data['parameter_sha256'],steps=steps)


def evaluate(a,t,s,model=None,coordination=None,decisions=None,observations=None):
    tags=dict(method=a,train_seed=t,seed=s);env=ResourceEnv(s,4,a,journal=old.Tags(decisions,tags) if decisions else None);obs=env.reset();reward=0.
    for tick in range(48):
        if observations is not None:observations.append({k:v.copy() for k,v in obs.items()})
        action=model.select(obs) if model else None;pairs,diag=env.resolve(action)
        if coordination:coordination.add(dict(tags,policy=json.dumps(model.describe(obs,action),sort_keys=True) if a in CORE else '{}',tick=tick,observation_sha256=hashlib.sha256(obs['state'].tobytes()).hexdigest(),submitted=json.dumps(int(action) if np.isscalar(action) else np.asarray(action).tolist()) if model else 'null',pairs=json.dumps(pairs),diagnostics=json.dumps(diag)))
        obs,r,done,info=env.step(action);reward+=r;check(diag==info['coordination'],'deterministic decoder')
    metrics=info['episode_metrics'];check(done and abs(20*reward+metrics['extended_cost'])<1e-5 and metrics['invalid_assignments']==0,'evaluation clock/reward/invalid');return dict(tags,**metrics,normalized_cost=metrics['extended_cost']/4,normalized_pending=metrics['terminal_pending']/4,normalized_failures=metrics['policy_failures']/4)


def fit_job(job):
    c,a,t,outname,start=job;out=Path(outname);torch.set_num_threads(1);directory=out/'N4'/a/str(t);directory.mkdir(parents=True);(directory/'checkpoints').mkdir();model=make_model(a,c,t);initial=learner.parameter_digest(model);probe=ResourceEnv(c['development_seeds'][0],4,a).reset();journals={key:Journal(directory/f'{key}.csv') for key in ('training_progress','training_episodes','development_episodes')};nodes=[];metas=[]
    def guard():check(not (out/'abort.json').exists(),'sibling technical abort');check(time.monotonic()-start<c['max_seconds'],'12hour hardstop')
    def assess(current,steps):
        guard();current.eval();meta=save_model(current,directory/'checkpoints'/f'step_{steps:09d}.pt',a,c,t,steps,probe);meta['path']=str(Path(meta['path']).relative_to(out));rows=[]
        for s in tqdm(c['development_seeds'],desc=f'DEV {a}/{t}/{steps}',leave=False):
            guard();observations=[] if a==CORE[0] and s==c['development_seeds'][0] else None;row=evaluate(a,t,s,current,observations=observations);row['checkpoint_steps']=steps;rows.append(row);nodes.append(row);journals['development_episodes'].add(row)
            if observations is not None:
                p=directory/'checkpoints'/f'observations_{steps:09d}.npz';np.savez_compressed(p,**{k:np.stack([o[k] for o in observations]) for k in observations[0]});meta['observations']=dict(path=str(p.relative_to(out)),sha256=sha(p),states=48,seed=s)
        meta['development_cost']=float(np.mean([r['extended_cost'] for r in rows]));metas.append(meta);current.train()
    try:
        assess(model,0);unit=1 if a=='ppo' else value_unit(4);cfg=dict(c,rng_namespace=f'{VERSION}:N4:{rng_role(a)}:',value_loss_normalizer=unit,behavior_value_atol=c['behavior_value_atol']*unit)
        audit=fit(model,cfg,t,lambda seed,w:ResourceEnv(seed,4,a,w,True),{key:old.Tags(journals[key],dict(n=4)) for key in ('training_progress','training_episodes')},assess,guard)
        return dict(method=a,train_seed=t,initial=initial,audit=audit,development=nodes,fixed=metas[-1],ancillary=min(metas,key=lambda r:(r['development_cost'],r['steps'])))
    finally:
        for j in journals.values():j.close()


def point_summary(rows,c,latency):
    means={a:{k:float(np.mean([r[k] for r in rows if r['method']==a])) for k in ('normalized_cost','availability','normalized_pending','normalized_failures')} for a in (*ARMS,'deadline')};candidate=means[CORE[0]];points={};checks={}
    for a in ARMS[1:]:
        v=1-candidate['normalized_cost']/means[a]['normalized_cost'];key='gain_vs_'+a;points[key]=v;checks[key]=v>=.005 if a==CORE[1] else v>0 if a in ('mappo','full_mappo') else v>=-.01
    for a in REFERENCES:
        for metric in ('availability','normalized_pending','normalized_failures'):
            key=metric+'_delta_vs_'+a;v=candidate[metric]-means[a][metric];points[key]=v;checks[key]=v>=-.01 if metric=='availability' else v<=.025 if metric=='normalized_pending' else v<=.25
    check(len(checks)==12,'12point endpoints');per={}
    for t in c['train_seeds']:
        costs={a:float(np.mean([r['normalized_cost'] for r in rows if r['method']==a and r['train_seed']==t])) for a in CORE[:2]};per[str(t)]=1-costs[CORE[0]]/costs[CORE[1]]
    extra=dict(every_replica_main_gain_positive=all(v>0 for v in per.values()),zero_invalid=sum(r['invalid_assignments'] for r in rows)==0,isolated_p95_at_most_50ms=latency['p95_ms']<=50)
    return dict(classification='TYPED_GIBBS_DEVELOPMENT_ONLY',means=means,endpoints=points,checks=checks,extra_checks=extra,per_replica_main_gain=per,latency=latency,development_screen_pass=all(checks.values()) and all(extra.values()),goal_pass=False,test_opened=False,teacher_assistance=False,limitation='Standard scalar joint PPO; exact learned two-type matching partition; fixed two physical profiles only. DEV point results not scientific CI/scaling evidence; no promotion without independent audit.')


def audit_training(out,c,results,dev,guard):
    check(len(results)==len(ARMS)*len(c['train_seeds']) and {(r['method'],r['train_seed']) for r in results}=={(a,t) for a in ARMS for t in c['train_seeds']},'complete fit grid');wanted={(a,t,s,step) for a in ARMS for t in c['train_seeds'] for s in c['development_seeds'] for step in (0,*c['checkpoints'])};check(len(dev)==len(wanted) and {(r['method'],r['train_seed'],r['seed'],r['checkpoint_steps']) for r in dev}==wanted,'complete DEV grid')
    for result in results:
        guard();a,t=result['method'],result['train_seed'];r=result['audit'];check(r['from_scratch'] and r['learned_steps']==c['env_steps'] and r['training_episodes']==c['env_steps']//48 and r['optimizer_steps']==c['env_steps']//c['rollout']*c['epochs']*(c['rollout']//c['minibatch']) and r['behavior_normalized_max_error']<=1 and r['changed'] and r['behavior_baselines_cached_before_current_batch_fit'],'fit scratch/budget/cache/replay');zero=load_model(out/'N4'/a/str(t)/'checkpoints/step_000000000.pt',a,c,t,0);final=load_model(out/result['fixed']['path'],a,c,t,c['env_steps']);check(learner.parameter_digest(zero)==learner.parameter_digest(make_model(a,c,t)) and learner.parameter_digest(final)==r['parameter_sha256'],'independent initial/final checkpoints')
        with (out/'N4'/a/str(t)/'training_progress.csv').open() as f:progress=list(csv.DictReader(f))
        check([int(row['learned_steps']) for row in progress]==list(range(c['rollout'],c['env_steps']+1,c['rollout'])) and all(float(row['behavior_normalized_max_error'])<=1 for row in progress) and all(sum(float(row[key]) for row in progress)>0 for key in ('actor_update_norm','critic_update_norm')),'complete updates/replay and nonzero net learning')
    seen=set();paired={}
    with (out/'training_episodes.csv').open() as f:
        for row in csv.DictReader(f):
            a,t,w,e=row['algorithm'],int(row['train_seed']),int(row['worker']),int(row['episode_index']);key=(t,w,e);label=(a,*key);check(label not in seen and a in ARMS and t in c['train_seeds'] and 0<=w<c['vector_envs'] and 0<=e<c['env_steps']//(48*c['vector_envs']),'unique train grid');seen.add(label);check(int(row['physical_seed'])==role_seed(t,f'{VERSION}:N4:training_physical:worker:{w}:episode:{e}') and paired.setdefault(key,row['population_sha256'])==row['population_sha256'] and float(row['invalid_assignments'])==0 and abs(float(row['episode_reward'])*20+float(row['extended_cost']))<1e-5,'paired train physics/reward')
    check(len(seen)==len(results)*c['env_steps']//48,'all train episodes');return dict(status='PASS',models=len(results),training_episodes=len(seen),training_decisions=len(results)*c['env_steps'],paired_training_tapes=len(paired),development_nodes=len(dev))


def audit_final(out,rows,c,guard):
    h=helpers();lookup={(r['method'],r['train_seed'],r['seed']):r for r in rows};wanted={(a,t,s) for a in ARMS for t in c['train_seeds'] for s in c['development_seeds']}|{('deadline',-1,s) for s in c['development_seeds']};check(len(rows)==len(wanted) and set(lookup)==wanted,'fixed-final grid');episodes={};count=0
    with (out/'decisions.csv').open() as f:
        for row in csv.DictReader(f):
            if count%1000==0:guard()
            key=(row['method'],int(row['train_seed']),int(row['seed']));tick=int(row['tick']);s=json.loads(row['state']);pairs=tuple(map(tuple,json.loads(row['action'])));check(key in wanted,'registered replay condition')
            if key not in episodes:
                initial,events,pop=h.initial_events(role_seed(key[2],f'{VERSION}:N4:physical'));episodes[key]=dict(state=initial,events=events,pop=pop,tick=0,metrics={})
            ep=episodes[key];check(ep['state']==s and ep['tick']==tick and json.loads(row['events'])==ep['events'][tick],'independent full state/event chain');following,parts=h.primitive_step(s,pairs,ep['events'][tick]);check(following==json.loads(row['following']) and abs(parts['objective']-float(row['cost']))<1e-8,'independent primitive transition');ep['state']=following;ep['tick']+=1
            for k,v in parts.items():ep['metrics'][k]=ep['metrics'].get(k,0)+v
            if tick==47:ep['cutoff']=dict(ep['metrics']);ep['pending']=sum(following['failed'][m] and m not in following['assigned'] for m in range(4))
            count+=1
    for key,ep in episodes.items():
        r=lookup[key];check(ep['tick']==60 and ep['pop']==r['population_sha256'] and all(abs(v-r[k])<1e-8 for k,v in ep['metrics'].items()),'all physical metrics/population');check(abs(r['availability']-(1-ep['cutoff']['unavailable_ticks']/192))<1e-10 and r['policy_failures']==ep['cutoff']['failures'] and r['terminal_pending']==ep['pending'],'H48health')
    check(count==len(rows)*60 and set(episodes)==wanted,'complete full replay');return dict(status='PASS',episodes=len(rows),physical_transitions=count,policy_decisions=48*len(rows))


def isolated_latency(out,c,choices):
    samples=[]
    for t in c['train_seeds']:
        meta=choices[f'{CORE[0]}:{t}'];model=load_model(out/meta['path'],CORE[0],c,t,c['env_steps']);obsmeta=meta['observations'];check(sha(out/obsmeta['path'])==obsmeta['sha256'],'latency observations binding')
        with np.load(out/obsmeta['path'],allow_pickle=False) as f:obs=[{k:f[k][i] for k in f.files} for i in range(48)]
        for o in obs[:8]:decode_tokens(model.select(o),o['pair_mask'])
        for repeat in range(3):
            for i,o in enumerate(obs):
                before=time.perf_counter_ns();decode_tokens(model.select(o),o['pair_mask']);samples.append(dict(train_seed=t,repeat=repeat,state_index=i,milliseconds=(time.perf_counter_ns()-before)/1e6))
    write_csv(out/'isolated_timing_samples.csv',samples);values=[r['milliseconds'] for r in samples];return dict(samples=len(samples),mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)),p99_ms=float(np.percentile(values,99)),choices_sha256=sha(out/'checkpoint_choices.json'),torch_threads=1,training_workers_active=False,includes='actor NN+broker only, no state/removal critic, observation/physics/network/learning')


def run(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();c=settings();torch.set_num_threads(1);manifest=dict(protocol=VERSION,status='RUNNING',git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())),goal_pass=False,test_opened=False,teacher_assistance=False,pretraining_steps=0)
    write_json(out/'manifest.json',manifest)
    def guard():check(time.monotonic()-start<c['max_seconds'],'12hour hardstop')
    try:
        check(not manifest['git']['dirty'] and not c['test_seeds'],'clean DEV-only code');check(manifest['git']['commit']==subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=ROOT,text=True).strip(),'pushed code gate');check(not any('from multiprocessing.spawn import spawn_main' in r for r in subprocess.check_output(['ps','-eo','args'],text=True).splitlines()),'no active training workers');write_json(out/'resolved_config.json',c);write_json(out/'physics.json',asdict(scaled_config(4)));write_json(out/'seed_audit.json',seed_audit());shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'seed_registry.json');(out/'source_snapshot').mkdir()
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,out/'source_snapshot'/p.name)
        manifest['source_seal']={str(p.relative_to(out)):sha(p) for p in (out/'source_snapshot').iterdir()};manifest['source_seal'].update({name:sha(out/name) for name in ('protocol.md','seed_registry.json','resolved_config.json','physics.json')});write_json(out/'manifest.json',manifest)
        counts={a:sum(p.numel() for p in make_model(a,c,c['train_seeds'][0]).parameters()) for a in ARMS};check(all(counts[a]==18883 for a in CORE),'declared capacity contract');initial={};critic_initial={}
        for t in c['train_seeds']:
            models={a:make_model(a,c,t) for a in ARMS};initial[str(t)]={a:learner.parameter_digest(m) for a,m in models.items()};critic_initial[str(t)]={a:old.raw_tensor_digest(m.critic.net if isinstance(m.critic,old.reference.ValueUnits) else m.critic) for a,m in models.items()};check(len({initial[str(t)][a] for a in CORE})==1 and len(set(critic_initial[str(t)].values()))==1,'matched private initial core/V tensors')
        write_json(out/'parameter_counts.json',counts);write_json(out/'initial_parameter_hashes.json',initial);write_json(out/'initial_critic_tensor_hashes.json',critic_initial);results=[]
        with ProcessPoolExecutor(max_workers=c['training_processes'],mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(fit_job,(c,a,t,str(out),start)) for a in ARMS for t in c['train_seeds']]
            try:
                for future in tqdm(as_completed(futures),total=len(futures),desc='21FULL random-init typed-Gibbs fits'):guard();results.append(future.result());write_json(out/'fit_results.partial.json',[{k:v for k,v in r.items() if k!='development'} for r in results])
            except Exception as exc:
                write_json(out/'abort.json',dict(error=str(exc)))
                for future in futures:future.cancel()
                raise
        results.sort(key=lambda r:(ARMS.index(r['method']),r['train_seed']));dev=[row for r in results for row in r['development']];choices={f"{r['method']}:{r['train_seed']}":r['fixed'] for r in results};write_json(out/'checkpoint_choices.json',choices);write_json(out/'ancillary_checkpoint_choices.json',{f"{r['method']}:{r['train_seed']}":r['ancillary'] for r in results});write_json(out/'training_audit.json',[r['audit'] for r in results]);write_csv(out/'development_episodes.csv',dev)
        for name in ('training_progress','training_episodes'):old.merge_journals(out,name,results)
        audit=audit_training(out,c,results,dev,guard);journals={name:Journal(out/f'{name}.csv') for name in ('coordination','decisions')};rows=[]
        try:
            for a in ARMS:
                for t in c['train_seeds']:
                    meta=choices[f'{a}:{t}'];check(meta['steps']==c['env_steps'],'fixed final checkpoint');model=load_model(out/meta['path'],a,c,t,c['env_steps'])
                    for s in tqdm(c['development_seeds'],desc=f'Fixed final {a}/{t}',leave=False):
                        guard();row=evaluate(a,t,s,model,journals['coordination'],journals['decisions']);rows.append(row);expected=next(r for r in dev if r['method']==a and r['train_seed']==t and r['seed']==s and r['checkpoint_steps']==c['env_steps']);check(all(abs(row[k]-expected[k])<1e-8 for k in ('extended_cost','availability','terminal_pending','policy_failures')),'final checkpoint replay')
            for s in tqdm(c['development_seeds'],desc='Deadline'):rows.append(evaluate('deadline',-1,s,coordination=journals['coordination'],decisions=journals['decisions']))
        finally:
            for j in journals.values():j.close()
        write_csv(out/'selected_episodes.csv',rows);audit['final']=audit_final(out,rows,c,guard);latency=isolated_latency(out,c,choices);write_json(out/'latency.json',latency);summary=point_summary(rows,c,latency);write_json(out/'summary.json',summary);write_json(out/'audit.json',audit)
        for name,digest in manifest['source_seal'].items():check(sha(out/name)==digest and (not name.startswith('source_snapshot/') or sha(Path(__file__).parent/Path(name).name)==digest),'source unchanged')
        check(sha(PLAN)==manifest['source_seal']['protocol.md'] and sha(REGISTRY)==manifest['source_seal']['seed_registry.json'],'protocol unchanged');manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,development_screen_pass=summary['development_screen_pass'],output_bindings={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'});write_json(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),status='PASS',development_screen_pass=summary['development_screen_pass'])));return summary
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);write_json(out/'manifest.json',manifest);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);args=p.parse_args();run(args.output_dir)
