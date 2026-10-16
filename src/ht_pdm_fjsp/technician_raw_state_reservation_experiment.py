"""Registered N4/K2 full development for raw-state categorical reservation PPO."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,multiprocessing as mp,platform,shutil,subprocess,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_capacity_factorized_experiment as reference
from ht_pdm_fjsp import technician_residual_capacity_experiment as old_rsc
from ht_pdm_fjsp.maintenance_opportunity_cost import Journal,write_json,write_csv
from ht_pdm_fjsp.maintenance_balanced_confirmation_mac import registered_integers
from ht_pdm_fjsp.technician_reservation_experiment import Tags
from ht_pdm_fjsp.technician_raw_state_reservation_policy import CORE_ARMS,RawStateReservationPolicy
from ht_pdm_fjsp import technician_joint_clearing_experiment as clearing_reference
from ht_pdm_fjsp.technician_raw_state_reservation_env import VERSION,ResourceEnv,scaled_config,value_unit,role_seed

learner=reference.learner;sha=reference.sha
ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'docs'/f'{VERSION}_plan.md';REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
ARMS=(*CORE_ARMS,'ppo','normalized_ppo','mappo','full_mappo')
FROZEN={f'frozen_{a}':a for a in CORE_ARMS[1:]}


def settings(profile='development'):
    if profile not in ('development','confirmation','engineering'):raise ValueError('profile')
    raw=json.loads(REGISTRY.read_text())['panels'][profile];c=reference.settings('engineering' if profile=='engineering' else 'development')
    c.update(profile=profile,train_seeds=raw['train_seeds'],analysis_seed=raw['analysis_seed'],hidden=64,
        development_seeds=list(range(raw['development_start'],raw['development_start']+raw['development_count'])),
        test_seeds=list(range(raw['test_start'],raw['test_start']+raw['test_count'])) if 'test_start' in raw else [],
        confidence=1-(.05/6)/13,teacher_assistance=False,pretraining_steps=0)
    return c


def rng_role(a):return 'graph' if a in CORE_ARMS else 'ppo' if a=='normalized_ppo' else a


def raw_tensor_digest(model):
    digest=hashlib.sha256()
    for p in model.parameters():
        digest.update(str(tuple(p.shape)).encode());digest.update(p.detach().numpy().tobytes())
    return digest.hexdigest()


def derived_streams(c,version):
    derived=[learner.role_seed(c['analysis_seed'],version+':bootstrap')]
    for n in c['scales']:
        derived += [role_seed(s,f'{version}:N{n}:physical') for s in (*c['development_seeds'],*c['test_seeds'])]
        for t in c['train_seeds']:
            for a in ('graph','ppo','mappo','full_mappo'):
                derived.append(learner.role_seed(t,f'{version}:initialization:{a}'))
                derived += [learner.role_seed(t,f'{version}:N{n}:{a}:{r}') for r in ('actions','minibatches','replay')]
            derived += [role_seed(t,f'{version}:N{n}:training_physical:worker:{w}:episode:{e}')
                        for w in range(c['vector_envs']) for e in range(c['env_steps']//(48*c['vector_envs']))]
    if version==VERSION:
        derived += [learner.role_seed(t,f'{VERSION}:initialization:critic') for t in c['train_seeds']]
    return derived


def seed_audit():
    history=set();hashes={};own_hash=sha(REGISTRY)
    paths=list((ROOT/'configs').glob('*seed_registry.json'))+list(Path('/home/bachng/ht-pdm-fjsp-worktrees').glob('*/configs/*seed_registry.json'))
    for p in paths:
        h=sha(p)
        if h==own_hash or h in hashes:continue
        hashes[h]=str(p);history|=registered_integers(json.loads(p.read_text()))
    groups=[];derived=[]
    for profile in ('development','confirmation','engineering'):
        c=settings(profile);groups.extend([set(c['train_seeds']),set(c['development_seeds']),set(c['test_seeds']),{c['analysis_seed']}]);derived+=derived_streams(c,VERSION)
    own=set.union(*groups);historical_derived=set()
    for module in (reference,old_rsc,clearing_reference):
        for profile in ('development','confirmation','engineering'):historical_derived.update(derived_streams(module.settings(profile),module.VERSION))
    diagnostic=ROOT/'configs/technician_capacity_onpolicy_credit_diagnostic_v1_seed_registry.json'
    if diagnostic.exists():
        reg=json.loads(diagnostic.read_text())['diagnostic'];v='technician_capacity_onpolicy_credit_diagnostic_v1'
        historical_derived.update(role_seed(s,f'{reference.VERSION}:N4:physical') for s in range(reg['physical_start'],reg['physical_start']+reg['physical_count']))
        historical_derived.update(learner.role_seed(s,f'{v}:{r}:actions') for s in reg['action_seeds'] for r in ('graph','ppo'))
    execution=ROOT/'configs/technician_joint_execution_diagnostic_v1_seed_registry.json'
    if execution.exists():
        reg=json.loads(execution.read_text())['diagnostic'];v='technician_joint_execution_diagnostic_v1'
        historical_derived.update(role_seed(s,f'{clearing_reference.VERSION}:N4:physical') for s in range(reg['physical_start'],reg['physical_start']+reg['physical_count']))
        historical_derived.update(learner.role_seed(s,f'{v}:{r}:actions') for s in range(reg['action_start'],reg['action_start']+reg['action_count']) for r in ('core','ppo','mappo'))
    if any(a&b for a,b in itertools_combinations(groups)) or own&history or len(derived)!=len(set(derived)) or set(derived)&(own|history|historical_derived):raise ValueError('seed collision')
    return dict(status='PASS',raw_unique=len(own),derived_unique=len(derived),historical_registries=hashes,
        known_historical_derived_checked=len(historical_derived),limitation='Raw historical registries across worktrees; derived streams explicitly checked for RSC, factorization, joint-clearing and two frozen diagnostic families.')


def itertools_combinations(groups):
    for i,a in enumerate(groups):
        for b in groups[i+1:]:yield a,b


def make_model(a,c,t):
    if a not in ARMS:raise ValueError('arm')
    with torch.random.fork_rng():
        torch.manual_seed(learner.role_seed(t,f'{VERSION}:initialization:'+rng_role(a)))
        if a in CORE_ARMS:model=RawStateReservationPolicy(a,4,2,c['hidden'])
        elif a=='normalized_ppo':model=reference.NormalizedPPO(c['baseline_hidden'])
        else:model=reference.FullMAPPO(c['baseline_hidden']) if a=='full_mappo' else learner.Model(a,c['baseline_hidden'])
    # Fresh matched random critic, private storage/updates in every method.
    with torch.random.fork_rng():
        torch.manual_seed(learner.role_seed(t,f'{VERSION}:initialization:critic'))
        critic=learner.mlp(71,c['baseline_hidden'],1)
    model.critic=critic if a in CORE_ARMS or a=='ppo' else reference.ValueUnits(critic,value_unit(4))
    return model


def load_model(path,a,c,t,steps):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if any(data.get(k)!=v for k,v in dict(protocol=VERSION,algorithm=a,train_seed=t,steps=steps).items()):raise RuntimeError('checkpoint metadata')
    model=make_model(a,c,t);model.load_state_dict(data['state_dict']);model.eval()
    if learner.parameter_digest(model)!=data['parameter_sha256']:raise RuntimeError('checkpoint tensor digest')
    return model


def save_model(model,path,a,c,t,steps,probe):
    if path.exists():raise FileExistsError('checkpoint overwrite')
    data=dict(protocol=VERSION,algorithm=a,train_seed=t,steps=steps,parameter_sha256=learner.parameter_digest(model),state_dict=model.state_dict())
    torch.save(data,path);loaded=load_model(path,a,c,t,steps)
    if not np.array_equal(model.select(probe),loaded.select(probe)):raise RuntimeError('checkpoint roundtrip')
    meta=dict(path=str(path),sha256=sha(path),parameter_sha256=data['parameter_sha256'])
    return meta


def _evaluate(a,t,s,model=None,coordination=None,decisions=None,observations=None):
    tags=dict(method=a,n=4,train_seed=t,seed=s);method=FROZEN.get(a,a)
    env=ResourceEnv(s,4,method,journal=Tags(decisions,tags) if decisions else None);obs=env.reset();reward=0.;times=[]
    for tick in range(48):
        if observations is not None:observations.append({k:v.copy() for k,v in obs.items()})
        start=time.perf_counter();action=model.select(obs) if model else None;pairs,diag=env.resolve(action);elapsed=(time.perf_counter()-start)*1000;times.append(elapsed)
        state_hash=hashlib.sha256(obs['state'].tobytes()).hexdigest();obs,r,done,info=env.step(action);reward+=r
        if diag!=info['coordination']:raise RuntimeError('decoder is not deterministic')
        if coordination:coordination.add(dict(tags,tick=tick,observation_sha256=state_hash,
            submitted=json.dumps(int(action) if np.isscalar(action) else np.asarray(action).tolist()) if model else 'null',
            pairs=json.dumps(pairs),diagnostics=json.dumps(diag,sort_keys=True),decision_ms=elapsed))
    metrics=info['episode_metrics']
    if not done or abs(-20*reward-metrics['extended_cost'])>1e-5 or metrics['invalid_assignments']:raise RuntimeError('evaluation reward/health reconciliation')
    return dict(tags,**metrics,decision_mean_ms=float(np.mean(times)),normalized_cost=metrics['extended_cost']/4,
        normalized_pending=metrics['terminal_pending']/4,normalized_failures=metrics['policy_failures']/4)


def evaluate(a,t,s,model=None,coordination=None,decisions=None,observations=None):
    previous=model.algorithm if model is not None else None
    if model is not None and a in FROZEN:model.algorithm=FROZEN[a]
    try:return _evaluate(a,t,s,model,coordination,decisions,observations)
    finally:
        if model is not None:model.algorithm=previous


def fit_job(job):
    c,a,t,out_name,start=job;out=Path(out_name);torch.set_num_threads(1)
    def guard():
        if (out/'abort.json').exists():raise RuntimeError('sibling technical abort')
        if time.monotonic()-start>c['max_seconds']:raise TimeoutError('12h hardstop')
    model=make_model(a,c,t);initial=learner.parameter_digest(model);directory=out/'N4'/a/str(t);directory.mkdir(parents=True);(directory/'checkpoints').mkdir()
    journals={key:Journal(directory/f'{key}.csv') for key in ('training_progress','training_episodes','development_episodes')};choices=[];dev=[]
    probe=ResourceEnv(c['development_seeds'][0],4,a).reset()
    def assess(current,steps,audit=None):
        guard();current.eval();meta=save_model(current,directory/'checkpoints'/f'step_{steps:09d}.pt',a,c,t,steps,probe);meta['path']=str(Path(meta['path']).relative_to(out));rows=[]
        for s in tqdm(c['development_seeds'],desc=f'Dev {a}/{t}/{steps}',leave=False):
            guard();observations=[] if a==CORE_ARMS[0] and s==c['development_seeds'][0] else None
            row=evaluate(a,t,s,current,observations=observations);row['checkpoint_steps']=steps;journals['development_episodes'].add(row);rows.append(row);dev.append(row)
            if observations is not None:
                p=directory/'checkpoints'/f'observations_{steps:09d}.npz';np.savez_compressed(p,**{k:np.stack([o[k] for o in observations]) for k in observations[0]})
                meta['observations']=dict(path=str(p.relative_to(out)),sha256=sha(p),states=48,seed=s)
        choices.append(dict(meta,steps=steps,development_cost=float(np.mean([r['extended_cost'] for r in rows]))));current.train()
    try:
        assess(model,0);unit=1 if a=='ppo' else value_unit(4)
        cfg=dict(c,rng_namespace=f'{VERSION}:N4:{rng_role(a)}:',value_loss_normalizer=unit,
            behavior_value_atol=c['behavior_value_atol']*unit,group_gradient_clipping=False,log_gradient_groups=a in CORE_ARMS)
        audit=learner.fit(model,cfg,t,lambda seed,w:ResourceEnv(seed,4,a,w,True),
            {key:Tags(journals[key],dict(n=4)) for key in ('training_progress','training_episodes')},assess,guard)
        audit.update(n=4,from_scratch=True,pretraining_steps=0)
        return dict(method=a,train_seed=t,initial=initial,parameters=sum(p.numel() for p in model.parameters()),audit=audit,development=dev,choice=min(choices,key=lambda d:(d['development_cost'],d['steps'])))
    finally:
        for journal in journals.values():journal.close()


def merge_journals(out,name,results):
    paths=[out/'N4'/r['method']/str(r['train_seed'])/f'{name}.csv' for r in results];fields=[]
    for p in paths:
        with p.open() as f:
            for key in csv.DictReader(f).fieldnames:
                if key not in fields:fields.append(key)
    with (out/f'{name}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for p in paths:
            with p.open() as g:writer.writerows(csv.DictReader(g))


def point_summary(rows,latency):
    means={a:{key:float(np.mean([r[key] for r in rows if r['method']==a])) for key in ('normalized_cost','availability','normalized_pending','normalized_failures')} for a in (*ARMS,'deadline',*FROZEN)}
    candidate=means[CORE_ARMS[0]];endpoints={};checks={}
    for a in ARMS[1:]:
        gain=1-candidate['normalized_cost']/means[a]['normalized_cost'];key='gain_vs_'+a;endpoints[key]=gain
        threshold=.005 if a=='independent_raw_ppo' else 0 if a in ('mappo','full_mappo') else -.01
        checks[key]=gain>0 if a in ('mappo','full_mappo') else gain>=threshold
    for a in ('ppo','normalized_ppo'):
        for metric,threshold,lower in (('availability',-.01,True),('normalized_pending',.025,False),('normalized_failures',.25,False)):
            key=metric+'_delta_vs_'+a;delta=candidate[metric]-means[a][metric];endpoints[key]=delta;checks[key]=delta>=threshold if lower else delta<=threshold
    if len(checks)!=13:raise RuntimeError('13endpoint family')
    invalid=sum(r['invalid_assignments'] for r in rows);latency_pass=latency['p95_ms']<=50
    return dict(classification='DEVELOPMENT_ONLY',goal_pass=False,test_opened=False,teacher_assistance=False,
        means=means,endpoints=endpoints,checks=checks,invalid_assignments=invalid,latency=latency,latency_pass=latency_pass,
        development_screen_pass=all(checks.values()) and latency_pass and invalid==0,
        limitation='DEV point gates only; no scientific CI or confirmation/larger-N authorization without independent audit.')


def isolated_latency(out,c,choices,guard):
    from ht_pdm_fjsp.technician_raw_state_reservation_policy import decode_tokens
    samples=[];bindings=[]
    for t in c['train_seeds']:
        choice=choices[f'{CORE_ARMS[0]}:{t}'];model=load_model(out/choice['path'],CORE_ARMS[0],c,t,choice['steps']);meta=choice['observations']
        if sha(out/meta['path'])!=meta['sha256']:raise RuntimeError('timing observations binding')
        bindings.append(dict(train_seed=t,checkpoint_steps=choice['steps'],checkpoint_sha256=choice['sha256'],**meta))
        with np.load(out/meta['path'],allow_pickle=False) as f:obs=[{k:f[k][i] for k in f.files} for i in range(48)]
        before_digest=learner.parameter_digest(model)
        for o in obs[:8]:decode_tokens(model.select(o),o['pair_mask'],CORE_ARMS[0],o['edges'])
        for repeat in range(3):
            for i,o in enumerate(obs):
                guard();before=time.perf_counter_ns();decode_tokens(model.select(o),o['pair_mask'],CORE_ARMS[0],o['edges']);elapsed=(time.perf_counter_ns()-before)/1e6
                samples.append(dict(train_seed=t,repeat=repeat,state_index=i,milliseconds=elapsed))
        if learner.parameter_digest(model)!=before_digest:raise RuntimeError('timing mutated model')
    write_csv(out/'isolated_timing_samples.csv',samples);times=[r['milliseconds'] for r in samples]
    return dict(samples=len(samples),mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),p99_ms=float(np.percentile(times,99)),
        choices_sha256=sha(out/'checkpoint_choices.json'),observation_bindings=bindings,torch_threads=torch.get_num_threads(),training_workers_active=False,
        includes='actor NN plus canonical token decoder; isolated after training; excludes observation/physical transition and transport')


def audit_training(out,c,results,choices,dev,guard):
    expected={(a,t) for a in ARMS for t in c['train_seeds']}
    if {(r['method'],r['train_seed']) for r in results}!=expected or len(results)!=len(expected):raise RuntimeError('training grid')
    paired={};seen=set()
    for r in results:
        a,t=r['method'],r['train_seed'];audit=r['audit'];guard()
        if audit['learned_steps']!=c['env_steps'] or audit['optimizer_steps']!=c['env_steps']//c['rollout']*c['epochs']*(c['rollout']//c['minibatch']) or audit['training_episodes']!=c['env_steps']//48 or audit['physical_ticks']!=c['env_steps']//48*60 or not audit['changed'] or audit['behavior_normalized_max_error']>1:raise RuntimeError('training budget/update contract')
        initial=load_model(out/'N4'/a/str(t)/'checkpoints/step_000000000.pt',a,c,t,0)
        if learner.parameter_digest(initial)!=learner.parameter_digest(make_model(a,c,t)):raise RuntimeError('scratch initialization')
        final=load_model(out/'N4'/a/str(t)/f"checkpoints/step_{c['env_steps']:09d}.pt",a,c,t,c['env_steps'])
        if learner.parameter_digest(final)!=audit['parameter_sha256']:raise RuntimeError('final checkpoint/update digest')
        with (out/'N4'/a/str(t)/'training_progress.csv').open() as f:progress=list(csv.DictReader(f))
        if [int(v['learned_steps']) for v in progress]!=list(range(c['rollout'],c['env_steps']+1,c['rollout'])):raise RuntimeError('complete update journal')
        for v in progress:
            expected_opts=int(v['learned_steps'])//c['rollout']*c['epochs']*(c['rollout']//c['minibatch'])
            if int(v['optimizer_steps'])!=expected_opts or float(v['behavior_normalized_max_error'])>1:raise RuntimeError('update journal budget/replay')
        if a in CORE_ARMS and any(sum(float(v[field]) for v in progress)<=0 for field in ('actor_update_norm','critic_update_norm')):raise RuntimeError('actor/critic did not update')
        data=[v for v in dev if v['method']==a and v['train_seed']==t]
        wanted={(step,s) for step in (0,*c['checkpoints']) for s in c['development_seeds']}
        if len(data)!=len(wanted) or {(v['checkpoint_steps'],v['seed']) for v in data}!=wanted:raise RuntimeError('DEV grid')
        score={step:float(np.mean([v['extended_cost'] for v in data if v['checkpoint_steps']==step])) for step in (0,*c['checkpoints'])}
        best=min(score,key=lambda step:(score[step],step));choice=choices[f'{a}:{t}']
        if choice['steps']!=best or choice['development_cost']!=score[best] or sha(out/choice['path'])!=choice['sha256']:raise RuntimeError('checkpoint selection')
    with (out/'training_episodes.csv').open() as f:
        for r in csv.DictReader(f):
            t,w,e=int(r['train_seed']),int(r['worker']),int(r['episode_index']);a=r['algorithm'];key=(t,w,e);identity=(a,*key)
            if identity in seen:raise RuntimeError('duplicate train episode')
            seen.add(identity);tape=(r['physical_seed'],r['population_sha256'])
            if paired.setdefault(key,tape)!=tape or int(r['physical_seed'])!=role_seed(t,f'{VERSION}:N4:training_physical:worker:{w}:episode:{e}'):raise RuntimeError('training physical pairing')
            if float(r['invalid_assignments']) or abs(float(r['episode_reward'])*20+float(r['extended_cost']))>1e-5:raise RuntimeError('training invalid/reward')
    expected_episodes={(a,t,w,e) for a,t in expected for w in range(c['vector_envs']) for e in range(c['env_steps']//(48*c['vector_envs']))}
    if seen!=expected_episodes:raise RuntimeError('complete training episode grid')
    return dict(status='PASS',from_scratch_models=len(results),training_episodes=len(seen),paired_training_tapes=len(paired),development_episodes=len(dev),teacher_assistance=False,pretraining_steps=0)


def audit_selected(out,c,rows,guard):
    from ht_pdm_fjsp.maintenance_dispatch import DispatchState
    from ht_pdm_fjsp.maintenance_waiting import transition,WaitingEnv
    from ht_pdm_fjsp.technician_reservation_env import deadline
    from ht_pdm_fjsp.maintenance_skill_contention_screen import pending
    lookup={(r['method'],r['train_seed'],r['seed']):r for r in rows};coord={};episodes={};population={}
    if len(lookup)!=len(rows):raise RuntimeError('selected episode duplicates')
    wanted={(a,t,s) for a in (*ARMS,*FROZEN) for t in c['train_seeds'] for s in c['development_seeds']}|{('deadline',-1,s) for s in c['development_seeds']}
    if set(lookup)!=wanted:raise RuntimeError('selected methods/replicas/seed grid')
    for r in rows:
        if population.setdefault(r['seed'],r['population_sha256'])!=r['population_sha256']:raise RuntimeError('selected physical pairing')
    with (out/'coordination.csv').open() as f:
        for r in csv.DictReader(f):
            key=r['method'],int(r['train_seed']),int(r['seed']),int(r['tick'])
            if key in coord:raise RuntimeError('duplicate coordination')
            coord[key]=r
    ticks=0
    with (out/'decisions.csv').open() as f:
        for r in csv.DictReader(f):
            if ticks%1000==0:guard()
            key=r['method'],int(r['train_seed']),int(r['seed']);tick=int(r['tick']);config=scaled_config(4)
            if key not in lookup:raise RuntimeError('unexpected physical trace')
            if key not in episodes:
                physical=role_seed(key[2],f'{VERSION}:N4:physical');initial=WaitingEnv(config);initial.reset(physical)
                probe=ResourceEnv(key[2],4,FROZEN.get(key[0],key[0]))
                episodes[key]=dict(state=initial.state,events=initial.events,tick=0,metrics={k:0. for k in initial.metrics},probe=probe,
                    broker=dict.fromkeys(('order_forward','order_reverse','conflicts','rejected_proposals'),0))
            ep=episodes[key];state=DispatchState(**{k:tuple(v) for k,v in json.loads(r['state']).items()});pairs=tuple(map(tuple,json.loads(r['action'])))
            if state!=ep['state'] or tick!=ep['tick'] or tuple(json.loads(r['events']))!=ep['events'][tick]:raise RuntimeError('physical state/event chain')
            if tick<48:
                cr=coord[key+(tick,)];probe=ep['probe'];probe.env.state=state;probe.env.time=tick;obs=probe.observe()
                if hashlib.sha256(obs['state'].tobytes()).hexdigest()!=cr['observation_sha256']:raise RuntimeError('observation hash')
                submitted=json.loads(cr['submitted'])
                if isinstance(submitted,list):submitted=np.asarray(submitted)
                decoded,diag=probe.resolve(submitted)
                if decoded!=pairs or diag!=json.loads(cr['diagnostics']) or list(map(list,pairs))!=json.loads(cr['pairs']):raise RuntimeError('canonical reservation decoding replay')
                for field in ep['broker']:ep['broker'][field]+=diag[field]
            elif deadline(config,state)!=pairs:raise RuntimeError('common tail policy')
            following,cost,components=transition(config,state,pairs,json.loads(r['events']))
            for k,v in components.items():ep['metrics'][k]+=v
            ep['state']=following;ep['tick']+=1
            if tick==47:
                ep['policy_cost']=ep['metrics']['objective'];ep['cutoff']=dict(ep['metrics']);ep['terminal_pending']=pending(config,following)
            if asdict(following)!={k:tuple(v) for k,v in json.loads(r['following']).items()} or abs(cost-float(r['cost']))>1e-8:raise RuntimeError('scalar physical transition')
            ticks+=1
    if ticks!=60*len(rows) or len(coord)!=48*len(rows) or set(episodes)!=set(lookup):raise RuntimeError('full selected trace grid')
    for key,ep in episodes.items():
        row=lookup[key]
        if ep['tick']!=60 or abs(ep['policy_cost']-row['policy_cost'])>1e-8 or any(abs(v-row[k])>1e-8 for k,v in ep['metrics'].items()):raise RuntimeError('selected aggregate physics')
        if abs(row['availability']-(1-ep['cutoff']['unavailable_ticks']/(4*48)))>1e-10 or row['policy_failures']!=ep['cutoff']['failures'] or row['terminal_pending']!=ep['terminal_pending']:raise RuntimeError('independent cutoff health')
        if any(row['broker_'+field]!=value for field,value in ep['broker'].items()):raise RuntimeError('broker aggregate reconciliation')
    return dict(status='PASS',episodes=len(rows),physical_transitions=ticks,coordination_decisions=len(coord),invalid_assignments=0)


def run(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();c=settings();torch.set_num_threads(1)
    manifest=dict(protocol=VERSION,status='RUNNING',goal_pass=False,test_opened=False,teacher_assistance=False,
        started_utc=datetime.now(timezone.utc).isoformat(),git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())))
    write_json(out/'manifest.json',manifest)
    def guard():
        if time.monotonic()-start>c['max_seconds']:raise TimeoutError('12h hardstop')
    try:
        if manifest['git']['dirty'] or c['test_seeds']:raise RuntimeError('clean DEV code with tests closed required')
        write_json(out/'resolved_config.json',c);write_json(out/'physics.json',asdict(scaled_config(4)));write_json(out/'seed_audit.json',seed_audit())
        shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'seed_registry.json');(out/'source_snapshot').mkdir()
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,out/'source_snapshot'/p.name)
        manifest['source_seal']={str(p.relative_to(out)):sha(p) for p in (out/'source_snapshot').iterdir()}
        manifest['source_seal'].update({name:sha(out/name) for name in ('protocol.md','seed_registry.json','resolved_config.json','physics.json')});write_json(out/'manifest.json',manifest)
        counts={a:sum(p.numel() for p in make_model(a,c,c['train_seeds'][0]).parameters()) for a in ARMS}
        if any(counts[a]!=19974 for a in CORE_ARMS):raise RuntimeError('core capacity contract')
        initial={str(t):{a:learner.parameter_digest(make_model(a,c,t)) for a in ARMS} for t in c['train_seeds']}
        raw_initial={str(t):{a:raw_tensor_digest(make_model(a,c,t)) for a in ARMS} for t in c['train_seeds']}
        if any(len({data[a] for a in CORE_ARMS})!=1 for data in initial.values()) or any(data['ppo']!=data['normalized_ppo'] for data in raw_initial.values()):raise RuntimeError('paired initial tensors')
        critic_initial={str(t):{a:raw_tensor_digest((m.critic.net if isinstance(m.critic,reference.ValueUnits) else m.critic)) for a in ARMS for m in [make_model(a,c,t)]} for t in c['train_seeds']}
        if any(len(set(data.values()))!=1 for data in critic_initial.values()):raise RuntimeError('matched fresh critic tensors')
        write_json(out/'parameter_counts.json',counts);write_json(out/'initial_parameter_hashes.json',initial);write_json(out/'initial_raw_tensor_hashes.json',raw_initial);write_json(out/'initial_critic_tensor_hashes.json',critic_initial)
        jobs=[(c,a,t,str(out),start) for a in ARMS for t in c['train_seeds']];results=[]
        with ProcessPoolExecutor(max_workers=c['training_processes'],mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(fit_job,j) for j in jobs]
            try:
                for future in tqdm(as_completed(futures),total=len(futures),desc='24 FULL random-init fits'):
                    guard();results.append(future.result());write_json(out/'fit_results.partial.json',[{k:v for k,v in r.items() if k!='development'} for r in results])
            except Exception as exc:
                write_json(out/'abort.json',dict(error=str(exc)))
                for f in futures:f.cancel()
                raise
        results.sort(key=lambda r:(ARMS.index(r['method']),r['train_seed']));dev=[v for r in results for v in r['development']]
        choices={f"{r['method']}:{r['train_seed']}":r['choice'] for r in results};write_json(out/'checkpoint_choices.json',choices);write_csv(out/'development_episodes.csv',dev)
        write_json(out/'training_audit.json',[r['audit'] for r in results])
        for name in ('training_progress','training_episodes'):merge_journals(out,name,results)
        audit=audit_training(out,c,results,choices,dev,guard)
        journals={key:Journal(out/f'{key}.csv') for key in ('coordination','decisions')};rows=[]
        try:
            for a in ARMS:
                for t in c['train_seeds']:
                    choice=choices[f'{a}:{t}'];model=load_model(out/choice['path'],a,c,t,choice['steps'])
                    for s in tqdm(c['development_seeds'],desc=f'Replay selected {a}/{t}',leave=False):
                        guard();r=evaluate(a,t,s,model,journals['coordination'],journals['decisions']);rows.append(r)
                        expected=[v for v in dev if v['method']==a and v['train_seed']==t and v['seed']==s and v['checkpoint_steps']==choice['steps']][0]
                        if any(abs(r[k]-expected[k])>1e-8 for k in ('extended_cost','availability','terminal_pending','policy_failures')):raise RuntimeError('selected DEV mismatch')
            for label in FROZEN:
                for t in c['train_seeds']:
                    choice=choices[f'{CORE_ARMS[0]}:{t}'];model=load_model(out/choice['path'],CORE_ARMS[0],c,t,choice['steps']);before=learner.parameter_digest(model)
                    for s in tqdm(c['development_seeds'],desc=f'Frozen {label}/{t}',leave=False):
                        guard();rows.append(evaluate(label,t,s,model,journals['coordination'],journals['decisions']))
                    if learner.parameter_digest(model)!=before:raise RuntimeError('frozen ablation changed weights')
            for s in tqdm(c['development_seeds'],desc='Deadline reference'):
                guard();rows.append(evaluate('deadline',-1,s,coordination=journals['coordination'],decisions=journals['decisions']))
        finally:
            for j in journals.values():j.close()
        write_csv(out/'selected_episodes.csv',rows);audit['selected']=audit_selected(out,c,rows,guard)
        latency=isolated_latency(out,c,choices,guard);write_json(out/'latency.json',latency);summary=point_summary(rows,latency)
        write_json(out/'summary.json',summary);write_json(out/'audit.json',audit)
        for name,h in manifest['source_seal'].items():
            if sha(out/name)!=h or (name.startswith('source_snapshot/') and sha(Path(__file__).parent/Path(name).name)!=h):raise RuntimeError('source changed during run')
        manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,development_screen_pass=summary['development_screen_pass'],
            output_bindings={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'})
        write_json(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),status=manifest['status'],development_screen_pass=summary['development_screen_pass'])));return manifest
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);write_json(out/'manifest.json',manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.output_dir)

if __name__=='__main__':main()
