"""Frozen N4 physical matching-mode diagnostic; no learning or scalable-mode claim."""
import argparse,csv,importlib.util,json,shutil,subprocess,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_broker_marginal_experiment as x
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import transition
from ht_pdm_fjsp.technician_reservation_env import deadline

ROOT=Path(__file__).resolve().parents[2];VERSION='technician_broker_mode_diagnostic_v1'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md';REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
SOURCE_COMMIT='dcef1f554bf54172aed7f61cfdefe8d8e16b70d9'


def check(ok,label):
    if not ok:raise AssertionError(label)
def read(p):return json.loads(p.read_text())
def helpers():
    p=ROOT/'scripts/verify_broker_marginal_development.py';spec=importlib.util.spec_from_file_location('mode_reference',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def seed_audit():
    reg=read(REGISTRY)['diagnostic'];seeds=list(range(reg['physical_start'],reg['physical_start']+reg['physical_count']));check(len(seeds)==64,'64fixed seeds')
    history=set();hashes={};own_hash=x.sha(REGISTRY)
    for p in [*(ROOT/'configs').glob('*seed_registry.json'),*Path('/home/bachng/ht-pdm-fjsp-worktrees').glob('*/configs/*seed_registry.json')]:
        h=x.sha(p)
        if h==own_hash or h in hashes:continue
        hashes[h]=str(p);history|=x.registered_integers(read(p))
    derived=[x.role_seed(s,f'{x.VERSION}:N4:physical') for s in seeds];known=set()
    for module in (x,x.reference,x.old_rsc,x.clearing_reference,x.raw_reference,x.budget_reference):
        for profile in ('development','confirmation','engineering'):known.update(x.derived_streams(module.settings(profile),module.VERSION))
    for name in ('technician_capacity_onpolicy_credit_diagnostic_v1','technician_joint_execution_diagnostic_v1'):
        p=ROOT/'configs'/f'{name}_seed_registry.json'
        if not p.exists():continue
        r=read(p)['diagnostic'];origin=x.reference.VERSION if 'onpolicy' in name else x.clearing_reference.VERSION
        known.update(x.role_seed(s,f'{origin}:N4:physical') for s in range(r['physical_start'],r['physical_start']+r['physical_count']))
        actions=r.get('action_seeds',list(range(r.get('action_start',0),r.get('action_start',0)+r.get('action_count',0))))
        roles=('graph','ppo') if 'onpolicy' in name else ('core','ppo','mappo')
        known.update(x.learner.role_seed(s,f'{name}:{role}:actions') for s in actions for role in roles)
    check(not set(seeds)&history and len(set(derived))==64 and not set(derived)&(history|set(seeds)|known),'fresh raw/derived diagnostic seeds')
    return dict(status='PASS',raw_unique=64,derived_unique=64,physical_seeds=seeds,historical_registries=hashes,known_historical_derived_checked=len(known))


def complete_obs(h,state,tick):
    o=h.observation(state,tick);workers=o['technicians'].copy();workers[:,2]=0
    o['local']=np.concatenate([o['machines'],np.tile(workers.ravel(),(4,1)),o['edges'].reshape(4,-1),np.tile(o['global_features'],(4,1)),np.eye(4,dtype=np.float32)],-1)
    o['full_local']=np.concatenate([np.tile(o['state'],(4,1)),np.eye(4,dtype=np.float32)],-1)
    o['local_mask']=np.c_[np.ones(4,bool),o['pair_mask']];o['joint_mask']=np.array([all(o['pair_mask'][m,j] for m,j in pairs) for pairs in h.baseline_catalogue()])
    return o


def generic_logmass(logp,assigned,priority,legal):
    winners=[(token-1,a) for a,token in enumerate(assigned) if token];total=0.
    for a,token in enumerate(assigned):
        if token:total+=logp[a,token]
        else:
            allowed=[0]+[r+1 for r,w in winners if legal[r,a] and (priority[r,w]>priority[r,a] or (priority[r,w]==priority[r,a] and w<a))]
            total+=np.logaddexp.reduce(logp[a,allowed])
    return float(total)


def decode_greedy(action,arm,state,o,h):
    if arm in x.CORE_ARMS:
        pairs,_,diag=h.independent_decode(action,o['pair_mask'],arm,o['edges']);return pairs,int(diag['rejected_proposals'])
    if arm in ('ppo','normalized_ppo'):return h.baseline_catalogue()[int(action)],0
    requests=[(m,int(token)-1) for m,token in enumerate(action) if token];pairs=[]
    check(all(o['pair_mask'][m,j] for m,j in requests),'greedy physical support')
    for j in range(2):
        group=[p for p in requests if p[1]==j]
        if group:pairs.append(max(group,key=lambda p:(h.urgency(state,p[0]),-p[0])))
    return tuple(sorted(pairs)),len(requests)-len(pairs)


def representative(pairs,arm,h):
    if arm in ('ppo','normalized_ppo'):return h.baseline_catalogue().index(pairs)
    result=np.zeros(2 if arm in x.CORE_ARMS else 4,dtype=np.int64)
    for m,j in pairs:result[j if arm in x.CORE_ARMS else m]=(m if arm in x.CORE_ARMS else j)+1
    return result


@torch.no_grad()
def query(model,arm,state,o,h):
    greedy=model.select(o);greedy_pairs,rejected=decode_greedy(greedy,arm,state,o,h);batch=x.learner.tensors([o])
    if arm in x.CORE_ARMS:
        scores=model.proposal_log_probs(batch)[0][0].numpy().astype(float);legal=o['pair_mask'];priority=h.priority_matrix(o,arm)
    else:
        scores=model.policy(batch)[0].numpy().astype(float)
        if arm not in ('ppo','normalized_ppo'):legal=o['pair_mask'].T;priority=np.tile([h.urgency(state,m) for m in range(4)],(2,1))
    if arm in ('ppo','normalized_ppo'):
        scores=np.where(o['joint_mask'],scores,-np.inf);logp=scores-np.logaddexp.reduce(scores)
        values=np.array([float(logp[i]) if o['joint_mask'][i] else -np.inf for i in range(21)]);expected_rejections=0.
    else:
        support=np.c_[np.ones(scores.shape[0],bool),legal.T];scores=np.where(support,scores,-np.inf);logp=scores-np.logaddexp.reduce(scores,axis=1)[:,None];values=[]
        for pairs in h.baseline_catalogue():
            if not all(o['pair_mask'][m,j] for m,j in pairs):values.append(-np.inf);continue
            assigned=np.zeros(scores.shape[0],int)
            for m,j in pairs:assigned[j if arm in x.CORE_ARMS else m]=(m if arm in x.CORE_ARMS else j)+1
            values.append(generic_logmass(logp,assigned,priority,legal))
        values=np.array(values);p=np.exp(logp[:,1:]);expected_rejections=float((p.sum(0)-(1-np.prod(1-p,axis=0))).sum())
    check(abs(float(np.exp(values).sum())-1)<2e-10,'physical matching mass normalization')
    mode_pairs=h.baseline_catalogue()[int(values.argmax())];greedy_log=float(values[h.baseline_catalogue().index(greedy_pairs)]);mode_log=float(values.max())
    check(mode_log>=greedy_log-2e-10,'physical mode maximality')
    if arm in ('ppo','normalized_ppo'):check(mode_pairs==greedy_pairs,'joint PPO mode must equal greedy')
    return dict(greedy=greedy,greedy_pairs=greedy_pairs,mode_pairs=mode_pairs,greedy_logmass=greedy_log,mode_logmass=mode_log,
        greedy_rejections=rejected,expected_latent_rejections=max(0.,expected_rejections),mismatch=int(greedy_pairs!=mode_pairs))


def collect(model,arm,t,seeds,mode,h,physical=None,policy=None,guard=lambda:None):
    check(mode in ('greedy','matching_mode'),'execution mode');episodes=[];config=x.scaled_config(4)
    for raw in tqdm(seeds,desc=f'{arm}/{t}/{mode}',leave=False,mininterval=1):
        physical_seed=x.role_seed(raw,f'{x.VERSION}:N4:physical');state,events,population=h.initial_events(physical_seed);metrics=defaultdict(float);mismatches=0;latent=0.;rejections=0;times=[]
        for tick in range(60):
            guard();tags=dict(method=arm,train_seed=t,seed=raw,mode=mode,tick=tick,physical_seed=physical_seed)
            if tick<48:
                o=complete_obs(h,state,tick);before=time.perf_counter();q=query(model,arm,state,o,h);times.append((time.perf_counter()-before)*1000)
                pairs=q['greedy_pairs'] if mode=='greedy' else q['mode_pairs'];submitted=q['greedy'] if mode=='greedy' else representative(pairs,arm,h)
                decoded,actual_rejected=decode_greedy(submitted,arm,state,o,h);check(decoded==pairs,'executed representative matching')
                mismatches+=q['mismatch'];latent+=q['expected_latent_rejections'];rejections+=actual_rejected
                if policy:policy.add(dict(tags,submitted=json.dumps(int(submitted) if np.isscalar(submitted) else np.asarray(submitted).tolist()),greedy_pairs=json.dumps(q['greedy_pairs']),mode_pairs=json.dumps(q['mode_pairs']),executed_pairs=json.dumps(pairs),greedy_logmass=q['greedy_logmass'],mode_logmass=q['mode_logmass'],same_state_mismatch=q['mismatch'],expected_latent_rejections=q['expected_latent_rejections'],executed_rejections=actual_rejected))
            else:pairs=deadline(config,DispatchState(**{k:tuple(v) for k,v in state.items()}))
            following,components=h.primitive_step(state,pairs,events[tick]);actual,cost,parts=transition(config,DispatchState(**{k:tuple(v) for k,v in state.items()}),pairs,events[tick])
            check(following=={k:list(v) for k,v in vars(actual).items()} and components==parts and abs(cost-components['objective'])<1e-10,'dictionary/scalar full transition')
            if physical:physical.add(dict(tags,state=json.dumps(state),pairs=json.dumps(pairs),events=json.dumps(events[tick]),following=json.dumps(following),cost=cost))
            for k,v in components.items():metrics[k]+=v
            state=following
            if tick==47:cutoff=dict(metrics);pending=sum(state['failed'][m] and m not in state['assigned'] for m in range(4))
        episodes.append(dict(method=arm,train_seed=t,seed=raw,mode=mode,physical_seed=physical_seed,population_sha256=population,
            **metrics,extended_cost=metrics['objective'],policy_cost=cutoff['objective'],tail_cost=metrics['objective']-cutoff['objective'],physical_ticks=60,
            availability=1-cutoff['unavailable_ticks']/192,terminal_pending=pending,policy_failures=cutoff['failures'],normalized_cost=metrics['objective']/4,
            normalized_pending=pending/4,normalized_failures=cutoff['failures']/4,same_state_mismatches=mismatches,expected_latent_rejections=latent,
            executed_rejections=rejections,diagnostic_query_mean_ms=float(np.mean(times))))
    return episodes


def summarize(rows,h):
    means={mode:{a:{k:float(np.mean([r[k] for r in rows if r['method']==a and r['mode']==mode])) for k in ('normalized_cost','availability','normalized_pending','normalized_failures')} for a in x.ARMS} for mode in ('greedy','matching_mode')}
    points,checks=h.endpoint_gates(means['matching_mode']);candidate=x.CORE_ARMS[0];own=1-means['matching_mode'][candidate]['normalized_cost']/means['greedy'][candidate]['normalized_cost'];per_replica={}
    for t in h.TRAINS:
        costs={mode:float(np.mean([r['normalized_cost'] for r in rows if r['method']==candidate and r['train_seed']==t and r['mode']==mode])) for mode in means}
        per_replica[str(t)]=dict(**costs,gain=1-costs['matching_mode']/costs['greedy'])
    greedy=[r for r in rows if r['method']==candidate and r['mode']=='greedy'];mismatch=sum(r['same_state_mismatches'] for r in greedy)/(48*len(greedy));invalid=sum(r['invalid_assignments'] for r in rows)
    screen=dict(own_cost_gain_at_least_half_percent=own>=.005,every_replica_improves=all(r['gain']>0 for r in per_replica.values()),primary_mismatch_at_least_one_percent=mismatch>=.01,all13_mode_point_gates=all(checks.values()),zero_invalid=invalid==0)
    return dict(classification='FROZEN_DIAGNOSTIC_DEV_ONLY',means=means,mode_endpoints=points,mode_checks=checks,candidate_own_gain=own,candidate_per_replica=per_replica,
        primary_greedy_state_mismatch_rate=mismatch,screen_checks=screen,future_polynomial_decoder_trial_motivated=all(screen.values()),goal_pass=False,test_opened=False,
        limitation='No CI or source-verdict change. Dense N4 mode is a reference, not scalable deployment. Query timing includes both diagnostic queries; representative-mode zero conflicts are not learning evidence.')


def run(source,verification,output):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(1)
    manifest=dict(protocol=VERSION,status='RUNNING',source=str(source),verification=str(verification),source_manifest_sha256=x.sha(source/'manifest.json'),choices_sha256=x.sha(source/'checkpoint_choices.json'),verification_manifest_sha256=x.sha(verification/'manifest.json'),goal_pass=False,test_opened=False,teacher_assistance=False,updates=0,
        git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())))
    x.write_json(out/'manifest.json',manifest)
    def guard():check(time.monotonic()-start<900,'15minute diagnostic hardstop')
    def bindings():
        check(x.sha(source/'manifest.json')==manifest['source_manifest_sha256'] and x.sha(verification/'manifest.json')==manifest['verification_manifest_sha256'],'source manifests unchanged')
        for name,value in read(source/'manifest.json')['source_seal'].items():
            if name.startswith('source_snapshot/'):check(x.sha(Path(x.__file__).parent/Path(name).name)==value,'frozen learning source unchanged')
        for folder,key in ((source,'output_bindings'),(verification,'outputs')):
            for name,meta in read(folder/'manifest.json')[key].items():guard();check(x.sha(folder/name)==meta['sha256'] and (folder/name).stat().st_size==meta['bytes'],'input binding '+name)
    try:
        m=read(source/'manifest.json');v=read(verification/'verification.json');check(not manifest['git']['dirty'] and m['status']=='COMPLETED' and m['audit_status']=='PASS' and m['git']['commit']==SOURCE_COMMIT and not m['git']['dirty'] and m['protocol']==x.VERSION and not m['development_screen_pass'] and not m['test_opened'],'frozen rejected source')
        check(read(verification/'manifest.json')['status']=='COMPLETED' and v['status']=='PASS' and v['source']==str(source) and v['source_manifest_sha256']==manifest['source_manifest_sha256'] and not v['original_development_pass'],'independent source verification')
        check(not any('from multiprocessing.spawn import spawn_main' in line for line in subprocess.check_output(['ps','-eo','args'],text=True).splitlines()),'no active training workers')
        bindings();audit=seed_audit();seeds=audit['physical_seeds'];x.write_json(out/'seed_audit.json',audit);shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'seed_registry.json');shutil.copy2(__file__,out/'diagnostic.py');shutil.copy2(ROOT/'scripts/verify_broker_marginal_development.py',out/'reference_helpers.py')
        seals={str(p.relative_to(out)):x.sha(p) for p in out.iterdir() if p.name in ('protocol.md','seed_registry.json','diagnostic.py','reference_helpers.py')};h=helpers();choices=read(source/'checkpoint_choices.json');c=x.settings();rows=[];checkpoints=[];population={}
        journals={key:x.Journal(out/f'{key}.csv') for key in ('physical','policy')}
        try:
            for arm in tqdm(x.ARMS,desc='8 frozen methods, both execution modes'):
                for t in h.TRAINS:
                    choice=choices[f'{arm}:{t}'];checkpoints.append(dict(method=arm,train_seed=t,path=choice['path'],sha256=choice['sha256'],steps=choice['steps']));model=x.load_model(source/choice['path'],arm,c,t,choice['steps']);before=x.learner.parameter_digest(model)
                    for mode in ('greedy','matching_mode'):
                        result=collect(model,arm,t,seeds,mode,h,journals['physical'],journals['policy'],guard);rows+=result
                        for r in result:check(population.setdefault(r['seed'],r['population_sha256'])==r['population_sha256'],'all-method/mode paired tape')
                        if arm in ('ppo','normalized_ppo') and mode=='matching_mode':
                            old=[r for r in rows if r['method']==arm and r['train_seed']==t and r['mode']=='greedy']
                            check(all(all(a[k]==b[k] for k in ('normalized_cost','availability','normalized_pending','normalized_failures')) for a,b in zip(old,result)),'joint PPO trajectories/outcomes mode equality')
                    check(x.learner.parameter_digest(model)==before,'no frozen weight updates')
        finally:
            for j in journals.values():j.close()
        expected={(a,t,s,mode) for a in x.ARMS for t in h.TRAINS for s in seeds for mode in ('greedy','matching_mode')};check(len(rows)==3072 and {(r['method'],r['train_seed'],r['seed'],r['mode']) for r in rows}==expected,'full paired3072grid')
        physical_counts=defaultdict(int);policy_counts=defaultdict(int);joint_traces={}
        for name,counts,limit in (('physical',physical_counts,60),('policy',policy_counts,48)):
            with (out/f'{name}.csv').open() as f:
                for record in csv.DictReader(f):
                    key=record['method'],int(record['train_seed']),int(record['seed']),record['mode'];tick=int(record['tick'])
                    check(key in expected and tick==counts[key] and tick<limit,'complete ordered diagnostic trace');counts[key]+=1
                    if name=='physical' and key[0] in ('ppo','normalized_ppo'):
                        pairkey=key[:3]+(tick,);values=tuple(record[field] for field in ('state','pairs','events','following','cost'))
                        if key[3]=='greedy':joint_traces[pairkey]=values
                        else:check(joint_traces[pairkey]==values,'joint PPO full trajectory equality')
            check(set(counts)==expected and all(v==limit for v in counts.values()),'all trace conditions present')
        x.write_csv(out/'episodes.csv',rows);x.write_json(out/'checkpoint_bindings.json',checkpoints);summary=summarize(rows,h);x.write_json(out/'summary.json',summary)
        bindings();check(all(x.sha(out/name)==value for name,value in seals.items()) and x.sha(Path(__file__))==seals['diagnostic.py'] and x.sha(PLAN)==seals['protocol.md'] and x.sha(REGISTRY)==seals['seed_registry.json'] and x.sha(ROOT/'scripts/verify_broker_marginal_development.py')==seals['reference_helpers.py'],'diagnostic code/protocol unchanged');manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,episodes=3072,physical_transitions=184320,policy_decisions=147456,
            future_polynomial_decoder_trial_motivated=summary['future_polynomial_decoder_trial_motivated'],output_bindings={str(p.relative_to(out)):dict(sha256=x.sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})
        x.write_json(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),status='PASS',screen=summary['screen_checks'])));return summary
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);x.write_json(out/'manifest.json',manifest);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.source,args.verification,args.output)
