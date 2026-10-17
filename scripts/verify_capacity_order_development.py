"""Independent fresh tensors, budgets, physics and enumerated matching-count prior."""
import argparse,csv,hashlib,importlib.util,itertools,json,shutil,subprocess,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1];VERSION='technician_capacity_order_ppo_v1';COMMIT='06c1638ed9beafe248c73f551eb28ff1bf8cc007'
CORE=('capacity_order_ppo','fixed_order_ppo','reverse_capacity_ppo','mask_only_order_ppo','independent_machine_q_ppo','independent_machine_p_ppo');ARMS=(*CORE,'raw_tech_ppo','ppo','normalized_ppo','mappo','full_mappo');REFS=('ppo','normalized_ppo');TRAINS=tuple(range(360000000,360000003));DEV=tuple(range(360100000,360100032));STEPS=(0,122880,245760,368640,491520)


def helpers():
    spec=importlib.util.spec_from_file_location('removal_reference',ROOT/'scripts/verify_broker_marginal_development.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);return h


def independent_initial(arm,train,h):
    from torch import nn
    def net(i,o):return nn.Sequential(nn.Linear(i,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,o))
    role='graph' if arm in CORE else 'ppo' if arm in REFS else arm
    with torch.random.fork_rng():
        torch.manual_seed(h.seed(train,f'maintenance_balanced_algorithm_benchmark_v1:{VERSION}:initialization:{role}'));m=nn.Module();m.actor=net(104,3) if arm in CORE else net(103,5) if arm=='raw_tech_ppo' else net(71,21) if arm in REFS else net(33,3)
        if arm in CORE:
            with torch.no_grad():m.actor[4].weight.zero_();m.actor[4].bias.zero_()
        if arm=='full_mappo':net(71,1);m.actor=net(75,3) # consume original constructor's dummy critic
    with torch.random.fork_rng():torch.manual_seed(h.seed(train,f'maintenance_balanced_algorithm_benchmark_v1:{VERSION}:initialization:critic_state'));state=net(71,1)
    if arm in CORE or arm in ('ppo','raw_tech_ppo'):m.critic=state
    else:m.critic=nn.Module();m.critic.net=state
    return m.state_dict()


def check_neural_group(data,arm,train,source,choices,c,h):
    from capacity_order_reference import verify_neural
    return verify_neural(data,arm,train,source,choices,c,h)


def endpoint_gates(means,h):
    p=means[CORE[0]];points={};checks={}
    for a in ARMS[1:]:
        name='gain_vs_'+a;v=1-p['normalized_cost']/means[a]['normalized_cost'];points[name]=v;checks[name]=v>=.005 if a in (CORE[1],'independent_machine_q_ppo') else v>0 if a in ('mappo','full_mappo') else v>=-.01
    for a in REFS:
        for metric in ('availability','normalized_pending','normalized_failures'):
            name=metric+'_delta_vs_'+a;v=p[metric]-means[a][metric];points[name]=v;checks[name]=v>=-.01 if metric=='availability' else v<=.025 if metric=='normalized_pending' else v<=.25
    h.check(len(checks)==16,'16endpoints');return points,checks


def seed_contract(source,h):
    registry=h.read(source/'seed_registry.json');audit=h.read(source/'seed_audit.json');h.check(registry['version']==VERSION,'seed version');groups=[];streams=[]
    for profile,p in registry['panels'].items():
        train=p['train_seeds'];dev=list(range(p['development_start'],p['development_start']+p['development_count']));test=list(range(p.get('test_start',0),p.get('test_start',0)+p.get('test_count',0)));groups.extend([set(train),set(dev),set(test),{p['analysis_seed']}]);workers=2 if profile=='engineering' else 16;steps=192 if profile=='engineering' else 491520
        streams.append(h.seed(p['analysis_seed'],f'maintenance_balanced_algorithm_benchmark_v1:{VERSION}:bootstrap'))
        streams.extend(h.seed(s,f'{VERSION}:N4:physical') for s in (*dev,*test))
        for t in train:
            streams.extend(h.seed(t,f'maintenance_balanced_algorithm_benchmark_v1:{VERSION}:initialization:{role}') for role in ('graph','raw_tech_ppo','ppo','mappo','full_mappo','critic_state'))
            streams.extend(h.seed(t,f'maintenance_balanced_algorithm_benchmark_v1:{VERSION}:N4:{role}:{stream}') for role in ('graph','raw_tech_ppo','ppo','mappo','full_mappo') for stream in ('actions','minibatches'))
            streams.extend(h.seed(t,f'{VERSION}:N4:training_physical:worker:{w}:episode:{e}') for w in range(workers) for e in range(steps//(48*workers)))
    raw=set.union(*groups)
    def integers(x):
        if isinstance(x,dict):return set.union(set(),*(integers(v) for v in x.values()))
        if isinstance(x,list):return set.union(set(),*(integers(v) for v in x))
        return {x} if isinstance(x,int) and not isinstance(x,bool) else set()
    historical=set()
    for digest,name in audit['historical_registries'].items():
        p=Path(name);h.check(h.sha(p)==digest,'historical registry binding');historical|=integers(h.read(p))
    h.check(audit['status']=='PASS' and audit['raw_unique']==len(raw)==365 and audit['derived_unique']==len(streams)==len(set(streams))==82424 and not audit['confirmation_states_constructed'],'independent seed cardinality')
    h.check(not any(a&b for i,a in enumerate(groups) for b in groups[i+1:]) and not raw&historical and not set(streams)&(raw|historical),'independent raw/derived separation')
    return dict(raw=len(raw),derived=len(streams),historical_registries=len(audit['historical_registries']),limitation='Current streams and historical raw registries independently rebuilt; historical derived namespaces rely on source audit, not exhaustively independently reconstructed.')


def budgets(source,c,choices,h,guard):
    for name,value in dict(env_steps=491520,vector_envs=16,rollout=3072,epochs=4,minibatch=256,learning_rate=.0003,clip=.2,entropy_weight=.01,value_weight=.5,gradient_clip=.5,teacher_assistance=False,pretraining_steps=0,training_processes=16).items():h.check(c[name]==value,'config '+name)
    h.check(c['train_seeds']==list(TRAINS) and c['development_seeds']==list(DEV) and c['test_seeds']==[] and c['checkpoints']==list(STEPS[1:]) and c['confidence']==1-(.05/42)/16,'registered panels/cadence/alpha')
    expected={(a,t) for a in ARMS for t in TRAINS};audits=h.read(source/'training_audit.json');h.check(len(audits)==33 and {(r['algorithm'],r['train_seed']) for r in audits}==expected,'33fit grid');counts=h.read(source/'parameter_counts.json');h.check(counts==dict(**{a:19908 for a in CORE},raw_tech_ppo=19974,ppo=18966,normalized_ppo=18966,mappo=15364,full_mappo=18052),'all declared capacities')
    initial_hashes=h.read(source/'initial_parameter_hashes.json');critic_hashes=h.read(source/'initial_critic_tensor_hashes.json')
    for r in tqdm(audits,desc='33independent scratch initializations'):
        guard();a,t=r['algorithm'],r['train_seed'];h.check(r['learned_steps']==491520 and r['training_episodes']==10240 and r['physical_ticks']==614400 and r['optimizer_steps']==7680 and r['updates']==160 and r['from_scratch'] and r['pretraining_steps']==0 and r['behavior_normalized_max_error']<=1 and r['behavior_baselines_cached_before_current_batch_fit'],'fit budget/behavior/cache')
        initial=torch.load(source/'N4'/a/str(t)/'checkpoints/step_000000000.pt',map_location='cpu',weights_only=True);reference=independent_initial(a,t,h);h.check(list(reference)==list(initial['state_dict']) and all(torch.equal(reference[k],initial['state_dict'][k]) for k in reference),'fresh private tensors '+a);h.check(all(initial[k]==v for k,v in dict(protocol=VERSION,algorithm=a,train_seed=t,steps=0).items()),'initial metadata');digest=hashlib.sha256();critic_digest=hashlib.sha256()
        for name,value in reference.items():
            digest.update(name.encode());digest.update(value.numpy().tobytes())
            if name.startswith('critic.'):critic_digest.update(str(tuple(value.shape)).encode());critic_digest.update(value.numpy().tobytes())
        h.check(digest.hexdigest()==initial['parameter_sha256']==initial_hashes[str(t)][a] and critic_digest.hexdigest()==critic_hashes[str(t)][a] and sum(v.numel() for v in reference.values())==counts[a],'initial hash/capacity');choice=choices[f'{a}:{t}'];h.check(choice['steps']==491520 and choice['path']==f'N4/{a}/{t}/checkpoints/step_000491520.pt','fixed endpoint');h.weights(source,choice);h.check(choice['parameter_sha256']==r['parameter_sha256'],'final optimizer/checkpoint digest')
    progress=defaultdict(list)
    for r in h.rows(source/'training_progress.csv'):progress[r['algorithm'],int(r['train_seed'])].append(r)
    h.check(set(progress)==expected,'progress grid')
    for a,t in expected:
        rr=progress[a,t];h.check([int(r['learned_steps']) for r in rr]==list(range(3072,491521,3072)) and all(int(r['optimizer_steps'])==int(r['learned_steps'])//3072*48 and float(r['behavior_normalized_max_error'])<=1 for r in rr),'complete160rollout updates');h.check(all(sum(float(r[k]) for r in rr)>0 for k in ('actor_update_norm','critic_update_norm')),'both groups learned')
    seen=set();population={}
    for r in tqdm(h.rows(source/'training_episodes.csv'),total=337920,desc='Independent paired training tapes'):
        if len(seen)%5000==0:guard()
        a,t,w,e=r['algorithm'],int(r['train_seed']),int(r['worker']),int(r['episode_index']);label=(a,t,w,e);h.check(label not in seen and (a,t) in expected and 0<=w<16 and 0<=e<640,'train episode grid');seen.add(label);k=t,w,e;physical=h.seed(t,f'{VERSION}:N4:training_physical:worker:{w}:episode:{e}')
        if k not in population:population[k]=h.initial_events(physical)[2]
        h.check(int(r['physical_seed'])==physical and r['population_sha256']==population[k] and int(r['physical_ticks'])==60 and float(r['invalid_assignments'])==0 and abs(float(r['episode_reward'])*20+float(r['extended_cost']))<1e-5,'actual train tapes/reward')
    h.check(len(seen)==337920 and len(population)==30720,'complete train grid');dev=list(h.rows(source/'development_episodes.csv'));wanted={(a,t,s,step) for a in ARMS for t in TRAINS for s in DEV for step in STEPS};h.check(len(dev)==5280 and {(r['method'],int(r['train_seed']),int(r['seed']),int(r['checkpoint_steps'])) for r in dev}==wanted,'full DEV node grid')
    return dict(models=33,training_decisions=16220160,training_episodes=337920,adam_updates=253440,paired_train_tapes=30720,development_nodes=5280)


def final_replay(source,choices,c,h,guard):
    rows=list(h.rows(source/'selected_episodes.csv'));lookup={(r['method'],int(r['train_seed']),int(r['seed'])):r for r in rows};wanted={(a,t,s) for a in ARMS for t in TRAINS for s in DEV}|{('deadline',-1,s) for s in DEV};h.check(len(rows)==1088 and set(lookup)==wanted,'1088final grid');coord={}
    for r in h.rows(source/'coordination.csv'):
        key=r['method'],int(r['train_seed']),int(r['seed']),int(r['tick']);h.check(key not in coord,'unique policy record');coord[key]=r
    h.check(len(coord)==52224,'policy count');episodes={};actor=defaultdict(list);count=0
    for r in tqdm(h.rows(source/'decisions.csv'),total=65280,desc='Independent final primitive replay'):
        if count%1000==0:guard()
        label=r['method'],int(r['train_seed']),int(r['seed']);tick=int(r['tick']);s=json.loads(r['state']);pairs=tuple(map(tuple,json.loads(r['action'])));h.check(label in wanted,'final condition')
        if label not in episodes:
            physical=h.seed(label[2],f'{VERSION}:N4:physical');initial,events,pop=h.initial_events(physical);episodes[label]=dict(state=initial,events=events,pop=pop,tick=0,metrics=defaultdict(float))
        ep=episodes[label];h.check(ep['state']==s and ep['tick']==tick and json.loads(r['events'])==ep['events'][tick],'seeded state/event chain');h.check(all(h.support(s)[m,j] for m,j in pairs),'urgent/available support')
        if tick<48:
            cr=coord[label+(tick,)];o=h.observation(s,tick);h.check(hashlib.sha256(o['state'].tobytes()).hexdigest()==cr['observation_sha256'] and tuple(map(tuple,json.loads(cr['pairs'])))==pairs,'policy/physics observation join');u=json.loads(cr['submitted']);a=label[0]
            if a in CORE:
                from capacity_order_reference import decode
                expected,diag=decode(np.asarray(u),o['pair_mask'],o['edges'],a);h.check(expected==pairs and diag==json.loads(cr['diagnostics']),'machine coordination')
            elif a=='raw_tech_ppo':
                expected=tuple(sorted((int(m)-1,j) for j,m in enumerate(u) if m));h.check(expected==pairs and len({m for m,j in pairs})==len(pairs),'raw technician prefix')
            elif a!='deadline':h.verify_baseline(u,o['pair_mask'],a,s,pairs)
            if a!='deadline':actor[label[:2]].append((o,u))
        if tick>=48 or label[0]=='deadline':
            score=np.array([[h.urgency(s,m)+h.R[m,j]/h.D[m,j] for j in range(2)] for m in range(4)]);h.near(sum(score[m,j] for m,j in pairs),h.top_two_optimum(score,h.support(s)),'common deadline objective')
        following,parts=h.primitive_step(s,pairs,ep['events'][tick]);h.check(following==json.loads(r['following']),'primitive following');h.near(parts['objective'],r['cost'],'primitive cost')
        for k,v in parts.items():ep['metrics'][k]+=v
        ep['state']=following;ep['tick']+=1;count+=1
        if tick==47:ep['cutoff']=dict(ep['metrics']);ep['pending']=sum(following['failed'][m] and m not in following['assigned'] for m in range(4))
    h.check(count==65280 and set(episodes)==wanted,'complete physical replay')
    for label,ep in episodes.items():
        r=lookup[label];h.check(ep['tick']==60 and ep['pop']==r['population_sha256'],'episode pop/final tick')
        for k,v in ep['metrics'].items():h.near(v,r[k],'episode '+k)
        cut=ep['cutoff'];values=dict(availability=1-cut['unavailable_ticks']/192,policy_failures=cut['failures'],normalized_failures=cut['failures']/4,terminal_pending=ep['pending'],normalized_pending=ep['pending']/4,normalized_cost=ep['metrics']['objective']/4,policy_cost=cut['objective'],tail_cost=ep['metrics']['objective']-cut['objective'])
        for k,v in values.items():h.near(v,r[k],'H48/H60 '+k)
    checks={}
    for (a,t),data in tqdm(actor.items(),desc='33manual actors/enumerated matching counts'):guard();checks[f'{a}:{t}']=check_neural_group(data,a,t,source,choices,c,h)
    means={a:{k:float(np.mean([float(r[k]) for r in rows if r['method']==a])) for k in ('normalized_cost','availability','normalized_pending','normalized_failures')} for a in (*ARMS,'deadline')};per={str(t):1-float(np.mean([float(r['normalized_cost']) for r in rows if r['method']==CORE[0] and int(r['train_seed'])==t]))/float(np.mean([float(r['normalized_cost']) for r in rows if r['method']==CORE[1] and int(r['train_seed'])==t])) for t in TRAINS};return dict(episodes=1088,physical_transitions=count,policy_decisions=52224,neural_checks=checks,means=means,per_replica_main_gain=per)


def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);h=helpers();torch.set_num_threads(1);start=time.monotonic();manifest=dict(status='RUNNING',source=str(source),source_manifest_sha256=h.sha(source/'manifest.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),goal_pass=False,test_opened=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def guard():h.check(time.monotonic()-start<900,'15minute verification hardstop')
    def bindings():
        h.check(h.sha(source/'manifest.json')==manifest['source_manifest_sha256'],'source manifest unchanged')
        for name,meta in h.read(source/'manifest.json')['output_bindings'].items():guard();h.check(h.sha(source/name)==meta['sha256'] and (source/name).stat().st_size==meta['bytes'],'source binding '+name)
    try:
        m=h.read(source/'manifest.json');h.check(m['status']=='COMPLETED' and m['audit_status']=='PASS' and m['git']['commit']==COMMIT and not m['git']['dirty'] and m['protocol']==VERSION and not m['test_opened'],'terminal committed DEV source');bindings()
        for name,digest in m['source_seal'].items():
            h.check(h.sha(source/name)==digest,'source seal')
            if name.startswith('source_snapshot/'):
                blob=subprocess.check_output(['git','show',f'{COMMIT}:src/ht_pdm_fjsp/{Path(name).name}'],cwd=ROOT);h.check(hashlib.sha256(blob).hexdigest()==digest,'committed source blob')
        for artifact,git_path in (('protocol.md',f'docs/{VERSION}_plan.md'),('seed_registry.json',f'configs/{VERSION}_seed_registry.json')):
            blob=subprocess.check_output(['git','show',f'{COMMIT}:{git_path}'],cwd=ROOT);h.check(hashlib.sha256(blob).hexdigest()==h.sha(source/artifact),'registered committed protocol')
        physics=dict(machines=4,technicians=2,horizon=60,failure_age=4,failure_probability=.45,service_time=h.D.tolist(),restoration=h.R.tolist(),max_age=8,preventive_cost=1.,corrective_cost=2.,unavailable_cost=12.,failure_cost=15.,waiting_limit=4,waiting_price=12.);h.check(h.read(source/'physics.json')==physics,'independent primitive constants');seeds=seed_contract(source,h)
        c=h.read(source/'resolved_config.json');choices=h.read(source/'checkpoint_choices.json');budget=budgets(source,c,choices,h,guard);budget['seed_contract']=seeds;selected=final_replay(source,choices,c,h,guard);summary=h.read(source/'summary.json');points,checks=endpoint_gates(selected['means'],h)
        for key,value in points.items():h.near(value,summary['endpoints'][key],'endpoint '+key)
        h.check(checks==summary['checks'],'16checks')
        for a,values in selected['means'].items():
            for k,v in values.items():h.near(v,summary['means'][a][k],'method mean')
        for t,v in selected['per_replica_main_gain'].items():h.near(v,summary['per_replica_main_gain'][t],'per-replica gain')
        timing=list(h.rows(source/'isolated_timing_samples.csv'));latency=h.read(source/'latency.json');h.check(len(timing)==432 and {(int(r['train_seed']),int(r['repeat']),int(r['state_index'])) for r in timing}=={(t,r,i) for t in TRAINS for r in range(3) for i in range(48)},'raw432timing grid');times=np.array([float(r['milliseconds']) for r in timing]);h.check(np.isfinite(times).all() and (times>=0).all() and latency['choices_sha256']==h.sha(source/'checkpoint_choices.json'),'timing finite/binding')
        for key,value in dict(mean_ms=np.mean(times),p95_ms=np.percentile(times,95),p99_ms=np.percentile(times,99)).items():h.near(value,latency[key],'raw timing math')
        extra=dict(every_replica_main_gain_positive=all(v>0 for v in selected['per_replica_main_gain'].values()),zero_invalid=True,isolated_p95_at_most_50ms=latency['p95_ms']<=50);passed=all(checks.values()) and all(extra.values());h.check(extra==summary['extra_checks'] and passed==summary['development_screen_pass']==m['development_screen_pass'] and not summary['goal_pass'] and not summary['test_opened'],'original verdict');bindings()
        result=dict(status='PASS',classification='DEVELOPMENT_VERIFICATION_ONLY',source=str(source),source_manifest_sha256=manifest['source_manifest_sha256'],budgets=budget,selected=selected,independent_points=points,independent_checks=checks,original_development_pass=passed,primary_latency=latency,goal_pass=False,test_opened=False,limitation='All final physics/actors/count priors/independent physical densities verified; historical optimizer minibatches/cache tensors not retained, not independently replayed. No scientific CI or scaling evidence from DEV.')
        (out/'verification.json').write_text(json.dumps(result,indent=2));shutil.copy2(__file__,out/'verifier.py');shutil.copy2(ROOT/'scripts/verify_broker_marginal_development.py',out/'reference_helpers.py');shutil.copy2(ROOT/'scripts/capacity_order_reference.py',out/'capacity_reference.py');shutil.copy2(ROOT/'docs/technician_capacity_order_v1_verification_plan.md',out/'protocol.md');manifest.update(status='COMPLETED',elapsed_seconds=time.monotonic()-start,outputs={p.name:dict(sha256=h.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(output=str(out),status='PASS',development_pass=passed)))
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.source,args.output)
