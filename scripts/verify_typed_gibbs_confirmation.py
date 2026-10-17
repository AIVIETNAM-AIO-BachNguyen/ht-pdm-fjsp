"""Independent complete N4 typed-Gibbs confirmation, freeze and crossed-CI audit."""
import argparse,hashlib,json,shutil,subprocess,time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
import typed_gibbs_confirmation_reference as ref

ROOT=Path(__file__).resolve().parents[1]
COMMIT='ed1002c497c85158eb243a69780a8906e68aab13'
VERSION='technician_typed_gibbs_confirmation_v1'
TRAINS=tuple(range(380300000,380300005));DEV=tuple(range(380400000,380400064));TEST=tuple(range(380500000,380500256));STEPS=(0,122880,245760,368640,491520)


def grid(c):
    return {(a,t) for a in ref.ARMS for t in c['train_seeds']}


def budgets(source,c,choices,h,guard=lambda:None):
    """Dynamic dimensions allow engineering fixtures; run() locks the full panel."""
    expected=grid(c);fits=len(expected);steps=c['env_steps'];episodes=fits*steps//48;updates=steps//c['rollout'];opts=updates*c['epochs']*(c['rollout']//c['minibatch']);audits=h.read(source/'training_audit.json');counts=h.read(source/'parameter_counts.json');initials=h.read(source/'initial_parameter_hashes.json');critics=h.read(source/'initial_critic_tensor_hashes.json')
    h.check(len(audits)==fits and {(r['algorithm'],r['train_seed']) for r in audits}==expected,'full learning grid')
    h.check(counts=={**{a:18883 for a in ref.CORE},'ppo':18966,'normalized_ppo':18966,'mappo':15364,'full_mappo':18052},'declared same capacities')
    for r in tqdm(audits,desc='Independent fresh constructor tensors'):
        guard();a,t=r['algorithm'],r['train_seed'];h.check(r['from_scratch'] and r['pretraining_steps']==0 and r['learned_steps']==steps and r['training_episodes']==steps//48 and r['physical_ticks']==steps//48*60 and r['optimizer_steps']==opts and r['updates']==updates and r['behavior_normalized_max_error']<=1 and r['behavior_baselines_cached_before_current_batch_fit'],'scratch budgets/behavior cache')
        initial=torch.load(source/'N4'/a/str(t)/'checkpoints/step_000000000.pt',map_location='cpu',weights_only=True);manual=ref.independent_initial(a,t,h);h.check(list(manual)==list(initial['state_dict']) and all(torch.equal(manual[k],initial['state_dict'][k]) for k in manual),'manual fresh initialization')
        h.check(all(initial[k]==v for k,v in dict(protocol=ref.FAMILY,algorithm=a,train_seed=t,steps=0).items()),'family checkpoint initialization metadata');digest=hashlib.sha256();cdigest=hashlib.sha256()
        for name,v in manual.items():
            digest.update(name.encode());digest.update(v.numpy().tobytes())
            if name.startswith('critic.'):cdigest.update(str(tuple(v.shape)).encode());cdigest.update(v.numpy().tobytes())
        h.check(sum(v.numel() for v in manual.values())==counts[a] and digest.hexdigest()==initial['parameter_sha256']==initials[str(t)][a] and cdigest.hexdigest()==critics[str(t)][a],'all initial digests/parameter capacities')
        choice=choices[f'{a}:{t}'];h.check(choice['steps']==steps and choice['path']==f'N4/{a}/{t}/checkpoints/step_{steps:09d}.pt','fixed final never selected from DEV');h.weights(source,choice);h.check(choice['parameter_sha256']==r['parameter_sha256'],'final optimizer/parameter binding')
    for t in c['train_seeds']:h.check(len({initials[str(t)][a] for a in ref.CORE})==1 and len(set(critics[str(t)].values()))==1,'paired fresh core and state critic')
    progress=defaultdict(list)
    for row in h.rows(source/'training_progress.csv'):progress[row['algorithm'],int(row['train_seed'])].append(row)
    h.check(set(progress)==expected,'complete progress identities')
    for identity,rows in progress.items():
        h.check([int(r['learned_steps']) for r in rows]==list(range(c['rollout'],steps+1,c['rollout'])),'all rollout updates')
        h.check(all(int(r['optimizer_steps'])==int(r['learned_steps'])//c['rollout']*c['epochs']*(c['rollout']//c['minibatch']) and float(r['behavior_normalized_max_error'])<=1 and np.isfinite(float(r['loss'])) and np.isfinite(float(r['preclip_global_gradient_norm'])) for r in rows),'finite complete optimizer progress')
        h.check(all(sum(float(r[k]) for r in rows)>0 for k in ('actor_update_norm','critic_update_norm')),'both networks learned')
    seen=set();tapes={};workers=c['vector_envs'];per_worker=steps//(48*workers)
    for row in tqdm(h.rows(source/'training_episodes.csv'),total=episodes,desc='Independent paired training populations'):
        if len(seen)%5000==0:guard()
        a,t,w,e=row['algorithm'],int(row['train_seed']),int(row['worker']),int(row['episode_index']);identity=(a,t,w,e);h.check(identity not in seen and (a,t) in expected and 0<=w<workers and 0<=e<per_worker,'unique complete episode identities');seen.add(identity);key=t,w,e;seed=h.seed(t,f'{ref.FAMILY}:N4:training_physical:worker:{w}:episode:{e}')
        if key not in tapes:tapes[key]=h.initial_events(seed)[2]
        h.check(int(row['physical_seed'])==seed and row['population_sha256']==tapes[key] and int(row['physical_ticks'])==60 and float(row['invalid_assignments'])==0 and abs(float(row['episode_reward'])*20+float(row['extended_cost']))<1e-5,'actual shared physical tapes and reward')
    h.check(len(seen)==episodes and len(tapes)==len(c['train_seeds'])*steps//48,'all training episodes/tapes');dev=list(h.rows(source/'development_episodes.csv'));wanted={(a,t,s,step) for a,t in expected for s in c['development_seeds'] for step in (0,*c['checkpoints'])}
    h.check(len(dev)==len(wanted) and {(r['method'],int(r['train_seed']),int(r['seed']),int(r['checkpoint_steps'])) for r in dev}==wanted,'complete CFDEV grid not selected')
    result=dict(models=fits,training_decisions=fits*steps,training_episodes=episodes,adam_updates=fits*opts,paired_training_tapes=len(tapes),development_nodes=len(dev));saved=h.read(source/'training_grid_audit.json')
    h.check(saved['status']=='PASS','source training grid audit')
    for name,value in result.items():
        if name!='adam_updates':h.check(saved[name]==value,'source training audit '+name)
    return result


def freeze(source,c,choices,h,manifest=None):
    frozen=h.read(source/'models_frozen.json');gate=h.read(source/'test_gate.json');expected={f'{a}:{t}' for a,t in grid(c)}
    h.check(set(choices)==set(frozen['models'])==expected and frozen['model_count']==len(expected) and frozen['fixed_steps']==c['env_steps'] and frozen['train_seeds']==c['train_seeds'],'all frozen final checkpoints')
    h.check(frozen['choices_sha256']==gate['choices_sha256']==h.sha(source/'checkpoint_choices.json') and gate['models_frozen_sha256']==h.sha(source/'models_frozen.json') and gate['training_grid_audit_sha256']==h.sha(source/'training_grid_audit.json') and gate['test_seeds']==c['test_seeds'] and gate['test_action_updates']==0,'phase seal binding')
    h.check(datetime.fromisoformat(frozen['frozen_utc'])<=datetime.fromisoformat(gate['opened_utc']),'freeze before open')
    if manifest is not None:h.check(manifest['model_seal_sha256']==gate['models_frozen_sha256'] and manifest['test_gate_sha256']==h.sha(source/'test_gate.json') and datetime.fromisoformat(manifest['started_utc'])<=datetime.fromisoformat(frozen['frozen_utc']),'source phase provenance')
    for key in sorted(expected):
        meta=frozen['models'][key];h.check(meta=={k:choices[key][k] for k in ('path','sha256','parameter_sha256','steps')} and meta['steps']==c['env_steps'] and h.sha(source/meta['path'])==meta['sha256'],'every final model immutable')
    return dict(status='PASS',model_count=len(expected),models_frozen_sha256=h.sha(source/'models_frozen.json'),test_gate_sha256=h.sha(source/'test_gate.json'),limitation='Frozen control flow, guard tests, hashes and recorded phase times checked; no independent hardware time attestation.')


def replay(source,c,choices,h,guard=lambda:None):
    rows=list(h.rows(source/'selected_test_episodes.csv'));lookup={(r['method'],int(r['train_seed']),int(r['seed'])):r for r in rows};wanted={(a,t,s) for a,t in grid(c) for s in c['test_seeds']}|{('deadline',-1,s) for s in c['test_seeds']};h.check(len(rows)==len(wanted) and set(lookup)==wanted,'all fixed-final TEST episodes')
    coords=iter(h.rows(source/'coordination.csv'));seen=set();populations={};count=policy=0;active=None;ep=None;actor_group=None;actor_data=[];neural={}
    def finish_episode():
        if ep is None:return
        r=lookup[active];h.check(ep['tick']==60 and ep['pop']==r['population_sha256'],'complete physical episode')
        for name,value in ep['metrics'].items():h.near(value,r[name],'episode '+name)
        cut=ep['cutoff'];computed=dict(availability=1-cut['unavailable_ticks']/192,policy_failures=cut['failures'],normalized_failures=cut['failures']/4,terminal_pending=ep['pending'],normalized_pending=ep['pending']/4,normalized_cost=ep['metrics']['objective']/4,policy_cost=cut['objective'],tail_cost=ep['metrics']['objective']-cut['objective'],extended_cost=ep['metrics']['objective'],technician_utilization=cut['service_ticks']/96,policy_overdue_ticks=cut['overdue_waiting_ticks'],broker_conflicts=ep['conflicts'],broker_rejected_proposals=ep['rejected'])
        for name,value in computed.items():h.near(value,r[name],'H48/H60 '+name)
    def finish_actor():
        if actor_group is None or actor_group[0]=='deadline':return
        a,t=actor_group;h.check(len(actor_data)==48*len(c['test_seeds']),'every model decision');neural[f'{a}:{t}']=ref.neural(actor_data,a,t,source,choices,c,h,guard)
    for row in tqdm(h.rows(source/'decisions.csv'),total=len(wanted)*60,desc='Independent complete TEST physics/actors'):
        if count%1000==0:guard()
        label=row['method'],int(row['train_seed']),int(row['seed']);tick=int(row['tick']);state=json.loads(row['state']);pairs=tuple(map(tuple,json.loads(row['action'])));h.check(label in wanted,'registered TEST condition')
        if label!=active:
            finish_episode();h.check(label not in seen and tick==0,'unique chronological episode');seen.add(label);active=label
            if label[:2]!=actor_group:finish_actor();actor_group=label[:2];actor_data=[]
            if label[2] not in populations:populations[label[2]]=h.initial_events(h.seed(label[2],f'{ref.FAMILY}:N4:physical'))
            initial,events,pop=populations[label[2]];ep=dict(state=initial,events=events,pop=pop,tick=0,metrics=defaultdict(float),conflicts=0,rejected=0)
        h.check(ep['tick']==tick and ep['state']==state and ep['events'][tick]==json.loads(row['events']),'independent seeded state/event chain');h.check(len({m for m,j in pairs})==len(pairs)==len({j for m,j in pairs}) and all(h.support(state)[m,j] for m,j in pairs),'physical matching support')
        if tick<48:
            cr=next(coords);identity=(cr['method'],int(cr['train_seed']),int(cr['seed']),int(cr['tick']));h.check(identity==label+(tick,),'complete deterministic policy/physics join');o=h.observation(state,tick);h.check(hashlib.sha256(o['state'].tobytes()).hexdigest()==cr['observation_sha256'] and tuple(map(tuple,json.loads(cr['pairs'])))==pairs,'raw observation join');submitted=json.loads(cr['submitted']);a=label[0]
            if a in ref.CORE:expected,diag=__import__('typed_gibbs_reference').decode(np.asarray(submitted),o['pair_mask']);h.check(expected==pairs,'canonical typed matching')
            else:
                if a!='deadline':h.verify_baseline(submitted,o['pair_mask'],a,state,pairs)
                requests=[(m,int(j)-1) for m,j in enumerate(submitted) if j] if a in ('mappo','full_mappo') else [];sizes=[sum(j==worker for m,j in requests) for worker in range(2)];diag=dict(conflicts=sum(v>1 for v in sizes),rejected_proposals=sum(max(0,v-1) for v in sizes),executed_starts=len(pairs))
            h.check(diag==json.loads(cr['diagnostics']),'independent conflicts/rejections/starts');ep['conflicts']+=diag['conflicts'];ep['rejected']+=diag['rejected_proposals']
            if a!='deadline':actor_data.append((o,submitted,json.loads(cr['policy'])))
            policy+=1
        if tick>=48 or label[0]=='deadline':
            score=np.array([[h.urgency(state,m)+h.R[m,j]/h.D[m,j] for j in range(2)] for m in range(4)]);h.near(sum(score[m,j] for m,j in pairs),h.top_two_optimum(score,h.support(state)),'common deadline action quality')
        following,parts=h.primitive_step(state,pairs,ep['events'][tick]);h.check(following==json.loads(row['following']),'independent primitive transition');h.near(parts['objective'],row['cost'],'primitive cost')
        for name,value in parts.items():ep['metrics'][name]+=value
        ep['state']=following;ep['tick']+=1;count+=1
        if tick==47:ep['cutoff']=dict(ep['metrics']);ep['pending']=sum(following['failed'][m] and m not in following['assigned'] for m in range(4))
    finish_episode();finish_actor();h.check(seen==wanted and count==len(wanted)*60 and policy==len(wanted)*48 and next(coords,None) is None,'complete physical/policy grids');h.check(set(neural)=={f'{a}:{t}' for a,t in grid(c)},'all manual actors checked')
    return rows,dict(episodes=len(rows),physical_transitions=count,policy_decisions=policy,neural_checks=neural,populations=len(populations))


def statistical_audit(source,c,rows,h,guard=lambda:None,draws=200000):
    lookup={(r['method'],int(r['train_seed']),int(r['seed'])):r for r in rows};values=np.array([[[[float(lookup[a,t,s][metric]) for metric in ref.FIELDS] for s in c['test_seeds']] for t in c['train_seeds']] for a in ref.ARMS]);saved=np.load(source/'test_metric_array.npy',allow_pickle=False);h.check(values.shape==(7,5,256,4),'full metric array');np.testing.assert_array_equal(saved,values)
    with np.load(source/'bootstrap_indices.npz',allow_pickle=False) as f:ti=f['train_indices'];si=f['seed_indices']
    with np.load(source/'bootstrap_endpoints.npz',allow_pickle=False) as f:actual=f['endpoints']
    h.check(ti.shape==(draws,5) and si.shape==(draws,256) and ti.dtype==si.dtype==np.uint8 and actual.shape==(draws,12) and np.isfinite(actual).all(),'complete bootstrap artifacts');expected=np.empty_like(actual);maximum=0.
    for start,t,s in tqdm(ref.regenerate_indices(h,c['analysis_seed'],5,256,draws=draws),total=(draws+255)//256,desc='Independent RNG and explicit Cartesian bootstrap'):
        guard();stop=start+len(t);np.testing.assert_array_equal(ti[start:stop],t);np.testing.assert_array_equal(si[start:stop],s)
        for begin in range(0,len(t),32):
            end=min(begin+32,len(t));part=ref.effects(ref.cartesian_means(values,t[begin:end],s[begin:end]));expected[start+begin:start+end]=part;delta=np.abs(part-actual[start+begin:start+end]);maximum=max(maximum,float(delta.max()));np.testing.assert_allclose(part,actual[start+begin:start+end],atol=1e-10,rtol=0)
    points=ref.effects(values.mean(axis=(1,2))[None])[0];endpoints=ref.bounds(points,expected);saved_bounds=h.read(source/'confidence_endpoints.json');h.check(set(saved_bounds)==set(endpoints),'twelve directional bounds')
    for name,row in endpoints.items():
        for key,value in row.items():
            if isinstance(value,(bool,str)):h.check(saved_bounds[name][key]==value,'CI '+key)
            else:h.near(saved_bounds[name][key],value,'CI '+key)
    means={a:{field:float(np.mean([float(r[field]) for r in rows if r['method']==a])) for field in ref.FIELDS} for a in (*ref.ARMS,'deadline')};per={str(t):1-float(np.mean([float(lookup[ref.CORE[0],t,s]['normalized_cost']) for s in c['test_seeds']]))/float(np.mean([float(lookup[ref.CORE[1],t,s]['normalized_cost']) for s in c['test_seeds']])) for t in c['train_seeds']}
    return dict(endpoints=endpoints,points=dict(zip(ref.NAMES,points.tolist())),means=means,per_replica_main_gain=per,bootstrap=dict(draws=draws,indices_regenerated=True,all_endpoint_values_checked=draws*12,max_absolute_error=maximum,quantile_method='linear',one_sided_alpha=ref.ALPHA,empirical_tail_rank=draws*ref.ALPHA,limitation='Empirical crossed bootstrap with five learning replicas; not distribution-free exact coverage.'))


def timing(source,c,choices,h):
    rows=list(h.rows(source/'isolated_timing_samples.csv'));latency=h.read(source/'latency.json');wanted={(t,r,s) for t in c['train_seeds'] for r in range(3) for s in range(48)};h.check(len(rows)==len(wanted) and {(int(r['train_seed']),int(r['repeat']),int(r['state_index'])) for r in rows}==wanted and latency['samples']==len(wanted) and not latency['training_workers_active'] and latency['torch_threads']==1 and latency['choices_sha256']==h.sha(source/'checkpoint_choices.json'),'complete isolated latency conditions');v=np.array([float(r['milliseconds']) for r in rows]);h.check(np.isfinite(v).all() and (v>=0).all(),'finite raw timings')
    for name,value in dict(mean_ms=np.mean(v),p95_ms=np.percentile(v,95),p99_ms=np.percentile(v,99)).items():h.near(latency[name],value,'latency '+name)
    for t in c['train_seeds']:
        meta=choices[f'{ref.CORE[0]}:{t}']['observations'];h.check(meta['states']==48 and meta['seed']==c['development_seeds'][0] and h.sha(source/meta['path'])==meta['sha256'],'latency CFDEV observation binding')
    return latency


def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);h=ref.helpers();torch.set_num_threads(1);start=time.monotonic();manifest=dict(status='RUNNING',source=str(source),source_manifest_sha256=h.sha(source/'manifest.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),goal_pass=False,test_opened=True)
    def write(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def guard():h.check(time.monotonic()-start<3600,'one-hour independent audit hardstop')
    def bindings():
        h.check(h.sha(source/'manifest.json')==manifest['source_manifest_sha256'],'source manifest unchanged')
        for name,meta in h.read(source/'manifest.json')['output_bindings'].items():guard();h.check(h.sha(source/name)==meta['sha256'] and (source/name).stat().st_size==meta['bytes'],'source output binding '+name)
    write()
    try:
        h.check(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip() and manifest['git_commit']==subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=ROOT,text=True).strip(),'clean pushed verifier source');m=h.read(source/'manifest.json');h.check(m['status']=='COMPLETED' and m['audit_status']=='PASS' and m['git']['commit']==COMMIT and not m['git']['dirty'] and m['protocol']==VERSION and m['family']==ref.FAMILY and m['test_opened'] and not m['goal_pass'],'terminal registered source confirmation');bindings()
        for name,digest in m['source_seal'].items():
            h.check(h.sha(source/name)==digest,'source snapshot seal')
            if name.startswith('source_snapshot/'):
                blob=subprocess.check_output(['git','show',f'{COMMIT}:src/ht_pdm_fjsp/{Path(name).name}'],cwd=ROOT);h.check(hashlib.sha256(blob).hexdigest()==digest,'frozen source Git blob');h.check(h.sha(ROOT/'src/ht_pdm_fjsp'/Path(name).name)==digest,'actual evaluation source unchanged')
        for name,git_path in (('protocol.md',f'docs/{VERSION}_plan.md'),('seed_registry.json',f'configs/{ref.FAMILY}_seed_registry.json')):h.check(hashlib.sha256(subprocess.check_output(['git','show',f'{COMMIT}:{git_path}'],cwd=ROOT)).hexdigest()==h.sha(source/name),'prospective protocol/registry')
        parent=h.read(source/'parent_bindings.json');pm=h.read(Path(parent['source'])/'manifest.json');vm=h.read(Path(parent['verification'])/'manifest.json');v=h.read(Path(parent['verification'])/'verification.json');h.check(h.sha(Path(parent['source'])/'manifest.json')==parent['source_manifest_sha256'] and h.sha(Path(parent['verification'])/'manifest.json')==parent['verification_manifest_sha256'] and pm['development_screen_pass'] and v['original_development_pass'] and v['status']=='PASS' and vm['status']=='COMPLETED','independent DEV entry gate')
        for base,entries in ((Path(parent['source']),pm['output_bindings']),(Path(parent['verification']),vm['outputs'])):
            for name,meta in entries.items():guard();h.check(h.sha(base/name)==meta['sha256'] and (base/name).stat().st_size==meta['bytes'],'parent unchanged output binding')
        for name,digest in pm['source_seal'].items():
            if name.startswith('source_snapshot/'):h.check(m['source_seal'][name]==digest,'unchanged original algorithm family')
        physics=dict(machines=4,technicians=2,horizon=60,failure_age=4,failure_probability=.45,service_time=h.D.tolist(),restoration=h.R.tolist(),max_age=8,preventive_cost=1.,corrective_cost=2.,unavailable_cost=12.,failure_cost=15.,waiting_limit=4,waiting_price=12.);h.check(h.read(source/'physics.json')==physics,'unchanged independently specified physics');seed=ref.seed_contract(source,h);c=h.read(source/'resolved_config.json')
        expected=dict(env_steps=491520,vector_envs=16,rollout=3072,epochs=4,minibatch=256,learning_rate=.0003,clip=.2,entropy_weight=.01,value_weight=.5,gradient_clip=.5,teacher_assistance=False,pretraining_steps=0,training_processes=16,profile='confirmation',analysis_seed=380600000)
        for name,value in expected.items():h.check(c[name]==value,'registered configuration '+name)
        h.check(c['train_seeds']==list(TRAINS) and c['development_seeds']==list(DEV) and c['test_seeds']==list(TEST) and c['checkpoints']==list(STEPS[1:]) and c['confidence']==1-ref.ALPHA,'locked complete panel and cadence');choices=h.read(source/'checkpoint_choices.json');budget=budgets(source,c,choices,h,guard);seal=freeze(source,c,choices,h,m);latency=timing(source,c,choices,h);rows,physical=replay(source,c,choices,h,guard);stats=statistical_audit(source,c,rows,h,guard);summary=h.read(source/'summary.json');checks={name:r['passed'] for name,r in stats['endpoints'].items()};extra=dict(every_replica_main_gain_positive=all(v>0 for v in stats['per_replica_main_gain'].values()),zero_invalid=True,isolated_p95_at_most_50ms=latency['p95_ms']<=50);passed=all(checks.values()) and all(extra.values());superiority=bool(passed and all(stats['endpoints']['gain_vs_'+a]['bound']>=.01 for a in ref.REFS))
        _,point_checks=ref.point_gates(stats['means'],h);h.check(point_checks==summary['point_checks'],'all original descriptive point gates')
        h.check(checks==summary['checks'] and extra==summary['extra_checks'] and passed==summary['confirmation_screen_pass']==m['confirmation_screen_pass'] and superiority==summary['ppo_superiority_supported'] and summary['test_opened'] and not summary['goal_pass'] and summary['independent_verification_pending'],'unchanged original scientific verdict')
        for a,values in stats['means'].items():
            for name,value in values.items():h.near(value,summary['means'][a][name],'TEST mean '+name)
        for name,value in stats['points'].items():h.near(value,summary['endpoints'][name],'TEST point')
        for name,value in stats['per_replica_main_gain'].items():h.near(value,summary['per_replica_main_gain'][name],'replica primary sign')
        h.check(summary['confidence_endpoints']==h.read(source/'confidence_endpoints.json'),'same source confidence record')
        bootstrap_meta=summary['bootstrap'];h.check(bootstrap_meta==dict(draws=200000,chunk=256,seed=h.seed(380600000,f'maintenance_balanced_algorithm_benchmark_v1:{ref.FAMILY}:bootstrap'),bit_generator='PCG64',paired=True,crossed=True,quantile_method='linear',sequential_round=7,family_alpha=.05/(7*8),one_sided_endpoint_alpha=ref.ALPHA,training_replicas=5,physical_seeds=256),'registered bootstrap metadata');bindings();freeze(source,c,choices,h,m)
        result=dict(status='PASS',source=str(source),source_manifest_sha256=manifest['source_manifest_sha256'],classification='INDEPENDENT_SEALED_TEST_CONFIRMATION_AUDIT',seed_contract=seed,budgets=budget,model_freeze=seal,test_physics=physical,statistics=stats,latency=latency,independent_checks=checks,extra_checks=extra,original_confirmation_pass=passed,ppo_superiority_supported=superiority,goal_pass=False,test_opened=True,limitation='Full recorded physics/actors/partition densities/bootstrap verified. Historical optimizer minibatch gradients were not retained, not replayed. Empirical five-replica uncertainty is not exact coverage. No scalability/generalization evidence here.')
        (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        for name in ('verify_typed_gibbs_confirmation.py','typed_gibbs_confirmation_reference.py','typed_gibbs_reference.py','verify_typed_gibbs_development.py','verify_broker_marginal_development.py'):shutil.copy2(ROOT/'scripts'/name,out/name)
        shutil.copy2(ROOT/'docs/technician_typed_gibbs_confirmation_verification_v1_plan.md',out/'protocol.md');manifest.update(status='COMPLETED',elapsed_seconds=time.monotonic()-start,original_confirmation_pass=passed,outputs={p.name:dict(sha256=h.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'});write();print(json.dumps(dict(output=str(out),status='PASS',original_confirmation_pass=passed,goal_pass=False)));return result
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);write();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();run(args.source,args.output)
