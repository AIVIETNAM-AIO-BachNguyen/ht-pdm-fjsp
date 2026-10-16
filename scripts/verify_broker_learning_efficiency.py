"""Independent budgets, weighted AUC and fixed-final primitive replay."""
import argparse,csv,hashlib,importlib.util,json,math,shutil,subprocess,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1];STEPS=(0,15360,30720,61440,92160,122880,184320,245760)
TRAINS=tuple(range(310000000,310000003));DEV=tuple(range(310100000,310100032));COMMIT='8a3c1f9e77449ccb3a51c91effe6005a90c28095'
def helpers():
    spec=importlib.util.spec_from_file_location('efficiency_reference',ROOT/'scripts/verify_broker_marginal_development.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.TRAINS=TRAINS;h.DEV=DEV;return h


def auc_reference(dev,h):
    grid={}
    for r in dev:
        label=r['method'],int(r['train_seed']),int(r['seed']),int(r['checkpoint_steps']);h.check(label not in grid,'unique DEV node');grid[label]=float(r['normalized_cost'])
    expected={(a,t,s,step) for a in h.ARMS for t in TRAINS for s in DEV for step in STEPS};h.check(set(grid)==expected,'full6144node grid')
    # Each point receives half of its adjacent interval lengths. This is a
    # different calculation from the production adjacent-pair trapezoid loop.
    weights=[(STEPS[1]-STEPS[0])/2]+[(STEPS[i+1]-STEPS[i-1])/2 for i in range(1,len(STEPS)-1)]+[(STEPS[-1]-STEPS[-2])/2]
    h.near(sum(weights),STEPS[-1],'AUC integration weights');result={}
    for a in h.ARMS:
        for t in TRAINS:
            for s in DEV:result[a,t,s]=math.fsum(w*grid[a,t,s,step] for w,step in zip(weights,STEPS))/STEPS[-1]
    return result


def endpoint_reference(means,auc,h):
    average={a:float(np.mean([v for (arm,t,s),v in auc.items() if arm==a])) for a in h.ARMS};p=means[h.CORE[0]];points=dict(auc_gain_vs_broker_proposal_ppo=1-average[h.CORE[0]]/average[h.CORE[1]]);checks={k:v>=.005 for k,v in points.items()}
    for a in h.ARMS[1:]:
        name='final_gain_vs_'+a;points[name]=1-p['normalized_cost']/means[a]['normalized_cost'];checks[name]=points[name]>0 if a in ('mappo','full_mappo') else points[name]>=-.01
    for a in ('ppo','normalized_ppo'):
        for metric in ('availability','normalized_pending','normalized_failures'):
            name=metric+'_delta_vs_'+a;v=p[metric]-means[a][metric];points[name]=v;checks[name]=v>=-.01 if metric=='availability' else v<=.025 if metric=='normalized_pending' else v<=.25
    per={}
    for t in TRAINS:
        values={a:float(np.mean([v for (arm,train,s),v in auc.items() if arm==a and train==t])) for a in h.CORE[:2]};per[str(t)]=dict(candidate_auc=values[h.CORE[0]],control_auc=values[h.CORE[1]],gain=1-values[h.CORE[0]]/values[h.CORE[1]])
    h.check(len(checks)==14,'14endpoints');return average,points,checks,per


def budgets(source,c,choices,h,guard):
    for k,v in dict(env_steps=245760,vector_envs=16,rollout=3072,epochs=4,minibatch=256,hidden=64,baseline_hidden=64,gamma=1.,learning_rate=.0003,clip=.2,entropy_weight=.01,value_weight=.5,gradient_clip=.5,teacher_assistance=False,pretraining_steps=0,training_processes=8).items():h.check(c[k]==v,'locked config '+k)
    h.check(c['train_seeds']==list(TRAINS) and c['development_seeds']==list(DEV) and c['test_seeds']==[] and c['checkpoints']==list(STEPS[1:]) and c['confidence']==1-(.05/20)/14,'panels/cadence/confidence')
    audit=h.read(source/'seed_audit.json');h.check(audit['status']=='PASS' and audit['raw_unique']==365 and audit['confirmation_states_constructed'] is False and audit['actual_learning_namespace']==h.VERSION,'raw/actual-role seed audit')
    fits=h.read(source/'training_audit.json');h.check(len(fits)==24 and {(r['algorithm'],r['train_seed']) for r in fits}=={(a,t) for a in h.ARMS for t in TRAINS},'24fit grid')
    h.check(h.read(source/'parameter_counts.json')==dict(**{a:19974 for a in h.CORE},ppo=18966,normalized_ppo=18966,mappo=15364,full_mappo=18052),'declared capacities')
    h.check(h.read(source/'physics.json')==dict(machines=4,technicians=2,horizon=60,failure_age=4,failure_probability=.45,service_time=h.D.tolist(),restoration=h.R.tolist(),max_age=8,preventive_cost=1.,corrective_cost=2.,unavailable_cost=12.,failure_cost=15.,waiting_limit=4,waiting_price=12.),'unchanged physical configuration')
    common=h.read(source/'initial_critic_tensor_hashes.json')
    for a in tqdm(fits,desc='Independent24scratch models'):
        guard();arm=a['algorithm'];t=a['train_seed'];h.check(a['from_scratch'] and a['pretraining_steps']==0 and a['learned_steps']==245760 and a['training_episodes']==5120 and a['physical_ticks']==307200 and a['optimizer_steps']==3840 and a['changed'] and a['behavior_normalized_max_error']<=1,'fit budget')
        expected=h.independent_initial(arm,t);initial=torch.load(source/'N4'/arm/str(t)/'checkpoints/step_000000000.pt',map_location='cpu',weights_only=True);h.check(list(expected)==list(initial['state_dict']) and all(torch.equal(expected[k],initial['state_dict'][k]) for k in expected),'independent random tensors')
        prefix='critic' if arm in h.CORE or arm=='ppo' else 'critic.net';digest=hashlib.sha256()
        for name,v in expected.items():
            if name.startswith(prefix+'.'):digest.update(str(tuple(v.shape)).encode());digest.update(v.numpy().tobytes())
        h.check(common[str(t)][arm]==digest.hexdigest(),'common initial critic hash');choice=choices[f'{arm}:{t}'];h.check(choice['steps']==245760 and choice['path']==f'N4/{arm}/{t}/checkpoints/step_000245760.pt','fixed endpoint choice');weights=h.weights(source,choice)
        final=torch.load(source/choice['path'],map_location='cpu',weights_only=True);h.check(final['parameter_sha256']==a['parameter_sha256'] and final['protocol']==h.VERSION and final['steps']==245760 and final['algorithm']==arm and final['train_seed']==t,'source metadata/final trained digest')
    h.check(all(len(set(v.values()))==1 for v in common.values()),'same common critic across all arms')
    progress=defaultdict(list)
    for r in h.rows(source/'training_progress.csv'):progress[r['algorithm'],int(r['train_seed'])].append(r)
    for a in h.ARMS:
        for t in TRAINS:
            group=progress[a,t];h.check([int(r['learned_steps']) for r in group]==list(range(3072,245761,3072)),'80complete updates');h.check(all(int(r['optimizer_steps'])==int(r['learned_steps'])//3072*48 and float(r['behavior_normalized_max_error'])<=1 for r in group),'update budget/replay')
            if a in h.CORE:h.check(all(sum(float(r[k]) for r in group)>0 for k in ('actor_update_norm','critic_update_norm')),'actor/critic updates')
    seen=set();population={}
    for r in tqdm(h.rows(source/'training_episodes.csv'),total=122880,desc='Independent train-tape audit'):
        if len(seen)%5000==0:guard()
        a,t,w,e=r['algorithm'],int(r['train_seed']),int(r['worker']),int(r['episode_index']);label=(a,t,w,e);h.check(label not in seen and a in h.ARMS and t in TRAINS and 0<=w<16 and 0<=e<320,'unique train grid');seen.add(label);k=(t,w,e);physical=h.seed(t,f'{h.VERSION}:N4:training_physical:worker:{w}:episode:{e}')
        if k not in population:population[k]=h.initial_events(physical)[2]
        h.check(int(r['physical_seed'])==physical and r['population_sha256']==population[k] and int(r['physical_ticks'])==60 and float(r['invalid_assignments'])==0 and abs(float(r['episode_reward'])*20+float(r['extended_cost']))<1e-5,'paired actual train tapes/reward')
    h.check(len(seen)==122880 and len(population)==15360,'complete train grid');dev=list(h.rows(source/'development_episodes.csv'));auc=auc_reference(dev,h);stored=list(h.rows(source/'auc_episodes.csv'));h.check(len(stored)==768,'768AUCrows');seen_auc=set()
    for r in stored:
        label=r['method'],int(r['train_seed']),int(r['seed']);h.check(label in auc and label not in seen_auc,'unique AUC row');seen_auc.add(label);h.near(auc[label],r['auc_cost_per_machine'],'per-seed AUC');h.check(int(r['nodes'])==8,'AUC node count')
    ancillary=h.read(source/'ancillary_checkpoint_choices.json')
    for a in h.ARMS:
        for t in TRAINS:
            score={step:float(np.mean([float(r['extended_cost']) for r in dev if r['method']==a and int(r['train_seed'])==t and int(r['checkpoint_steps'])==step])) for step in STEPS};choice=ancillary[f'{a}:{t}'];h.check(choice['steps']==min(score,key=lambda step:(score[step],step)) and choice['development_cost']==score[choice['steps']],'ancillary choice separated');h.weights(source,choice);h.near(choices[f'{a}:{t}']['development_cost'],score[245760],'fixed-final DEV cost')
    return dict(models=24,training_decisions=5898240,training_episodes=122880,adam_steps=92160,paired_training_tapes=15360,development_nodes=6144,auc_rows=768),auc


def run(source,out):
    source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);h=helpers();torch.set_num_threads(1);start=time.monotonic();manifest=dict(status='RUNNING',source=str(source),source_manifest_sha256=h.sha(source/'manifest.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),goal_pass=False,test_opened=False)
    def guard():h.check(time.monotonic()-start<900,'15minute verification hardstop')
    def bindings():
        h.check(h.sha(source/'manifest.json')==manifest['source_manifest_sha256'],'unchanged source manifest')
        for name,meta in h.read(source/'manifest.json')['output_bindings'].items():guard();h.check(h.sha(source/name)==meta['sha256'] and (source/name).stat().st_size==meta['bytes'],'output binding '+name)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        m=h.read(source/'manifest.json');h.check(m['status']=='COMPLETED' and m['audit_status']=='PASS' and m['git']['commit']==COMMIT and not m['git']['dirty'] and not m['test_opened'] and m['protocol']=='technician_broker_learning_efficiency_v1','frozen DEV study');bindings()
        for name,digest in m['source_seal'].items():
            h.check(h.sha(source/name)==digest,'source seal')
            if name.startswith('source_snapshot/'):
                blob=subprocess.check_output(['git','show',f'{COMMIT}:src/ht_pdm_fjsp/{Path(name).name}'],cwd=ROOT);h.check(hashlib.sha256(blob).hexdigest()==digest,'committed snapshot')
        c=h.read(source/'resolved_config.json');choices=h.read(source/'checkpoint_choices.json');budget,auc=budgets(source,c,choices,h,guard);selected=h.verify_selected(source,choices,guard);latency=h.verify_primary_timing(source,choices);average,points,checks,per=endpoint_reference(selected['means'],auc,h);summary=h.read(source/'summary.json')
        for a in h.ARMS:h.near(average[a],summary['auc_means'][a],'AUC mean')
        for k,v in points.items():h.near(v,summary['endpoints'][k],'endpoint '+k)
        h.check(checks==summary['checks'],'all14point gates')
        for a,metrics in selected['means'].items():
            for k,v in metrics.items():h.near(v,summary['final_means'][a][k],'fixed-final mean')
        for t,data in per.items():
            for k,v in data.items():h.near(v,summary['per_replica_auc'][t][k],'replica AUC')
        invalid=sum(float(r['invalid_assignments']) for r in h.rows(source/'selected_episodes.csv'));extra=dict(every_replica_auc_gain_positive=all(v['gain']>0 for v in per.values()),zero_invalid=invalid==0,isolated_p95_at_most_50ms=latency['p95_ms']<=50);passed=all(checks.values()) and all(extra.values());h.check(extra==summary['extra_checks'] and passed==summary['development_screen_pass']==m['development_screen_pass'] and summary['primary_fixed_steps']==245760 and not summary['goal_pass'] and not summary['test_opened'],'study verdict');bindings()
        result=dict(status='PASS',classification='DEVELOPMENT_VERIFICATION_ONLY',source=str(source),source_manifest_sha256=manifest['source_manifest_sha256'],budgets=budget,selected=selected,independent_points=points,independent_checks=checks,per_replica_auc=per,original_development_pass=passed,primary_latency=latency,goal_pass=False,test_opened=False,limitation='No CI or scientific claim. Complete final physical/actor replay; recorded nonfinal DEV nodes audited, not all independently re-simulated. Historical minibatch training gradients not retained.')
        (out/'verification.json').write_text(json.dumps(result,indent=2));shutil.copy2(__file__,out/'verifier.py');shutil.copy2(ROOT/'scripts/verify_broker_marginal_development.py',out/'reference_helpers.py');shutil.copy2(ROOT/'docs/technician_broker_learning_efficiency_v1_verification_plan.md',out/'protocol.md');manifest.update(status='COMPLETED',elapsed_seconds=time.monotonic()-start,outputs={p.name:dict(sha256=h.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(output=str(out),status='PASS',development_pass=passed)))
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.source,args.output)
