"""Read-only independent preimage-sum and primitive replay of frozen mode diagnostic."""
import argparse,csv,hashlib,importlib.util,itertools,json,shutil,subprocess,time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mode_audit_primitives',ROOT/'scripts/verify_broker_marginal_development.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
CAT=h.baseline_catalogue();INDEX={v:i for i,v in enumerate(CAT)}
MODES=('greedy','matching_mode');METRICS=('normalized_cost','availability','normalized_pending','normalized_failures')


def pair(value):return tuple(map(tuple,json.loads(value)))
def near(a,b,label):h.check(abs(float(a)-float(b))<1e-8,label)


@lru_cache(maxsize=32)
def translator(core,ranking):
    """Explicit 25/81 latent proposals mapped to the existing 21 physical tokens."""
    proposals=np.array(list(itertools.product(range(5 if core else 3),repeat=2 if core else 4)),int)
    matrix=np.zeros((len(proposals),21))
    for i,u in enumerate(proposals):
        result=[]
        for resource in range(4 if core else 2):
            requesters=[a for a,token in enumerate(u) if token==resource+1]
            if not requesters:continue
            winner=min(requesters,key=lambda a:ranking[resource*2+a] if core else ranking[a])
            result.append((resource,winner) if core else (winner,resource))
        matrix[i,INDEX[tuple(sorted(result))]]=1
    return proposals,matrix


def dense_mass(probabilities,core,ranking):
    u,matrix=translator(core,tuple(ranking));mass=np.ones((len(probabilities),len(u)))
    for a in range(u.shape[1]):mass*=probabilities[:,a,u[:,a]]
    return mass@matrix


def reference_laws(observations,states,arm,w):
    if arm in h.CORE:
        logp=h.core_log_probs(observations,w);priority=h.priority_matrix(observations[0],arm)
        rank=tuple(int(a!=max(range(2),key=lambda j:(priority[m,j],-j))) for m in range(4) for a in range(2))
        return dense_mass(np.exp(logp),True,rank),logp.argmax(-1)
    if arm in ('ppo','normalized_ppo'):
        logits=h.mlp(np.stack([o['state'] for o in observations]).astype(float),w,'actor')
        mask=np.array([[all(o['pair_mask'][m,j] for m,j in p) for p in CAT] for o in observations])
    else:
        inputs=[]
        for o in observations:
            if arm=='full_mappo':local=np.c_[np.tile(o['state'],(4,1)),np.eye(4,dtype=np.float32)]
            else:
                workers=o['technicians'].copy();workers[:,2]=0
                local=np.c_[o['machines'],np.tile(workers.ravel(),(4,1)),o['edges'].reshape(4,-1),np.tile(o['global_features'],(4,1)),np.eye(4,dtype=np.float32)]
            inputs.append(local)
        logits=h.mlp(np.stack(inputs).astype(float),w,'actor');mask=np.stack([np.c_[np.ones(4,bool),o['pair_mask']] for o in observations])
    logits=np.where(mask,logits,-np.inf);prob=np.exp(logits-np.logaddexp.reduce(logits,axis=-1)[...,None]);greedy=logits.argmax(-1)
    if arm in ('ppo','normalized_ppo'):return prob,greedy
    rankings=defaultdict(list)
    for i,state in enumerate(states):
        order=sorted(range(4),key=lambda m:(-h.urgency(state,m),m));rank=tuple(order.index(m) for m in range(4));rankings[rank].append(i)
    law=np.zeros((len(observations),21))
    for rank,indices in rankings.items():law[indices]=dense_mass(prob[indices],False,rank)
    return law,greedy


def decode(u,arm,state,o):
    if arm in h.CORE:return h.independent_decode(u,o['pair_mask'],arm,o['edges'])[0]
    if arm in ('ppo','normalized_ppo'):return CAT[int(u)]
    result=[]
    for j in range(2):
        group=[m for m,t in enumerate(u) if t==j+1]
        if group:result.append((max(group,key=lambda m:(h.urgency(state,m),-m)),j))
    return tuple(sorted(result))


def audit_actor(data,arm,train,source,choices):
    observations,states,records=map(list,zip(*data));w=h.weights(source,choices[f'{arm}:{train}']);law,greedy=reference_laws(observations,states,arm,w)
    h.check(np.allclose(law.sum(1),1,atol=2e-12,rtol=0),'dense all-state normalization');error=0.
    greedy_indices=[i for i,r in enumerate(records) if r['mode']=='greedy']
    h.verify_actor([observations[i] for i in greedy_indices],[json.loads(records[i]['submitted']) for i in greedy_indices],arm,w)
    for i,(o,state,r) in enumerate(data):
        gp=pair(r['greedy_pairs']);mp=pair(r['mode_pairs']);chosen=law[i,INDEX[mp]];maximum=law[i].max();tolerance=5e-6+1e-6*abs(np.log(chosen))
        h.check(chosen>0 and np.log(maximum)-np.log(chosen)<=tolerance,'independent near-exact physical mode')
        reference_pairs=decode(greedy[i],arm,state,o)
        h.check(reference_pairs==gp,'independent physical greedy matching')
        for pairs,field in ((gp,'greedy_logmass'),(mp,'mode_logmass')):
            expected=float(np.log(law[i,INDEX[pairs]]));actual=float(r[field]);delta=abs(expected-actual);error=max(error,delta)
            h.check(delta<=5e-6+1e-6*abs(expected),'independent model matching mass')
        submitted=json.loads(r['submitted']);h.check(decode(submitted,arm,state,o)==pair(r['executed_pairs']),'submitted arbitration')
        rejected=0 if arm in ('ppo','normalized_ppo') else int(np.count_nonzero(submitted))-len(pair(r['executed_pairs']))
        near(rejected,r['executed_rejections'],'actual proposal rejections')
        h.check(int(r['same_state_mismatch'])==int(gp!=mp),'same-state mismatch record')
    return dict(method=arm,train_seed=train,states=len(data),max_logmass_absolute_error=error)


def summary(rows):
    means={mode:{a:{k:float(np.mean([float(r[k]) for r in rows if r['method']==a and r['mode']==mode])) for k in METRICS} for a in h.ARMS} for mode in MODES}
    points,checks=h.endpoint_gates(means['matching_mode']);a=h.CORE[0];own=1-means['matching_mode'][a]['normalized_cost']/means['greedy'][a]['normalized_cost'];rep={}
    for t in h.TRAINS:
        costs={mode:float(np.mean([float(r['normalized_cost']) for r in rows if r['method']==a and int(r['train_seed'])==t and r['mode']==mode])) for mode in MODES};rep[str(t)]=dict(**costs,gain=1-costs['matching_mode']/costs['greedy'])
    greedy=[r for r in rows if r['method']==a and r['mode']=='greedy'];mismatch=sum(int(r['same_state_mismatches']) for r in greedy)/(48*len(greedy))
    screen=dict(own_cost_gain_at_least_half_percent=own>=.005,every_replica_improves=all(r['gain']>0 for r in rep.values()),primary_mismatch_at_least_one_percent=mismatch>=.01,all13_mode_point_gates=all(checks.values()),zero_invalid=sum(float(r['invalid_assignments']) for r in rows)==0)
    return dict(means=means,mode_endpoints=points,mode_checks=checks,candidate_own_gain=own,candidate_per_replica=rep,primary_greedy_state_mismatch_rate=mismatch,screen_checks=screen,future_polynomial_decoder_trial_motivated=all(screen.values()))


def run(source,out):
    source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();manifest=dict(status='RUNNING',source=str(source),source_manifest_sha256=h.sha(source/'manifest.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),goal_pass=False,test_opened=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def guard():h.check(time.monotonic()-start<900,'15minute verification hardstop')
    def bindings():
        h.check(h.sha(source/'manifest.json')==manifest['source_manifest_sha256'],'unchanged diagnostic manifest')
        for name,meta in h.read(source/'manifest.json')['output_bindings'].items():guard();h.check(h.sha(source/name)==meta['sha256'] and (source/name).stat().st_size==meta['bytes'],'diagnostic binding '+name)
    try:
        m=h.read(source/'manifest.json');h.check(m['status']=='COMPLETED' and m['audit_status']=='PASS' and not m['test_opened'] and m['updates']==0,'frozen diagnostic only');bindings()
        parent=Path(m['source']);h.check(h.sha(parent/'manifest.json')==m['source_manifest_sha256'] and h.sha(parent/'checkpoint_choices.json')==m['choices_sha256'],'original source/selection binding');choices=h.read(parent/'checkpoint_choices.json')
        rows=list(h.rows(source/'episodes.csv'));lookup={(r['method'],int(r['train_seed']),int(r['seed']),r['mode']):r for r in rows};expected={(a,t,s,mode) for a in h.ARMS for t in h.TRAINS for s in range(300100000,300100064) for mode in MODES};h.check(len(rows)==3072 and set(lookup)==expected,'complete3072 grid')
        episodes={};data=[];group=None;actors=[];policy_iter=iter(h.rows(source/'policy.csv'));count=0;decisions=0
        for r in tqdm(h.rows(source/'physical.csv'),total=184320,desc='Independent primitive replay'):
            if count%1000==0:guard()
            key=r['method'],int(r['train_seed']),int(r['seed']),r['mode'];tick=int(r['tick']);h.check(key in expected,'registered condition');state=json.loads(r['state']);pairs=pair(r['pairs'])
            if key not in episodes:
                physical=h.seed(key[2],f'{h.VERSION}:N4:physical');initial,events,pop=h.initial_events(physical);episodes[key]=dict(state=initial,events=events,pop=pop,metrics=defaultdict(float),tick=0,mismatch=0,rejections=0,latent=0.)
            ep=episodes[key];h.check(ep['tick']==tick and state==ep['state'] and ep['events'][tick]==json.loads(r['events']),'independent entire state/event chain');h.check(int(r['physical_seed'])==h.seed(key[2],f'{h.VERSION}:N4:physical'),'paired physical seed')
            if tick<48:
                pr=next(policy_iter);h.check((pr['method'],int(pr['train_seed']),int(pr['seed']),pr['mode'],int(pr['tick']))==key+(tick,),'ordered policy/physics join');h.check(pair(pr['executed_pairs'])==pairs,'executed physical matching')
                if group!=key[:2]:
                    if data:actors.append(audit_actor(data,*group,parent,choices))
                    group=key[:2];data=[]
                data.append((h.observation(state,tick),state,pr));ep['mismatch']+=int(pr['same_state_mismatch']);ep['rejections']+=int(pr['executed_rejections']);ep['latent']+=float(pr['expected_latent_rejections']);decisions+=1
            else:
                o=h.observation(state,tick);scores=np.array([[h.urgency(state,m)+h.R[m,j]/h.D[m,j] for j in range(2)] for m in range(4)]);near(sum(scores[m,j] for m,j in pairs),h.top_two_optimum(scores,o['pair_mask']),'unchanged deadline tail objective')
            following,parts=h.primitive_step(state,pairs,ep['events'][tick]);h.check(following==json.loads(r['following']),'independent next state');near(parts['objective'],r['cost'],'physical cost')
            for k,v in parts.items():ep['metrics'][k]+=v
            ep['state']=following;ep['tick']+=1;count+=1
            if tick==47:ep['cutoff']=dict(ep['metrics']);ep['pending']=sum(following['failed'][m] and m not in following['assigned'] for m in range(4))
        if data:actors.append(audit_actor(data,*group,parent,choices))
        h.check(next(policy_iter,None) is None and count==184320 and decisions==147456 and set(episodes)==expected and len(actors)==24,'complete replay and actor grid')
        for key,ep in episodes.items():
            r=lookup[key];h.check(ep['tick']==60 and ep['pop']==r['population_sha256'],'population/final tick')
            for k,v in ep['metrics'].items():near(v,r[k],'episode component '+k)
            cut=ep['cutoff'];values=dict(extended_cost=ep['metrics']['objective'],normalized_cost=ep['metrics']['objective']/4,policy_cost=cut['objective'],tail_cost=ep['metrics']['objective']-cut['objective'],availability=1-cut['unavailable_ticks']/192,policy_failures=cut['failures'],normalized_failures=cut['failures']/4,terminal_pending=ep['pending'],normalized_pending=ep['pending']/4,same_state_mismatches=ep['mismatch'],executed_rejections=ep['rejections'],expected_latent_rejections=ep['latent'])
            for k,v in values.items():near(v,r[k],'derived episode '+k)
        recomputed=summary(rows);original=h.read(source/'summary.json')
        for key,value in recomputed.items():h.check(original[key]==value,'independent summary '+key)
        bindings();result=dict(status='PASS',source=str(source),source_manifest_sha256=manifest['source_manifest_sha256'],episodes=3072,physical_transitions=count,policy_decisions=decisions,actor_checks=actors,summary=recomputed,goal_pass=False,test_opened=False,limitation='No training replay or scientific test. Independent float64 NumPy scores checked against original float32 with explicit roundoff tolerance; reference mode remains dense N4.')
        (out/'verification.json').write_text(json.dumps(result,indent=2));shutil.copy2(__file__,out/'verifier.py');shutil.copy2(ROOT/'scripts/verify_broker_marginal_development.py',out/'reference_helpers.py');manifest.update(status='COMPLETED',elapsed_seconds=time.monotonic()-start,outputs={p.name:dict(sha256=h.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(output=str(out),status='PASS')))
    except Exception as e:
        manifest.update(status='FAILED',error=f'{type(e).__name__}: {e}',elapsed_seconds=time.monotonic()-start);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args();run(args.source,args.output)
