"""Independent read-only reconciliation of decoding audit artifacts, no training."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys
from collections import Counter, defaultdict

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUN=ROOT/'lab_results/maintenance_decoding_audit_v1_full_cpu_20261008T135102Z'
SOURCE=ROOT/'lab_results/maintenance_solver_comparison_v1_full_cpu_20261007T151955Z'
WORK=Path('/Users/bachng/.codex/worktrees/maintenance-solver-comparison/marl')
sys.path.insert(0,str(WORK/'src'))
def read(name):return json.loads((RUN/name).read_text())
def rows(name):
    with (RUN/name).open(newline='') as f:yield from csv.DictReader(f)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

m=read('manifest.json');cfg=m['config'];original=read('summary.json')
assert m['status']=='COMPLETED' and m['dirty']==''
assert m['actual_counts']==m['expected_counts']
for name,meta in m['artifacts'].items():
    p=RUN/name
    assert p.stat().st_size==meta['bytes'] and digest(p)==meta['sha256'],name
prov=read('source_provenance.json')
assert digest(SOURCE/'manifest.json')==prov['source_manifest_sha256']==m['source_manifest_sha256']
for name,meta in prov['verified_artifacts'].items():
    p=SOURCE/name
    assert digest(p)==meta['sha256'] and p.stat().st_size==meta['bytes'],name
snapshot_files=0
for folder in ('source_snapshot','inherited_source_snapshot'):
    for p in (RUN/folder).iterdir():
        candidates=[WORK/'src/ht_pdm_fjsp'/p.name,WORK/'docs'/p.name,WORK/'configs'/p.name,WORK/p.name]
        current=next(x for x in candidates if x.is_file())
        committed=subprocess.check_output(['git','-C',str(WORK),'show',m['commit']+':'+str(current.relative_to(WORK))])
        assert p.read_bytes()==committed==current.read_bytes(),p.name
        snapshot_files+=1
print('Hashes, frozen inputs and committed sources verified',flush=True)

metrics=['objective','base_cost','maintenance_cost','failure_cost','unavailability_cost','waiting_cost','failures',
    'waiting_violation','terminal_pending','corrective_requests','served_requests','censored_requests',
    'terminal_zero_wait_censored','request_overdue_observed','max_wait','inference_seconds',
    'decision_latency_p50','decision_latency_p95',
    'failed_waiting_ticks','overdue_waiting_ticks','service_ticks','unavailable_ticks']
physical=['objective','base_cost','maintenance_cost','failure_cost','unavailability_cost','waiting_cost',
          'failures','preventive_jobs','corrective_jobs','jobs','completed_jobs','unavailable_ticks',
          'failed_waiting_ticks','service_ticks','overdue_waiting_ticks','max_wait','waiting_violation',
          'terminal_pending','terminal_failed','corrective_requests','served_requests','censored_requests',
          'request_overdue_observed','terminal_zero_wait_censored']
groups=defaultdict(list);replicas=defaultdict(list);pairs={};keys=set();decisions=0
for r in rows('episodes.csv'):
    r={k:(v if k in ('controller','algorithm','decoder','condition') else float(v)) for k,v in r.items()}
    key=(r['controller'],r['train_seed'],r['cohort'],r['condition'],r['horizon'],r['seed'])
    assert key not in keys;keys.add(key)
    group=(r['controller'],r['condition'],int(r['horizon']))
    groups[group].append(r);replicas[group+(int(r['train_seed']),)].append(r)
    if r['algorithm'] in ('cooperative_local_ppo','cooperative_full_ppo'):
        pkey=(r['algorithm'],int(r['train_seed']),int(r['cohort']),r['condition'],int(r['horizon']),int(r['seed']))
        pairs.setdefault(pkey,{})[r['decoder']]=r
    assert r['invalid_assignments']==0
    assert r['maximum_mass_error']<=1e-5 and r['maximum_map_selection_regret']<=1e-6
    assert abs(r['objective']-r['base_cost']-r['waiting_cost'])<1e-8
    assert abs(r['base_cost']-r['maintenance_cost']-r['failure_cost']-r['unavailability_cost'])<1e-8
    assert r['corrective_requests']==r['failures']==r['served_requests']+r['censored_requests']
    assert r['terminal_pending']==r['censored_requests']
    assert r['waiting_cost']==12*r['overdue_waiting_ticks']
    assert r['unavailability_cost']==6*r['unavailable_ticks']
    assert r['failure_cost']==15*r['failures']
    decisions+=int(r['horizon'])
assert len(keys)==317440 and decisions==5713920
controller_seeds={a:cfg['train_seeds'] for a in (
    'central_matching_ppo__greedy','central_fixed_allocation_ppo__greedy',
    'cooperative_local_ppo__greedy','cooperative_local_ppo__joint_map',
    'cooperative_full_ppo__greedy','cooperative_full_ppo__joint_map')}
controller_seeds.update(selected_rule=[-1],joint_rollout=[-1])
expected_keys={(a,seed,cohort,condition,h,shock) for a,seeds in controller_seeds.items()
    for seed,cohort,condition,h,shock in itertools.product(seeds,cfg['cohorts'],cfg['conditions'],cfg['horizons'],cfg['shocks'])}
assert keys==expected_keys
assert all(len(v)==(640 if a in ('selected_rule','joint_rollout') else 6400) for (a,c,h),v in groups.items())
def mean(v,k='objective'):return st.mean(r[k] for r in v)
means={f'{a}:{c}:H{h}':{metric:mean(v,metric) for metric in metrics} for (a,c,h),v in groups.items()}
for key,v in original['aggregate'].items():
    for k,x in v.items():assert abs(x-means[key][k])<1e-8,(key,k)

contrasts={}
for label,result in original['contrasts'].items():
    arm,cond,htext=label.split(':');h=int(htext[1:]);deltas=[]
    for seed in cfg['train_seeds']:
        a=replicas[(arm+'__joint_map',cond,h,seed)]
        b=replicas[(arm+'__greedy',cond,h,seed)]
        deltas.append(mean(a)-mean(b))
    critical=2.6850108468164553 if result['confirmatory'] else 2.2621571627982055
    half=critical*st.stdev(deltas)/math.sqrt(10);delta=st.mean(deltas)
    ci=[delta-half,delta+half]
    assert abs(delta-result['difference'])<1e-8
    assert max(abs(a-b) for a,b in zip(ci,result['ci']))<1e-8
    assert sum(x < -1e-9 for x in deltas)==result['wins']
    contrasts[label]={**result,'replica_differences':dict(zip(cfg['train_seeds'],deltas))}
reference_matches=0
for label,result in original['reference_contrasts'].items():
    comparison,cond,htext=label.split(':');left,right=comparison.split('_vs_');h=int(htext[1:])
    deltas=[]
    for seed in cfg['train_seeds']:
        a=replicas[(left,cond,h,seed)]
        b=replicas[(right,cond,h,seed)] if right.endswith('__greedy') else groups[(right,cond,h)]
        deltas.append(mean(a)-mean(b))
    half=2.2621571627982055*st.stdev(deltas)/math.sqrt(10);delta=st.mean(deltas)
    assert abs(delta-result['difference'])<1e-8
    assert max(abs(a-b) for a,b in zip([delta-half,delta+half],result['ci']))<1e-8
    reference_matches+=1
pair_stats=defaultdict(Counter)
changed=[]
for key,v in pairs.items():
    assert set(v)=={'greedy','joint_map'}
    a,b=v['joint_map'],v['greedy'];arm,seed,cohort,cond,h,shock=key
    delta=a['objective']-b['objective']
    pair_stats[f'{arm}:{cond}:H{h}'].update(episodes=1,cost_equal=delta==0,
        map_lower_cost=delta<0,map_higher_cost=delta>0,physical_metrics_equal=all(a[k]==b[k] for k in physical))
    if cond=='specialized' and h==12 and delta:
        changed.append(dict(algorithm=arm,train_seed=seed,cohort=cohort,shock=shock,delta=delta))
print('All episode and contrast statistics independently matched',flush=True)

requests=defaultdict(Counter);waits=defaultdict(Counter);request_rows=0
for r in rows('request_metrics.csv'):
    group=f'{r["controller"]}:{r["condition"]}:H{r["horizon"]}'
    wait=int(r['wait']);served=r['served']=='True';censored=r['censored']=='True'
    assert served!=censored and wait==int(r['end'])-int(r['arrival'])
    assert (r['overdue_observed']=='True')==(wait>4)
    requests[group].update(requests=1,served=served,censored=censored,observed_overdue=wait>4,
        terminal_zero_wait_censored=censored and wait==0,
        served_overdue=served and wait>4,censored_overdue=censored and wait>4,
        served_overdue_ticks=max(wait-4,0) if served else 0,
        censored_overdue_ticks=max(wait-4,0) if censored else 0,
        censored_observed_wait_sum=wait if censored else 0)
    if served:waits[group][wait]+=1
    request_rows+=1
for key,v in requests.items():
    a,c,h=key.split(':');group=groups[(a,c,int(h[1:]))]
    for raw,metric in [('requests','corrective_requests'),('served','served_requests'),('censored','censored_requests'),
        ('observed_overdue','request_overdue_observed'),('terminal_zero_wait_censored','terminal_zero_wait_censored')]:
        assert v[raw]==round(sum(r[metric] for r in group))
    assert v['served_overdue_ticks']+v['censored_overdue_ticks']==round(sum(r['overdue_waiting_ticks'] for r in group))
    v['censored_waiting_cost_fraction']=v['censored_overdue_ticks']/(v['served_overdue_ticks']+v['censored_overdue_ticks']) if v['served_overdue_ticks']+v['censored_overdue_ticks'] else 0.
    if v['served']:
        target=math.ceil(.95*v['served']);seen=0
        for wait,n in sorted(waits[key].items()):
            seen+=n
            if seen>=target:break
        v.update(served_wait_p95_nearest_rank=wait,served_wait_max=max(waits[key]),
                 served_wait_mean=sum(w*n for w,n in waits[key].items())/v['served'])
    v['censored_fraction']=v['censored']/v['requests']
    for k,x in read('request_summary.json')['groups'][key].items():assert abs(v[k]-x)<1e-8
latency=defaultdict(list);latency_rows=0
for r in rows('latency.csv'):
    latency[f'{r["controller"]}:{r["condition"]}'].append(float(r['seconds'])*1000);latency_rows+=1
assert latency_rows==8928
latency={key:dict(n=len(v),mean_ms=st.mean(v),median_ms=st.median(v),
    p95_ms=sorted(v)[math.ceil(.95*len(v))-1]) for key,v in latency.items()}
print('Request reconciliation and common-state latency checked',flush=True)

audit_stats=defaultdict(Counter);audit_count=0;candidate_total=0;max_error=0
for r in rows('decoding_audit.csv'):
    actions=json.loads(r['candidate_actions']);scores=json.loads(r['candidate_log_probabilities'])
    greedy=json.loads(r['greedy_action']);mapped=json.loads(r['map_action'])
    assert len(actions)==len(scores)==int(r['candidate_count'])
    mass=sum(math.exp(x) for x in scores)
    assert abs(mass-float(r['probability_mass']))<1e-8 and abs(mass-1)<1e-5
    best=max(scores);choice=next(i for i,x in enumerate(scores) if x>=best-1e-6)
    assert mapped==actions[choice]
    assert float(r['greedy_log_probability'])==scores[actions.index(greedy)]
    assert float(r['map_log_probability'])==scores[choice]
    assert (r['disagreement']=='True')==(greedy!=mapped)
    assert abs(scores[choice]-scores[actions.index(greedy)]-float(r['map_log_probability_gain']))<1e-8
    key=f'{r["algorithm"]}:{r["condition"]}'
    audit_stats[key].update(states=1,disagreements=greedy!=mapped,
        strictly_better_probability=scores[choice]>scores[actions.index(greedy)]+1e-6,
        tied_probability_different_action=greedy!=mapped and abs(scores[choice]-scores[actions.index(greedy)])<=1e-6)
    max_error=max(max_error,abs(mass-1));audit_count+=1;candidate_total+=len(scores)
assert audit_count==960
save('independent_analysis.json',dict(artifact_hashes=len(m['artifacts']),source_artifact_hashes=len(prov['verified_artifacts']),
    committed_source_files=snapshot_files,episodes=len(keys),decision_intervals=decisions,
    request_rows=request_rows,latency_rows=latency_rows,decoder_audits=audit_count,candidate_probabilities=candidate_total,
    aggregate=means,contrasts=contrasts,reference_contrasts_reconciled=reference_matches,
    episode_pair_counts=dict(pair_stats),primary_nonzero_episode_differences=changed,
    request_summary=dict(requests),latency=latency,common_state_audit=dict(audit_stats),maximum_logged_mass_error=max_error))

# Re-score logged DEVELOPMENT states only on Mac using teacher-forced original serial
# forward pass, independently of the optimized MAP scoring implementation. Not a full run.
from ht_pdm_fjsp.maintenance_solver_comparison import load_checkpoint
from ht_pdm_fjsp.maintenance_solver_policy import feature_batch, CATALOGUE
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig
from ht_pdm_fjsp.maintenance_joint_map import matching_log_probabilities, best_index
import torch
torch.set_num_threads(1)
states=read('latency_states.json');models={};maximum_likelihood_error=0;greedy_mismatches=[];support_mismatches=[]
maximum_selected_likelihood_error=0;maximum_optimized_likelihood_error=0;large_error_rows=[];map_mismatches=[]
for r in rows('decoding_audit.csv'):
    key=(r['algorithm'],int(r['train_seed']))
    if key not in models:
        record=prov['checkpoint_selection'][f'{key[0]}:{key[1]}']
        models[key]=load_checkpoint(SOURCE/record['checkpoint'])[0]
    model=models[key];snapshot=states[int(r['state_index'])];cd=snapshot['config']
    c=WaitingConfig(**{**cd,'service_time':tuple(map(tuple,cd['service_time'])),
                         'restoration':tuple(map(tuple,cd['restoration']))})
    s=DispatchState(**{k:tuple(v) for k,v in snapshot['state'].items()});h=snapshot['remaining']
    actions=[tuple(map(tuple,a)) for a in json.loads(r['candidate_actions'])]
    batch=feature_batch([c],[s],[h]);ids=torch.where(batch['matching_mask'][0])[0]
    if actions!=[CATALOGUE[int(i)] for i in ids]:support_mismatches.append(r['state_index'])
    raw=torch.tensor([[next((j+1 for m,j in a if m==i),0) for i in range(4)] for a in actions],dtype=torch.long)
    expanded={k:v.expand(len(actions),*v.shape[1:]) for k,v in batch.items()}
    with torch.no_grad():
        _,scores,_,_=model._serial(expanded,action=raw,compute_value=False)
    logged=torch.tensor(json.loads(r['candidate_log_probabilities']),dtype=torch.float64)
    error=float((scores.double()-logged).abs().max());maximum_likelihood_error=max(maximum_likelihood_error,error)
    greedy_index=actions.index(tuple(map(tuple,json.loads(r['greedy_action']))))
    map_index=actions.index(tuple(map(tuple,json.loads(r['map_action']))))
    selected_error=float((scores.double()[[greedy_index,map_index]]-logged[[greedy_index,map_index]]).abs().max())
    maximum_selected_likelihood_error=max(maximum_selected_likelihood_error,selected_error)
    if error>1e-5:
        large_error_rows.append(dict(algorithm=key[0],seed=key[1],state_index=r['state_index'],
            maximum_error=error,selected_error=selected_error,minimum_logged_score=float(logged.min())))
    ids,optimized,_=matching_log_probabilities(model,c,s,h)
    maximum_optimized_likelihood_error=max(maximum_optimized_likelihood_error,float((optimized.double()-logged).abs().max()))
    if CATALOGUE[int(ids[best_index(optimized)])]!=actions[map_index]:
        map_mismatches.append(dict(algorithm=key[0],seed=key[1],state_index=r['state_index']))
    greedy,_=model.select(c,s,h)
    if greedy!=tuple(map(tuple,json.loads(r['greedy_action']))):
        greedy_mismatches.append(dict(algorithm=key[0],seed=key[1],state_index=r['state_index']))
    for a in actions:validate_matching(c,s,a)
save('development_actor_replay.json',dict(states=audit_count,models=len(models),candidate_likelihoods=candidate_total,
    maximum_likelihood_difference=maximum_likelihood_error,greedy_action_mismatches=greedy_mismatches,
    support_mismatches=support_mismatches,comparison_tolerance=1e-5,
    maximum_selected_likelihood_difference=maximum_selected_likelihood_error,
    maximum_optimized_mac_vs_linux_difference=maximum_optimized_likelihood_error,
    rows_exceeding_1e_minus_5=large_error_rows,map_action_mismatches=map_mismatches,
    scope='All logged common DEVELOPMENT states, CPU Mac vs source CPU Linux; no test trajectory or training rerun.'))
print('Development actor replay finished:',maximum_likelihood_error,'action mismatches:',len(greedy_mismatches),flush=True)
