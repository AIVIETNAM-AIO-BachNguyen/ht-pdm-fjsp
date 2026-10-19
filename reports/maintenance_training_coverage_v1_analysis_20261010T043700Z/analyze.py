"""Read-only independent arithmetic/provenance audit; never mutates raw run."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import csv, hashlib, itertools, json, math, statistics, subprocess, sys

ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl')
RUN=ROOT/'lab_results/maintenance_training_coverage_v1_full_cpu_20261009T044236Z'
OUT=Path(__file__).resolve().parent
WORK=Path('/Users/bachng/.codex/worktrees/maintenance-solver-comparison/marl')
C='central_matching_ppo';L='cooperative_local_ppo';F='cooperative_full_ppo'
ARMS=(C,L,F);CONDS=('homogeneous','specialized','nominal','skill_mask')
REGS=('legacy','covered');SEEDS=list(range(180000,180010))
SCHEDULES={'legacy':['specialized']*4+['nominal'],'covered':['specialized','skill_mask','specialized','skill_mask','nominal']}
METRICS=('objective','base_cost','maintenance_cost','failure_cost','unavailability_cost','waiting_cost','failures','unavailable_ticks','waiting_violation','terminal_pending','corrective_requests','served_requests','censored_requests','request_overdue_observed','max_wait','p95_wait','inference_seconds','communication_bytes','communication_rounds','served_waiting_cost','censored_waiting_cost')
PHYS=('objective','base_cost','maintenance_cost','failure_cost','unavailability_cost','waiting_cost','unavailable_ticks','failed_waiting_ticks','overdue_waiting_ticks','jobs','failures','terminal_failed','terminal_pending','max_wait','waiting_violation','corrective_requests','served_requests','censored_requests','request_overdue_observed','served_waiting_cost','censored_waiting_cost')
def js(p):return json.loads((RUN/p).read_text())
def write(name,data): (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as s:
  for b in iter(lambda:s.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(name):
 with (RUN/name).open() as s:yield from csv.DictReader(s)
def key(r):return (r['training_regime'],r['controller'],int(r['train_seed']),int(r['cohort']),r['condition'],int(r['horizon']),int(r['seed']))
def gkey(r):return (r['training_regime'],r['controller'],r['condition'],int(r['horizon']))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)
def interval(x,confidence):
 t={.95:2.2621571627982055,.9875:3.1109348231794737}[confidence]
 half=t*statistics.stdev(x)/math.sqrt(len(x));m=statistics.mean(x)
 return [m-half,m+half]
manifest=js('manifest.json')
assert manifest['status']=='COMPLETED' and not manifest['dirty']
assert manifest['commit']=='53d180a5fc18c53e850eff7278d753fbe295d647'
assert manifest['expected_counts']==manifest['actual_counts']
expected_counts=dict(models=60,training_steps=29491200,training_episodes=2457600,optimizer_steps=614400,checkpoint_files=360,development_episodes=40960,test_episodes=317440,test_decision_intervals=5713920,test_machine_rows=1269760,latency_rows=8928)
assert manifest['actual_counts']==expected_counts
for f,meta in manifest['artifacts'].items():
 p=RUN/f
 assert p.resolve().is_relative_to(RUN.resolve()) and p.stat().st_size==meta['bytes']
 assert sha(p)==meta['sha256'],f
print('All411artifact hashes/sizes PASS',flush=True)
snapshot_count=0
for p in (RUN/'source_snapshot').iterdir():
 paths=list(WORK.rglob(p.name)) if p.suffix=='.py' else [WORK/('configs' if p.suffix=='.json' else 'docs' if p.suffix=='.md' else '')/p.name]
 paths=[x for x in paths if '.venv' not in x.parts and 'artifacts' not in x.parts]
 assert len(paths)==1,(p,paths)
 rel=paths[0].relative_to(WORK)
 original=subprocess.check_output(['git','show',f"{manifest['commit']}:{rel}"],cwd=WORK)
 assert original==p.read_bytes(),rel
 snapshot_count+=1
cfg=js('benchmark_config.json')
assert cfg==manifest['config']
assert cfg['train_seeds']==SEEDS and cfg['test_seeds']==list(range(185000,185020)) and cfg['development_seeds']==list(range(184000,184010))
assert cfg['env_steps']==491520 and cfg['selection']=='final_checkpoint_only' and cfg['decoder']=='greedy'
coverage=js('training_coverage.json');select=js('checkpoint_selection.json')
assert len(coverage)==len(select)==60
for seed in SEEDS:
 rr=[r for r in coverage if r['train_seed']==seed]
 assert len(rr)==6 and len({r['initial_parameters_sha256'] for r in rr})==1
 assert len({r['training_seed_sequence_sha256'] for r in rr})==1
for r in coverage:
 assert (r['env_steps'],r['episodes'],r['optimizer_steps'])==(491520,40960,10240)
 assert r['mixture_counts']=={c:n*8192 for c,n in Counter(SCHEDULES[r['training_regime']]).items()}
 assert r['parameters_changed'] and r['sampled_probability_audit'] and r['complete_episode_returns']
 assert r['reserved_seed_training_transitions']==r['teacher_label_rows']==0
for ident,r in select.items():
 assert r['selection']=='FINAL_ONLY' and r['selected_steps']==491520
 assert sha(RUN/r['checkpoint'])==r['checkpoint_sha256']
 assert sha(RUN/r['checkpoint'])==sha((RUN/r['checkpoint']).parent/'checkpoints/step_00491520.pt')
# Complete declared evaluation grid, no duplicated episode keys.
expected={(reg,a,s,cs,c,h,shock) for reg,a,seeds in [(r,a,SEEDS) for r,a in itertools.product(REGS,ARMS)]+[('reference',a,[-1]) for a in cfg['controls']] for s,cs,c,h,shock in itertools.product(seeds,cfg['test_cohorts'],CONDS,[12,24],cfg['test_seeds'])}
sums=defaultdict(Counter);ns=Counter();replicas=defaultdict(Counter);rn=Counter();episodes={}
for r in rows('episodes.csv'):
 k=key(r);assert k in expected and k not in episodes
 v={m:float(r[m]) for m in set(PHYS)|set(METRICS)}
 assert all(math.isfinite(x) for x in v.values()) and float(r['invalid_assignments'])==0
 close(v['objective'],v['base_cost']+v['waiting_cost']);close(v['base_cost'],v['maintenance_cost']+v['failure_cost']+v['unavailability_cost'])
 close(v['waiting_cost'],12*v['overdue_waiting_ticks']);close(v['failure_cost'],15*v['failures']);close(v['unavailability_cost'],6*v['unavailable_ticks'])
 close(v['corrective_requests'],v['failures']);close(v['corrective_requests'],v['served_requests']+v['censored_requests']);close(v['censored_requests'],v['terminal_pending'])
 close(v['served_waiting_cost']+v['censored_waiting_cost'],v['waiting_cost'])
 assert int(r['checkpoint_step'])==(491520 if k[0]!='reference' else -1)
 episodes[k]=tuple(v[m] for m in PHYS)
 g=gkey(r);ns[g]+=1;sums[g].update({m:v[m] for m in METRICS});rn[g+(k[2],)]+=1;replicas[g+(k[2],)].update({m:v[m] for m in METRICS})
assert set(episodes)==expected
assert sum(k[5] for k in episodes)==5713920
aggregate={':'.join((g[0],g[1],f'{g[2]}_H{g[3]}')):{m:sums[g][m]/ns[g] for m in METRICS} for g in ns}
means={g:{m:v[m]/rn[g] for m in METRICS} for g,v in replicas.items()}
def av(reg,a,c,h,m='objective'):return sums[(reg,a,c,h)][m]/ns[(reg,a,c,h)]
original=js('summary.json')
assert set(aggregate)==set(original['aggregate'])
for g,metrics in aggregate.items():
 for m,v in metrics.items():close(v,original['aggregate'][g][m])
contrasts={};paired=[]
for c,h in itertools.product(CONDS,(12,24)):
 for a,label in [(a,'coverage') for a in ARMS]+[(a,'gap_closure') for a in (L,F)]+[(a,'residual_gap') for a in (L,F)]:
  ds=[]
  for seed in SEEDS:
   legacy=means[('legacy',a,c,h,seed)]['objective'];covered=means[('covered',a,c,h,seed)]['objective']
   lc=means[('legacy',C,c,h,seed)]['objective'];cc=means[('covered',C,c,h,seed)]['objective']
   d=covered-legacy if label=='coverage' else (covered-cc)-(legacy-lc) if label=='gap_closure' else covered-cc
   ds.append(d)
  primary=c=='skill_mask' and h==24 and a in (L,F) and label in ('coverage','gap_closure')
  confidence=.9875 if primary else .95;ci=interval(ds,confidence)
  old=av('legacy',a,c,h);new=av('covered',a,c,h)
  oldgap=old-av('legacy',C,c,h);newgap=new-av('covered',C,c,h)
  reduction=1-new/old;closure=(oldgap-newgap)/oldgap if oldgap>0 else None
  gates=dict(practical_effect=reduction>=.02 if label=='coverage' else closure is not None and closure>=.25,
      wins_80pct=sum(d< -1e-9 for d in ds)>=8,ci_upper_negative=ci[1]<0,
      waiting_guard=av('covered',a,c,h,'waiting_violation')<=av('legacy',a,c,h,'waiting_violation')+.02,
      pending_guard=av('covered',a,c,h,'terminal_pending')<=av('legacy',a,c,h,'terminal_pending')+.05,
      specialized_preserved=av('covered',a,'specialized',12)<=1.05*av('legacy',a,'specialized',12),
      nominal_preserved=av('covered',a,'nominal',12)<=1.05*av('legacy',a,'nominal',12)) if label!='residual_gap' else {}
  status=('NOT_APPLICABLE' if label=='gap_closure' and oldgap<=0 else 'PASS' if all(gates.values()) else 'FAIL') if primary else 'EXPLORATORY'
  ident=f'{label}:{a}:{c}_H{h}'
  record=dict(primary=primary,status=status,mean_difference=statistics.mean(ds),ci=ci,confidence=confidence,wins=sum(d< -1e-9 for d in ds),legacy_cost=old,covered_cost=new,priced_reduction=reduction,legacy_gap_to_central=oldgap,covered_gap_to_central=newgap,gap_fraction_closed=closure,gates=gates,seed_differences=ds)
  logged=original['contrasts'][ident]
  for m in ('primary','status','wins','gates'):assert record[m]==logged[m],(ident,m)
  for m in ('mean_difference','legacy_cost','covered_cost','priced_reduction','legacy_gap_to_central','covered_gap_to_central'):close(record[m],logged[m])
  for x,y in zip(ci,logged['ci']):close(x,y)
  if closure is None:assert logged['gap_fraction_closed'] is None
  else:close(closure,logged['gap_fraction_closed'])
  contrasts[ident]=record
# Pair-count diagnostics, equality does not prove general equivalence.
pair_counts={}
for a in ARMS:
 for c,h in itertools.product(CONDS,(12,24)):
  count=Counter();seed_changes=Counter()
  for s,cs,shock in itertools.product(SEEDS,cfg['test_cohorts'],cfg['test_seeds']):
   old=episodes[('legacy',a,s,cs,c,h,shock)];new=episodes[('covered',a,s,cs,c,h,shock)]
   d=new[0]-old[0];count['total']+=1;count['better' if d<0 else 'worse' if d>0 else 'equal_cost']+=1
   count['equal_physical_metrics']+=new==old
   if d:seed_changes[str(s)]+=1
  pair_counts[f'{a}:{c}_H{h}']={**count,'changed_by_seed':dict(seed_changes)}
write('independent_analysis.json',dict(manifest={k:manifest[k] for k in ('status','commit','started','finished','actual_counts')},hash_audit=dict(files=411,source_snapshots=snapshot_count),aggregate=aggregate,contrasts=contrasts,pair_counts=pair_counts))
print('Episode grid, cost arithmetic, all56contrasts and aggregate statistics PASS',flush=True)
print(json.dumps({k:v for k,v in contrasts.items() if v['primary']},indent=2),flush=True)
# Actual four machine rows per episode, not inferred from episode count.
machine_count=0;previous=None;total=Counter();seen=set()
MACHINE=('objective','base_cost','maintenance_cost','failure_cost','unavailability_cost','waiting_cost','unavailable_ticks','failed_waiting_ticks','overdue_waiting_ticks','terminal_failed','terminal_pending')
def finish_machine(k,tot,seen):
 assert seen==set(range(4));ep=dict(zip(PHYS,episodes[k]))
 for m in MACHINE:close(tot[m],ep[m])
 close(tot['service_starts'],ep['jobs']);close(tot['violation']>0,ep['waiting_violation']);close(tot['max_wait'],ep['max_wait'])
for r in rows('machine_metrics.csv'):
 k=key(r)
 if previous is not None and k!=previous:finish_machine(previous,total,seen);total=Counter();seen=set()
 assert int(r['machine']) not in seen;seen.add(int(r['machine']));previous=k
 for m in MACHINE+('service_starts','violation'):total[m]+=float(r[m])
 total['max_wait']=max(total['max_wait'],float(r['max_wait']));machine_count+=1
finish_machine(previous,total,seen);assert machine_count==1269760
print('All1,269,760machine rows reconciled',flush=True)
# Request data, exact per-episode waiting decomposition including censoring.
request_groups=defaultdict(Counter);hist=defaultdict(lambda: {'served':Counter(),'censored':Counter()});request_count=0;per_episode=defaultdict(Counter);reqkeys=set()
for r in rows('request_metrics.csv'):
 k=key(r);assert k in episodes
 ident=k+(int(r['machine']),int(r['request']));assert ident not in reqkeys;reqkeys.add(ident)
 wait=int(r['wait']);arrival=int(r['arrival']);end=int(r['end']);served=r['served']=='True';censored=r['censored']=='True'
 assert served!=censored and wait==end-arrival and 0<=arrival<=end<=k[5]
 assert not censored or end==k[5]
 overdue=max(wait-4,0);scost=12*overdue if served else 0;ccost=12*overdue if censored else 0
 close(overdue,float(r['overdue_ticks']));close(scost,float(r['served_waiting_cost']));close(ccost,float(r['censored_waiting_cost']));close(12*overdue,float(r['observed_waiting_cost']))
 assert (r['overdue_observed']=='True')==(wait>4)
 values=dict(requests=1,served=int(served),censored=int(censored),overdue_ticks=overdue,served_waiting_cost=scost,censored_waiting_cost=ccost,overdue_requests=int(wait>4),served_overdue=int(served and wait>4),censored_overdue=int(censored and wait>4))
 g=gkey(r);request_groups[g].update(values);per_episode[k].update(values);hist[g]['served' if served else 'censored'][wait]+=1;request_count+=1
for k,vals in episodes.items():
 ep=dict(zip(PHYS,vals));q=per_episode[k]
 for qname,ename in [('requests','corrective_requests'),('served','served_requests'),('censored','censored_requests'),('overdue_ticks','overdue_waiting_ticks'),('overdue_requests','request_overdue_observed'),('served_waiting_cost','served_waiting_cost'),('censored_waiting_cost','censored_waiting_cost')]:close(q[qname],ep[ename])
assert request_count==manifest['request_rows']==858570
request_summary={}
for g,q in request_groups.items():
 out=dict(q);out.update(censored_fraction=q['censored']/q['requests'],censored_waiting_cost_fraction=q['censored_waiting_cost']/(q['censored_waiting_cost']+q['served_waiting_cost']) if q['censored_waiting_cost']+q['served_waiting_cost'] else None)
 for kind in ('served','censored'):
  hh=hist[g][kind];n=sum(hh.values());out[f'{kind}_wait_histogram']=dict(sorted(hh.items()));out[f'mean_{kind}_wait']=sum(w*n for w,n in hh.items())/n if n else None
 request_summary[':'.join((g[0],g[1],f'{g[2]}_H{g[3]}'))]=out
logged=js('request_wait_decomposition.json')
for g,q in logged.items():
 for m,v in q.items():close(v,request_summary[g][m])
write('request_audit.json',dict(rows=request_count,machine_rows=machine_count,groups=request_summary))
print('All858570requests reconciled incl.censored waiting cost',flush=True)
# Latency inventory, actual same states and checkpoint attributions; no physics replay.
lg=defaultdict(list);lkeys=set()
for r in rows('latency.csv'):
 k=(r['training_regime'],r['controller'],int(r['train_seed']));ident=k+(int(r['repeat']),int(r['state_index']))
 assert ident not in lkeys;lkeys.add(ident);lg[k].append(float(r['seconds']))
assert len(lkeys)==8928 and len(lg)==62 and all(len(v)==144 for v in lg.values())
latency={}
for reg,a in itertools.product(REGS,ARMS):
 values=list(itertools.chain.from_iterable(v for k,v in lg.items() if k[:2]==(reg,a)))
 latency[f'{reg}:{a}']=dict(mean_ms=1000*statistics.mean(values),count=len(values))
for a in cfg['controls']:latency[f'reference:{a}']=dict(mean_ms=1000*statistics.mean(lg[('reference',a,-1)]),count=144)
write('latency_audit.json',latency)
# Training rows stream: seed sequence derived on central_legacy; all others match digests.
coverage_by={(r['training_regime'],r['algorithm'],r['train_seed']):r for r in coverage}
train_counter=Counter();seedhash={};config_hash={};nominal_hash={};mix=defaultdict(Counter);maxerr=0.;bad=0
for i,r in enumerate(rows('training_episodes.csv'),1):
 k=(r['training_regime'],r['algorithm'],int(r['train_seed']));idx=int(r['episode']);train_counter[k]+=1
 assert idx==train_counter[k] and idx<=40960 and r['condition']==SCHEDULES[k[0]][(idx-1)%5]
 assert int(r['rollout_update'])==(idx-1)//128+1
 assert int(r['physical_steps'])==((idx-1)//32+1)*384
 cs=int(r['config_seed']);es=int(r['environment_seed'])
 if k[0]=='legacy' and k[1]==C:
  def role(role):return int.from_bytes(hashlib.sha256(f'{k[2]}:{role}'.encode()).digest()[:8],'big')
  assert cs==role(f'solver_comparison_v1_configuration:{idx}') and es==role(f'solver_comparison_v1_environment:{idx}')
 seedhash.setdefault(k,hashlib.sha256()).update(f'{idx}:{cs}:{es};'.encode())
 config_hash.setdefault(k,hashlib.sha256()).update(f"{idx}:{r['condition']}:{r['config_sha256']};".encode())
 if r['condition']=='nominal':nominal_hash.setdefault(k,hashlib.sha256()).update(f"{idx}:{r['config_sha256']};".encode())
 mix[k][r['condition']]+=1
 obj=float(r['objective']);basecost=float(r['base_cost']);waitcost=float(r['waiting_cost'])
 close(obj,basecost+waitcost);close(basecost,float(r['maintenance_cost'])+float(r['failure_cost'])+float(r['unavailability_cost']))
 close(waitcost,12*float(r['overdue_waiting_ticks']));close(float(r['failure_cost']),15*float(r['failures']));close(float(r['unavailability_cost']),6*float(r['unavailable_ticks']))
 close(sum(json.loads(r['machine_costs'])),obj);assert float(r['invalid_assignments'])==0
 maxerr=max(maxerr,float(r['reward_reconciliation_error']))
 if i%409600==0:print(f'Training rows checked {i}/2457600',flush=True)
assert sum(train_counter.values())==2457600 and len(train_counter)==60 and set(train_counter.values())=={40960}
assert maxerr<=5e-5
for k,n in train_counter.items():
 assert seedhash[k].hexdigest()==coverage_by[k]['training_seed_sequence_sha256']
 assert dict(mix[k])==coverage_by[k]['mixture_counts']
for s in SEEDS:
 keys=[k for k in train_counter if k[2]==s]
 assert len({seedhash[k].hexdigest() for k in keys})==1
 assert len({nominal_hash[k].hexdigest() for k in keys})==1
 for reg in REGS:assert len({config_hash[k].hexdigest() for k in keys if k[0]==reg})==1
# Rollout logs verify loss finite and exact optimizer steps; expose gradients/entropy for descriptive use.
progress=defaultdict(list)
for r in rows('training_progress.csv'):
 k=(r['training_regime'],r['algorithm'],int(r['train_seed']));u=int(r['rollout_update']);assert u==len(progress[k])+1
 assert int(r['physical_steps'])==1536*u and int(r['episodes'])==128*u and int(r['optimizer_steps'])==32*u
 numeric={m:float(v) for m,v in r.items() if m not in ('training_regime','algorithm')};assert all(math.isfinite(v) for v in numeric.values())
 progress[k].append(numeric)
assert len(progress)==60 and all(len(v)==320 for v in progress.values())
write('training_audit.json',dict(rows=sum(train_counter.values()),progress_rows=sum(map(len,progress.values())),maximum_reward_reconciliation_error=maxerr,seed_sequences_matched=True,nominal_configurations_matched=True,within_regime_configuration_sequences_matched=True,
      replicas={':'.join(map(str,k)):dict(mixture=dict(mix[k]),seed_sequence_sha256=seedhash[k].hexdigest(),last20_progress_mean={m:statistics.mean(v[m] for v in progress[k][-20:]) for m in ('loss','policy_loss','value_loss','entropy','gradient_norm','clipped_fraction','mean_episode_cost')}) for k in train_counter}))
print('Training audit PASS; ready for interpretation',flush=True)
