import csv,gc,hashlib,itertools,json,math,random,statistics,subprocess,sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from tqdm import tqdm
root=Path.cwd();raw=root/'lab_results/maintenance_production_full_cpu_20261004T155224Z'
out=root/'reports'/('maintenance_production_impact_analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
out.mkdir(parents=True,exist_ok=False);(out/'analyze.py').write_bytes(Path(__file__).read_bytes())
def load(n):return json.loads((raw/n).read_text())
def read(n):
 with (raw/n).open(newline='') as f:return list(csv.DictReader(f))
def save(n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
mf=load('manifest.json');panel=mf['panel'];stored=load('summary.json');policy=load('policy.json')
assert mf['status']=='COMPLETED' and mf['profile']=='full' and not mf['source_dirty']
assert mf['source_commit']=='27abfabd6a9787ddaeabad0f2caeddd99cdb2801'
assert all((raw/n).is_file() for n in mf['outputs'])
pkg=out/'pinned'/'ht_pdm_fjsp';pkg.mkdir(parents=True);(pkg/'__init__.py').write_text('')
for source,key,dest in [('src/ht_pdm_fjsp/maintenance_production_impact.py','source_sha256',pkg/'maintenance_production_impact.py'),('src/ht_pdm_fjsp/maintenance_coupling.py','dependency_sha256',pkg/'maintenance_coupling.py'),('docs/maintenance_production_impact_v1_plan.md','protocol_sha256',out/'protocol_snapshot.md')]:
 data=subprocess.check_output(['git','show',mf['source_commit']+':'+source]);assert hashlib.sha256(data).hexdigest()==mf[key];dest.write_bytes(data)
sys.path.insert(0,str(out/'pinned'))
from ht_pdm_fjsp import maintenance_production_impact as m
assert panel==m.panels('full')
rows=read('episodes.csv');dev=read('development.csv');tr=read('decisions.csv');jobs=read('job_completions.csv');co=read('coordination.csv');cfgs=load('resolved_config.json')
fields=('instance_seed','pressure','skill','length','episode_seed','policy');cf=fields[:4]
def key(r):return tuple(r[f] for f in fields)
expected=set(itertools.product(map(str,panel['configs']),('low','high'),('uniform','heterogeneous'),map(str,panel['lengths']),map(str,panel['episodes']),m.POLICIES))
assert len(rows)==len({key(r) for r in rows})==11520 and {key(r) for r in rows}==expected
assert len(dev)==2304 and len(tr)==121703 and len(jobs)==138240 and len(co)==len(cfgs)==192
assert (raw/'episodes.csv').read_bytes()==(raw/'episodes.partial.csv').read_bytes()
assert not set(panel['configs'])&set(panel['dev_configs']) and not set(panel['episodes'])&set(panel['dev_episodes'])
numeric=[f for f in rows[0] if f not in fields+('feasibility_audit','work_balance_audit')]
for r in rows+dev:
 assert r['feasibility_audit']==r['work_balance_audit']=='PASS'
 assert all(math.isfinite(float(r[f])) for f in numeric) and float(r['invalid_requests'])==0
 assert float(r['productive_steps'])==float(r['required_work']) and int(r['completed_jobs'])==6*int(r['length'])
 assert float(r['makespan'])>=float(r['ideal_makespan']) and math.isclose(float(r['normalized_makespan']),float(r['makespan'])/float(r['ideal_makespan']),abs_tol=1e-10)
 assert math.isclose(float(r['cost']),sum(float(r['cost_'+f]) for f in m.COMPONENTS),abs_tol=1e-8)
de=set()
for pol,grid in [(m.POLICIES[1],(0,1,2,4)),(m.POLICIES[2],(0,2,4,6))]:de.update(itertools.product([pol],map(str,grid),map(str,panel['dev_configs']),('low','high'),('uniform','heterogeneous'),map(str,panel['lengths']),map(str,panel['dev_episodes'])))
assert {tuple(r[f] for f in ('policy','parameter')+fields[:-1]) for r in dev}==de
devmeans={}
for pol,param,grid in [(m.POLICIES[1],'offset',(0,1,2,4)),(m.POLICIES[2],'lead',(0,2,4,6))]:
 scores={v:statistics.mean(float(r['normalized_makespan']) for r in dev if r['policy']==pol and int(r['parameter'])==v) for v in grid}
 assert policy[param]==min(grid,key=lambda v:(scores[v],v));devmeans[param]=scores
expected_cfg=set(itertools.product(map(str,panel['configs']),('low','high'),('uniform','heterogeneous'),map(str,panel['lengths'])))
assert {tuple(str(r[f]) for f in cf) for r in cfgs}=={tuple(r[f] for f in cf) for r in co}==expected_cfg
assert all(r['feasibility_audit']==r['work_balance_audit']=='PASS' and int(r['invalid_requests'])==0 for r in co)
for r in cfgs:assert json.loads(json.dumps(asdict(m.scenario(*(r[f] for f in cf),6))))==r['config']
for i in panel['configs']:
 cs=[m.scenario(i,p,s,64,6) for p,s in m.CELLS]
 assert len({(c.jobs,c.maintenance.initial_ages,c.maintenance.failure_age,c.maintenance.probability) for c in cs})==1
 assert len({tuple(statistics.mean(row) for row in c.maintenance.service) for c in cs})==1

# Independent arithmetic/bootstrap on instance means, with locked draw seed.
groups=defaultdict(list)
for r in rows:groups[tuple(r[f] for f in cf)+(r['policy'],)].append(r)
means={k:{f:statistics.mean(float(r[f]) for r in rs) for f in numeric} for k,rs in groups.items()}
def boot(v):
 rng=random.Random(882100);n=len(v);draws=sorted(sum(v[int(rng.random()*n)] for _ in range(n))/n for _ in range(10000))
 return dict(mean=statistics.mean(v),ci95=[draws[250],draws[9750]],n_instances=n)
def assert_endpoint(a,b):
 assert a['n_instances']==b['n_instances'] and math.isclose(a['mean'],b['mean'],abs_tol=1e-10)
 assert all(math.isclose(x,y,abs_tol=1e-10) for x,y in zip(a['ci95'],b['ci95']))
cells={};instances=[]
def vals(p,sk,l,pol,f='makespan'):return [means[str(i),p,sk,str(l),pol][f] for i in panel['configs']]
I,H,S=m.POLICIES
for p,sk,l in itertools.product(('low','high'),('uniform','heterogeneous'),panel['lengths']):
 name=f'{p}_{sk}_jobs{l}';iv,hv,sv=(vals(p,sk,l,pol) for pol in m.POLICIES)
 c={'metrics':{pol:{f:statistics.mean(vals(p,sk,l,pol,f)) for f in numeric} for pol in m.POLICIES}}
 for pol in m.POLICIES:
  for f,v in stored['cells'][name]['metrics'][pol].items():assert math.isclose(v,c['metrics'][pol][f],abs_tol=1e-10)
 for label,values in [('H_minus_S',[x-y for x,y in zip(hv,sv)]),('H_minus_S_relative',[(x-y)/x for x,y in zip(hv,sv)]),('I_minus_H',[x-y for x,y in zip(iv,hv)]),('H_minus_S_late_service',[x-y for x,y in zip(vals(p,sk,l,H,'late_active_service_steps'),vals(p,sk,l,S,'late_active_service_steps'))])]:
  c[label]=boot(values);assert_endpoint(c[label],stored['cells'][name][label])
 c['H_minus_S_cost_relative']=boot([(x-y)/x for x,y in zip(vals(p,sk,l,H,'cost'),vals(p,sk,l,S,'cost'))])
 c['S_vs_H_counts']=dict(wins=sum(x<y-1e-8 for x,y in zip(sv,hv)),ties=sum(abs(x-y)<1e-8 for x,y in zip(sv,hv)),losses=sum(x>y+1e-8 for x,y in zip(sv,hv)))
 for i in panel['configs']:
  record=dict(instance_seed=i,pressure=p,skill=sk,length=l)
  for pol in m.POLICIES:record.update({pol[0]+'_'+f:means[str(i),p,sk,str(l),pol][f] for f in numeric})
  instances.append(record)
 cells[name]=c
hhu,hhe,lhu,lhe=[vals(p,sk,64,H,'normalized_makespan') for p,sk in [('high','uniform'),('high','heterogeneous'),('low','uniform'),('low','heterogeneous')]]
interaction=boot([(a-b)-(c-d) for a,b,c,d in zip(hhe,hhu,lhe,lhu)]);assert_endpoint(interaction,stored['interaction'])
capacity={sk:boot([a-b for a,b in zip(vals('high',sk,64,H,'normalized_makespan'),vals('low',sk,64,H,'normalized_makespan'))]) for sk in ('uniform','heterogeneous')}
for sk in capacity:assert_endpoint(capacity[sk],stored['capacity_burden'][sk])
pr=cells['high_heterogeneous_jobs64'];sh=cells['high_heterogeneous_jobs16']
gates=dict(recovery=pr['H_minus_S_relative']['mean']>=.05 and pr['H_minus_S']['ci95'][0]>0,resource_specific_burden=interaction['ci95'][0]>0,persistence=pr['H_minus_S_relative']['mean']>=.5*max(sh['H_minus_S_relative']['mean'],0) and pr['metrics'][H]['late_waiting']>0)
gates['problem_support']=all(gates.values());assert gates==stored['gates']
ts={key(r)+(r['step'],):r for r in tr};jc={key(r)+(r['machine'],r['job']):r for r in jobs}
expected_tr={key(r)+(str(t),) for r in rows if r['episode_seed']==str(panel['episodes'][0]) for t in range(int(r['makespan']))}
expected_jc={key(r)+(str(machine),str(job)) for r in rows if r['episode_seed']==str(panel['episodes'][0]) for machine in range(6) for job in range(int(r['length']))}
assert len(ts)==len(tr) and set(ts)==expected_tr and len(jc)==len(jobs) and set(jc)==expected_jc
bykey={key(r):r for r in rows};nrep=0;nt=0;nj=0;maxerr=0.
for r in tqdm(cfgs,desc='replay stored production trace panel',unit='cell'):
 cfg=m.scenario(*(r[f] for f in cf),6)
 for pol in m.POLICIES:
  identity=tuple(str(r[f]) for f in cf)+(str(panel['episodes'][0]),pol)
  result,trace,completion=m.episode(cfg,pol,panel['episodes'][0],offset=policy['offset'],lead=policy['lead'],trace=True)
  for f,v in result.items():
   if f=='decision_seconds':continue
   if isinstance(v,str):assert v==bykey[identity][f]
   else:
    error=abs(v-float(bykey[identity][f]));maxerr=max(maxerr,error);assert error<1e-8
  for t in trace:
   old=ts[identity+(str(t['step']),)]
   for f,v in t.items():
    if f in ('state','work','action','next_state','next_work'):assert json.loads(v)==json.loads(old[f])
    else:assert math.isclose(float(old[f]),v,abs_tol=1e-8)
   nt+=1
  for job in completion:
   old=jc[identity+(str(job['machine']),str(job['job']))]
   assert all(int(old[f])==v for f,v in job.items());nj+=1
  nrep+=1
 del cfg;gc.collect()
assert nt==len(tr) and nj==len(jobs)
# How often a failure is followed immediately by service rather than additional
# lost-production waiting. Descriptive mechanism, only recorded shock panel.
failure_events=0;immediate_service=0
for r in tr:
 if r['pressure']!='high' or r['skill']!='heterogeneous' or r['length']!='64' or r['policy']!=H or not int(r['failures']):continue
 before=json.loads(r['state']);after=json.loads(r['next_state'])
 new=[i for i,(a,b) in enumerate(zip(before['failed'],after['failed'])) if not a and b]
 nxt=ts.get(key(r)+(str(int(r['step'])+1),));failure_events+=len(new)
 if nxt:
  servicing=json.loads(nxt['next_state'])['assigned'];immediate_service+=sum(i in servicing for i in new)
audit=dict(verification_status='ANALYZED',raw=str(raw),source_commit=mf['source_commit'],hashes_match=True,full_cartesian_complete=True,episodes=len(rows),development=len(dev),coordination=len(co),decisions=len(tr),job_completions=len(jobs),summary_arithmetic_match=True,development_selection_matches=True,replayed_episodes=nrep,replayed_decisions=nt,replayed_job_completions=nj,max_replay_error=maxerr,independent_reproduction=False,replay_scope='first shared shock per cell; entire stored trace/completion panel, pinned common implementation',file_sha256={n:hashlib.sha256((raw/n).read_bytes()).hexdigest() for n in mf['outputs']})
save('audit.json',audit);save('statistics.json',dict(cells=cells,interaction=interaction,capacity_burden=capacity,gates=gates,development=devmeans,primary_H_trace_failures=failure_events,primary_H_trace_next_tick_service=immediate_service))
with (out/'instances.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(instances[0]));w.writeheader();w.writerows(instances)
print('ANALYSIS_DIR',out,flush=True)
print(json.dumps(dict(primary=pr,interaction=interaction,capacity=capacity,gates=gates,development=devmeans,failures=failure_events,immediate_service=immediate_service),indent=2),flush=True)
