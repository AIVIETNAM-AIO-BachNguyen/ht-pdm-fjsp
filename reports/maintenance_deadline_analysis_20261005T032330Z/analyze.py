import csv, hashlib, itertools, json, math, random, statistics, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl')
RAW=ROOT/'lab_results/maintenance_deadline_full_cpu_20261005T023359Z'
OUT=ROOT/'reports'/('maintenance_deadline_analysis_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
OUT.mkdir();(OUT/'analyze.py').write_text(Path(__file__).read_text())
manifest=json.loads((RAW/'manifest.json').read_text());assert manifest['status']=='COMPLETED' and not manifest['source_dirty']
commit='a7089acfd9074a4c1e7c9023673f0f39446562b4';assert manifest['source_commit']==commit
PIN=OUT/'pinned'/'ht_pdm_fjsp';PIN.mkdir(parents=True);(PIN/'__init__.py').write_text('')
files=dict(source='maintenance_deadline_packing.py',repair='maintenance_repair_impact.py',production='maintenance_production_impact.py',coupling='maintenance_coupling.py')
for key,name in files.items():
 data=subprocess.check_output(['git','show',f'{commit}:src/ht_pdm_fjsp/{name}'],cwd=ROOT)
 assert hashlib.sha256(data).hexdigest()==manifest['hashes'][key];(PIN/name).write_bytes(data)
protocol=subprocess.check_output(['git','show',f'{commit}:docs/maintenance_deadline_packing_v1_plan.md'],cwd=ROOT)
assert hashlib.sha256(protocol).hexdigest()==manifest['hashes']['protocol'];(OUT/'protocol_snapshot.md').write_bytes(protocol)
sys.path.insert(0,str(PIN.parent));from ht_pdm_fjsp import maintenance_deadline_packing as m
panel=manifest['panel'];assert panel==m.panels('full');policy=json.loads((RAW/'policy.json').read_text())['settings']
template,_,_=m.episode(m.scenario(930100,'low','uniform',3,1,4),m.POLICIES[0],931100)
numeric={k for k,v in template.items() if isinstance(v,(int,float))}|{'instance_seed','length','ratio','episode_seed'}
def read(name):
 with (RAW/name).open() as f:return [{k:float(v) if k in numeric else v for k,v in r.items()} for r in csv.DictReader(f)]
rows=read('episodes.csv');dev=read('development.csv');m.validate(rows,panel);m.validate(dev,panel,True)
assert (RAW/'episodes.csv').read_bytes()==(RAW/'episodes.partial.csv').read_bytes()
for ratio in (1,2,4):
 for pol in m.POLICIES:assert policy[str(ratio)][pol]==m.select(dev,ratio,pol)
configs=json.loads((RAW/'resolved_config.json').read_text());assert len(configs)==576
for r in configs:
 p=m.scenario(r['instance_seed'],r['pressure'],r['skill'],r['length'],r['ratio'],6)
 assert json.loads(json.dumps(m.asdict(p)))==r['config']
with (RAW/'coordination.csv').open() as f:
 coord=list(csv.DictReader(f));assert len(coord)==576 and all(x['feasibility_audit']==x['work_balance_audit']=='PASS' and float(x['invalid_requests'])==0 for x in coord)
 print('AUDIT: complete source, Cartesian panels, development minima, resolved scenarios, cost/work feasibility',flush=True)
# Independent instance aggregates and locked percentile bootstrap.
rng=random.Random(942100);idx=np.array([rng.choices(range(24),k=24) for _ in range(10000)])
def boot(v):
 arr=np.asarray(v);draw=np.sort(arr[idx].mean(axis=1))
 return dict(mean=float(arr.mean()),ci95=[float(draw[250]),float(draw[9750])],n_instances=len(v))
g=defaultdict(list)
for r in rows:g[tuple(r[k] for k in ('instance_seed','pressure','skill','length','ratio','policy'))].append(r)
seeds=panel['configs'];instances=[];stats={}
for (p,sk),l,ratio in itertools.product(m.CELLS,panel['lengths'],panel['ratios']):
 records=[]
 for i in seeds:
  x=dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=ratio)
  for pol,label in zip(m.POLICIES,('H','S','D')):
   rs=g[i,p,sk,l,ratio,pol];vals=[r['makespan'] for r in rs]
   for f in ('makespan','failures','waiting','failed_steps','planned_steps','pm_service_steps','cm_service_steps','pm_jobs','cm_jobs','cost'):
    x[label+'_'+f]=statistics.mean(r[f] for r in rs)
   x[label+'_p95']=sorted(vals)[18];x[label+'_worst20']=statistics.mean(sorted(vals)[-4:]);x[label+'_max']=max(vals)
  records.append(x);instances.append(x)
 key=f'{p}_{sk}_jobs{l}_ratio{ratio}'
 stats[key]={}
 for name,values in [('D_vs_S_relative',[(x['D_makespan']-x['S_makespan'])/x['S_makespan'] for x in records]),('D_vs_H_relative',[(x['D_makespan']-x['H_makespan'])/x['H_makespan'] for x in records]),('H_recovery_D',[(x['H_makespan']-x['D_makespan'])/x['H_makespan'] for x in records]),('D_minus_S_absolute',[x['D_makespan']-x['S_makespan'] for x in records]),('D_vs_S_p95_relative',[(x['D_p95']-x['S_p95'])/x['S_p95'] for x in records]),('D_vs_H_p95_relative',[(x['D_p95']-x['H_p95'])/x['H_p95'] for x in records])]:stats[key][name]=boot(values)
 stats[key]['D_wins_S']=sum(x['D_makespan']<x['S_makespan'] for x in records);stats[key]['D_ties_S']=sum(x['D_makespan']==x['S_makespan'] for x in records);stats[key]['D_losses_S']=sum(x['D_makespan']>x['S_makespan'] for x in records)
 stats[key]['D_wins_H']=sum(x['D_makespan']<x['H_makespan'] for x in records)
summary=json.loads((RAW/'summary.json').read_text())
for key,r in stats.items():
 for name in ('D_vs_S_relative','D_vs_H_relative','H_recovery_D','D_minus_S_absolute','D_vs_S_p95_relative','D_vs_H_p95_relative'):
  assert abs(r[name]['mean']-summary['cells'][key][name]['mean'])<1e-12
  assert all(abs(a-b)<1e-12 for a,b in zip(r[name]['ci95'],summary['cells'][key][name]['ci95']))
gates=dict(mean_adequacy=stats['high_heterogeneous_jobs64_ratio2']['D_vs_S_relative']['ci95'][1]<=.02,heterogeneous_tail_guard=stats['high_heterogeneous_jobs64_ratio4']['D_vs_S_p95_relative']['ci95'][1]<=.05,uniform_tail_guard=stats['high_uniform_jobs64_ratio4']['D_vs_H_p95_relative']['ci95'][1]<=.05)
gates['overall_adequacy']=all(gates.values());assert gates==summary['gates']
# Full stored first-shock replay streamed: bounded memory, no fresh full run.
identity=('instance_seed','pressure','skill','length','ratio','episode_seed','policy')
def key_of(r):return tuple(float(r[k]) if k in numeric else r[k] for k in identity)
def grouped(file,key):
 with (RAW/file).open() as f:
  for ident,part in itertools.groupby(csv.DictReader(f),key=key):yield ident,list(part)
trace_iter=iter(grouped('decisions.csv',key_of));job_iter=iter(grouped('job_completions.csv',key_of))
bykey={tuple(r[k] for k in identity):r for r in rows};maxerror=0.;nt=nj=ne=0;plans=0;dispatch=defaultdict(int)
for ident,tt in trace_iter:
 jobident,jj=next(job_iter);assert jobident==ident
 i,p,sk,l,ratio,e,pol=ident;cfg=m.scenario(int(i),p,sk,int(l),int(ratio),6)
 r,ts,cs=m.episode(cfg,pol,int(e),trace=True,**policy[str(int(ratio))][pol]);expected=bykey[ident]
 for k,v in r.items():
  if k in numeric and k!='decision_seconds':maxerror=max(maxerror,abs(expected[k]-v));assert expected[k]==v
 assert len(ts)==len(tt) and len(cs)==len(jj)
 for a,b in zip(ts,tt):
  for k,v in a.items():
   if isinstance(v,(str,bool)):assert str(v)==b[k]
   else:maxerror=max(maxerror,abs(v-float(b[k])));assert v==float(b[k])
 for a,b in zip(cs,jj):assert all(v==int(float(b[k])) for k,v in a.items())
 # Separate deadline-slot invariants and observed dispatch duration counts.
 for t in ts:
  before=json.loads(t['state']);after=json.loads(t['next_state'])
  for worker,machine in enumerate(after['assigned']):
   if machine>=0 and before['remaining'][worker]==0:
    d=cfg.maintenance.service[machine][worker]
    dispatch[int(i),p,sk,int(l),int(ratio),pol,'CM' if before['failed'][machine] else 'PM',d]+=1
  if pol==m.POLICIES[2]:
   plans+=1;slots=json.loads(t['deadline_plan']);a=json.loads(t['action'])
   assert t['waiting']==0 and len({x['machine'] for x in slots})==len(slots)
   for worker in range(len(before['remaining'])):
    chain=[x for x in slots if x['worker']==worker]
    shift=max(0,min((x['deadline']-x['forward_start'] for x in chain),default=0))
    previous=before['remaining'][worker]+sum(cfg.maintenance.service[machine][worker]*(cfg.ratio if before['failed'][machine] else 1) for machine in before['queues'][worker])
    for slot in chain:
     assert slot['start']==slot['forward_start']+shift and slot['shift']==shift and slot['start']>=previous and slot['start']>=slot['release']
     assert slot['late']==max(0,slot['start']-slot['deadline'])
     if slot['start']==0:assert a[slot['machine']]==slot['worker']+1 and not before['remaining'][slot['worker']] and not before['queues'][slot['worker']]
     previous=slot['start']+slot['duration']

 nt+=len(tt);nj+=len(jj);ne+=1
 if ne%288==0:print(f'REPLAY: {ne}/1728 first-shock episodes; {nt} decisions',flush=True)
assert next(job_iter,None) is None and nt==summary['audits']['decisions'] and nj==414720 and ne==1728
# Opened regressions are audited separately from the scientific panel.
reg=read('regression_episodes.csv');expected={(record['instance_seed'],e,pol) for record in panel['regressions'] for e,pol in itertools.product(record['episodes'],m.POLICIES)}
assert len(reg)==180 and {(int(r['instance_seed']),int(r['episode_seed']),r['policy']) for r in reg}==expected
regkey=lambda r:(int(r['instance_seed']),int(r['episode_seed']),r['policy'])
regjobs=iter(grouped('regression_job_completions.csv',regkey));rtotal=rjtotal=0
regmap={(int(r['instance_seed']),int(r['episode_seed']),r['policy']):r for r in reg}
regconfigs={record['instance_seed']:m.scenario(record['instance_seed'],'high','heterogeneous',64,4,6) for record in panel['regressions']}
for (i,e,pol),stored in grouped('regression_decisions.csv',regkey):
 ident,jobs=next(regjobs);assert ident==(i,e,pol)
 params=dict(offset=4) if pol==m.POLICIES[0] else dict(lead=6) if pol==m.POLICIES[1] else policy['4'][pol]
 r,ts,cs=m.episode(regconfigs[i],pol,e,trace=True,**params);rr=regmap[i,e,pol]
 assert all(rr[k]==v for k,v in r.items() if k in numeric and k!='decision_seconds')
 assert len(ts)==len(stored) and len(cs)==len(jobs)
 for a,b in zip(ts,stored):assert all(str(v)==b[k] if isinstance(v,str) else v==float(b[k]) for k,v in a.items())
 for a,b in zip(cs,jobs):assert all(v==int(float(b[k])) for k,v in a.items())
 rtotal+=len(ts);rjtotal+=len(cs)
assert next(regjobs,None) is None and rjtotal==3456 and rtotal==summary['regressions']['decisions']
for rr in reg:
 cfg=regconfigs[int(rr['instance_seed'])];params=dict(offset=4) if rr['policy']==m.POLICIES[0] else dict(lead=6) if rr['policy']==m.POLICIES[1] else policy['4'][rr['policy']]
 r,_,_=m.episode(cfg,rr['policy'],int(rr['episode_seed']),**params)
 assert all(rr[k]==v for k,v in r.items() if k in numeric and k!='decision_seconds')
 for field in ('makespan','failures'):
  if field not in summary['regressions']['cases'][str(int(rr['instance_seed']))][rr['policy']]:continue
  actual=statistics.mean(x[field] for x in reg if x['instance_seed']==rr['instance_seed'] and x['policy']==rr['policy'])
  assert actual==summary['regressions']['cases'][str(int(rr['instance_seed']))][rr['policy']][field]
# Source-matched old H/S case results must reproduce earlier transferred runs.
old_backlog=ROOT/'lab_results/maintenance_backlog_full_cpu_20261004T173740Z/episodes.csv'
old_repair=ROOT/'lab_results/maintenance_repair_full_cpu_20261004T162209Z/episodes.csv'
for path,target in [(old_backlog,{920211,920216}),(old_repair,{900215})]:
 with path.open() as f:
  for r in csv.DictReader(f):
   if int(r['instance_seed']) not in target or r['ratio']!='4' or r['pressure']!='high' or r['skill']!='heterogeneous' or r['length']!='64' or r['policy'] not in m.POLICIES[:2]:continue
   rr=regmap[int(r['instance_seed']),int(r['episode_seed']),r['policy']]
   assert all(rr[k]==float(r[k]) for k in template if k in numeric and k!='decision_seconds')
def savecsv(name,data):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
savecsv('instances.csv',instances)
savecsv('observed_dispatch.csv',[dict(zip(('instance_seed','pressure','skill','length','ratio','policy','type','pm_row_duration'),key),count=count) for key,count in dispatch.items()])
choices=[]
for ratio in (1,2,4):
 for pol in m.POLICIES:
  for params in m.grid(pol):
   rs=[r for r in dev if r['ratio']==ratio and r['policy']==pol and r['parameter']==json.dumps(params,sort_keys=True)]
   choices.append(dict(ratio=ratio,policy=pol,parameter=json.dumps(params,sort_keys=True),normalized_makespan=statistics.mean(r['normalized_makespan'] for r in rs),selected=params==policy[str(ratio)][pol]))
savecsv('development_choices.csv',choices)
(OUT/'statistics.json').write_text(json.dumps(dict(cells=stats,gates=gates),indent=2))
audit=dict(status='PASS',raw=str(RAW),source_commit=commit,all_hashes_match=True,eval=len(rows),development=len(dev),configs=len(configs),replayed_first_shock_episodes=ne,replayed_decisions=nt,replayed_completions=nj,deadline_plans_independently_audited=plans,regression_episodes_replayed=len(reg),regression_decisions_replayed=rtotal,regression_completions_replayed=rjtotal,old_case_H_S_match=True,max_numeric_error=maxerror,independent_bootstrap_matches=True,development_selection_matches=True,scope='shared-implementation replay and independent statistics/slot invariants; not independent simulator reproduction',raw_unmodified=True)
(OUT/'audit.json').write_text(json.dumps(audit,indent=2));print('REPORT_DIR',OUT,flush=True);print(json.dumps(audit),flush=True)
