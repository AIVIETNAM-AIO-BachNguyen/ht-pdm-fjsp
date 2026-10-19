import csv, hashlib, itertools, json, math, random, statistics, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl')
RAW=ROOT/'lab_results/maintenance_backlog_full_cpu_20261004T173740Z'
OUT=ROOT/'reports'/('maintenance_backlog_analysis_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
OUT.mkdir();(OUT/'analyze.py').write_text(Path(__file__).read_text())
manifest=json.loads((RAW/'manifest.json').read_text());assert manifest['status']=='COMPLETED' and not manifest['source_dirty']
commit='3be08ac5550a64603ce88634f1043a3e00c6817f';assert manifest['source_commit']==commit
PIN=OUT/'pinned'/'ht_pdm_fjsp';PIN.mkdir(parents=True);(PIN/'__init__.py').write_text('')
files=dict(source='maintenance_backlog_adaptive.py',repair='maintenance_repair_impact.py',production='maintenance_production_impact.py',coupling='maintenance_coupling.py')
for key,name in files.items():
 data=subprocess.check_output(['git','show',f'{commit}:src/ht_pdm_fjsp/{name}'],cwd=ROOT)
 assert hashlib.sha256(data).hexdigest()==manifest['hashes'][key];(PIN/name).write_bytes(data)
protocol=subprocess.check_output(['git','show',f'{commit}:docs/maintenance_backlog_adaptive_v1_plan.md'],cwd=ROOT)
assert hashlib.sha256(protocol).hexdigest()==manifest['hashes']['protocol'];(OUT/'protocol_snapshot.md').write_bytes(protocol)
sys.path.insert(0,str(PIN.parent));from ht_pdm_fjsp import maintenance_backlog_adaptive as m
panel=manifest['panel'];assert panel==m.panels('full');policy=json.loads((RAW/'policy.json').read_text())['settings']
template,_,_=m.episode(m.scenario(910100,'low','uniform',3,1,4),m.POLICIES[0],911100)
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
# Independently compute all instance aggregates and frozen percentile bootstrap.
rng=random.Random(922100);idx=np.array([rng.choices(range(24),k=24) for _ in range(10000)])
def boot(v):
 arr=np.asarray(v);assert len(arr)==24;draw=np.sort(arr[idx].mean(axis=1))
 return dict(mean=float(arr.mean()),ci95=[float(draw[250]),float(draw[9750])],n_instances=24)
g=defaultdict(list)
for r in rows:g[tuple(r[k] for k in ('instance_seed','pressure','skill','length','ratio','policy'))].append(r)
seeds=panel['configs'];instances=[];stats={}
for (p,sk),l,ratio in itertools.product(m.CELLS,panel['lengths'],panel['ratios']):
 records=[]
 for i in seeds:
  x=dict(instance_seed=i,pressure=p,skill=sk,length=l,ratio=ratio)
  for pol,label in zip(m.POLICIES,('H','A','S')):
   rs=g[i,p,sk,l,ratio,pol];vals=[r['makespan'] for r in rs]
   for f in ('makespan','failures','waiting','failed_steps','planned_steps','pm_service_steps','cm_service_steps','pm_jobs','cm_jobs','cost'):
    x[label+'_'+f]=statistics.mean(r[f] for r in rs)
   x[label+'_p95']=sorted(vals)[18];x[label+'_worst20']=statistics.mean(sorted(vals)[-4:]);x[label+'_max']=max(vals)
  x['A_vs_S_relative']=(x['A_makespan']-x['S_makespan'])/x['S_makespan'];x['H_minus_A_relative']=(x['H_makespan']-x['A_makespan'])/x['H_makespan']
  x['A_vs_S_p95_relative']=(x['A_p95']-x['S_p95'])/x['S_p95'];records.append(x);instances.append(x)
 key=f'{p}_{sk}_jobs{l}_ratio{ratio}'
 stats[key]=dict(A_vs_S_relative=boot([r['A_vs_S_relative'] for r in records]),H_minus_A_relative=boot([r['H_minus_A_relative'] for r in records]),A_vs_S_p95_relative=boot([r['A_vs_S_p95_relative'] for r in records]),S_recovery_relative_A=boot([(r['A_makespan']-r['S_makespan'])/r['A_makespan'] for r in records]),S_minus_A_absolute=boot([r['S_makespan']-r['A_makespan'] for r in records]),wins_S=sum(r['S_makespan']<r['A_makespan'] for r in records),ties=sum(r['S_makespan']==r['A_makespan'] for r in records),losses_S=sum(r['S_makespan']>r['A_makespan'] for r in records))
summary=json.loads((RAW/'summary.json').read_text())
for key,r in stats.items():
 for name in ('A_vs_S_relative','H_minus_A_relative','A_vs_S_p95_relative'):
  assert abs(r[name]['mean']-summary['cells'][key][name]['mean'])<1e-12
  assert all(abs(a-b)<1e-12 for a,b in zip(r[name]['ci95'],summary['cells'][key][name]['ci95']))
primary=stats['high_heterogeneous_jobs64_ratio2']['A_vs_S_relative'];tail=stats['high_heterogeneous_jobs64_ratio4']['A_vs_S_p95_relative']
gates=dict(mean_adequacy=primary['ci95'][1]<=.02,tail_guard=tail['ci95'][1]<=.05);gates['overall_adequacy']=all(gates.values());assert gates==summary['gates']
# Exact outcome equality when gain0; runtime is observational.
for ratio in (1,2):
 for i,(p,sk),l in itertools.product(seeds,m.CELLS,panel['lengths']):
  assert all(all(a[k]==b[k] for k in template if k!='decision_seconds') for a,b in zip(g[i,p,sk,l,ratio,m.POLICIES[0]],g[i,p,sk,l,ratio,m.POLICIES[1]]))
# Full stored first-shock replay streamed: bounded memory, no fresh full run.
identity=('instance_seed','pressure','skill','length','ratio','episode_seed','policy')
def key_of(r):return tuple(float(r[k]) if k in numeric else r[k] for k in identity)
def grouped(file,key):
 with (RAW/file).open() as f:
  for ident,part in itertools.groupby(csv.DictReader(f),key=key):yield ident,list(part)
trace_iter=iter(grouped('decisions.csv',key_of));job_iter=iter(grouped('job_completions.csv',key_of))
bykey={tuple(r[k] for k in identity):r for r in rows};maxerror=0.;nt=nj=ne=0
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
 nt+=len(tt);nj+=len(jj);ne+=1
 if ne%288==0:print(f'REPLAY: {ne}/1728 first-shock episodes; {nt} decisions',flush=True)
assert next(job_iter,None) is None and nt==summary['audits']['decisions'] and nj==414720 and ne==1728
# Selected-case all20shock/all7variants replay and no pre-intervention divergence.
cf=read('counterfactual_episodes.csv');assert len(cf)==140
cfkeys={(int(r['episode_seed']),r['variant']) for r in cf};names=summary['counterfactual']['variants']
assert cfkeys==set(itertools.product(panel['case_episodes'],names))
cfbykey={(int(r['episode_seed']),r['variant']):r for r in cf}
cfkey=lambda r:(int(r['episode_seed']),r['variant'])
cjob_iter=iter(grouped('counterfactual_job_completions.csv',cfkey));ctot=cjt=0;baseline=None;cfs=[]
for ident,tt in grouped('counterfactual_decisions.csv',cfkey):
 jobident,jj=next(cjob_iter);assert ident==jobident
 e,name=ident;parts=name.split('_');intervention=None if name=='S_baseline' else (int(parts[-1][1:]),int(parts[2][1:]),0)
 r,ts,cs=m.episode(m.scenario(900215,'high','heterogeneous',64,4,6),m.POLICIES[2],e,lead=6,trace=True,intervention=intervention)
 old=cfbykey[ident]
 assert all(old[k]==v for k,v in r.items() if k in numeric and k!='decision_seconds')
 assert len(ts)==len(tt) and len(cs)==len(jj)
 for a,b in zip(ts,tt):assert all(str(v)==b[k] if isinstance(v,(str,bool)) else v==float(b[k]) for k,v in a.items())
 for a,b in zip(cs,jj):assert all(v==int(float(b[k])) for k,v in a.items())
 if name=='S_baseline':baseline=ts
 else:
  assert all(all(a[k]==b[k] for k in ('state','work','action','next_state','next_work','events')) for a,b in zip(baseline[:intervention[0]],ts[:intervention[0]]))
 assert sum(t['intervention_applied'] for t in ts)==int(old['applied']=='True')
 ctot+=len(tt);cjt+=len(jj)
assert next(cjob_iter,None) is None and cjt==140*384
baseline={int(r['episode_seed']):r for r in cf if r['variant']=='S_baseline'}
for name in names:
 rs=[r for r in cf if r['variant']==name];dif=[baseline[int(r['episode_seed'])]['makespan']-r['makespan'] for r in rs]
 cfs.append(dict(variant=name,n=len(rs),applied=sum(r['applied']=='True' for r in rs),makespan=statistics.mean(r['makespan'] for r in rs),failures=statistics.mean(r['failures'] for r in rs),min=min(r['makespan'] for r in rs),max=max(r['makespan'] for r in rs),wins=sum(d>0 for d in dif),ties=sum(d==0 for d in dif),losses=sum(d<0 for d in dif),mean_saving=statistics.mean(dif)))
 assert statistics.mean(r['makespan'] for r in rs)==names[name]['mean_makespan']
# Dev table exposes selection tradeoffs, no retuning.
devtable=[]
for ratio in (1,2,4):
 for pol in m.POLICIES:
  for params in m.grid(pol):
   rs=[r for r in dev if r['ratio']==ratio and r['policy']==pol and r['parameter']==json.dumps(params,sort_keys=True)]
   devtable.append(dict(ratio=ratio,policy=pol,parameter=json.dumps(params,sort_keys=True),normalized_makespan=statistics.mean(r['normalized_makespan'] for r in rs),primary_cell=statistics.mean(r['normalized_makespan'] for r in rs if r['pressure']=='high' and r['skill']=='heterogeneous' and r['length']==64),selected=params==policy[str(ratio)][pol]))
def savecsv(name,data):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
savecsv('instances.csv',instances);savecsv('counterfactual_summary.csv',cfs);savecsv('development_choices.csv',devtable)
(OUT/'statistics.json').write_text(json.dumps(dict(cells=stats,gates=gates,counterfactual=cfs),indent=2))
audit=dict(status='PASS',raw=str(RAW),source_commit=commit,all_hashes_match=True,complete_eval=len(rows),complete_dev=len(dev),resolved_configs=len(configs),replayed_first_shock_episodes=ne,replayed_decisions=nt,replayed_completions=nj,replayed_counterfactual_episodes=len(cf),replayed_counterfactual_decisions=ctot,replayed_counterfactual_completions=cjt,max_numeric_error=maxerror,independent_bootstrap_matches=True,development_selection_matches=True,scope='shared-implementation exact replay, independent statistics; not independent simulator reproduction',raw_unmodified=True)
(OUT/'audit.json').write_text(json.dumps(audit,indent=2));print('REPORT_DIR',OUT,flush=True);print(json.dumps(audit),flush=True)
