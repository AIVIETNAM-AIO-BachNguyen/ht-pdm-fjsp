import csv,gc,hashlib,itertools,json,math,random,statistics,subprocess,sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from tqdm import tqdm
root=Path.cwd();raw=root/'lab_results/maintenance_repair_full_cpu_20261004T162209Z'
out=root/'reports'/('maintenance_repair_impact_analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(parents=True,exist_ok=False)
(out/'analyze.py').write_bytes(Path(__file__).read_bytes())
def load(n):return json.loads((raw/n).read_text())
def read(n):
 with (raw/n).open(newline='') as f:return list(csv.DictReader(f))
def save(n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
mf=load('manifest.json');p=mf['panel'];orig=load('summary.json');policy=load('policy.json')
assert mf['status']=='COMPLETED' and mf['profile']=='full' and not mf['source_dirty'] and mf['source_commit']=='c8f6abb5f68b01c77bb1699d558936028b5721fa'
assert all((raw/n).is_file() for n in mf['outputs'])
pkg=out/'pinned'/'ht_pdm_fjsp';pkg.mkdir(parents=True);(pkg/'__init__.py').write_text('')
for file,key,dest in [('maintenance_repair_impact.py','source_sha256',pkg/'maintenance_repair_impact.py'),('maintenance_production_impact.py','production_dependency_sha256',pkg/'maintenance_production_impact.py'),('maintenance_coupling.py','coupling_dependency_sha256',pkg/'maintenance_coupling.py')]:
 data=subprocess.check_output(['git','show',mf['source_commit']+':src/ht_pdm_fjsp/'+file]);assert hashlib.sha256(data).hexdigest()==mf[key];dest.write_bytes(data)
data=subprocess.check_output(['git','show',mf['source_commit']+':docs/maintenance_repair_impact_v1_plan.md']);assert hashlib.sha256(data).hexdigest()==mf['protocol_sha256'];(out/'protocol_snapshot.md').write_bytes(data)
sys.path.insert(0,str(out/'pinned'))
from ht_pdm_fjsp import maintenance_repair_impact as m
assert p==m.panels('full')
rows=read('episodes.csv');dev=read('development.csv');co=read('coordination.csv');cfgs=load('resolved_config.json')
fields=('instance_seed','pressure','skill','length','ratio','episode_seed','policy');cf=fields[:5]
def key(r):return tuple(r[f] for f in fields)
expect=set(itertools.product(map(str,p['configs']),('low','high'),('uniform','heterogeneous'),map(str,p['lengths']),map(str,p['ratios']),map(str,p['episodes']),m.POLICIES))
assert len(rows)==len({key(r) for r in rows})==34560 and {key(r) for r in rows}==expect
assert len(dev)==6912 and len(co)==len(cfgs)==576
assert hashlib.sha256((raw/'episodes.csv').read_bytes()).digest()==hashlib.sha256((raw/'episodes.partial.csv').read_bytes()).digest()
assert not set(p['configs'])&set(p['dev_configs']) and not set(p['episodes'])&set(p['dev_episodes'])
numeric=[f for f in rows[0] if f not in fields+('feasibility_audit','work_balance_audit')]
for r in rows+dev:
 assert r['feasibility_audit']==r['work_balance_audit']=='PASS' and float(r['invalid_requests'])==0
 assert all(math.isfinite(float(r[f])) for f in numeric)
 assert float(r['pm_jobs'])+float(r['cm_jobs'])==float(r['jobs'])
 assert float(r['pm_service_steps'])+float(r['cm_service_steps'])==float(r['planned_steps'])
 assert float(r['productive_steps'])==float(r['required_work']) and int(r['completed_jobs'])==6*int(r['length'])
 assert float(r['makespan'])>=float(r['ideal_makespan']) and math.isclose(float(r['normalized_makespan']),float(r['makespan'])/float(r['ideal_makespan']),abs_tol=1e-10)
 assert math.isclose(float(r['cost']),sum(float(r['cost_'+f]) for f in m.COMPONENTS),abs_tol=1e-8)
de=set()
for pol,grid in [(m.POLICIES[1],(0,1,2,4)),(m.POLICIES[2],(0,2,4,6))]:de.update(itertools.product([pol],map(str,grid),map(str,p['dev_configs']),('low','high'),('uniform','heterogeneous'),map(str,p['lengths']),map(str,p['ratios']),map(str,p['dev_episodes'])))
assert {tuple(r[f] for f in ('policy','parameter')+fields[:-1]) for r in dev}==de
devscores={}
for ratio in p['ratios']:
 for pol,param,grid in [(m.POLICIES[1],'offset',(0,1,2,4)),(m.POLICIES[2],'lead',(0,2,4,6))]:
  scores={v:statistics.mean(float(r['normalized_makespan']) for r in dev if int(r['ratio'])==ratio and r['policy']==pol and int(r['parameter'])==v) for v in grid}
  assert policy['settings'][str(ratio)][param]==min(grid,key=lambda v:(scores[v],v));devscores[f'ratio{ratio}_{param}']=scores
expected_cfg=set(itertools.product(map(str,p['configs']),('low','high'),('uniform','heterogeneous'),map(str,p['lengths']),map(str,p['ratios'])))
assert {tuple(str(r[f]) for f in cf) for r in cfgs}=={tuple(r[f] for f in cf) for r in co}==expected_cfg
assert all(r['feasibility_audit']==r['work_balance_audit']=='PASS' and int(r['invalid_requests'])==0 for r in co)
for r in cfgs:assert json.loads(json.dumps(asdict(m.scenario(*(r[f] for f in cf),6))))==r['config']
for i,l in itertools.product(p['configs'],p['lengths']):
 cs=[m.scenario(i,pressure,skill,l,ratio,6) for pressure,skill,ratio in itertools.product(('low','high'),('uniform','heterogeneous'),p['ratios'])]
 assert len({(c.jobs,c.maintenance.initial_ages,c.maintenance.failure_age,c.maintenance.probability) for c in cs})==1
 assert len({tuple(statistics.mean(row) for row in c.maintenance.service) for c in cs})==1

# Independent numerical aggregation and paired bootstrap using the locked draws.
groups=defaultdict(list)
for r in rows:groups[tuple(r[f] for f in cf)+(r['policy'],)].append(r)
means={k:{f:statistics.mean(float(r[f]) for r in rs) for f in numeric} for k,rs in groups.items()}
rng=random.Random(902100);indices=np.array([[int(rng.random()*24) for _ in range(24)] for _ in range(10000)])
def boot(v):
 assert len(v)==24
 draws=np.sort(np.asarray(v,dtype=float)[indices].mean(axis=1))
 return dict(mean=statistics.mean(v),ci95=[float(draws[250]),float(draws[9750])],n_instances=24)
def check(a,b):
 assert a['n_instances']==b['n_instances'] and math.isclose(a['mean'],b['mean'],abs_tol=1e-10)
 assert all(math.isclose(x,y,abs_tol=1e-10) for x,y in zip(a['ci95'],b['ci95']))
F,H,S=m.POLICIES
def vals(pres,sk,l,ratio,pol,f='makespan'):return [means[str(i),pres,sk,str(l),str(ratio),pol][f] for i in p['configs']]
cells={};instances=[]
for ratio,pres,sk,l in itertools.product(p['ratios'],('low','high'),('uniform','heterogeneous'),p['lengths']):
 name=f'ratio{ratio}_{pres}_{sk}_jobs{l}';stored=orig['ratios'][str(ratio)]['cells'][f'{pres}_{sk}_jobs{l}']
 fv,hv,sv=(vals(pres,sk,l,ratio,pol) for pol in (F,H,S));cell=dict(metrics={pol:{f:statistics.mean(vals(pres,sk,l,ratio,pol,f)) for f in numeric} for pol in (F,H,S)})
 for pol in (F,H,S):
  for f,v in stored['metrics'][pol].items():assert math.isclose(v,cell['metrics'][pol][f],abs_tol=1e-10)
 for label,v in [('H_minus_S',[a-b for a,b in zip(hv,sv)]),('H_minus_S_relative',[(a-b)/a for a,b in zip(hv,sv)]),('F_minus_H',[a-b for a,b in zip(fv,hv)]),('H_minus_S_late_service',[a-b for a,b in zip(vals(pres,sk,l,ratio,H,'late_active_service_steps'),vals(pres,sk,l,ratio,S,'late_active_service_steps'))])]:
  cell[label]=boot(v);check(cell[label],stored[label])
 cell['H_minus_S_relative_cost']=boot([(a-b)/a for a,b in zip(vals(pres,sk,l,ratio,H,'cost'),vals(pres,sk,l,ratio,S,'cost'))])
 cell['S_vs_H_counts']=dict(wins=sum(a<b-1e-8 for a,b in zip(sv,hv)),ties=sum(abs(a-b)<1e-8 for a,b in zip(sv,hv)),losses=sum(a>b+1e-8 for a,b in zip(sv,hv)))
 for i in p['configs']:
  r=dict(instance_seed=i,pressure=pres,skill=sk,length=l,ratio=ratio)
  for pol in (F,H,S):r.update({pol[0]+'_'+f:means[str(i),pres,sk,str(l),str(ratio),pol][f] for f in numeric})
  instances.append(r)
 cells[name]=cell
capacity={};interactions={}
for ratio in p['ratios']:
 for sk in ('uniform','heterogeneous'):
  cap=boot([a-b for a,b in zip(vals('high',sk,64,ratio,H,'normalized_makespan'),vals('low',sk,64,ratio,H,'normalized_makespan'))]);check(cap,orig['ratios'][str(ratio)]['capacity_burden'][sk]);capacity[f'ratio{ratio}_{sk}']=cap
 a,b,c,d=[vals(pres,sk,64,ratio,H,'normalized_makespan') for pres,sk in [('high','heterogeneous'),('high','uniform'),('low','heterogeneous'),('low','uniform')]]
 inter=boot([(x-y)-(z-w) for x,y,z,w in zip(a,b,c,d)]);check(inter,orig['ratios'][str(ratio)]['interaction']);interactions[str(ratio)]=inter
f1=vals('high','heterogeneous',64,1,F);f2=vals('high','heterogeneous',64,2,F)
mechanism=dict(absolute=boot([b-a for a,b in zip(f1,f2)]),relative=boot([(b-a)/a for a,b in zip(f1,f2)]))
for label in mechanism:check(mechanism[label],orig['physical_mechanism'][label])
h=vals('high','heterogeneous',64,2,H);s=vals('high','heterogeneous',64,2,S)
adequacy=boot([(a-b)/b for a,b in zip(h,s)]);check(adequacy,orig['H_vs_S_noninferiority'])
primary=cells['ratio2_high_heterogeneous_jobs64']
gates=dict(physical_mechanism=mechanism['relative']['mean']>=.05 and mechanism['absolute']['ci95'][0]>0,coordination_useful=primary['H_minus_S_relative']['mean']>=.05 and primary['H_minus_S']['ci95'][0]>0,threshold_adequacy=adequacy['ci95'][1]<=.02)
assert gates==orig['gates']
extra=boot([((a-b)/a)-((c-d)/c) for a,b,c,d in zip(h,s,vals('low','heterogeneous',64,2,H),vals('low','heterogeneous',64,2,S))])

# Stream the entire recorded trace/completion panel, avoiding large JSON tables
# in memory. This replay uses the pinned shared implementation, not a new panel.
bykey={key(r):r for r in rows};nrep=0;nt=0;nj=0;maxerr=0.
with (raw/'decisions.csv').open(newline='') as tf,(raw/'job_completions.csv').open(newline='') as jf:
 tg=iter(itertools.groupby(csv.DictReader(tf),key=key));jg=iter(itertools.groupby(csv.DictReader(jf),key=key))
 for rec in tqdm(cfgs,desc='replay stored repair impact traces',unit='cell'):
  cfg=m.scenario(*(rec[f] for f in cf),6)
  for pol in (F,H,S):
   identity=tuple(str(rec[f]) for f in cf)+(str(p['episodes'][0]),pol)
   result,trace,jobs=m.episode(cfg,pol,p['episodes'][0],**policy['settings'][str(rec['ratio'])],trace=True)
   for f,v in result.items():
    if f=='decision_seconds':continue
    if isinstance(v,str):assert v==bykey[identity][f]
    else:
     err=abs(v-float(bykey[identity][f]));maxerr=max(maxerr,err);assert err<1e-8
   ident,stream=next(tg);assert ident==identity
   olds=list(stream);assert len(olds)==len(trace)==result['makespan']
   for old,t in zip(olds,trace):
    for f,v in t.items():
     if f in ('state','work','action','next_state','next_work'):assert json.loads(v)==json.loads(old[f])
     else:assert math.isclose(v,float(old[f]),abs_tol=1e-8)
    nt+=1
   ident,stream=next(jg);assert ident==identity
   olds=list(stream);assert len(olds)==len(jobs)==6*rec['length']
   assert len({(old['machine'],old['job']) for old in olds})==len(olds)
   for old,c in zip(olds,jobs):assert all(int(old[f])==v for f,v in c.items());nj+=1
   nrep+=1
  del cfg;gc.collect()
 assert next(tg,None) is None and next(jg,None) is None
assert nrep==1728 and nt==399576 and nj==414720
audit=dict(verification_status='ANALYZED',raw=str(raw),source_commit=mf['source_commit'],hashes_match=True,full_cartesian_complete=True,episodes=len(rows),development=len(dev),config_coordination=len(cfgs),decisions=nt,job_completions=nj,summary_arithmetic_match=True,development_selection_matches=True,replayed_episodes=nrep,replayed_decisions=nt,replayed_job_completions=nj,max_replay_error=maxerr,independent_reproduction=False,replay_scope='first shared shock per config/cell/ratio; entire recorded trace/completion panel',file_sha256={n:hashlib.sha256((raw/n).read_bytes()).hexdigest() for n in mf['outputs']})
save('audit.json',audit);save('statistics.json',dict(cells=cells,capacity_burden=capacity,interactions=interactions,mechanism=mechanism,threshold_adequacy=adequacy,gates=gates,development=devscores,exploratory_extra_relative_recovery_high_minus_low=extra))
with (out/'instances.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(instances[0]));w.writeheader();w.writerows(instances)
print('ANALYSIS_DIR',out,flush=True);print(json.dumps(dict(primary=primary,mechanism=mechanism,adequacy=adequacy,gates=gates,extra=extra),indent=2),flush=True)
