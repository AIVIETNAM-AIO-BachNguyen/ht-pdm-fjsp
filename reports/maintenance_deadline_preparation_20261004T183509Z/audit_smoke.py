import csv,hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl')
WORK=Path('/Users/bachng/.codex/worktrees/maintenance-coupling/marl')
sys.path.insert(0,str(WORK/'src'))
from ht_pdm_fjsp import maintenance_deadline_packing as m
OUT=ROOT/'reports'/('maintenance_deadline_preparation_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));OUT.mkdir()
(OUT/'audit_smoke.py').write_text(Path(__file__).read_text())
run=sorted(p for p in (WORK/'artifacts').glob('maintenance_deadline_smoke_*') if p.is_dir())[-1]
manifest=json.loads((run/'manifest.json').read_text());summary=json.loads((run/'summary.json').read_text());panel=manifest['panel'];policy=json.loads((run/'policy.json').read_text())['settings']
assert manifest['status']=='COMPLETED' and not manifest['full_panel_opened'] and panel==m.panels('smoke')
sources={'source':Path(m.__file__),'repair':Path(m.repair.__file__),'production':Path(m.prod.__file__),'coupling':Path(m.base.__file__),'protocol':WORK/'docs/maintenance_deadline_packing_v1_plan.md'}
for k,p in sources.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['hashes'][k]
template,_,_=m.episode(m.scenario(930100,'low','uniform',3,1,4),m.POLICIES[0],931100)
numeric={k for k,v in template.items() if isinstance(v,(int,float))}|{'instance_seed','length','ratio','episode_seed'}
def read(name):
 with (run/name).open() as f:return [{k:float(v) if k in numeric else v for k,v in r.items()} for r in csv.DictReader(f)]
rows=read('episodes.csv');dev=read('development.csv');m.validate(rows,panel);m.validate(dev,panel,True)
assert (run/'episodes.csv').read_bytes()==(run/'episodes.partial.csv').read_bytes()
for ratio in (1,2,4):
 for pol in m.POLICIES:
  candidates=m.grid(pol)
  means={json.dumps(params,sort_keys=True):sum(r['normalized_makespan'] for r in dev if r['ratio']==ratio and r['policy']==pol and r['parameter']==json.dumps(params,sort_keys=True))/sum(1 for r in dev if r['ratio']==ratio and r['policy']==pol and r['parameter']==json.dumps(params,sort_keys=True)) for params in candidates}
  best=min(candidates,key=lambda p:(means[json.dumps(p,sort_keys=True)],tuple(p.values())))
  assert best==policy[str(ratio)][pol]
traces=read('decisions.csv');jobs=read('job_completions.csv');maxerror=0.;nt=nj=0;plans=0

def check(r,rr,ts,cs,tt,jj,cfg):
 global maxerror,plans
 for k,v in rr.items():
  if k in numeric and k!='decision_seconds':maxerror=max(maxerror,abs(r[k]-v));assert r[k]==v
 assert len(ts)==len(tt) and len(cs)==len(jj)
 for x,y in zip(ts,tt):
  for k,v in x.items():
   if isinstance(v,str):assert v==y[k]
   else:maxerror=max(maxerror,abs(float(y[k])-v));assert v==float(y[k])
  slots=json.loads(y['deadline_plan']);state=json.loads(y['state'])
  if r['policy']==m.POLICIES[2]:
   plans+=1;assert float(y['waiting'])==0
   assert len({s['machine'] for s in slots})==len(slots)
   for t in range(len(state['remaining'])):
    chain=[s for s in slots if s['worker']==t];deadline_shift=max(0,min((s['deadline']-s['forward_start'] for s in chain),default=0))
    committed=state['remaining'][t]+sum(cfg.maintenance.service[machine][t]*(cfg.ratio if state['failed'][machine] else 1) for machine in state['queues'][t])
    previous=committed
    for s in chain:
     assert s['start']==s['forward_start']+deadline_shift and s['shift']==deadline_shift
     assert s['start']>=previous and s['start']>=s['release'] and s['late']==max(0,s['start']-s['deadline'])
     assert s['duration']==cfg.maintenance.service[s['machine']][t]*(cfg.ratio if state['failed'][s['machine']] else 1)
     previous=s['start']+s['duration']
   action=json.loads(y['action'])
   assert sum(a>0 for a in action)==sum(s['start']==0 for s in slots)
   for s in slots:
    if s['start']==0:assert action[s['machine']]==s['worker']+1 and state['remaining'][s['worker']]==0 and not state['queues'][s['worker']]
 for x,y in zip(cs,jj):assert all(v==int(float(y[k])) for k,v in x.items())

for r in rows:
 cfg=m.scenario(int(r['instance_seed']),r['pressure'],r['skill'],int(r['length']),int(r['ratio']),4)
 rr,ts,cs=m.episode(cfg,r['policy'],int(r['episode_seed']),trace=True,**policy[str(int(r['ratio']))][r['policy']])
 if int(r['episode_seed'])==panel['episodes'][0]:
  same=lambda x:all(x[k]==r[k] for k in ('instance_seed','pressure','skill','length','ratio','episode_seed','policy'))
  tt=[x for x in traces if same(x)];jj=[x for x in jobs if same(x)]
  check(r,rr,ts,cs,tt,jj,cfg);nt+=len(ts);nj+=len(cs)
 else:
  assert all(r[k]==v for k,v in rr.items() if k in numeric and k!='decision_seconds')
reg=read('regression_episodes.csv');rt=read('regression_decisions.csv');rj=read('regression_job_completions.csv');nr=nc=0
assert len(reg)==9
for r in reg:
 cfg=m.scenario(int(r['instance_seed']),'high','heterogeneous',64,4,6)
 params=dict(offset=4) if r['policy']==m.POLICIES[0] else dict(lead=6) if r['policy']==m.POLICIES[1] else policy['4'][r['policy']]
 rr,ts,cs=m.episode(cfg,r['policy'],int(r['episode_seed']),trace=True,**params)
 same=lambda x:all(x[k]==r[k] for k in ('instance_seed','episode_seed','policy'))
 tt=[x for x in rt if same(x)];jj=[x for x in rj if same(x)]
 check(r,rr,ts,cs,tt,jj,cfg);nr+=len(ts);nc+=len(cs)
full=m.panels('full');fresh=set(full['configs']+full['dev_configs']+full['episodes']+full['dev_episodes']);oldcount=0
for folder in [ROOT/'artifacts',ROOT/'lab_results',WORK/'artifacts']:
 for p in folder.rglob('manifest.json'):
  data=json.loads(p.read_text())
  if data.get('protocol')==m.PROTOCOL:continue
  def ints(o):
   if isinstance(o,dict):return set().union(*(ints(v) for v in o.values())) if o else set()
   if isinstance(o,list):return set().union(*(ints(v) for v in o)) if o else set()
   return {o} if isinstance(o,int) and not isinstance(o,bool) else set()
  assert not fresh&ints(data),(str(p),fresh&ints(data));oldcount+=1
assert nt==summary['audits']['decisions'] and nj==summary['audits']['job_completions']
assert nr==summary['regressions']['decisions'] and nc==3456
assert not any(summary['gates'].values())
audit=dict(status='PASS',scope='shared-implementation replay and separate slot invariants; not independent scientific reproduction',smoke=str(run),source_commit=manifest['source_commit'],source_dirty=manifest['source_dirty'],episodes=len(rows),development=len(dev),configs=24,decisions_replayed=nt,job_completions_replayed=nj,regression_episodes=len(reg),regression_decisions_replayed=nr,regression_job_completions_replayed=nc,plans_audited=plans,max_numeric_error=maxerror,sealed_panel_unopened=True,dev_selection_independently_matches=True,prior_local_manifests_seed_audited=oldcount,seed_freshness_scope='local manifests only; untransferred remote experiments unknown',hashes=manifest['hashes'])
(OUT/'smoke_audit.json').write_text(json.dumps(audit,indent=2));print('REPORT_DIR',OUT);print(json.dumps(audit))
