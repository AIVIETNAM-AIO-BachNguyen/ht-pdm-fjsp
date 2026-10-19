import csv, json, hashlib, sys
from pathlib import Path
root=Path('/Users/bachng/.codex/worktrees/maintenance-coupling/marl')
sys.path.insert(0,str(root/'src'))
from ht_pdm_fjsp import maintenance_backlog_adaptive as m
out=sorted(p for p in (root/'artifacts').glob('maintenance_backlog_smoke_*') if p.is_dir())[-1]
manifest=json.loads((out/'manifest.json').read_text());panel=manifest['panel'];policy=json.loads((out/'policy.json').read_text())['settings']
assert manifest['status']=='COMPLETED' and not manifest['full_panel_opened']
template,_,_=m.episode(m.scenario(910100,'low','uniform',3,1,4),m.POLICIES[0],911100)
numeric={k for k,v in template.items() if isinstance(v,(int,float))}|{'instance_seed','length','ratio','episode_seed'}
def read(name):
    with (out/name).open() as f:return [{k:float(v) if k in numeric else v for k,v in r.items()} for r in csv.DictReader(f)]
rows=read('episodes.csv');dev=read('development.csv');m.validate(rows,panel);m.validate(dev,panel,True)
assert (out/'episodes.csv').read_bytes()==(out/'episodes.partial.csv').read_bytes()
for ratio in (1,2,4):
    for pol in m.POLICIES:assert policy[str(ratio)][pol]==m.select(dev,ratio,pol)
for key,path in [('source',Path(m.__file__)),('repair',Path(m.repair.__file__)),('production',Path(m.prod.__file__)),('coupling',Path(m.base.__file__)),('protocol',root/'docs/maintenance_backlog_adaptive_v1_plan.md')]:
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['hashes'][key]
traces=read('decisions.csv');jobs=read('job_completions.csv');nstep=njobs=0
for r in rows:
    cfg=m.scenario(int(r['instance_seed']),r['pressure'],r['skill'],int(r['length']),int(r['ratio']),panel['machines'])
    rr,tt,jj=m.episode(cfg,r['policy'],int(r['episode_seed']),trace=True,**policy[str(int(r['ratio']))][r['policy']])
    assert all(abs(r[k]-v)<1e-8 for k,v in rr.items() if k in numeric and k!='decision_seconds')
    if int(r['episode_seed'])!=panel['episodes'][0]:continue
    def same(x):return all(x[k]==r[k] for k in ('instance_seed','pressure','skill','length','ratio','episode_seed','policy'))
    ts=[x for x in traces if same(x)];cs=[x for x in jobs if same(x)]
    assert len(tt)==len(ts) and len(jj)==len(cs)
    for x,y in zip(tt,ts):
        assert all(str(v)==y[k] if isinstance(v,str) else str(v)==y[k] if isinstance(v,bool) else abs(float(y[k])-v)<1e-8 for k,v in x.items())
    for x,y in zip(jj,cs):assert all(int(float(y[k]))==v for k,v in x.items())
    nstep+=len(tt);njobs+=len(jj)
cf=read('counterfactual_episodes.csv');ct=read('counterfactual_decisions.csv');cj=read('counterfactual_job_completions.csv');csteps=cjobs=0
for r in cf:
    parts=r['variant'].split('_');intervention=None if r['variant']=='S_baseline' else (int(parts[-1][1:]),int(parts[2][1:]),0)
    cfg=m.scenario(900215,'high','heterogeneous',64,4,6)
    rr,tt,jj=m.episode(cfg,m.POLICIES[2],int(r['episode_seed']),lead=6,trace=True,intervention=intervention)
    assert all(abs(r[k]-v)<1e-8 for k,v in rr.items() if k in numeric and k!='decision_seconds')
    ts=[x for x in ct if x['variant']==r['variant']];cs=[x for x in cj if x['variant']==r['variant']]
    assert len(tt)==len(ts) and len(jj)==len(cs)
    for x,y in zip(tt,ts):
        assert all(str(v)==y[k] if isinstance(v,(str,bool)) else abs(float(y[k])-v)<1e-8 for k,v in x.items())
    for x,y in zip(jj,cs):assert all(int(float(y[k]))==v for k,v in x.items())
    csteps+=len(tt);cjobs+=len(jj)
full=m.panels('full');fresh=set(full['configs']+full['dev_configs']+full['episodes']+full['dev_episodes'])
prior=[]
for directory in [Path('/Users/bachng/Coding/Reinforcement Learning/marl/artifacts'),Path('/Users/bachng/Coding/Reinforcement Learning/marl/lab_results'),root/'artifacts']:
    for p in directory.rglob('manifest.json'):
        data=json.loads(p.read_text())
        if data.get('protocol')==m.PROTOCOL:continue
        def ints(o):
            if isinstance(o,dict):return set().union(*(ints(v) for v in o.values())) if o else set()
            if isinstance(o,list):return set().union(*(ints(v) for v in o)) if o else set()
            return {o} if isinstance(o,int) and not isinstance(o,bool) else set()
        overlap=fresh&ints(data)
        assert not overlap,(str(p),overlap)
        prior.append(str(p))
report=dict(status='PASS',scope='shared-implementation consistency replay; not independent reproduction',smoke=str(out),episodes=len(rows),development=len(dev),configs=24,decisions_replayed=nstep,job_completions_replayed=njobs,counterfactual_episodes=len(cf),counterfactual_decisions_replayed=csteps,counterfactual_job_completions_replayed=cjobs,max_numeric_error=0,sealed_panel_unopened=True,prior_manifests_seed_audited=len(prior),seed_freshness_scope='locally available manifests only; untransferred remote experiments unknown',hashes=manifest['hashes'])
Path('reports/maintenance_backlog_checks_20261005/smoke_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
