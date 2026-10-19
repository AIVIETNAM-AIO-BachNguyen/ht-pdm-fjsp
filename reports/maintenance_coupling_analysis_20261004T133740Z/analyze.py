import csv, gc, hashlib, importlib.util, itertools, json, math, random, statistics, subprocess, sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

root=Path.cwd(); raw=root/'lab_results/maintenance_coupling_full_cpu_20261004T133132Z'
out=root/'reports'/('maintenance_coupling_analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
out.mkdir(parents=True,exist_ok=False)
(out/'analyze.py').write_bytes(Path(__file__).read_bytes())
def j(name): return json.loads((raw/name).read_text())
def csvread(name):
 with (raw/name).open(newline='') as f: return list(csv.DictReader(f))
def save(name,data): (out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
manifest=j('manifest.json'); panel=manifest['panel']; summary=j('summary.json'); policy=j('policy.json')
assert manifest['status']=='COMPLETED' and manifest['profile']=='full' and not manifest['source_dirty']
assert manifest['source_commit']=='07697c29a6e563b762ef6497b5949366ae85c554'
assert all((raw/name).is_file() for name in manifest['outputs'])
for src,key,target in [('src/ht_pdm_fjsp/maintenance_coupling.py','source_sha256','source_snapshot.py'),('docs/maintenance_coupling_v1_plan.md','protocol_sha256','protocol_snapshot.md')]:
 data=subprocess.check_output(['git','show',manifest['source_commit']+':'+src]); assert hashlib.sha256(data).hexdigest()==manifest[key]; (out/target).write_bytes(data)
spec=importlib.util.spec_from_file_location('verified_coupling',out/'source_snapshot.py'); mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
rows=csvread('episodes.csv'); dev=csvread('development.csv'); oracle=csvread('exact_oracle.csv'); trace=csvread('decisions.csv'); coords=csvread('coordination.csv')
keyfields=('instance_seed','pressure','skill','episode_seed','policy')
keys={tuple(r[k] for k in keyfields) for r in rows}
expected=set(itertools.product(map(str,panel['configs']),('low','high'),('uniform','heterogeneous'),map(str,panel['episodes']),mod.POLICIES))
assert len(rows)==len(keys)==7680 and keys==expected
assert len(dev)==288 and len(trace)==2304 and len(oracle)==32 and len(coords)==48
assert (raw/'episodes.csv').read_bytes()==(raw/'episodes.partial.csv').read_bytes()
assert not set(panel['configs'])&set(panel['dev_configs']) and not set(panel['episodes'])&set(panel['dev_episodes'])
devkeys={(r['offset'],r['instance_seed'],r['pressure'],r['skill'],r['episode_seed']) for r in dev}
assert devkeys==set(itertools.product(('0','1','2'),map(str,panel['dev_configs']),('low','high'),('uniform','heterogeneous'),map(str,panel['dev_episodes'])))
dev_means={o:statistics.mean(float(r['cost']) for r in dev if int(r['offset'])==o) for o in (0,1,2)}
assert policy['offset']==min(dev_means,key=lambda o:(dev_means[o],o))==1
for r in rows:
 assert math.isclose(float(r['cost']),float(r['terminal_cost'])+sum(float(r['cost_'+k]) for k in mod.COMPONENTS),abs_tol=1e-8)
 assert float(r['invalid_requests'])==0 and all(math.isfinite(float(r[k])) for k in r if k not in keyfields)
configs=j('resolved_config.json'); assert len(configs)==48
for record in configs:
 cfg=mod.scenario(record['instance_seed'],record['pressure'],record['skill'],panel['horizon'])
 assert json.loads(json.dumps(asdict(cfg)))==record['config']
for seed in panel['configs']:
 assert len({tuple(statistics.mean(row) for row in mod.scenario(seed,p,s,12).service) for p,s in mod.CELLS})==1

# Independent aggregation/bootstrap arithmetic; cluster units are instances.
means={}; metrics={}; cells={}; deltas={}; inst_rows=[]
for p,s in mod.CELLS:
 cell=p+'_'+s;cells[cell]={};metrics[cell]={}
 for pol in mod.POLICIES:
  selected=[r for r in rows if r['pressure']==p and r['skill']==s and r['policy']==pol]
  metrics[cell][pol]={k:statistics.mean(float(r[k]) for r in selected) for k in selected[0] if k not in keyfields}
  for seed in panel['configs']:
   rs=[r for r in selected if int(r['instance_seed'])==seed]
   means[seed,p,s,pol]=statistics.mean(float(r['cost']) for r in rs)
 def boot(values):
  rng=random.Random(782100);rep=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(10000))
  return {'mean':sum(values)/len(values),'ci95':[rep[250],rep[9750]],'n_instances':len(values)}
 for label,a,b in [('B_minus_C',mod.POLICIES[1],mod.POLICIES[2]),('A_minus_B',mod.POLICIES[0],mod.POLICIES[1]),('D_minus_C',mod.POLICIES[3],mod.POLICIES[2])]:
  values=[means[i,p,s,a]-means[i,p,s,b] for i in panel['configs']]
  cells[cell][label]=boot(values)
  for field in ('mean','n_instances'): assert math.isclose(cells[cell][label][field],summary['cells'][cell][label][field],abs_tol=1e-10)
  for x,y in zip(cells[cell][label]['ci95'],summary['cells'][cell][label]['ci95']): assert math.isclose(x,y,abs_tol=1e-10)
  if label=='B_minus_C': deltas[p,s]=values
 rel=statistics.mean((means[i,p,s,mod.POLICIES[1]]-means[i,p,s,mod.POLICIES[2]])/means[i,p,s,mod.POLICIES[1]] for i in panel['configs'])
 assert math.isclose(rel,summary['cells'][cell]['relative_savings'],abs_tol=1e-12)
 cells[cell]['relative_savings']=rel
 cells[cell]['ratio_of_grand_means_saving']=1-metrics[cell][mod.POLICIES[2]]['cost']/metrics[cell][mod.POLICIES[1]]['cost']
 for i in panel['configs']:
  inst_rows.append({'instance_seed':i,'pressure':p,'skill':s,**{pol:means[i,p,s,pol] for pol in mod.POLICIES},'B_minus_C':means[i,p,s,mod.POLICIES[1]]-means[i,p,s,mod.POLICIES[2]],'D_minus_C':means[i,p,s,mod.POLICIES[3]]-means[i,p,s,mod.POLICIES[2]]})
interaction=boot([deltas['high','heterogeneous'][i]-deltas['high','uniform'][i]-deltas['low','heterogeneous'][i]+deltas['low','uniform'][i] for i in range(12)])
assert math.isclose(interaction['mean'],summary['interaction']['mean'],abs_tol=1e-10)
assert all(math.isclose(x,y,abs_tol=1e-10) for x,y in zip(interaction['ci95'],summary['interaction']['ci95']))
high=cells['high_heterogeneous']; primary=high['relative_savings']>=.05 and high['B_minus_C']['ci95'][0]>0
strong=primary and interaction['ci95'][0]>0 and high['D_minus_C']['ci95'][0]>0
assert primary==summary['primary_gate'] and strong==summary['stronger_mechanism_gate']

# Decision/transition replay uses the pinned implementation: a consistency audit,
# not an independent implementation or a new scientific evaluation panel.
bykey={tuple(r[k] for k in keyfields):r for r in rows}
tracekey={tuple(r[k] for k in keyfields)+(r['step'],):r for r in trace}
replayed=0;trace_count=0; max_error=0.
for seed in panel['configs']:
 for p,s in mod.CELLS:
  solver=mod.Solver(mod.scenario(seed,p,s,12),policy['offset'],policy['depth'],policy['tail'])
  for episode_seed in panel['episodes']:
   for pol in mod.POLICIES:
    result,t=mod.episode(solver,pol,episode_seed,trace=episode_seed==panel['episodes'][0])
    key=(str(seed),p,s,str(episode_seed),pol);stored=bykey[key]
    for k,v in result.items():
     if k=='decision_seconds':continue
     error=abs(v-float(stored[k]));max_error=max(max_error,error);assert error<1e-8
    for decision in t:
     r=tracekey[key+(str(decision['step']),)]
     for k in ('state','action','next_state'): assert json.loads(r[k])==json.loads(decision[k])
     for k,v in decision.items():
      if k not in ('state','action','next_state','step'): assert math.isclose(float(r[k]),v,abs_tol=1e-8)
     trace_count+=1
    replayed+=1
  del solver;gc.collect()
 print('replayed instance',seed,flush=True)
for seed in panel['configs'][:2]:
 for p,s in mod.CELLS:
  solver=mod.Solver(mod.scenario(seed,p,s,4),policy['offset'],policy['depth'],policy['tail'])
  optimal=solver.oracle_value(mod.initial(solver.cfg),4)
  for pol in mod.POLICIES:
   r=next(r for r in oracle if int(r['instance_seed'])==seed and r['pressure']==p and r['skill']==s and r['policy']==pol)
   value=solver.exact_policy_cost(pol,4)
   assert math.isclose(value,float(r['exact_cost']),abs_tol=1e-8) and math.isclose(optimal,float(r['optimal_cost']),abs_tol=1e-8)
   assert math.isclose(value-optimal,float(r['gap']),abs_tol=1e-8) and value>=optimal-1e-8
  del solver;gc.collect()

other=j('manifest.json')
excluded=[]
for path in (root/'lab_results').glob('*/manifest.json'):
 if '20261004' in path.parent.name and path.parent!=raw:
  m=json.loads(path.read_text());excluded.append({'run':path.parent.name,'local_manifest_status':m.get('status'),'reason':'different experiment; not included in coupling analysis'})
audit={'source':str(raw),'source_commit':manifest['source_commit'],'complete':True,'expected_rows':len(rows),'decision_rows':len(trace),'oracle_rows':len(oracle),'development_rows':len(dev),'coordination_rows':len(coords),'replayed_episodes':replayed,'replayed_decisions':trace_count,'max_replay_error':max_error,'summary_arithmetic_match':True,'hashes_match':True,'independent_reproduction':False,'excluded':excluded,'file_sha256':{name:hashlib.sha256((raw/name).read_bytes()).hexdigest() for name in manifest['outputs']}}
save('audit.json',audit);save('statistics.json',{'cells':cells,'interaction':interaction,'primary_gate':primary,'stronger_mechanism_gate':strong,'development_mean_costs':dev_means,'metrics':metrics})
with (out/'instances.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(inst_rows[0]));w.writeheader();w.writerows(inst_rows)
print('ANALYSIS_DIR',out,flush=True)
print(json.dumps({'gates':[primary,strong],'counts':[replayed,trace_count],'dev_cost':dev_means,'high_heterogeneous':high,'interaction':interaction,'excluded':excluded}),flush=True)
