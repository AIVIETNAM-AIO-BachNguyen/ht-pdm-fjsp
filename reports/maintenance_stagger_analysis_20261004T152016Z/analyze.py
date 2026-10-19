import csv, gc, hashlib, itertools, json, math, random, statistics, subprocess, sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

root=Path.cwd(); raw=root/'lab_results/maintenance_stagger_full_cpu_20261004T150819Z'
out=root/'reports'/('maintenance_stagger_analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
out.mkdir(parents=True,exist_ok=False)
(out/'analyze.py').write_bytes(Path(__file__).read_bytes())
def load(name): return json.loads((raw/name).read_text())
def read(name):
    with (raw/name).open(newline='') as f: return list(csv.DictReader(f))
def save(name,data): (out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
manifest=load('manifest.json'); panel=manifest['panel']; original=load('summary.json'); policy=load('policy.json')
assert manifest['status']=='COMPLETED' and manifest['profile']=='full' and not manifest['source_dirty']
assert manifest['source_commit']=='ccaff3e1e36078b9258cbb209089cce6b5ca684c'
assert all((raw/n).is_file() for n in manifest['outputs'])
pkg=out/'pinned'/'ht_pdm_fjsp'; pkg.mkdir(parents=True); (pkg/'__init__.py').write_text('')
for source,key,target in [('src/ht_pdm_fjsp/maintenance_stagger.py','source_sha256',pkg/'maintenance_stagger.py'),('src/ht_pdm_fjsp/maintenance_coupling.py','dependency_sha256',pkg/'maintenance_coupling.py'),('docs/maintenance_stagger_v1_plan.md','protocol_sha256',out/'protocol_snapshot.md')]:
    data=subprocess.check_output(['git','show',manifest['source_commit']+':'+source]); assert hashlib.sha256(data).hexdigest()==manifest[key]; target.write_bytes(data)
sys.path.insert(0,str(out/'pinned'))
from ht_pdm_fjsp import maintenance_stagger as mod
base=mod.base; policies=mod.POLICIES
assert panel==mod.panels('full')
keys=('instance_seed','pressure','skill','planned_price','horizon','episode_seed','policy')
def identity(r): return tuple(r[k] for k in keys)
rows=read('episodes.csv'); dev=read('development.csv'); traces=read('decisions.csv'); coords=read('coordination.csv')
expected=set(itertools.product(map(str,panel['configs']),('low','high'),('uniform','heterogeneous'),map(str,panel['prices']),map(str,panel['horizons']),map(str,panel['episodes']),policies))
assert len(rows)==len({identity(r) for r in rows})==34560 and {identity(r) for r in rows}==expected
assert len(dev)==7680 and len(traces)==20736 and len(coords)==384
assert (raw/'episodes.partial.csv').read_bytes()==(raw/'episodes.csv').read_bytes()
assert not set(panel['configs'])&set(panel['dev_configs']) and not set(panel['episodes'])&set(panel['dev_episodes'])
devkeys=('policy','parameter','instance_seed','pressure','skill','planned_price','horizon','episode_seed')
de=set()
for pol,grid in [(policies[0],range(3)),(policies[1],range(5))]:
    de.update(itertools.product([pol],map(str,grid),map(str,panel['dev_configs']),('low','high'),('uniform','heterogeneous'),map(str,panel['prices']),map(str,panel['horizons']),map(str,panel['dev_episodes'])))
assert len(de)==len(dev) and {tuple(r[k] for k in devkeys) for r in dev}==de
devmeans={}
for price in panel['prices']:
    for pol,grid,param in [(policies[0],range(3),'offset'),(policies[1],range(5),'lead')]:
        vals={v:statistics.mean(float(r['cost_per_step']) for r in dev if int(r['planned_price'])==price and r['policy']==pol and int(r['parameter'])==v) for v in grid}
        assert policy['settings'][str(price)][param]==min(vals,key=lambda v:(vals[v],v))
        devmeans[f'price{price}_{pol}']=vals
fields=[k for k in rows[0] if k not in keys]
for r in rows+dev:
    assert all(math.isfinite(float(r[k])) for k in fields)
    assert float(r['invalid_requests'])==0
    stage=sum(float(r['cost_'+k]) for k in base.COMPONENTS)
    assert math.isclose(stage,float(r['stage_cost']),abs_tol=1e-8)
    assert math.isclose(stage+float(r['terminal_cost']),float(r['cost']),abs_tol=1e-8)
    assert math.isclose(float(r['cost'])/int(r['horizon']),float(r['cost_per_step']),abs_tol=1e-10)
    assert math.isclose(1-(float(r['planned_steps'])+float(r['failed_steps']))/(2*int(r['horizon'])),float(r['operating_proxy']),abs_tol=1e-10)
configs=load('resolved_config.json'); assert len(configs)==384
configkeys=('instance_seed','pressure','skill','planned_price','horizon')
assert len({tuple(r[k] for k in configkeys) for r in configs})==384
for r in configs:
    cfg=mod.cell_config(*(r[k] for k in configkeys)); assert json.loads(json.dumps(asdict(cfg)))==r['config']
for seed in panel['configs']:
    assert len({tuple(statistics.mean(row) for row in mod.cell_config(seed,p,s,1,12).service) for p,s in base.CELLS})==1
assert {tuple(str(r[k]) for k in configkeys) for r in configs}=={tuple(r[k] for k in configkeys) for r in rows}
assert {tuple(r[k] for k in configkeys) for r in coords}=={tuple(r[k] for k in configkeys) for r in rows}
assert all(r['reservation_audit']==r['state_audit']=='PASS' and int(r['invalid_requests'])==0 for r in coords)

# Independent aggregation and paired cluster bootstrap arithmetic.
grouped=defaultdict(list)
for r in rows: grouped[tuple(r[k] for k in configkeys)+(r['policy'],)].append(r)
means={k:{f:statistics.mean(float(r[f]) for r in rs) for f in fields} for k,rs in grouped.items()}
def boot(vals):
    # Match locked random.choices draws with independently expressed arithmetic.
    rng=random.Random(862100); draws=sorted(sum(vals[int(rng.random()*len(vals))] for _ in vals)/len(vals) for _ in range(10000))
    return dict(mean=statistics.mean(vals),ci95=[draws[250],draws[9750]],n_instances=len(vals))
cells={}; instance_rows=[]
for p,s,price,h in itertools.product(('low','high'),('uniform','heterogeneous'),panel['prices'],panel['horizons']):
    name=f'{p}_{s}_price{price}_h{h}'
    def vals(pol,f='cost'): return [means[str(i),p,s,str(price),str(h),pol][f] for i in panel['configs']]
    b,e,c=map(vals,policies)
    cell={'metrics':{pol:{f:statistics.mean(vals(pol,f)) for f in fields} for pol in policies}}
    endpoints={'B_minus_E':[x-y for x,y in zip(b,e)],'B_minus_C':[x-y for x,y in zip(b,c)],'E_minus_C_relative':[(x-y)/y for x,y in zip(e,c)],'B_minus_E_relative':[(x-y)/x for x,y in zip(b,e)],'E_minus_B_operating':[x-y for x,y in zip(vals(policies[1],'operating_proxy'),vals(policies[0],'operating_proxy'))],'C_minus_B_operating':[x-y for x,y in zip(vals(policies[2],'operating_proxy'),vals(policies[0],'operating_proxy'))]}
    for label,vs in endpoints.items():
        cell[label]=boot(vs)
        for k in ('mean','n_instances'): assert math.isclose(cell[label][k],original['cells'][name][label][k],abs_tol=1e-10)
        assert all(math.isclose(x,y,abs_tol=1e-10) for x,y in zip(cell[label]['ci95'],original['cells'][name][label]['ci95']))
    cell['E_vs_B_counts']={'wins':sum(x<y-1e-8 for x,y in zip(e,b)),'ties':sum(abs(x-y)<1e-8 for x,y in zip(e,b)),'losses':sum(x>y+1e-8 for x,y in zip(e,b))}
    for pol in policies:
        for f,of in [('cost','cost_means'),('cost_per_step','cost_per_step'),('operating_proxy','operating_proxy')]:
            assert math.isclose(cell['metrics'][pol][f],original['cells'][name][of][pol],abs_tol=1e-10)
    for i in panel['configs']:
        record=dict(instance_seed=i,pressure=p,skill=s,planned_price=price,horizon=h,service=json.dumps(asdict(mod.cell_config(i,p,s,price,h))['service']))
        for pol in policies:
            tag=pol[0]; record.update({tag+'_'+f:means[str(i),p,s,str(price),str(h),pol][f] for f in fields})
        instance_rows.append(record)
    cells[name]=cell
pr=cells['high_heterogeneous_price1_h24']
gates=dict(comparator_valid=pr['B_minus_C']['ci95'][0]>0,E_cost_noninferior=pr['E_minus_C_relative']['ci95'][1]<=.05,E_useful=pr['B_minus_E']['ci95'][0]>0 and pr['B_minus_E_relative']['mean']>=.05,operating_guard=pr['E_minus_B_operating']['ci95'][0]>=-.02)
gates['joint_operational']=all(gates.values()); assert gates==original['gates']

# Replay only the entire recorded trace panel, using pinned implementation.
# No fresh scientific seeds; runtime fields excluded. This is not independent reproduction.
bykey={identity(r):r for r in rows}; ts={identity(r)+(r['step'],):r for r in traces}
expected_trace={k+(str(t),) for k in expected if k[-2]==str(panel['episodes'][0]) for t in range(int(k[4]))}
assert len(ts)==len(traces) and set(ts)==expected_trace
maxerr=0.; nreplay=0; ntrace=0
for r in tqdm(configs,desc='audit recorded trace panel',unit='cell'):
    cfg=mod.cell_config(*(r[k] for k in configkeys)); setting=policy['settings'][str(r['planned_price'])]
    solver=mod.Solver(cfg,**setting,depth=policy['depth'],tail=policy['tail'])
    for pol in policies:
        key=tuple(str(r[k]) for k in configkeys)+(str(panel['episodes'][0]),pol)
        result,trace=mod.measure(solver,pol,panel['episodes'][0],True)
        for f,v in result.items():
            if f=='decision_seconds': continue
            err=abs(v-float(bykey[key][f])); maxerr=max(maxerr,err); assert err<1e-8
        for t in trace:
            old=ts[key+(str(t['step']),)]
            for f,v in t.items():
                if f in ('state','action','next_state'): assert json.loads(v)==json.loads(old[f])
                else: assert math.isclose(float(old[f]),float(v),abs_tol=1e-8)
            ntrace+=1
        nreplay+=1
    del solver; gc.collect()
audit=dict(verification_status='ANALYZED',raw_directory=str(raw),source_commit=manifest['source_commit'],hashes_match=True,complete_cartesian_panel=True,episodes=len(rows),development=len(dev),decisions=len(traces),coordination=len(coords),configs=len(configs),summary_arithmetic_match=True,development_selection_matches=True,replayed_episodes=nreplay,replayed_decisions=ntrace,max_replay_error=maxerr,independent_reproduction=False,replay_scope='first fixed episode shock per cell; all stored decisions; pinned shared implementation',file_sha256={n:hashlib.sha256((raw/n).read_bytes()).hexdigest() for n in manifest['outputs']})
save('audit.json',audit); save('statistics.json',dict(cells=cells,gates=gates,development=devmeans))
with (out/'instances.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(instance_rows[0]));w.writeheader();w.writerows(instance_rows)
print('ANALYSIS_DIR',out,flush=True)
print(json.dumps(dict(primary=pr,gates=gates),indent=2),flush=True)
