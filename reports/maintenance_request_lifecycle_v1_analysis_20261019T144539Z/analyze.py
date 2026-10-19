"""Read-only analysis of the completed fixed request-lifecycle diagnostic.

No simulator modules are imported, no controllers are run and no new seeds or
counterfactual policies are evaluated. Raw artifacts remain unchanged.
"""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, required=True, help='Full original raw run (not compact evidence capsule)')
parser.add_argument('--output-dir', type=Path, required=True, help='New analysis directory; existing paths are refused')
args = parser.parse_args()
RAW = args.run_dir.resolve()
OUT = args.output_dir.resolve()
if not RAW.is_dir() or not (RAW/'decisions.jsonl').is_file():
    parser.error('--run-dir must contain the full completed run including decisions.jsonl')
if OUT.is_relative_to(RAW):
    parser.error('--output-dir must be outside the read-only raw run')
if not __debug__:
    parser.error('run without -O: verification requires enabled assertions')
if OUT.exists():
    parser.error('--output-dir already exists; choose a new timestamped directory')
# The independent analysis below is unchanged; only I/O paths are now explicit.
OUT.mkdir(parents=True, exist_ok=False)
VERSION = 'maintenance_request_lifecycle_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def save_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def role(seed, label):
    return int.from_bytes(hashlib.sha256(f'{seed}:{label}'.encode()).digest()[:8], 'big')


manifest = json.loads((RAW/'manifest.json').read_text())
assert manifest['status'] == 'COMPLETED'
sealed = {p.name:sha(p) for p in RAW.iterdir() if p.is_file()}
for name, expected in manifest['artifact_hashes'].items():
    assert sealed[name] == expected, name
for path, expected in manifest['source_hashes'].items():
    target = RAW/'source_snapshot'/Path(path).name if path.startswith('src/') else RAW/('protocol.md' if path.startswith('docs/') else 'seed_registry.json')
    assert sha(target) == expected, path
assert (RAW/'episodes.csv').read_bytes() == (RAW/'episodes.partial.csv').read_bytes()
rows = list(csv.DictReader((RAW/'episodes.csv').open()))
for row in rows:
    for k, v in list(row.items()):
        if k not in ('cell','requester','dispatcher'):
            row[k] = v == 'True' if k == 'completed_wait_defined' else int(v) if k in ('seed','physical_seed') else float(v)
configs = json.loads((RAW/'configs.json').read_text())
requesters = ('threshold','resource_aware')
dispatchers = ('fcfs','risk_matching','rollout')
seeds = list(range(61000000,61000064))
expected_grid = set(itertools.product(configs,requesters,dispatchers,seeds))
key = lambda r:(r['cell'],r['requester'],r['dispatcher'],r['seed'])
by_key = {key(r):r for r in rows}
assert len(rows) == len(by_key) == 3456 and set(by_key) == expected_grid
groups = collections.defaultdict(list)
for row in rows: groups[key(row)[:3]].append(row)
for panel in groups.values():panel.sort(key=lambda r:r['seed'])
metrics = [k for k in rows[0] if k not in ('cell','requester','dispatcher','seed','physical_seed','completed_wait_defined')]
means = {k:{m:float(np.mean([r[m] for r in panel])) for m in metrics} for k,panel in groups.items()}
saved = json.loads((RAW/'summary.json').read_text())
for row in saved['means']:
    values=means[row['cell'],row['requester'],row['dispatcher']]
    assert row['count']==64
    for m in row:
        if m in values:assert abs(row[m]-values[m])<1e-9,(row['cell'],m)

# Independently reconstruct the registered bootstrap using count weights.
for contrast in saved['contrasts']:
    cell=contrast['cell']; a=tuple(contrast['treatment'].split('/')); b=tuple(contrast['baseline'].split('/'))
    delta=np.asarray([r['objective'] for r in groups[(cell,*a)]])-np.asarray([r['objective'] for r in groups[(cell,*b)]])
    rng=np.random.default_rng(role(61300000,f'{VERSION}:full:{cell}:{a}:{b}'))
    draws=rng.integers(0,64,(2000,64))
    counts=np.asarray([np.bincount(draw,minlength=64) for draw in draws])
    interval=np.quantile(counts@delta/64,[.025,.975])
    assert abs(delta.mean()-contrast['mean_cost_difference'])<1e-9
    assert np.allclose(interval,contrast['ci95_exploratory'],atol=1e-9)


def request_record(m, sequence, at, kind):
    return dict(machine=m,sequence=sequence,created_at=at,kind=kind,
                started_at=None,technician=None,upgraded_at=None)


def scalar_step(c, state, tick, submissions, matching, events):
    """Separate scalar reconstruction of the published source contract."""
    p={k:list(v) for k,v in state['physical'].items()}
    requests=[dict(r) if r is not None else None for r in state['requests']]
    generations=list(state['generations']); lifecycle=[]
    def event(label,r,at):lifecycle.append(dict(event=label,at=at,**r))
    assert len(submissions)==len(set(submissions))
    for m in sorted(submissions):
        assert requests[m] is None and not p['failed'][m] and m not in p['assigned']
        generations[m]+=1; requests[m]=request_record(m,generations[m],tick,'preventive')
        event('created',requests[m],tick)
    machines=[m for m,j in matching]; workers=[j for m,j in matching]
    assert len(machines)==len(set(machines)) and len(workers)==len(set(workers))
    pm=cm=0; start_wait=[]
    for m,j in sorted(matching):
        assert requests[m] is not None and requests[m]['started_at'] is None
        assert p['remaining'][j]==0 and m not in p['assigned'] and c['service_time'][m][j]>0
        r=requests[m]; r['started_at']=tick; r['technician']=j
        start_wait.append(tick-r['created_at']); event('started',r,tick)
        p['remaining'][j]=c['service_time'][m][j];p['assigned'][j]=m
        cm+=int(p['failed'][m]);pm+=int(not p['failed'][m])
    serving={m for m in p['assigned'] if m>=0}
    queued=[dict(r) for r in requests if r is not None and r['started_at'] is None]
    down=sum(f or m in serving for m,f in enumerate(p['failed']))
    failed_wait=sum(f and m not in serving for m,f in enumerate(p['failed']))
    failures=0; hazard=0
    for m in range(c['machines']):
        if m in serving:continue
        if p['failed'][m]:p['pending_wait'][m]+=1;continue
        p['ages'][m]=min(c['max_age'],p['ages'][m]+1)
        if p['ages'][m]>=c['failure_age']:
            hazard+=1
            if events[m]:
                p['failed'][m]=True;p['pending_wait'][m]=0;failures+=1
    completed=0;completed_wait=[]
    for j,m in enumerate(p['assigned']):
        if m<0:continue
        p['remaining'][j]-=1
        if p['remaining'][j]==0:
            p['ages'][m]=math.floor(p['ages'][m]*(1-c['restoration'][m][j]))
            p['failed'][m]=False;p['pending_wait'][m]=0;p['assigned'][j]=-1;completed+=1
    for m,r in enumerate(requests):
        if r is not None and r['started_at'] is not None:
            if m not in p['assigned']:
                completed_wait.append(r['started_at']-r['created_at']);event('completed',r,tick+1);requests[m]=None
        elif p['failed'][m]:
            if r is None:
                generations[m]+=1;r=request_record(m,generations[m],tick+1,'corrective');event('created',r,tick+1)
            elif r['kind']=='preventive':
                r['kind']='corrective';r['upgraded_at']=tick+1;event('upgraded',r,tick+1)
            requests[m]=r
    mc=pm*c['preventive_cost']+cm*c['corrective_cost']
    uc=down*c['unavailable_cost'];fc=failures*c['failure_cost'];cost=mc+uc+fc
    ages=[tick+1-r['created_at'] for r in requests if r is not None and r['started_at'] is None]
    info=dict(objective=cost,maintenance_cost=mc,unavailability_cost=uc,failure_cost=fc,
        preventive_jobs=pm,corrective_jobs=cm,jobs=pm+cm,completed_jobs=completed,
        failures=failures,unavailable_ticks=down,failed_waiting_ticks=failed_wait,
        service_ticks=len(serving),invalid_assignments=0,
        requests_created=sum(e['event']=='created' for e in lifecycle),
        requests_upgraded=sum(e['event']=='upgraded' for e in lifecycle),service_starts=len(matching),
        requests_completed=len(completed_wait),completed_wait_total=sum(completed_wait),
        queued_pm_ticks=sum(r['kind']=='preventive' for r in queued),
        queued_cm_ticks=sum(r['kind']=='corrective' for r in queued),request_wait_ticks=len(queued),
        max_wait_age=max([0,*start_wait,*ages]),lifecycle=lifecycle)
    following=dict(physical=p,requests=requests,generations=generations)
    return following,info,hazard


mechanisms=collections.defaultdict(collections.Counter)
fingerprints={};active_key=None;seen=set();nsteps=0
with (RAW/'decisions.jsonl').open() as stream:
    for line in stream:
        r=json.loads(line);k=key(r);c=configs[k[0]];tick=r['tick']
        if k!=active_key:
            if active_key is not None:
                assert current_tick==64
                for m,v in accumulated.items():assert v==by_key[active_key][m],(active_key,m,v,by_key[active_key][m])
                fingerprints[active_key]=(physical_hash.hexdigest(),admission_hash.hexdigest())
            assert k not in seen;seen.add(k);active_key=k;current_tick=0
            physical_hash=hashlib.sha256();admission_hash=hashlib.sha256();accumulated=collections.Counter()
            pseed=role(k[3],VERSION+':physical');assert pseed==by_key[k]['physical_seed']
            initial=random.Random(role(pseed,'initial'));shocks=random.Random(role(pseed,'failures'))
            previous=dict(physical=dict(ages=[initial.randrange(c['failure_age']) for _ in range(4)],
                failed=[False]*4,pending_wait=[0]*4,remaining=[0]*2,assigned=[-1]*2),requests=[None]*4,generations=[0]*4)
        assert tick==current_tick and previous==r['state']
        events=[shocks.random()<c['failure_probability'] for _ in range(4)]
        assert r['events']==events
        following,info,hazard=scalar_step(c,r['state'],tick,r['submissions'],r['matching'],events)
        if following!=r['following'] or info!=r['info'] or r['reward']!=-info['objective']:
            print('SCALAR_DIFFERENCE',k,tick,{m:(info.get(m),r['info'].get(m)) for m in set(info)|set(r['info']) if info.get(m)!=r['info'].get(m)},
                  'state',following if following!=r['following'] else 'MATCH',flush=True)
            raise AssertionError((k,tick))
        for m,v in info.items():
            if m=='lifecycle':continue
            if m=='max_wait_age':accumulated[m]=max(accumulated[m],v)
            else:accumulated[m]+=v
        physical_hash.update(json.dumps([r['state']['physical'],r['matching'],following['physical']],sort_keys=True).encode())
        admission_hash.update(json.dumps(r['submissions']).encode())
        if tick<48:
            stat=mechanisms[k[:3]];stat['ticks']+=1;stat['hazard_exposure']+=hazard
            stat['failures']+=info['failures'];stat['pm_starts']+=info['preventive_jobs'];stat['cm_starts']+=info['corrective_jobs']
            stat['upgrades']+=info['requests_upgraded'];stat['cm_wait_ticks']+=info['queued_cm_ticks']
            scores=r['diagnostic']['root_scores']
            if scores:
                base=scores[0]['matching'];chosen=r['matching']
                stat['forecast_roots_considered']+=len(scores)
                if chosen!=base:
                    stat['changed_matching_ticks']+=1
                    if sorted(m for m,j in chosen)==sorted(m for m,j in base):stat['same_subset_assignment_changes']+=1
                    else:stat['subset_or_timing_changes']+=1
                    stat['cardinality_changes']+=int(len(chosen)!=len(base))
                    stat['wait_instead_of_service']+=int(not chosen and bool(base))
            p=r['state']['physical'];selected_m={m for m,j in r['matching']};selected_j={j for m,j in r['matching']}
            free=[j for j,left in enumerate(p['remaining']) if left==0 and j not in selected_j]
            unserved=[m for m,failed in enumerate(p['failed']) if failed and m not in p['assigned'] and m not in selected_m
                      and any(c['service_time'][m][j]>0 for j in free)]
            stat['idle_eligible_cm_ticks']+=int(bool(unserved));stat['idle_eligible_cm_machine_ticks']+=len(unserved)
            stat['started_long_cm']+=sum(p['failed'][m] and c['service_time'][m][j]>=6 for m,j in r['matching'])
        previous=following;current_tick+=1;nsteps+=1
    assert current_tick==64
    for m,v in accumulated.items():assert v==by_key[active_key][m],(active_key,m)
    fingerprints[active_key]=(physical_hash.hexdigest(),admission_hash.hexdigest())
assert seen==expected_grid and nsteps==221184

requester_identity=[]
for cell,d in itertools.product(configs,dispatchers):
    a=[fingerprints[cell,'threshold',d,s] for s in seeds];b=[fingerprints[cell,'resource_aware',d,s] for s in seeds]
    requester_identity.append(dict(cell=cell,dispatcher=d,
        identical_physical_and_matching_episodes=sum(x[0]==y[0] for x,y in zip(a,b)),
        identical_admission_episodes=sum(x[1]==y[1] for x,y in zip(a,b)),count=64))
save_csv('requester_identity.csv',requester_identity)
mean_rows=[dict(cell=k[0],requester=k[1],dispatcher=k[2],**v) for k,v in means.items()]
save_csv('means.csv',mean_rows)
save_csv('registered_contrasts.csv',[{**r,'ci95_exploratory':json.dumps(r['ci95_exploratory'])} for r in saved['contrasts']])
save_csv('mechanisms.csv',[dict(cell=k[0],requester=k[1],dispatcher=k[2],**{m:v.get(m,0) for m in sorted(set.union(*(set(x) for x in mechanisms.values())))}) for k,v in mechanisms.items()])

comparison=[]
for cell in configs:
    a=means[cell,'threshold','rollout'];b=means[cell,'threshold','risk_matching']
    contrast=next(r for r in saved['contrasts'] if r['cell']==cell and r['treatment']=='threshold/rollout')
    panel_a=groups[cell,'threshold','rollout'];panel_b=groups[cell,'threshold','risk_matching']
    diff=np.array([a['objective']-b['objective'] for a,b in zip(panel_a,panel_b)])
    comparison.append(dict(cell=cell,fcfs_cost=means[cell,'threshold','fcfs']['objective'],
        risk_cost=b['objective'],rollout_cost=a['objective'],cost_gain_pct=100*(1-a['objective']/b['objective']),
        difference_ci_low=contrast['ci95_exploratory'][0],difference_ci_high=contrast['ci95_exploratory'][1],
        availability_change_pp=100*(a['availability']-b['availability']),
        policy_failure_change=a['policy_failures']-b['policy_failures'],
        cm_wait_change=a['queued_cm_ticks']-b['queued_cm_ticks'],
        all_request_wait_change=a['request_wait_ticks']-b['request_wait_ticks'],
        H_pending_change=a['H_pending']-b['H_pending'],end_pending_change=a['end_pending']-b['end_pending'],
        maintenance_cost_change=a['maintenance_cost']-b['maintenance_cost'],
        unavailability_cost_change=a['unavailability_cost']-b['unavailability_cost'],
        failure_cost_change=a['failure_cost']-b['failure_cost'],
        policy_cost_change=a['policy_cost']-b['policy_cost'],tail_cost_change=a['tail_cost']-b['tail_cost'],
        winning_seeds=int(sum(diff<0)),equal_seeds=int(sum(diff==0)),losing_seeds=int(sum(diff>0))))
save_csv('rollout_vs_risk.csv',comparison)

# Export standard scientific plots, rather than a bespoke interactive graphic.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
profiles=['interchangeable','reciprocal','specialist_generalist']
labels=['Interchangeable','Reciprocal specialization','Generalist + specialist']
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
for ax,metric,title,limit in zip(axes,['cost_gain_pct','availability_change_pp'],
        ['Cost reduction vs risk matching (%)','Availability change (percentage points)'],[12,8]):
    matrix=np.array([[next(r[metric] for r in comparison if r['cell']==f'{p}_scale{s}') for s in (1,2,3)] for p in profiles])
    im=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit)
    ax.set_xticks(range(3),['1','2','3']);ax.set_xlabel('Service duration multiplier')
    ax.set_yticks(range(3),labels);ax.set_title(title,fontsize=11,pad=12)
    for i,j in itertools.product(range(3),repeat=2):
        ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.55*limit else 'black',fontweight='bold')
    fig.colorbar(im,ax=ax,shrink=.7)
fig.suptitle('Rollout benefits depend on the physical profile\nThreshold requester; 64 paired seeds per cell',fontsize=13)
fig.savefig(OUT/'profile_tradeoffs.png',dpi=180,bbox_inches='tight',pad_inches=.15);fig.savefig(OUT/'profile_tradeoffs.svg',bbox_inches='tight',pad_inches=.15);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4.6),layout='constrained')
x=np.arange(9);width=.24
for offset,metric,label,color in [(-1,'maintenance_cost_change','Maintenance','#577590'),(0,'unavailability_cost_change','Unavailability','#ed9b40'),(1,'failure_cost_change','Failures','#bc4749')]:
    ax.bar(x+offset*width,[r[metric] for r in comparison],width,label=label,color=color)
ax.scatter(x,[r['rollout_cost']-r['risk_cost'] for r in comparison],color='black',marker='D',label='Total',s=24,zorder=5)
ax.axhline(0,color='#555555',linewidth=.7)
ax.set_xticks(x,[p+' / '+str(s) for p in ('Equal','Reciprocal','Generalist') for s in (1,2,3)],rotation=25,ha='right')
ax.set_ylabel('Rollout minus risk cost; negative is cheaper')
ax.set_title('Economic cost differences by component (H48 + common tail16)')
ax.legend(ncols=4,frameon=False,loc='lower left')
fig.savefig(OUT/'cost_components.png',dpi=180,bbox_inches='tight',pad_inches=.15);fig.savefig(OUT/'cost_components.svg',bbox_inches='tight',pad_inches=.15);plt.close(fig)

for name,expected in sealed.items():assert sha(RAW/name)==expected,name
save_json('validation.json',dict(status='PASS',raw_hashes_before_and_after=sealed,
    expected_episode_grid=3456,scalar_transitions_checked=nsteps,
    independent_scalar_reconstruction=True,summary_mean_rows_checked=54,bootstrap_contrasts_checked=45,
    forecast_banks_reexecuted=False,new_controller_runs=0,new_seeds=0))
print(json.dumps(dict(status='PASS',output=str(OUT),scalar_transitions=nsteps)))
print('REQUESTER IDENTITIES',json.dumps(requester_identity))
print('MECHANISMS threshold',json.dumps({str(k):dict(v) for k,v in mechanisms.items() if k[1]=='threshold' and k[2] in ('rollout','risk_matching')}))
