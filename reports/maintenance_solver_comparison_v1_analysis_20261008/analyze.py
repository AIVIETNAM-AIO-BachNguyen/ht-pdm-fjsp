"""Read-only independent reconciliation of the completed lab run. No training."""
import csv
import hashlib
import itertools
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'lab_results/maintenance_solver_comparison_v1_full_cpu_20261007T151955Z'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, '/Users/bachng/.codex/worktrees/maintenance-solver-comparison/marl/src')
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, transition
from ht_pdm_fjsp.maintenance_dispatch import DispatchState

def read(name):
    return json.loads((RUN / name).read_text())

def rows(name):
    with (RUN / name).open(newline='') as f:
        yield from csv.DictReader(f)

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

manifest = read('manifest.json')
bad = []
for name, meta in manifest['artifacts'].items():
    path = RUN / name
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    if path.stat().st_size != meta['bytes'] or h.hexdigest() != meta['sha256']:
        bad.append(name)
assert not bad, bad
print('Artifact hashes verified:', len(manifest['artifacts']), flush=True)

metrics = ['objective', 'base_cost', 'waiting_cost', 'maintenance_cost', 'failure_cost',
           'unavailability_cost', 'waiting_violation', 'failures', 'utilization',
           'terminal_pending', 'corrective_requests', 'served_requests',
           'censored_requests', 'communication_bytes', 'communication_rounds',
           'invalid_assignments', 'collision_rejections', 'inference_seconds']
sums = defaultdict(Counter)
counts = Counter()
replica = defaultdict(list)
profile = defaultdict(list)
episode_keys = set()
episode_replay_expected = {}
for r in rows('episodes.csv'):
    group = (r['controller'], r['condition'], int(r['horizon']))
    key = (r['controller'], int(r['train_seed']), int(r['cohort']), r['condition'], int(r['horizon']), int(r['seed']))
    assert key not in episode_keys
    episode_keys.add(key)
    counts[group] += 1
    vals = {m: float(r[m]) for m in metrics}
    sums[group].update(vals)
    assert abs(vals['objective']-vals['base_cost']-vals['waiting_cost']) < 1e-8
    assert abs(vals['base_cost']-vals['maintenance_cost']-vals['failure_cost']-vals['unavailability_cost']) < 1e-8
    assert vals['invalid_assignments'] == vals['collision_rejections'] == 0
    replica[group + (int(r['train_seed']),)].append(vals['objective'])
    profile[group + (int(r['cohort']),)].append(vals['objective'])
    episode_replay_expected[key] = (vals['objective'], vals['base_cost'], vals['waiting_cost'])
assert len(episode_keys) == 286720
means = {f'{a}:{c}:H{h}': {'episodes': counts[(a,c,h)], **{m:sums[(a,c,h)][m]/counts[(a,c,h)] for m in metrics}}
         for a,c,h in counts}
assert all(n == (640 if a in manifest['config']['controls'] else 6400) for (a,c,h),n in counts.items())

contrasts = {}
for label, original in read('summary.json')['contrasts'].items():
    a,b,c,h = original['left'],original['right'],original['condition'],original['horizon']
    units = manifest['config']['test_cohorts'] if original['unit']=='cohort' else manifest['config']['train_seeds']
    source = profile if original['unit']=='cohort' else replica
    def mean(a,u):
        key=(a,c,h,u)
        if key not in source: key=(a,c,h,-1)
        return st.mean(source[key])
    differences=[mean(a,u)-mean(b,u) for u in units]
    delta=st.mean(differences)
    n=len(differences)
    critical = {(10,True):3.110934823179474,(10,False):2.2621571627982055,
                (32,True):2.6519126890308002,(32,False):2.0395134463964085}[(n,original['confirmatory'])]
    error=critical*st.stdev(differences)/math.sqrt(n)
    ci=[delta-error,delta+error]
    assert abs(delta-original['paired_difference']) < 1e-8, label
    assert max(abs(x-y) for x,y in zip(ci,original['ci'])) < 1e-8, label
    wins=sum(x<0 for x in differences)
    assert wins==original['wins']
    contrasts[label]={'delta':delta,'ci':ci,'wins':wins,'replicates':n,
                      'unit':original['unit'],'differences':differences,
                      'status':original['status'],'priced_reduction':original['priced_reduction'],
                      'base_reduction':original['base_reduction']}
    if n==10:
        observed=abs(sum(differences))
        contrasts[label]['exact_sign_flip_two_sided_p']=sum(abs(sum(x*s for x,s in zip(differences, signs))) >= observed-1e-10
            for signs in itertools.product((-1,1),repeat=n))/2**n
print('Episodes and all contrasts independently reconciled', flush=True)

requests=defaultdict(Counter)
waits=defaultdict(list)
request_count=0
for r in rows('request_metrics.csv'):
    request_count+=1
    group=(r['controller'],r['condition'],int(r['horizon']))
    served=r['served']=='True'; censored=r['censored']=='True'; wait=int(r['wait'])
    assert served != censored
    assert wait == int(r['end'])-int(r['arrival'])
    requests[group].update(total=1,served=served,censored=censored,
        overdue=r['overdue_observed']=='True',terminal_zero_censored=censored and wait==0)
    if served: waits[group].append(wait)
reqout={}
for group,v in requests.items():
    assert v['total']==round(sums[group]['corrective_requests'])
    assert v['served']==round(sums[group]['served_requests'])
    assert v['censored']==round(sums[group]['censored_requests'])
    w=sorted(waits[group])
    reqout[f'{group[0]}:{group[1]}:H{group[2]}']={**v,'censored_fraction':v['censored']/v['total'],
        'observed_overdue_fraction':v['overdue']/v['total'],
        'served_wait_mean':st.mean(w),'served_wait_p95_nearest_rank':w[math.ceil(.95*len(w))-1],
        'served_wait_max':max(w)}

latencies=defaultdict(list)
for r in rows('latency.csv'):
    latencies[(r['controller'],r['condition'])].append(float(r['seconds'])*1000)
latout={f'{a}:{c}':{'n':len(v),'mean_ms':st.mean(v),'median_ms':st.median(v),
                  'p95_ms':sorted(v)[math.ceil(.95*len(v))-1]}
        for (a,c),v in latencies.items()}

coverage=read('training_coverage.json')
training=defaultdict(Counter)
training_signatures={}
max_reward_error=0
trainrows=0
for r in rows('training_episodes.csv'):
    trainrows+=1
    key=(r['algorithm'],int(r['train_seed']))
    training[key].update(episodes=1,**{r['condition']:1})
    index=(int(r['train_seed']),int(r['episode']))
    signature=(r['config_seed'],r['environment_seed'],r['config_sha256'],r['condition'])
    if index not in training_signatures: training_signatures[index]=signature
    assert training_signatures[index]==signature
    assert abs(sum(json.loads(r['machine_costs']))-float(r['objective'])) < 1e-7
    max_reward_error=max(max_reward_error,abs(float(r['reward_reconciliation_error'])))
assert trainrows==2048000
for key,v in training.items(): assert v['episodes']==40960 and v['specialized']==32768 and v['nominal']==8192
progress=defaultdict(list)
for r in rows('training_progress.csv'): progress[(r['algorithm'],int(r['train_seed']))].append(r)
for key,v in progress.items():
    assert len(v)==320
    assert int(v[-1]['physical_steps'])==491520 and int(v[-1]['optimizer_steps'])==10240
    assert [int(r['rollout_update']) for r in v]==list(range(1,321))
selections=read('checkpoint_selection.json')['selections']
selectedupdates=defaultdict(Counter)
for v in selections.values():
    assert v['status']=='FEASIBLE'
    candidates=[c for c in v['candidates'] if c['update'] and all(c['development_guards'].values())]
    best=min(candidates,key=lambda x:(x['development_primary_cost'],x['steps']))
    assert best['steps']==v['selected_steps']
    selectedupdates[v['algorithm']][str(v['selected_update'])]+=1
print('Requests, latency, full training logs and checkpoint choices reconciled', flush=True)

save('independent_analysis.json', {'manifest_status':manifest['status'],'commit':manifest['commit'],
    'verified_artifact_hashes':len(manifest['artifacts']),'episodes':len(episode_keys),
    'request_rows':request_count,'training_episode_rows':trainrows,'training_progress_rows':sum(map(len,progress.values())),
    'matched_training_configuration_streams':len(training_signatures),
    'maximum_logged_reward_reconciliation_error':max_reward_error,'means':means,'contrasts':contrasts,
    'requests':reqout,'latency':latout,'selected_updates':dict(selectedupdates),
    'training_seconds_by_arm':{a:sum(v['training_wall_seconds'] for v in coverage if v['algorithm']==a)
        for a in manifest['config']['arms']}})

# Replay every physical transition with the verified experiment implementation.
# This validates dynamics/actions/costs, not learned policy inference.
configs={k:WaitingConfig(**{**v,'service_time':tuple(map(tuple,v['service_time'])),
                            'restoration':tuple(map(tuple,v['restoration']))})
         for k,v in read('evaluation_configs.json').items() if k.startswith('test_')}
streams={}
previous=None; expectedstate=None; tick=0; total=[0.,0.,0.]; decisions=0; episodes=0
for r in rows('decisions.csv'):
    key=(r['controller'],int(r['train_seed']),int(r['cohort']),r['condition'],int(r['horizon']),int(r['seed']))
    configkey=f'test_{r["cohort"]}_{r["condition"]}_H{r["horizon"]}'
    cfg=configs[configkey]
    streamkey=(configkey,int(r['seed']))
    if streamkey not in streams:
        env=WaitingEnv(cfg);env.reset(streamkey[1]);streams[streamkey]=(env.state,env.events)
    initial,events=streams[streamkey]
    if key!=previous:
        if previous is not None:
            assert tick==previous[4]
            assert max(abs(x-y) for x,y in zip(total,episode_replay_expected[previous]))<1e-8
            episodes+=1
        previous=key;tick=0;expectedstate=initial;total=[0.,0.,0.]
    state=DispatchState(**{k:tuple(v) for k,v in json.loads(r['state']).items()})
    assert state==expectedstate and int(r['time'])==tick
    action=tuple(map(tuple,json.loads(r['action'])))
    following,_,info=transition(cfg,state,action,events[tick])
    loggedfollowing=DispatchState(**{k:tuple(v) for k,v in json.loads(r['following']).items()})
    assert following==loggedfollowing
    loggedinfo=json.loads(r['physical_metrics'])
    assert info.keys()==loggedinfo.keys()
    assert all(abs(info[k]-loggedinfo[k])<1e-8 for k in info)
    total=[x+info[m] for x,m in zip(total,('objective','base_cost','waiting_cost'))]
    expectedstate=following;tick+=1;decisions+=1
    if decisions%500000==0: print('Physical transitions reconciled:',decisions,flush=True)
assert tick==previous[4]
assert max(abs(x-y) for x,y in zip(total,episode_replay_expected[previous]))<1e-8
episodes+=1
assert decisions==5160960 and episodes==286720
save('physical_replay_audit.json',{'decision_rows':decisions,'episodes':episodes,'mismatches':0,
    'checks':['initial state and shock stream','state continuity','valid assignment','following state',
              'all physical tick metrics','episode cost reconciliation'],
    'limitation':'Uses frozen experiment physics; does not independently reimplement physics or replay actor inference.'})
print('Completed full physical replay:',decisions,episodes,flush=True)
