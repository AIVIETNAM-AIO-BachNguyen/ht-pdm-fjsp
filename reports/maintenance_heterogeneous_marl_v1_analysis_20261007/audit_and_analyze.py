"""Read-only audit of the pulled run; all output goes to this report directory."""
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import types
from collections import defaultdict, Counter

ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl')
RAW=ROOT/'lab_results/maintenance_heterogeneous_marl_v1_full_cpu_20261006T150059Z'
OUT=Path(__file__).resolve().parent
manifest=json.loads((RAW/'manifest.json').read_text());summary=json.loads((RAW/'summary.json').read_text())

def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

print('Stage 1: manifest/file/source hashes',flush=True)
issues=[]
for name,meta in manifest['artifacts'].items():
    p=RAW/name
    if not p.is_file() or p.stat().st_size!=meta['bytes'] or sha(p)!=meta['sha256']:issues.append(name)
assert not issues,issues
assert manifest['status']=='COMPLETED' and manifest['dirty']==''
assert manifest['actual_counts']==manifest['expected_counts']==dict(models=30,training_steps=14745600,
    training_episodes=1228800,optimizer_steps=307200,development_episodes=63200,test_episodes=144000,
    test_decision_intervals=2592000,test_machine_rows=576000)
source_verified=[]
for p in (RAW/'source_snapshot').iterdir():
    if p.suffix=='.py':relative='src/ht_pdm_fjsp/'+p.name
    elif p.suffix=='.md':relative='docs/'+p.name
    elif p.suffix=='.json':relative='configs/'+p.name
    else:relative=p.name
    content=subprocess.check_output(['git','show',manifest['commit']+':'+relative],cwd=ROOT)
    assert hashlib.sha256(content).hexdigest()==sha(p),relative
    source_verified.append(relative)
write('integrity_audit.json',dict(status='PASS',artifact_files=len(manifest['artifacts']),
    bytes=sum(v['bytes'] for v in manifest['artifacts'].values()),commit=manifest['commit'],
    source_files_against_git=len(source_verified),counts=manifest['actual_counts'],
    minutes=(datetime.fromisoformat(manifest['finished'])-datetime.fromisoformat(manifest['started'])).total_seconds()/60))
# Import exactly the verified source snapshot, without changing the working checkout.
package=types.ModuleType('ht_pdm_fjsp');package.__path__=[str(RAW/'source_snapshot')]
sys.modules['ht_pdm_fjsp']=package
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_heterogeneous_marl as runner
from ht_pdm_fjsp import maintenance_heterogeneous_model as world
from ht_pdm_fjsp import maintenance_heterogeneous_policy as policy
from ht_pdm_fjsp.maintenance_dispatch import role_seed,eligible_edges,DispatchState
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig,WaitingEnv

torch.set_num_threads(1)
cfg=manifest['config'];refs=json.loads((RAW/'reference_selection.json').read_text())['selected']
registry=json.loads((RAW/'source_snapshot'/f'{runner.VERSION}_seed_registry.json').read_text())
panels=[set(cfg[k]) for k in ('train_seeds','development_cohorts','development_seeds','test_cohorts','test_seeds')]
assert all(not (a&b) for a,b in itertools.combinations(panels,2))
history=set(registry['declared_seed_values'])|set(range(201,301))|set(range(601,701))
assert not history&set.union(*panels)
forbidden=history|set.union(*panels)
coverage=json.loads((RAW/'training_coverage.json').read_text())
selection=json.loads((RAW/'checkpoint_selection.json').read_text())['selections']
assert len(coverage)==len(selection)==30
assert {r['parameter_count'] for r in coverage}=={72258}
for seed in cfg['train_seeds']:
    own=[r for r in coverage if r['train_seed']==seed]
    assert len(own)==3 and len({r['initial_parameters_sha256'] for r in own})==1
assert all(r['env_steps']==491520 and r['episodes']==40960 and r['optimizer_steps']==10240 and r['parameters_changed'] for r in coverage)
for arm,seed in itertools.product(policy.ARMS,cfg['train_seeds']):
    assert all(role_seed(seed,'heterogeneous_learning_'+role) not in forbidden for role in ('actions','minibatches'))
print('Stage 2: training roles/budgets and paired configuration streams',flush=True)
train_counts=Counter();train_digests=defaultdict(hashlib.sha256);sampled_configuration_checks=0
for row in tqdm(csv.DictReader((RAW/'training_episodes.csv').open()),total=1228800,desc='audit training rows'):
    seed=int(row['train_seed']);index=int(row['episode']);arm=row['algorithm'];key=(arm,seed)
    train_counts[key]+=1;assert index==train_counts[key]
    cs=role_seed(seed,f'heterogeneous_learning_configuration:{index}')
    es=role_seed(seed,f'heterogeneous_learning_environment:{index}')
    assert cs==int(row['config_seed']) and es==int(row['environment_seed']) and cs not in forbidden and es not in forbidden and cs!=es
    cond='nominal' if index%5==0 else 'specialized';assert row['condition']==cond
    assert float(row['objective'])==float(row['base_cost'])+float(row['waiting_cost'])
    train_digests[key].update(f"{index}:{cs}:{es}:{row['config_sha256']}\n".encode())
    if index<=5 or index>40955 or index%1024==0:
        c=world.cohort_config(cs,cond,12)
        assert hashlib.sha256(json.dumps(asdict(c),sort_keys=True).encode()).hexdigest()==row['config_sha256']
        sampled_configuration_checks+=1
assert set(train_counts.values())=={40960}
for seed in cfg['train_seeds']:
    assert len({train_digests[(arm,seed)].hexdigest() for arm in policy.ARMS})==1
progress=list(csv.DictReader((RAW/'training_progress.csv').open()))
assert len(progress)==9600
for arm,seed in itertools.product(policy.ARMS,cfg['train_seeds']):
    own=[r for r in progress if r['algorithm']==arm and int(r['train_seed'])==seed]
    assert [int(r['rollout_update']) for r in own]==list(range(1,321))
    assert int(own[-1]['physical_steps'])==491520 and int(own[-1]['optimizer_steps'])==10240
    assert all(math.isfinite(float(r[k])) for r in own for k in ('loss','gradient_norm','entropy','value_loss'))
write('training_audit.json',dict(status='PASS',episodes=sum(train_counts.values()),progress_rows=len(progress),
    paired_configuration_streams=True,reserved_seed_overlap=0,sampled_config_hash_checks=sampled_configuration_checks,
    identical_initial_parameters=True,identical_parameter_counts=True,development_infeasible=[k for k,v in selection.items() if v['status']!='FEASIBLE']))
print('Stage 3: independent episode statistics and checkpoint selection',flush=True)
rows=[];group=defaultdict(list);episode_map={}
int_keys=('train_seed','checkpoint_step','cohort','horizon','seed')
str_keys=('condition','controller','worker_utilization')
for row in csv.DictReader((RAW/'episodes.csv').open()):
    r={k:int(v) if k in int_keys else v if k in str_keys else float(v) for k,v in row.items()}
    key=(r['train_seed'],r['cohort'],r['condition'],r['horizon'],r['seed'],r['controller'])
    assert key not in episode_map;episode_map[key]=r;rows.append(r)
    group[(r['controller'],r['train_seed'],r['condition'],r['horizon'])].append(r)
assert len(rows)==144000
expected=set()
for cs,cond,h,es,controller in itertools.product(cfg['test_cohorts'],world.CONDITIONS,cfg['horizons'],cfg['test_seeds'],cfg['controls']):
    expected.add((-1,cs,cond,h,es,controller))
for arm,seed,cs,cond,h,es in itertools.product(policy.ARMS,cfg['train_seeds'],cfg['test_cohorts'],world.CONDITIONS,cfg['horizons'],cfg['test_seeds']):
    expected.add((seed,cs,cond,h,es,arm))
assert set(episode_map)==expected
means={key:{metric:float(np.mean([r[metric] for r in panel])) for metric in ('objective','base_cost','waiting_violation')}
       for key,panel in group.items()}

def simpson(f,a,b,n=8192):
    x=np.linspace(a,b,n+1);y=f(x);return (b-a)/n/3*(y[0]+y[-1]+4*y[1:-1:2].sum()+2*y[2:-1:2].sum())

def t_pvalue(t,df=9):
    x=abs(t);constant=math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi)
    if x<=1:tail=.5-constant*simpson(lambda u:(1+u*u/df)**(-(df+1)/2),0,x)
    else:tail=constant*simpson(lambda u:u**(df-1)*(u*u+1/df)**(-(df+1)/2),0,1/x)
    return float(min(1,2*tail))

primary=[];detailed=[]
for key,result in summary['contrasts'].items():
    left,right=result['left'],result['right'];cond,h=result['condition'],result['horizon']
    lv=np.array([means[(left,seed,cond,h)]['objective'] for seed in cfg['train_seeds']])
    rv=np.array([means[(right,seed if right in policy.ARMS else -1,cond,h)]['objective'] for seed in cfg['train_seeds']])
    d=lv-rv;sd=d.std(ddof=1);critical=runner.CI97 if result['confidence_level']==.975 else runner.CI95
    ci=[float(d.mean()-critical*sd/math.sqrt(10)),float(d.mean()+critical*sd/math.sqrt(10))]
    assert np.allclose(ci,result['ci'],rtol=0,atol=1e-10)
    assert abs(float(d.mean())-result['paired_difference'])<1e-10
    assert abs(1-lv.mean()/rv.mean()-result['priced_reduction'])<1e-12
    assert int((d < -1e-9).sum())==result['training_seed_wins']
    p=t_pvalue(float(d.mean()/sd*math.sqrt(10))) if sd else (0. if d.mean() else 1.)
    flips=np.asarray(list(itertools.product((-1,1),repeat=10)))@d/10
    flip_p=float((abs(flips)>=abs(d.mean())-1e-12).mean())
    own=dict(contrast=key,**result,p_two_sided_t=p,exact_sign_flip_p=flip_p,differences=d.tolist())
    detailed.append(own)
    if cond=='specialized' and h==12:primary.append(own)
write('independent_statistics.json',dict(status='PASS',all_contrasts_recomputed=len(detailed),primary=primary,
    contrasts=detailed,replication_unit='10 matched training replicas; fixed cohort/shock panel',
    t_assumption='Approximate normality of paired replica differences; n=10. Sign-flip sensitivity requires symmetry/exchangeability.'))
# Recompute development-only references and guarded checkpoint selection.
dev=list(csv.DictReader((RAW/'development_episodes.csv').open()));assert len(dev)==63200
for cond in world.CONDITIONS:
    costs={name:np.mean([float(r['objective']) for r in dev if r['condition']==cond and r['controller']==name]) for name in world.REFERENCES}
    assert min(world.REFERENCES,key=lambda name:(costs[name],name))==refs[cond]
for arm,seed in itertools.product(policy.ARMS,cfg['train_seeds']):
    candidates=[]
    for steps in (0,122880,245760,368640,491520):
        panel=[r for r in dev if r['controller']==arm and int(r['train_seed'])==seed and int(r['checkpoint_step'])==steps]
        assert len(panel)==400
        if not steps:continue
        specialized=[r for r in panel if r['condition']=='specialized'];nominal=[r for r in panel if r['condition']=='nominal']
        primary_cost=np.mean([float(r['objective']) for r in specialized])
        nominal_ref=[r for r in dev if r['controller']==refs['nominal'] and r['condition']=='nominal']
        special_ref=[r for r in dev if r['controller']==refs['specialized'] and r['condition']=='specialized']
        guard=np.mean([float(r['base_cost']) for r in nominal])<=1.1*np.mean([float(r['base_cost']) for r in nominal_ref])
        guard=guard and np.mean([float(r['waiting_violation']) for r in specialized])<=.02+np.mean([float(r['waiting_violation']) for r in special_ref])
        candidates.append((guard,primary_cost,steps))
    viable=[r for r in candidates if r[0]]
    chosen=min(viable,key=lambda r:(r[1],r[2]))[2] if viable else 491520
    assert selection[f'{arm}:{seed}']['selected_steps']==chosen
# Saved selected weights and symmetric-worker diagnostic.
loaded={}
for arm,seed in itertools.product(policy.ARMS,cfg['train_seeds']):
    path=RAW/selection[f'{arm}:{seed}']['checkpoint'];net,payload=runner.load_checkpoint(path)
    assert payload['physical_steps']==selection[f'{arm}:{seed}']['selected_steps']
    loaded[(arm,seed)]=net
c=world.cohort_config(153000,'nominal',12)
state=DispatchState((2,)*4,(False,)*4,(0,)*4,(0,)*4,(-1,)*4)
batch=policy.feature_batch([c],[state],[12]);symmetry=[]
for seed in cfg['train_seeds']:
    net=loaded[(policy.MARL,seed)];dist,_=net.distribution(batch)
    spread=(dist.logits[0,:,1:].max(-1).values-dist.logits[0,:,1:].min(-1).values).detach().tolist()
    marl_pairs,diag=net.select(c,state,12)
    central_pairs,_=loaded[(policy.CENTRAL,seed)].select(c,state,12)
    assert max(spread)==0 and len(marl_pairs)<=1
    symmetry.append(dict(train_seed=seed,free_worker_logit_spreads=spread,
        marl_proposals=diag['raw_action'],marl_matching=marl_pairs,central_matching=central_pairs))
write('worker_symmetry_diagnostic.json',dict(status='STRUCTURAL_LIMITATION_CONFIRMED',
    state=asdict(state),config=asdict(c),replicas=symmetry,
    explanation='Identical free technician embeddings/edge features yield equal per-agent technician logits. Deterministic argmax nominates the same smallest free worker. Public machine rank does not supply a worker label. Simultaneous distinct-worker assignments are not representable by this deterministic head in homogeneous conditions.'))
# Cohort tables and training progression are descriptive, not new confirmatory gates.
cohort_table=[]
for cond,h,cs in itertools.product(world.CONDITIONS,cfg['horizons'],cfg['test_cohorts']):
    rec=dict(condition=cond,horizon=h,cohort=cs)
    for name in [*policy.ARMS,'joint_rollout','fixed_allocation_rollout',refs[cond]]:
        panel=[r for r in rows if r['condition']==cond and r['horizon']==h and r['cohort']==cs and r['controller']==name]
        rec[name]=float(np.mean([r['objective'] for r in panel]))
    cohort_table.append(rec)
write('cohort_costs.json',cohort_table)
write('pre_decision_audit.json',dict(status='PASS',episodes=144000,development_rows=63200,
    all_primary=primary,checkpoint_selection_recomputed=True,selected_models_reloaded=30,
    structural_nominal_limitation=True))
print('Stage 4: all recorded physical transitions, proposal arbitration and episode costs',flush=True)
# Batched model replay is independent of the audit environment and receives only logged current state.
replay_buffers=defaultdict(list);policy_checks=0;policy_mismatches=[];max_prob_error=0.;max_value_error=0.

def flush(buffer_key):
    global policy_checks,max_prob_error,max_value_error
    buffer=replay_buffers[buffer_key]
    if not buffer:return
    arm,seed=buffer_key;net=loaded[(arm,seed)]
    features=policy.feature_batch([r[0] for r in buffer],[r[1] for r in buffer],[r[2] for r in buffer])
    action,lp,value=net.sample(features,deterministic=True)
    for i,(config,state,h,key,t,diag,pairs) in enumerate(buffer):
        recorded=diag['raw_action'];got=action[i].tolist()
        policy_checks+=1
        max_prob_error=max(max_prob_error,abs(float(lp[i])-diag['log_probability']))
        max_value_error=max(max_value_error,abs(float(value[i])-diag['value']))
        if got!=recorded:
            scalar_pairs,scalar_diag=net.select(config,state,h)
            policy_mismatches.append(dict(key=key,time=t,batch_raw=got,recorded_raw=recorded,
                batch_physical=net.decode(config,state,h,action[i])[0],logged_physical=pairs,
                scalar_physical=scalar_pairs,scalar_raw=scalar_diag['raw_action'],
                scalar_probability_error=abs(scalar_diag['log_probability']-diag['log_probability'])))
    buffer.clear()

last=None;env=None;time_in_episode=0;decisions=0;episodes_finished=0;nominal=Counter();physical_cost_checks=0
nominal_by_seed=defaultdict(Counter)

def finish():
    global episodes_finished
    if last is None:return
    row=episode_map[last]
    assert time_in_episode==last[3]
    for k,v in env.metrics.items():assert abs(v-row[k])<1e-9,(last,k,v,row[k])
    episodes_finished+=1

for row in tqdm(csv.DictReader((RAW/'decisions.csv').open()),total=2592000,desc='physical and policy audit'):
    key=(int(row['train_seed']),int(row['cohort']),row['condition'],int(row['horizon']),int(row['seed']),row['controller'])
    if key!=last:
        finish();c=world.cohort_config(key[1],key[2],key[3]);env=WaitingEnv(c);env.reset(key[4]);last=key;time_in_episode=0
    t=int(row['time']);assert t==time_in_episode
    raw_state=json.loads(row['state']);state=DispatchState(**{k:tuple(v) for k,v in raw_state.items()})
    assert state==env.state
    pairs=tuple(tuple(p) for p in json.loads(row['action']));diag=json.loads(row['controller_diagnostics'])
    if key[-1] in policy.ARMS:
        arm=key[-1];recorded=diag['raw_action']
        if arm==policy.MARL:
            edges=eligible_edges(c,state)
            assert all(p==0 or edges[m][p-1] for m,p in enumerate(recorded))
            resolved,rejections=world.resolve(recorded,t)
            assert resolved==pairs and rejections==diag['collision_rejections']
            assert sum(x>0 for x in recorded)==diag['proposal_attempts']
            if key[2]=='nominal':
                stats=nominal_by_seed[key[0]]
                for target in (stats,nominal):
                    target['intervals']+=1;target['starts']+=len(pairs)
                    target['attempts']+=sum(x>0 for x in recorded);target['rejections']+=rejections
                    target['multi_start_intervals']+=len(pairs)>1
                    target['distinct_worker_proposals']+=len({x for x in recorded if x})>1
                    target['multiple_free_workers']+=sum(r==0 for r in state.remaining)>=2
                    target['lost_with_idle_capacity']+=rejections if sum(r==0 for r in state.remaining)>len(pairs) else 0
        else:assert policy.CATALOGUE[recorded]==pairs
        bkey=(arm,key[0]);replay_buffers[bkey].append((c,state,key[3]-t,key,t,diag,pairs))
        if len(replay_buffers[bkey])>=256:flush(bkey)
    _,_,_,info=env.step(pairs)
    assert info==json.loads(row['physical_metrics']),(key,t,'info')
    assert asdict(env.state)=={k:tuple(v) for k,v in json.loads(row['following']).items()}
    physical_cost_checks+=1;time_in_episode+=1;decisions+=1
finish()
for key in replay_buffers:flush(key)
assert decisions==2592000 and episodes_finished==144000 and policy_checks==2160000
# Each machine's cost/availability row reconciles to its own physical episode.
print('Stage 5: machine-level cost reconciliation',flush=True)
machine_sums=defaultdict(lambda:defaultdict(float));machine_counts=Counter()
for row in tqdm(csv.DictReader((RAW/'machine_metrics.csv').open()),total=576000,desc='machine costs'):
    key=(int(row['train_seed']),int(row['cohort']),row['condition'],int(row['horizon']),int(row['seed']),row['controller'])
    machine_counts[key]+=1
    assert abs(float(row['objective'])-float(row['base_cost'])-float(row['waiting_cost']))<1e-9
    for field in ('objective','base_cost','maintenance_cost','failure_cost','unavailability_cost','waiting_cost','unavailable_ticks','failed_waiting_ticks','overdue_waiting_ticks'):
        machine_sums[key][field]+=float(row[field])
assert len(machine_counts)==144000 and set(machine_counts.values())=={4}
for key,total in machine_sums.items():
    for field,value in total.items():assert abs(value-episode_map[key][field])<1e-9,(key,field)
write('physical_policy_audit.json',dict(status='PASS' if not policy_mismatches else 'NUMERICAL_ACTION_MISMATCH',
    physical_transitions=physical_cost_checks,episodes_replayed=episodes_finished,machine_rows=576000,
    checkpoint_batched_decisions=policy_checks,policy_action_mismatches=len(policy_mismatches),
    max_batch_probability_error=max_prob_error,max_batch_value_error=max_value_error,
    mismatches=policy_mismatches,nominal_marl=dict(nominal),
    nominal_by_seed={str(k):dict(v) for k,v in nominal_by_seed.items()},
    verification_scope='Logged actions/states/costs and selected policy checkpoints only; no re-training or original training reproduction.'))
print(json.dumps(dict(status='ANALYZED',physical_decisions=decisions,policy_checks=policy_checks,
    policy_mismatches=len(policy_mismatches),nominal_symmetry_counters=dict(nominal))),flush=True)
