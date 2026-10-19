from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import asdict
import csv,json,math,statistics,hashlib
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import passive_technician_heuristic_assignment as exp
from ht_pdm_fjsp.passive_technician_marl import PassiveTechnicianEnv,PassiveState

ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl')
RAW=ROOT/'lab_results/ra_qmix_heuristic_assignment_full_cpu_20261004T151339Z'
OUT=ROOT/'reports/ra_qmix_heuristic_assignment_analysis_20261005T025516Z'
torch.set_num_threads(1)
def read(name):
    with (RAW/name).open() as f:return list(csv.DictReader(f))
def numrows(rows):
    return [{k:(v if k in ('algorithm','cell','config') else float(v)) for k,v in row.items()} for row in rows]
def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):exp.base._write_csv(OUT/name,rows)
manifest=json.loads((RAW/'manifest.json').read_text());summary=json.loads((RAW/'summary.json').read_text())
assert manifest['status']=='COMPLETED' and manifest['git_revision']=='9cff694b3b19f45c4219adf7728ebb5d59900c88' and not manifest['git_dirty']
sourcehash=hashlib.sha256((RAW/'source_snapshot.py').read_bytes()).hexdigest()
assert sourcehash==manifest['source_sha256']==hashlib.sha256(Path(exp.__file__).read_bytes()).hexdigest()
assert all((RAW/p).is_file() for p in manifest['outputs'])
cfg=exp.settings('full');resolved=json.loads((RAW/'resolved_config.json').read_text())
assert cfg==resolved['settings']
assert json.loads((RAW/'feature_contract.json').read_text())==exp.FEATURE_CONTRACT
assert json.loads((RAW/'seed_audit.json').read_text())==exp.seed_audit(cfg)
for n in range(2,6):
    for c,config in exp.cells('full',n).items():assert json.loads(json.dumps(asdict(config)))==resolved['cells'][str(n)][c]
episodes=numrows(read('episodes.csv'));dev=numrows(read('development_episodes.csv'));refs=numrows(read('reference_episodes.csv'))
assert (len(episodes),len(dev),len(refs))==(13500,1800,680)
for rows in (episodes,dev):
    assert len({(r['machines'],r['algorithm'],r['train_seed'],r['cell'],r['eval_seed']) for r in rows})==len(rows)
    assert all(r['invalid_requests']==0 and abs(r['objective']-sum(r[k] for k in ('maintenance_cost','queue_waiting_cost','failure_cost','downtime_cost')))<1e-9 for r in rows)
assert read('episodes.csv')==read('episodes.partial.csv')
assert read('coordination.csv')==read('episodes.csv')+read('development_episodes.csv')
pairs,recomputed=exp.summarize(episodes,cfg,True)
assert recomputed['primary']==summary['primary']
oldpairs=numrows(read('paired_seed_metrics.csv'))
assert all(abs(a[k]-b[k])<1e-10 for a,b in zip(pairs,oldpairs) for k in a)
csvout('paired_seed_metrics.csv',pairs)
parameters=numrows(read('parameter_counts.csv'));assert len(parameters)==90
for n in range(2,6):assert len({r['parameters'] for r in parameters if r['machines']==n})==1
coverage=[]
for p in RAW.rglob('training_coverage.json'):
    r=json.loads(p.read_text());coverage.append(r)
    assert r['env_steps']==60000 and r['optimizer_updates']==14873 and r['target_syncs']==29
    assert r['evaluation_config_training_transitions']==0 and r['teacher_labels_seen']==(0 if r['algorithm']==exp.ARMS[0] else 60000)
    assert r['training_episodes']==(10000 if r['machines']==2 else 5000)
    seed=r['train_seed'];assert r['rng_seeds']==dict(initialization=seed,env_a=seed+1000,env_b=seed+2000,exploration=seed+3000,replay=seed+4000)
assert len(coverage)==90 and sum(r['env_steps'] for r in coverage)==5400000
assert sum(r['optimizer_updates'] for r in coverage)==1338570
training_counts=Counter(); last_episode={}
with (RAW/'training_episodes.csv').open() as f:
    for r in csv.DictReader(f):
        key=(int(r['machines']),r['algorithm'],int(r['train_seed']))
        training_counts[key]+=1
        assert int(r['episode'])==training_counts[key] and float(r['invalid_requests'])==0
        assert r['config']==('train_a' if int(r['episode'])%2 else 'train_b')
        assert int(r['env_steps'])==int(r['episode'])*(6 if key[0]==2 else 12)
assert sum(training_counts.values())==600000 and len(training_counts)==90
for k,v in training_counts.items():assert v==(10000 if k[0]==2 else 5000)
logs=numrows(read('training_progress.csv'));assert len(logs)==13410
loggroups=defaultdict(list)
for r in logs:
    assert all(math.isfinite(r[k]) for k in ('td_loss','loss','gradient_norm','assignment_ce'))
    if r['algorithm']==exp.ARMS[0]:assert r['assignment_ce']==0
    assert abs(r['loss']-r['td_loss']-r['assignment_ce'])<1e-5
    assert r['env_steps']==512+4*(r['update']-1)
    assert r['target_syncs']==r['update']//500
    loggroups[r['machines'],r['algorithm'],r['train_seed']].append(r)
for rr in loggroups.values():assert [int(r['update']) for r in rr]==list(range(100,14873,100))+[14873]

def group_summary(rows):
    groups=defaultdict(list)
    for r in rows:groups[int(r['machines']),r['cell'],r.get('algorithm','heuristic_reference')].append(r)
    result=[]
    for (n,c,a),rr in sorted(groups.items()):
        row=dict(machines=n,cell=c,algorithm=a,episodes=len(rr))
        for k in ('objective','jobs','failures','collisions','waiting','busy_requests','maintenance_cost','queue_waiting_cost','failure_cost','downtime_cost'):
            row[k]=statistics.fmean(r[k] for r in rr)
        row['episode_cost_p95']=sorted(r['objective'] for r in rr)[math.ceil(.95*len(rr))-1]
        result.append(row)
    return result
cells=group_summary(episodes);development=group_summary(dev);reference=group_summary(refs)
csvout('cell_means.csv',cells);csvout('development_means.csv',development);csvout('reference_means.csv',reference)
index={(r['machines'],r['cell'],r['algorithm']):r for r in cells}
size_results=[]
for n in range(2,6):
    per_seed=[];nominal_delta=[]
    for s in cfg['train_seeds']:
        avg=lambda a,cs:statistics.fmean(r['objective'] for r in episodes if r['machines']==n and r['algorithm']==a and r['train_seed']==s and r['cell'] in cs)
        a,b=avg(exp.ARMS[0],('medium','high')),avg(exp.ARMS[1],('medium','high'))
        nom_a,nom_b=avg(exp.ARMS[0],('nominal',)),avg(exp.ARMS[1],('nominal',))
        per_seed.append(dict(machines=n,train_seed=s,td_cost=a,heuristic_cost=b,delta=b-a,nominal_td=nom_a,nominal_heuristic=nom_b,nominal_delta=nom_b-nom_a))
        nominal_delta.append(nom_b-nom_a)
    delta=statistics.fmean(r['delta'] for r in per_seed);baseline=statistics.fmean(r['td_cost'] for r in per_seed)
    size_results.append(dict(machines=n,diagnostic_td=baseline,diagnostic_heuristic=baseline+delta,relative_reduction=-delta/baseline,
        diagnostic_wins=sum(r['delta']<0 for r in per_seed),nominal_losses=sum(r['nominal_delta']>1e-8 for r in per_seed),
        nominal_td=index[n,'nominal',exp.ARMS[0]]['objective'],nominal_heuristic=index[n,'nominal',exp.ARMS[1]]['objective']))
    csvout(f'size_{n}_paired.csv',per_seed)
csvout('size_summary.csv',size_results)
exact=numrows(read('small_exact_metrics.csv'));assert len(exact)==120
oracles={c:exp.base.ExactOracle(config,progress=False) for c,config in exp.cells('full',2).items()}
# Independently check all stored training labels against the same-source exact DP.
labels_checked=0
for c in ('train_a','train_b'):
    for r in json.loads((RAW/'machines_2'/f'oracle_{c}_teacher.json').read_text()):
        st=r['state'];state=PassiveState(tuple(st['ages']),tuple(st['failed']),tuple(st['busy_until']),tuple(st['assigned_machine']),tuple(tuple(q) for q in st['queues']))
        node=oracles[c].nodes[int(r['time']),state];action=tuple(r['actions'])
        assert action in node['actions'] and max(node['q95'])-node['q95'][node['actions'].index(action)]<1e-8
        labels_checked+=1
assert labels_checked==826
lookup={(int(r['machines']),r['algorithm'],int(r['train_seed']),r['cell'],int(r['eval_seed'])):r for r in episodes+dev}
by_model=defaultdict(list)
for key,r in lookup.items():by_model[key[:3]].append(r)
replayed=0;exact_rechecked=0;machine_rows=[];traces=[];posthoc_train=[]

def detailed_rollout(model,config,seed,n,arm,trainseed):
    env=PassiveTechnicianEnv(config);env.reset()
    jobs=[0]*n;failed_defer=[0]*n;actionable_failed=[0]*n;failed_ticks=[0]*n
    for events in exp.failure_grid(config,seed):
        obs,masks=env.observations(),env.action_masks();action=exp.greedy(model,obs,masks,config)
        for i,a in enumerate(action):
            if env.state.failed[i]:
                failed_ticks[i]+=1
                if any(masks[i][1:]):
                    actionable_failed[i]+=1;failed_defer[i]+=int(a==0)
        state,cost,info=env.transition(env.state,action,env.time,events)
        # Service starts reset a machine's age; request arrival itself need not start it.
        for i in range(n):jobs[i]+=int(state.ages[i]==0)
        if n in (4,5) and trainseed==99000 and seed==99200:
            traces.append(dict(machines=n,algorithm=arm,train_seed=trainseed,eval_seed=seed,time=env.time,
                ages=json.dumps(env.state.ages),failed=json.dumps(env.state.failed),actions=json.dumps(action),
                heuristic_actions=json.dumps(exp.heuristic_action(obs,masks,config)),step_cost=cost))
        env.state,env.time=state,env.time+1
    return jobs,failed_defer,actionable_failed,failed_ticks

for (n,arm,s),rr in tqdm(sorted(by_model.items()),desc='Replay 90 checkpoints',unit='model'):
    path=RAW/f'machines_{n}'/arm/f'train_seed_{s}'/'model.pt'
    payload=torch.load(path,weights_only=True)
    assert payload['teacher_definition_sha256']==sourcehash and payload['train_seed']==s and payload['algorithm']==arm
    model=exp.load_checkpoint(path);configs=exp.cells('full',n)
    for r in rr:
        actual=exp.rollout(configs[r['cell']],int(r['eval_seed']),model)
        for k,v in actual.items():assert abs(v-r[k])<1e-8,(n,arm,s,r['cell'],r['eval_seed'],k,v,r[k])
        replayed+=1
        if r['cell']=='nominal':
            jobs,defer,failed,total=detailed_rollout(model,configs['nominal'],int(r['eval_seed']),n,arm,s)
            for i in range(n):machine_rows.append(dict(machines=n,algorithm=arm,train_seed=s,eval_seed=int(r['eval_seed']),machine=i,jobs=jobs[i],failed_defer=defer[i],actionable_failed=failed[i],failed_ticks=total[i]))
    if n==2:
        for c in ('development','nominal','medium','high'):
            actual=exp.small_exact_metrics(model,configs[c],c,s,arm,oracles[c])
            stored=next(r for r in exact if r['cell']==c and r['train_seed']==s and r['algorithm']==arm)
            for k,v in actual.items():
                if isinstance(v,(int,float)):assert abs(v-stored[k])<1e-8
            assert actual['policy_cost']>=actual['oracle_cost']-1e-8
            exact_rechecked+=1
    # Exploratory posthoc training-configuration deployment checks. Same existing
    # exogenous panel, no fitting/tuning and no new confirmatory gate.
    for c in ('train_a','train_b'):
        for ev in cfg['evaluation_seeds']:
            posthoc_train.append(dict(machines=n,cell=c,algorithm=arm,train_seed=s,**exp.rollout(configs[c],ev,model)))
    if len(by_model)>0 and replayed%1700==0:print('Replay progress',replayed,flush=True)
assert replayed==15300 and exact_rechecked==120
for r in tqdm(refs,desc='Replay heuristic references',unit='episode'):
    actual=exp.rollout(exp.cells('full',int(r['machines']))[r['cell']],int(r['eval_seed']))
    assert all(abs(v-r[k])<1e-8 for k,v in actual.items())
csvout('nominal_machine_metrics.csv',machine_rows);csvout('representative_nominal_traces.csv',traces)
csvout('exploratory_training_config_episodes.csv',posthoc_train);csvout('exploratory_training_config_means.csv',group_summary(posthoc_train))
mg=defaultdict(list)
for r in machine_rows:mg[r['machines'],r['algorithm'],r['machine']].append(r)
machine_summary=[]
for (n,a,m),rr in sorted(mg.items()):machine_summary.append(dict(machines=n,algorithm=a,machine=m,mean_jobs=statistics.fmean(r['jobs'] for r in rr),
    zero_service_episode_fraction=statistics.fmean(r['jobs']==0 for r in rr),mean_failed_ticks=statistics.fmean(r['failed_ticks'] for r in rr),
    failed_actionable_defer_fraction=sum(r['failed_defer'] for r in rr)/max(1,sum(r['actionable_failed'] for r in rr))))
csvout('nominal_machine_summary.csv',machine_summary)
late=[]
for n in range(2,6):
    for a in exp.arms_for(n):
        rr=[r for r in logs if r['machines']==n and r['algorithm']==a and r['update']>=14000]
        late.append(dict(machines=n,algorithm=a,td_loss=statistics.fmean(r['td_loss'] for r in rr),ce=statistics.fmean(r['assignment_ce'] for r in rr)))
csvout('late_losses.csv',late)
checks=dict(status='ANALYZED',manifest='COMPLETED',source_snapshot_hash_verified=sourcehash,checkpoints_replayed=90,
    evaluation_episodes_replayed=replayed,heuristic_reference_episodes_replayed=680,small_exact_policy_metrics_recomputed=exact_rechecked,
    training_oracle_labels_checked=labels_checked,training_episode_rows_checked=sum(training_counts.values()),training_log_rows_checked=len(logs),
    budget_counts_and_seed_audit=True,primary_recomputed=recomputed['primary'],size_summary=size_results,
    prior_failed_run_excluded='ra_qmix_heuristic_assignment_full_cpu_20261004T124714Z (22 completed models)',
    limitations=['no retraining reproduction','same-source DP, not independent oracle','individual training transitions not persisted: cannot independently replay training reward reconciliation',
        'training-config replays and behavioral decomposition are exploratory; no gate changes'])
dump('analysis_checks.json',checks)
print(json.dumps(dict(primary=recomputed['primary'],sizes=size_results,replayed=replayed),indent=2),flush=True)
