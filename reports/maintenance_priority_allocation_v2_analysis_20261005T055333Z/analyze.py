import argparse,csv,hashlib,itertools,json,math,statistics,sys
from collections import Counter,defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);parser.add_argument('--run',required=True)
a=parser.parse_args();report=Path(a.report);run=Path(a.run)
sys.path.insert(0,str(report/'pinned_repository/src'))
from ht_pdm_fjsp import maintenance_dispatch as envmod
from ht_pdm_fjsp import maintenance_priority_allocation as exp
from ht_pdm_fjsp.maintenance_dispatch_policy import batch_observations, batch_sequences, MODES, RULE_MODE

torch.set_num_threads(1)
def rows(name):
    with (run/name).open() as f: yield from csv.DictReader(f)
def jsonread(name):return json.loads((run/name).read_text())
def save(name,data): (report/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def writecsv(name,data):
    if not data:return
    with (report/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def near(x,y):
    global max_error
    diff=abs(float(x)-float(y));max_error=max(max_error,diff)
    assert diff<1e-7,(x,y,diff)
def key(r):return (r['algorithm'],int(r['train_seed']),r['family'],int(r['eval_seed']))
max_error=0.;counts={};manifest=jsonread('manifest.json');old_summary=jsonread('summary.json');cfg=manifest['settings']
assert manifest['status']=='COMPLETED' and manifest['protocol']==exp.PROTOCOL
assert manifest['git_revision']=='cb1a655063843befeff0da0667eb6d190f2f5f07' and manifest['git_dirty'] is False
assert manifest['device']=='cpu' and not manifest['sealed_test_evaluated']
assert cfg==exp.settings('full') and all(manifest['audits'].values())
for filename in manifest['outputs']:assert (run/filename).is_file(),filename
for name,h in manifest['source_hashes'].items():
    path=report/'pinned_repository/src/ht_pdm_fjsp'/name
    if name.endswith('_plan.md'):path=report/'pinned_repository/docs'/name
    if name.endswith('_seed_registry.json'):path=report/'pinned_repository/configs'/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==h
    assert hashlib.sha256((run/'source_snapshot'/name).read_bytes()).hexdigest()==h
assert hashlib.sha256(json.dumps(manifest['source_hashes'],sort_keys=True).encode()).hexdigest()==manifest['source_bundle_sha256']
assert exp.seed_audit(cfg)==jsonread('seed_audit.json')
models={}
for path in sorted(run.glob('*/train_seed_*/model.pt')):
    model,payload=exp.load_checkpoint(path)
    assert payload['source_bundle_sha256']==manifest['source_bundle_sha256'] and payload['settings']==cfg
    models[payload['algorithm'],payload['train_seed']]=model
assert len(models)==30
expected_models={(mode,seed) for mode in MODES for seed in cfg['train_seeds']};assert set(models)==expected_models
counts['models']=len(models)
for path in run.glob('*/train_seed_*/training_coverage.json'):
    data=json.loads(path.read_text());assert data['env_steps']==120000 and data['episodes']==10000 and data['optimizer_steps']==4000 and data['rollout_updates']==250
    assert data['teacher_label_rows']==data['evaluation_config_training_transitions']==0
coverage=list(run.glob('*/train_seed_*/training_coverage.json'));assert len(coverage)==30
configs=jsonread('evaluation_configs.json')
for family in (*envmod.FAMILIES,'development'):
    seeds=cfg['development_seeds'] if family=='development' else cfg['evaluation_seeds']
    assert set(configs[family])=={str(s) for s in seeds}
    for seed in seeds:assert json.loads(json.dumps(asdict(envmod.family_config(family,seed,'full'))))==configs[family][str(seed)]
# Independently reconcile all training rows and common random populations.
training=defaultdict(list);population={};training_count=0
for r in tqdm(rows('training_episodes.csv'),desc='audit training records',total=300000):
    mode,seed=r['algorithm'],int(r['train_seed']);episode=int(r['episode'])
    assert (mode,seed) in expected_models
    training[mode,seed].append(episode)
    population_key=(seed,episode)
    record=(r['config_seed'],r['environment_seed'],r['config_sha256'],r['machines'],r['technicians'])
    if population_key in population:assert population[population_key]==record
    else:population[population_key]=record
    assert int(r['physical_steps'])==12*episode
    near(r['maintenance_cost'],float(r['preventive_jobs'])+2*float(r['corrective_jobs']))
    near(r['failure_cost'],15*float(r['failures']));near(r['unavailability_cost'],6*float(r['unavailable_ticks']))
    near(r['unavailable_ticks'],float(r['failed_waiting_ticks'])+float(r['service_ticks']))
    near(r['objective'],sum(float(r[f]) for f in ['maintenance_cost','unavailability_cost','failure_cost']))
    assert float(r['invalid_assignments'])==0
    training_count+=1
assert training_count==300000
assert all(rr==list(range(1,10001)) for rr in training.values()) and len(training)==30
counts['training_episodes']=training_count
progress=list(rows('training_progress.csv'));assert len(progress)==7500
progress_groups=defaultdict(list)
for r in progress:
    progress_groups[r['algorithm'],int(r['train_seed'])].append(r)
    assert all(math.isfinite(float(r[f])) for f in ['policy_loss','value_loss','entropy','loss','gradient_norm','mean_episode_cost','mean_matching_size','clipped_fraction'])
for modelkey,rr in progress_groups.items():
    assert [int(r['rollout_update']) for r in rr]==list(range(1,251))
    for r in rr:
        t=int(r['rollout_update']);assert int(r['physical_steps'])==t*480 and int(r['episodes'])==40*t and int(r['optimizer_steps'])==16*t
counts={k:v for k,v in counts.items() if v is not None};counts['training_progress']=7500
learning_curves=[]
for (mode,seed),rr in progress_groups.items():
    for block,start in [('first20',0),('last20',230)]:
        subset=rr[start:start+20]
        learning_curves.append(dict(algorithm=mode,train_seed=seed,block=block,mean_cost=statistics.fmean(float(r['mean_episode_cost']) for r in subset),mean_matching_size=statistics.fmean(float(r['mean_matching_size']) for r in subset),entropy=statistics.fmean(float(r['entropy']) for r in subset),value_loss=statistics.fmean(float(r['value_loss']) for r in subset)))
writecsv('learning_curves.csv',learning_curves)
episodes=list(rows('episodes.csv'));dev=list(rows('development_episodes.csv'));refs=list(rows('reference_episodes.csv'))
assert len(episodes)==10500 and len(dev)==600 and len(refs)==370
assert list(rows('episodes.partial.csv'))==episodes
assert list(rows('coordination.csv'))==episodes+dev
assert len({key(r) for r in episodes+dev+refs})==len(episodes+dev+refs)
all_results={key(r):r for r in episodes+dev+refs}
for r in all_results.values():
    near(r['objective'],float(r['maintenance_cost'])+float(r['unavailability_cost'])+float(r['failure_cost']))
    near(r['maintenance_cost'],float(r['preventive_jobs'])+2*float(r['corrective_jobs']))
    near(r['unavailability_cost'],6*float(r['unavailable_ticks']));near(r['failure_cost'],15*float(r['failures']))
    near(r['unavailable_ticks'],float(r['failed_waiting_ticks'])+float(r['service_ticks']))
    assert float(r['invalid_assignments'])==0
counts.update(test_episodes=len(episodes),development_episodes=len(dev),reference_episodes=len(refs))
parameters=list(rows('parameter_counts.csv'));assert len(parameters)==30 and len({r['parameters'] for r in parameters})==1
# Replay every physical test trace, aggregate machine/resource accounting, and
# batch-check each stored deployed policy/sequence on every test state.
machine_metrics={}
machine_count=0
for r in itertools.chain(rows('machine_metrics.csv'),rows('reference_machine_metrics.csv')):
    mk=(*key(r),int(r['machine']));assert mk not in machine_metrics
    machine_metrics[mk]=r;machine_count+=1
assert len(machine_metrics)==42100 # 40800 learned incl development + 1300 rule test.
counts['machine_rows']=len(machine_metrics)
replayed=steps=checked_policy_states=0;dispatch=Counter();behavior=Counter();witnesses=[]
policy_mismatches=[];max_policy_margin=0.0
def check_policy_batch(buffer,modelkey):
    global checked_policy_states,max_policy_margin
    if not buffer:return
    model=models[modelkey];observations=[r[0] for r in buffer]
    batch=batch_observations(observations);stored=batch_sequences([r[1] for r in buffer])
    with torch.no_grad():
        seq,_,_,_=model.decode(batch,modelkey[0],deterministic=True)
        replay=model.decode(batch,modelkey[0],stored)
        encoded=model.encode(batch)
        b,n,h=encoded[0].shape;k=encoded[2].shape[2]
        used_m=torch.zeros(b,n,dtype=torch.bool);used_t=torch.zeros(b,k,dtype=torch.bool)
        prefix=torch.zeros(b,h);count=torch.zeros(b);indices=torch.arange(b)
        maximum=torch.zeros(b)
        for stage in range(stored.shape[1]):
            requested=stored[:,stage,0];active=requested!=-2
            machine=torch.where(requested<0,torch.full_like(requested,n),requested)
            pair_active=active&(machine!=n)
            md,mask=model.distributions(encoded,batch,used_m,used_t,prefix,count,modelkey[0])
            margin=(md.logits.max(-1).values-md.logits[indices,machine])*active
            maximum=torch.maximum(maximum,margin)
            td=model.technician_distribution(encoded,batch,mask,machine,pair_active,prefix)
            technician=torch.where(pair_active,stored[:,stage,1],torch.zeros_like(machine))
            if modelkey[0]!='fixed_allocation':
                margin=(td.logits.max(-1).values-td.logits[indices,technician])*pair_active
                maximum=torch.maximum(maximum,margin)
            selected=machine.clamp_max(n-1)
            used_m[indices[pair_active],selected[pair_active]]=True
            used_t[indices[pair_active],technician[pair_active]]=True
            new=encoded[2][indices,selected,technician]
            prefix=torch.where(pair_active[:,None],(prefix*count[:,None]+new)/(count+1).clamp_min(1)[:,None],prefix)
            count=count+pair_active.float()
    assert torch.isfinite(replay[1]).all() and torch.isfinite(replay[2]).all()
    max_policy_margin=max(max_policy_margin,float(maximum.max()))
    assert float(maximum.max())<1e-5,('recorded action below greedy tolerance',modelkey,float(maximum.max()))
    for i,(emitted,record) in enumerate(zip(seq.tolist(),buffer)):
        local=[tuple(t) for t in emitted if t[0]!=-2]
        if local!=record[1]:
            policy_mismatches.append(dict(algorithm=modelkey[0],train_seed=modelkey[1],family=record[2][0],eval_seed=record[2][1],time=record[2][2],stored_sequence=record[1],local_sequence=local,max_recorded_choice_logit_margin=float(maximum[i])))
    checked_policy_states+=len(buffer)
for filename in ['decisions.csv','reference_decisions.csv']:
    iterator=itertools.groupby(rows(filename),key=key);buffers=defaultdict(list)
    for k,group in tqdm(iterator,desc='replay '+filename,total=10500 if filename=='decisions.csv' else 350):
        mode,seed,family,ev=k;config=envmod.family_config(family,ev,'full');env=envmod.DispatchEnv(config);obs=env.reset(ev)
        trace=list(group);assert len(trace)==config.horizon
        starts=[0]*config.machines;waits=[0]*config.machines;unavail=[0]*config.machines;maxwait=[0]*config.machines
        actionable=[0]*config.machines;deferred=[0]*config.machines;early=[0]*config.machines;earlydefer=[0]*config.machines
        edges=[[0]*config.technicians for _ in range(config.machines)];busy=[0]*config.technicians
        maxduration=max(max(row) for row in config.service_time)
        for r in trace:
            assert int(r['time'])==env.time
            before=env.state
            for field in ['ages','failed','pending_wait','remaining','assigned']:assert tuple(json.loads(r[field]))==getattr(before,field)
            pairs=tuple(tuple(p) for p in json.loads(r['pairs']));tokens=[tuple(p) for p in json.loads(r['sequence'])]
            assert pairs==tuple(t for t in tokens if t[0]>=0)
            if mode==RULE_MODE:assert pairs==exp.rule_action(obs)
            else:
                buffer=buffers[mode,seed];buffer.append((obs,tokens,(family,ev,env.time)))
                if len(buffer)==128:check_policy_batch(buffer,(mode,seed));buffer.clear()
            selected={m for m,j in pairs};occupied={m for m in before.assigned if m>=0}|selected
            technicians={j for m,j in pairs};mask=envmod.eligible_edges(config,before)
            near(r['unavailability_cost'],sum(before.failed[m] or m in occupied for m in range(config.machines))*config.unavailable_cost)
            near(r['maintenance_cost'],sum(config.corrective_cost if before.failed[m] else config.preventive_cost for m,j in pairs))
            behavior[mode,family,'stop_step']+=not pairs
            behavior[mode,family,'idle_possible_stop']+=not pairs and any(any(row) for row in mask)
            behavior[mode,family,'steps']+=1
            for m,j in pairs:
                starts[m]+=1;edges[m][j]+=1
                dispatch[mode,family,'CM' if before.failed[m] else 'PM',config.service_time[m][j]]+=1
            for j in range(config.technicians):busy[j]+=int(before.remaining[j]>0 or j in technicians)
            for m in range(config.machines):
                unavail[m]+=int(before.failed[m] or m in occupied);waits[m]+=int(before.failed[m] and m not in occupied);maxwait[m]=max(maxwait[m],before.pending_wait[m])
                if before.failed[m] and any(mask[m]):
                    actionable[m]+=1;deferred[m]+=int(m not in selected)
                    if config.horizon-env.time>maxduration:
                        early[m]+=1;earlydefer[m]+=int(m not in selected)
            obs,_,_,info=env.step(pairs)
            for field,value in info.items():near(r[field],value)
            steps+=1
        rr=all_results[k]
        for field,value in env.metrics.items():near(rr[field],value)
        assert sum(f and m not in env.state.assigned for m,f in enumerate(env.state.failed))==int(rr['failed_pending_final'])
        assert sum(r>0 for r in env.state.remaining)==int(rr['unfinished_services'])
        assert busy==json.loads(rr['technician_busy_ticks'])
        for m in range(config.machines):
            mr=machine_metrics[*k,m]
            expected=dict(service_starts=starts[m],failed_waiting_ticks=waits[m],unavailable_ticks=unavail[m],max_pending_wait=max(maxwait[m],env.state.pending_wait[m]),pending_wait_final=env.state.pending_wait[m],actionable_failed_decisions=actionable[m],failed_deferred_decisions=deferred[m],early_actionable_failed_decisions=early[m],early_failed_deferred_decisions=earlydefer[m])
            for field,value in expected.items():near(mr[field],value)
            assert (mr['failed_at_terminal']=='True')==env.state.failed[m]
            assert json.loads(mr['technician_service_starts'])==edges[m]
        replayed+=1
    for modelkey,buffer in buffers.items():check_policy_batch(buffer,modelkey)
counts.update(replayed_test_episodes=replayed,replayed_physical_steps=steps,checkpoint_policy_states=checked_policy_states)
assert replayed==10850 and steps==119350 and checked_policy_states==115500
development_policy_differences=[]
for r in tqdm(dev+[r for r in refs if r['family']=='development'],desc='replay development episodes'):
    mode,seed,family,ev=key(r);config=envmod.family_config(family,ev,'full')
    result,_,mm=exp.rollout(config,ev,mode,models.get((mode,seed)))
    for field,value in result.items():
        if isinstance(value,(int,float)):
            if abs(float(r[field])-value)>1e-7:development_policy_differences.append(dict(algorithm=mode,train_seed=seed,eval_seed=ev,field=field,stored=float(r[field]),local=value))
        else:
            if r[field]!=value:development_policy_differences.append(dict(algorithm=mode,train_seed=seed,eval_seed=ev,field=field,stored=r[field],local=value))
    if mode!=RULE_MODE:
        for mr in mm:
            observed=machine_metrics[*key(r),mr['machine']]
            for field,value in mr.items():
                pass # Test machine metrics fully audited from stored physical traces above.
counts['local_development_rollouts']=620
save('policy_tie_sensitivity.json',dict(max_greedy_logit_margin=max_policy_margin,sequence_mismatch_count=len(policy_mismatches),sequence_mismatches=policy_mismatches,development_differences=development_policy_differences))
writecsv('dispatch_counts.csv',[dict(algorithm=k[0],family=k[1],type=k[2],duration=k[3],starts=v) for k,v in sorted(dispatch.items())])
writecsv('behavior_counts.csv',[dict(algorithm=k[0],family=k[1],metric=k[2],count=v) for k,v in sorted(behavior.items())])
# Small DP and independently implemented policy evaluation / occupancy regrets.
small_cfg=envmod.family_config('small',0,'full');oracle=exp.SmallOracle(small_cfg)
assert len(oracle.nodes)==1475 and oracle.optimum==30.5
oracle_rows=list(rows('small_oracle_actions.csv'))
qtab={}
for r in oracle_rows:
    state=envmod.DispatchState(**{k:tuple(v) for k,v in json.loads(r['state']).items()})
    action=tuple(tuple(p) for p in json.loads(r['pairs']))
    qtab[int(r['time']),state,action]=float(r['cost_to_go'])
assert len(qtab)==sum(len(node['actions']) for node in oracle.nodes.values())
for k,node in oracle.nodes.items():
    for action,q in zip(node['actions'],node['q']):near(qtab[*k,action],q)
exact=[]
keys=list(oracle.nodes)
for modelkey in tqdm([('risk_skill_rule',-1),*sorted(models)],desc='exact deployed policies'):
    mode,seed=modelkey;chosen={}
    for begin in range(0,len(keys),128):
        chunk=keys[begin:begin+128];obs=[envmod.observation(small_cfg,state,t) for t,state in chunk]
        if mode==RULE_MODE:actions=[exp.rule_action(o) for o in obs]
        else:
            with torch.no_grad():seq,_,_,_=models[modelkey].decode(batch_observations(obs),mode,deterministic=True)
            actions=[tuple(tuple(p) for p in ss if p[0]>=0) for ss in seq.tolist()]
        for k,action in zip(chunk,actions):chosen[k]=oracle.nodes[k]['actions'].index(tuple(sorted(action)))
    values={}
    for k in sorted(keys,key=lambda k:k[0],reverse=True):
        t,state=k;node=oracle.nodes[k]
        values[k]=sum(p*(cost+(values[t+1,nextstate] if t+1<small_cfg.horizon else 0)) for p,nextstate,cost in node['edges'][chosen[k]])
    cost=statistics.fmean(values[0,state] for state in oracle.initials)
    occupancy={(0,state):1/len(oracle.initials) for state in oracle.initials};selection=allocation=0.
    for t in range(small_cfg.horizon):
        for k,weight in list(occupancy.items()):
            if k[0]!=t:continue
            node=oracle.nodes[k];index=chosen[k];action=node['actions'][index];machine_set={m for m,j in action}
            conditional=min(q for q,a in zip(node['q'],node['actions']) if {m for m,j in a}==machine_set)
            selection+=weight*(conditional-min(node['q']));allocation+=weight*(node['q'][index]-conditional)
            for p,nextstate,_ in node['edges'][index]:occupancy[t+1,nextstate]=occupancy.get((t+1,nextstate),0)+weight*p
    near(selection+allocation,cost-oracle.optimum)
    exact.append(dict(algorithm=mode,train_seed=seed,exact_policy_cost=cost,exact_optimum=oracle.optimum,optimality_gap=cost-oracle.optimum,exact_selection_regret=selection,exact_allocation_regret=allocation,maximum_relative_reduction=(cost-oracle.optimum)/cost,five_percent_possible_unrestricted=cost-oracle.optimum+1e-8>=.05*cost))
observed={(r['algorithm'],int(r['train_seed'])):r for r in rows('exact_headroom.csv')}
local_exact_differences=[]
for r in exact:
    old=observed[r['algorithm'],r['train_seed']]
    for field in ['exact_policy_cost','exact_optimum','optimality_gap','exact_selection_regret','exact_allocation_regret','maximum_relative_reduction']:
        if abs(r[field]-float(old[field]))>1e-7:local_exact_differences.append(dict(algorithm=r['algorithm'],train_seed=r['train_seed'],field=field,stored=float(old[field]),local=r[field]))
save('local_exact_differences.json',local_exact_differences)
writecsv('exact_recomputed.csv',exact);counts['exact_policy_rows']=len(exact);counts['exact_oracle_states']=len(keys);counts['exact_oracle_actions']=len(qtab)
# Independent paired statistics; compute t(df9) cutoffs by numeric integration.
def two_sided_p(t,df=9):
    begin=math.atan(abs(t)/math.sqrt(df));end=math.pi/2
    x=np.linspace(begin,end,4097);y=np.cos(x)**(df-1);step=(end-begin)/4096
    integral=step/3*(y[0]+y[-1]+4*y[1:-1:2].sum()+2*y[2:-1:2].sum())
    return float(2*math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(math.pi)*integral)
def critical(alpha):
    lo,hi=0.,20.
    for _ in range(55):
        mid=(lo+hi)/2
        if two_sided_p(mid)>alpha:lo=mid
        else:hi=mid
    return (lo+hi)/2
crit975=critical(.025);crit95=critical(.05)
assert abs(crit975-2.685010847)<1e-8 and abs(crit95-2.262157163)<1e-8
family_means={}
for mode in (*MODES,RULE_MODE):
    for seed in cfg['train_seeds']:
        for family in envmod.FAMILIES:
            panel=[float(r['objective']) for r in (refs if mode==RULE_MODE else episodes) if r['algorithm']==mode and r['family']==family and (mode==RULE_MODE or int(r['train_seed'])==seed)]
            assert len(panel)==50
            family_means[mode,seed,family]=statistics.fmean(panel)
def stats(arm,control,families,confirmatory=False):
    aa=[statistics.fmean(family_means[arm,s,f] for f in families) for s in cfg['train_seeds']]
    bb=[statistics.fmean(family_means[control,s,f] for f in families) for s in cfg['train_seeds']]
    delta=np.array(aa)-np.array(bb);mean=float(delta.mean());sd=float(delta.std(ddof=1));se=sd/math.sqrt(10)
    radius=(crit975 if confirmatory else crit95)*se
    p=two_sided_p(mean/se) if se else (1. if mean==0 else 0.)
    flips=np.array(list(itertools.product([-1,1],repeat=10)))
    permutation=float(np.mean(np.abs((flips*delta).mean(axis=1))>=abs(mean)-1e-10))
    nominal_arm=statistics.fmean(family_means[arm,s,f] for s in cfg['train_seeds'] for f in exp.NOMINAL)
    nominal_control=statistics.fmean(family_means[control,s,f] for s in cfg['train_seeds'] for f in exp.NOMINAL)
    wins=int((delta<-1e-8).sum());baseline=statistics.fmean(bb)
    return dict(algorithm=arm,control=control,families=list(families),mean_cost=statistics.fmean(aa),control_cost=baseline,mean_delta=mean,relative_reduction=-mean/baseline,ci=[mean-radius,mean+radius],ci_level=.975 if confirmatory else .95,wins=wins,ties=int((np.abs(delta)<=1e-8).sum()),losses=int((delta>1e-8).sum()),paired_t_p=p,bonferroni_p=min(1.,2*p) if confirmatory else None,sign_flip_p_exploratory=permutation,paired_dz=mean/sd if sd else None,nominal_relative_increase=nominal_arm/nominal_control-1,gate=dict(effect_at_least_5percent=mean<=-.05*baseline,wins_at_least_8=wins>=8,ci_upper_below_zero=mean+radius<0,nominal_within_10percent=nominal_arm<=1.1*nominal_control),seed_deltas=delta.tolist(),confirmatory=confirmatory)
primary=[stats('learned_both',control,exp.PRIMARY,True) for control in MODES[1:]]
for r in primary:
    old=old_summary['contrasts'][r['control']]
    for field,new in [('control_cost',r['control_cost']),('learned_cost',r['mean_cost']),('mean_delta',r['mean_delta']),('relative_reduction',r['relative_reduction']),('nominal_relative_increase',r['nominal_relative_increase'])]:near(old[field],new)
    # Published constants rounded at 10 digits; independent CI difference tolerance.
    assert max(abs(x-y) for x,y in zip(r['ci'],old['ci97_5_bonferroni']))<1e-8
    assert all(r['gate'].values())==old['passed']
reference=[stats(mode,RULE_MODE,exp.PRIMARY) for mode in MODES]
for r in reference:
    old=old_summary['reference_contrasts'][r['algorithm']]
    near(r['mean_delta'],old['mean_delta']);assert max(abs(x-y) for x,y in zip(r['ci'],old['ci95_descriptive']))<1e-8
secondary=[stats('learned_both',control,(family,)) for family in envmod.FAMILIES for control in (*MODES[1:],RULE_MODE)]
family_metrics=[]
for mode in (*MODES,RULE_MODE):
    for family in envmod.FAMILIES:
        rr=[r for r in (refs if mode==RULE_MODE else episodes) if r['algorithm']==mode and r['family']==family]
        metrics={f:statistics.fmean(float(r[f]) for r in rr) for f in ['objective','maintenance_cost','unavailability_cost','failure_cost','preventive_jobs','corrective_jobs','jobs','completed_jobs','failures','unavailable_ticks','failed_waiting_ticks','service_ticks','failed_pending_final','unfinished_services','actionable_failed_decisions','failed_deferred_decisions','early_actionable_failed_decisions','early_failed_deferred_decisions']}
        near(metrics['objective'],old_summary['family_mean_costs'][mode][family])
        mm=[r for k,r in machine_metrics.items() if k[0]==mode and k[2]==family]
        waits=np.array([float(r['max_pending_wait']) for r in mm])
        metrics.update(max_machine_wait=float(waits.max()),p95_machine_wait=float(np.quantile(waits,.95)),machines_failed_at_terminal=statistics.fmean(r['failed_at_terminal']=='True' for r in mm),machines_failed_unserved_at_terminal=statistics.fmean(r['failed_at_terminal']=='True' and float(r['pending_wait_final'])>0 and float(r['service_starts'])==0 for r in mm))
        metrics['early_failed_deferral_fraction']=metrics['early_failed_deferred_decisions']/metrics['early_actionable_failed_decisions'] if metrics['early_actionable_failed_decisions'] else None
        family_metrics.append(dict(algorithm=mode,family=family,episodes=len(rr),**metrics))
writecsv('family_metrics.csv',family_metrics)
writecsv('paired_seed_recomputed.csv',[dict(control=r['control'],train_seed=s,delta=d) for r in primary for s,d in zip(cfg['train_seeds'],r['seed_deltas'])])
statistics_output=dict(primary=primary,reference_descriptive=reference,secondary_descriptive=secondary,critical975=crit975,critical95=crit95,inference='paired training seed, conditional on common finite evaluation episode/configuration panel; no independent configuration population CI',normality_assumption='n=10; paired-t normality not established; exploratory exact sign-flip sensitivity assumes symmetric paired differences and does not replace locked gates')
save('statistics.json',statistics_output)
# Original recorded exact arithmetic and paired CSV, separate from local tie-dependent values.
recorded_exact=list(rows('exact_small_metrics.csv'));recorded_headroom={(r['algorithm'],r['train_seed']):r for r in rows('exact_headroom.csv')}
for r in recorded_exact:
    h=recorded_headroom[r['algorithm'],r['train_seed']]
    for field in ['exact_policy_cost','exact_optimum','optimality_gap','exact_selection_regret','exact_allocation_regret']:assert r[field]==h[field]
    cost=float(r['exact_policy_cost']);opt=float(r['exact_optimum']);gap=float(r['optimality_gap'])
    assert abs(cost-opt-gap)<1e-8 and abs(float(r['exact_selection_regret'])+float(r['exact_allocation_regret'])-gap)<1e-8
    assert abs(gap/cost-float(h['maximum_relative_reduction']))<1e-8
paired_csv=list(rows('paired_seed_metrics.csv'))
for r in primary:
    rr=[r0 for r0 in paired_csv if r0['control']==r['control']];assert len(rr)==10
    assert all(abs(float(r0['cost_delta'])-d)<1e-8 for r0,d in zip(rr,r['seed_deltas']))
elapsed=(datetime.fromisoformat(manifest['completed_at'])-datetime.fromisoformat(manifest['started_at'])).total_seconds()
audit=dict(status='ANALYZED',run=str(run),commit=manifest['git_revision'],raw_unmodified=True,source_bundle_sha256=manifest['source_bundle_sha256'],counts=counts,max_numeric_error=max_error,engineering_audits=len(manifest['audits']),all_engineering_gates_passed=True,seed_overlap=False,sealed_panels_closed=True,device=manifest['device'],elapsed_seconds=elapsed,greedy_sequence_tie_mismatches=len(policy_mismatches),maximum_recorded_choice_logit_margin=max_policy_margin,local_exact_difference_count=len(local_exact_differences),local_development_difference_count=len(development_policy_differences),scope='All physical test traces replayed exactly; recorded checkpoint choices verified within 1e-5 logit tolerance with CPU/runtime tie differences reported; local exact values recomputed using shared dynamics; independent paired statistics, not separate simulator or training reproduction',recorded_exact_arithmetic_verified=True,paired_seed_csv_verified=True,scientific_primary_passed=all(all(r['gate'].values()) for r in primary))
save('audit.json',audit)
print(json.dumps({'audit':audit,'primary':primary,'reference':reference,'exact_means':{mode:{f:statistics.fmean(r[f] for r in exact if r['algorithm']==mode) for f in ['exact_policy_cost','optimality_gap','exact_selection_regret','exact_allocation_regret']} for mode in (*MODES,RULE_MODE)}},indent=2))
