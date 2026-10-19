"""Read-only validation and analysis of the transferred, frozen joint-RL run."""
from collections import defaultdict, Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'lab_results/maintenance_joint_rl_v1_full_cpu_20261011T045434Z').exists())
RUN=ROOT/'lab_results/maintenance_joint_rl_v1_full_cpu_20261011T045434Z'
SNAP=RUN/'source_snapshot'
OUT=ROOT/'reports'/('maintenance_joint_rl_v1_analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
OUT.mkdir()
import ht_pdm_fjsp
ht_pdm_fjsp.__path__.insert(0,str(SNAP))
from ht_pdm_fjsp import maintenance_joint_rl as e
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, transition
from ht_pdm_fjsp.maintenance_solver_diagnostics import Requests
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
torch.set_num_threads(1)

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def eq(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a),set(b))
        for k in a:eq(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):eq(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert np.isclose(a,b,atol=1e-10,rtol=1e-10),(a,b)
    else:assert a==b,(a,b)

m=json.loads((RUN/'manifest.json').read_text());cfg=m['settings']
assert m['status']=='COMPLETED' and m['audit']=='PASS' and not m['dirty']
assert m['commit']=='ed032858ebf174738e9c00bfe2dce72b7fff4277'
assert cfg==e.settings('full') and m['actual_counts']==m['expected_counts']==e.counts(cfg)
for filename,meta in m['artifacts'].items():
    p=RUN/filename
    assert p.stat().st_size==meta['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
worktree=Path('/Users/bachng/.codex/worktrees/maintenance-joint-rl-v1/marl')
for p in SNAP.glob('*.py'):
    data=subprocess.check_output(['git','show',m['commit']+':src/ht_pdm_fjsp/'+p.name],cwd=worktree)
    assert data==p.read_bytes()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:assert m['thread_environment'][k]=='1'
audit=dict(status='PASS',file_hashes=len(m['artifacts']),source_snapshot_matches_commit=True,
    bytes=sum(v['bytes'] for v in m['artifacts'].values()),expected_counts=e.counts(cfg))

df=pd.read_csv(RUN/'episodes.csv');dev=pd.read_csv(RUN/'development_episodes.csv')
train=pd.read_csv(RUN/'training_episodes.csv');progress=pd.read_csv(RUN/'training_progress.csv')
dec=pd.read_csv(RUN/'decisions.csv');coord=pd.read_csv(RUN/'coordination.csv')
req=pd.read_csv(RUN/'requests.csv');carry=pd.read_csv(RUN/'carryover.csv')
times=pd.read_csv(RUN/'time_profiles.csv')
keys=['condition','cohort','shock','controller','train_seed'];request_keys=keys+['scope','machine','request']
index=df.set_index(keys);assert index.index.is_unique
expected=set(itertools.product(cfg['conditions'],cfg['test_cohorts'],cfg['test_shocks']))
expected={(*cell,name,seed) for cell in expected for name,seed in [('best_rule',-1),*itertools.product(cfg['arms'],cfg['train_seeds'])]}
assert set(index.index)==expected and len(df)==21504
assert df.equals(pd.read_csv(RUN/'episodes.partial.csv'))
assert len(train)==102400 and len(dev)==6848 and len(dec)==10080 and len(progress)==1600
assert len(coord)==len(df) and set(coord.set_index(keys).index)==expected
assert (coord.invalid_actions==0).all() and (coord.scalar_batch_replay=='PASS').all() and (coord.fixed_action_audit=='PASS').all()
assert (coord.base_decisions==48).all() and (coord.tail_decisions==12).all()
for prefix in ['base','tail','extended']:
    component=sum(df[f'{prefix}_{k}'] for k in ['maintenance_cost','unavailability_cost','failure_cost','waiting_cost'])
    np.testing.assert_allclose(component,df[f'{prefix}_objective'],atol=1e-10)
    np.testing.assert_allclose(df[f'{prefix}_waiting_cost'],12*df[f'{prefix}_overdue_waiting_ticks'])
    np.testing.assert_allclose(df[f'{prefix}_unavailability_cost'],6*df[f'{prefix}_unavailable_ticks'])
    np.testing.assert_allclose(df[f'{prefix}_failure_cost'],15*df[f'{prefix}_failures'])
for column in df:
    if column.startswith('base_') and 'tail_'+column[5:] in df:
        np.testing.assert_allclose(df[column]+df['tail_'+column[5:]],df['extended_'+column[5:]])
np.testing.assert_allclose(df.extended_cost_per_machine_tick,df.extended_objective/240)
assert (df[df.controller=='fixed_ppo'].nonfixed_assignments==0).all()
assert not req.duplicated(request_keys).any()
assert (req.wait==req.end-req.arrival).all() and (req.overdue==(req.wait>4)).all()
assert (req.served!=req.censored).all() and (req.wait>=0).all()
request_groups=req.groupby(keys+['scope'])
for scope,prefix in [('cutoff','cutoff'),('extended','extended')]:
    sub=req[req.scope==scope].groupby(keys)
    for metric,values in [('requests',sub.size()),('served_requests',sub.served.sum()),
                         ('censored_requests',sub.censored.sum()),('overdue_requests',sub.overdue.sum()),('observed_wait',sub.wait.sum())]:
        np.testing.assert_allclose(values.reindex(index.index,fill_value=0),index[f'{prefix}_{metric}'])
    served=req[(req.scope==scope)&req.served].groupby(keys).wait
    for metric,values in [('served_wait_mean',served.mean()),('served_wait_p95',served.agg(lambda x:np.quantile(x,.95,method='inverted_cdf')))]:
        np.testing.assert_allclose(values.reindex(index.index),index[f'{prefix}_{metric}'],equal_nan=True)
np.testing.assert_allclose(carry.groupby(keys).size().reindex(index.index,fill_value=0),index.carryover_requests)
np.testing.assert_allclose(carry.groupby(keys).started_in_tail.sum().reindex(index.index,fill_value=0),index.carryover_started_in_tail)
cut=req[(req.scope=='cutoff')&req.censored].set_index(keys+['machine','request'])
ext=req[req.scope=='extended'].set_index(keys+['machine','request'])
cc=carry.set_index(keys+['machine','request']);assert set(cc.index)==set(cut.index)
np.testing.assert_array_equal(cc.arrival,cut.loc[cc.index].arrival)
np.testing.assert_array_equal(cc.wait_at_cutoff,cut.loc[cc.index].wait)
np.testing.assert_array_equal(cc.started_in_tail,ext.loc[cc.index].served)
np.testing.assert_array_equal(cc.wait_at_start_or_end,ext.loc[cc.index].wait)
np.testing.assert_array_equal(cc.censored_after_tail,ext.loc[cc.index].censored)
assert len(times)==2*21*60 and not times.duplicated(['condition','controller','train_seed','time']).any()
assert (times.episodes==512).all()
for metric in ['objective','maintenance_cost','unavailability_cost','failure_cost','waiting_cost']:
    np.testing.assert_allclose(times.groupby(['condition','controller','train_seed'])[metric].sum().sort_index(),
        df.groupby(['condition','controller','train_seed'])['extended_'+metric].mean().sort_index())

exclusions=json.loads((SNAP/'maintenance_joint_rl_v1_profile_exclusions.json').read_text())
e.exclusions=lambda:exclusions
panels,registry=e.build_panels(cfg)
resolved=json.loads((RUN/'resolved_config.json').read_text())
eq(json.loads(json.dumps(registry)),resolved['physical_profiles'])
assert len({e.boundary.signature(c) for c in panels.values()})==144
own=json.loads((SNAP/'maintenance_joint_rl_v1_seed_registry.json').read_text())['panels']
historical=e.integers(exclusions['historical_raw_panels'])
for p in SNAP.glob('*seed_registry.json'):
    if p.name!='maintenance_joint_rl_v1_seed_registry.json':historical|=e.integers(json.loads(p.read_text()))
for row in own.values():
    groups=[set(v) if isinstance(v,list) else {v} for v in row.values()]
    assert not any(a&b for a,b in itertools.combinations(groups,2))
    assert not set.union(*groups)&historical
assert not e.integers(own['full'])&e.integers(own['smoke'])
sampler=e.training_sampler(cfg,panels,historical|e.integers(own))
for seed in tqdm(cfg['train_seeds'],desc='Regenerate paired training profiles'):
    left=train[(train.train_seed==seed)&(train.algorithm=='joint_ppo')].reset_index(drop=True)
    right=train[(train.train_seed==seed)&(train.algorithm=='fixed_ppo')].reset_index(drop=True)
    for name in ['episode','rollout_update','config_seed','environment_seed','generator_attempt','condition','config_sha256','learned_steps','physical_steps']:
        assert left[name].equals(right[name]),name
    assert len(left)==5120 and left.episode.tolist()==list(range(1,5121))
    for r in left.itertuples():
        c,condition,cs,es,attempt=sampler(seed,r.episode-1)
        assert (condition,cs,es,attempt)==(r.condition,r.config_seed,r.environment_seed,r.generator_attempt)
        assert hashlib.sha256(json.dumps(asdict(c),sort_keys=True).encode()).hexdigest()==r.config_sha256
        assert r.rollout_update==(r.episode-1)//64+1
        assert r.learned_steps==((r.episode-1)//32+1)*32*48
        assert r.physical_steps==((r.episode-1)//32+1)*32*60
coverage=json.loads((RUN/'training_coverage.json').read_text())
selection=json.loads((RUN/'checkpoint_selection.json').read_text())
assert selection['locked_before_test']
models={};learning=[]
for r in tqdm(coverage,desc='Verify checkpoints and dev-only selection'):
    name,seed=r['algorithm'],r['train_seed']
    assert r['learned_steps']==245760 and r['physical_steps']==307200 and r['episodes']==5120 and r['optimizer_steps']==3840
    assert r['mixture_counts']=={'dense':2560,'skill_mask':2560} and r['parameter_count']==19650
    pp=progress[(progress.algorithm==name)&(progress.train_seed==seed)]
    assert pp.rollout_update.tolist()==list(range(1,81)) and np.isfinite(pp.select_dtypes('number').values).all()
    np.testing.assert_array_equal(pp.learned_steps,pp.rollout_update*3072)
    np.testing.assert_array_equal(pp.physical_steps,pp.rollout_update*3840)
    np.testing.assert_array_equal(pp.optimizer_steps,pp.rollout_update*48)
    initial,_=e.load_checkpoint(RUN/name/f'seed_{seed}/checkpoints/update_0000.pt')
    actual_initial=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in initial.state_dict().values())).hexdigest()
    assert actual_initial==r['initial_parameters_sha256']
    chosen=selection['selections'][f'{name}:{seed}']
    assert chosen['update']==e.choose_checkpoint(chosen['candidates'])['update']>0
    for candidate in chosen['candidates']:
        sub=dev[(dev.controller==name)&(dev.train_seed==seed)&(dev.checkpoint_update==candidate['update'])]
        assert len(sub)==64 and not sub.duplicated(['condition','cohort','shock']).any()
        assert np.isclose(sub.extended_objective.mean(),candidate['mean_extended_cost'])
        _,payload=e.load_checkpoint(RUN/candidate['checkpoint'])
        assert payload['learned_steps']==candidate['update']*3072
    model,payload=e.load_checkpoint(RUN/chosen['selected_checkpoint'])
    assert (RUN/chosen['selected_checkpoint']).read_bytes()==(RUN/chosen['checkpoint']).read_bytes()
    assert any(not torch.equal(v,initial.state_dict()[k]) for k,v in model.state_dict().items())
    assert (df[(df.controller==name)&(df.train_seed==seed)].checkpoint_update==chosen['update']).all()
    models[name,seed]=model
    first=next(x['mean_extended_cost'] for x in chosen['candidates'] if x['update']==0)
    last=next(x['mean_extended_cost'] for x in chosen['candidates'] if x['update']==80)
    learning.append(dict(algorithm=name,seed=seed,initial_dev=first,last_dev=last,selected_dev=chosen['mean_extended_cost'],
        selected_update=chosen['update'],selected_gain_percent=100*(first-chosen['mean_extended_cost'])/first))
for seed in cfg['train_seeds']:
    pair=[r for r in coverage if r['train_seed']==seed]
    assert len({r['initial_parameters_sha256'] for r in pair})==1
for condition in cfg['conditions']:
    mean=dev[(dev.condition==condition)&(dev.train_seed==-1)].groupby('controller').extended_objective.mean()
    assert selection['selected_rules'][condition]==min(cfg['rules'],key=lambda x:mean[x])

replay_requests=[];replay_carry=[];observed=defaultdict(Counter)
diagnostic_max_difference=defaultdict(float)
logged_groups=dec.groupby(keys,sort=False)
assert len(logged_groups)==168
for identity,rows in tqdm(logged_groups,desc='Replay sealed audit trajectories',total=168):
    condition,cohort,shock,name,seed=identity;result=index.loc[identity]
    c=replace(panels['test',cohort,condition],horizon=60);env=WaitingEnv(c)
    es=role_seed(shock,f'{e.VERSION}:physical:{cohort}');env.reset(es)
    base=Requests(env.state);extended=Requests(env.state);totals=defaultdict(float);inside=defaultdict(float)
    assert rows.time.tolist()==list(range(60))
    for row in rows.itertuples():
        state=DispatchState(**{k:tuple(v) for k,v in json.loads(row.state).items()})
        pairs=tuple(map(tuple,json.loads(row.action)));assert state==env.state and row.env_seed==es
        if row.time<48 and name!='best_rule':
            action,diag=models[name,seed].select(c,state,48-row.time,48)
            assert action==pairs
            logged_diag=json.loads(row.controller_diagnostics)
            assert diag['action_index']==logged_diag['action_index']
            for field in ['log_probability','value']:
                diagnostic_max_difference[field]=max(diagnostic_max_difference[field],abs(diag[field]-logged_diag[field]))
                assert np.isclose(diag[field],logged_diag[field],atol=5e-5,rtol=1e-5)
        else:
            rule='deadline_matching' if row.time>=48 else selection['selected_rules'][condition]
            assert pairs==e.prior.rule_action(c,state,60-row.time if row.time>=48 else 48-row.time,rule)
        following,cost,info=transition(c,state,pairs,env.events[row.time]);env.step(pairs)
        assert following==env.state
        eq(json.loads(json.dumps(asdict(following))),json.loads(row.following));eq(info,json.loads(row.physical_metrics))
        totals['base' if row.time<48 else 'tail']+=cost
        extended.step(row.time,state,pairs,following)
        if row.time<48:
            base.step(row.time,state,pairs,following)
            feasible=e.feasible(c,state);fixed=e.fixed_candidates(c,state)
            counter=observed[condition,name]
            counter['ticks']+=1;counter['choice_ticks']+=len(feasible)>1
            counter['wait_with_choice']+=not pairs and len(feasible)>1
            counter['assignment_choices']+=sum(tuple(m for m,j in a)==tuple(m for m,j in pairs) for a in feasible)>1
            counter['nonfixed']+=pairs not in fixed
            counter['starts']+=bool(pairs)
            if name=='fixed_ppo':assert pairs in fixed
        if 12<=row.time<36:
            for k,v in info.items():inside[k]+=v
            for k,v in e.boundary.snapshot(state).items():inside[k]+=v
        if row.time==47:
            cut=[{**r,'overdue':r['wait']>4} for r in base.finish(48)]
            cutoff=e.boundary.snapshot(following)
    end=[{**r,'overdue':r['wait']>4} for r in extended.finish(60)]
    for prefix,total in totals.items():assert result[prefix+'_objective']==total
    assert result.extended_objective==sum(totals.values())
    for k,v in inside.items():
        if k=='objective':assert np.isclose(result.interior_cost_per_machine_tick,v/96)
        elif k in ['pending','failed']:assert np.isclose(result['interior_'+k+'_mean'],v/24)
    for k,v in cutoff.items():assert result['cutoff_'+k]==v
    for k,v in e.boundary.snapshot(env.state).items():assert result['post_tail_'+k]==v
    for scope,records in [('cutoff',cut),('extended',end)]:
        for r in records:replay_requests.append(dict(zip(keys,identity))|dict(scope=scope)|r)
    later={(r['machine'],r['request']):r for r in end}
    for r in cut:
        if r['censored']:
            final=later[r['machine'],r['request']]
            replay_carry.append(dict(zip(keys,identity))|dict(machine=r['machine'],request=r['request'],arrival=r['arrival'],
                wait_at_cutoff=r['wait'],started_in_tail=final['served'],wait_at_start_or_end=final['wait'],censored_after_tail=final['censored']))
rr=pd.DataFrame(replay_requests);rc=pd.DataFrame(replay_carry)
logged_ids=set(logged_groups.groups)
for generated,actual,k in [(rr,req,request_keys),(rc,carry,keys+['machine','request'])]:
    subset=actual[actual[keys].apply(tuple,axis=1).isin(logged_ids)]
    cols=list(generated.columns)
    pd.testing.assert_frame_equal(generated[cols].sort_values(k).reset_index(drop=True),subset[cols].sort_values(k).reset_index(drop=True),check_dtype=False)
audit.update(independent_replay_decisions=10080,independent_replay_episodes=168,episode_reconciliation=21504,
    request_rows_reconciled=len(req),carryover_rows_reconciled=len(carry),regenerated_training_profiles=51200,
    checkpoint_files_verified=120,paired_training_tapes='PASS',development_selection='PASS',physical_profile_splits='PASS')
audit['diagnostic_float_max_difference']=dict(diagnostic_max_difference)
audit['diagnostic_float_tolerance']='atol5e-5/rtol1e-5; Mac single-item inference vs Linux batched inference'
save('integrity_audit.json',audit)

summary=json.loads((RUN/'summary.json').read_text())
grouped=df.groupby(['condition','controller','train_seed','cohort']).mean(numeric_only=True)
def matrix(condition,name,metric):
    if name=='best_rule':
        row=grouped.loc[condition,name,-1].reindex(cfg['test_cohorts'])[metric].values
        return np.tile(row,(10,1))
    return np.array([grouped.loc[condition,name,seed].reindex(cfg['test_cohorts'])[metric].values for seed in cfg['train_seeds']])
rng=np.random.default_rng(role_seed(cfg['analysis_seed'],f'{e.VERSION}:crossed_bootstrap'))
rdraw=rng.integers(10,size=(20000,10));pdraw=rng.integers(64,size=(20000,64))
def independent_samples(left,right,relative):
    output=np.zeros(20000)
    for start in range(0,20000,500):
        ids=(rdraw[start:start+500,:,None],pdraw[start:start+500,None,:])
        delta=(left[ids]-right[ids]).mean(axis=2).mean(axis=1)
        output[start:start+500]=delta/left[ids].mean(axis=2).mean(axis=1) if relative else delta
    return output
for condition in cfg['conditions']:
    for left in ['best_rule','fixed_ppo']:
        a,b=matrix(condition,left,'extended_cost_per_machine_tick'),matrix(condition,'joint_ppo','extended_cost_per_machine_tick')
        record=summary['primary_contrasts'][f'{condition}:joint_vs_{left}']
        ci=np.quantile(independent_samples(a,b,True),[.00625,.99375])
        np.testing.assert_allclose(record['confidence_interval'],ci,atol=1e-13)
        assert np.isclose(record['estimate'],(a-b).mean()/a.mean())
    for metric in ['interior_pending_mean','interior_failed_mean','interior_overdue_per_tick']:
        a,b=matrix(condition,'joint_ppo',metric),matrix(condition,'best_rule',metric)
        record=summary['exploratory_service'][f'{condition}:{metric}']
        np.testing.assert_allclose(record['confidence_interval'],np.quantile(independent_samples(a,b,False),[.025,.975]),atol=1e-13)
audit['recomputed_intervals']=10;save('integrity_audit.json',audit)
for condition in cfg['conditions']:
    for name,metrics in summary['means'][condition].items():
        sub=df[(df.condition==condition)&(df.controller==name)]
        for metric,value in metrics.items():assert np.isclose(value,sub[metric].mean())

learn=pd.DataFrame(learning);learn.to_csv(OUT/'learning_by_seed.csv',index=False)
devcurve=dev[dev.controller.isin(cfg['arms'])].groupby(['condition','controller','checkpoint_update']).extended_objective.mean().reset_index()
devcurve.to_csv(OUT/'development_learning_curves.csv',index=False)
traincurve=progress.groupby(['algorithm','rollout_update'])[['mean_extended_episode_cost','entropy','policy_loss','value_loss','gradient_norm','clipped_fraction']].mean().reset_index()
traincurve.to_csv(OUT/'training_learning_curves.csv',index=False)
fullmeans=df.groupby(['condition','controller']).mean(numeric_only=True)
fullmeans.to_csv(OUT/'all_mean_metrics.csv')
seedrows=[]
for condition in cfg['conditions']:
    for seed in cfg['train_seeds']:
        a=matrix(condition,'best_rule','extended_cost_per_machine_tick')[0].mean()
        j=matrix(condition,'joint_ppo','extended_cost_per_machine_tick')[cfg['train_seeds'].index(seed)].mean()
        f=matrix(condition,'fixed_ppo','extended_cost_per_machine_tick')[cfg['train_seeds'].index(seed)].mean()
        seedrows.append(dict(condition=condition,seed=seed,rule_cost=a,joint_cost=j,fixed_cost=f,
            joint_vs_rule_percent=100*(a-j)/a,joint_vs_fixed_percent=100*(f-j)/f))
pd.DataFrame(seedrows).to_csv(OUT/'test_by_seed.csv',index=False)
components=[];behaviors=[]
for condition in cfg['conditions']:
    for metric in ['maintenance_cost','unavailability_cost','failure_cost','waiting_cost','objective']:
        rule=fullmeans.loc[condition,'best_rule']['extended_'+metric]
        joint=fullmeans.loc[condition,'joint_ppo']['extended_'+metric]
        components.append(dict(condition=condition,component=metric,rule=rule,joint=joint,difference=joint-rule))
    for name in ['best_rule','joint_ppo','fixed_ppo']:
        row=fullmeans.loc[condition,name]
        behaviors.append(dict(condition=condition,controller=name,
            wait_with_choice_per_episode=row.wait_with_choice,wait_per_all_base_ticks=row.wait_with_choice/48,
            same_subset_assignment_choice_ticks=row.assignment_alternatives,
            nonfixed_assignments_per_episode=row.nonfixed_assignments,
            nonfixed_given_selected_subset_has_choices=row.nonfixed_assignments/row.assignment_alternatives,
            preventive_starts=row.base_preventive_jobs,corrective_starts=row.base_corrective_jobs))
pd.DataFrame(components).to_csv(OUT/'cost_decomposition.csv',index=False)
pd.DataFrame(behaviors).to_csv(OUT/'behavior_full_panel.csv',index=False)
save('behavior_audit_subpanel.json',{f'{c}:{name}':dict(counts) for (c,name),counts in observed.items()})
elapsed=(datetime.fromisoformat(m['finished'])-datetime.fromisoformat(m['started'])).total_seconds()
analysis=dict(run=RUN.name,commit=m['commit'],report=str(OUT),elapsed_seconds=elapsed,
    primary=summary['primary_contrasts'],service=summary['exploratory_service'],selected_rules=selection['selected_rules'],
    development_initial_to_final={name:dict(initial=learn[learn.algorithm==name].initial_dev.mean(),
        final=learn[learn.algorithm==name].last_dev.mean(),selected=learn[learn.algorithm==name].selected_dev.mean(),
        improved_seeds=int((learn[learn.algorithm==name].selected_gain_percent>0).sum())) for name in cfg['arms']},
    selection_updates={name:dict(Counter(int(x) for x in learn[learn.algorithm==name].selected_update)) for name in cfg['arms']},
    components=components,behaviors=behaviors,
    best_seed_not_selected_from_test=True,training_elapsed_seconds=sum(r['elapsed_seconds'] for r in coverage),
    scope='ANALYZED; statistical intervals and locked replay subpanel VERIFIED; no complete training rerun or full test rerun')
save('analysis_numbers.json',analysis)
print(json.dumps(analysis,indent=2))
