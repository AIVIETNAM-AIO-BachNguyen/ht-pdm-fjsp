"""Audit immutable returned artifacts; recompute inference and descriptive strata."""
from dataclasses import asdict
from datetime import datetime,timezone
from collections import defaultdict
import hashlib,itertools,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'lab_results/maintenance_policy_continuation_v1_full_cpu_20261011T081315Z'
WT=Path('/Users/bachng/.codex/worktrees/maintenance-joint-rl-v1/marl')
OUT=ROOT/'reports'/('maintenance_policy_continuation_v1_analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
OUT.mkdir()
import ht_pdm_fjsp
ht_pdm_fjsp.__path__.insert(0,str(RUN/'source_snapshot'))
from ht_pdm_fjsp import maintenance_policy_continuation as e
from ht_pdm_fjsp.maintenance_dispatch import role_seed,validate_matching
from ht_pdm_fjsp.maintenance_waiting import transition
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_policy_forks import ARMS,LABELS,COMPONENTS,root_actions
for module in (e,e.joint,e.previous):module.ROOT=WT
torch.set_num_threads(1)

def save(name,value):
 (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def same(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
def identity(row):return tuple(row[k] for k in e.KEYS)

m=json.loads((RUN/'manifest.json').read_text());cfg=m['settings'];locked=json.loads((RUN/'source_contract.json').read_text())
assert m['status']=='COMPLETED' and m['audit']=='PASS' and not m['dirty'] and m['commit']=='3fd164ce912f09ed8ddfea7b099ac99ff6abddc7'
assert cfg==e.settings('full') and m['expected_counts']==e.counts(cfg)
for name,rec in m['artifacts'].items():
 p=(RUN/name).resolve();assert p.is_relative_to(RUN.resolve())
 assert p.stat().st_size==rec['bytes'] and e.sha(p)==rec['sha256']
for p in (RUN/'source_snapshot').iterdir():
 relative='src/ht_pdm_fjsp/'+p.name if p.suffix=='.py' else p.name if p.name in ('pyproject.toml','uv.lock') else 'configs/'+p.name
 assert subprocess.check_output(['git','show',m['commit']+':'+relative],cwd=WT)==p.read_bytes(),relative
assert subprocess.check_output(['git','show',m['commit']+':docs/maintenance_policy_continuation_v1_plan.md'],cwd=WT)==(RUN/'plan.md').read_bytes()
assert locked==e.contract();source=ROOT/'lab_results'/locked['state_source_run']
models,panels,source_rows,source_audit=e.verify_source(source,locked)
assert e.seed_audit(cfg)==m['seed_audit'] and source_audit==m['source_audit']
for seed in cfg['model_seeds']:
 p=RUN/'frozen_models'/f'seed_{seed}.pt'
 assert e.sha(p)==locked['joint_source_contract']['models'][str(seed)]['sha256']
 model,_=e.joint.load_checkpoint(p)
 assert e.previous.parameters_sha(model)==e.previous.parameters_sha(models[seed])==m['frozen_parameter_hashes'][str(seed)]
 models[seed]=model

rows=pd.read_csv(RUN/'states.csv').to_dict('records');d=pd.DataFrame(rows)
assert d.equals(pd.read_csv(RUN/'states.partial.csv'))
expected=set(itertools.product(cfg['conditions'],cfg['cohorts'],cfg['physical_shocks'],cfg['model_seeds'],cfg['origins'],cfg['landmarks']))
index={identity(r):i for i,r in enumerate(rows)};assert len(index)==len(rows)==2560 and set(index)==expected
source_index={identity(r):r for r in source_rows}
roots=pd.read_csv(RUN/'root_values.csv');root_groups={k:g.to_dict('records') for k,g in roots.groupby(list(e.KEYS))}
audits=pd.read_csv(RUN/'rollout_audits.csv');audit_index={identity(r):r for r in audits.to_dict('records')}
assert len(audit_index)==len(audits)==2560 and set(audit_index)==expected
assert len(roots)==8834 and not roots.duplicated(list(e.KEYS)+['continuation','action_index']).any()
with np.load(RUN/'raw_forecast_costs.npz') as archive:raw={k:archive[k] for k in archive.files}
with np.load(RUN/'raw_boundary_states.npz') as archive:bound={k:archive[k] for k in archive.files}
assert set(raw)=={'costs','base','tail'}
for key in raw:assert raw[key].shape==(2560,2,4,32) and np.isfinite(raw[key]).all()
np.testing.assert_array_equal(raw['costs'],raw['base']+raw['tail'])
for stage in ('cutoff','terminal'):
 for key in ('ages','failed','pending_wait','remaining','assigned'):
  a=bound[stage+'_'+key];assert a.shape==(2560,2,4,32,2 if key in ('remaining','assigned') else 4)
  if key=='remaining':assert (a>=0).all()
assert (audits.feasibility=='PASS').all() and (audits.components=='PASS').all()
assert (audits.training_clock=='H48_L12_PASS').all() and (audits.scalar_replay=='NOT_SCHEDULED').all()
augmented=[];roots_count=0;transitions=0;rl_rows=0;tail_rows=0;hist=np.zeros(21,int)
for i,row in enumerate(tqdm(rows,desc='Audit all saved states, roots and raw cost arrays')):
 ident=identity(row);old=source_index[ident];c=panels[row['cohort'],row['condition']];s=e.state(row['state'])
 for key in ('state','rl_action','rule_action'):assert row[key]==old[key]
 assert row['source_physical_seed']==old['physical_seed'] and row['forecast_seed']==e.forecast_seed(cfg,row)
 rl=e.action(row['rl_action']);ref=e.action(row['rule_action']);actions,labels=root_actions(s,rl,ref)
 assert row['pm_exposure']==int(labels[0]!=labels[2]) and row['now_exposure']==int(bool(rl))
 assert row['structural_choice']==int(len(e.joint.feasible(c,s))>1)
 assert row['unique_roots']==len(actions) and row['remaining_physical_ticks']==60-row['tick'] and row['learned_intervals']==48-row['tick'] and row['tail_intervals']==12
 roots_count+=len(actions);nt=2*len(actions)*32*(60-row['tick']);nr=len(actions)*32*(47-row['tick']);n_tail=2*len(actions)*32*12
 assert row['forecast_transitions']==nt;transitions+=nt;rl_rows+=nr;tail_rows+=n_tail
 audit=audit_index[ident];assert audit['transitions']==nt and audit['batched_decision_rows']==audit['rl_continuation_transition_rows']==nr and audit['tail_transition_rows']==n_tail
 counts=np.asarray(json.loads(audit['rl_action_histogram']));assert counts.shape==(21,) and counts.sum()==nr;hist+=counts
 group=root_groups[ident];assert len(group)==2*len(actions);lookup={}
 for root in group:
  arm=ARMS.index(root['continuation']);pairs=e.action(root['action']);validate_matching(c,s,pairs)
  assert pairs in actions and root['action_index']==CATALOGUE.index(pairs)
  aliases=json.loads(root['labels']);assert set(aliases)=={label for label,ix in zip(LABELS,labels) if actions[ix]==pairs}
  label=LABELS.index(aliases[0]);samples=raw['costs'][i,arm,label]
  for alias in aliases:
   for key in raw:np.testing.assert_array_equal(raw[key][i,arm,LABELS.index(alias)],raw[key][i,arm,label])
   for key in bound:np.testing.assert_array_equal(bound[key][i,arm,LABELS.index(alias)],bound[key][i,arm,label])
  np.testing.assert_array_equal(samples,json.loads(root['scenario_costs']))
  same(samples.mean(),root['total_mean']);same(samples.std(ddof=1)/np.sqrt(32),root['mc_se'])
  same(raw['base'][i,arm,label].mean(),root['base_mean']);same(raw['tail'][i,arm,label].mean(),root['tail_mean'])
  same(sum(root[k] for k in COMPONENTS),root['total_mean'])
  same((bound['cutoff_remaining'][i,arm,label]>0).sum(1).mean(),root['cutoff_busy_workers'])
  same((bound['terminal_remaining'][i,arm,label]>0).sum(1).mean(),root['terminal_busy_workers'])
  same(bound['terminal_failed'][i,arm,label].sum(1).mean(),root['terminal_failed_machines'])
  lookup[arm,pairs]=root
 for pairs in actions:same(lookup[0,pairs]['first_step_mean'],lookup[1,pairs]['first_step_mean'])
 extra={}
 for contrast,label,exposure in [('pm',2,row['pm_exposure']),('now',1,row['now_exposure']),('reference',3,row['structural_choice'])]:
  for arm_idx,arm in enumerate(ARMS):
   samples=raw['costs'][i,arm_idx,0]-raw['costs'][i,arm_idx,label]
   prefix=contrast if arm_idx==0 and contrast!='reference' else f'{contrast}_deadline' if contrast!='reference' else f'{arm}_reference'
   same(samples.mean() if exposure else 0,row[prefix+'_delta']);same(samples.std(ddof=1)/np.sqrt(32) if exposure else 0,row[prefix+'_mc_se'])
   extra[f'{contrast}_{arm}_base_delta']=float((raw['base'][i,arm_idx,0]-raw['base'][i,arm_idx,label]).mean())
   extra[f'{contrast}_{arm}_tail_delta']=float((raw['tail'][i,arm_idx,0]-raw['tail'][i,arm_idx,label]).mean())
   for component in COMPONENTS:extra[f'{contrast}_{arm}_{component}_delta']=lookup[arm_idx,rl][component]-lookup[arm_idx,actions[labels[label]]][component]
  if contrast!='reference':
   interaction=(raw['costs'][i,0,0]-raw['costs'][i,0,label])-(raw['costs'][i,1,0]-raw['costs'][i,1,label])
   same(interaction.mean() if exposure else 0,row[contrast+'_interaction_delta'])
   same(interaction.std(ddof=1)/np.sqrt(32) if exposure else 0,row[contrast+'_interaction_mc_se'])
   same(row[contrast+'_delta']/(4*(60-row['tick'])),row[contrast+'_normalized_delta'])
 pm=[machine for machine,worker in rl if not s.failed[machine]];risk=[machine for machine in pm if s.ages[machine]+1>=c.failure_age]
 available={machine for pairs in e.joint.feasible(c,s) for machine,worker in pairs if not s.failed[machine]}
 risky={machine for machine in available if s.ages[machine]+1>=c.failure_age}
 assert (row['at_risk_opportunities'],row['young_opportunities'])==(len(risky),len(available-risky))
 assert (row['at_risk_pm_starts'],row['young_pm_starts'])==(len(risk),len(pm)-len(risk))
 extra['pm_timing']='NO_PM' if not pm else 'ALL_RISK' if len(risk)==len(pm) else 'ALL_YOUNG' if not risk else 'MIXED_RISK'
 extra['pending_failed']=sum(s.failed[machine] and machine not in s.assigned for machine in range(4))
 augmented.append({**row,**extra})
actual=m['actual_counts'];assert (roots_count,transitions,rl_rows,tail_rows)==(actual['unique_roots'],actual['forecast_transitions'],actual['batched_decision_rows'],actual['tail_transition_rows'])
np.testing.assert_array_equal(hist,actual['rl_action_histogram']);assert audits.near_tie_scalar_rows.sum()==actual['near_tie_scalar_rows']
d=pd.DataFrame(augmented)

# Independent scalar replay: prospectively fixed first historical cohort/shock,
# ALL ten models/conditions/origins/landmarks, scenario indices0/31; no outcome filter.
scalar_transitions=scalar_states=0;component_checks=0
subset=[r for r in rows if r['cohort']==cfg['cohorts'][0] and r['shock']==cfg['physical_shocks'][0]]
assert len(subset)==160
for row in tqdm(subset,desc='Scalar check fixed 160-state audit subset, scenarios0/31'):
 i=index[identity(row)];c=panels[row['cohort'],row['condition']];s=e.state(row['state']);rl=e.action(row['rl_action']);ref=e.action(row['rule_action'])
 actions,labels=root_actions(s,rl,ref);t0=row['tick'];model=models[row['model_seed']]
 shocks=np.random.default_rng(row['forecast_seed']).random((60-t0,32,4))<c.failure_probability
 for arm_idx,arm in enumerate(ARMS):
  for ix,root in enumerate(actions):
   label=labels.index(ix)
   for scenario in (0,31):
    current=s;base=tail=0.;components={k:0. for k in COMPONENTS}
    for offset in range(60-t0):
     t=t0+offset
     action=root if offset==0 else model.select(c,current,48-t,48)[0] if arm=='rl' and t<48 else e.prior.rule_action(c,current,60-t,'deadline_matching')
     current,cost,info=transition(c,current,action,shocks[offset,scenario]);scalar_transitions+=1
     if t<48:base+=cost
     else:tail+=cost
     for k in COMPONENTS:components[k]+=info[k]
     if t==47:
      for k,v in asdict(current).items():np.testing.assert_array_equal(v,bound['cutoff_'+k][i,arm_idx,label,scenario])
    assert base==raw['base'][i,arm_idx,label,scenario] and tail==raw['tail'][i,arm_idx,label,scenario]
    assert sum(components.values())==base+tail;component_checks+=1
    for k,v in asdict(current).items():np.testing.assert_array_equal(v,bound['terminal_'+k][i,arm_idx,label,scenario])
 scalar_states+=1

# Independent crossed bootstrap by resampling-frequency weights; do not call summarize/crossed_ratio.
rng=np.random.default_rng(role_seed(cfg['analysis_seed'],e.VERSION+':crossed_bootstrap'))
rd=rng.integers(10,size=(20000,10));pd_=rng.integers(8,size=(20000,8))
rw=np.stack([(rd==i).sum(1) for i in range(10)],1);pw=np.stack([(pd_==j).sum(1) for j in range(8)],1)
published=json.loads((RUN/'summary.json').read_text());recomputed={};cells=[];secondary=[]
for condition,contrast in itertools.product(cfg['conditions'],('pm','now')):
 num=np.zeros((10,8));den=np.zeros((10,8))
 for i,seed in enumerate(cfg['model_seeds']):
  for j,cohort in enumerate(cfg['cohorts']):
   group=d[(d.condition==condition)&(d.model_seed==seed)&(d.cohort==cohort)]
   num[i,j]=group[contrast+'_delta'].sum();den[i,j]=group[contrast+'_exposure'].sum()
   cells.append(dict(condition=condition,contrast=contrast,model_seed=seed,cohort=cohort,numerator=num[i,j],exposure=int(den[i,j])))
 draws=np.einsum('bi,ij,bj->b',rw,num,pw)/np.einsum('bi,ij,bj->b',rw,den,pw)
 ci=np.quantile(draws,[.00625,.99375]);mean=num.sum()/den.sum();old=published['primary_contrasts'][condition+':'+contrast]
 same(mean,old['estimate']);same(ci,old['confidence_interval']);assert int(den.sum())==old['exposure'] and np.isfinite(draws).all()
 recomputed[condition+':'+contrast]=dict(estimate=float(mean),confidence_interval=ci.tolist(),exposure=int(den.sum()),status=old['status'])
 for metric in (contrast+'_deadline_delta',contrast+'_interaction_delta'):
  secondary.append(dict(condition=condition,contrast=contrast,metric=metric,eligible=int(den.sum()),mean=float(d[(d.condition==condition)&(d[contrast+'_exposure']==1)][metric].mean())))
np.testing.assert_array_equal(pd.DataFrame(cells).exposure,pd.read_csv(RUN/'profile_cells.csv').exposure)
same(pd.DataFrame(cells).numerator,pd.read_csv(RUN/'profile_cells.csv').numerator)

def group_table(frame,keys,metric):
 return frame.groupby(keys).agg(n=(metric,'size'),mean=(metric,'mean'),negative=(metric,lambda a:(a<0).sum()),positive=(metric,lambda a:(a>0).sum()),median_mc_se=(metric.replace('_delta','_mc_se'),'median')).reset_index()
for contrast in ('pm','now'):
 eligible=d[d[contrast+'_exposure']==1]
 for name,keys in [('timing',['condition','origin','pm_timing']),('landmark',['condition','origin','tick']),('model',['condition','model_seed']),('profile',['condition','cohort']),('action_kind',['condition','origin','rl_kind'])]:
  group_table(eligible,keys,contrast+'_delta').to_csv(OUT/f'{contrast}_{name}.csv',index=False)
 eligible.groupby(['condition','origin'])[[col for col in d.columns if col.startswith(contrast+'_rl_')]].mean().to_csv(OUT/f'{contrast}_phase_components.csv')
behaviors=[]
for condition,group in d.groupby('condition'):
 choices=group[group.structural_choice==1];wait=group[group.rl_wait==1]
 behavior=dict(condition=condition,states=len(group),choice_states=len(choices),strategic_wait=int(choices.rl_wait.sum()),
  reference_strategic_wait=int(choices.reference_wait.sum()),forced_wait=int((group.structural_choice==0).sum()),total_wait=len(wait),
  pm_roots=int(group.pm_exposure.sum()),pm_machine_starts=int(group.pm_starts.sum()))
 for risk in ('at_risk','young'):
  behavior[risk+'_opportunities']=int(group[risk+'_opportunities'].sum());behavior[risk+'_starts']=int(group[risk+'_pm_starts'].sum())
 behaviors.append(behavior)
strategic=d[(d.structural_choice==1)&(d.rl_wait==1)]
# For RL WAIT roots, compare WAIT to the frozen reference root on identical futures;
# no optimization over saved alternatives or claim of optimal WAIT.
strategic.to_csv(OUT/'strategic_wait_states.csv',index=False)
group_table(d[d.structural_choice==1],['condition','origin'],'rl_reference_delta').to_csv(OUT/'reference_root.csv',index=False)
d.to_csv(OUT/'state_diagnostics.csv',index=False);pd.DataFrame(behaviors).to_csv(OUT/'behavior.csv',index=False)
pd.DataFrame(secondary).to_csv(OUT/'continuation_comparison.csv',index=False)
save('primary_recomputed.json',recomputed)
save('audit.json',dict(status='PASS',verification_status='ANALYZED with exact scalar audit subset, not a new full-run replication',
 run_manifest_sha256=e.sha(RUN/'manifest.json'),hashed_files=len(m['artifacts']),artifact_bytes=sum(r['bytes'] for r in m['artifacts'].values()),snapshot_matches_commit=True,
 source_audit=source_audit,states=2560,unique_roots=roots_count,raw_cost_samples=2560*2*4*32,forecast_transitions_reported=transitions,
 exact_raw_arrays_metrics_checked=True,frozen_models=10,scalar_states=scalar_states,scalar_scenarios=[0,31],scalar_transition_checks=scalar_transitions,
 scalar_return_component_sum_checks=component_checks,primary_cis_independently_recomputed=4,full_lab_runtime_seconds=m['elapsed_seconds'],
 scope='same historical occupancy panel, exploratory single-step root interventions; deterministic frozen policy; future tapes fixed'))
print('OUTPUT',OUT,flush=True);print(json.dumps(recomputed,indent=2),flush=True)
print(pd.DataFrame(behaviors).to_string(index=False),flush=True)
