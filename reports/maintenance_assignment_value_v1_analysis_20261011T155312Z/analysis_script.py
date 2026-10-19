"""Read-only audit and analysis of transferred assignment-value artifacts; no new experiment."""
import json,hashlib,itertools,subprocess,sys,time
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
ROOT=Path('/Users/bachng/Coding/Reinforcement Learning/marl')
WT=Path('/Users/bachng/.codex/worktrees/maintenance-joint-rl-v1/marl')
RUN=ROOT/'lab_results/maintenance_assignment_value_v1_full_cpu_20261011T153135Z'
OUT=ROOT/'reports'/('maintenance_assignment_value_v1_analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
OUT.mkdir(exist_ok=False);print('REPORT',OUT,flush=True)
sys.path.insert(0,str(WT/'src'))
from ht_pdm_fjsp import maintenance_assignment_value_benchmark as b
from ht_pdm_fjsp import maintenance_assignment_value_statistics as stats
from ht_pdm_fjsp import maintenance_assignment_value_training as training
from ht_pdm_fjsp.maintenance_assignment_value import counts
from ht_pdm_fjsp.maintenance_assignment_value_source import sha,verify_source,parameters_sha
from ht_pdm_fjsp.maintenance_assignment_value_policy import RelativeValueHead,ValueController,CONTRACT,inputs
from ht_pdm_fjsp import maintenance_assignment_value_forks as f
from ht_pdm_fjsp.maintenance_assignment_value_data import all_assignment_classes
from ht_pdm_fjsp.maintenance_autoregressive_policy import feature_batch
from ht_pdm_fjsp.maintenance_dispatch import DispatchState,validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig,WaitingEnv,transition
from ht_pdm_fjsp.maintenance_opportunity_cost import rule_action
from ht_pdm_fjsp.maintenance_timing_statistics import crossed_ratio

torch.set_num_threads(1)
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def eq(a,z,tol=1e-9):
 if isinstance(a,dict):
  assert set(a)==set(z),(set(a),set(z))
  for k in a:eq(a[k],z[k],tol)
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(z)
  for x,y in zip(a,z,strict=True):eq(x,y,tol)
 elif isinstance(a,(int,float,np.number)) and not isinstance(a,(bool,np.bool_)):
  assert np.isclose(a,z,rtol=tol,atol=tol),(a,z)
 else:assert a==z,(a,z)
def state(v):
 v=json.loads(v) if isinstance(v,str) else v
 return DispatchState(**{k:tuple(a) for k,a in v.items()})
def action(v):
 v=json.loads(v) if isinstance(v,str) else v
 return tuple(tuple(e) for e in v)
def config(v):return WaitingConfig(**{k:tuple(tuple(a) for a in x) if k in ('service_time','restoration') else x for k,x in v.items()})
m=json.loads((RUN/'manifest.json').read_text());cfg=m['settings'];assert m['status']=='COMPLETED' and m['audit']=='PASS' and not m['dirty']
assert m['commit']=='e821e7f9ccfd9a5cdd06e233dbc39ae39c98bb5c'
eq(cfg,b.settings('full'));eq(counts(cfg),m['expected_counts']);eq(m['actual_counts'],{k:v for k,v in counts(cfg).items() if k!='max_forecast_transitions'})
for n,r in m['artifacts'].items():
 p=RUN/n;assert p.resolve().is_relative_to(RUN.resolve());assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],n
source_count=0
for p in (RUN/'source_snapshot').iterdir():
 rel='src/ht_pdm_fjsp/'+p.name if p.suffix=='.py' else p.name if p.name in ('pyproject.toml','uv.lock') else 'configs/'+p.name
 original=subprocess.check_output(['git','show',m['commit']+':'+rel],cwd=ROOT)
 assert p.read_bytes()==original==(WT/rel).read_bytes(),rel;source_count+=1
assert (RUN/'plan.md').read_bytes()==subprocess.check_output(['git','show',m['commit']+':docs/maintenance_assignment_value_v1_plan.md'],cwd=ROOT)
eq(b.seed_audit(cfg),m['seed_audit']);panels,registries,physical=b.metadata();eq(physical,m['physical_split_audit'])
resolved=json.loads((RUN/'resolved_config.json').read_text());eq(json.loads(json.dumps(registries['full'])),resolved['physical_profiles'])
configs={(r['split'],r['cohort'],r['condition']):config(r['config']) for r in resolved['physical_profiles']}
source_dir=ROOT/'lab_results'/json.loads((RUN/'source_contract.json').read_text())['source_run']
models,contract,sa=verify_source(source_dir,cfg);eq(sa,m['source_audit']);eq(contract,json.loads((RUN/'source_contract.json').read_text()))
frames={name:pd.read_csv(RUN/(name+'.csv')) for name in ('episodes','episodes.partial','coordination','decisions','pm_choices','source_trajectories','label_states','label_pairs','ranking_states','ranking_pairs','ranking_predictions','timing_states','timing_predictions','training_progress')}
assert frames['episodes'].equals(frames['episodes.partial']);assert len(frames['episodes'])==1728 and len(frames['coordination'])==1728
assert len(frames['decisions'])==4320 and len(frames['pm_choices'])==23029 and len(frames['source_trajectories'])==27648
assert len(frames['label_states'])==1728 and len(frames['label_pairs'])==2992 and len(frames['ranking_pairs'])==972
assert (frames['coordination'].invalid_actions==0).all() and all((frames['coordination'][k]=='PASS').all() for k in ('scalar_batch_replay','matching_feasibility','token_padding_audit'))
for phase in ('base','tail','extended'):
 ep=frames['episodes'];np.testing.assert_array_equal(ep[phase+'_objective'],ep[[phase+'_'+k for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost')]].sum(axis=1))
summary,cells=stats.summarize(frames['episodes'].to_dict('records'),cfg);eq(summary,json.loads((RUN/'summary.json').read_text()))
diagnostic,dcells=stats.diagnostic_summary(frames['ranking_states'].to_dict('records'),frames['ranking_pairs'].to_dict('records'),frames['timing_states'].to_dict('records'),cfg)
eq(diagnostic,json.loads((RUN/'diagnostic_summary.json').read_text()));save('summary_verified.json',summary);save('diagnostic_summary_verified.json',diagnostic)
pd.DataFrame(cells).to_csv(OUT/'verified_profile_differences.csv',index=False)
# Paired labels, raw costs, boundaries, RNG tapes and phase budgets independently reconciled.
forecast_meta={};compute={};npz_count=0
for p in tqdm(sorted((RUN/'forecasts').rglob('*.npz')),desc='Audit saved tapes / costs'):
 with np.load(p) as z:
  ident=json.loads(str(z['identity_json']));phase=ident['phase'];cost=z['costs'];roots=ident['roots'];n,s=cost.shape;t=ident['tick'];c=config(ident['config']);ss=state(ident['state'])
  assert n==len(roots) and z['shocks'].shape==(60-t,s,4)
  assert ident['seed']==b.future_seed(cfg,ident,phase)
  expected=np.random.default_rng(ident['seed']).random((60-t,s,4))<c.failure_probability
  np.testing.assert_array_equal(expected,z['shocks']);np.testing.assert_array_equal(cost,z['base']+z['tail'])
  np.testing.assert_array_equal(cost,sum(z['components_'+k] for k in f.COMPONENTS))
  assert np.isfinite(cost).all();assert z['base'].shape==z['tail'].shape==cost.shape
  for a in roots:validate_matching(c,ss,action(a))
  for name,v in ident['counts'].items():compute[name]=compute.get(name,0)+v
  for k,v in dict(forecast_transitions=n*s*(60-t),forecast_roots=n,forecast_futures=n*s,**{phase+'_states':1,phase+'_transitions':n*s*(60-t)}).items():compute[k]=compute.get(k,0)+v
  forecast_meta[p.relative_to(RUN).as_posix()]=ident;npz_count+=1
for k,v in compute.items():assert m['effective_compute'][k]==v,(k,v)
for row in tqdm(frames['label_pairs'].to_dict('records'),desc='Recompute train/development labels'):
 ident=forecast_meta[row['forecast']];roots=[action(x) for x in ident['roots']];left,right=roots.index(action(row['left'])),roots.index(action(row['right']))
 with np.load(RUN/row['forecast']) as z:
  delta=(z['costs'][right]-z['costs'][left])/20
 eq(float(delta.mean()),row['target_normalized']);eq(float(delta.std(ddof=1)/np.sqrt(len(delta))),row['paired_mc_se_normalized'])
for row in frames['ranking_states'].to_dict('records'):
 with np.load(RUN/row['selection_forecast']) as z:assert f.choose(z['costs'],tuple(json.loads(row['original_class'])))==row['reference_root']
 with np.load(RUN/row['evaluation_forecast']) as z:
  for name in ('frozen_ppo','true_value','shuffled_value','simulation_reference'):
   gain,se=f.paired_gain(z['costs'],0,row[name+'_chosen']);eq(gain,row[name+'_gain']);eq(se,row[name+'_mc_se'])
for row in frames['ranking_pairs'].to_dict('records'):
 ident=forecast_meta[row['evaluation_forecast']];roots=[action(x) for x in ident['roots']]
 with np.load(RUN/row['evaluation_forecast']) as z:
  delta=(z['costs'][roots.index(action(row['right']))]-z['costs'][roots.index(action(row['left']))])/20
 eq(float(delta.mean()),row['measured_delta']);eq(float(delta.std(ddof=1)/np.sqrt(len(delta))),row['paired_mc_se'])
 for name in ('true_value','shuffled_value'):eq((row[name+'_predicted_delta']-row['measured_delta'])**2,row[name+'_squared_error'])
for row in frames['timing_states'].to_dict('records'):
 with np.load(RUN/row['selection_forecast']) as z:assert f.choose(z['costs'],tuple(range(len(z['costs']))))==row['selected_root']
 with np.load(RUN/row['evaluation_forecast']) as z:
  for name,index in [('correction',row['selected_root']),('no_pm',row['no_pm_root'])]:
   gain,se=f.paired_gain(z['costs'],0,index);eq(gain,row[name+'_gain']);eq(se,row[name+'_mc_se'])
# Checkpoint selection, tensor labels and controls. Never retrain or select using test.
selection=json.loads((RUN/'checkpoint_selection.json').read_text());coverage=json.loads((RUN/'training_coverage.json').read_text());assert selection['locked_before_test']
controllers={};chosen=[]
for seed,train_seed in zip(cfg['model_seeds'],cfg['train_seeds'],strict=True):
 tensors={split:torch.load(RUN/'datasets'/f'{split}_{seed}.pt',weights_only=True) for split in ('train','development')}
 for split,ds in tensors.items():
  rows=frames['label_pairs'].query('split==@split and model_seed==@seed')
  np.testing.assert_array_equal(ds['targets'].numpy(),rows.target_normalized.to_numpy(dtype='float32'))
 perm=torch.load(RUN/'datasets'/f'shuffle_{seed}.pt',weights_only=True);assert torch.equal(perm,training.shuffled_permutation(tensors['train']['conditions'],train_seed))
 assert torch.equal(tensors['train']['conditions'],tensors['train']['conditions'][perm])
 controllers[seed]={}
 cov=next(r for r in coverage if r['source_seed']==seed);assert cov['train_labels']==len(tensors['train']['targets']) and cov['development_labels']==len(tensors['development']['targets'])
 assert parameters_sha(models[seed])==cov['source_parameters_sha256']
 for name in ('true_value','shuffled_value'):
  rec=selection['selections'][str(seed)][name];best=min(rec['candidates'],key=lambda r:(r['true_development_mse'],r['epoch']));eq(best,rec['selected'])
  for cand in rec['candidates']:
   payload=torch.load(RUN/cand['checkpoint'],weights_only=True);head=RelativeValueHead();head.load_state_dict(payload['state_dict']);head.eval()
   assert payload['feature_contract']==CONTRACT and payload['source_sha256']==contract['models'][str(seed)]['sha256']
   eq(training.mse(head,tensors['development']),cand['true_development_mse'],tol=2e-6)
   if cand==best:controllers[seed][name]=ValueController(models[seed],head,name)
  chosen.append(dict(source_seed=seed,controller=name,epoch=best['epoch'],development_mse=best['true_development_mse']))
 assert selection['selections'][str(seed)]['true_value']['selected']['matched_initial_sha']==selection['selections'][str(seed)]['shuffled_value']['selected']['matched_initial_sha']
assert len(frames['training_progress'])==480 and sum(r['matched_optimizer_steps'] for r in coverage)==m['effective_compute']['optimizer_steps']==3360
pd.DataFrame(chosen).to_csv(OUT/'checkpoint_selection_verified.csv',index=False)
# Reconstruct physical transitions from logged tapes, and inference for logged test decisions.
source_replay=0
for row in tqdm(frames['source_trajectories'].to_dict('records'),desc='Replay logged source transitions'):
 c=configs[row['split'],row['cohort'],row['condition']];ss=state(row['state']);a=action(row['action'])
 following,cost,info=transition(c,ss,a,json.loads(row['shock_bits']));eq(asdict(following),asdict(state(row['following'])));eq(info,json.loads(row['metrics']));source_replay+=1
physical_replay=0;policy_diffs=[];decode_rows=[];env_cache={}
for row in tqdm(frames['decisions'].to_dict('records'),desc='Replay logged test decisions'):
 c=configs['test',row['cohort'],row['condition']];ss=state(row['state']);a=action(row['action']);key=(row['cohort'],row['condition'],row['shock'])
 if key not in env_cache:
  env=WaitingEnv(c);env.reset(b.physical_seed(row['cohort'],row['shock']),b.initial_state(c,row['cohort']));env_cache[key]=env.events
 following,cost,info=transition(c,ss,a,env_cache[key][row['time']]);eq(asdict(following),asdict(state(row['following'])));eq(info,json.loads(row['physical_metrics']))
 if row['phase']=='base':
  ctrl=models[row['train_seed']] if row['controller']=='frozen_ppo' else controllers[row['train_seed']][row['controller']]
  selected,details=ctrl.select(c,ss,48-row['time'],48)
  if selected!=a:policy_diffs.append(dict(condition=row['condition'],cohort=row['cohort'],seed=row['train_seed'],time=row['time'],controller=row['controller']))
  diag=json.loads(row['controller_diagnostics'])
  if diag['assignment_decode'] is not None:
   d=diag['assignment_decode'];original=action(d['original']);selected=action(d['selected'])
   assert selected==a and {m for m,j in original}=={m for m,j in a}
   assert {e for e in original if ss.failed[e[0]]}=={e for e in a if ss.failed[e[0]]}
   decode_rows.append(dict(condition=row['condition'],controller=row['controller'],source_seed=row['train_seed'],cohort=row['cohort'],time=row['time'],changed=d['changed'],alternatives=len(d['candidates'])>1))
 else:assert rule_action(c,ss,60-row['time'],'deadline_matching')==a
 physical_replay+=1
pd.DataFrame(decode_rows).to_csv(OUT/'logged_assignment_decode_audit.csv',index=False)
# Exact selected checkpoint predictions on all288 ranking states.
for row in frames['ranking_predictions'].to_dict('records'):
 rs=frames['ranking_states'];found=rs[(rs.condition==row['condition'])&(rs.cohort==row['cohort'])&(rs.model_seed==row['model_seed'])&(rs.tick==row['tick'])].iloc[0]
 ident=forecast_meta[found.evaluation_forecast];c=config(ident['config']);ss=state(ident['state']);roots=[action(a) for a in ident['roots']]
 for name,ctrl in controllers[row['model_seed']].items():
  with torch.no_grad():pred=ctrl.head(inputs(feature_batch([c],[ss],[48-row['tick']],48),ctrl.base))[0].numpy()
  indices=[f.CATALOGUE.index(a) for a in roots]
  np.testing.assert_allclose(pred[indices],json.loads(row[name+'_relative_scores']),rtol=1e-5,atol=2e-6)
  assert roots[json.loads(row['chosen'])[name]]==ctrl.select(c,ss,48-row['tick'],48)[0]
# Bounded validation on supplied forecast tapes, not a fresh full run.
chosen_files={}
for name,ident in forecast_meta.items():chosen_files.setdefault((ident['phase'],ident['condition'],ident['model_seed']),[]).append(name)
sampled=[];replay_futures=0;replay_transitions=0
for group,names in tqdm(list(chosen_files.items()),desc='Replay stratified saved future tapes'):
 ordered=sorted(names,key=lambda n:(forecast_meta[n]['tick'],forecast_meta[n]['cohort']))
 for name in dict.fromkeys([ordered[0],ordered[-1]]):
  ident=forecast_meta[name];seed=ident['model_seed'];ctrl=controllers[seed]['true_value'] if ident['phase'].startswith('timing') and name.endswith('_true_value.npz') else models[seed]
  with np.load(RUN/name) as z:
   roots=ident['roots'];ri=sorted(set([0,len(roots)-1]));si=sorted(set([0,z['costs'].shape[1]-1]))
   subset=lambda a:a[ri][:,si]
   v=dict(roots=tuple(action(roots[i]) for i in ri),costs=subset(z['costs']),base=subset(z['base']),tail=subset(z['tail']),first=subset(z['first']),shocks=z['shocks'][:,si],trace=None,
     components={k:subset(z['components_'+k]) for k in f.COMPONENTS},
     cutoff={k:subset(z['cutoff_'+k]) for k in ('ages','failed','pending_wait','remaining','assigned')},terminal={k:subset(z['terminal_'+k]) for k in ('ages','failed','pending_wait','remaining','assigned')})
   assert f.scalar_replay(config(ident['config']),state(ident['state']),v,ctrl,ident['tick'])
   replay_futures+=len(ri)*len(si);replay_transitions+=len(ri)*len(si)*(60-ident['tick']);sampled.append(name)
audit=dict(status='PASS',artifact_hashes_checked=len(m['artifacts']),raw_bytes=sum(r['bytes'] for r in m['artifacts'].values()),snapshot_files_match_commit=source_count,
 source_checkpoint_artifacts_checked=sa['verified_source_artifacts'],source_models_checked=3,primary_summary_recomputed=True,diagnostic_summary_recomputed=True,
 saved_forecast_files_reconciled=npz_count,all_raw_labels_recomputed=2992,all_ranking_pairs_recomputed=972,
 source_logged_transitions_replayed=source_replay,test_logged_transitions_replayed=physical_replay,test_logged_policy_action_differences=policy_diffs,
 ranking_model_prediction_states_verified=288,sampled_future_files=len(sampled),sampled_future_paths=replay_futures,sampled_future_transitions=replay_transitions,
 full_training_rerun=False,new_full_outcomes=False,raw_artifacts_modified=False,lab_access=False,
 reproducibility_scope='All supplied summary/label/tape/accounting aggregates, logged decisions, selected predictions verified; bounded scalar future replay. Entire training and full counterfactual policy computation not rerun.')
save('audit.json',audit);save('sampled_future_files.json',sampled)
print('AUDIT_DONE',audit,flush=True)
# Post-hoc diagnostics: paired clustered MSE reduction, zero predictor, signs and overrides.
extra={};rows=[];pairs=frames['ranking_pairs'];ranking=frames['ranking_states'];timing=frames['timing_states'];lp=frames['label_pairs']
for cond,g in pairs.groupby('condition'):
 zero=g.measured_delta**2;noise=g.paired_mc_se**2
 item=dict(pair_count=len(g),zero_mse=float(zero.mean()),mc_noise_mse=float(noise.mean()),true_mse=float(g.true_value_squared_error.mean()),shuffle_mse=float(g.shuffled_value_squared_error.mean()),
  true_vs_shuffle_mse_reduction=float(1-g.true_value_squared_error.mean()/g.shuffled_value_squared_error.mean()),true_vs_zero_mse_reduction=float(1-g.true_value_squared_error.mean()/zero.mean()),metrics={})
 for subset,x in [('all_classes',g),('original_PPO_class',g[g.original_class])]:
  item['metrics'][subset]={}
  for name in ('true_value','shuffled_value'):
   pred=x[name+'_predicted_delta'];truth=x.measured_delta;resolved=truth.abs()>1.96*x.paired_mc_se
   item['metrics'][subset][name]=dict(n=len(x),mse=float(x[name+'_squared_error'].mean()),sign_correct=int((pred*truth>0).sum()),sign_accuracy=float((pred*truth>0).mean()),
    prediction_std=float(pred.std()),target_std=float(truth.std()),correlation=float(pred.corr(truth)),resolved_n=int(resolved.sum()),resolved_sign_correct=int((pred[resolved]*truth[resolved]>0).sum()),
    resolved_sign_accuracy=float((pred[resolved]*truth[resolved]>0).mean()))
 for baseline in ('shuffled_value','zero'):
  gx=g.copy();gx['paired_mse_improvement']=(gx.shuffled_value_squared_error if baseline=='shuffled_value' else gx.measured_delta**2)-gx.true_value_squared_error
  result,c=stats.conditional(gx.to_dict('records'),cfg,cond,'paired_mse_improvement',cfg['test_cohorts'],exposure_key='pair_count')
  item['paired_mse_improvement_vs_'+baseline]=result;rows+=c
 extra[cond]=item
 for name in ('true_value','shuffled_value','simulation_reference'):
  x=ranking[(ranking.condition==cond)&(ranking.exposure==1)];switch=x[name+'_chosen']!=0;gain=x[name+'_gain'];se=x[name+'_mc_se']
  item[name+'_overrides']=dict(eligible=len(x),switched=int(switch.sum()),positive=int((gain[switch]>0).sum()),negative=int((gain[switch]<0).sum()),mc_positive95=int((gain[switch]-1.96*se[switch]>0).sum()),mc_negative95=int((gain[switch]+1.96*se[switch]<0).sum()))
 for name in ('frozen_ppo','true_value'):
  x=timing[(timing.condition==cond)&(timing.controller==name)&(timing.exposure==1)];switch=x.selected_root!=0
  item[name+'_timing']=dict(eligible=len(x),switched=int(switch.sum()),positive=int((x.correction_gain[switch]>0).sum()),negative=int((x.correction_gain[switch]<0).sum()),mc_positive95=int((x.correction_gain[switch]-1.96*x.correction_mc_se[switch]>0).sum()))
label_coverage=lp.groupby(['split','condition','tick']).size().rename('labels').reset_index();label_coverage.to_csv(OUT/'label_coverage_by_clock.csv',index=False)
extra['label_coverage']=dict(train_labels=int(sum(lp.split=='train')),development_labels=int(sum(lp.split=='development')),training_tick0_fraction=float((lp.query('split=="train"').tick==0).mean()),development_tick0_fraction=float((lp.query('split=="development"').tick==0).mean()))
pd.DataFrame(rows).to_csv(OUT/'supplementary_mse_cluster_cells.csv',index=False);save('supplementary_diagnostics.json',extra)
# Find harmful original-class overrides and export all original gains, including zeros.
ranking.to_csv(OUT/'assignment_state_gains.csv',index=False)
ranking[(ranking.exposure==1)&(ranking.true_value_chosen!=0)].sort_values('true_value_gain').head(10).to_csv(OUT/'largest_harmful_overrides.csv',index=False)
print('SUPPLEMENTARY',json.dumps(extra,indent=2),flush=True)
(OUT/'analysis_script.py').write_bytes(Path(__file__).read_bytes())
print('COMPLETE',OUT,flush=True)
