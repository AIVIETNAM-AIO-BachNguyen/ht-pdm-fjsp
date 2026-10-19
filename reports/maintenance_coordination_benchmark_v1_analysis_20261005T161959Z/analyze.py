"""Read-only raw-data statistics, budget checks and report tables."""
import csv
import hashlib
import json
import math
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

raw,out = map(Path,sys.argv[1:])
ARMS=('central_ar','central_proposal','ippo_local','mappo_local','mappo_message')
REFS=('risk_skill_rule','deadline_matching','lookahead_matching')
FAMILIES=('n3_nominal','n3_pressure','n4_nominal','n4_pressure','n5_pressure','n5_k3_sparse','small','n4_pressure_long')
PRIMARY=FAMILIES[:4]
NOMINAL=('n3_nominal','n4_nominal')
CRIT=3.110934823179475
metrics=('base_cost','service_adjusted_cost','objective','waiting_cost','overdue_waiting_ticks','maintenance_cost','failure_cost','unavailability_cost','failures','preventive_jobs','corrective_jobs','jobs','completed_jobs','unavailable_ticks','failed_waiting_ticks','rejected_proposals','proposal_count','contested_technicians','avoidable_unassigned_requests','message_scalars','failed_deferred_decisions','terminal_failed_never_serviced_fraction','wait_violation_machine_fraction')
def read(name):
 with (raw/name).open() as f:return list(csv.DictReader(f))
def write(name,rows):
 with (out/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(name,obj): (out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def key(r):return r['algorithm'],int(r['train_seed']),r['family'],int(r['eval_seed'])
def q95(v):
 s=sorted(v);return s[math.ceil(.95*len(s))-1]
manifest=json.loads((raw/'manifest.json').read_text());cfg=manifest['settings']
summary=json.loads((raw/'summary.json').read_text())
assert manifest['status']=='COMPLETED' and manifest['git_revision']=='0cafebbfb54bd9835dac210d9435befcc9c42f42'
assert manifest['git_dirty'] is False
panels={name:read(name) for name in ['episodes.csv','development_episodes.csv','reference_episodes.csv','message_off_episodes.csv']}
allrows=sum(panels.values(),[])
checks={}
checks['unique_all_episode_keys']=len({key(r) for r in allrows})==len(allrows)
checks['complete_raw_episode_keys']=True
for filename,algs,seeds,families,evals in [
 ('episodes.csv',ARMS,cfg['train_seeds'],FAMILIES,cfg['evaluation_seeds']),
 ('development_episodes.csv',ARMS,cfg['train_seeds'],('development',),cfg['development_seeds']),
 ('message_off_episodes.csv',('mappo_message_off',),cfg['train_seeds'],FAMILIES,cfg['evaluation_seeds'])]:
 expected={(a,s,f,e) for a in algs for s in seeds for f in families for e in evals}
 checks['complete_raw_episode_keys'] &= {key(r) for r in panels[filename]}==expected and len(panels[filename])==len(expected)
expected={(a,-1,f,e) for a in REFS for f in (*FAMILIES,'development') for e in (cfg['development_seeds'] if f=='development' else cfg['evaluation_seeds'])}
checks['complete_raw_episode_keys'] &= {key(r) for r in panels['reference_episodes.csv']}==expected and len(expected)==len(panels['reference_episodes.csv'])
checks['episode_economic_and_priced_costs']=all(abs(float(r['base_cost'])-sum(float(r[k]) for k in ('maintenance_cost','failure_cost','unavailability_cost')))<1e-8 and abs(float(r['objective'])-float(r['base_cost'])-12*float(r['overdue_waiting_ticks']))<1e-8 and abs(float(r['service_adjusted_cost'])-float(r['objective']))<1e-8 and float(r['invalid_assignments'])==0 for r in allrows)
checks['unchanged_raw_partial_matches_final']=(raw/'episodes.partial.csv').read_bytes()==(raw/'episodes.csv').read_bytes()
training=defaultdict(lambda:dict(episodes=0,steps=0,overdue=0.0,first_cost=[],last_cost=[],sequence_hash=hashlib.sha256()))
checks['training_record_budget_and_costs']=True
with (raw/'training_episodes.csv').open() as f:
 for r in csv.DictReader(f):
  a,s=r['algorithm'],int(r['train_seed']);t=training[a,s]
  t['episodes']+=1;t['steps']=int(r['physical_steps']);t['overdue']+=float(r['overdue_waiting_ticks'])
  checks['training_record_budget_and_costs'] &= int(r['episode'])==t['episodes'] and t['steps']==12*t['episodes'] and abs(float(r['base_cost'])-sum(float(r[k]) for k in ('maintenance_cost','failure_cost','unavailability_cost')))<1e-8 and abs(float(r['objective'])-float(r['base_cost'])-12*float(r['overdue_waiting_ticks']))<1e-8
  fields=[r[k] for k in ('episode','physical_steps','config_seed','environment_seed','physical_config_sha256')]
  t['sequence_hash'].update(json.dumps(fields).encode())
  if t['episodes']<=500:t['first_cost'].append(float(r['objective']))
  if t['episodes']>9500:t['last_cost'].append(float(r['objective']))
checks['training_record_budget_and_costs'] &= set(training)=={(a,s) for a in ARMS for s in cfg['train_seeds']} and all(t['episodes']==10000 and t['steps']==120000 for t in training.values())
checks['same_physical_training_panels']=all(len({training[a,s]['sequence_hash'].hexdigest() for a in ARMS})==1 for s in cfg['train_seeds'])
tr=[]
for (a,s),t in training.items():
 tr.append(dict(algorithm=a,train_seed=s,episodes=t['episodes'],physical_steps=t['steps'],overdue_training_ticks=t['overdue'],first_500_mean_cost=st.fmean(t['first_cost']),last_500_mean_cost=st.fmean(t['last_cost']),population_sha256=t['sequence_hash'].hexdigest()))
write('training_audit.csv',tr)
progress=read('training_progress.csv'); pg=defaultdict(list)
for r in progress:pg[r['algorithm'],int(r['train_seed'])].append(r)
checks['progress_updates_and_optimizer_budget']=len(pg)==50 and all(len(rr)==250 and int(rr[-1]['optimizer_steps'])==4000 and int(rr[-1]['physical_steps'])==120000 for rr in pg.values()) and all(math.isfinite(float(r[k])) for r in progress for k in ['loss','policy_loss','value_loss','entropy','gradient_norm'])
checks['checkpoint_count']=len(list(raw.glob('*/train_seed_*/model.pt')))==50
machine=read('machine_metrics.csv');mg=defaultdict(list)
for r in machine:mg[r['algorithm'],int(r['train_seed']),r['family']].append(r)
checks['machine_panel_keys']=len({(*key(r),int(r['machine'])) for r in machine})==len(machine)
expected_machine={(*key(r),m) for r in allrows for m in range(3 if r['family']=='development' else 2 if r['family']=='small' else int(r['family'][1]))}
checks['machine_panel_keys'] &= {(*key(r),int(r['machine'])) for r in machine}==expected_machine
sourcehash={n:hashlib.sha256((raw/'source_snapshot'/n).read_bytes()).hexdigest() for n in manifest['source_hashes']}
checks['frozen_source_hashes']=sourcehash==manifest['source_hashes'] and hashlib.sha256(json.dumps(sourcehash,sort_keys=True).encode()).hexdigest()==manifest['source_bundle_sha256']
# Per seed and family to preserve the declared independent unit.
g=defaultdict(list)
for r in allrows:
 if r['family']!='development':g[r['algorithm'],int(r['train_seed']),r['family']].append(r)
family=[]
for (a,s,f),rr in sorted(g.items()):
 ms=mg[a,s,f];waits=[int(r['max_pending_wait']) for r in ms]
 family.append(dict(algorithm=a,train_seed=s,family=f,episodes=len(rr),**{k:st.fmean(float(r[k]) for r in rr) for k in metrics},wait_p95=q95(waits),max_wait=max(waits),wait_violations=sum(w>4 for w in waits),machine_records=len(ms),terminal_failed_never_started=sum(r['failed_at_terminal']=='True' and int(r['service_starts'])==0 for r in ms)))
write('family_seed_metrics.csv',family)
fs={(r['algorithm'],r['train_seed'],r['family']):r for r in family}
def mean(a,s,k,families=PRIMARY):return st.fmean(fs[a,s,f][k] for f in families)
seedrows=[]
for a in (*ARMS,'mappo_message_off',*REFS):
 for s in ([-1] if a in REFS else cfg['train_seeds']):
  seedrows.append(dict(algorithm=a,train_seed=s,**{k:mean(a,s,k) for k in metrics},nominal_base_cost=mean(a,s,'base_cost',NOMINAL),n3_pressure_wait_p95=fs[a,s,'n3_pressure']['wait_p95'],n4_pressure_wait_p95=fs[a,s,'n4_pressure']['wait_p95']))
write('primary_seed_metrics.csv',seedrows)
primary=[]
for a in (*ARMS,'mappo_message_off',*REFS):
 rr=[r for r in seedrows if r['algorithm']==a]
 primary.append(dict(algorithm=a,training_seeds=len(rr),**{k:st.fmean(r[k] for r in rr) for k in metrics},nominal_base_cost=st.fmean(r['nominal_base_cost'] for r in rr)))
write('primary_metrics.csv',primary)
comparisons=[(n,h['learner'],h['control'],h['metric'],True) for n,h in summary['hypotheses'].items()]+[
 ('D_global_vs_local','central_proposal','mappo_local','service_adjusted_cost',False),
 ('D_proposal_vs_autoregressive','central_proposal','central_ar','base_cost',False),
 ('D_message_off_vs_on','mappo_message_off','mappo_message','service_adjusted_cost',False),
 ('D_n4_ar_vs_deadline','central_ar','deadline_matching','base_cost',False)]
paired=[];contrasts=[]
for name,a,b,metric,confirmatory in comparisons:
 families=('n4_pressure',) if name=='D_n4_ar_vs_deadline' else PRIMARY
 rr=[]
 for s in cfg['train_seeds']:
  x=mean(a,s,metric,families);y=mean(b,-1 if b in REFS else s,metric,families)
  rr.append(dict(contrast=name,train_seed=s,learner=x,control=y,difference=x-y))
 ds=[r['difference'] for r in rr];avg=st.fmean(ds);crit=CRIT if confirmatory else 2.2621571628540993
 half=crit*st.stdev(ds)/math.sqrt(10);base=st.fmean(r['control'] for r in rr)
 result=dict(contrast=name,learner=a,control=b,metric=metric,confirmatory=confirmatory,learner_mean=st.fmean(r['learner'] for r in rr),control_mean=base,difference=avg,cost_change_percent=100*avg/base,seed_wins=sum(d<0 for d in ds),confidence=.9875 if confirmatory else .95,ci_low=avg-half,ci_high=avg+half,ci_low_scaled_percent=100*(avg-half)/base,ci_high_scaled_percent=100*(avg+half)/base)
 if confirmatory:
  h=summary['hypotheses'][name]
  checks.setdefault('raw_hypothesis_statistics_match',True)
  checks['raw_hypothesis_statistics_match'] &= abs(avg-h['mean_paired_difference'])<1e-10 and abs(avg/base+h['relative_reduction'])<1e-10 and all(abs(v-w)<1e-10 for v,w in zip([avg-half,avg+half],h['ci_9875'])) and result['seed_wins']==h['seed_wins']
  costs=[mean(a,s,'base_cost') for s in cfg['train_seeds']];controls=[mean(b,-1 if b in REFS else s,'base_cost') for s in cfg['train_seeds']]
  sla=all(fs[a,s,f]['wait_p95']<=4 for s in cfg['train_seeds'] for f in ['n3_pressure','n4_pressure']) and all(q95([int(r['max_pending_wait']) for r in machine if r['algorithm']==a and r['family']==f])<=4 for f in ['n3_pressure','n4_pressure'])
  gates=dict(practical_reduction=avg/base<=-.05,seed_wins=result['seed_wins']>=8,uncertainty=avg+half<0,pressure_wait_p95=sla,base_cost_guard=st.fmean(costs)<=1.1*st.fmean(controls),nominal_guard=st.fmean(mean(a,s,'base_cost',NOMINAL) for s in cfg['train_seeds'])<=1.1*st.fmean(mean(b,-1 if b in REFS else s,'base_cost',NOMINAL) for s in cfg['train_seeds']),reference_eligible=summary['reference_selection']['feasible_reference_exists'] if b in REFS else True)
  checks.setdefault('raw_hypothesis_gates_match',True);checks['raw_hypothesis_gates_match'] &= gates==h['gates'] and h['status']==('PASS' if all(gates.values()) else 'FAIL')
 contrasts.append(result);paired.extend(rr)
write('contrasts.csv',contrasts);write('paired_seed_metrics.csv',paired)
# Aggregate per-family machine distribution, including message-off omitted by runner summary.
aggregate=[]
for a in (*ARMS,'mappo_message_off',*REFS):
 for f in FAMILIES:
  rr=[r for r in family if r['algorithm']==a and r['family']==f]
  ms=[r for r in machine if r['algorithm']==a and r['family']==f]
  waits=[int(r['max_pending_wait']) for r in ms]
  aggregate.append(dict(algorithm=a,family=f,**{k:st.fmean(r[k] for r in rr) for k in metrics},wait_p95=q95(waits),max_wait=max(waits),wait_violation_fraction=st.fmean(w>4 for w in waits),terminal_failed_never_started_fraction=st.fmean(r['failed_at_terminal']=='True' and int(r['service_starts'])==0 for r in ms)))
write('family_metrics.csv',aggregate)
exact=read('exact_small_metrics.csv');ex=[]
for a in (*ARMS,*REFS):
 rr=[r for r in exact if r['algorithm']==a]
 ex.append(dict(algorithm=a,seeds=len(rr),**{k:st.fmean(float(r[k]) for r in rr) for k in ['exact_optimum','exact_policy_cost','optimality_gap','exact_selection_regret','exact_allocation_regret']},near_optimal_seeds=sum(float(r['optimality_gap'])<=1/6+1e-8 for r in rr)))
checks['exact_panel']=len(exact)==53 and all(abs(float(r['regret_decomposition_error']))<1e-8 for r in exact)
write('exact_metrics.csv',ex)
checks['all_checks_pass']=all(checks.values());assert checks['all_checks_pass'],checks
result=dict(run=raw.name,git_revision=manifest['git_revision'],runtime_minutes=(datetime.fromisoformat(manifest['completed_at'])-datetime.fromisoformat(manifest['started_at'])).total_seconds()/60,raw_checks=checks,counts=dict(models=50,test_learned=len(panels['episodes.csv']),test_reference=1200,test_message_off=len(panels['message_off_episodes.csv']),development_learned=1000,development_reference=60,training_records=sum(t['episodes'] for t in training.values()),training_physical_steps=sum(t['steps'] for t in training.values()),optimizer_steps=sum(int(rr[-1]['optimizer_steps']) for rr in pg.values()),machine_records=len(machine)),primary=primary,contrasts=contrasts,exact=ex)
dump('statistics.json',result)
inventory=[]
for p in sorted(raw.rglob('*')):
 if p.is_file():inventory.append(dict(path=str(p.relative_to(raw)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
write('raw_inventory.csv',inventory)
print(json.dumps(result,indent=2))
