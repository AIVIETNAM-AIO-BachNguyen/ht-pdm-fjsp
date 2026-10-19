import csv,hashlib,itertools,json,math,statistics,sys,shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

run=Path(sys.argv[1]);out=Path(sys.argv[2])
ARMS=('local_cost','context_cost','local_wait','context_wait')
PRIMARY=('n3_nominal','n3_pressure','n4_nominal','n4_pressure')
PRESSURE=('n3_pressure','n4_pressure');NOMINAL=('n3_nominal','n4_nominal')
FAMILIES=(*PRIMARY,'n5_pressure','n5_k3_sparse','small','n4_pressure_long')
manifest=json.loads((run/'manifest.json').read_text());original=json.loads((run/'summary.json').read_text())
assert manifest['status']=='COMPLETED' and all(manifest['audits'].values())
assert manifest['git_revision']=='e2ec53e3239881aaf2d7db57aa7b72c64342165f'
seeds=manifest['settings']['train_seeds'];evals=manifest['settings']['evaluation_seeds']
assert seeds==list(range(122000,122010)) and evals==list(range(124000,124050))

def read(name):
    with (run/name).open() as f:return list(csv.DictReader(f))

def save(name,rows):
    with (out/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def pvalue_t(t,df=9):
    if math.isinf(t):return 0.
    a=math.atan(abs(t)/math.sqrt(df));b=math.pi/2;n=4096;h=(b-a)/n
    f=lambda x:math.cos(x)**(df-1)
    integral=h/3*(f(a)+f(b)+sum((4 if i%2 else 2)*f(a+i*h) for i in range(1,n)))
    return 2*math.gamma((df+1)/2)/math.gamma(df/2)/math.sqrt(math.pi)*integral
assert abs(pvalue_t(2.685010847)-.025)<1e-9

def paired(values,controls,learners,critical=2.685010847):
    mean=statistics.fmean(values);sd=statistics.stdev(values);se=sd/math.sqrt(len(values));radius=critical*se
    control=statistics.fmean(controls);learned=statistics.fmean(learners)
    t=mean/se if se>0 else (0. if mean==0 else math.copysign(math.inf,mean))
    signflip=sum(abs(statistics.fmean(s*d for s,d in zip(sign,values)))>=abs(mean)-1e-10
                 for sign in itertools.product((-1,1),repeat=len(values)))/2**len(values)
    return dict(control_mean=control,learned_mean=learned,delta=mean,
                relative_reduction=-mean/control if control else None,improving_seeds=sum(x< -1e-8 for x in values),
                ci=[mean-radius,mean+radius],critical=critical,df=len(values)-1,t=t,
                p_two_sided=pvalue_t(t),p_bonferroni_two=min(1.,2*pvalue_t(t)),
                signflip_p_two_sided_descriptive=signflip,paired_sd=sd,paired_effect=mean/sd if sd else None,
                seed_deltas=values)

episodes=read('episodes.csv');refs=read('reference_episodes.csv')
keys={(r['algorithm'],int(r['train_seed']),r['family'],int(r['eval_seed'])) for r in episodes}
assert keys=={(a,s,f,e) for a in ARMS for s in seeds for f in FAMILIES for e in evals}
assert len(keys)==len(episodes)==16000
numeric=[k for k in episodes[0] if k not in ('algorithm','train_seed','family','eval_seed','technician_busy_ticks')]
groups=defaultdict(list)
for row in episodes:
    groups[row['algorithm'],int(row['train_seed']),row['family']].append(row)
for key,rr in groups.items():assert len(rr)==50
idx={(a,s,f,k):statistics.fmean(float(r[k]) for r in rr) for (a,s,f),rr in groups.items() for k in numeric}

def panel(a,s,k,fams):return statistics.fmean(idx[a,s,f,k] for f in fams)
def overall(a,k,fams):return statistics.fmean(panel(a,s,k,fams) for s in seeds)

comparisons={};pair_rows=[]
for name,learner,control,metric,families,crit in (
    ('H_context','context_cost','local_cost','base_cost',PRIMARY,2.685010847),
    ('H_wait','context_wait','context_cost','overdue_ticks_per_machine',PRESSURE,2.685010847),
    ('waiting_economic_cost','context_wait','context_cost','base_cost',PRIMARY,2.262157163),
    ('waiting_nominal_cost','context_wait','context_cost','base_cost',NOMINAL,2.262157163),
    ('waiting_with_local_head','local_wait','local_cost','overdue_ticks_per_machine',PRESSURE,2.262157163),
    ('context_given_waiting','context_wait','local_wait','base_cost',PRIMARY,2.262157163),
    ('long_horizon_waiting','context_wait','context_cost','overdue_ticks_per_machine',('n4_pressure_long',),2.262157163)):
    av=[panel(learner,s,metric,families) for s in seeds];bv=[panel(control,s,metric,families) for s in seeds]
    dd=[a-b for a,b in zip(av,bv)]
    comparisons[name]=dict(learner=learner,control=control,metric=metric,
                           confirmatory=name in ('H_context','H_wait'),**paired(dd,bv,av,crit))
    if name not in ('H_context','H_wait'):
        comparisons[name].pop('p_bonferroni_two')
    pair_rows.extend(dict(contrast=name,train_seed=s,learner=a,control=b,delta=d) for s,a,b,d in zip(seeds,av,bv,dd))
for name in ('H_context','H_wait'):
    expected=original['hypotheses'][name];actual=comparisons[name]
    for k1,k2 in [('control_mean','control_mean'),('learned_mean','learned_mean'),('delta','mean_delta'),
                  ('relative_reduction','relative_reduction'),('improving_seeds','improving_seeds')]:
        assert abs(actual[k1]-expected[k2])<1e-10,(name,k1)
    assert max(abs(a-b) for a,b in zip(actual['ci'],expected['ci97_5_bonferroni']))<1e-10

machines=read('machine_metrics.csv');refmachines=read('reference_machine_metrics.csv')
mgroups=defaultdict(list)
for r in machines+refmachines:
    if r['family']!='development':mgroups[r['algorithm'],r['family']].append(r)
family_rows=[];machine_rows=[]
for arm in (*ARMS,'risk_skill_rule'):
    for f in FAMILIES:
        rr=mgroups[arm,f];waits=sorted(int(r['max_pending_wait']) for r in rr)
        p95=waits[math.ceil(.95*len(waits))-1]
        violation=sum(w>4 for w in waits);never=sum(r['failed_at_terminal']=='True' and int(r['service_starts'])==0 for r in rr)
        machine_rows.append(dict(algorithm=arm,family=f,machine_records=len(rr),p95_max_wait=p95,
              maximum_wait=max(waits),wait_violation_records=violation,wait_violation_fraction=violation/len(rr),
              terminal_failed_never_serviced_records=never,terminal_failed_never_serviced_fraction=never/len(rr)))
        assert len(rr)==(2 if f=='small' else int(f[1]))*50*(1 if arm=='risk_skill_rule' else 10)
        metric={k: (statistics.fmean(float(r[k]) for r in refs if r['family']==f)
                    if arm=='risk_skill_rule' else overall(arm,k,(f,))) for k in numeric}
        family_rows.append(dict(algorithm=arm,family=f,**metric,p95_max_wait=p95,maximum_wait=max(waits)))
        if arm in ARMS:
            assert p95==original['pooled_p95_max_wait'][arm][f]
            assert abs(metric['wait_violation_machine_fraction']-violation/len(rr))<1e-10
            assert abs(metric['terminal_failed_never_serviced_fraction']-never/len(rr))<1e-10

primary_rows=[]
for arm in (*ARMS,'risk_skill_rule'):
    rr=[r for r in family_rows if r['algorithm']==arm]
    primary_rows.append(dict(algorithm=arm,base_cost=statistics.fmean(r['base_cost'] for r in rr if r['family'] in PRIMARY),
        service_adjusted_cost=statistics.fmean(r['service_adjusted_cost'] for r in rr if r['family'] in PRIMARY),
        pressure_overdue_per_machine=statistics.fmean(r['overdue_ticks_per_machine'] for r in rr if r['family'] in PRESSURE),
        nominal_base_cost=statistics.fmean(r['base_cost'] for r in rr if r['family'] in NOMINAL)))
rule_primary=primary_rows[-1]['base_cost']
for a in ARMS:
    av=[panel(a,s,'base_cost',PRIMARY) for s in seeds]
    comparisons[a+'_vs_rule']=dict(confirmatory=False,**paired([x-rule_primary for x in av],[rule_primary]*10,av,2.262157163))
    comparisons[a+'_vs_rule'].pop('p_bonferroni_two')

# Re-evaluate precisely locked success criteria; no outcome-dependent amendment.
ctx=comparisons['H_context'];wait=comparisons['H_wait']
nom_c=overall('context_cost','base_cost',NOMINAL);nom_l=overall('local_cost','base_cost',NOMINAL)
nom_w=overall('context_wait','base_cost',NOMINAL)
primary_c=overall('context_cost','base_cost',PRIMARY);primary_w=overall('context_wait','base_cost',PRIMARY)
ctx_gate=dict(practical_5_percent=ctx['relative_reduction']>=.05,wins_8_of_10=ctx['improving_seeds']>=8,
              uncertainty=ctx['ci'][1]<0,nominal=nom_c<=1.1*nom_l)
wait_gate=dict(practical_50_percent=wait['relative_reduction']>=.5,wins_8_of_10=wait['improving_seeds']>=8,
               uncertainty=wait['ci'][1]<0,nominal=nom_w<=1.1*nom_c,economic=primary_w<=1.1*primary_c,
               p95_bound=all(next(r for r in machine_rows if r['algorithm']=='context_wait' and r['family']==f)['p95_max_wait']<=4 for f in PRESSURE))
assert all(ctx_gate.values())==original['hypotheses']['H_context']['passed']
assert all(wait_gate.values())==original['hypotheses']['H_wait']['passed']

logs=read('training_progress.csv');loggroups=defaultdict(list)
for r in logs:loggroups[r['algorithm'],int(r['train_seed'])].append(r)
assert len(logs)==10000 and len(loggroups)==40
curve=[]
for a in ARMS:
    for phase in ('first20','last20'):
        rr=[r for s in seeds for r in (loggroups[a,s][:20] if phase=='first20' else loggroups[a,s][-20:])]
        curve.append(dict(algorithm=a,phase=phase,**{k:statistics.fmean(float(r[k]) for r in rr) for k in
            ('mean_episode_cost','mean_matching_size','entropy','value_loss','gradient_norm')}))

exact=read('exact_small_metrics.csv');exact_rows=[]
for a in (*ARMS,'risk_skill_rule'):
    rr=[r for r in exact if r['algorithm']==a]
    exact_rows.append(dict(algorithm=a,rows=len(rr),**{k:statistics.fmean(float(r[k]) for r in rr) for k in
        ('exact_policy_cost','exact_optimum','optimality_gap','exact_selection_regret','exact_allocation_regret')},
        close_optimal_seeds=sum(float(r['optimality_gap'])<.2 for r in rr)))

hashrows=[]
for p in sorted(run.rglob('*')):
    if p.is_file():
        h=hashlib.sha256()
        with p.open('rb') as f:
            while block:=f.read(1024*1024):h.update(block)
        hashrows.append(dict(file=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=h.hexdigest()))
save('raw_file_sha256.csv',hashrows)
save('family_metrics.csv',family_rows);save('machine_tail_metrics.csv',machine_rows);save('primary_metrics.csv',primary_rows)
save('paired_seed_recomputed.csv',pair_rows);save('learning_curves.csv',curve);save('exact_aggregates.csv',exact_rows)
result=dict(verification_status='ANALYZED',raw_run=str(run),git_revision=manifest['git_revision'],
    primary=primary_rows,contrasts=comparisons,gates=dict(H_context=ctx_gate,H_wait=wait_gate),
    nominal_cost_increase=nom_w/nom_c-1,primary_cost_increase=primary_w/primary_c-1,
    elapsed_seconds=(datetime.fromisoformat(manifest['completed_at'])-datetime.fromisoformat(manifest['started_at'])).total_seconds(),
    locked_summary_recomputed=True,exact=exact_rows,learning_curves=curve,
    fallacy_scan=dict(coverage='11/11',simpson='family effects reported; nominal and pressure not substituted for primary',
        ecological='CI unit training seed; machine/tick summaries are descriptive',berkson='synthetic benchmark population; no selective seed exclusion',
        collider='no post-treatment covariate adjustment; actionable-deferral ratios are descriptive',base_rate='all machine-record denominators reported',
        regression_to_mean='fresh matched seed panels; original poor run not a within-panel control',survivorship='40/40 models and all test keys; no best checkpoint/seed selection',
        look_elsewhere='two locked Bonferroni tests; all other contrasts descriptive',forking_paths='unchanged practical margins and p95/economic gates',
        causal='factor intervention evaluated within this simulator/budget; no MARL-necessity claim',reverse_causality='context and waiting factors assigned before training'),
    limits=['No independent training rerun or independent simulator reproduction',
            'Paired-t uncertainty conditions on this evaluation panel and uses ten training seeds; normality unverified, signflip sensitivity descriptive',
            '4-tick limit and 12/tick price synthetic; soft cost and pooled p95 do not establish a hard bound',
            'Long horizon OOD and base-only N2 exact results are secondary'])
(out/'statistics.json').write_text(json.dumps(result,indent=2)+'\n')
shutil.copy2(__file__,out/'analyze.py')
print(json.dumps(dict(primary=primary_rows,contrasts={k:v for k,v in comparisons.items() if k in ('H_context','H_wait','waiting_economic_cost','context_wait_vs_rule','waiting_with_local_head','context_given_waiting')},gates=result['gates'],exact=exact_rows,elapsed_seconds=result['elapsed_seconds']),indent=2))
