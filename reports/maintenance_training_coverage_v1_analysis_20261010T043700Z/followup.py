from pathlib import Path
from collections import Counter,defaultdict
import csv,json,math,statistics,itertools
P=Path(__file__).resolve().parent
RUN=Path('/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_training_coverage_v1_full_cpu_20261009T044236Z')
d=json.loads((P/'independent_analysis.json').read_text());cfg=json.loads((RUN/'benchmark_config.json').read_text())
def rows(f):
 with (RUN/f).open() as s:yield from csv.DictReader(s)
refs=['risk_matching','deadline_matching','lookahead_matching','adaptive_isolated_matching'];arms=cfg['arms'];regs=('legacy','covered')
# Confirm development selection solely from complete declared raw development grid.
expected={(reg,a,s,cs,c,shock) for reg,a,seeds in [(r,a,cfg['train_seeds']) for r,a in itertools.product(regs,arms)]+[('reference',a,[-1]) for a in refs] for s,cs,c,shock in itertools.product(seeds,cfg['development_cohorts'],cfg['conditions'],cfg['development_seeds'])}
seen=set();sums=Counter();ns=Counter()
for r in rows('development_episodes.csv'):
 k=(r['training_regime'],r['controller'],int(r['train_seed']),int(r['cohort']),r['condition'],int(r['seed']))
 assert k in expected and k not in seen and int(r['horizon'])==12
 assert int(r['checkpoint_step'])==(491520 if k[0]!='reference' else -1)
 seen.add(k)
 if k[0]=='reference':sums[(k[4],k[1])]+=float(r['objective']);ns[(k[4],k[1])]+=1
assert seen==expected and len(seen)==40960
rule_means={c:{a:sums[(c,a)]/ns[(c,a)] for a in refs} for c in cfg['conditions']}
selected={c:min(refs,key=lambda a:(rule_means[c][a],a)) for c in cfg['conditions']}
assert selected==json.loads((RUN/'reference_selection.json').read_text())['selected']
# Supplemental exact sign flips: diagnostic only, symmetry assumption; do not substitute gate.
robust={}
for k,v in d['contrasts'].items():
 if not v['primary']:continue
 x=v['seed_differences'];obs=abs(statistics.mean(x))
 p=sum(abs(sum(a*b for a,b in zip(signs,x))/10)>=obs-1e-10 for signs in itertools.product((-1,1),repeat=10))/1024
 loo=[statistics.mean(x[:i]+x[i+1:]) for i in range(10)]
 robust[k]=dict(two_sided_exact_sign_flip_p=p,bonferroni4_p=min(1,4*p),leave_one_replica_out_mean_range=[min(loo),max(loo)],
    max_difference=max(x),max_difference_seed=cfg['train_seeds'][x.index(max(x))],min_difference=min(x),
    caveat='Posthoc sensitivity, sign-flip needs symmetric null paired differences; no deleted replica, no replacement confirmatory verdict.')
# Raw seed-level means and failure/wait component tradeoffs.
acc=defaultdict(Counter);n=Counter()
for r in rows('episodes.csv'):
 if r['controller'] not in arms:continue
 k=(r['training_regime'],r['controller'],r['condition'],int(r['horizon']),int(r['train_seed']))
 n[k]+=1
 acc[k].update({m:float(r[m]) for m in ('objective','waiting_cost','waiting_violation','terminal_pending','failure_cost','unavailability_cost','failures')})
perseed={':'.join(map(str,k)):{m:v/n[k] for m,v in total.items()} for k,total in acc.items()}
seed=180005;a='cooperative_full_ppo'
case={}
for c in cfg['conditions']:
 for h in cfg['horizons']:
  old=perseed[f'legacy:{a}:{c}:{h}:{seed}'];new=perseed[f'covered:{a}:{c}:{h}:{seed}']
  case[f'{c}_H{h}']=dict(legacy=old,covered=new,difference=new['objective']-old['objective'])
output=dict(development_rows=40960,selected_rules=selected,development_rule_means=rule_means,primary_posthoc_sensitivity=robust,seed_means=perseed,full_seed180005_case=case)
(P/'supplemental_audit.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'robustness':robust,'seed180005':case},indent=2))
