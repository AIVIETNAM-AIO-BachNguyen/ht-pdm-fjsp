import csv,json,os
from pathlib import Path
p=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(p/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
s=json.loads((p/'statistics.json').read_text())
with (p/'family_metrics.csv').open() as f:families=list(csv.DictReader(f))
with (p/'paired_seed_recomputed.csv').open() as f:paired=list(csv.DictReader(f))
fig,(a,b,c)=plt.subplots(1,3,figsize=(14.2,4.6),layout='constrained')
r=s['contrasts']['H_context'];d=r['delta'];lo,hi=r['ci']
a.errorbar(d,0,xerr=[[d-lo],[hi-d]],fmt='o',color='#196d91',capsize=5,linewidth=2)
a.scatter(-.05*r['control_mean'],0,color='#b06c27',marker='D',s=55,label='Required 5% reduction')
a.axvline(0,color='#888',ls='--',lw=1);a.set_ylim(-.7,.7);a.set_yticks([])
a.set_xlim(-8,1.5);a.set_xlabel('Base cost difference: context minus local\n97.5% paired CI, 10 training seeds')
a.set_title('Context benefit: gate fails',fontweight='bold');a.legend(loc='lower left',fontsize=8)
a.grid(axis='x',alpha=.15)
rr=[r for r in paired if r['contrast']=='H_wait']
for r in rr:
    b.plot([0,1],[float(r['control']),float(r['learner'])],'-o',color='#327f80',alpha=.55,lw=1,ms=4)
b.set_xticks([0,1],['Cost only','Cost + overdue price']);b.set_xlim(-.15,1.15);b.set_ylim(-.035,1.42)
b.set_ylabel('Overdue waiting ticks / machine / episode\nEqual mean of N3 and N4 pressure families')
b.set_title('Waiting benefit: gate passes',fontweight='bold')
b.text(.5,1.34,'99.55% less overdue waiting; 10/10 wins',ha='center',fontsize=9)
b.grid(axis='y',alpha=.15)
colors={'local_cost':'#8c929d','context_cost':'#216f9e','local_wait':'#99bfa4','context_wait':'#287c5b','risk_skill_rule':'#bc772d'}
labels={'local_cost':'Local / cost','context_cost':'Context / cost','local_wait':'Local / wait','context_wait':'Context / wait','risk_skill_rule':'Rule'}
offsets={'local_cost':(6,-17),'context_cost':(6,9),'local_wait':(7,-20),'context_wait':(-28,10),'risk_skill_rule':(-37,-18)}
for r in families:
    if r['family']!='n4_pressure':continue
    arm=r['algorithm'];x=float(r['base_cost']);y=float(r['p95_max_wait'])
    c.scatter(x,y,s=55,color=colors[arm],zorder=3)
    c.annotate(labels[arm],(x,y),textcoords='offset points',xytext=offsets[arm],fontsize=8)
c.axhline(4,color='#aa4b43',ls='--',lw=1,label='Waiting threshold (4 ticks)')
c.set_ylim(0,11);c.set_xlim(234.7,250);c.set_xlabel('Mean base economic cost / episode')
c.set_ylabel('Pooled p95 maximum pending wait (ticks)')
c.set_title('N4 pressure: cost vs service',fontweight='bold');c.grid(alpha=.15)
c.legend(loc='upper right',fontsize=8)
for ax in (a,b,c):ax.spines[['top','right']].set_visible(False)
fig.savefig(p/'context_waiting_results.png',dpi=180)
fig.savefig(p/'context_waiting_results.svg')
plt.close(fig)
