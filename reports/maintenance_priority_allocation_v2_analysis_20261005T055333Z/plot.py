import csv,json,os
from pathlib import Path
p=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(p/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
s=json.loads((p/'statistics.json').read_text())
with (p/'family_metrics.csv').open() as f:rows=list(csv.DictReader(f))
fig,(left,right)=plt.subplots(1,2,figsize=(11,5.2),layout='constrained')
for i,r in enumerate(s['primary']):
    delta=r['mean_delta'];lo,hi=r['ci'];left.errorbar(delta,i,xerr=[[delta-lo],[hi-delta]],fmt='o',color='#166b91',capsize=5,linewidth=2)
    left.scatter(-.05*r['control_cost'],i,marker='D',color='#b87320',s=50,zorder=3,label='5% practical threshold' if i==0 else None)
left.axvline(0,color='gray',linestyle='--',linewidth=1)
left.set_yticks([0,1],['vs fixed priority','vs fixed allocation']);left.set_xlim(-9,1.8);left.set_ylim(1.6,-.7)
left.set_xlabel('Cost difference: learned both minus control\n97.5% paired CI, Bonferroni for two contrasts')
left.set_title('Locked primary panel: both gates fail',fontweight='bold');left.legend(loc='lower left',fontsize=8);left.grid(axis='x',alpha=.15)
modes=['learned_both','fixed_priority','fixed_allocation','risk_skill_rule'];labels=['Learn both','Fixed\npriority','Fixed\nallocation','Rule']
rr=[next(r for r in rows if r['algorithm']==m and r['family']=='n4_pressure') for m in modes]
components=[('Maintenance',[float(r['maintenance_cost']) for r in rr],'#75a8ab'),('In-service loss',[6*float(r['service_ticks']) for r in rr],'#9ca9c7'),('Failed waiting loss',[6*float(r['failed_waiting_ticks']) for r in rr],'#d89664'),('Failure events',[float(r['failure_cost']) for r in rr],'#a3515c')]
bottom=np.zeros(4)
for label,values,color in components:
    right.bar(np.arange(4),values,bottom=bottom,label=label,color=color,width=.65);bottom+=values
for i,r in enumerate(rr):right.text(i,bottom[i]+4,f'{bottom[i]:.1f}',ha='center',fontsize=9)
right.set_xticks(np.arange(4),labels);right.set_ylim(0,290);right.set_ylabel('Mean total cost / episode');right.set_title('N=4 pressure: less failure, more waiting',fontweight='bold');right.legend(loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=8,ncol=2,frameon=False)
for ax in (left,right):ax.spines[['top','right']].set_visible(False)
fig.savefig(p/'cost_and_waiting.png',dpi=180);fig.savefig(p/'cost_and_waiting.svg');plt.close(fig)
