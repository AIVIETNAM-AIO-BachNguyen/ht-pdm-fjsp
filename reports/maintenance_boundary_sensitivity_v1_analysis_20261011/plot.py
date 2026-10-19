"""Render explanatory scientific figures from independently audited statistics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
s=json.loads((OUT/'independent_statistics.json').read_text())
assert not s['physical_replay_pending'] and s['physical_replay']=='PASS'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(12.5,5),gridspec_kw={'width_ratios':[1,1.15]})
fig.subplots_adjust(left=.07,right=.98,top=.79,bottom=.18,wspace=.48)
r=s['means']['skill_mask:12']; names=['best_rule','fixed_zero','fixed_tail']
base=np.array([r[n]['base_objective'] for n in names]);tail=np.array([r[n]['tail_objective'] for n in names])
x=np.arange(3)
ax.bar(x,base,color='#4682b4',label='Before cutoff: 12 ticks')
ax.bar(x,tail,bottom=base,color='#edb45a',label='Shared-rule continuation: 12 ticks')
for i in range(3):
    ax.text(i,base[i]/2,f'{base[i]:.1f}',ha='center',va='center',color='white',fontweight='bold')
    ax.text(i,base[i]+tail[i]/2,f'{tail[i]:.1f}',ha='center',va='center')
    ax.text(i,base[i]+tail[i]+7,f'{base[i]+tail[i]:.1f}',ha='center',fontweight='bold')
ax.set_xticks(x,['Best rule','Fixed:\nzero terminal','Fixed:\n12-tick tail'])
ax.set_ylabel('Mean priced cost (simulation units)')
ax.set_ylim(0,520);ax.set_title('H12: include the next 12 ticks',fontweight='bold',fontsize=12,pad=10)
ax.legend(loc='upper center',frameon=False,fontsize=9)

labels=['Planner over\nbest rule','Flexible over\nfixed allocator']; contrasts=['interior_headroom','interior_assignment']
for i,label in enumerate(contrasts):
    c=s['contrasts'][f'skill_mask:48:{label}'];point=c['relative_improvement']*100
    low,high=np.array(c['relative_confidence_interval'])*100
    bx.errorbar(point,1-i,xerr=[[point-low],[high-point]],fmt='o',markersize=8,capsize=5,color=['#4682b4','#a66b2c'][i],linewidth=2)
    bx.text(-.55,1-i+.18,f'{point:.3f}%  [{low:.3f}, {high:.3f}]',ha='left',va='bottom',fontsize=10)
bx.axvline(0,color='#bbb',linewidth=1)
bx.axvline(1,color='#555',linestyle='--',linewidth=1.3)
bx.text(1,-.42,'1% research margin',ha='center',va='center',fontsize=9,color='#555')
bx.set_yticks([1,0],labels);bx.set_ylim(-.6,1.6);bx.set_xlim(-.65,3.05)
bx.set_xlabel('Relative improvement in interior cost (%)')
bx.set_title('H48: interior cost improvement',fontweight='bold',fontsize=12,pad=10)
fig.suptitle('Boundary sensitivity — primary skill-mask condition',fontsize=15,fontweight='bold',y=.98)
fig.text(.5,.015,'H48 interior = ticks 12–35. Intervals: 98.33% paired-profile bootstrap, conditional on 8 shocks/profile.',ha='center',fontsize=9,color='#444')
fig.savefig(OUT/'boundary_summary.png',dpi=180)
fig.savefig(OUT/'boundary_summary.pdf')
plt.close(fig)
