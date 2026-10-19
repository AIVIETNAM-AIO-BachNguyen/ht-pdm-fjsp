import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
S=json.loads((P.parent.parent/'lab_results/maintenance_heterogeneous_marl_v1_full_cpu_20261006T150059Z/summary.json').read_text())
A=json.loads((P/'independent_statistics.json').read_text())['primary']
labels=['Selected rule','Central full matching','Central fixed allocation','Machine-agent PPO','Joint rollout']
controllers=['adaptive_isolated_matching','central_matching_ppo','central_fixed_allocation_ppo','machine_cooperative_ppo','joint_rollout']
colors=['#72808c','#2563a6','#278b62','#cf574c','#9b85bd']
values=[S['mechanisms']['aggregate'][f'specialized_H12:{a}']['objective'] for a in controllers]
errors=[]
for a in controllers:
 r=next((x for x in A if x['left']==a and x['right']=='adaptive_isolated_matching'),None)
 errors.append((r['ci'][1]-r['ci'][0])/2 if r else 0)
fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.7,1]})
y=np.arange(len(labels));ax.barh(y,values,color=colors,height=.62,xerr=errors,capsize=3,alpha=.92)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,218)
for i,v in enumerate(values):ax.text(v+2.5,i,f'{v:.2f}',va='center',fontsize=10)
ax.set_xlabel('Mean priced cost (lower is better)');ax.set_title('Primary: specialized technicians, N4/K2/H12')
ax.text(0,-.16,'Learned arms: 10 training replicas × 500 shared evaluation episodes.\nError bars: 95% training-replica CI; controls have no training replicas.',transform=ax.transAxes,fontsize=9,va='top')
base=[S['mechanisms']['aggregate'][f'nominal_H12:{a}']['base_cost'] for a in controllers[1:4]]
bx.bar(np.arange(3),base,color=colors[1:4],width=.55)
bx.set_xticks(np.arange(3),['Central\nfull','Central\nfixed','Machine\nagents']);bx.set_ylim(0,235)
for i,v in enumerate(base):bx.text(i,v+3,f'{v:.2f}',ha='center')
bx.axhline(168.048*1.1,color='#707070',linestyle='--',linewidth=1,label='Nominal guard: +10%')
bx.set_title('Nominal: homogeneous N4/K4/H12');bx.set_ylabel('Mean base cost');bx.legend(loc='upper left',fontsize=8)
for x in (ax,bx):x.spines[['top','right']].set_visible(False);x.grid(axis='x' if x==ax else 'y',alpha=.14);x.set_axisbelow(True)
fig.suptitle('Heterogeneous maintenance v1 — allocation gain small; MARL head has worker-symmetry limitation',fontsize=12)
fig.tight_layout(rect=[0,.07,1,.95]);fig.savefig(P/'primary_cost_and_nominal.png',dpi=200,bbox_inches='tight');fig.savefig(P/'primary_cost_and_nominal.pdf',bbox_inches='tight')
plt.close(fig)
