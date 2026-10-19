import csv,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
raw,out=map(Path,sys.argv[1:])
s=json.loads((out/'statistics.json').read_text())
cs={r['contrast']:r for r in s['contrasts']}
order=['H_learning','H_multiagent','H_critic','H_message']
labels=['Central PPO vs deadline (base)','MAPPO + message vs deadline (base)','Central critic vs IPPO (priced)','Messages vs no messages (priced)']
fig,ax=plt.subplots(figsize=(10.2,4.7))
for i,n in enumerate(order):
 r=cs[n];v=r['cost_change_percent'];low=r['ci_low_scaled_percent'];high=r['ci_high_scaled_percent']
 ax.errorbar(v,3-i,xerr=[[v-low],[high-v]],fmt='o',color='#a43a30' if low>0 else '#22587c',capsize=4,lw=2,ms=6)
 ax.text(high+.18,3-i,f"{v:+.2f}%  ({r['seed_wins']}/10 wins)",va='center',fontsize=10)
ax.axvline(0,color='#89939f',lw=1)
ax.axvline(-5,color='#427d54',ls='--',lw=1.3,label='Target: at least 5% lower cost')
ax.set_yticks(range(4),labels[::-1]);ax.set_xlim(-5.6,8.4);ax.set_ylim(-.6,3.6)
ax.set_xlabel('Paired cost difference / mean control cost (%) — lower is better')
ax.set_title('No confirmatory learning, critic or message benefit',loc='left',fontweight='bold',pad=14)
ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
ax.legend(loc='lower right',frameon=False,fontsize=9)
fig.text(.04,.025,'Bars: 98.75% paired t intervals, 10 training seeds, four locked contrasts.\nIntervals are cost-difference intervals scaled by the control mean; evaluation panel is fixed.',fontsize=9,color='#536070')
fig.tight_layout(rect=[0,.105,1,1]);fig.savefig(out/'confirmatory_contrasts.png',dpi=180);fig.savefig(out/'confirmatory_contrasts.pdf');plt.close(fig)
# Train curves describe progress only; heterogeneous training mix differs from test panel.
with (raw/'training_progress.csv').open() as f:rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(9.2,4.5))
colors=['#22587c','#b07132','#7972a2','#358d82','#ae4650']
for a,color in zip(['central_ar','central_proposal','ippo_local','mappo_local','mappo_message'],colors):
 x=sorted({int(r['physical_steps']) for r in rows if r['algorithm']==a})
 y=[np.mean([float(r['mean_episode_cost']) for r in rows if r['algorithm']==a and int(r['physical_steps'])==t]) for t in x]
 # Ten-update trailing moving average, descriptive and explicitly labeled.
 smooth=[np.mean(y[max(0,i-9):i+1]) for i in range(len(y))]
 ax.plot(np.array(x)/1000,smooth,label=a,color=color,lw=1.7)
ax.set_xlabel('Training transitions per model (thousands)');ax.set_ylabel('Mean priced training episode cost')
ax.set_title('Learning occurs; lower training cost is not a test win',loc='left',fontweight='bold')
ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=9)
fig.text(.07,.02,'10 training seeds averaged; trailing 10-update smoothing. Training includes N2/N3/N4 and differs from the primary test panel.',fontsize=8)
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'training_curves.png',dpi=180);plt.close(fig)
