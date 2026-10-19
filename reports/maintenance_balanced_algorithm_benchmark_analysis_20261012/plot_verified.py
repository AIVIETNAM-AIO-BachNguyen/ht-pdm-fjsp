"""Plot verified numerical outputs; no inference, training or simulator run."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser(); parser.add_argument('directory',type=Path)
args=parser.parse_args(); out=args.directory
analysis=json.loads((out/'analysis.json').read_text())
intervals=json.loads((out/'registered_intervals_verified.json').read_text())
assert analysis['verification']['registered_statistics_recalculation']=='PASS'
algorithms=('ppo','mappo','iql','qmix')
colors={'ppo':'#2468b4','mappo':'#e18a28','iql':'#b84851','qmix':'#418965'}
paths=[out/'cost_vs_h6.png',out/'development_learning_curves.png']
assert not any(p.exists() for p in paths)
fig,ax=plt.subplots(figsize=(8,3.6))
for i,a in enumerate(algorithms):
    v=intervals[f'{a}_cost_vs_isolated_h6']
    ax.errorbar(100*v['estimate'],i,xerr=[[100*(v['estimate']-v['lower'])],[100*(v['upper']-v['estimate'])]],
        fmt='o',capsize=5,color=colors[a],markersize=7)
ax.axvline(0,color='#555',lw=1); ax.axvline(1,color='#999',lw=1,ls=':')
ax.set_yticks(range(4),[a.upper() for a in algorithms]); ax.invert_yaxis()
ax.set_xlabel('Cost reduction vs isolated h6 (%) — positive is better')
ax.set_title('Registered crossed-bootstrap intervals (99.8214% each)')
ax.grid(axis='x',alpha=.15); fig.tight_layout(); fig.savefig(paths[0],dpi=180); plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
steps=[0,122880,245760,368640,491520]
for ax,a in zip(axes.ravel(),algorithms):
    curves=np.asarray([[analysis['development_curves'][f'{a}:{t}:{k}'] for k in steps] for t in range(46000000,46000005)])
    for curve in curves: ax.plot(np.asarray(steps)/1000,curve,color=colors[a],lw=1,alpha=.3)
    ax.plot(np.asarray(steps)/1000,curves.mean(0),color=colors[a],lw=2.5,marker='o')
    ax.set_title(a.upper()); ax.set_ylabel('Development H+L cost'); ax.set_xlabel('Learned transitions (thousands)'); ax.grid(alpha=.15)
fig.suptitle('Development only: thin lines = five training replicas; thick line = mean',fontsize=11)
fig.tight_layout(); fig.savefig(paths[1],dpi=180); plt.close(fig)
bindings={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in paths}
(out/'plot_manifest.json').write_text(json.dumps(dict(status='COMPLETED',
    source_analysis_sha256=hashlib.sha256((out/'analysis.json').read_bytes()).hexdigest(),
    plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),figures=bindings,
    note='Numerical analysis finished successfully; its optional plot phase lacked matplotlib in the temporary learner venv. These figures use the existing main venv without rerunning inference.'),indent=2)+'\n')
print(json.dumps({'status':'COMPLETED','figures':list(bindings)}))
