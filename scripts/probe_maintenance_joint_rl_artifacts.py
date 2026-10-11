from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from ht_pdm_fjsp import maintenance_joint_rl as e

torch.set_num_threads(1)
cfg=e.settings('smoke');cfg.update(horizon=48,tail_ticks=12,hidden=32,evaluation_batch=32)
out=Path('artifacts')/('maintenance_joint_rl_v1_artifact_probe_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
out.mkdir()
panels,_=e.build_panels(cfg)
journals={k:e.CountingJournal(out/f'{k}.csv') for k in ['development_episodes','episodes.partial','decisions','requests','carryover','coordination']}
try:
    for algorithm in e.ARMS:
        model=e.model_for(cfg,cfg['train_seeds'][0],algorithm)
        e.evaluate(panels,cfg,'test',algorithm,cfg['train_seeds'][0],model,0,journals=journals)
    e.evaluate(panels,cfg,'test','best_rule',rule='deadline_matching',journals=journals)
finally:
    for j in journals.values():j.close()
torch.save(model.state_dict(),out/'model_size.pt')
full=e.counts(e.settings('full'))
ratio=full['test_episodes']/12
sizes={p.name:p.stat().st_size for p in out.iterdir() if p.is_file()}
projected={name:size*(full['logged_test_decisions']/720 if name=='decisions.csv' else ratio)
    for name,size in sizes.items() if name not in ['model_size.pt','development_episodes.csv']}
projected['episodes.csv']=projected['episodes.partial.csv']
projected['development_episodes.csv']=sizes['episodes.partial.csv']*(full['development_episodes']/12)
projected['checkpoints_and_selected_models']=sizes['model_size.pt']*20*6
bench=sorted(Path('artifacts').glob('maintenance_joint_rl_v1_engineering_benchmark_*/training_episodes.csv'))[-1]
projected['training_episodes.csv']=bench.stat().st_size*full['training_episodes']/128
result=dict(classification='ENGINEERING_ONLY',full_test_outcomes_opened=False,
    probe_conditions='H48/L12 on reserved smoke profiles/tapes and initial policies; no performance selection',
    raw_sizes=sizes,projected_bytes=projected,projected_total_bytes=sum(projected.values()),
    uncertainty='request/action CSV lengths and failure density vary; add snapshots, JSON and logs')
(out/'size_estimate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(output=str(out),projected_total_mb=sum(projected.values())/1e6,projected_bytes=projected),indent=2))
