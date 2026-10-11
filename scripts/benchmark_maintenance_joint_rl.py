from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import torch
from ht_pdm_fjsp import maintenance_joint_rl as e

torch.set_num_threads(1)
cfg=e.settings('smoke')
cfg.update(horizon=48,tail_ticks=12,hidden=32,vector_envs=32,rollout=3072,
    env_steps=3072,minibatch=256,epochs=4,checkpoint_updates=[1],evaluation_batch=32)
out=Path('artifacts')/('maintenance_joint_rl_v1_engineering_benchmark_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
out.mkdir()
panels,_=e.build_panels(cfg);audit,forbidden=e.seed_audit(e.settings('smoke'))
spec=e.training_sampler(cfg,panels,forbidden)
journals={k:e.CountingJournal(out/f'{k}.csv') for k in ['training_episodes','training_progress']}
records=[]
try:
    for algorithm in e.ARMS:
        model=e.model_for(cfg,cfg['train_seeds'][0],algorithm)
        coverage=e.fit(model,cfg,cfg['train_seeds'][0],spec,journals,lambda *args:None)
        start=time.perf_counter()
        rows=e.evaluate(panels,cfg,'test',algorithm,cfg['train_seeds'][0],model,1)
        elapsed=time.perf_counter()-start
        records.append(dict(algorithm=algorithm,coverage=coverage,evaluation_seconds=elapsed,
            evaluation_physical_ticks=len(rows)*60,evaluation_episodes=len(rows),
            evaluation_batch_size=4))
finally:
    for j in journals.values():j.close()
report=dict(classification='ENGINEERING_ONLY',full_test_outcomes_opened=False,
    settings=cfg,records=records,seed_audit=audit,
    conservative_projection_note='one-update cold initialization and tiny evaluation batch on Mac; lab measurement may differ',
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (e.ROOT/'src/ht_pdm_fjsp').glob('maintenance_joint*.py')})
(out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(output=str(out),records=records),indent=2))
