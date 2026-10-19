import csv,json,sys,shutil
from pathlib import Path
from collections import Counter
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_context_waiting as study
from ht_pdm_fjsp import maintenance_waiting as envmod
from ht_pdm_fjsp import maintenance_dispatch as physics
from ht_pdm_fjsp.maintenance_dispatch_policy import batch_observations

torch.set_num_threads(1)
run=Path(sys.argv[1]);out=Path(sys.argv[2]);manifest=json.loads((run/'manifest.json').read_text())
assert manifest['git_revision']=='e2ec53e3239881aaf2d7db57aa7b72c64342165f'
for name,data in study.source_snapshot().items():
    assert data==(run/'source_snapshot'/name).read_bytes(),name
profile=manifest['profile'];config_cache={};models={};counts=Counter();details=[];buffer=[];current=None

def flush():
    if not buffer:return
    arm,seed=current
    model=models.setdefault(current,None)
    if model is None:
        model,_=study.load_checkpoint(run/arm/f'train_seed_{seed}'/'model.pt')
        models[current]=model
    observations=[]
    for row in buffer:
        k=(arm,row['family'],int(row['eval_seed']))
        if k not in config_cache:
            config_cache[k]=envmod.family_config(k[1],k[2],profile,study.factors(arm)[1])
        state=physics.DispatchState(*(tuple(json.loads(row[f])) for f in
            ('ages','failed','pending_wait','remaining','assigned')))
        observations.append(envmod.observation(config_cache[k],state,int(row['time'])))
    with torch.no_grad():
        sequences=model.decode(batch_observations(observations),'learned_both',deterministic=True)[0].tolist()
    for row,tokens in zip(buffer,sequences):
        expected_pairs=[tuple(p) for p in json.loads(row['pairs'])]
        actual_pairs=[tuple(p) for p in tokens if p[0]>=0]
        expected_tokens=[tuple(p) for p in json.loads(row['sequence'])]
        actual_tokens=[tuple(p) for p in tokens if p[0]!=-2]
        counts['states']+=1
        counts['sequence_mismatches']+=int(expected_tokens!=actual_tokens)
        counts['physical_matching_mismatches']+=int(sorted(expected_pairs)!=sorted(actual_pairs))
        if expected_tokens!=actual_tokens and len(details)<100:
            details.append(dict(arm=arm,train_seed=seed,family=row['family'],eval_seed=row['eval_seed'],
                time=row['time'],logged_sequence=expected_tokens,local_batched_sequence=actual_tokens,
                physical_matching_changed=sorted(expected_pairs)!=sorted(actual_pairs)))
    buffer.clear()

with (run/'decisions.csv').open() as f:
    for row in tqdm(csv.DictReader(f),total=202000,desc='checkpoint decisions on logged states',unit='state'):
        k=(row['algorithm'],int(row['train_seed']))
        if current!=k:flush();current=k
        buffer.append(row)
        if len(buffer)==256:flush()
flush()
assert counts['states']==202000 and len(models)==40
result=dict(status='MATCHED' if not counts['physical_matching_mismatches'] else 'DIFFERENT_MATCHINGS',
    **dict(counts),models=40,batch_size=256,source_snapshot_matches=True,details=details,
    scope='local batched checkpoint inference on original logged states, not independent training reproduction')
(out/'policy_replay.json').write_text(json.dumps(result,indent=2)+'\n')
shutil.copy2(__file__,out/'policy_replay.py')
print(json.dumps(result,indent=2))
