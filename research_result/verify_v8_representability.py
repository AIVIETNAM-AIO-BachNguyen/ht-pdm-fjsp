"""Independent exact-collision and split accounting audit for V8 DEV probe."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import catalogue, configs, vector_observation
from ht_pdm_fjsp.maintenance_framework_v6_policy import EdgeSharingActor
from ht_pdm_fjsp.maintenance_requests import state_from_dict, submit_requests

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'research_result/v8_headroom_dev_20261020T052426Z'
ACTIONS=catalogue()


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def observation(row):
    cfg=configs()[row['case']]
    old=state_from_dict(row['original'])
    admitted=submit_requests(cfg,old,row['tick'],tuple(row['submissions']))
    return vector_observation(cfg,admitted,row['tick'])


def edge_key(obs,m,j):
    # A separate extraction from the V7 actor's 2 + 8 + 2 + 2 contract.
    indexes=[0,1]+list(range(2+8*m,10+8*m)) + [34+3*j,35+3*j] + \
        [40+2*(2*m+j),41+2*(2*m+j)]
    return obs[indexes].tobytes()


def independent_census(rows):
    groups=defaultdict(list)
    non_tie=0
    for row in tqdm(rows,desc='Independent exact collision audit',unit='state'):
        obs=observation(row)
        legal=sorted(int(a) for a in row['legal_actions'])
        cost={int(a):float(v) for a,v in row['scores'].items()}
        assert set(legal)==set(cost)
        for i,left in enumerate(legal):
            for right in legal[i+1:]:
                margin=cost[right]-cost[left]
                if margin==0:continue
                l=set(ACTIONS[left]);r=set(ACTIONS[right]); pieces=[]
                for m,j in sorted(l-r):pieces.append((1,m,j,edge_key(obs,m,j)))
                for m,j in sorted(r-l):pieces.append((-1,m,j,edge_key(obs,m,j)))
                groups[(left,right,tuple(sorted(pieces)))].append((int(row['state_id']),
                    row['case'],row['source_arm'],int(row['train_seed']),margin,obs.tobytes()))
                non_tie+=1
    result={'pair_records':non_tie,'collision_key_count':len(groups),'thresholds':{}}
    for threshold in (1,5,10):
        key_count=pair_count=0; covered=set();by_case=defaultdict(int);by_source=defaultdict(int)
        for entries in groups.values():
            eligible=[]
            for first in entries:
                if first[4]<threshold:continue
                for second in entries:
                    if second[4]>-threshold or first[5]==second[5]:continue
                    eligible.append((first,second))
            if not eligible:continue
            key_count+=1;pair_count+=len(eligible)
            distinct={x for pair in eligible for x in pair}
            for x in distinct:
                covered.add(x[0]);by_case[x[1]]+=1
                by_source[f'{x[2]}/{x[3]}']+=1
        result['thresholds'][str(threshold)]={'collision_keys':key_count,
            'opposite_preference_record_pairs':pair_count,'unique_states':len(covered),
            'states':sorted(covered),'by_case':dict(sorted(by_case.items())),
            'by_source_seed':dict(sorted(by_source.items()))}
    return result


def audit(run):
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='COMPLETED' and manifest['profile']=='dev'
    assert manifest['input_states']==720
    for name in ('states.jsonl','manifest.json','summary.json'):
        assert sha(SOURCE/name)==manifest['input_hashes'][name]
    for name,digest in manifest['source_hashes'].items():
        assert sha(ROOT/name)==digest
    for name,digest in manifest['output_hashes'].items():
        assert sha(run/name)==digest
    rows=[json.loads(line) for line in (SOURCE/'states.jsonl').open()]
    assert len(rows)==len({r['state_id'] for r in rows})==720
    actual=independent_census(rows)
    reported=json.loads((run/'collisions.json').read_text())
    assert actual['pair_records']==reported['pair_records']
    assert actual['collision_key_count']==reported['collision_key_count']
    for threshold,cell in actual['thresholds'].items():
        for key,value in cell.items():assert reported['thresholds'][threshold][key]==value
    for threshold,cell in reported['thresholds'].items():
        for item in cell['examples']:
            a=rows[item['state_a']];b=rows[item['state_b']]
            assert observation(a).tobytes()!=observation(b).tobytes()
            left,right=item['left_action'],item['right_action']
            ca={int(k):float(v) for k,v in a['scores'].items()}
            cb={int(k):float(v) for k,v in b['scores'].items()}
            assert ca[right]-ca[left]>=int(threshold)
            assert cb[right]-cb[left]<=-int(threshold)
            # Equality of actor action-margin follows from the exact signed edge inputs.
            torch.manual_seed(77);actor=EdgeSharingActor('shared_edge').eval()
            with torch.no_grad():
                logits=actor(torch.tensor(np.stack([observation(a),observation(b)])))
            assert abs(float((logits[0,left]-logits[0,right])-
                             (logits[1,left]-logits[1,right])))<1e-5
    counts={'train':0,'selection':0,'probe':0}
    for row in rows:
        seed=int(row['physical_seed'])
        if 69100000<=seed<=69100019:counts['train']+=1
        elif 69100020<=seed<=69100023:counts['selection']+=1
        elif 69100024<=seed<=69100031:counts['probe']+=1
        else:raise AssertionError('unregistered seed')
    capacity=json.loads((run/'capacity.json').read_text())
    assert counts==capacity['split_counts']
    assert capacity['starts']==list(range(70200000,70200005)) and capacity['epochs']==100
    for family in ('edge','flat','joint'):
        cells=capacity['models'][family]
        assert len(cells)==5 and [x['seed'] for x in cells]==capacity['starts']
        assert all(1<=x['selected_epoch']<=100 for x in cells)
        for x in cells:
            for split,n in counts.items():
                metric=x[split]
                assert metric['states']==n and 0<=metric['gap_ge_5']<=n
                assert 0<=metric['planner_action_match']<=n
                assert np.isfinite(metric['mse']) and np.isfinite(metric['mean_root_regret'])
    return dict(status='PASS',run=str(run),states=720,pair_records=actual['pair_records'],
                material_states=actual['thresholds']['5']['unique_states'],split_counts=counts,
                checks=['source/output hashes','independent observation and signed-edge census',
                        'distinct full observation and actor-margin example replay',
                        'physical-seed split and finite fit-metric accounting'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    result=audit(args.run_dir.resolve())
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
