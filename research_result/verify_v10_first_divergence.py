"""Independent same-tape action, first-divergence and long-horizon root audit."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON,FrameworkEnv,catalogue,configs,vector_observation
from ht_pdm_fjsp.maintenance_framework_v6_policy import EdgeSharingActor
from ht_pdm_fjsp.maintenance_framework_v7_campaign import VERSION as V7_VERSION
from ht_pdm_fjsp.maintenance_request_policies import choose,request_action,risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings,state_from_dict,submit_requests,transition

ROOT=Path(__file__).resolve().parent.parent
V10=ROOT/'research_result/framework_v10_dev_20261020T072256Z'
VERSION='v10_first_divergence_dev_v1'
TARGET=('interchangeable_scale1','reciprocal_scale2')
TRAIN=tuple(range(70400000,70400005))
DEV=tuple(range(70410000,70410016))
ACTIONS=catalogue()
METRICS=('objective','maintenance_cost','unavailability_cost','failure_cost','tail_cost',
         'H_availability','H_failures','queued_cm_ticks','queued_pm_ticks',
         'H_pending','end_pending','service_starts','requests_created','requests_completed')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def close(left,right,label):
    if not np.isclose(float(left),float(right),atol=1e-8,rtol=0):
        raise AssertionError(f'{label}: {left} != {right}')


def model(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    actor=EdgeSharingActor('shared_edge');actor.load_state_dict(payload['state_dict']);actor.eval()
    return actor


def actor_action(actor,cfg,state,tick):
    submitted=request_action(cfg,state,tick,'threshold')
    admitted=submit_requests(cfg,state,tick,submitted)
    legal=sorted(ACTIONS.index(a) for a in legal_matchings(cfg,admitted))
    obs=vector_observation(cfg,admitted,tick)
    with torch.no_grad():
        logits=actor(torch.as_tensor(obs).reshape(1,56))[0]
        masked=torch.full_like(logits,-1e9);masked[legal]=logits[legal]
        return int(np.argmax(torch.softmax(masked,dim=0).cpu().numpy()))


def rollout_action(cfg,state,tick,case,physical):
    forecast=role_seed(70420000,f'{V7_VERSION}:forecast:{case}:{physical}:{tick}')
    _,matching,_=choose(cfg,state,tick,'threshold','rollout',
        policy_horizon=HORIZON,forecast_seed=forecast,scenarios=16,forecast_horizon=6)
    return ACTIONS.index(matching)


def class_of(left,right):
    a,b=ACTIONS[left],ACTIONS[right]
    if a==b:return 'match'
    if not a and b:return 'wait_vs_service'
    if a and not b:return 'service_vs_wait'
    if len(a)!=len(b):return 'lower_cardinality' if len(a)<len(b) else 'higher_cardinality'
    return 'machine_subset' if {m for m,_ in a}!={m for m,_ in b} else 'technician_assignment'


def root_score(cfg,state,tick,root,events):
    short=[];full=[];steps=0
    for tape in events:
        current=state;short_cost=0.;total=0.
        for offset,t in enumerate(range(tick,cfg.horizon)):
            submissions=request_action(cfg,current,t,'threshold')
            pending=submit_requests(cfg,current,t,submissions)
            matching=ACTIONS[root] if offset==0 else risk_matching(cfg,pending,t)
            if offset==0 and matching not in legal_matchings(cfg,pending):
                raise AssertionError('root infeasible')
            current,cost,_=transition(cfg,current,t,submissions,matching,tape[offset])
            total+=cost;steps+=1
            if offset<6:short_cost+=cost
        short.append(short_cost);full.append(total)
    return float(np.mean(short)),float(np.mean(full)),steps


def forecast_events(cfg,case,train,physical,tick,count):
    seed=role_seed(70470000,f'{VERSION}:{case}:{train}:{physical}:{tick}')
    rng=random.Random(seed)
    return seed,tuple(tuple(tuple(rng.random()<cfg.failure_probability
                                 for _ in range(cfg.machines))
                          for _ in range(tick,cfg.horizon)) for _ in range(count))


def replay(case,physical,expected_actions,actor=None):
    cfg=configs()[case];env=FrameworkEnv(cfg);env.reset(seed=physical)
    states=[]
    for tick,expected in enumerate(expected_actions):
        state=env.inner.state
        states.append(json.loads(json.dumps(__import__('dataclasses').asdict(state))))
        action=(actor_action(actor,cfg,state,tick) if actor is not None
                else rollout_action(cfg,state,tick,case,physical))
        if action!=expected:raise AssertionError(f'action mismatch {case}/{physical}/{tick}')
        env.step(action)
    if len(expected_actions)!=HORIZON:raise AssertionError('action length')
    return states,env.episode_metrics()


def audit(path):
    out=Path(path).resolve();manifest=json.loads((out/'manifest.json').read_text())
    if manifest['status']!='COMPLETED' or manifest['profile']!='dev':raise AssertionError('incomplete DEV')
    if manifest['cases']!=list(TARGET) or manifest['training_seeds']!=list(TRAIN) or manifest['physical_seeds']!=list(DEV):
        raise AssertionError('grid changed')
    for rel,digest in manifest['source_hashes'].items():
        if sha(ROOT/rel)!=digest:raise AssertionError(f'source hash {rel}')
    for name,digest in manifest['artifact_hashes'].items():
        if sha(out/name)!=digest:raise AssertionError(f'artifact hash {name}')
    for name,key in (('manifest.json','v10_manifest_sha256'),('selection.json','v10_selection_sha256'),
                     ('episodes.csv','v10_episodes_sha256')):
        if sha(V10/name)!=manifest[key]:raise AssertionError(f'V10 source hash {name}')
    selection=json.loads((V10/'selection.json').read_text())
    actors={}
    for train in TRAIN:
        item=selection['assignment_credit'][str(train)];path=V10/item['checkpoint']
        if sha(path)!=item['sha256'] or sha(path)!=manifest['selected_checkpoint_hashes'][str(train)]:
            raise AssertionError('checkpoint hash')
        actors[train]=model(path)
    original={(r['case'],r['controller'],r['train_seed'],int(r['seed'])):r
              for r in rows(V10/'episodes.csv') if r['case'] in TARGET and
              r['controller'] in ('assignment_credit','rollout')}
    roll_actions=defaultdict(list);actor_actions=defaultdict(list)
    for r in rows(out/'rollout_actions.csv'):
        roll_actions[(r['case'],int(r['physical_seed']))].append((int(r['tick']),int(r['action'])))
    for r in rows(out/'v10_actions.csv'):
        actor_actions[(r['case'],int(r['train_seed']),int(r['physical_seed']))].append(
            (int(r['tick']),int(r['action'])))
    rollout_rows={(r['case'],int(r['physical_seed'])):r for r in rows(out/'rollout_episodes.csv')}
    pair_rows={(r['case'],int(r['train_seed']),int(r['physical_seed'])):r for r in rows(out/'pairs.csv')}
    if len(roll_actions)!=len(TARGET)*len(DEV) or len(actor_actions)!=len(TARGET)*len(TRAIN)*len(DEV):
        raise AssertionError('action grid')
    if len(rollout_rows)!=len(roll_actions) or len(pair_rows)!=len(actor_actions):
        raise AssertionError('episode grid')
    snapshots={}
    for case in tqdm(TARGET,desc='Independent rollout replay',unit='case'):
        for physical in DEV:
            key=(case,physical);sequence=roll_actions[key]
            if [tick for tick,_ in sequence]!=list(range(HORIZON)):raise AssertionError('rollout ticks')
            states,metrics=replay(case,physical,[a for _,a in sequence])
            snapshots[key]=states
            saved=rollout_rows[key];prior=original[(case,'rollout','',physical)]
            for metric in METRICS:
                close(metrics[metric],saved[metric],f'rollout {metric}')
                close(metrics[metric],prior[metric],f'prior rollout {metric}')
    divergence_map={(r['case'],r['train_seed'],r['physical_seed']):r
                    for r in map(json.loads,(out/'divergences.jsonl').open())}
    result=defaultdict(list);forecast_transitions=0;divergences=0
    for case in TARGET:
        cfg=configs()[case]
        for train in tqdm(TRAIN,desc=f'Independent {case} V10 replay',unit='seed'):
            for physical in DEV:
                key=(case,train,physical);sequence=actor_actions[key]
                if [tick for tick,_ in sequence]!=list(range(HORIZON)):raise AssertionError('V10 ticks')
                actions=[a for _,a in sequence];states,metrics=replay(case,physical,actions,actors[train])
                roll=[a for _,a in roll_actions[(case,physical)]];other=rollout_rows[(case,physical)]
                saved=pair_rows[key];prior=original[(case,'assignment_credit',str(train),physical)]
                for metric in METRICS:
                    close(metrics[metric],saved[f'v10_{metric}'],f'V10 {metric}')
                    close(metrics[metric],prior[metric],f'prior V10 {metric}')
                    close(other[metric],saved[f'rollout_{metric}'],f'pair rollout {metric}')
                close(metrics['objective']-float(other['objective']),saved['episode_delta'],'episode delta')
                first=next((t for t,(a,b) in enumerate(zip(actions,roll)) if a!=b),None)
                assert int(saved['divergence_tick'])==(first if first is not None else -1)
                if first is not None:
                    assert all(actions[t]==roll[t] and states[t]==snapshots[(case,physical)][t]
                               for t in range(first))
                    assert states[first]==snapshots[(case,physical)][first]
                    assert saved['category']==class_of(actions[first],roll[first])
                    assert int(saved['v10_action'])==actions[first] and int(saved['rollout_action'])==roll[first]
                    assert int(saved['v10_duration'])==sum(cfg.service_time[m][j] for m,j in ACTIONS[actions[first]])
                    assert int(saved['rollout_duration'])==sum(cfg.service_time[m][j] for m,j in ACTIONS[roll[first]])
                    raw=divergence_map.pop((case,train,physical))
                    assert raw['original']==states[first]
                    seed,bank=forecast_events(cfg,case,train,physical,first,16)
                    assert int(saved['forecast_seed'])==seed==raw['forecast_seed']
                    state=state_from_dict(states[first])
                    qleft=root_score(cfg,state,first,actions[first],bank)
                    qright=root_score(cfg,state,first,roll[first],bank)
                    close(qleft[0],raw['q6']['v10'],'q6 V10')
                    close(qright[0],raw['q6']['rollout'],'q6 rollout')
                    close(qleft[1],raw['qfull']['v10'],'qfull V10')
                    close(qright[1],raw['qfull']['rollout'],'qfull rollout')
                    close(qleft[0]-qright[0],saved['q6_delta'],'q6 delta')
                    close(qleft[1]-qright[1],saved['qfull_delta'],'qfull delta')
                    count=qleft[2]+qright[2]
                    if int(saved['forecast_transitions'])!=count:raise AssertionError('forecast count')
                    forecast_transitions+=count;divergences+=1
                else:
                    if saved['category']!='match' or int(saved['forecast_transitions'])!=0:
                        raise AssertionError('no-divergence record')
                result[f'{case}/{train}'].append(saved)
    if divergence_map:raise AssertionError('extra divergence states')
    summary=json.loads((out/'summary.json').read_text())
    if set(summary)!=set(result):raise AssertionError('summary keys')
    for key,cell in result.items():
        item=summary[key]
        assert item['pairs']==len(DEV)
        assert item['divergence_categories']==dict(Counter(r['category'] for r in cell))
        close(item['mean_episode_delta'],np.mean([float(r['episode_delta']) for r in cell]),'summary cost')
        diverged=[r for r in cell if int(r['divergence_tick'])>=0]
        for name,col in [('mean_q6_delta','q6_delta'),('mean_qfull_delta','qfull_delta'),
                         ('first_divergence_mean','divergence_tick')]:
            if diverged:close(item[name],np.mean([float(r[col]) for r in diverged]),name)
            elif item[name] is not None:raise AssertionError('empty summary')
        assert item['full_root_favors_rollout_ge5']==sum(float(r['qfull_delta'])>=5 for r in cell)
        assert item['short_root_favors_rollout_ge5']==sum(float(r['q6_delta'])>=5 for r in cell)
    expected=dict(pairs=len(actor_actions),rollout_episodes=len(roll_actions),
                  divergences=divergences,decisions=(len(actor_actions)+len(roll_actions))*HORIZON,
                  forecast_transitions=forecast_transitions)
    if manifest['counts']!=expected:raise AssertionError('manifest counts')
    return dict(status='PASS',pairs=len(actor_actions),rollout_episodes=len(roll_actions),
        divergences=divergences,forecast_transitions=forecast_transitions,
        source_commit=manifest['git_commit'],manifest_sha256=sha(out/'manifest.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=Path(a.output_dir).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    result=audit(a.input_dir)
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(**result,output_dir=str(out))))

if __name__=='__main__':main()
