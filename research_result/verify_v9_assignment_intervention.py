"""Independent action/metric replay of the V9 assignment-only DEV intervention."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES,HORIZON,FrameworkEnv,catalogue,configs
from ht_pdm_fjsp.maintenance_framework_v9_campaign import load_checkpoint

ROOT=Path(__file__).resolve().parent.parent
V9=ROOT/'research_result/framework_v9_dev_20261020T062502Z'
ACTIONS=catalogue()
ARMS=('all_action','chosen_action')
TRAIN=tuple(range(70300000,70300005))
DEV=tuple(range(70310000,70310016))
METRICS=('objective','maintenance_cost','unavailability_cost','failure_cost','tail_cost',
         'H_availability','H_failures','queued_cm_ticks','queued_pm_ticks',
         'H_pending','end_pending','service_starts','requests_created','requests_completed')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(left,right):return np.isclose(float(left),float(right),atol=1e-8,rtol=0)


def action_from_actor(actor,obs,legal):
    with torch.no_grad():
        raw=actor(torch.as_tensor(obs).reshape(1,56)).squeeze(0)
        masked=torch.full_like(raw,-1e9)
        masked[list(legal)]=raw[list(legal)]
        probs=torch.softmax(masked,dim=0).cpu().numpy()
    return int(np.argmax(probs))


def independently_swap(config,chosen,legal):
    chosen_machines=frozenset(m for m,_ in ACTIONS[chosen])
    options=[a for a in legal if frozenset(m for m,_ in ACTIONS[a])==chosen_machines]
    def duration(a):return sum(config.service_time[m][j] for m,j in ACTIONS[a])
    minimum=min(map(duration,options))
    return chosen if duration(chosen)==minimum else min(a for a in options if duration(a)==minimum)


def verify(path):
    out=Path(path).resolve();manifest=json.loads((out/'manifest.json').read_text())
    if manifest['status']!='COMPLETED':raise AssertionError('incomplete artifact')
    if manifest['profile']!='dev':raise AssertionError('not full DEV')
    if tuple(manifest['physical_seeds'])!=DEV or tuple(manifest['training_seeds'])!=TRAIN:
        raise AssertionError('seed grid changed')
    for rel,digest in manifest['source_hashes'].items():
        if sha(ROOT/rel)!=digest:raise AssertionError(f'source hash changed: {rel}')
    for name,digest in manifest['artifact_hashes'].items():
        if sha(out/name)!=digest:raise AssertionError(f'artifact hash changed: {name}')
    for filename,field in (('manifest.json','v9_manifest_sha256'),
                           ('selection.json','v9_selection_sha256'),
                           ('episodes.csv','v9_episodes_sha256')):
        if sha(V9/filename)!=manifest[field]:raise AssertionError(f'V9 input changed {filename}')
    selection=json.loads((V9/'selection.json').read_text())
    models={}
    for arm in ARMS:
        for train in TRAIN:
            item=selection[arm][str(train)];checkpoint=V9/item['checkpoint']
            if sha(checkpoint)!=item['sha256'] or sha(checkpoint)!=manifest['selected_checkpoint_hashes'][f'{arm}/{train}']:
                raise AssertionError('checkpoint hash mismatch')
            models[(arm,train)]=load_checkpoint(checkpoint)
    source={ (r['case'],r['controller'],int(r['train_seed']),int(r['seed'])):r
             for r in csv.DictReader((V9/'episodes.csv').open(newline='')) if r['controller'] in ARMS}
    episodes=list(csv.DictReader((out/'episodes.csv').open(newline='')))
    actions=defaultdict(list)
    for row in csv.DictReader((out/'actions.csv').open(newline='')):
        key=(row['case'],row['arm'],int(row['train_seed']),int(row['physical_seed']),int(row['treatment']))
        actions[key].append(row)
    expected={(case,arm,train,physical) for case in CASES for arm in ARMS
              for train in TRAIN for physical in DEV}
    seen=set();cell=defaultdict(list);change_total=0;decisions=0
    for row in tqdm(episodes,desc='Independent assignment audit',unit='pair'):
        case=row['case'];arm=row['arm'];train=int(row['train_seed']);physical=int(row['physical_seed'])
        key=(case,arm,train,physical)
        if key in seen or key not in expected:raise AssertionError('duplicate/out-of-grid episode')
        seen.add(key);config=configs()[case];actor=models[(arm,train)]
        for treated in (0,1):
            env=FrameworkEnv(config);env.reset(seed=physical)
            log=actions.pop((*key,treated),None)
            if log is None or len(log)!=HORIZON:raise AssertionError('action log length')
            changes=0;saved=0;starts=0;ties=0
            for tick,event in enumerate(log):
                if int(event['tick'])!=tick:raise AssertionError('tick ordering')
                legal=tuple(np.flatnonzero(env.action_masks()))
                chosen=action_from_actor(actor,env._observation(),legal)
                executed=independently_swap(config,chosen,legal) if treated else chosen
                if int(event['actor_action'])!=chosen or int(event['executed_action'])!=executed:
                    raise AssertionError(f'action mismatch {key}/{treated}/{tick}')
                original_duration=sum(config.service_time[m][j] for m,j in ACTIONS[chosen])
                executed_duration=sum(config.service_time[m][j] for m,j in ACTIONS[executed])
                if int(event['actor_duration'])!=original_duration or int(event['executed_duration'])!=executed_duration:
                    raise AssertionError('duration mismatch')
                if treated:
                    changes+=chosen!=executed;saved+=original_duration-executed_duration
                    ties+=chosen==executed and sum(
                        frozenset(m for m,_ in ACTIONS[a])==frozenset(m for m,_ in ACTIONS[chosen]) and
                        sum(config.service_time[m][j] for m,j in ACTIONS[a])==original_duration
                        for a in legal)>1
                starts+=len(ACTIONS[executed]);decisions+=1
                _,_,done,_,_=env.step(executed)
            if not done:raise AssertionError('unterminated episode')
            metrics=env.episode_metrics();prefix='treatment' if treated else 'control'
            for name in METRICS:
                if not close(metrics[name],row[f'{prefix}_{name}']):
                    raise AssertionError(f'metric mismatch {key}/{prefix}/{name}')
                if not treated and not close(metrics[name],source[key][name]):
                    raise AssertionError(f'V9 control mismatch {key}/{name}')
            for name,value in [('H_service_starts',starts),('changed_assignments',changes),
                               ('known_duration_saved',saved),('actor_minimum_ties',ties)]:
                if not close(value,row[f'{prefix}_{name}']):raise AssertionError(f'meta mismatch {name}')
            change_total+=changes
        cell[(case,arm,train)].append(row)
    if seen!=expected or actions:raise AssertionError('incomplete episode/action grid')
    if manifest['counts']!=dict(pairs=len(expected),episodes=2*len(expected),decisions=decisions):
        raise AssertionError('manifest counts')
    summary=json.loads((out/'summary.json').read_text())
    if set(summary)!={'/'.join(map(str,key)) for key in cell}:raise AssertionError('summary grid')
    for key,items in cell.items():
        if len(items)!=len(DEV):raise AssertionError('summary cell length')
        saved=summary['/'.join(map(str,key))]
        if saved['episodes']!=len(DEV) or saved['total_changed_assignments']!=sum(
            int(r['treatment_changed_assignments']) for r in items):raise AssertionError('summary count')
        if saved['total_known_duration_saved']!=sum(int(r['treatment_known_duration_saved']) for r in items):
            raise AssertionError('summary duration')
        for field,prefix in [('control_mean_objective','control'),('treatment_mean_objective','treatment')]:
            if not close(saved[field],np.mean([float(r[f'{prefix}_objective']) for r in items])):
                raise AssertionError('summary objective')
        for field,value in saved['paired_mean'].items():
            if not close(value,np.mean([float(r[f'treatment_{field}'])-float(r[f'control_{field}'])
                                        for r in items])):raise AssertionError(f'summary paired {field}')
        counts=saved['cost_pairs']
        if counts!=dict(better=sum(float(r['treatment_objective'])<float(r['control_objective']) for r in items),
                        equal=sum(float(r['treatment_objective'])==float(r['control_objective']) for r in items),
                        worse=sum(float(r['treatment_objective'])>float(r['control_objective']) for r in items)):
            raise AssertionError('summary cost pairs')
    return dict(status='PASS',pairs=int(len(expected)),episodes=int(2*len(expected)),decisions=int(decisions),
                changed_assignments=int(change_total),source_commit=manifest['git_commit'],
                intervention_manifest_sha256=sha(out/'manifest.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=Path(a.output_dir).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    result=verify(a.input_dir)
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(**result,output_dir=str(out))))

if __name__=='__main__':main()
