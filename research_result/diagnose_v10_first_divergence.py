"""Same-tape V10 versus rollout first-divergence and full-horizon root probe."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import CASES,HORIZON,FrameworkEnv,catalogue,configs
from ht_pdm_fjsp.maintenance_framework_v7_campaign import V7RuleController
from ht_pdm_fjsp.maintenance_framework_v9_campaign import legal_present,policy_action
from ht_pdm_fjsp.maintenance_framework_v10_campaign import load_checkpoint
from ht_pdm_fjsp.maintenance_request_policies import request_action,risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings,state_from_dict,submit_requests,transition

ROOT=Path(__file__).resolve().parent.parent
V10=ROOT/'research_result/framework_v10_dev_20261020T072256Z'
PLAN=ROOT/'research_result/v10_first_divergence_plan_vi.md'
VERSION='v10_first_divergence_dev_v1'
TARGET=('interchangeable_scale1','reciprocal_scale2')
TRAIN=tuple(range(70400000,70400005))
DEV=tuple(range(70410000,70410016))
FORECAST_ANCHOR=70470000
ACTIONS=catalogue()
METRICS=('objective','maintenance_cost','unavailability_cost','failure_cost','tail_cost',
         'H_availability','H_failures','queued_cm_ticks','queued_pm_ticks',
         'H_pending','end_pending','service_starts','requests_created','requests_completed')
SOURCES=tuple(ROOT/path for path in (
    'research_result/diagnose_v10_first_divergence.py',
    'research_result/v10_first_divergence_plan_vi.md',
    'src/ht_pdm_fjsp/maintenance_framework_v1.py',
    'src/ht_pdm_fjsp/maintenance_framework_v7_campaign.py',
    'src/ht_pdm_fjsp/maintenance_framework_v9_campaign.py',
    'src/ht_pdm_fjsp/maintenance_framework_v10_campaign.py',
    'src/ht_pdm_fjsp/maintenance_requests.py',
    'src/ht_pdm_fjsp/maintenance_request_policies.py',
    'src/ht_pdm_fjsp/maintenance_dispatch.py',
))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def add_row(path,row):
    path=Path(path);first=not path.exists()
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if first:writer.writeheader()
        writer.writerow(row)


def category(left,right):
    a,b=ACTIONS[left],ACTIONS[right]
    if a==b:return 'match'
    if not a and b:return 'wait_vs_service'
    if a and not b:return 'service_vs_wait'
    if len(a)<len(b):return 'lower_cardinality'
    if len(a)>len(b):return 'higher_cardinality'
    return ('machine_subset' if {m for m,_ in a}!={m for m,_ in b}
            else 'technician_assignment')


def duration(config,action):
    return sum(config.service_time[m][j] for m,j in ACTIONS[action])


def load_inputs():
    manifest=json.loads((V10/'manifest.json').read_text())
    if manifest['status']!='COMPLETED' or manifest['profile']!='dev':
        raise RuntimeError('V10 source not complete')
    selection=json.loads((V10/'selection.json').read_text())
    if sha(V10/'selection.json')!=manifest['selection_sha256']:
        raise RuntimeError('selection hash changed')
    models={};hashes={}
    for seed in TRAIN:
        item=selection['assignment_credit'][str(seed)]
        path=V10/item['checkpoint']
        if sha(path)!=item['sha256']:raise RuntimeError('checkpoint hash changed')
        models[seed]=load_checkpoint(path)
        hashes[str(seed)]=item['sha256']
    refs={(r['case'],r['controller'],r['train_seed'],int(r['seed'])):r
          for r in csv.DictReader((V10/'episodes.csv').open(newline=''))
          if r['case'] in TARGET and r['controller'] in ('assignment_credit','rollout')}
    if len(refs)!=(len(TARGET)*len(DEV)*(len(TRAIN)+1)):
        raise RuntimeError('source episode grid incomplete')
    return models,hashes,refs,manifest


def episode(case,physical,actor,forecast_seed,deadline):
    cfg=configs()[case];env=FrameworkEnv(cfg);env.reset(seed=physical)
    actions=[];states=[]
    rule=V7RuleController('rollout',forecast_seed,deadline) if actor is None else None
    for tick in range(HORIZON):
        deadline();state=env.inner.state;states.append(asdict(state))
        submissions,_,legal,obs=legal_present(cfg,state,tick)
        if actor is None:
            supplied,matching=rule.choose(cfg,state,tick,case=case,seed=physical)
            if supplied!=submissions:raise AssertionError('requester mismatch')
            action=ACTIONS.index(matching)
        else:action=policy_action(actor,obs,legal)
        if action not in legal:raise AssertionError('illegal action')
        actions.append(action);_,_,done,_,_=env.step(action)
    if not done:raise AssertionError('unterminated episode')
    return actions,states,env.episode_metrics()


def forecast_root(config,state,tick,root,bank,deadline):
    totals=[];shorts=[];steps=0
    for events in bank:
        current=state;total=0.;short=0.
        for offset,t in enumerate(range(tick,config.horizon)):
            deadline()
            submissions=request_action(config,current,t,'threshold')
            admitted=submit_requests(config,current,t,submissions)
            action=root if offset==0 else ACTIONS.index(risk_matching(config,admitted,t))
            if offset==0 and ACTIONS[action] not in legal_matchings(config,admitted):
                raise AssertionError('illegal root')
            current,cost,_=transition(config,current,t,submissions,ACTIONS[action],events[offset])
            total+=cost;steps+=1
            if offset<6:short+=cost
        totals.append(total);shorts.append(short)
    return float(np.mean(shorts)),float(np.mean(totals)),steps


def bank_for(config,case,train,physical,tick,scenarios):
    seed=role_seed(FORECAST_ANCHOR,f'{VERSION}:{case}:{train}:{physical}:{tick}')
    rng=random.Random(seed)
    events=tuple(tuple(tuple(rng.random()<config.failure_probability
                              for _ in range(config.machines))
                       for _ in range(tick,config.horizon)) for _ in range(scenarios))
    return seed,events


def summarize(rows):
    result={}
    for case in TARGET:
        for train in TRAIN:
            cell=[r for r in rows if r['case']==case and r['train_seed']==train]
            if not cell:continue
            result[f'{case}/{train}']=dict(pairs=len(cell),
                divergence_categories=dict(Counter(r['category'] for r in cell)),
                mean_episode_delta=float(np.mean([r['episode_delta'] for r in cell])),
                mean_q6_delta=float(np.mean([r['q6_delta'] for r in cell if r['divergence_tick']>=0]))
                    if any(r['divergence_tick']>=0 for r in cell) else None,
                mean_qfull_delta=float(np.mean([r['qfull_delta'] for r in cell if r['divergence_tick']>=0]))
                    if any(r['divergence_tick']>=0 for r in cell) else None,
                first_divergence_mean=float(np.mean([r['divergence_tick'] for r in cell if r['divergence_tick']>=0]))
                    if any(r['divergence_tick']>=0 for r in cell) else None,
                full_root_favors_rollout_ge5=sum(r['qfull_delta']>=5 for r in cell),
                short_root_favors_rollout_ge5=sum(r['q6_delta']>=5 for r in cell))
    return result


def run(profile,output):
    if profile not in ('smoke','dev'):raise ValueError(profile)
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    start=time.monotonic();limit=180 if profile=='smoke' else 900
    def deadline():
        if time.monotonic()-start>=limit:raise TimeoutError('first divergence timeout')
    rel=[str(p.relative_to(ROOT)) for p in SOURCES]
    dirty=subprocess.check_output(['git','diff','--name-only','HEAD','--',*rel],cwd=ROOT,text=True).splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*rel],cwd=ROOT,text=True).splitlines()
    if profile=='dev' and (dirty or untracked):raise RuntimeError('uncommitted first-divergence source')
    models,hashes,refs,_=load_inputs()
    trains=TRAIN if profile=='dev' else TRAIN[:1]
    physicals=DEV if profile=='dev' else DEV[:1]
    scenarios=16 if profile=='dev' else 1
    manifest=dict(status='RUNNING',phase='preflight',profile=profile,version=VERSION,
        started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        invocation=[sys.executable,*sys.argv],runtime=dict(python=platform.python_version(),
        platform=platform.platform()),cwd=str(ROOT),timeout_seconds=limit,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES},
        v10_manifest_sha256=sha(V10/'manifest.json'),v10_selection_sha256=sha(V10/'selection.json'),
        v10_episodes_sha256=sha(V10/'episodes.csv'),selected_checkpoint_hashes=hashes,
        cases=list(TARGET),training_seeds=list(trains),physical_seeds=list(physicals),
        forecast_anchor=FORECAST_ANCHOR,scenarios=scenarios)
    save(out/'manifest.json',manifest);shutil.copy2(PLAN,out/'protocol.md')
    try:
        rollouts={}
        manifest['phase']='rollout_replay';save(out/'manifest.json',manifest)
        for case in tqdm(TARGET,desc='Rollout unique cases',unit='case'):
            for physical in physicals:
                actions,states,metrics=episode(case,physical,None,70420000,deadline)
                ref=refs[(case,'rollout','',physical)]
                for metric in METRICS:
                    if not np.isclose(float(metrics[metric]),float(ref[metric]),atol=1e-8,rtol=0):
                        raise RuntimeError(f'rollout replay mismatch {case}/{physical}/{metric}')
                rollouts[(case,physical)]=(actions,states,metrics)
                add_row(out/'rollout_episodes.partial.csv',dict(case=case,physical_seed=physical,
                    **{k:metrics[k] for k in METRICS}))
                for tick,action in enumerate(actions):
                    add_row(out/'rollout_actions.partial.csv',dict(case=case,physical_seed=physical,
                        tick=tick,action=action))
        manifest['phase']='paired_replay_and_forecast';save(out/'manifest.json',manifest)
        rows=[];forecast_transitions=0;divergences=0
        grid=[(case,train,physical) for case in TARGET for train in trains for physical in physicals]
        with (out/'divergences.jsonl').open('x') as state_stream:
            for case,train,physical in tqdm(grid,desc='V10 first divergence',unit='pair'):
                deadline();cfg=configs()[case]
                actions,states,metrics=episode(case,physical,models[train],70420000,deadline)
                ref=refs[(case,'assignment_credit',str(train),physical)]
                for metric in METRICS:
                    if not np.isclose(float(metrics[metric]),float(ref[metric]),atol=1e-8,rtol=0):
                        raise RuntimeError(f'V10 replay mismatch {case}/{train}/{physical}/{metric}')
                other,other_states,other_metrics=rollouts[(case,physical)]
                for tick,action in enumerate(actions):
                    add_row(out/'v10_actions.partial.csv',dict(case=case,train_seed=train,
                        physical_seed=physical,tick=tick,action=action))
                first=next((t for t,(a,b) in enumerate(zip(actions,other)) if a!=b),None)
                row=dict(case=case,train_seed=train,physical_seed=physical,
                         episode_delta=metrics['objective']-other_metrics['objective'],
                         divergence_tick=first if first is not None else -1,
                         category='match' if first is None else category(actions[first],other[first]),
                         v10_action=-1 if first is None else actions[first],
                         rollout_action=-1 if first is None else other[first],
                         v10_duration=0 if first is None else duration(cfg,actions[first]),
                         rollout_duration=0 if first is None else duration(cfg,other[first]),
                         q6_delta=0.,qfull_delta=0.,forecast_seed=0,forecast_transitions=0)
                if first is not None:
                    if any(actions[t]!=other[t] or states[t]!=other_states[t] for t in range(first)):raise AssertionError('pre-divergence path differs')
                    if states[first]!=other_states[first]:raise AssertionError('first divergence state differs')
                    divergences+=1
                    seed,bank=bank_for(cfg,case,train,physical,first,scenarios)
                    state=state_from_dict(states[first]);q={}
                    for name,action in (('v10',actions[first]),('rollout',other[first])):
                        q[name]=forecast_root(cfg,state,first,action,bank,deadline)
                    row.update(q6_delta=q['v10'][0]-q['rollout'][0],
                               qfull_delta=q['v10'][1]-q['rollout'][1],
                               forecast_seed=seed,forecast_transitions=q['v10'][2]+q['rollout'][2])
                    forecast_transitions+=row['forecast_transitions']
                    state_stream.write(json.dumps(dict(**row,original=states[first],
                        q6={'v10':q['v10'][0],'rollout':q['rollout'][0]},
                        qfull={'v10':q['v10'][1],'rollout':q['rollout'][1]}),sort_keys=True)+'\n')
                for name,metric in (('v10',metrics),('rollout',other_metrics)):
                    row.update({f'{name}_{k}':metric[k] for k in METRICS})
                rows.append(row);add_row(out/'pairs.partial.csv',row)
        for name in ('rollout_episodes','rollout_actions','v10_actions','pairs'):
            shutil.copy2(out/f'{name}.partial.csv',out/f'{name}.csv')
        save(out/'summary.json',summarize(rows))
        if manifest['source_hashes']!={str(p.relative_to(ROOT)):sha(p) for p in SOURCES}:
            raise RuntimeError('source changed during run')
        manifest.update(status='COMPLETED',phase='done',elapsed_seconds=time.monotonic()-start,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            counts=dict(pairs=len(rows),rollout_episodes=len(rollouts),divergences=divergences,
                        decisions=(len(rows)+len(rollouts))*HORIZON,
                        forecast_transitions=forecast_transitions),
            artifact_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
        save(out/'manifest.json',manifest)
        return out
    except Exception as exc:
        manifest.update(status='INCOMPLETE' if isinstance(exc,TimeoutError) else 'FAILED',
            error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        save(out/'manifest.json',manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=('smoke','dev'),required=True)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args();print(json.dumps(dict(status='COMPLETED',output_dir=str(run(a.profile,a.output_dir)))))

if __name__=='__main__':main()
