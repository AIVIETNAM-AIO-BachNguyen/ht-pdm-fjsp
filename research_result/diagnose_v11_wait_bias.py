"""DEV-only stored-Q and deterministic-action audit for V11 continuation bias."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES,HORIZON,FrameworkEnv,catalogue,configs
from ht_pdm_fjsp.maintenance_framework_v11_campaign import load_checkpoint,legal_present,policy_action
from ht_pdm_fjsp.maintenance_request_policies import risk_matching

ROOT=Path(__file__).resolve().parent.parent
V11=ROOT/'research_result/framework_v11_dev_20261020T115151Z'
PLAN=ROOT/'research_result/v11_wait_bias_diagnostic_plan_vi.md'
VERSION='v11_wait_bias_dev_v1'
ARMS=('reference_credit','self_credit')
TRAIN=tuple(range(70500000,70500005))
DEV=tuple(range(70510000,70510016))
ACTIONS=catalogue()
METRICS=('objective','maintenance_cost','unavailability_cost','failure_cost','tail_cost',
         'H_availability','H_failures','queued_cm_ticks','queued_pm_ticks',
         'H_pending','end_pending','service_starts','requests_created','requests_completed')
SOURCES=tuple(ROOT/path for path in (
    'research_result/diagnose_v11_wait_bias.py',
    'research_result/v11_wait_bias_diagnostic_plan_vi.md',
    'src/ht_pdm_fjsp/maintenance_framework_v11_campaign.py',
    'src/ht_pdm_fjsp/maintenance_framework_v1.py',
    'src/ht_pdm_fjsp/maintenance_request_policies.py',
    'src/ht_pdm_fjsp/maintenance_requests.py',
    'src/ht_pdm_fjsp/maintenance_dispatch.py',
))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def csv_rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def add_row(path,row):
    path=Path(path);first=not path.exists()
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if first:writer.writeheader()
        writer.writerow(row)


def q_wait_delta(values,legal):
    service=[i for i in legal if ACTIONS[i]]
    if 0 not in legal or not service:return None
    return float(values[0]-min(values[i] for i in service))


def load_inputs():
    manifest=json.loads((V11/'manifest.json').read_text())
    if manifest['status']!='COMPLETED' or manifest['profile']!='dev':
        raise RuntimeError('V11 incomplete')
    selection=json.loads((V11/'selection.json').read_text())
    if sha(V11/'selection.json')!=manifest['selection_sha256']:
        raise RuntimeError('V11 selection hash')
    models={};hashes={}
    for arm in ARMS:
        for train in TRAIN:
            selected=selection[arm][str(train)]
            for label,iteration,path in (('initial',0,V11/'checkpoints'/arm/str(train)/'0.pt'),
                                         ('selected',selected['iteration'],V11/selected['checkpoint'])):
                digest=sha(path)
                if label=='selected' and digest!=selected['sha256']:
                    raise RuntimeError('selected model hash')
                models[(arm,train,label)]=load_checkpoint(path)
                hashes[f'{arm}/{train}/{label}']=digest
    dev={(r['case'],r['controller'],int(r['train_seed']),int(r['checkpoint_iteration']),int(r['seed'])):r
         for r in csv_rows(V11/'development.csv') if r['controller'] in ARMS}
    selected={(r['case'],r['controller'],int(r['train_seed']),int(r['seed'])):r
              for r in csv_rows(V11/'episodes.csv') if r['controller'] in ARMS}
    if len(dev)!=len(CASES)*len(ARMS)*len(TRAIN)*7*len(DEV):
        raise RuntimeError('DEV checkpoint grid')
    if len(selected)!=len(CASES)*len(ARMS)*len(TRAIN)*len(DEV):
        raise RuntimeError('selected grid')
    return manifest,selection,models,hashes,dev,selected


def scan_q(profile,out,deadline):
    q=csv_rows(V11/'counterfactual_states.csv')
    if len(q)!=len(ARMS)*len(TRAIN)*6*len(CASES)*32:
        raise RuntimeError('incomplete V11 Q grid')
    if profile=='smoke':
        q=[r for arm in ARMS for r in [x for x in q if x['arm']==arm and
                                       int(x['train_seed'])==TRAIN[0]][:24]]
    rows=[]
    for row in tqdm(q,desc='V11 paired Q-WAIT scan',unit='state'):
        deadline();legal=tuple(json.loads(row['legal']))
        self_values={int(k):float(v) for k,v in json.loads(row['scores_self']).items()}
        ref_values={int(k):float(v) for k,v in json.loads(row['scores_reference']).items()}
        if set(self_values)!=set(ref_values)!=set(legal):raise AssertionError('Q legal roots')
        if set(self_values)!=set(legal):raise AssertionError('Q bank incomplete')
        d_self=q_wait_delta(self_values,legal);d_ref=q_wait_delta(ref_values,legal)
        if d_self is None:continue
        item=dict(arm=row['arm'],train_seed=int(row['train_seed']),iteration=int(row['iteration']),
            case=row['case'],physical_seed=int(row['physical_seed']),tick=int(row['tick']),
            clock_bin=min(int(row['tick'])//16,2),source_action=int(row['source_action']),
            legal_count=len(legal),q_self_wait_delta=d_self,q_ref_wait_delta=d_ref,
            q_preference_difference=d_ref-d_self,
            self_favors_wait=int(d_self<0),ref_favors_wait=int(d_ref<0),
            self_favors_wait_ge5=int(d_self<=-5),ref_favors_wait_ge5=int(d_ref<=-5))
        rows.append(item);add_row(out/'q_wait.partial.csv',item)
    if not rows:raise RuntimeError('no service-legal Q states')
    shutil.copy2(out/'q_wait.partial.csv',out/'q_wait.csv')
    return rows


def replay(profile,out,selection,models,dev,selected,deadline):
    trains=TRAIN if profile=='dev' else TRAIN[:1]
    physicals=DEV if profile=='dev' else DEV[:1]
    grid=[(case,arm,train,label,physical) for case in CASES for arm in ARMS
          for train in trains for label in ('initial','selected') for physical in physicals]
    episodes=[]
    for case,arm,train,label,physical in tqdm(grid,desc='V11 initial/selected action replay',unit='episode'):
        deadline();cfg=configs()[case];actor=models[(arm,train,label)]
        iteration=0 if label=='initial' else selection[arm][str(train)]['iteration']
        env=FrameworkEnv(cfg);env.reset(seed=physical)
        waits=0;service_starts=0;legal_service=0
        for tick in range(HORIZON):
            state=env.inner.state
            _,admitted,legal,obs=legal_present(cfg,state,tick)
            action=policy_action(actor,obs,legal)
            risk=ACTIONS.index(risk_matching(cfg,admitted,tick))
            has_service=any(ACTIONS[i] for i in legal)
            queued_cm=sum(r is not None and r.started_at is None and r.kind=='corrective'
                          for r in admitted.requests)
            item=dict(case=case,arm=arm,train_seed=train,checkpoint_label=label,
                checkpoint_iteration=iteration,physical_seed=physical,tick=tick,
                legal_service=int(has_service),chosen_action=action,risk_action=risk,
                legal_count=len(legal),chosen_cardinality=len(ACTIONS[action]),
                queued_cm=queued_cm)
            add_row(out/'actions.partial.csv',item)
            legal_service+=has_service;waits+=action==0 and has_service
            service_starts+=len(ACTIONS[action])
            env.step(action)
        metrics=env.episode_metrics();source=dev[(case,arm,train,iteration,physical)]
        for metric in METRICS:
            if not np.isclose(float(metrics[metric]),float(source[metric]),atol=1e-8,rtol=0):
                raise RuntimeError(f'DEV replay {case}/{arm}/{train}/{label}/{metric}')
            if label=='selected':
                original=selected[(case,arm,train,physical)]
                if not np.isclose(float(metrics[metric]),float(original[metric]),atol=1e-8,rtol=0):
                    raise RuntimeError(f'selected replay {metric}')
        record=dict(case=case,arm=arm,train_seed=train,checkpoint_label=label,
                    checkpoint_iteration=iteration,physical_seed=physical,
                    legal_service_opportunities=legal_service,wait_with_legal_service=waits,
                    H_service_starts=service_starts,
                    **{k:metrics[k] for k in METRICS})
        episodes.append(record);add_row(out/'episodes.partial.csv',record)
    shutil.copy2(out/'actions.partial.csv',out/'actions.csv')
    shutil.copy2(out/'episodes.partial.csv',out/'episodes.csv')
    return episodes


def summarize(q,episodes):
    result={'q':{},'episodes':{}}
    for arm in ARMS:
        for train in TRAIN:
            cell=[r for r in q if r['arm']==arm and r['train_seed']==train]
            if cell:
                result['q'][f'{arm}/{train}']=dict(states=len(cell),
                    self_favors_wait=sum(r['self_favors_wait'] for r in cell),
                    ref_favors_wait=sum(r['ref_favors_wait'] for r in cell),
                    self_favors_wait_ge5=sum(r['self_favors_wait_ge5'] for r in cell),
                    ref_favors_wait_ge5=sum(r['ref_favors_wait_ge5'] for r in cell),
                    mean_q_preference_difference=float(np.mean([r['q_preference_difference'] for r in cell])))
            for label in ('initial','selected'):
                for case in CASES:
                    rows=[r for r in episodes if r['arm']==arm and r['train_seed']==train and
                          r['checkpoint_label']==label and r['case']==case]
                    if rows:
                        result['episodes'][f'{case}/{arm}/{train}/{label}']=dict(episodes=len(rows),
                            legal_service_opportunities=sum(r['legal_service_opportunities'] for r in rows),
                            wait_with_legal_service=sum(r['wait_with_legal_service'] for r in rows),
                            mean_H_service_starts=float(np.mean([r['H_service_starts'] for r in rows])),
                            mean_queued_cm_ticks=float(np.mean([r['queued_cm_ticks'] for r in rows])),
                            mean_objective=float(np.mean([r['objective'] for r in rows])))
    return result


def run(profile,output):
    if profile not in ('smoke','dev'):raise ValueError(profile)
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    start=time.monotonic();limit=180 if profile=='smoke' else 600
    def deadline():
        if time.monotonic()-start>=limit:raise TimeoutError('V11 WAIT bias timeout')
    rel=[str(p.relative_to(ROOT)) for p in SOURCES]
    dirty=subprocess.check_output(['git','diff','--name-only','HEAD','--',*rel],cwd=ROOT,text=True).splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*rel],cwd=ROOT,text=True).splitlines()
    if profile=='dev' and (dirty or untracked):raise RuntimeError('uncommitted diagnostic source')
    source,selection,models,hashes,development,selected=load_inputs()
    manifest=dict(status='RUNNING',phase='preflight',profile=profile,version=VERSION,
        started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        invocation=[sys.executable,*sys.argv],cwd=str(ROOT),
        runtime=dict(python=platform.python_version(),platform=platform.platform()),
        timeout_seconds=limit,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES},
        v11_manifest_sha256=sha(V11/'manifest.json'),v11_selection_sha256=sha(V11/'selection.json'),
        v11_development_sha256=sha(V11/'development.csv'),v11_episodes_sha256=sha(V11/'episodes.csv'),
        v11_q_sha256=sha(V11/'counterfactual_states.csv'),checkpoint_hashes=hashes,
        cases=list(CASES),training_seeds=list(TRAIN if profile=='dev' else TRAIN[:1]),
        physical_seeds=list(DEV if profile=='dev' else DEV[:1]))
    save(out/'manifest.json',manifest);shutil.copy2(PLAN,out/'protocol.md')
    try:
        manifest['phase']='q_scan';save(out/'manifest.json',manifest)
        q=scan_q(profile,out,deadline)
        manifest['phase']='policy_replay';save(out/'manifest.json',manifest)
        episodes=replay(profile,out,selection,models,development,selected,deadline)
        save(out/'summary.json',summarize(q,episodes))
        if manifest['source_hashes']!={str(p.relative_to(ROOT)):sha(p) for p in SOURCES}:
            raise RuntimeError('source changed during diagnostic')
        manifest.update(status='COMPLETED',phase='done',elapsed_seconds=time.monotonic()-start,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            counts=dict(q_states=len(q),episodes=len(episodes),decisions=len(episodes)*HORIZON),
            artifact_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
        save(out/'manifest.json',manifest);return out
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
