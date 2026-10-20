"""Paired closed-loop V9 DEV assignment-only intervention."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES,HORIZON,FrameworkEnv,configs
from ht_pdm_fjsp.maintenance_framework_v9_campaign import legal_present,policy_action
from diagnose_v9_actions import ACTIONS,ARMS,DEV,ROOT,TRAIN,V9,load_sources,sha,save,add_row

PLAN=ROOT/'research_result/v9_assignment_intervention_plan_vi.md'
VERSION='v9_assignment_only_closed_loop_dev_v1'
METRICS=('objective','maintenance_cost','unavailability_cost','failure_cost','tail_cost',
         'H_availability','H_failures','queued_cm_ticks','queued_pm_ticks',
         'H_pending','end_pending','service_starts','requests_created','requests_completed')
SOURCES=tuple(ROOT/path for path in (
    'research_result/intervene_v9_assignment.py',
    'research_result/diagnose_v9_actions.py',
    'research_result/v9_assignment_intervention_plan_vi.md',
    'src/ht_pdm_fjsp/maintenance_framework_v9_campaign.py',
    'src/ht_pdm_fjsp/maintenance_framework_v1.py',
    'src/ht_pdm_fjsp/maintenance_request_policies.py',
    'src/ht_pdm_fjsp/maintenance_requests.py',
    'src/ht_pdm_fjsp/maintenance_dispatch.py',
))


def duration(config,action):
    return sum(config.service_time[m][j] for m,j in ACTIONS[action])


def assignment_only(config,chosen,legal):
    """Keep actor-selected machines; minimize known service duration, actor wins ties."""
    machines={m for m,_ in ACTIONS[chosen]}
    same=[a for a in legal if {m for m,_ in ACTIONS[a]}==machines]
    if chosen not in same:raise AssertionError('chosen action not legal')
    best=min(same,key=lambda a:(duration(config,a),a!=chosen,a))
    if {m for m,_ in ACTIONS[best]}!=machines:raise AssertionError('machine set changed')
    return best


def episode(config,actor,physical,treatment,deadline,action_writer=None,identity=None):
    env=FrameworkEnv(config);env.reset(seed=physical)
    starts=0;changes=0;saved=0;ties=0
    for tick in range(HORIZON):
        deadline()
        _,_,legal,obs=legal_present(config,env.inner.state,tick)
        chosen=policy_action(actor,obs,legal)
        action=assignment_only(config,chosen,legal) if treatment else chosen
        if treatment:
            changes+=action!=chosen
            saved+=duration(config,chosen)-duration(config,action)
            ties+=action==chosen and sum(
                {m for m,_ in ACTIONS[a]}=={m for m,_ in ACTIONS[chosen]} and
                duration(config,a)==duration(config,chosen) for a in legal)>1
        starts+=len(ACTIONS[action])
        if action_writer is not None:
            action_writer(dict(**identity,treatment=int(treatment),tick=tick,
                actor_action=chosen,executed_action=action,
                actor_duration=duration(config,chosen),executed_duration=duration(config,action)))
        _,_,terminated,_,_=env.step(action)
    if not terminated:raise AssertionError('episode did not finish')
    metrics=env.episode_metrics()
    if starts>metrics['service_starts']:raise AssertionError('start count')
    return metrics,dict(H_service_starts=starts,changed_assignments=changes,
                        known_duration_saved=saved,actor_minimum_ties=ties)


def summarize(rows):
    cells=defaultdict(list)
    for row in rows:
        cells[(row['case'],row['arm'],row['train_seed'])].append(row)
    result={}
    for key,cell in sorted(cells.items()):
        if len(cell) not in (1,16):raise AssertionError('incomplete seed cell')
        fields=('objective','maintenance_cost','unavailability_cost','failure_cost',
                'tail_cost','H_availability','H_failures','queued_cm_ticks',
                'queued_pm_ticks','H_pending','end_pending','service_starts',
                'H_service_starts','changed_assignments','known_duration_saved')
        result['/'.join(map(str,key))]={
            'episodes':len(cell),
            'control_mean_objective':float(np.mean([x['control_objective'] for x in cell])),
            'treatment_mean_objective':float(np.mean([x['treatment_objective'] for x in cell])),
            'paired_mean':{field:float(np.mean([x[f'treatment_{field}']-x[f'control_{field}']
                                              for x in cell])) for field in fields
                           if field not in ('changed_assignments','known_duration_saved')},
            'total_changed_assignments':sum(x['treatment_changed_assignments'] for x in cell),
            'total_known_duration_saved':sum(x['treatment_known_duration_saved'] for x in cell),
            'cost_pairs':dict(better=sum(x['treatment_objective']<x['control_objective'] for x in cell),
                              equal=sum(x['treatment_objective']==x['control_objective'] for x in cell),
                              worse=sum(x['treatment_objective']>x['control_objective'] for x in cell))}
    return result


def run(profile,output):
    if profile not in ('smoke','dev'):raise ValueError(profile)
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    start=time.monotonic();limit=180 if profile=='smoke' else 900
    def deadline():
        if time.monotonic()-start>=limit:raise TimeoutError('assignment intervention timeout')
    rel=[str(p.relative_to(ROOT)) for p in SOURCES]
    dirty=subprocess.check_output(['git','diff','--name-only','HEAD','--',*rel],cwd=ROOT,text=True).splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*rel],
                                      cwd=ROOT,text=True).splitlines()
    if profile=='dev' and (dirty or untracked):raise RuntimeError('uncommitted intervention source')
    models,reference,_,hashes=load_sources()
    seeds=DEV if profile=='dev' else DEV[:1]
    manifest=dict(status='RUNNING',profile=profile,phase='preflight',version=VERSION,
        started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        invocation=[sys.executable,*sys.argv],cwd=str(ROOT),
        runtime=dict(python=platform.python_version(),platform=platform.platform()),
        timeout_seconds=limit,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES},
        v9_manifest_sha256=sha(V9/'manifest.json'),v9_selection_sha256=sha(V9/'selection.json'),
        v9_episodes_sha256=sha(V9/'episodes.csv'),selected_checkpoint_hashes=hashes,
        physical_seeds=list(seeds),training_seeds=list(TRAIN),cases=list(CASES),arms=list(ARMS))
    save(out/'manifest.json',manifest);shutil.copy2(PLAN,out/'protocol.md')
    try:
        manifest['phase']='paired_episodes';save(out/'manifest.json',manifest)
        rows=[]
        grid=[(case,arm,train,physical) for case in CASES for arm in ARMS
              for train in TRAIN for physical in seeds]
        for case,arm,train,physical in tqdm(grid,desc='V9 assignment intervention',unit='pair'):
            deadline();cfg=configs()[case];actor=models[(arm,train)]
            ident=dict(case=case,arm=arm,train_seed=train,physical_seed=physical)
            control,cmeta=episode(cfg,actor,physical,False,deadline,
                                  lambda row:add_row(out/'actions.partial.csv',row),ident)
            original=reference[(case,arm,train,physical)]
            for metric in METRICS:
                if not np.isclose(float(control[metric]),float(original[metric]),atol=1e-8,rtol=0):
                    raise RuntimeError(f'control replay mismatch {ident} {metric}')
            treated,tmeta=episode(cfg,actor,physical,True,deadline,
                                  lambda row:add_row(out/'actions.partial.csv',row),ident)
            row=dict(**ident)
            for prefix,metrics,meta in (('control',control,cmeta),('treatment',treated,tmeta)):
                row.update({f'{prefix}_{k}':metrics[k] for k in METRICS})
                row.update({f'{prefix}_{k}':v for k,v in meta.items()})
            rows.append(row);add_row(out/'episodes.partial.csv',row)
        shutil.copy2(out/'episodes.partial.csv',out/'episodes.csv')
        shutil.copy2(out/'actions.partial.csv',out/'actions.csv')
        save(out/'summary.json',summarize(rows))
        if manifest['source_hashes']!={str(p.relative_to(ROOT)):sha(p) for p in SOURCES}:
            raise RuntimeError('source changed during run')
        manifest.update(status='COMPLETED',phase='done',elapsed_seconds=time.monotonic()-start,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            counts=dict(pairs=len(rows),episodes=2*len(rows),decisions=2*len(rows)*HORIZON),
            artifact_hashes={p.name:sha(p) for p in out.iterdir()
                             if p.is_file() and p.name!='manifest.json'})
        save(out/'manifest.json',manifest)
        return out
    except Exception as exc:
        manifest.update(status='INCOMPLETE' if isinstance(exc,TimeoutError) else 'FAILED',
            error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        save(out/'manifest.json',manifest);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('smoke','dev'),required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    print(json.dumps(dict(status='COMPLETED',output_dir=str(run(args.profile,args.output_dir)))))

if __name__=='__main__':main()
