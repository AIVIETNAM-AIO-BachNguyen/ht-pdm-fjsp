"""Prelocked DEV-only selected V10 versus rollout action/value trace."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
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

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import CASES,HORIZON,FrameworkEnv,catalogue,configs,vector_observation
from ht_pdm_fjsp.maintenance_framework_v10_campaign import load_checkpoint,legal_present,policy_action
from ht_pdm_fjsp.maintenance_framework_v7_policy import canonical_mask
from ht_pdm_fjsp.maintenance_request_policies import choose,risk_matching
from ht_pdm_fjsp.maintenance_requests import state_from_dict,validate_state

ROOT=Path(__file__).resolve().parent.parent
PLAN=ROOT/'research_result/v10_rollout_gap_diagnostic_plan_vi.md'
V10=ROOT/'research_result/framework_v10_dev_20261020T072256Z'
VERSION='v10_selected_rollout_gap_dev_v1'
ARMS=('assignment_credit','joint_credit')
TRAIN=tuple(range(70400000,70400005))
DEV=tuple(range(70410000,70410016))
FORECAST_ANCHOR=70460000
ACTIONS=catalogue()
SOURCES=tuple(ROOT/path for path in (
    'research_result/diagnose_v10_rollout_gap.py',
    'research_result/v10_rollout_gap_diagnostic_plan_vi.md',
    'src/ht_pdm_fjsp/maintenance_framework_v10_campaign.py',
    'src/ht_pdm_fjsp/maintenance_framework_v7_policy.py',
    'src/ht_pdm_fjsp/maintenance_framework_v1.py',
    'src/ht_pdm_fjsp/maintenance_request_policies.py',
    'src/ht_pdm_fjsp/maintenance_requests.py',
    'src/ht_pdm_fjsp/maintenance_dispatch.py',
))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def add_row(path,row):
    path=Path(path);new=not path.exists()
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if new:writer.writeheader()
        writer.writerow(row)

def clock_bin(tick):return ('early','middle','late')[tick//16]

def rank(case,arm,train,physical,tick):
    key=f'{VERSION}:{case}:{arm}:{train}:{physical}:{tick}'
    return hashlib.sha256(key.encode()).hexdigest()

def action_class(left,right):
    """Mutually exclusive action difference, with class names relative to right."""
    a,b=ACTIONS[left],ACTIONS[right]
    if a==b:return 'match'
    if not a and b:return 'wait_vs_service'
    if a and not b:return 'service_vs_wait'
    if len(a)<len(b):return 'lower_cardinality'
    if len(a)>len(b):return 'higher_cardinality'
    if {m for m,_ in a}!={m for m,_ in b}:return 'machine_subset'
    return 'technician_assignment'


def best_root(scores,risk):
    ordered=(risk,*(a for a in sorted(scores) if a!=risk))
    return min(ordered,key=lambda a:scores[a])


def load_sources():
    manifest=json.loads((V10/'manifest.json').read_text())
    if manifest['status']!='COMPLETED' or manifest['profile']!='dev':
        raise RuntimeError('V10 source run incomplete')
    selection_path=V10/'selection.json'
    if sha(selection_path)!=manifest['selection_sha256']:
        raise RuntimeError('selection hash changed')
    selection=json.loads(selection_path.read_text())
    models={};hashes={}
    for arm in ARMS:
        for seed in TRAIN:
            item=selection[arm][str(seed)]
            path=V10/item['checkpoint']
            if sha(path)!=item['sha256']:raise RuntimeError('selected model hash changed')
            models[(arm,seed)]=load_checkpoint(path)
            hashes[f'{arm}/{seed}']=item['sha256']
    episode_path=V10/'episodes.csv'
    reference={(r['case'],r['controller'],int(r['train_seed']),int(r['seed'])):r
        for r in csv.DictReader(episode_path.open(newline='')) if r['controller'] in ARMS}
    if len(reference)!=len(CASES)*len(ARMS)*len(TRAIN)*len(DEV):
        raise RuntimeError('source learned episode grid incomplete')
    return models,reference,manifest,hashes


def trace(profile,models,reference,out,deadline):
    seeds=DEV if profile=='dev' else DEV[:1]
    grid=[(case,arm,train,physical) for case in CASES for arm in ARMS
          for train in TRAIN for physical in seeds]
    candidates=defaultdict(list)
    summary=dict(episodes=0,decisions=0,candidate_states=0,
                 by_source={},by_case_arm_bin={})
    for case,arm,train,physical in tqdm(grid,desc='V10 selected action replay',unit='episode'):
        deadline();cfg=configs()[case];env=FrameworkEnv(cfg);env.reset(seed=physical)
        actor=models[(arm,train)];starts=0
        for tick in range(HORIZON):
            old=env.inner.state
            submissions,admitted,legal,obs=legal_present(cfg,old,tick)
            chosen=policy_action(actor,obs,legal)
            risk=ACTIONS.index(risk_matching(cfg,admitted,tick))
            category=action_class(chosen,risk)
            nonempty=any(ACTIONS[index] for index in legal)
            queued_cm=sum(r is not None and r.started_at is None and r.kind=='corrective'
                          for r in admitted.requests)
            queued_pm=sum(r is not None and r.started_at is None and r.kind=='preventive'
                          for r in admitted.requests)
            record=dict(case=case,source_arm=arm,train_seed=train,physical_seed=physical,
                tick=tick,clock_bin=clock_bin(tick),rank=rank(case,arm,train,physical,tick),
                chosen_action=chosen,risk_action=risk,category=category,
                legal_count=len(legal),legal_nonempty=int(nonempty),
                chosen_cardinality=len(ACTIONS[chosen]),risk_cardinality=len(ACTIONS[risk]),
                queued_cm=queued_cm,queued_pm=queued_pm,
                free_technicians=sum(x==0 for x in old.physical.remaining),
                chosen_duration=sum(cfg.service_time[m][j] for m,j in ACTIONS[chosen]),
                risk_duration=sum(cfg.service_time[m][j] for m,j in ACTIONS[risk]))
            add_row(out/'actions.partial.csv',record)
            summary['decisions']+=1;starts+=len(ACTIONS[chosen])
            key=f'{case}/{arm}/{train}'
            cell=summary['by_source'].setdefault(key,dict(decisions=0,wait_with_legal_service=0,
                wait_without_legal_service=0,service_starts=0,risk_service_starts=0,
                queued_cm_ticks_preaction=0,categories=Counter()))
            cell['decisions']+=1
            cell['wait_with_legal_service']+=chosen==0 and nonempty
            cell['wait_without_legal_service']+=chosen==0 and not nonempty
            cell['service_starts']+=len(ACTIONS[chosen]);cell['risk_service_starts']+=len(ACTIONS[risk])
            cell['queued_cm_ticks_preaction']+=queued_cm
            cell['categories'][category]+=1
            bin_key=f'{case}/{arm}/{train}/{clock_bin(tick)}'
            bin_cell=summary['by_case_arm_bin'].setdefault(bin_key,dict(decisions=0,
                wait_with_legal_service=0,lower_cardinality=0,machine_subset=0,
                technician_assignment=0))
            bin_cell['decisions']+=1
            bin_cell['wait_with_legal_service']+=chosen==0 and nonempty
            for label in ('lower_cardinality','machine_subset','technician_assignment'):
                bin_cell[label]+=category==label
            if len(legal)>=2:
                summary['candidate_states']+=1
                candidates[(case,arm,train,clock_bin(tick))].append(dict(
                    case=case,source_arm=arm,train_seed=train,physical_seed=physical,
                    tick=tick,clock_bin=clock_bin(tick),rank=record['rank'],
                    original=asdict(old),chosen_action=chosen,risk_action=risk))
            _,_,terminated,_,_=env.step(chosen)
        if not terminated:raise AssertionError('episode did not finish')
        metrics=env.episode_metrics()
        old=reference[(case,arm,train,physical)]
        for metric in ('objective','H_availability','H_failures','queued_cm_ticks',
                       'queued_pm_ticks','H_pending','end_pending','service_starts',
                       'maintenance_cost','unavailability_cost','failure_cost',
                       'tail_cost','requests_created','requests_completed'):
            if not np.isclose(float(metrics[metric]),float(old[metric]),atol=1e-8,rtol=0):
                raise RuntimeError(f'episode replay mismatch {metric}')
        if starts>metrics['service_starts']:
            raise RuntimeError('H service starts exceed full episode starts')
        summary['episodes']+=1
        add_row(out/'episodes.partial.csv',dict(case=case,source_arm=arm,train_seed=train,
            physical_seed=physical,objective=metrics['objective'],H_availability=metrics['H_availability'],
            H_failures=metrics['H_failures'],queued_cm_ticks=metrics['queued_cm_ticks'],
            H_service_starts=starts,service_starts=metrics['service_starts'],
            H_pending=metrics['H_pending'],end_pending=metrics['end_pending']))
    cap=2 if profile=='dev' else 1
    selected=[];coverage={}
    for case in CASES:
        for arm in ARMS:
            for train in TRAIN:
                for b in ('early','middle','late'):
                    key=(case,arm,train,b)
                    cell=sorted(candidates[key],key=lambda r:r['rank'])
                    chosen=cell[:cap];selected+=chosen
                    coverage['/'.join(map(str,key))]=dict(eligible=len(cell),selected=len(chosen))
    selected.sort(key=lambda r:r['rank'])
    summary['sampling']=coverage
    return selected,summary


def root_probe(profile,selected,models,out,summary,deadline):
    scenarios=16 if profile=='dev' else 1
    summary['root_states']=0;summary['forecast_transitions']=0
    summary['root_by_source']={}
    with (out/'states.jsonl').open('x') as stream:
        for state_id,row in enumerate(tqdm(selected,desc='V10 bounded root-value check',unit='state')):
            deadline();case=row['case'];cfg=configs()[case];tick=row['tick']
            state=state_from_dict(row['original']);validate_state(cfg,state,tick)
            _,admitted,legal,obs=legal_present(cfg,state,tick)
            if len(legal)<2:raise RuntimeError('selected state not eligible')
            risk=ACTIONS.index(risk_matching(cfg,admitted,tick))
            forecast_seed=role_seed(FORECAST_ANCHOR,f'{VERSION}:{row["rank"]}')
            _,planner_matching,diagnostic=choose(cfg,state,tick,'threshold','rollout',
                policy_horizon=HORIZON,forecast_seed=forecast_seed,
                scenarios=scenarios,forecast_horizon=6,check_timeout=deadline)
            scores={ACTIONS.index(tuple(x['matching'])):float(x['expected_cost'])
                    for x in diagnostic['root_scores']}
            if set(scores)!=set(legal):raise RuntimeError('incomplete root scores')
            planner=ACTIONS.index(planner_matching)
            if planner!=best_root(scores,risk):raise RuntimeError('planner tie rule')
            allowed=np.zeros(len(ACTIONS),dtype=bool);allowed[list(legal)]=True
            canonical=tuple(i for i,ok in enumerate(canonical_mask(cfg,ACTIONS,allowed)) if ok)
            canonical_best=min(scores[i] for i in canonical)
            canonical_gap=canonical_best-scores[planner]
            source=policy_action(models[(row['source_arm'],row['train_seed'])],obs,legal)
            if source!=row['chosen_action'] or risk!=row['risk_action']:
                raise RuntimeError('source/risk choice changed')
            record=dict(state_id=state_id,**row,forecast_seed=forecast_seed,
                legal_actions=legal,planner_action=planner,scores=scores,
                planner_canonical=int(planner in canonical),canonical_gap=canonical_gap,
                forecast_transitions=diagnostic['forecast_transitions'])
            stream.write(json.dumps(record,sort_keys=True)+'\n')
            summary['root_states']+=1
            summary['forecast_transitions']+=diagnostic['forecast_transitions']
            key=f'{case}/{row["source_arm"]}/{row["train_seed"]}'
            cell=summary['root_by_source'].setdefault(key,dict(states=0,source_regret_sum=0.,
                risk_regret_sum=0.,source_gap_ge_1=0,source_gap_ge_5=0,
                source_gap_ge_10=0,planner_excluded_by_canonical=0,categories=Counter()))
            cell['states']+=1
            for arm in ARMS:
                pred=policy_action(models[(arm,row['train_seed'])],obs,legal)
                gap=scores[pred]-scores[planner]
                risk_gap=scores[risk]-scores[planner]
                add_row(out/'choices.partial.csv',dict(state_id=state_id,case=case,
                    source_arm=row['source_arm'],train_seed=row['train_seed'],
                    physical_seed=row['physical_seed'],tick=tick,clock_bin=row['clock_bin'],
                    policy_arm=arm,policy_action=pred,source_action=source,
                    risk_action=risk,planner_action=planner,
                    versus_risk=action_class(pred,risk),
                    versus_planner=action_class(pred,planner),
                    cardinality_delta_vs_planner=len(ACTIONS[pred])-len(ACTIONS[planner]),
                    full_regret=gap,risk_regret=risk_gap))
                if arm==row['source_arm']:
                    cell['source_regret_sum']+=gap;cell['risk_regret_sum']+=risk_gap
                    cell['planner_excluded_by_canonical']+=planner not in canonical
                    cell['categories'][action_class(pred,planner)]+=1
                    for threshold in (1,5,10):cell[f'source_gap_ge_{threshold}']+=gap>=threshold
    return summary


def run(profile,output):
    if profile not in ('smoke','dev'):raise ValueError(profile)
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    start=time.monotonic();limit=300 if profile=='smoke' else 1200
    def deadline():
        if time.monotonic()-start>=limit:raise TimeoutError('V10 action diagnostic timeout')
    rel=[str(p.relative_to(ROOT)) for p in SOURCES]
    dirty=subprocess.check_output(['git','diff','--name-only','HEAD','--',*rel],cwd=ROOT,text=True).splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*rel],
                                      cwd=ROOT,text=True).splitlines()
    if profile=='dev' and (dirty or untracked):raise RuntimeError('uncommitted diagnostic source')
    models,reference,source_manifest,hashes=load_sources()
    manifest=dict(status='RUNNING',profile=profile,phase='preflight',
        started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        invocation=[sys.executable,*sys.argv],cwd=str(ROOT),
        runtime=dict(python=platform.python_version(),platform=platform.platform()),
        timeout_seconds=limit,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES},
        v10_manifest_sha256=sha(V10/'manifest.json'),v10_selection_sha256=sha(V10/'selection.json'),
        v10_episodes_sha256=sha(V10/'episodes.csv'),selected_checkpoint_hashes=hashes,
        forecast_anchor=FORECAST_ANCHOR,physical_seeds=list(DEV if profile=='dev' else DEV[:1]))
    save(out/'manifest.json',manifest);shutil.copy2(PLAN,out/'protocol.md')
    try:
        manifest['phase']='trajectory_trace';save(out/'manifest.json',manifest)
        selected,summary=trace(profile,models,reference,out,deadline)
        shutil.copy2(out/'actions.partial.csv',out/'actions.csv')
        shutil.copy2(out/'episodes.partial.csv',out/'episodes.csv')
        manifest['phase']='root_probe';save(out/'manifest.json',manifest)
        summary=root_probe(profile,selected,models,out,summary,deadline)
        shutil.copy2(out/'choices.partial.csv',out/'choices.csv')
        save(out/'summary.json',summary)
        if manifest['source_hashes']!={str(p.relative_to(ROOT)):sha(p) for p in SOURCES}:
            raise RuntimeError('source changed during run')
        manifest.update(status='COMPLETED',phase='done',elapsed_seconds=time.monotonic()-start,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            counts=dict(episodes=summary['episodes'],decisions=summary['decisions'],
                        root_states=summary['root_states'],
                        forecast_transitions=summary['forecast_transitions']),
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
