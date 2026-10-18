"""Sealed C1 confirmation with frozen request adapter and DEV-selected rule."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import technician_diverse_maintenance_experiment as core
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_opportunity_cost import Journal, write_csv, write_json
from ht_pdm_fjsp.technician_diverse_maintenance_env import VERSION, family_config
from ht_pdm_fjsp.technician_relational_prefix_distillation import (
    DEFAULT_TEACHERS, freeze_teachers, prepare_warm_job,
)


ROOT=Path(__file__).resolve().parents[2]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def settings(profile):
    if profile not in ('smoke','confirmation'):
        raise ValueError('sealed confirmation or engineering smoke only')
    cfg=core.settings('smoke' if profile=='smoke' else 'development')
    if profile=='smoke':
        return cfg
    raw=read_json(core.REGISTRY)['panels']['confirmation']
    cfg.update(profile='confirmation',train_seeds=raw['train_seeds'],
        development_seeds=list(range(raw['development_start'],
            raw['development_start']+raw['development_count'])),
        test_seeds=list(range(raw['test_start'],raw['test_start']+raw['test_count'])),
        analysis_seed=raw['analysis_seed'],training_processes=8,
        max_seconds=259200,bootstrap_samples=40000)
    return cfg


def prior_gate(profile,prior):
    manifest=read_json(prior/'manifest.json')
    summary=read_json(prior/'summary.json')
    audit=read_json(prior/'audit.json')
    assert manifest['protocol']==VERSION and manifest['status']=='COMPLETED'
    assert audit['status']=='PASS'
    if profile=='confirmation':
        if (manifest['profile']!='development' or manifest['test_opened']
                or not summary['development_screen_pass']):
            raise RuntimeError('C1 development gate failed or scientific TEST opened')
    else:
        if manifest['profile']!='smoke':
            raise RuntimeError('engineering smoke requires smoke prior')
    if sha(prior/'protocol.md')!=sha(core.PLAN) or sha(prior/'seed_registry.json')!=sha(core.REGISTRY):
        raise RuntimeError('C1 protocol/seed registry changed')
    for snapshot in (prior/'source_snapshot').glob('*.py'):
        current=ROOT/'src/ht_pdm_fjsp'/snapshot.name
        if not current.is_file() or sha(snapshot)!=sha(current):
            raise RuntimeError(f'C1 frozen source changed: {snapshot.name}')
    selection=read_json(prior/'reference_selection.json')
    if selection['selected_rule'] not in ('fcfs','risk') or selection['test_opened']:
        raise RuntimeError('C1 reference selection was not frozen on DEV')
    return dict(path=str(prior),manifest_sha256=sha(prior/'manifest.json'),
        summary_sha256=sha(prior/'summary.json'),audit_sha256=sha(prior/'audit.json'),
        reference_selection_sha256=sha(prior/'reference_selection.json'),
        selected_rule=selection['selected_rule'],
        engineering_gate_bypass=profile=='smoke')


def independent_development_gate(prior,audit_dir):
    """Bind an independent C1 DEV PASS to the exact frozen DEV artifact."""
    if audit_dir is None:
        raise ValueError('C1 scientific confirmation requires independent DEV audit')
    audit_dir=Path(audit_dir).resolve()
    report=read_json(audit_dir/'audit.json')
    verifier=audit_dir/'audit_diverse_maintenance_development.py'
    if (report.get('status')!='PASS' or report.get('classification')!='INDEPENDENT_DEVELOPMENT_AUDIT'
            or report.get('development_screen_pass') is not True
            or report.get('scientific_test_opened') is not False
            or Path(report.get('source_artifact','')).resolve()!=prior
            or report.get('source_manifest_sha256')!=sha(prior/'manifest.json')
            or report.get('verifier_sha256')!=sha(verifier)):
        raise RuntimeError('C1 independent DEV audit failed or does not bind to DEV')
    return dict(path=str(audit_dir),report_sha256=sha(audit_dir/'audit.json'),
        verifier_sha256=sha(verifier))


def expected_grid(cfg):
    configs=[tuple(c) for c in cfg['configs']]
    return {(f,n,arm,t,s) for f,n in configs
        for arm in (*core.ARMS,'fcfs','risk')
        for t in (cfg['train_seeds'] if arm in core.ARMS else [-1])
        for s in cfg['test_seeds']}


def summarize(rows,cfg,selected_rule,coordination):
    configs=[tuple(c) for c in cfg['configs']]
    training=cfg['train_seeds'];physical=cfg['test_seeds']
    key=lambda r:(r['family'],int(r['n']),r['method'],int(r['train_seed']),int(r['seed']))
    lookup={key(r):r for r in rows}
    if len(rows)!=len(expected_grid(cfg)) or set(lookup)!=expected_grid(cfg):
        raise RuntimeError('C1 exact confirmation TEST grid')
    fields=('normalized_cost','availability','normalized_pending','normalized_failures')
    arrays={}
    for f,n in configs:
        arrays[f,n]={}
        for arm in core.ARMS:
            arrays[f,n][arm]=np.asarray([[[float(lookup[f,n,arm,t,s][m]) for m in fields]
                for s in physical] for t in training])
        for method in ('fcfs','risk'):
            panel=np.asarray([[float(lookup[f,n,method,-1,s][m]) for m in fields]
                for s in physical])
            arrays[f,n][method]=np.tile(panel[None],(len(training),1,1))
    means={c:{a:v.mean((0,1)) for a,v in panel.items()} for c,panel in arrays.items()}
    families=sorted({f for f,_ in configs})
    def estimates(values):
        out={}
        full=np.mean([values[c]['reservation_ppo'][0] for c in configs],axis=0)
        masked=np.mean([values[c]['mask_only_ppo'][0] for c in configs],axis=0)
        rule=np.mean([values[c][selected_rule][0] for c in configs],axis=0)
        out['pooled_gain_full_vs_mask']=1-full/masked
        out['pooled_gain_full_vs_heuristic']=1-full/rule
        for f in families:
            group=[c for c in configs if c[0]==f]
            cf=np.mean([values[c]['reservation_ppo'][0] for c in group],axis=0)
            cm=np.mean([values[c]['mask_only_ppo'][0] for c in group],axis=0)
            out[f'{f}_gain_full_vs_mask']=1-cf/cm
            for name,index in (('availability',1),('pending_per_machine',2),
                               ('failures_per_machine',3)):
                out[f'{f}_{name}_full_minus_mask']=np.mean([
                    values[c]['reservation_ppo'][index]-values[c]['mask_only_ppo'][index]
                    for c in group],axis=0)
        return out
    point=estimates(means)
    samples=200 if cfg['profile']=='smoke' else 40000
    rng=np.random.default_rng(role_seed(cfg['analysis_seed'],VERSION+':confirmation_bootstrap'))
    draws={name:[] for name in point}
    nt,ns=len(training),len(physical)
    for start in tqdm(range(0,samples,128),desc='C1 crossed bootstrap',unit='batch'):
        batch=min(128,samples-start)
        indices=rng.integers(nt,size=(batch,nt))
        tw=(indices[:,:,None]==np.arange(nt)).mean(1)
        sampled={}
        for c,panel in arrays.items():
            indices=rng.integers(ns,size=(batch,ns))
            ew=(indices[:,:,None]==np.arange(ns)).mean(1)
            sampled[c]={a:np.einsum('bt,bs,tsd->db',tw,ew,v,optimize=True)
                for a,v in panel.items()}
        for name,values in estimates(sampled).items():draws[name].extend(values.tolist())
    alpha=(.05 if cfg['profile']=='smoke' else .025)/10
    intervals={name:dict(estimate=float(value),
        lower=float(np.quantile(draws[name],alpha)),
        upper=float(np.quantile(draws[name],1-alpha))) for name,value in point.items()}
    guards=dict(
        pooled_full_mask_gain=point['pooled_gain_full_vs_mask']>=.01 and
            intervals['pooled_gain_full_vs_mask']['lower']>0,
        family_point_noninferiority=all(point[f'{f}_gain_full_vs_mask']>=-.01 for f in families),
        pooled_heuristic_noninferiority=intervals['pooled_gain_full_vs_heuristic']['lower']>=-.01)
    for f in families:
        guards[f'{f}_health']=(
            intervals[f'{f}_availability_full_minus_mask']['lower']>=-.01 and
            intervals[f'{f}_pending_per_machine_full_minus_mask']['upper']<=.025 and
            intervals[f'{f}_failures_per_machine_full_minus_mask']['upper']<=.25)
    largest=max(n for _,n in configs)
    full_coord=[r for r in coordination if int(r['n'])==largest and
        r['method']=='reservation_ppo']
    select_times=np.asarray([float(r['decision_ms']) for r in full_coord])
    end_to_end=np.asarray([float(r['decision_ms'])+float(r['environment_ms'])
        for r in full_coord])
    guards['largest_select_latency']=float(np.quantile(select_times,.95))<=50
    guards['zero_invalid_assignments']=all(float(r['invalid_assignments'])==0 for r in rows)
    guards={name:bool(value) for name,value in guards.items()}
    per_replica={f'{f}:N{n}':{a:{str(t):dict(
        normalized_cost=float(arrays[f,n][a][i,:,0].mean()),
        availability=float(arrays[f,n][a][i,:,1].mean()),
        normalized_pending=float(arrays[f,n][a][i,:,2].mean()),
        normalized_failures=float(arrays[f,n][a][i,:,3].mean()))
        for i,t in enumerate(training)} for a in core.ARMS} for f,n in configs}
    leave_one=[]
    if nt>1:
        for omit in range(nt):
            kept=[i for i in range(nt) if i!=omit]
            panel={c:{a:v[kept].mean((0,1)) for a,v in arms.items()}
                for c,arms in arrays.items()}
            leave_one.append(dict(omitted_train_seed=training[omit],
                estimates={k:float(v) for k,v in estimates(panel).items()}))
    return dict(classification='ENGINEERING_ONLY' if cfg['profile']=='smoke' else 'C1_CONFIRMATION',
        goal_pass=all(guards.values()),guards=guards,intervals=intervals,
        inference_family_size=10,alpha_per_endpoint=alpha,
        bootstrap_draws=samples,
        selected_reference=selected_rule,
        means={f'{f}:N{n}':{a:dict(zip(fields,v.tolist()))
            for a,v in panel.items()} for (f,n),panel in means.items()},
        per_replica=per_replica,leave_one_bundle_out=leave_one,
        latency=dict(largest_n=largest,model_select_p95_ms=float(np.quantile(select_times,.95)),
            end_to_end_p95_ms=float(np.quantile(end_to_end,.95)),
            model_select_samples=len(select_times)))


def secondary_summary(rows,cfg):
    configs=[tuple(c) for c in cfg['configs']]
    expected={(f,n,arm,t,s,condition) for f,n in configs for arm in core.ARMS
        for t in cfg['train_seeds'] for s in cfg['test_seeds']
        for condition in (('final','zero_shot') if arm=='reservation_ppo' else ('final',))}
    key=lambda r:(r['family'],r['n'],r['method'],r['train_seed'],r['seed'],r['condition'])
    if len(rows)!=len(expected) or {key(r) for r in rows}!=expected:
        raise RuntimeError('C1 secondary TEST grid')
    result={}
    for f,n in configs:
        label=f'{f}:N{n}';result[label]={}
        for arm in core.ARMS:
            result[label][arm]={}
            conditions=('final','zero_shot') if arm=='reservation_ppo' else ('final',)
            for condition in conditions:
                panel=[r for r in rows if (r['family'],r['n'],r['method'],r['condition'])==
                    (f,n,arm,condition)]
                result[label][arm][condition]=dict(
                    mean_cost_per_machine=float(np.mean([r['normalized_cost'] for r in panel])),
                    per_replica={str(t):float(np.mean([r['normalized_cost'] for r in panel
                        if r['train_seed']==t])) for t in cfg['train_seeds']})
    return dict(classification='ENGINEERING_ONLY' if cfg['profile']=='smoke' else 'C1_SECONDARY',
        episodes=len(rows),results=result,
        conditions='target-scale fine-tuned selected, fixed-final, N4 warm zero-shot and scratch target-scale training reported separately')


def run(profile,out,prior,teacher_dir=None,independent_audit=None):
    cfg=settings(profile);out=Path(out).resolve();prior=Path(prior).resolve()
    if out.exists():raise FileExistsError('fresh confirmation artifact required')
    gate=prior_gate(profile,prior)
    if profile=='confirmation':
        gate['independent_audit']=independent_development_gate(prior,independent_audit)
    out.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(1)
    git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()))
    if profile=='confirmation' and git['dirty']:
        raise RuntimeError('commit and push verified confirmation code before FULL')
    manifest=dict(protocol=VERSION,status='RUNNING',profile=profile,git=git,
        test_opened=False,prior_development_gate=gate,
        invocation=dict(executable=sys.executable,argv=sys.argv,cwd=str(Path.cwd())),
        started_utc=datetime.now(timezone.utc).isoformat(),
        runtime=dict(platform=platform.platform(),python=platform.python_version(),
            numpy=np.__version__,torch=torch.__version__,device='cpu'),
        phase_wall_seconds={})
    write_json(out/'manifest.json',manifest)
    def guard():
        if time.monotonic()-start>cfg['max_seconds']:raise TimeoutError('C1 confirmation hardstop')
    try:
        seed_report=core.seed_audit()
        freeze_teachers(teacher_dir or DEFAULT_TEACHERS,out,cfg)
        write_json(out/'resolved_config.json',cfg)
        write_json(out/'seed_audit.json',seed_report)
        configs=[tuple(c) for c in cfg['configs']]
        write_json(out/'physics.json',{f'{f}:N{n}':family_config(f,n).__dict__ for f,n in configs})
        snap=out/'source_snapshot';snap.mkdir()
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
        history=out/'registry_snapshot';history.mkdir()
        for name in seed_report['historical_registry_hashes']:
            shutil.copy2(ROOT/'configs'/name,history/name)
        shutil.copy2(core.PLAN,out/'protocol.md');shutil.copy2(core.REGISTRY,out/'seed_registry.json')
        shutil.copy2(prior/'reference_selection.json',out/'frozen_reference_selection.json')
        seals={str(p.relative_to(out)):sha(p) for p in
            [*snap.iterdir(),*history.iterdir(),out/'protocol.md',out/'seed_registry.json',
             out/'physics.json',out/'resolved_config.json',out/'frozen_reference_selection.json',
             *(out/'teachers').iterdir()]}
        manifest['source_seal']=seals;write_json(out/'manifest.json',manifest)
        begin=time.monotonic();manifest['stage']='teacher_bank_and_distillation';write_json(out/'manifest.json',manifest)
        warm={}
        with ProcessPoolExecutor(max_workers=cfg['training_processes'],mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(prepare_warm_job,(cfg,t,str(out),start)) for t in cfg['train_seeds']]
            for future in tqdm(as_completed(futures),total=len(futures),desc='C1 confirmation teacher banks',unit='bundle'):
                result=future.result();guard();warm[str(result['train_seed'])]=result
        write_json(out/'warm_choices.json',warm)
        manifest['phase_wall_seconds']['teacher_bank_and_distillation']=time.monotonic()-begin
        begin=time.monotonic();manifest['stage']='training_pool';write_json(out/'manifest.json',manifest)
        jobs=[(cfg,f,n,a,t,str(out),start) for f,n in configs for a in core.ARMS for t in cfg['train_seeds']]
        results={}
        with ProcessPoolExecutor(max_workers=cfg['training_processes'],mp_context=mp.get_context('spawn')) as pool:
            futures={pool.submit(core.fit_job,job):job for job in jobs}
            for future in tqdm(as_completed(futures),total=len(futures),desc='C1 confirmation models',unit='model'):
                result=future.result();guard()
                key=f"{result['family']}:N{result['n']}:{result['arm']}:{result['train_seed']}"
                results[key]=result
                manifest['completed_models']=len(results);write_json(out/'manifest.json',manifest)
        manifest['phase_wall_seconds']['training_including_development']=time.monotonic()-begin
        choices={};audits=[];devrows=[];initial={};parameters={}
        journals={name:Journal(out/f'{name}.csv') for name in
            ('training_progress','training_episodes','development_episodes')}
        try:
            for _,f,n,a,t,_,_ in jobs:
                key=f'{f}:N{n}:{a}:{t}';result=results[key]
                choices[key]=result['choice'];audits.append(result['audit'])
                devrows.extend(result['development_rows']);initial[key]=result['initial']
                parameters[key]=result['parameters']
                directory=out/f'{f}_N{n}'/a/str(t)
                for name in journals:
                    with (directory/f'{name}.csv').open(newline='') as stream:
                        for row in csv.DictReader(stream):journals[name].add(row)
        finally:
            for journal in journals.values():journal.close()
        for f,n in configs:
            for t in cfg['train_seeds']:
                if len({initial[f'{f}:N{n}:{a}:{t}'] for a in core.ARMS})!=1:
                    raise RuntimeError('C1 paired initial weights')
        write_json(out/'checkpoint_choices.json',choices)
        write_json(out/'training_audit.json',audits)
        write_json(out/'initial_parameter_hashes.json',initial)
        write_json(out/'parameter_counts.json',parameters)
        write_json(out/'checkpoint_selection.json',dict(
            rule='minimum mean confirmation DEV cost per machine, earliest tie',
            choices_sha256=sha(out/'checkpoint_choices.json'),
            frozen_reference_rule=gate['selected_rule'],
            frozen_reference_selection_sha256=sha(out/'frozen_reference_selection.json'),
            development_episode_count=len(devrows),test_opened=False))
        if profile=='confirmation' and sha(prior/'reference_selection.json')!=gate['reference_selection_sha256']:
            raise RuntimeError('C1 prior rule selection changed before TEST')
        manifest.update(stage='test',test_opened=True,
            choices_sha256=sha(out/'checkpoint_choices.json'),
            choices_sealed_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/'manifest.json',manifest)
        begin=time.monotonic();test=[];secondary=[];latencies={}
        coordination=Journal(out/'coordination.csv');decisions=Journal(out/'decisions.csv')
        try:
            for f,n in configs:
                controllers=[(a,t) for a in core.ARMS for t in cfg['train_seeds']]
                controllers.extend((a,-1) for a in ('fcfs','risk'))
                for arm,t in controllers:
                    choice=choices.get(f'{f}:N{n}:{arm}:{t}')
                    model=core.load_model(out/choice['path'],f,n,arm,t,choice['steps']) if choice else None
                    latency=latencies.setdefault(f'{f}:N{n}:{arm}',[])
                    for physical in tqdm(cfg['test_seeds'],
                            desc=f'C1 TEST {f}/N{n}/{arm}/{t}',unit='episode'):
                        guard();test.append(core.evaluate(f,n,arm,t,physical,model,
                            coordination,decisions,latency if model is not None else None))
        finally:
            coordination.close();decisions.close()
        write_csv(out/'episodes.csv',test)
        coord=read_rows(out/'coordination.csv')
        summary=summarize(test,cfg,gate['selected_rule'],coord)
        write_json(out/'summary.json',summary)
        manifest['phase_wall_seconds']['selected_policy_test']=time.monotonic()-begin
        begin=time.monotonic();manifest['stage']='secondary_test';write_json(out/'manifest.json',manifest)
        for f,n in configs:
            for arm in core.ARMS:
                for t in cfg['train_seeds']:
                    conditions=('final','zero_shot') if arm=='reservation_ppo' else ('final',)
                    for condition in conditions:
                        step=0 if condition=='zero_shot' else cfg['env_steps']
                        path=out/f'{f}_N{n}'/arm/str(t)/'checkpoints'/f'step_{step:09d}.pt'
                        model=core.load_model(path,f,n,arm,t,step)
                        for physical in tqdm(cfg['test_seeds'],
                                desc=f'C1 secondary {f}/N{n}/{arm}/{condition}/{t}',unit='episode'):
                            guard();row=core.evaluate(f,n,arm,t,physical,model)
                            row['condition']=condition;secondary.append(row)
        write_csv(out/'secondary_episodes.csv',secondary)
        write_json(out/'secondary_summary.json',secondary_summary(secondary,cfg))
        manifest['phase_wall_seconds']['secondary_test']=time.monotonic()-begin
        # Physical and statistical confirmation is replayed again by an independent verifier.
        expected=expected_grid(cfg)
        keys={(r['family'],r['n'],r['method'],r['train_seed'],r['seed']) for r in test}
        if len(test)!=len(expected) or keys!=expected:raise RuntimeError('C1 TEST grid audit')
        paired={}
        for row in test:
            k=row['family'],row['n'],row['seed']
            tape=row['physical_seed'],row['population_sha256']
            if k in paired and paired[k]!=tape:raise RuntimeError('C1 TEST physical pairing')
            paired[k]=tape
            if row['invalid_assignments'] or row['requests']!=row['completions']+row['tail_pending']+row['active_end']:
                raise RuntimeError('C1 TEST feasibility or request conservation')
        for row in secondary:
            k=row['family'],row['n'],row['seed']
            if paired[k]!=(row['physical_seed'],row['population_sha256']) or row['invalid_assignments']:
                raise RuntimeError('C1 secondary physical pairing')
        if sha(out/'checkpoint_choices.json')!=manifest['choices_sha256']:
            raise RuntimeError('C1 checkpoint choices changed after TEST')
        if any(sha(out/path)!=digest for path,digest in seals.items()):
            raise RuntimeError('C1 source seal changed')
        write_json(out/'audit.json',dict(status='PASS',training_models=len(audits),
            development_episodes=len(devrows),test_episodes=len(test),
            test_physical_ticks=60*len(test),secondary_episodes=len(secondary),
            paired_test_tapes=len(paired),invalid_assignments=0,
            frozen_reference=gate['selected_rule'],
            scientific_test_opened=profile=='confirmation'))
        manifest.update(status='COMPLETED',stage='complete',audit_status='PASS',
            goal_pass=summary['goal_pass'],elapsed_seconds=time.monotonic()-start,
            completed_utc=datetime.now(timezone.utc).isoformat())
        manifest['output_bindings']={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size)
            for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        write_json(out/'manifest.json',manifest)
        return manifest
    except Exception as exc:
        write_json(out/'abort.json',dict(reason='TECHNICAL_FAILURE',error=f'{type(exc).__name__}: {exc}'))
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',
            elapsed_seconds=time.monotonic()-start)
        write_json(out/'manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('smoke','confirmation'),required=True)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--development-artifact',required=True,type=Path)
    parser.add_argument('--teacher-dir',type=Path)
    parser.add_argument('--independent-development-audit',type=Path)
    args=parser.parse_args()
    result=run(args.profile,args.output_dir,args.development_artifact,args.teacher_dir,args.independent_development_audit)
    print(json.dumps(dict(status=result['status'],output_dir=str(args.output_dir),
        goal_pass=result['goal_pass'])))


if __name__=='__main__':main()
