"""Exploratory B3 same-weight capacity intervention on opened DEV only."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import technician_prefix_capacity_v3_experiment as x
from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors
from ht_pdm_fjsp.technician_prefix_capacity_v3_env import ResourceEnv

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/technician_prefix_capacity_v3_dev_diagnostic_plan.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def capacity_off(full):
    """Identical fitted full tensors, with only the two capacity input slots off."""
    model=copy.deepcopy(full)
    model.capacity_suppressed_inputs=0
    def suppress(module,args):
        value=args[0]
        model.capacity_suppressed_inputs+=int(torch.count_nonzero(value[...,-2:]))
        return (torch.cat([value[...,:-2],torch.zeros_like(value[...,-2:])],-1),)
    model.actor.register_forward_pre_hook(suppress)
    model.wait.register_forward_pre_hook(suppress)
    return model


def rollout(seed,n,algorithm,model):
    env=ResourceEnv(seed,n,algorithm);obs=env.reset()
    for _ in range(48):
        obs,_,done,info=env.step(model.select(obs))
    assert done
    return info['episode_metrics']


def run(source,audit_source,output,profile):
    started=time.monotonic();output=Path(output);source=Path(source);audit_source=Path(audit_source)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),output/Path(__file__).name)
    shutil.copy2(PLAN,output/'diagnostic_plan.md')
    manifest=dict(status='RUNNING',classification='EXPLORATORY_DEV_DIAGNOSTIC',profile=profile,
        source_artifact=str(source),source_manifest_sha256=sha(source/'manifest.json'),
        independent_audit_artifact=str(audit_source),independent_audit_sha256=sha(audit_source/'audit.json'),
        script_sha256=sha(output/Path(__file__).name),plan_sha256=sha(output/'diagnostic_plan.md'),
        started_utc=datetime.now(timezone.utc).isoformat(),
        invocation=dict(executable=sys.executable,argv=sys.argv,cwd=str(Path.cwd())))
    save(output/'manifest.json',manifest)
    try:
        original=json.loads((source/'manifest.json').read_text())
        independent=json.loads((audit_source/'audit.json').read_text())
        assert original['protocol']=='technician_prefix_capacity_v3'
        assert original['profile']=='development' and original['status']=='COMPLETED'
        assert original['audit_status']=='PASS' and not original['test_opened']
        assert independent['status']=='PASS' and independent['verdict']=='DEV_GATE_FAIL'
        assert independent['test_opened'] is False and independent['source_artifact']==str(source)
        assert independent['source_manifest_sha256']==manifest['source_manifest_sha256']
        assert independent['verifier_sha256']==sha(audit_source/'audit_prefix_capacity_development.py')
        for name,digest in original['source_seal'].items():
            assert sha(source/name)==digest,name
        for name,meta in original['output_bindings'].items():
            path=source/name
            assert path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256'],name
        cfg=json.loads((source/'resolved_config.json').read_text())
        assert not cfg['test_seeds']
        choices=json.loads((source/'checkpoint_choices.json').read_text())
        with (source/'development_episodes.csv').open(newline='') as stream:
            prior={(int(r['n']),r['method'],int(r['train_seed']),int(r['checkpoint_steps']),int(r['seed'])):r
                for r in csv.DictReader(stream)}
        torch.set_num_threads(1)
        seeds=cfg['development_seeds'][:2] if profile=='smoke' else cfg['development_seeds']
        records=[]
        for n in cfg['scales']:
            for train in cfg['train_seeds']:
                choice=choices[f'{n}:reservation_ppo:{train}']
                full=x.load_model(source/choice['path'],n,cfg,train,'reservation_ppo',choice['steps'])
                cap=capacity_off(full)
                all_off=x.make_model('mask_only_ppo',n,cfg,train)
                all_off.load_state_dict(full.state_dict());all_off.eval()
                for seed in tqdm(seeds,desc=f'B3 DEV diagnosis N{n}/bundle{train}',unit='episode'):
                    if time.monotonic()-started>3600:raise TimeoutError('registered DEV diagnosis hardstop')
                    env=ResourceEnv(seed,n,'reservation_ppo');obs=env.reset()
                    cap_changed=all_changed=cap_slots=all_slots=mask_checks=capacity_stages=relational_stages=0
                    for _ in range(48):
                        action=full.select(obs)
                        candidate_cap=cap.select(obs)
                        candidate_all=all_off.select(obs)
                        cap_changed+=int(np.any(action!=candidate_cap))
                        all_changed+=int(np.any(action!=candidate_all))
                        cap_slots+=int(np.count_nonzero(action!=candidate_cap))
                        all_slots+=int(np.count_nonzero(action!=candidate_all))
                        batch=tensors({name:np.asarray(value)[None] for name,value in obs.items()})
                        selected=torch.as_tensor(action[None],dtype=torch.long)
                        traces=[[],[],[]]
                        with torch.no_grad():
                            for model,trace in zip((full,cap,all_off),traces,strict=True):
                                model.trace(batch,recorded=selected,audit=trace)
                        for fst,cst,ast in zip(*traces,strict=True):
                            if not torch.equal(fst['legal'],cst['legal']) or not torch.equal(fst['legal'],ast['legal']):
                                raise RuntimeError('intervention changed legal reservation set')
                            if torch.count_nonzero(ast['prefix']) or torch.count_nonzero(ast['candidate_capacity']) or torch.count_nonzero(ast['wait_capacity']):
                                raise RuntimeError('all-prefix-off channel leakage')
                            capacity_stages+=int(bool(torch.count_nonzero(fst['candidate_capacity']) or torch.count_nonzero(fst['wait_capacity'])))
                            relational_stages+=int(bool(torch.count_nonzero(fst['prefix'])))
                            mask_checks+=1
                        obs,_,done,info=env.step(action)
                    assert done
                    first=info['episode_metrics']
                    second=rollout(seed,n,'reservation_ppo',cap)
                    third=rollout(seed,n,'mask_only_ppo',all_off)
                    if len({first['population_sha256'],second['population_sha256'],third['population_sha256']})!=1:
                        raise RuntimeError('physical population mismatch')
                    if any(r['invalid_assignments'] for r in (first,second,third)):
                        raise RuntimeError('invalid assignment')
                    saved=prior[n,'reservation_ppo',train,choice['steps'],seed]
                    if abs(first['extended_cost']-float(saved['extended_cost']))>1e-8 or first['population_sha256']!=saved['population_sha256']:
                        raise RuntimeError('full trajectory does not replay frozen DEV row')
                    records.append(dict(n=n,train_seed=train,seed=seed,checkpoint_steps=choice['steps'],
                        full_cost=first['extended_cost'],capacity_off_cost=second['extended_cost'],
                        all_prefix_off_cost=third['extended_cost'],
                        full_availability=first['availability'],capacity_off_availability=second['availability'],all_prefix_off_availability=third['availability'],
                        full_pending=first['terminal_pending']/n,capacity_off_pending=second['terminal_pending']/n,all_prefix_off_pending=third['terminal_pending']/n,
                        full_failures=first['policy_failures']/n,capacity_off_failures=second['policy_failures']/n,all_prefix_off_failures=third['policy_failures']/n,
                        capacity_off_changed_states=cap_changed,all_prefix_off_changed_states=all_changed,
                        capacity_off_slot_changes=cap_slots,all_prefix_off_slot_changes=all_slots,
                        nonzero_capacity_stages=capacity_stages,nonzero_relational_stages=relational_stages,
                        mask_checks=mask_checks,population_sha256=first['population_sha256']))
        with (output/'episodes.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        by_scale={}
        for n in cfg['scales']:
            panel=[r for r in records if r['n']==n]
            mean=lambda field,part:float(np.mean([r[field] for r in part]))
            def record(part):
                return dict(episodes=len(part),
                    cost_per_machine={name:mean(name+'_cost',part)/n for name in ('full','capacity_off','all_prefix_off')},
                    capacity_off_changed_states=sum(r['capacity_off_changed_states'] for r in part),
                    all_prefix_off_changed_states=sum(r['all_prefix_off_changed_states'] for r in part),
                    capacity_off_slot_changes=sum(r['capacity_off_slot_changes'] for r in part),
                    all_prefix_off_slot_changes=sum(r['all_prefix_off_slot_changes'] for r in part),
                    nonzero_capacity_stages=sum(r['nonzero_capacity_stages'] for r in part),
                    nonzero_relational_stages=sum(r['nonzero_relational_stages'] for r in part),
                    mask_checks=sum(r['mask_checks'] for r in part))
            by_scale[str(n)]=dict(**record(panel),per_replica={str(t):record([r for r in panel if r['train_seed']==t])
                for t in cfg['train_seeds']})
        save(output/'summary.json',dict(classification='EXPLORATORY_DEV_DIAGNOSTIC',
            scientific_matched_control_verdict='DEV_GATE_FAIL',same_weight_interventions_only=True,by_scale=by_scale))
        manifest.update(status='COMPLETED',test_opened=False,episodes=len(records),
            completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
            capacity_suppressed_inputs='actor and WAIT last two slots only',
            output_bindings={p.name:dict(sha256=sha(p),bytes=p.stat().st_size)
                for p in output.iterdir() if p.is_file() and p.name!='manifest.json'})
        save(output/'manifest.json',manifest)
        return manifest
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-started)
        save(output/'manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact',type=Path)
    parser.add_argument('--audit',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--profile',choices=('smoke','full'),required=True)
    args=parser.parse_args()
    result=run(args.artifact.resolve(),args.audit.resolve(),args.output_dir.resolve(),args.profile)
    print(json.dumps(dict(status=result['status'],output_dir=str(args.output_dir),episodes=result['episodes'])))


if __name__=='__main__':main()
