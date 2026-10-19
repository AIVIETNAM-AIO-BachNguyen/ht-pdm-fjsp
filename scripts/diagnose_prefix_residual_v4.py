"""Exploratory B4 same-weight direct-residual intervention on opened DEV only."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import technician_prefix_residual_v4_experiment as x
from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors
from ht_pdm_fjsp.technician_prefix_residual_v4_env import ResourceEnv

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/technician_prefix_residual_v4_dev_diagnostic_plan.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def residual_off(full):
    """Keep the key map in graph encoding; zero only later residual key outputs."""
    model=copy.deepcopy(full)
    state=dict(calls=0,encoded_calls=0,suppressed_nonzero=0)
    def suppress(module,args,output):
        call=state['calls'];state['calls']+=1
        if call==0:
            state['encoded_calls']+=1
            return output
        state['suppressed_nonzero']+=int(torch.count_nonzero(output))
        return torch.zeros_like(output)
    model.key.register_forward_hook(suppress)
    original_trace=model.trace
    def trace(*args,**kwargs):
        state['calls']=0
        result=original_trace(*args,**kwargs)
        k=args[0]['edges'].shape[2]
        if state['calls']!=1+2*k:
            raise RuntimeError('unexpected shared key call contract')
        return result
    model.trace=trace
    model.residual_intervention_state=state
    return model


def rollout(seed,n,model):
    env=ResourceEnv(seed,n,'reservation_ppo');obs=env.reset()
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
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        started_utc=datetime.now(timezone.utc).isoformat(),
        invocation=dict(executable=sys.executable,argv=sys.argv,cwd=str(Path.cwd())))
    save(output/'manifest.json',manifest)
    try:
        original=json.loads((source/'manifest.json').read_text())
        independent=json.loads((audit_source/'audit.json').read_text())
        assert original['protocol']=='technician_prefix_residual_v4'
        assert original['profile']=='development' and original['status']=='COMPLETED'
        assert original['audit_status']=='PASS' and not original['test_opened']
        assert independent['status']=='PASS' and independent['verdict']=='DEV_GATE_FAIL'
        assert independent['test_opened'] is False and independent['source_artifact']==str(source)
        assert independent['source_manifest_sha256']==manifest['source_manifest_sha256']
        assert independent['verifier_sha256']==sha(audit_source/'audit_prefix_residual_development.py')
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
                off=residual_off(full)
                for seed in tqdm(seeds,desc=f'B4 DEV residual diagnosis N{n}/bundle{train}',unit='episode'):
                    if time.monotonic()-started>3600:raise TimeoutError('registered DEV diagnosis hardstop')
                    env=ResourceEnv(seed,n,'reservation_ppo');obs=env.reset()
                    changed=slot_changes=mask_checks=residual_stages=0
                    for _ in range(48):
                        action=full.select(obs);candidate=off.select(obs)
                        changed+=int(np.any(action!=candidate))
                        slot_changes+=int(np.count_nonzero(action!=candidate))
                        batch=tensors({name:np.asarray(value)[None] for name,value in obs.items()})
                        with torch.no_grad():
                            original_encoded=full.encode(batch)
                            off.residual_intervention_state['calls']=0
                            intervened_encoded=off.encode(batch)
                            if off.residual_intervention_state['calls']!=1:
                                raise RuntimeError('encoder key call contract')
                            if not all(torch.equal(a,b) for a,b in zip(original_encoded,intervened_encoded,strict=True)):
                                raise RuntimeError('graph encoding changed')
                            selected=torch.as_tensor(action[None],dtype=torch.long)
                            traces=[[],[]]
                            full.trace(batch,recorded=selected,audit=traces[0])
                            off.trace(batch,recorded=selected,audit=traces[1])
                        for fst,ost in zip(*traces,strict=True):
                            for field in ('legal','resource','stats','prefix','candidate_capacity','wait_capacity','base_logits'):
                                if not torch.equal(fst[field],ost[field]):
                                    raise RuntimeError(f'residual intervention changed {field}')
                            if torch.count_nonzero(ost['prefix_residual']):
                                raise RuntimeError('direct residual was not zeroed')
                            if not torch.equal(fst['full_logits'],
                                               ost['full_logits']+fst['prefix_residual']):
                                raise RuntimeError('direct residual logit mismatch')
                            residual_stages+=int(bool(torch.count_nonzero(fst['prefix_residual'])))
                            mask_checks+=1
                        obs,_,done,info=env.step(action)
                    assert done
                    first=info['episode_metrics'];second=rollout(seed,n,off)
                    if first['population_sha256']!=second['population_sha256']:
                        raise RuntimeError('physical population mismatch')
                    if first['invalid_assignments'] or second['invalid_assignments']:
                        raise RuntimeError('invalid assignment')
                    saved=prior[n,'reservation_ppo',train,choice['steps'],seed]
                    if abs(first['extended_cost']-float(saved['extended_cost']))>1e-8 or first['population_sha256']!=saved['population_sha256']:
                        raise RuntimeError('full trajectory does not replay frozen DEV row')
                    records.append(dict(n=n,train_seed=train,seed=seed,checkpoint_steps=choice['steps'],
                        full_cost=first['extended_cost'],residual_off_cost=second['extended_cost'],
                        full_availability=first['availability'],residual_off_availability=second['availability'],
                        full_pending=first['terminal_pending']/n,residual_off_pending=second['terminal_pending']/n,
                        full_failures=first['policy_failures']/n,residual_off_failures=second['policy_failures']/n,
                        residual_off_changed_states=changed,residual_off_slot_changes=slot_changes,
                        nonzero_residual_stages=residual_stages,mask_checks=mask_checks,
                        population_sha256=first['population_sha256']))
        with (output/'episodes.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        by_scale={}
        for n in cfg['scales']:
            panel=[r for r in records if r['n']==n]
            mean=lambda field,part:float(np.mean([r[field] for r in part]))
            def record(part):
                return dict(episodes=len(part),
                    cost_per_machine={name:mean(name+'_cost',part)/n for name in ('full','residual_off')},
                    residual_off_changed_states=sum(r['residual_off_changed_states'] for r in part),
                    residual_off_slot_changes=sum(r['residual_off_slot_changes'] for r in part),
                    nonzero_residual_stages=sum(r['nonzero_residual_stages'] for r in part),
                    mask_checks=sum(r['mask_checks'] for r in part))
            by_scale[str(n)]=dict(**record(panel),per_replica={str(t):record([r for r in panel if r['train_seed']==t])
                for t in cfg['train_seeds']})
        save(output/'summary.json',dict(classification='EXPLORATORY_DEV_DIAGNOSTIC',
            scientific_matched_control_verdict='DEV_GATE_FAIL',same_weight_interventions_only=True,by_scale=by_scale))
        manifest.update(status='COMPLETED',test_opened=False,episodes=len(records),
            completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
            intervention='shared key map first encoder call preserved, later direct-residual calls zeroed',
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
