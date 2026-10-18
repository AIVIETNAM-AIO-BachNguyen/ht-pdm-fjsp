"""Exploratory same-weights prefix intervention on opened B1 development seeds."""

from __future__ import annotations

import argparse
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

from ht_pdm_fjsp import technician_prefix_experiment as x
from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors
from ht_pdm_fjsp.technician_prefix_env import ResourceEnv


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def run(source,output,profile):
    start=time.monotonic()
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),output/Path(__file__).name)
    manifest=dict(status='RUNNING',classification='EXPLORATORY_DEV_DIAGNOSTIC',profile=profile,
        source_artifact=str(source),source_manifest_sha256=sha(source/'manifest.json'),
        script_sha256=sha(output/Path(__file__).name),started_utc=datetime.now(timezone.utc).isoformat(),
        invocation=dict(executable=sys.executable,argv=sys.argv,cwd=str(Path.cwd())))
    save_json(output/'manifest.json',manifest)
    try:
        original=json.loads((source/'manifest.json').read_text())
        cfg=json.loads((source/'resolved_config.json').read_text())
        choices=json.loads((source/'checkpoint_choices.json').read_text())
        assert original['status']=='COMPLETED' and original['audit_status']=='PASS'
        assert original['profile']=='development' and not original['test_opened']
        for name,digest in original['source_seal'].items():
            assert sha(source/name)==digest,name
        for name,meta in original['output_bindings'].items():
            path=source/name
            assert path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256'],name
        torch.set_num_threads(1)
        seeds=cfg['development_seeds'][:2] if profile=='smoke' else cfg['development_seeds']
        records=[]
        for n in cfg['scales']:
            for train in cfg['train_seeds']:
                choice=choices[f'{n}:reservation_ppo:{train}']
                full=x.load_model(source/choice['path'],n,cfg,train,'reservation_ppo',choice['steps'])
                without=x.make_model('mask_only_ppo',n,cfg,train)
                without.load_state_dict(full.state_dict());without.eval()
                for seed in tqdm(seeds,desc=f'Diagnostic N{n}/bundle{train}',unit='episode'):
                    if time.monotonic()-start>3600:raise TimeoutError('registered diagnostic hardstop')
                    env=ResourceEnv(seed,n,'reservation_ppo');obs=env.reset()
                    disagreements=slot_disagreements=nonzero_prefix=mask_checks=0
                    for tick in range(48):
                        a=full.select(obs);b=without.select(obs)
                        disagreements+=int(np.any(a!=b));slot_disagreements+=int(np.count_nonzero(a!=b))
                        batch=tensors({name:np.asarray(value)[None] for name,value in obs.items()})
                        selected=torch.as_tensor(a[None],dtype=torch.long)
                        full_trace=[];mask_trace=[]
                        with torch.no_grad():
                            full.trace(batch,recorded=selected,audit=full_trace)
                            without.trace(batch,recorded=selected,audit=mask_trace)
                        for fst,mst in zip(full_trace,mask_trace,strict=True):
                            if not torch.equal(fst['legal'],mst['legal']):raise RuntimeError('feasible masks differ')
                            if torch.count_nonzero(mst['prefix']):raise RuntimeError('masked prefix nonzero')
                            nonzero_prefix+=int(bool(torch.count_nonzero(fst['prefix'])))
                            mask_checks+=1
                        obs,_,done,info=env.step(a)
                    assert done
                    first=info['episode_metrics']
                    other=ResourceEnv(seed,n,'mask_only_ppo');obs=other.reset()
                    for tick in range(48):obs,_,done,info=other.step(without.select(obs))
                    assert done
                    second=info['episode_metrics']
                    if first['population_sha256']!=second['population_sha256'] or first['invalid_assignments'] or second['invalid_assignments']:
                        raise RuntimeError('physical pairing or legality failed')
                    records.append(dict(n=n,train_seed=train,seed=seed,checkpoint_steps=choice['steps'],
                        full_cost=first['extended_cost'],without_prefix_cost=second['extended_cost'],
                        full_availability=first['availability'],without_prefix_availability=second['availability'],
                        full_pending=first['terminal_pending']/n,without_prefix_pending=second['terminal_pending']/n,
                        full_failures=first['policy_failures']/n,without_prefix_failures=second['policy_failures']/n,
                        action_disagreements=disagreements,slot_disagreements=slot_disagreements,
                        nonzero_prefix_stages=nonzero_prefix,mask_checks=mask_checks,
                        population_sha256=first['population_sha256']))
        with (output/'episodes.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        by_scale={}
        for n in cfg['scales']:
            panel=[r for r in records if r['n']==n]
            by_scale[str(n)]=dict(episodes=len(panel),
                mean_full_cost_per_machine=float(np.mean([r['full_cost']/n for r in panel])),
                mean_without_prefix_cost_per_machine=float(np.mean([r['without_prefix_cost']/n for r in panel])),
                same_weights_cost_gain_full_vs_without=float(1-np.mean([r['full_cost'] for r in panel])/np.mean([r['without_prefix_cost'] for r in panel])),
                states_with_changed_action=sum(r['action_disagreements'] for r in panel),
                state_count=48*len(panel),
                slot_changes=sum(r['slot_disagreements'] for r in panel),
                nonzero_prefix_stages=sum(r['nonzero_prefix_stages'] for r in panel),
                mask_checks=sum(r['mask_checks'] for r in panel),
                per_replica={str(t):dict(
                    mean_full_cost_per_machine=float(np.mean([r['full_cost']/n for r in panel if r['train_seed']==t])),
                    mean_without_prefix_cost_per_machine=float(np.mean([r['without_prefix_cost']/n for r in panel if r['train_seed']==t])),
                    changed_states=sum(r['action_disagreements'] for r in panel if r['train_seed']==t))
                    for t in cfg['train_seeds']})
        save_json(output/'summary.json',dict(classification='EXPLORATORY_DEV_DIAGNOSTIC',by_scale=by_scale))
        manifest.update(status='COMPLETED',completed_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-start,episodes=len(records),test_opened=False,
            output_bindings={p.name:dict(sha256=sha(p),bytes=p.stat().st_size)
                for p in output.iterdir() if p.is_file() and p.name!='manifest.json'})
        save_json(output/'manifest.json',manifest)
        return manifest
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        save_json(output/'manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--profile',choices=('smoke','full'),required=True)
    args=parser.parse_args()
    result=run(args.artifact.resolve(),args.output_dir.resolve(),args.profile)
    print(json.dumps(dict(status=result['status'],output_dir=str(args.output_dir),episodes=result['episodes'])))


if __name__=='__main__':main()
