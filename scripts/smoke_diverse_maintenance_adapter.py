"""Small read-only heuristic smoke of the preregistered request adapter."""

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

from tqdm.auto import tqdm

from ht_pdm_fjsp.technician_diverse_maintenance_env import (
    DiverseEnv, FAMILIES, SCALES,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, payload):
    Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')


def run(output):
    started=time.monotonic()
    output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve()
    package=Path(DiverseEnv.__module__.replace('.', '/'))
    root=Path(__file__).resolve().parents[1]
    env_source=root/'src'/f'{package}.py'
    registry=root/'configs/technician_diverse_maintenance_v1_seed_registry.json'
    protocol=root/'docs/technician_diverse_maintenance_v1_plan.md'
    seeds=json.loads(registry.read_text())['panels']['smoke']
    dev=list(range(seeds['development_start'],
        seeds['development_start']+seeds['development_count']))
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for p in (source,env_source,registry,protocol):
        shutil.copy2(p,snapshot/p.name)
    manifest=dict(status='RUNNING',classification='ENGINEERING_ADAPTER_SMOKE',
        started_utc=datetime.now(timezone.utc).isoformat(),
        invocation=dict(executable=sys.executable,argv=sys.argv,cwd=str(Path.cwd())),
        source_hashes={p.name:sha(snapshot/p.name) for p in (source,env_source,registry,protocol)},
        seeds=dev,scientific_test_opened=False)
    save(output/'manifest.json',manifest)
    try:
        rows=[]
        jobs=[(family,n,method,seed) for family in FAMILIES for n in SCALES
            for method in ('fcfs','risk') for seed in dev]
        for family,n,method,seed in tqdm(jobs,desc='Adapter smoke',unit='episode'):
            if time.monotonic()-started>300:raise TimeoutError('adapter smoke hardstop')
            env=DiverseEnv(seed,n,family,method)
            obs=env.reset();reward=0.
            for _ in range(48):
                obs,r,done,info=env.step(None);reward+=r
            if not done:raise RuntimeError('short episode')
            metrics=info['episode_metrics']
            if abs(-20*reward-metrics['extended_cost'])>1e-7:
                raise RuntimeError('reward/cost mismatch')
            if metrics['invalid_assignments']:
                raise RuntimeError('invalid assignment')
            rows.append(dict(family=family,n=n,method=method,seed=seed,
                physical_seed=metrics['physical_seed'],
                population_sha256=metrics['population_sha256'],
                extended_cost=metrics['extended_cost'],
                cost_per_machine=metrics['extended_cost']/n,
                availability=metrics['availability'],
                pending_per_machine=metrics['terminal_pending']/n,
                failures_per_machine=metrics['policy_failures']/n,
                requests=metrics['requests'],starts=metrics['starts'],
                completions=metrics['completions'],
                pending_end=metrics['tail_pending'],active_end=metrics['active_end'],
                invalid_assignments=metrics['invalid_assignments']))
        paired={}
        for row in rows:
            key=row['family'],row['n'],row['seed']
            tape=row['physical_seed'],row['population_sha256']
            if key in paired and paired[key]!=tape:
                raise RuntimeError('heuristic physical tapes differ')
            paired[key]=tape
            if row['requests']!=row['completions']+row['pending_end']+row['active_end']:
                raise RuntimeError('request conservation artifact')
        with (output/'episodes.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        save(output/'summary.json',dict(classification='ENGINEERING_ADAPTER_SMOKE',
            episodes=len(rows),paired_tapes=len(paired),
            by_configuration={f'{family}:N{n}':{
                method:sum(r['cost_per_machine'] for r in rows if
                    r['family']==family and r['n']==n and r['method']==method)/len(dev)
                for method in ('fcfs','risk')}
                for family in FAMILIES for n in SCALES}))
        manifest.update(status='COMPLETED',elapsed_seconds=time.monotonic()-started,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            output_bindings={p.name:dict(sha256=sha(p),bytes=p.stat().st_size)
                for p in output.iterdir() if p.is_file() and p.name!='manifest.json'})
        save(output/'manifest.json',manifest)
        return manifest
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',
            elapsed_seconds=time.monotonic()-started)
        save(output/'manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.output_dir.resolve())
    print(json.dumps(dict(status=result['status'],output_dir=str(args.output_dir))))


if __name__=='__main__':main()
