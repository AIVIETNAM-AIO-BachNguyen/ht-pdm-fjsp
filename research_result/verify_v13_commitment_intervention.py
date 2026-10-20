"""Independent action, trigger, metric and bootstrap audit for V13 commitment replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, HORIZON, FrameworkEnv, catalogue, configs, vector_observation
from ht_pdm_fjsp.maintenance_framework_v13_policy import OpportunityCostActor
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests

ROOT=Path(__file__).resolve().parent.parent
ACTIONS=catalogue()
TARGETS=('interchangeable_scale1','reciprocal_scale2')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def close(a,b,label):
    if not np.isclose(float(a),float(b),rtol=0,atol=1e-7):
        raise AssertionError(f'{label}: {a} != {b}')


def actor_choice(actor,cfg,admitted,tick,legal):
    obs=vector_observation(cfg,admitted,tick)
    with torch.no_grad():logits=actor(torch.as_tensor(obs).reshape(1,56))[0]
    masked=torch.full((len(ACTIONS),),-1e9,dtype=torch.float32)
    indices=[ACTIONS.index(a) for a in legal]
    masked[indices]=logits[indices]
    return ACTIONS[int(np.argmax(torch.softmax(masked,dim=0).numpy()))]


def replay(case,seed,actor,original_actions,edited_actions,expected,deadline):
    cfg=configs()[case]
    out=[]
    original_first=None
    for arm,actions in (('original',original_actions),('edited',edited_actions)):
        env=FrameworkEnv(cfg);env.reset(seed=seed)
        risk=dict(exposed=0,feasible=0,constrained=0)
        pm_starts=0
        first=None;reissued=False
        for tick,record in enumerate(actions):
            if time.monotonic()>deadline:raise TimeoutError('V13 intervention audit timeout')
            assert int(record['tick'])==tick
            before=env.inner.state
            admitted=submit_requests(cfg,before,tick,request_action(cfg,before,tick,'threshold'))
            legal=legal_matchings(cfg,admitted)
            selected=actor_choice(actor,cfg,admitted,tick,legal)
            chosen=tuple(tuple(x) for x in json.loads(record['selected']))
            executed=tuple(tuple(x) for x in json.loads(record['executed']))
            assert selected==chosen and executed in legal
            if arm=='original':assert chosen==executed
            elif int(expected['triggered']) and tick==int(expected['start_tick']):
                pair=(int(expected['occupied_machine']),int(expected['technician']))
                assert pair in chosen
                assert executed==tuple(p for p in chosen if p!=pair)
            else:assert executed==chosen
            if arm=='edited' and int(expected['triggered']) and tick>int(expected['start_tick']):
                if any(m==int(expected['occupied_machine']) for m,_ in executed):reissued=True
            served={m for m,_ in executed}
            tick_risk=dict(exposed=0,feasible=0,constrained=0)
            target=None
            for machine,req in enumerate(admitted.requests):
                if req is None or req.started_at is not None or req.kind!='preventive':continue
                if admitted.physical.failed[machine] or admitted.physical.ages[machine]+1<cfg.failure_age:continue
                if machine in served:continue
                tick_risk['exposed']+=1
                possible=any(any(m==machine for m,_ in action) for action in legal)
                if possible:tick_risk['feasible']+=1
                else:
                    tick_risk['constrained']+=1
                    if target is None:target=machine
            for key in risk:
                risk[key]+=tick_risk[key]
                assert int(record[key])==tick_risk[key]
            if arm=='original' and first is None and target is not None:
                busy=[j for j in range(cfg.technicians) if cfg.service_time[target][j]>0
                      and admitted.physical.remaining[j]>0]
                if busy:
                    j=sorted(busy,key=lambda x:(-admitted.physical.remaining[x],x))[0]
                    occupied=admitted.physical.assigned[j]
                    req=admitted.requests[occupied]
                    assert req is not None and req.technician==j
                    first=(tick,target,j,occupied,req.started_at,
                           admitted.physical.remaining[j],cfg.service_time[occupied][j])
            _,_,_,_,info=env.step(ACTIONS.index(executed))
            pm_starts+=sum(e['event']=='started' and e['kind']=='preventive' for e in info['lifecycle'])
        met=env.episode_metrics()
        prefix='original' if arm=='original' else 'edited'
        for field,name in (('objective','cost'),('failure_cost','failure_cost'),
                           ('H_failures','H_failures'),('H_availability','H_availability')):
            close(met[field],expected[f'{prefix}_{name}'],(case,seed,arm,field))
        assert pm_starts==int(expected[f'{prefix}_H_pm_starts'])
        for key in risk:
            suffix={'exposed':'H_risk_exposed','feasible':'H_risk_feasible_exposed',
                    'constrained':'H_risk_capacity_exposed'}[key]
            assert risk[key]==int(expected[f'{prefix}_{suffix}'])
        out.append((met,risk,pm_starts))
        if arm=='original':original_first=first
    first=original_first
    if first is None:
        assert int(expected['triggered'])==0
        for key in ('trigger_tick','target_machine','technician','occupied_machine','start_tick','remaining','duration'):
            assert expected[key]==''
    else:
        assert int(expected['triggered'])==1
        for key,value in zip(('trigger_tick','target_machine','technician','occupied_machine',
                              'start_tick','remaining','duration'),first):
            assert int(expected[key])==value
        pair=[first[3],first[2]]
        assert pair in json.loads(original_actions[first[4]]['executed'])
    assert int(expected['reissued'])==int(reissued)
    a,b=out
    close(float(b[0]['objective'])-float(a[0]['objective']),expected['cost_delta'],'cost delta')
    close(float(b[0]['H_failures'])-float(a[0]['H_failures']),expected['failure_delta'],'failure delta')
    assert b[2]-a[2]==int(expected['pm_starts_delta'])
    assert b[1]['exposed']-a[1]['exposed']==int(expected['exposure_delta'])


def independently_summarize(rows,profile,manifest):
    seeds=range(72500000,72500005) if profile=='dev' else (72520000,)
    physicals=range(72510000,72510016) if profile=='dev' else (72520000,72520001)
    table={(r['case'],int(r['train_seed']),int(r['seed'])):r for r in rows}
    rng=np.random.default_rng(int(manifest['bootstrap_seed']))
    # Runner initializes a fresh RNG per case; repeat that independent stream per case.
    result={}
    for case in CASES:
        cube=np.asarray([[float(table[(case,s,p)]['cost_delta']) for p in physicals] for s in seeds])
        local=np.random.default_rng(int(manifest['bootstrap_seed']))
        draws=int(manifest['bootstrap_draws'])
        boot=np.empty(draws)
        for i in range(0,draws,1000):
            n=min(1000,draws-i)
            ti=local.integers(0,len(seeds),size=(n,len(seeds)))
            pi=local.integers(0,len(physicals),size=(n,len(seeds),len(physicals)))
            boot[i:i+n]=np.take_along_axis(cube[ti],pi,axis=-1).mean(axis=(1,2))
        group=[table[(case,s,p)] for s in seeds for p in physicals]
        means=cube.mean(axis=1)
        ci=np.quantile(boot,[.025,.975])
        result[case]=dict(cost_delta=float(cube.mean()),ci95=[float(v) for v in ci],
            by_train_seed={str(s):float(means[k]) for k,s in enumerate(seeds)},
            trigger_fraction=float(np.mean([int(r['triggered']) for r in group])),
            triggered_count=sum(int(r['triggered']) for r in group),
            conditional_delta=float(np.mean([float(r['cost_delta']) for r in group if int(r['triggered'])]))
                if any(int(r['triggered']) for r in group) else None,
            mean_exposure_delta=float(np.mean([float(r['exposure_delta']) for r in group])),
            mean_failure_delta=float(np.mean([float(r['failure_delta']) for r in group])),
            mean_pm_starts_delta=float(np.mean([float(r['pm_starts_delta']) for r in group])),
            directional_gate=bool(profile=='dev' and cube.mean()<=-10 and ci[1]<0 and sum(means<0)>=4))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();run=args.run_dir.resolve();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    deadline=time.monotonic()+1200
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='COMPLETED'
    for filename,digest in manifest['artifact_hashes'].items():assert sha(run/filename)==digest
    assert sha(ROOT/'research_result/run_v13_commitment_intervention.py')==manifest['script_sha256']
    assert sha(ROOT/'research_result/v13_commitment_intervention_protocol_vi.md')==manifest['protocol_sha256']
    source=Path(manifest['source_run'])
    assert sha(source/'manifest.json')==manifest['source_manifest_sha256']
    assert sha(source/'episodes.csv')==manifest['source_episodes_sha256']
    assert sha(source/'selection.json')==manifest['source_selection_sha256']
    selection=json.loads((source/'selection.json').read_text())
    actors={}
    for seed,digest in manifest['checkpoint_hashes'].items():
        cp=source/selection['competition'][seed]['checkpoint']
        assert sha(cp)==digest
        payload=torch.load(cp,map_location='cpu',weights_only=True)
        actor=OpportunityCostActor(competition=True)
        actor.load_state_dict(payload['state_dict']);actor.eval()
        actors[int(seed)]=actor
    torch.set_num_threads(1)
    rows=read_csv(run/'episodes.csv');actions=read_csv(run/'actions.csv')
    by_action={(r['case'],int(r['train_seed']),int(r['seed']),r['arm'],int(r['tick'])):r for r in actions}
    assert len(by_action)==len(actions)==len(rows)*2*HORIZON
    source_rows={(r['case'],int(r['train_seed']),int(r['seed'])):r
                 for r in read_csv(source/'episodes.csv') if r['controller']=='competition'}
    for row in tqdm(rows,desc='Independent V13 commitment audit',unit='pair'):
        case=row['case'];train=int(row['train_seed']);seed=int(row['seed'])
        assert case in CASES and train in actors
        original=[by_action[(case,train,seed,'original',t)] for t in range(HORIZON)]
        edited=[by_action[(case,train,seed,'edited',t)] for t in range(HORIZON)]
        replay(case,seed,actors[train],original,edited,row,deadline)
        source_row=source_rows[(case,train,seed)]
        for key,field in (('original_cost','objective'),('original_failure_cost','failure_cost'),
                          ('original_H_failures','H_failures'),('original_H_availability','H_availability')):
            close(row[key],source_row[field],f'source {case}/{train}/{seed}/{key}')
    profile=manifest['profile']
    seeds=range(72500000,72500005) if profile=='dev' else (72520000,)
    physicals=range(72510000,72510016) if profile=='dev' else (72520000,72520001)
    assert {(r['case'],int(r['train_seed']),int(r['seed'])) for r in rows}=={
        (case,s,p) for case in CASES for s in seeds for p in physicals}
    summary=json.loads((run/'summary.json').read_text())
    assert summary['status']=='COMPLETED' and summary['episodes']==len(rows)
    independent=independently_summarize(rows,profile,manifest)
    for case in CASES:
        for key,value in independent[case].items():
            stored=summary['by_case'][case][key]
            if isinstance(value,dict):
                for seed,v in value.items():close(v,stored[seed],(case,key,seed))
            elif isinstance(value,list):
                for v,s in zip(value,stored):close(v,s,(case,key))
            elif value is None or isinstance(value,bool):assert stored==value
            else:close(value,stored,(case,key))
    assert summary['joint_gate']==all(independent[c]['directional_gate'] for c in TARGETS)
    close(summary['joint_cost_delta'],np.mean([independent[c]['cost_delta'] for c in TARGETS]),'joint')
    audit=dict(status='PASS',episodes=len(rows),actions=len(actions),profile=profile,
               run_manifest_sha256=sha(run/'manifest.json'),source_manifest_sha256=sha(source/'manifest.json'),
               verifier_sha256=sha(Path(__file__)))
    (out/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))

if __name__=='__main__':main()
