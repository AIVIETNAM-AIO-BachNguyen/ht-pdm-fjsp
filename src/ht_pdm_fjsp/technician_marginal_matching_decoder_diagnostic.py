"""Frozen N4 diagnostic: standard versus exact physical-probability decoding."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,multiprocessing as mp,shutil,subprocess,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_marginal_matching_experiment as x
from ht_pdm_fjsp.maintenance_dispatch import DispatchState,role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv,transition
from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue

VERSION='technician_marginal_matching_decoder_diagnostic_v1'
ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'


def settings():
    raw=json.loads(REGISTRY.read_text())['panels']['diagnostic']
    return dict(protocol=VERSION,seeds=raw['evaluation_seeds'],analysis_seed=raw['analysis_seed'],
        train_seeds=x.settings('development')['train_seeds'],scales=[4],training_processes=8,torch_threads=1,
        max_seconds=1800,teacher_assistance=False,training_decisions=0,scientific_test_opened=False)


def seed_audit():
    own=x.registered_integers(json.loads(REGISTRY.read_text()));history=set();hashes={}
    for f in sorted((ROOT/'configs').glob('*seed_registry.json')):
        if f!=REGISTRY:history|=x.registered_integers(json.loads(f.read_text()));hashes[f.name]=x.sha(f)
    expected=set(range(210000000,210000064))|{210100000,210200000,210200001}
    if own!=expected or own&history:raise ValueError('diagnostic raw seed overlap/panel')
    physical=[role_seed(s,x.VERSION+':N4:physical') for s in sorted(own-{210100000})]
    if len(set(physical))!=66 or set(physical)&(own|history):raise ValueError('diagnostic derived seed overlap')
    return dict(status='PASS',raw_unique=67,derived_physical_unique=66,historical_registry_hashes=hashes,
        intentional_pairing='same fresh physical episode across every frozen model/decoder; source train seeds identify frozen replicas')


@torch.no_grad()
def decoder_choices(model,obs):
    """Enumerate ONLY for this diagnostic, without cost/future-event lookahead."""
    if model.algorithm not in x.CORE_ARMS or obs['pair_mask'].shape!=(4,2):raise ValueError('N4 core diagnostic only')
    pairs=[a for a in catalogue(4,2) if all(obs['pair_mask'][m,j] for m,j in a)]
    tokens=np.zeros((len(pairs),2),dtype=np.int64)
    for row,a in enumerate(pairs):
        for m,j in a:tokens[row,j]=m+1
    batch=x.learner.tensors([obs]*len(pairs));logs,_=model.components(batch,torch.from_numpy(tokens))
    physical=model.physical_log(logs).numpy().astype(float);probs=np.exp(physical)
    residual=abs(probs.sum()-1)
    if not np.isfinite(physical).all() or residual>2e-5:raise RuntimeError('diagnostic physical support normalization')
    greedy=model.select(obs);where=np.flatnonzero(np.all(tokens==greedy[:-1],axis=1))
    if len(where)!=1:raise RuntimeError('standard matching outside physical support')
    winner=int(np.argmax(physical));index=int(where[0]);order=int(logs[winner].argmax())
    if model.algorithm=='forward_order_ppo':order=0
    elif model.algorithm=='reverse_order_ppo':order=1
    exact=np.r_[tokens[winner],order].astype(np.int64)
    return greedy,exact,dict(support=len(pairs),matching_tokens=json.dumps(tokens.tolist()),
        component_logs=json.dumps(logs.numpy().tolist()),physical_log_probabilities=json.dumps(physical.tolist()),
        standard_index=index,exact_index=winner,standard_probability=float(probs[index]),exact_probability=float(probs[winner]),
        physical_entropy=float(-(probs*physical).sum()),normalization_error=float(residual),
        disagrees=int(not np.array_equal(greedy[:-1],exact[:-1])))


def validate_source(directory):
    source=Path(directory).resolve();m=json.loads((source/'manifest.json').read_text());c=json.loads((source/'resolved_config.json').read_text());s=json.loads((source/'summary.json').read_text())
    if (m['protocol']!=x.VERSION or m['status']!='COMPLETED' or m['profile']!='development'
        or m.get('audit_status')!='PASS' or m['test_opened'] or s.get('development_screen_pass') is not False
        or c!=x.settings('development')):raise ValueError('requires completed audited failed-screen full development, tests closed')
    x.validate_summary_grid(s,c);x.validate_bound_training(source,m,c)
    for name,meta in m['output_bindings'].items():
        p=source/name
        if not p.is_file() or p.stat().st_size!=meta['bytes'] or x.sha(p)!=meta['sha256']:raise ValueError('changed parent output '+name)
    for name,digest in m['source_seal'].items():
        if x.sha(source/name)!=digest:raise ValueError('changed parent sealed input')
        if name.startswith('source_snapshot/') and x.sha(Path(x.__file__).parent/Path(name).name)!=digest:raise ValueError('imported source differs from frozen parent')
    choices=json.loads((source/'checkpoint_choices.json').read_text())
    if any(choices[k]['steps']!=v for k,v in s['selected_steps'].items()):raise ValueError('parent selected steps differ')
    return c,choices,dict(status='PASS',path=str(source),manifest_sha256=x.sha(source/'manifest.json'),source_commit=m['git']['commit'],selected_checkpoints={k:dict(sha256=v['sha256'],parameter_sha256=v['parameter_sha256'],steps=v['steps'],path=v['path']) for k,v in choices.items()})


def evaluate_job(job):
    cfg,source_name,parent_cfg,choice,a,t,decoder,out_name,start=job;source=Path(source_name);out=Path(out_name)
    torch.set_num_threads(1)
    def timeout():
        if (out/'abort.json').exists():raise RuntimeError('sibling technical abort')
        if time.monotonic()-start>cfg['max_seconds']:raise TimeoutError('registered diagnostic global deadline')
    directory=out/'jobs'/a/decoder/str(t);directory.mkdir(parents=True)
    journals={k:x.Journal(directory/f'{k}.csv') for k in ('episodes','decisions','coordination','decoder_choices','starts')}
    model=x.load_model(source/choice['path'],4,parent_cfg,t,a,choice['steps']) if choice else None
    initial=x.learner.parameter_digest(model) if model else None;rows=[]
    try:
        for seed in tqdm(cfg['seeds'],desc=f'Diagnostic {a}/{decoder}/{t}',leave=False):
            timeout();tags=dict(n=4,method=a,train_seed=t,seed=seed,decoder=decoder)
            env=x.ResourceEnv(seed,4,a,journal=x.Tags(journals['decisions'],tags));obs=env.reset();reward=0.;conflicts=rejected=0
            for tick in range(48):
                timeout();extra=None
                if a in x.CORE_ARMS:
                    standard,exact,extra=decoder_choices(model,obs);action=standard if decoder=='standard' else exact
                else:action=model.select(obs) if model else None
                before=env.env.state;pairs,diag=env.resolve(action)
                for m,j in pairs:
                    journals['starts'].add(dict(tags,tick=tick,machine=m,technician=j,duration=env.config.service_time[m][j],
                        restoration=env.config.restoration[m][j],corrective=int(before.failed[m]),age=before.ages[m],
                        free_technicians=sum(r==0 for r in before.remaining),urgent_edges=int(obs['pair_mask'].sum())))
                if extra is not None:journals['decoder_choices'].add(dict(tags,tick=tick,standard=json.dumps(standard.tolist()),exact=json.dumps(exact.tolist()),**extra))
                observation_hash=hashlib.sha256(obs['state'].tobytes()).hexdigest()
                obs,r,done,info=env.step(action);reward+=r;conflicts+=diag['conflicts'];rejected+=diag['rejected_proposals']
                journals['coordination'].add(dict(tags,tick=tick,observation_sha256=observation_hash,
                    submitted='null' if action is None else json.dumps(int(action) if np.isscalar(action) else np.asarray(action).tolist()),**diag))
            metrics=info['episode_metrics']
            if not done or abs(-20*reward-metrics['extended_cost'])>1e-5 or metrics['invalid_assignments']:raise RuntimeError('diagnostic episode reward/legality')
            row=dict(tags,**metrics,reward=reward,conflicts=conflicts,rejected_proposals=rejected,
                normalized_cost=metrics['extended_cost']/4,normalized_pending=metrics['terminal_pending']/4,
                normalized_failures=metrics['policy_failures']/4);journals['episodes'].add(row);rows.append(row)
    finally:
        for j in journals.values():j.close()
    final=x.learner.parameter_digest(model) if model else None
    if initial!=final or (choice and final!=choice['parameter_sha256']):raise RuntimeError('frozen model changed')
    audit=replay_job(directory,rows,a,t,decoder,timeout)
    x.write_json(directory/'audit.json',dict(audit,initial_parameter_sha256=initial,final_parameter_sha256=final))
    return dict(method=a,train_seed=t,decoder=decoder,rows=rows,audit=audit,initial_parameter_sha256=initial,final_parameter_sha256=final)


def replay_job(directory,rows,a,t,decoder,timeout=None):
    """Independent physical pass: complete state/event/action chain and aggregates."""
    lookup={int(r['seed']):r for r in rows};coord={};seen=set();episodes={};count=0
    with (directory/'coordination.csv').open() as stream:
        for r in csv.DictReader(stream):
            key=int(r['seed']),int(r['tick'])
            if key in coord:raise RuntimeError('duplicate coordination')
            coord[key]=r
    with (directory/'decisions.csv').open() as stream:
        for r in csv.DictReader(stream):
            if timeout and count%1000==0:timeout()
            seed,tick=int(r['seed']),int(r['tick']);key=seed,tick
            if key in seen or seed not in lookup:raise RuntimeError('duplicate/unexpected physical record')
            seen.add(key)
            if seed not in episodes:
                probe=x.ResourceEnv(seed,4,a);probe.reset();raw=WaitingEnv(probe.config);raw.reset(role_seed(seed,x.VERSION+':N4:physical'))
                episodes[seed]=dict(probe=probe,state=raw.state,events=raw.events,metrics=dict.fromkeys(raw.metrics,0.),tick=0,broker=dict.fromkeys((*x.COUNTERS,'conflicts','rejected_proposals'),0))
            ep=episodes[seed];probe=ep['probe'];state=DispatchState(**{k:tuple(v) for k,v in json.loads(r['state']).items()});pairs=tuple(map(tuple,json.loads(r['action'])))
            if state!=ep['state'] or tick!=ep['tick'] or tuple(json.loads(r['events']))!=ep['events'][tick]:raise RuntimeError('physical state/event chain')
            if tick<48:
                cr=coord.get(key)
                if cr is None:raise RuntimeError('missing coordination record')
                probe.env.state=state;probe.env.time=tick;obs=probe.observe()
                if hashlib.sha256(obs['state'].tobytes()).hexdigest()!=cr['observation_sha256']:raise RuntimeError('observation replay')
                submitted=json.loads(cr['submitted'])
                if isinstance(submitted,list):submitted=np.asarray(submitted,dtype=int)
                actual,diag=probe.resolve(submitted)
                if actual!=pairs or any(int(cr[k])!=v for k,v in diag.items()):raise RuntimeError('action/coordination replay')
                for k in ep['broker']:ep['broker'][k]+=diag[k]
            else:
                from ht_pdm_fjsp.technician_reservation_env import deadline
                if deadline(probe.config,state)!=pairs:raise RuntimeError('common tail replay')
            following,cost,parts=transition(probe.config,state,pairs,json.loads(r['events']))
            if asdict(following)!={k:tuple(v) for k,v in json.loads(r['following']).items()} or abs(cost-float(r['cost']))>1e-8:raise RuntimeError('scalar transition replay')
            for k,v in parts.items():ep['metrics'][k]+=v
            ep.update(state=following,tick=tick+1)
            if tick==47:
                ep['cutoff']=dict(ep['metrics']);ep['pending']=sum(failed and m not in following.assigned for m,failed in enumerate(following.failed))
            count+=1
    if count!=60*len(rows) or len(coord)!=48*len(rows) or set(episodes)!=set(lookup):raise RuntimeError('exact physical grid')
    for seed,ep in episodes.items():
        row=lookup[seed];m=ep['metrics'];cut=ep['cutoff']
        if ep['tick']!=60 or any(abs(v-row[k])>1e-8 for k,v in m.items()):raise RuntimeError('physical aggregate replay')
        if (abs(row['policy_cost']-cut['objective'])>1e-8 or abs(row['availability']-(1-cut['unavailable_ticks']/192))>1e-8
            or row['terminal_pending']!=ep['pending'] or row['policy_failures']!=cut['failures']
            or any(row['broker_'+k]!=v for k,v in ep['broker'].items())):raise RuntimeError('health/broker aggregate replay')
    return dict(status='PASS',method=a,train_seed=t,decoder=decoder,episodes=len(rows),physical_ticks_replayed=count,invalid_assignments=0)


def audit_decoder_log(directory,a,decoder,rows):
    coord={(int(r['seed']),int(r['tick'])):r for r in csv.DictReader((directory/'coordination.csv').open())};seen=set()
    for r in csv.DictReader((directory/'decoder_choices.csv').open()):
        key=int(r['seed']),int(r['tick'])
        if key in seen:raise RuntimeError('duplicate decoder record')
        seen.add(key)
        logs=np.asarray(json.loads(r['component_logs']),float);tokens=np.asarray(json.loads(r['matching_tokens']),int)
        physical=(logs[:,0] if a=='forward_order_ppo' else logs[:,1] if a=='reverse_order_ppo' else np.logaddexp(logs[:,0],logs[:,1])-math.log(2))
        probabilities=np.exp(physical);index=int(r['standard_index']);winner=int(r['exact_index'])
        if (logs.shape!=(len(tokens),2) or tokens.shape[1:]!=(2,) or abs(probabilities.sum()-1)>2e-5
            or not np.allclose(physical,json.loads(r['physical_log_probabilities']),atol=1e-6,rtol=0)
            or winner!=int(np.argmax(json.loads(r['physical_log_probabilities'])))):raise RuntimeError('physical decoder score replay')
        standard=np.asarray(json.loads(r['standard']));exact=np.asarray(json.loads(r['exact']));submitted=np.asarray(json.loads(coord[key]['submitted']))
        if (not np.array_equal(tokens[index],standard[:-1]) or not np.array_equal(tokens[winner],exact[:-1])
            or not np.array_equal(submitted,standard if decoder=='standard' else exact)
            or int(r['disagrees'])!=int(not np.array_equal(standard[:-1],exact[:-1]))
            or abs(float(r['standard_probability'])-probabilities[index])>1e-6
            or abs(float(r['exact_probability'])-probabilities[winner])>1e-6):raise RuntimeError('decoder action replay')
    if len(seen)!=(48*len(rows) if a in x.CORE_ARMS else 0):raise RuntimeError('decoder log grid')
    return len(seen)


def summary(rows,out,cfg):
    metrics=('normalized_cost','availability','normalized_pending','normalized_failures','conflicts','rejected_proposals')
    expected={(a,t,d,s) for a,t,d in jobspecs(cfg) for s in cfg['seeds']}
    lookup={(r['method'],r['train_seed'],r['decoder'],r['seed']):r for r in rows}
    if len(rows)!=len(expected) or set(lookup)!=expected:raise RuntimeError('exact diagnostic episode grid')
    populations={}
    for r in rows:
        if populations.setdefault(r['seed'],r['population_sha256'])!=r['population_sha256']:raise RuntimeError('unpaired diagnostic physical tape')
        if r['invalid_assignments'] or abs(r['reward']*20+r['extended_cost'])>1e-5:raise RuntimeError('diagnostic health/reward')
    means={};replicas={}
    for a,t,d in jobspecs(cfg):
        key=a+':'+d;group=[lookup[a,t,d,s] for s in cfg['seeds']]
        replicas.setdefault(key,[]).append(dict(train_seed=t,**{m:float(np.mean([r[m] for r in group])) for m in metrics}))
    for key,group in replicas.items():means[key]={m:float(np.mean([r[m] for r in group])) for m in metrics}
    effects={}
    for a in x.CORE_ARMS:
        std,exact=means[a+':standard'],means[a+':exact']
        effects[a]=dict(cost_gain=1-exact['normalized_cost']/std['normalized_cost'],
            availability_difference=exact['availability']-std['availability'],pending_difference=exact['normalized_pending']-std['normalized_pending'],
            failures_difference=exact['normalized_failures']-std['normalized_failures'],
            paired_cost_differences=[lookup[a,t,'exact',s]['normalized_cost']-lookup[a,t,'standard',s]['normalized_cost'] for t in cfg['train_seeds'] for s in cfg['seeds']])
    decode={};starts={}
    for r in csv.DictReader((out/'decoder_choices.csv').open()):decode.setdefault(r['method']+':'+r['decoder'],[]).append(r)
    for r in csv.DictReader((out/'starts.csv').open()):starts.setdefault(r['method']+':'+r['decoder'],[]).append(r)
    return dict(classification='FROZEN_DEVELOPMENT_DIAGNOSTIC',goal_pass=False,scientific_test_opened=False,
        training_decisions=0,teacher_assistance=False,means=means,per_replica_means=replicas,decoder_effects=effects,
        decoder_diagnostics={key:dict(states=len(group),disagreement_rate=float(np.mean([int(r['disagrees']) for r in group])),
            mean_physical_entropy=float(np.mean([float(r['physical_entropy']) for r in group])),
            mean_standard_probability=float(np.mean([float(r['standard_probability']) for r in group])),
            mean_exact_probability=float(np.mean([float(r['exact_probability']) for r in group])),
            max_normalization_error=max(float(r['normalization_error']) for r in group)) for key,group in decode.items()},
        service_diagnostics={key:dict(starts=len(group),mean_duration=float(np.mean([float(r['duration']) for r in group])),
            long_service_fraction=float(np.mean([float(r['duration'])>1 for r in group])),
            mean_restoration=float(np.mean([float(r['restoration']) for r in group])),
            corrective_fraction=float(np.mean([float(r['corrective']) for r in group]))) for key,group in starts.items()},
        inference_limit='exact decoding enumerates N4 catalogue for diagnosis only; no scalable-quality or causal-duration claim')


def jobspecs(cfg):
    return [(a,t,d) for a in x.arms(4) for d in (('standard','exact') if a in x.CORE_ARMS else ('standard',)) for t in cfg['train_seeds']]+[('deadline',-1,'standard')]


def run(source_dir,output_dir):
    cfg=settings();out=Path(output_dir).resolve();source=Path(source_dir).resolve();start=time.monotonic();torch.set_num_threads(1)
    if out.exists() and any(out.iterdir()):raise FileExistsError('diagnostic prior run overwrite')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('commit verified diagnostic code before full run')
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(protocol=VERSION,status='RUNNING',stage='validating_source',started_utc=datetime.now(timezone.utc).isoformat(),
        scientific_test_opened=False,teacher_assistance=False,git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=False))
    x.write_json(out/'manifest.json',manifest);journals={}
    try:
        parent_cfg,choices,provenance=validate_source(source)
        x.write_json(out/'source_provenance.json',provenance);x.write_json(out/'resolved_config.json',cfg);x.write_json(out/'seed_audit.json',seed_audit())
        source_meta=out/'parent_snapshot';source_meta.mkdir()
        for name in ('manifest.json','resolved_config.json','summary.json','checkpoint_choices.json','training_audit.json','audit.json','protocol.md','seed_registry.json'):
            shutil.copy2(source/name,source_meta/name)
        for key,choice in choices.items():
            target=out/'frozen_checkpoints'/choice['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/choice['path'],target)
        snap=out/'source_snapshot';snap.mkdir()
        for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,snap/f.name)
        shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'seed_registry.json')
        sealed={str(p.relative_to(out)):x.sha(p) for base in (source_meta,snap,out/'frozen_checkpoints') for p in base.rglob('*') if p.is_file()}
        sealed.update({name:x.sha(out/name) for name in ('protocol.md','seed_registry.json','source_provenance.json','resolved_config.json','seed_audit.json')})
        manifest.update(stage='frozen_evaluation',source_seal=sealed);x.write_json(out/'manifest.json',manifest)
        specs=jobspecs(cfg);jobs=[(cfg,str(source),parent_cfg,choices.get(f'4:{a}:{t}'),a,t,d,str(out),start) for a,t,d in specs]
        results={}
        with ProcessPoolExecutor(max_workers=cfg['training_processes'],mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(evaluate_job,job) for job in jobs]
            for result in x.pool_results(futures,out,manifest,start,'Frozen decoder diagnostic'):
                key=result['method'],result['train_seed'],result['decoder'];results[key]=result
                manifest['completed_jobs']=len(results);x.write_json(out/'manifest.json',manifest)
        journals={k:x.Journal(out/f'{k}.csv') for k in ('episodes','decisions','coordination','decoder_choices','starts')};rows=[];audits=[];decoder_states=0
        for a,t,d in specs:
            result=results[a,t,d];directory=out/'jobs'/a/d/str(t);rows.extend(result['rows']);audits.append(dict(result['audit'],initial_parameter_sha256=result['initial_parameter_sha256'],final_parameter_sha256=result['final_parameter_sha256']))
            decoder_states+=audit_decoder_log(directory,a,d,result['rows'])
            for name,journal in journals.items():
                with (directory/f'{name}.csv').open() as stream:
                    for row in csv.DictReader(stream):journal.add(row)
        for journal in journals.values():journal.close()
        journals.clear();report=summary(rows,out,cfg);x.write_json(out/'summary.json',report)
        replay_count=sum(r['physical_ticks_replayed'] for r in audits)
        if len(rows)!=2752 or replay_count!=165120 or decoder_states!=92160 or len(results)!=43:raise RuntimeError('full diagnostic registered budget')
        x.write_json(out/'audit.json',dict(status='PASS',jobs=audits,episodes=len(rows),physical_ticks_replayed=replay_count,
            decoder_states_replayed=decoder_states,invalid_assignments=0,unchanged_models=True,scientific_test_opened=False))
        if any(x.sha(out/name)!=digest for name,digest in sealed.items()):raise RuntimeError('diagnostic sealed input changed')
        if x.sha(source/'manifest.json')!=provenance['manifest_sha256'] or any(x.sha(source/r['path'])!=r['sha256'] for r in choices.values()):raise RuntimeError('parent frozen source/checkpoint changed')
        if x.sha(PLAN)!=sealed['protocol.md'] or x.sha(REGISTRY)!=sealed['seed_registry.json'] or any(x.sha(Path(__file__).parent/f.name)!=sealed['source_snapshot/'+f.name] for f in snap.iterdir()):raise RuntimeError('working diagnostic source changed')
        if time.monotonic()-start>cfg['max_seconds']:raise TimeoutError('registered diagnostic hard stop')
        manifest.update(status='COMPLETED',stage='complete',audit_status='PASS',elapsed_seconds=time.monotonic()-start,completed_utc=datetime.now(timezone.utc).isoformat(),goal_pass=False)
        manifest['output_bindings']={str(p.relative_to(out)):dict(sha256=x.sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'}
        x.write_json(out/'manifest.json',manifest);print(json.dumps(dict(status='COMPLETED',audit='PASS',output=str(out))))
        return manifest
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);x.write_json(out/'manifest.json',manifest);raise
    finally:
        for journal in journals.values():journal.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.source,a.output_dir)

if __name__=='__main__':main()
