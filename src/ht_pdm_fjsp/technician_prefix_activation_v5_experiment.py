"""Prospective development and sealed confirmation for resource reservation PPO."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import itertools
import multiprocessing as mp
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_balanced_algorithms as learner
from ht_pdm_fjsp.maintenance_balanced_confirmation_mac import registered_integers
from ht_pdm_fjsp.maintenance_opportunity_cost import Journal,write_csv,write_json
from ht_pdm_fjsp.technician_prefix_activation_v5_env import VERSION,ResourceEnv,scaled_config
from ht_pdm_fjsp.technician_prefix_activation_v5_policy import ReservationPolicy

ROOT=Path(__file__).resolve().parents[2]
REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
GRAPH_ARMS=('reservation_ppo','mask_only_ppo','independent_ppo')
ARMS=(*GRAPH_ARMS,'ppo','mappo')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(profile):
    if profile not in ('smoke','development','confirmation'): raise ValueError('unknown profile')
    raw=json.loads(REGISTRY.read_text())['panels'][profile];smoke=profile=='smoke';final=profile=='confirmation'
    return dict(profile=profile,train_seeds=raw['train_seeds'],analysis_seed=raw['analysis_seed'],
        development_seeds=list(range(raw['development_start'],raw['development_start']+raw['development_count'])),
        test_seeds=list(range(raw['test_start'],raw['test_start']+raw['test_count'])) if 'test_start' in raw else [],
        scales=[4,8] if smoke else [4,8,16,32] if final else [4,8],
        env_steps=384 if smoke else 491520,vector_envs=2 if smoke else 16,
        training_processes=8 if (smoke or final) else 4,
        rollout=192 if smoke else 3072,checkpoints=[192,384] if smoke else [245760,368640,491520],
        hidden=64,mixer_hidden=32,gamma=1.,learning_rate=.0003,epochs=2 if smoke else 4,
        minibatch=48 if smoke else 256,clip=.2,entropy_weight=.01,value_weight=.5,gradient_clip=.5,
        teacher_bank_steps=192 if smoke else 122880,teacher_epochs=2 if smoke else 32,
        teacher_minibatch=48 if smoke else 512,teacher_temperature=1.5,
        behavior_log_atol=2e-5,behavior_log_rtol=0.,behavior_value_atol=2e-5,behavior_value_rtol=1e-6,
        torch_threads=1,bootstrap_samples=200 if smoke else 40000,
        confirmation_round=1,round_alpha=.025,
        confidence=1-(.025 if final else .05)/17,max_seconds=172800)


def arms(n): return ARMS if n==4 else GRAPH_ARMS


def eligible_steps(a,cfg):return cfg['checkpoints'] if a in GRAPH_ARMS else [491520]


def assessed_steps(a,cfg):return eligible_steps(a,cfg)


def seed_audit():
    history=set();hashes={}
    for p in sorted((ROOT/'configs').glob('*seed_registry.json')):
        if p!=REGISTRY:
            history|=registered_integers(json.loads(p.read_text()));hashes[p.name]=sha(p)
    groups=[];derived=[]
    for profile in ('smoke','development','confirmation'):
        cfg=settings(profile)
        groups.extend([set(cfg['train_seeds']),set(cfg['development_seeds']),set(cfg['test_seeds']),{cfg['analysis_seed']}])
        for n in cfg['scales']:
            for s in (*cfg['development_seeds'],*cfg['test_seeds']):
                from ht_pdm_fjsp.maintenance_dispatch import role_seed
                derived.append(role_seed(s,f'{VERSION}:N{n}:physical'))
            for s in cfg['train_seeds']:
                for role in ('actions','minibatches','replay'):
                    derived.append(learner.role_seed(s,f'{VERSION}:N{n}:'+role))
                for w in range(cfg['vector_envs']):
                    # Audit actual environment stream seeds, not only raw seeds.
                    from ht_pdm_fjsp.maintenance_dispatch import role_seed
                    for e in range(cfg['env_steps']//(48*cfg['vector_envs'])):
                        derived.append(role_seed(s,f'{VERSION}:N{n}:training_physical:worker:{w}:episode:{e}'))
    for profile in ('smoke','development','confirmation'):
        cfg=settings(profile)
        for s in cfg['train_seeds']:
            for role in ('initialization:reservation_pair','teacher_actions','teacher_minibatches','teacher_augmentation'):
                derived.append(learner.role_seed(s,f'{VERSION}:'+role))
            from ht_pdm_fjsp.maintenance_dispatch import role_seed
            for w in range(cfg['vector_envs']):
                for e in range(cfg['teacher_bank_steps']//(48*cfg['vector_envs'])):
                    derived.append(role_seed(s,f'{VERSION}:N4:training_physical:worker:{1000+w}:episode:{e}'))
    own=set.union(*groups)
    if any(a&b for a,b in itertools.combinations(groups,2)) or own&history: raise ValueError('raw seed overlap')
    if len(derived)!=len(set(derived)) or set(derived)&(own|history): raise ValueError('derived seed overlap')
    return dict(status='PASS',raw_unique=len(own),derived_unique=len(derived),historical_registry_hashes=hashes,
        intentional_pairing='same physical tapes across methods and same initial parameters for reservation/independent')


def make_model(algorithm,n,cfg,seed):
    # Shared initial weights for the architecture ablation, independent of global RNG.
    role='reservation_pair' if algorithm in GRAPH_ARMS else algorithm
    with torch.random.fork_rng():
        torch.manual_seed(learner.role_seed(seed,f'{VERSION}:initialization:{role}' if algorithm in GRAPH_ARMS else f'{VERSION}:N{n}:initialization:{role}'))
        if algorithm in GRAPH_ARMS: return ReservationPolicy(algorithm,cfg['hidden'])
        model=learner.Model(algorithm,cfg['hidden'])
        if n!=4:
            k=n//2
            model.actor=learner.mlp(15+7*k+n,cfg['hidden'],k+1)
            model.critic=learner.mlp(6*n+3*k+4*n*k+9,cfg['hidden'],1)
        return model


def save_model(model,path,n,cfg,seed,steps,probe):
    if path.exists(): raise FileExistsError('checkpoint overwrite')
    payload=dict(protocol=VERSION,algorithm=model.algorithm,n=n,train_seed=seed,steps=steps,
        hidden=cfg['hidden'],parameter_sha256=learner.parameter_digest(model),state_dict=model.state_dict())
    torch.save(payload,path)
    loaded=load_model(path,n,cfg,seed,model.algorithm,steps)
    if not np.array_equal(model.select(probe),loaded.select(probe)): raise RuntimeError('checkpoint prediction roundtrip')
    return dict(path=str(path),sha256=sha(path),parameter_sha256=payload['parameter_sha256'])


def load_model(path,n,cfg,seed,algorithm,steps):
    p=torch.load(path,map_location='cpu',weights_only=True)
    if any(p.get(k)!=v for k,v in dict(protocol=VERSION,algorithm=algorithm,n=n,train_seed=seed,steps=steps,hidden=cfg['hidden']).items()):
        raise RuntimeError('checkpoint metadata mismatch')
    model=make_model(algorithm,n,cfg,seed);model.load_state_dict(p['state_dict']);model.eval()
    if learner.parameter_digest(model)!=p['parameter_sha256']: raise RuntimeError('checkpoint parameter digest')
    return model


class Tags:
    def __init__(self,journal,tags): self.journal=journal;self.tags=tags
    def add(self,row): self.journal.add(dict(self.tags,**row))


def evaluate(method,n,train_seed,seed,model=None,coordination=None,decisions=None,latencies=None):
    tags=dict(method=method,n=n,train_seed=train_seed,seed=seed)
    env=ResourceEnv(seed,n,method,journal=Tags(decisions,tags) if decisions else None)
    obs=env.reset();rewards=0.;conflicts=rejected=0;times=[];environment_times=[]
    for tick in range(48):
        before=time.perf_counter();action=model.select(obs) if model else None
        elapsed=(time.perf_counter()-before)*1000;times.append(elapsed)
        state_hash=hashlib.sha256(obs['state'].tobytes()).hexdigest()
        before=time.perf_counter();obs,r,done,info=env.step(action);environment_times.append((time.perf_counter()-before)*1000)
        rewards+=r;diag=info['coordination'];conflicts+=diag['conflicts'];rejected+=diag['rejected_proposals']
        if coordination:
            coordination.add(dict(tags,tick=tick,observation_sha256=state_hash,
                submitted=json.dumps(int(action) if np.isscalar(action) else np.asarray(action).tolist()) if model else 'null',
                **diag,decision_ms=elapsed,environment_ms=environment_times[-1]))
    m=info['episode_metrics']
    if not done or abs(-20*rewards-m['extended_cost'])>1e-5 or m['invalid_assignments']: raise RuntimeError('episode reconciliation')
    if latencies is not None: latencies.extend(times)
    return dict(tags,**m,conflicts=conflicts,rejected_proposals=rejected,decision_mean_ms=float(np.mean(times)),
        decision_p95_ms=float(np.quantile(times,.95)),environment_mean_ms=float(np.mean(environment_times)),
        normalized_cost=m['extended_cost']/n,normalized_pending=m['terminal_pending']/n,
        normalized_failures=m['policy_failures']/n)


def estimates(means,scales):
    """Cost gains and health differences for the registered full/mask contrast."""
    result={}
    full=[];masked=[]
    for n in scales:
        current=means[n]['reservation_ppo'];reference=means[n]['mask_only_ppo']
        full.append(current[0]);masked.append(reference[0])
        result[f'N{n}_gain_full_vs_mask']=1-current[0]/reference[0]
        for label,index in (('availability',1),('pending_per_machine',2),('failures_per_machine',3)):
            result[f'N{n}_{label}_full_minus_mask']=current[index]-reference[index]
    result['pooled_gain_full_vs_mask']=1-sum(full)/sum(masked)
    return result


def summarize(rows,cfg,latencies):
    metrics=('normalized_cost','availability','normalized_pending','normalized_failures')
    trains=cfg['train_seeds'];tests=cfg['test_seeds'];lookup={(r['n'],r['method'],r['train_seed'],r['seed']):r for r in rows}
    expected={(n,a,t,s) for n in cfg['scales'] for a in arms(n) for t in trains for s in tests}
    expected|={(n,'deadline',-1,s) for n in cfg['scales'] for s in tests}
    if len(rows)!=len(expected) or set(lookup)!=expected: raise RuntimeError('exact evaluation grid')
    arrays={n:{a:np.array([[[lookup[n,a,t,s][m] for m in metrics] for s in tests] for t in trains]) for a in arms(n)} for n in cfg['scales']}
    for n in cfg['scales']:
        reference=np.array([[lookup[n,'deadline',-1,s][m] for m in metrics] for s in tests])
        arrays[n]['deadline']=np.tile(reference[None],(len(trains),1,1))
    means={n:{a:v.mean((0,1)) for a,v in panel.items()} for n,panel in arrays.items()}
    point=estimates(means,cfg['scales']);draws={key:[] for key in point};rng=np.random.default_rng(learner.role_seed(cfg['analysis_seed'],VERSION+':bootstrap'))
    nt,ns=len(trains),len(tests)
    for start in range(0,cfg['bootstrap_samples'],128):
        b=min(128,cfg['bootstrap_samples']-start)
        tc=np.eye(nt)[rng.integers(nt,size=(b,nt))].mean(1)
        sampled={}
        for n,panel in arrays.items():
            sc=np.eye(ns)[rng.integers(ns,size=(b,ns))].mean(1)
            sampled[n]={a:np.einsum('bt,bs,tsk->bk',tc,sc,v,optimize=True).T for a,v in panel.items()}
        for key,value in estimates(sampled,cfg['scales']).items():draws[key].extend(value.tolist())
    tail=(.025/17) if cfg['profile']=='confirmation' else (.05/17)
    intervals={key:dict(estimate=float(value),lower=float(np.quantile(draws[key],tail)),upper=float(np.quantile(draws[key],1-tail))) for key,value in point.items()}
    guards={'pooled_prefix_gain':intervals['pooled_gain_full_vs_mask']['estimate']>=.01 and intervals['pooled_gain_full_vs_mask']['lower']>0,
        'all_scales_point_noninferior':all(point[f'N{n}_gain_full_vs_mask']>=-.01 for n in cfg['scales'])}
    for n in cfg['scales']:
        guards[f'N{n}_health']=(intervals[f'N{n}_availability_full_minus_mask']['lower']>=-.01 and
            intervals[f'N{n}_pending_per_machine_full_minus_mask']['upper']<=.025 and
            intervals[f'N{n}_failures_per_machine_full_minus_mask']['upper']<=.25)
        guards[f'N{n}_latency']=float(np.quantile(latencies[f'{n}:reservation_ppo'],.95))<=50
    leave_one_out=[]
    if nt>1:
        for omit in range(nt):
            retained=[i for i in range(nt) if i!=omit]
            loo={n:{a:v[retained].mean((0,1)) for a,v in panel.items()} for n,panel in arrays.items()}
            leave_one_out.append(dict(omitted_train_seed=trains[omit],estimates={key:float(value) for key,value in estimates(loo,cfg['scales']).items()}))
    return dict(classification='ENGINEERING_ONLY' if cfg['profile']=='smoke' else 'CONFIRMATORY',
        goal_pass=all(guards.values()),guards=guards,intervals=intervals,family_size=17,
        one_sided_family_alpha=.025 if cfg['profile']=='confirmation' else .05,
        bootstrap_unit='training replica crossed with paired episodes by scale',
        leave_one_replica_out=leave_one_out,
        means={str(n):{a:dict(zip(metrics,v.tolist())) for a,v in panel.items()} for n,panel in means.items()},
        per_replica={str(n):{a:[dict(zip(metrics,r.tolist())) for r in v.mean(1)] for a,v in panel.items()} for n,panel in arrays.items()},
        latency={key:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.quantile(v,.95)),p99_ms=float(np.quantile(v,.99)),samples=len(v)) for key,v in latencies.items()})


def development_summary(rows,cfg,references=None):
    selected={};means={};selected_metrics={}
    for n in cfg['scales']:
        means[n]={};selected_metrics[n]={}
        for a in arms(n):
            costs=[];panel=[]
            for t in cfg['train_seeds']:
                candidates=[r for r in rows if r['n']==n and r['method']==a and r['train_seed']==t]
                bystep={step:np.mean([r['extended_cost'] for r in candidates if r['checkpoint_steps']==step]) for step in eligible_steps(a,cfg)}
                step=min(bystep,key=lambda s:(bystep[s],s));selected[f'{n}:{a}:{t}']=step;costs.append(bystep[step])
                panel.extend(r for r in candidates if r['checkpoint_steps']==step)
            means[n][a]=float(np.mean(costs))
            selected_metrics[n][a]={m:float(np.mean([r[m] for r in panel])) for m in ('availability','normalized_pending','normalized_failures','invalid_assignments')}
    full=sum(means[n]['reservation_ppo']/n for n in cfg['scales'])
    mask=sum(means[n]['mask_only_ppo']/n for n in cfg['scales'])
    pooled_gain=1-full/mask
    scale_gains={n:1-means[n]['reservation_ppo']/means[n]['mask_only_ppo'] for n in cfg['scales']}
    health={n:dict(
        availability=selected_metrics[n]['reservation_ppo']['availability']>=selected_metrics[n]['mask_only_ppo']['availability']-.01,
        pending=selected_metrics[n]['reservation_ppo']['normalized_pending']<=selected_metrics[n]['mask_only_ppo']['normalized_pending']+.025,
        failures=selected_metrics[n]['reservation_ppo']['normalized_failures']<=selected_metrics[n]['mask_only_ppo']['normalized_failures']+.25,
        invalid=selected_metrics[n]['reservation_ppo']['invalid_assignments']==0 and selected_metrics[n]['mask_only_ppo']['invalid_assignments']==0)
        for n in cfg['scales']}
    gate=pooled_gain>=.005 and all(v>=-.01 for v in scale_gains.values()) and all(all(v.values()) for v in health.values())
    return dict(classification='DEVELOPMENT_ONLY',test_opened=False,means=means,selected_steps=selected,
        selected_metrics=selected_metrics,pooled_gain_full_vs_mask=float(pooled_gain),scale_gains=scale_gains,
        health_guards=health,development_screen_pass=bool(gate))


def fit_job(job):
    """Independent deterministic CPU training job; never opens test seeds."""
    cfg,n,a,t,out_name,global_start=job
    out=Path(out_name);torch.set_num_threads(1)
    def timeout():
        if (out/'abort.json').exists():raise RuntimeError('sibling job failed; technical abort')
        if time.monotonic()-global_start>cfg['max_seconds']:raise TimeoutError('registered global deadline')
    timeout();model=make_model(a,n,cfg,t)
    initial=learner.parameter_digest(model)
    from ht_pdm_fjsp.technician_prefix_activation_v5_distillation import load_warm,load_reference
    warm=load_warm(model,out,t) if a in GRAPH_ARMS else None
    if warm is None:
        meta=json.loads((out/'teachers/references.json').read_text())['references'][f'{a}:{t}']
        model=load_reference(out,meta)
        initial=learner.parameter_digest(model)
    parameters=sum(p.numel() for p in model.parameters())
    directory=out/f'N{n}'/a/str(t);directory.mkdir(parents=True)
    checkpoints=directory/'checkpoints';checkpoints.mkdir()
    journals={name:Journal(directory/f'{name}.csv') for name in ('training_progress','training_episodes','development_episodes')}
    probe=ResourceEnv(cfg['development_seeds'][0],n,a).reset();candidates=[];devrows=[];precursor_actions={}
    def assess(current,steps,audit=None):
        if a in GRAPH_ARMS and steps==cfg['env_steps']//2:
            precursor_actions['sha256']=current.training_action_sha256.hexdigest()
        if a=='reservation_ppo' and steps==cfg['env_steps']//2:
            if current.algorithm!='mask_only_ppo':raise RuntimeError('prefix switch requires mask-only precursor')
            current.algorithm='reservation_ppo'
        timeout();path=checkpoints/f'step_{steps:09d}.pt'
        meta=save_model(current,path,n,cfg,t,steps,probe);meta['path']=str(path.relative_to(out))
        rows=[];current.eval()
        for seed in tqdm(cfg['development_seeds'],desc=f'Dev N{n}/{a}/{t}/{steps}',leave=False):
            timeout();row=evaluate(a,n,t,seed,current);row['checkpoint_steps']=steps
            journals['development_episodes'].add(row);devrows.append(row);rows.append(row)
        current.train()
        if steps in eligible_steps(a,cfg):candidates.append(dict(meta,steps=steps,development_cost=float(np.mean([r['extended_cost'] for r in rows]))))
    try:
        before=time.monotonic()
        if warm is None:
            assess(model,491520)
            audit=dict(algorithm=a,train_seed=t,learned_steps=0,physical_ticks=0,training_episodes=0,optimizer_steps=0,
                changed=False,behavior_max_error=0.,behavior_normalized_max_error=0.,parameter_sha256=learner.parameter_digest(model),reference=True,
                foundation_training_steps=491520,source_selected_steps=meta['steps'],source_sha256=meta['sha256'])
            return dict(n=n,algorithm=a,train_seed=t,initial=initial,parameters=parameters,devrows=devrows,
                audit=dict(audit,n=n,seconds_including_development=time.monotonic()-before),choice=candidates[0])
        if a=='reservation_ppo':model.algorithm='mask_only_ppo'
        model.training_action_sha256=hashlib.sha256()
        class TrainTags:
            def __init__(self,journal):self.journal=journal
            def add(self,row):self.journal.add(dict(row,training_mode=row['algorithm'],algorithm=a,n=n))
        train_journals={name:TrainTags(journals[name]) for name in ('training_progress','training_episodes')}
        audit=learner.fit(model,dict(cfg,rng_namespace=f'{VERSION}:N{n}:'),t,
            lambda seed,worker:ResourceEnv(seed,n,a,worker,True),train_journals,assess,timeout)
        import csv
        with (directory/'training_progress.csv').open() as stream:
            precursor=[r for r in csv.DictReader(stream) if int(r['learned_steps'])==cfg['env_steps']//2]
        if len(precursor)!=1:raise RuntimeError('missing branch update')
        return dict(n=n,algorithm=a,train_seed=t,initial=initial,parameters=parameters,devrows=devrows,
            warm_parameter_sha256=warm['parameter_sha256'],precursor_parameter_sha256=precursor[0]['parameter_sha256'],
            precursor_action_sha256=precursor_actions['sha256'],
            audit=dict(audit,n=n,seconds_including_development=time.monotonic()-before,reference=False,
                warm_state_exposures=warm['state_exposures'],warm_optimizer_steps=warm['optimizer_steps'],bank_sha256=warm['bank_sha256'],
                warm_parameter_sha256=warm['parameter_sha256'],precursor_parameter_sha256=precursor[0]['parameter_sha256'],
                precursor_action_sha256=precursor_actions['sha256']),
            choice=min(candidates,key=lambda v:(v['development_cost'],v['steps'])))
    finally:
        for journal in journals.values():journal.close()


def pool_results(futures,out,manifest,start,description):
    """Expose technical failure before pool shutdown; stop sibling computations."""
    for future in tqdm(as_completed(futures),total=len(futures),desc=description,unit='job'):
        try:
            result=future.result()
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            write_json(Path(out)/'abort.json',dict(reason='TECHNICAL_FAILURE',error=error))
            manifest.update(status='FAILED',error=error,abort_reason='TECHNICAL_FAILURE',elapsed_seconds=time.monotonic()-start)
            write_json(Path(out)/'manifest.json',manifest)
            for pending in futures:pending.cancel()
            raise
        yield result


def run(profile,output_dir,teacher_dir=None,development_artifact=None):
    cfg=settings(profile);out=Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('prior run overwrite')
    dev_gate=None
    if profile=='confirmation':
        if development_artifact is None:raise ValueError('sealed confirmation requires a passed development artifact')
        prior=Path(development_artifact).resolve()
        prior_manifest=json.loads((prior/'manifest.json').read_text())
        prior_summary=json.loads((prior/'summary.json').read_text())
        prior_audit=json.loads((prior/'audit.json').read_text())
        if (prior_manifest['protocol']!=VERSION or prior_manifest['profile']!='development' or
            prior_manifest['status']!='COMPLETED' or prior_audit['status']!='PASS' or
            not prior_summary['development_screen_pass']):
            raise RuntimeError('development gate failed')
        dev_gate=dict(path=str(prior),manifest_sha256=sha(prior/'manifest.json'),
            summary_sha256=sha(prior/'summary.json'),audit_sha256=sha(prior/'audit.json'))
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);start=time.monotonic()
    git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()))
    if profile!='smoke' and git['dirty']: raise RuntimeError('commit verified code before research run')
    manifest=dict(protocol=VERSION,status='RUNNING',profile=profile,git=git,test_opened=False,development_gate=dev_gate,
        invocation=dict(executable=sys.executable,argv=sys.argv,cwd=str(Path.cwd())),
        started_utc=datetime.now(timezone.utc).isoformat(),runtime=dict(platform=platform.platform(),
            python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu'))
    write_json(out/'manifest.json',manifest);journals={}
    def timeout():
        if time.monotonic()-start>cfg['max_seconds']: raise TimeoutError('fixed hard deadline; incomplete')
    try:
        seed_report=seed_audit()
        from ht_pdm_fjsp.technician_prefix_activation_v5_distillation import freeze_teachers,prepare_warm_job,DEFAULT_TEACHERS
        teacher_report=freeze_teachers(teacher_dir or DEFAULT_TEACHERS,out,cfg)
        write_json(out/'resolved_config.json',cfg);write_json(out/'seed_audit.json',seed_report)
        write_json(out/'physics.json',{str(n):asdict(scaled_config(n)) for n in cfg['scales']})
        write_json(out/'graph_sizes.json',{str(n):dict(edges=n*n//2,
            catalogue_count=sum(math.comb(n,r)*math.comb(n//2,r)*math.factorial(r) for r in range(n//2+1)),
            enumerated_catalogue=n==4) for n in cfg['scales']})
        write_json(out/'execution_contract.json',dict(actor_information='current full graph plus public normalized row/column identities and competing-machine attention',
            coordination='identical mask-only precursor for 245760 PPO decisions, then full selected-edge attention, capacity and direct residual versus continued mask-only; shared reservation mask',
            ablation='mask-only shares full reservation feasibility, resource summary, and order; both selected-edge attention and selected-pair occupancy scalars are zero. Independent removes reservation and prefix',
            value_encoder='separate actor/value encoders; value output N/4 calibration',
            public_remaining_workers='indices >=j in both arms',teacher_augmentation='paired machine and technician bijections',
            attention='four heads plus incident-edge machine pooling',execution='communicating resource agents and shared reservation service',teacher_assistance=True,
            teacher_budget='122880 shared N4 states and 32 mask-only distillation epochs;491520 PPO decisions per target scale with a shared 245760-decision mask-only precursor;491520 foundation teacher decisions',
            latency_scope='single-state neural decision; observation/arbitration/physical stepping measured separately; no network transport claim'))
        snap=out/'source_snapshot';snap.mkdir()
        for p in Path(__file__).parent.glob('*.py'): shutil.copy2(p,snap/p.name)
        history=out/'registry_snapshot';history.mkdir()
        for name in seed_report['historical_registry_hashes']:shutil.copy2(ROOT/'configs'/name,history/name)
        shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'seed_registry.json')
        seals={str(p.relative_to(out)):sha(p) for p in [*snap.iterdir(),*history.iterdir(),out/'protocol.md',out/'seed_registry.json',out/'physics.json',out/'resolved_config.json',*(out/'teachers').iterdir()]}
        manifest['source_seal']=seals;write_json(out/'manifest.json',manifest)
        journals={name:Journal(out/f'{name}.csv') for name in ('training_progress','training_episodes','development_episodes','episodes.partial','coordination','decisions')}
        choices={};audits=[];devrows=[];parameters={};initials={}
        manifest['stage']='teacher_bank_and_distillation';write_json(out/'manifest.json',manifest)
        warm_results={}
        with ProcessPoolExecutor(max_workers=cfg['training_processes'],mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(prepare_warm_job,(cfg,t,str(out),start)) for t in cfg['train_seeds']]
            for result in pool_results(futures,out,manifest,start,'Teacher banks and paired warm starts'):
                timeout();warm_results[str(result['train_seed'])]=result
        write_json(out/'warm_choices.json',warm_results)
        warm_seals={str(p.relative_to(out)):sha(p) for p in (out/'warm').rglob('*') if p.is_file()}
        warm_seals['warm_choices.json']=sha(out/'warm_choices.json')
        jobs=[(cfg,n,a,t,str(out),start) for n in cfg['scales'] for a in arms(n) for t in cfg['train_seeds']]
        manifest['stage']='training_pool';write_json(out/'manifest.json',manifest)
        results={};partial_choices={}
        with ProcessPoolExecutor(max_workers=cfg['training_processes'],mp_context=mp.get_context('spawn')) as pool:
            futures={pool.submit(fit_job,job):job for job in jobs}
            for result in pool_results(futures,out,manifest,start,'Training models'):
                timeout();key=f'{result["n"]}:{result["algorithm"]}:{result["train_seed"]}'
                results[key]=result;partial_choices[key]=result['choice']
                write_json(out/'checkpoint_choices.partial.json',partial_choices)
                manifest['completed_models']=len(results);write_json(out/'manifest.json',manifest)
        # Deterministic merge order, independent of OS process completion order.
        import csv
        for _,n,a,t,_,_ in jobs:
            key=f'{n}:{a}:{t}';result=results[key];directory=out/f'N{n}'/a/str(t)
            audits.append(result['audit']);devrows.extend(result['devrows']);choices[key]=result['choice']
            parameters[f'{n}:{a}']=result['parameters'];initials[key]=result['initial']
            for name in ('training_progress','training_episodes','development_episodes'):
                with (directory/f'{name}.csv').open() as stream:
                    for row in csv.DictReader(stream):journals[name].add(row)
        write_json(out/'training_audit.json',audits)
        write_json(out/'checkpoint_choices.json',choices);choice_hash=sha(out/'checkpoint_choices.json')
        write_json(out/'parameter_counts.json',parameters);write_json(out/'initial_parameter_hashes.json',initials)
        for n in cfg['scales']:
            for t in cfg['train_seeds']:
                if len({initials[f'{n}:{a}:{t}'] for a in GRAPH_ARMS})!=1:raise RuntimeError('paired initial weights')
                full=results[f'{n}:reservation_ppo:{t}'];mask=results[f'{n}:mask_only_ppo:{t}']
                if full['warm_parameter_sha256']!=mask['warm_parameter_sha256']:
                    raise RuntimeError('common warm checkpoint mismatch')
                if full['precursor_parameter_sha256']!=mask['precursor_parameter_sha256']:
                    raise RuntimeError('common precursor parameter mismatch')
                if full['precursor_action_sha256']!=mask['precursor_action_sha256']:
                    raise RuntimeError('common precursor action sequence mismatch')
                for name in ('training_progress','training_episodes'):
                    def precursor_rows(arm):
                        with (out/f'N{n}'/arm/str(t)/f'{name}.csv').open() as stream:
                            return [{k:v for k,v in row.items() if k!='algorithm'} for row in csv.DictReader(stream)
                                if int(row['learned_steps'])<=cfg['env_steps']//2]
                    if precursor_rows('reservation_ppo')!=precursor_rows('mask_only_ppo'):
                        raise RuntimeError(f'common precursor {name} mismatch')
        if profile=='development':
            references=[]
            for n in cfg['scales']:
                for s in tqdm(cfg['development_seeds'],desc=f'Dev reference N{n}'):
                    timeout();references.append(evaluate('deadline',n,-1,s))
            write_csv(out/'development_reference.csv',references)
            summary=development_summary(devrows,cfg,references);write_json(out/'summary.json',summary);rows=[]
        else:
            manifest.update(stage='test',test_opened=True,choices_sha256=choice_hash,choices_sealed_utc=datetime.now(timezone.utc).isoformat());write_json(out/'manifest.json',manifest)
            rows=[];latencies={}
            for n in cfg['scales']:
                controllers=[(a,t) for a in arms(n) for t in cfg['train_seeds']]+[('deadline',-1)]
                for a,t in controllers:
                    choice=choices.get(f'{n}:{a}:{t}')
                    model=load_model(out/choice['path'],n,cfg,t,a,choice['steps']) if choice else None
                    key=f'{n}:{a}';latencies.setdefault(key,[])
                    for s in tqdm(cfg['test_seeds'],desc=f'Test N{n}/{a}/{t}'):
                        timeout();row=evaluate(a,n,t,s,model,journals['coordination'],journals['decisions'],latencies[key])
                        rows.append(row);journals['episodes.partial'].add(row)
            write_csv(out/'episodes.csv',rows);summary=summarize(rows,cfg,latencies);write_json(out/'summary.json',summary)
        for journal in journals.values():journal.close()
        journals.clear()
        audit=audit_run(out,cfg,devrows,rows,audits,choices,timeout)
        if sha(out/'checkpoint_choices.json')!=choice_hash or any(sha(out/p)!=digest for p,digest in {**seals,**warm_seals}.items()): raise RuntimeError('source or checkpoint choices changed')
        if any(sha(out/c['path'])!=c['sha256'] for c in choices.values()): raise RuntimeError('checkpoint content changed')
        if sha(PLAN)!=seals['protocol.md'] or sha(REGISTRY)!=seals['seed_registry.json'] or any(sha(Path(__file__).parent/p.name)!=seals[f'source_snapshot/{p.name}'] for p in snap.iterdir()):
            raise RuntimeError('working source changed during experiment')
        if any(sha(ROOT/'configs'/name)!=digest for name,digest in seed_report['historical_registry_hashes'].items()):raise RuntimeError('historical registry changed during experiment')
        write_json(out/'audit.json',audit)
        manifest.update(status='COMPLETED',audit_status='PASS',stage='complete',elapsed_seconds=time.monotonic()-start,
            training_models=len(audits),test_episodes=len(rows),development_episodes=len(devrows),
            goal_pass=summary.get('goal_pass',False),completed_utc=datetime.now(timezone.utc).isoformat())
        manifest['output_bindings']={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        timeout();write_json(out/'manifest.json',manifest);print(json.dumps(dict(output_dir=str(out),status='COMPLETED',summary=summary)));return manifest
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);write_json(out/'manifest.json',manifest);raise
    finally:
        for journal in journals.values():journal.close()


def audit_run(out,cfg,devrows,rows,audits,choices,timeout=None):
    import csv
    expected={(n,a,t) for n in cfg['scales'] for a in arms(n) for t in cfg['train_seeds']}
    if len(audits)!=len(expected) or {(r['n'],r['algorithm'],r['train_seed']) for r in audits}!=expected:raise RuntimeError('training grid')
    trained={identity for identity in expected if identity[1] in GRAPH_ARMS}
    references=[r for r in audits if r['reference']]
    if len(references)!=2*len(cfg['train_seeds']) or any(r['learned_steps']!=0 or r['foundation_training_steps']!=491520 for r in references):raise RuntimeError('foundation reference budget')
    for r in [r for r in audits if not r['reference']]:
        if r['learned_steps']!=cfg['env_steps'] or r['physical_ticks']!=cfg['env_steps']//48*60 or r['training_episodes']!=cfg['env_steps']//48 or not r['changed'] or r['behavior_normalized_max_error']>1:
            raise RuntimeError('training budget/likelihood audit')
    expected_dev={(n,a,t,step,s) for n,a,t in expected for step in assessed_steps(a,cfg) for s in cfg['development_seeds']}
    if len(devrows)!=len(expected_dev) or {(r['n'],r['method'],r['train_seed'],r['checkpoint_steps'],r['seed']) for r in devrows}!=expected_dev: raise RuntimeError('development grid')
    paired={};seen=set()
    with (out/'training_episodes.csv').open() as f:
        for r in csv.DictReader(f):
            n=int(r['n']);a=r['algorithm'];t=int(r['train_seed']);w=int(r['worker']);e=int(r['episode_index'])
            key=(n,t,w,e);identity=key+(a,)
            if identity in seen:raise RuntimeError('duplicate training episode')
            seen.add(identity);tape=(r['physical_seed'],r['population_sha256'])
            if paired.setdefault(key,tape)!=tape:raise RuntimeError('physical training pairing')
            if abs(float(r['episode_reward'])*20+float(r['extended_cost']))>1e-5:raise RuntimeError('training reward reconciliation')
    expected_episode={(n,t,w,e,a) for n,a,t in trained for w in range(cfg['vector_envs']) for e in range(cfg['env_steps']//(48*cfg['vector_envs']))}
    if seen!=expected_episode:raise RuntimeError('training episode grid')
    from ht_pdm_fjsp.technician_prefix_activation_v5_distillation import audit_bank,load_reference
    warm=json.loads((out/'warm_choices.json').read_text());refs=json.loads((out/'teachers/references.json').read_text())['references']
    for t in cfg['train_seeds']:
        if timeout:timeout()
        panel=warm[str(t)];bank_path=out/'warm'/str(t)/'bank.npz'
        if sha(bank_path)!=panel['bank']['bank_sha256']:raise RuntimeError('shared bank hash')
        with np.load(bank_path) as bank:
            actions=ResourceEnv(t,4,'ppo').actions
            audit_bank(bank,load_reference(out,refs[f'ppo:{t}']),actions,cfg)
            bank_rewards=bank['reward'].reshape(-1,48,cfg['vector_envs']).sum(1)
        ablation=list(panel['arms'].values())
        if len({r['initial_parameter_sha256'] for r in ablation})!=1 or len({r['bank_sha256'] for r in ablation})!=1 or len({r['augmentation_sha256'] for r in ablation})!=1:raise RuntimeError('warm ablation pairing')
        for r in ablation:
            if r['state_exposures']!=cfg['teacher_bank_steps'] or r['state_presentations']!=cfg['teacher_epochs']*cfg['teacher_bank_steps'] or r['epochs']!=cfg['teacher_epochs']:raise RuntimeError('distillation budget')
        episodes=list(csv.DictReader((out/'warm'/str(t)/'bank_episodes.csv').open()))
        identities={(int(r['worker']),int(r['episode_index'])) for r in episodes}
        expected_bank={(1000+w,e) for w in range(cfg['vector_envs']) for e in range(cfg['teacher_bank_steps']//(48*cfg['vector_envs']))}
        if identities!=expected_bank or len(episodes)!=len(expected_bank):raise RuntimeError('bank episode grid')
        for r in episodes:
            if abs(float(r['episode_reward'])*20+float(r['extended_cost']))>1e-5:raise RuntimeError('bank episode reward')
            if abs(bank_rewards[int(r['episode_index']),int(r['worker'])-1000]-float(r['episode_reward']))>2e-4:raise RuntimeError('bank stored rewards')
            from ht_pdm_fjsp.maintenance_dispatch import role_seed
            physical=role_seed(t,f'{VERSION}:N4:training_physical:worker:{r["worker"]}:episode:{r["episode_index"]}')
            if physical!=int(r['physical_seed']):raise RuntimeError('bank physical stream')
        for r in audits:
            if r['train_seed']==t and not r['reference'] and (r['bank_sha256']!=panel['bank']['bank_sha256'] or r['warm_state_exposures']!=cfg['teacher_bank_steps']):raise RuntimeError('fine-tuning bank provenance')
    if cfg['test_seeds']:
        pops={}
        for r in rows:
            key=r['n'],r['seed']
            if pops.setdefault(key,r['population_sha256'])!=r['population_sha256']:raise RuntimeError('test physical pairing')
        # Independently replay all physical transitions, rather than relying on wrappers.
        from ht_pdm_fjsp.maintenance_dispatch import DispatchState,role_seed
        from ht_pdm_fjsp.maintenance_waiting import transition,WaitingEnv
        replay_count=0;replay_episodes={};prefix_mask_audits=0;prefix_nonzero=0;residual_nonzero=0;residual_choice_changes=0
        audit_models={};lookup={(r['n'],r['method'],r['train_seed'],r['seed']):r for r in rows}
        recorded_coord={}
        with (out/'coordination.csv').open() as f:
            for r in csv.DictReader(f):
                key=(int(r['n']),r['method'],int(r['train_seed']),int(r['seed']),int(r['tick']))
                if key in recorded_coord:raise RuntimeError('duplicate coordination record')
                recorded_coord[key]=r
        with (out/'decisions.csv').open() as f:
            for r in csv.DictReader(f):
                if timeout and replay_count%1000==0:timeout()
                n=int(r['n']);c=scaled_config(n)
                state=DispatchState(**{k:tuple(v) for k,v in json.loads(r['state']).items()})
                pairs=tuple(map(tuple,json.loads(r['action'])))
                key=n,r['method'],int(r['train_seed']),int(r['seed']);tick=int(r['tick'])
                if key not in lookup:raise RuntimeError('unexpected replay episode')
                if key not in replay_episodes:
                    physical=role_seed(key[3],f'{VERSION}:N{n}:physical');initial=WaitingEnv(c);initial.reset(physical)
                    replay_episodes[key]=dict(state=initial.state,events=initial.events,tick=0,
                        metrics={k:0. for k in initial.metrics},probe=ResourceEnv(key[3],n,key[1]))
                ep=replay_episodes[key]
                if tick!=ep['tick'] or state!=ep['state'] or tuple(json.loads(r['events']))!=ep['events'][tick]:raise RuntimeError('physical tape/state sequence mismatch')
                if tick<48:
                    cr=recorded_coord.get(key+(tick,))
                    if cr is None:raise RuntimeError('missing coordination record')
                    probe=ep['probe'];probe.env.state=state;probe.env.time=tick
                    obs=probe.observe()
                    if hashlib.sha256(obs['state'].tobytes()).hexdigest()!=cr['observation_sha256']:raise RuntimeError('observation hash mismatch')
                    submitted=json.loads(cr['submitted'])
                    if isinstance(submitted,list):submitted=np.asarray(submitted,dtype=int)
                    decoded,diag=probe.resolve(submitted)
                    if decoded!=pairs or any(int(cr[k])!=v for k,v in diag.items()):raise RuntimeError('coordination/action replay mismatch')
                    if (key[1]=='reservation_ppo' and key[3]==cfg['test_seeds'][0]):
                        model_key=n,key[2]
                        if model_key not in audit_models:
                            audit_models[model_key]={}
                            for arm in ('reservation_ppo','mask_only_ppo'):
                                choice=choices[f'{n}:{arm}:{key[2]}']
                                audit_models[model_key][arm]=load_model(out/choice['path'],n,cfg,key[2],arm,choice['steps'])
                        batch=learner.tensors({name:np.asarray(value)[None] for name,value in obs.items()})
                        recorded=torch.as_tensor(np.asarray(submitted)[None],dtype=torch.long)
                        full_trace=[];mask_trace=[]
                        with torch.no_grad():
                            audit_models[model_key]['reservation_ppo'].trace(batch,recorded=recorded,audit=full_trace)
                            audit_models[model_key]['mask_only_ppo'].trace(batch,recorded=recorded,audit=mask_trace)
                        for full_step,mask_step in zip(full_trace,mask_trace,strict=True):
                            for field in ('legal','resource','stats'):
                                if not torch.equal(full_step[field],mask_step[field]):raise RuntimeError('prefix/mask trajectory feasibility mismatch')
                            if torch.count_nonzero(mask_step['prefix']):raise RuntimeError('mask-only leaked selected-edge prefix')
                            if torch.count_nonzero(mask_step['candidate_capacity']) or torch.count_nonzero(mask_step['wait_capacity']):
                                raise RuntimeError('mask-only leaked selected-pair capacity')
                            if torch.count_nonzero(mask_step['prefix_residual']):
                                raise RuntimeError('mask-only leaked direct prefix residual')
                            if not torch.allclose(full_step['full_logits']-full_step['base_logits'],
                                                  full_step['prefix_residual'],atol=1e-6,rtol=0):
                                raise RuntimeError('direct residual logit arithmetic')
                            residual_nonzero+=int(bool(torch.count_nonzero(full_step['prefix_residual'])))
                            legal=full_step['legal']
                            full_choice=full_step['full_logits'].masked_fill(~legal,-1e9).argmax(-1)
                            base_choice=full_step['base_logits'].masked_fill(~legal,-1e9).argmax(-1)
                            residual_choice_changes+=int(bool(torch.any(full_choice!=base_choice)))
                            prefix_nonzero+=int(bool(torch.count_nonzero(full_step['prefix'])))
                        prefix_mask_audits+=1
                else:
                    from ht_pdm_fjsp.technician_prefix_activation_v5_env import deadline
                    if deadline(c,state)!=pairs:raise RuntimeError('common tail policy mismatch')
                following,cost,components=transition(c,state,pairs,json.loads(r['events']))
                for k,v in components.items():ep['metrics'][k]+=v
                ep['state']=following;ep['tick']+=1
                if tick==47:ep['policy_cost']=ep['metrics']['objective']
                actual={k:list(v) for k,v in asdict(following).items()}
                if actual!=json.loads(r['following']) or abs(cost-float(r['cost']))>1e-8:raise RuntimeError('scalar transition replay')
                replay_count+=1
        if replay_count!=60*len(rows):raise RuntimeError('physical test replay count')
        if len(recorded_coord)!=48*len(rows) or set(replay_episodes)!=set(lookup):raise RuntimeError('coordination/replay population')
        for key,ep in replay_episodes.items():
            row=lookup[key]
            if ep['tick']!=60 or abs(ep['policy_cost']-row['policy_cost'])>1e-8 or any(abs(v-row[k])>1e-8 for k,v in ep['metrics'].items()):raise RuntimeError('aggregate physical cost/metric replay')
    else: replay_count=0;prefix_mask_audits=0;prefix_nonzero=0;residual_nonzero=0;residual_choice_changes=0
    return dict(status='PASS',training_models=len(audits),training_episodes=len(seen),paired_training_tapes=len(paired),
        development_episodes=len(devrows),test_physical_ticks_replayed=replay_count,choices=len(choices),invalid_assignments=0,
        prefix_mask_trajectory_decisions_audited=prefix_mask_audits,full_nonzero_prefix_stages=prefix_nonzero,
        full_nonzero_residual_stages=residual_nonzero,full_same_state_residual_choice_changes=residual_choice_changes,
        teacher_bank_exposures=cfg['teacher_bank_steps']*len(cfg['train_seeds']),
        per_graph_bundle=dict(shared_teacher_bank_exposures=cfg['teacher_bank_steps'],fine_tuning_per_scale=cfg['env_steps'],
            pretraining_presentations_per_arm=cfg['teacher_bank_steps']*cfg['teacher_epochs'],foundation_ppo_steps=491520),
        reference_models=len(references))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=('smoke','development','confirmation'),required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--teacher-dir');p.add_argument('--development-artifact')
    args=p.parse_args();run(args.profile,args.output_dir,args.teacher_dir,args.development_artifact)

if __name__=='__main__':main()
