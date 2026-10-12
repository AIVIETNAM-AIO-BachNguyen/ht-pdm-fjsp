"""Prospective joint-headroom confirmation against restricted economic matching."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import itertools
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_lookahead_boundary as old
from ht_pdm_fjsp import maintenance_restricted_dp as dp
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays, batch_tick, reference_batch
from ht_pdm_fjsp.maintenance_opportunity_cost import Journal, write_csv, write_json
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

VERSION='maintenance_joint_frontier_mac_v1'
ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
CHEAP=('deadline_matching','isolated_h6','isolated_endpoint')
PLANNERS=('joint_rollout','subset_rollout')
METHODS=(*CHEAP,*PLANNERS)
SOURCES=tuple(dict.fromkeys(('maintenance_joint_frontier_mac.py','maintenance_restricted_dp.py',*old.SOURCES)))


def settings(profile):
    if profile not in ('smoke','full'): raise ValueError('unknown profile')
    raw=json.loads(REGISTRY.read_text())['panels'][profile]
    return dict(profile=profile,horizon=48,tail_ticks=12,scenarios=128 if profile=='full' else 8,
        development_seeds=list(range(raw['development_start'],raw['development_start']+raw['development_count'])),
        confirmation_seeds=list(range(raw['confirmation_start'],raw['confirmation_start']+raw['confirmation_count'])),
        analysis_seed=raw['analysis_seed'],forecast_seed=raw['forecast_seed'],
        bootstrap_samples=20000 if profile=='full' else 200,confidence=1-.05/8,
        max_seconds=7200 if profile=='full' else 600,landmarks=list(old.LANDMARKS),
        audit_seeds=2,train_seeds=[],training='NOT_APPLICABLE',candidate=old.CANDIDATE)


def physical_seed(seed): return role_seed(seed,f'{VERSION}:physical')


def bank_seed(cfg,seed,tick,phase='selection'):
    if phase not in ('selection','holdout') or tick not in range(cfg['horizon']): raise ValueError('forecast role')
    return role_seed(cfg['forecast_seed'],f'{VERSION}:{phase}:{seed}:{tick}')


def seed_audit(cfg):
    if cfg!=settings(cfg['profile']): raise ValueError('locked settings changed')
    groups=[];derived=[];historical=set();hashes={}
    for p in sorted((ROOT/'configs').glob('*seed_registry.json')):
        if p!=REGISTRY:
            historical |= old.integers(json.loads(p.read_text()));hashes[p.name]=old.screen.sha(p)
    prior=set()
    for module in (old.screen.base,old.screen.previous,old.screen):
        for profile in ('smoke','full'):
            c=module.settings(profile)
            prior.update(role_seed(s,f'{module.VERSION}:physical') for phase in ('screening','confirmation') for s in c[f'{phase}_seeds'])
            prior.add(role_seed(c['analysis_seed'],f'{module.VERSION}:bootstrap'))
    for module in (old.previous,old):
        for profile in ('smoke','full'):
            c=module.settings(profile)
            prior.update(module.physical_seed(s) for s in c['evaluation_seeds'])
            prior.add(role_seed(c['analysis_seed'],f'{module.VERSION}:bootstrap'))
            prior.update(module.forecast_seed(c,s,t) for s,t in itertools.product(c['evaluation_seeds'],range(c['horizon'])))
            prior.update(module.forecast_seed(c,s,t,'holdout') for s,t in itertools.product(c['evaluation_seeds'],c['landmarks']))
    for profile in ('smoke','full'):
        c=settings(profile)
        groups += [set(c['development_seeds']),set(c['confirmation_seeds']),{c['analysis_seed']},{c['forecast_seed']}]
        derived += [physical_seed(s) for s in c['development_seeds']+c['confirmation_seeds']]
        derived += [bank_seed(c,s,t) for s,t in itertools.product(c['confirmation_seeds'],range(48))]
        derived += [bank_seed(c,s,t,'holdout') for s,t in itertools.product(c['confirmation_seeds'],c['landmarks'])]
        derived.append(role_seed(c['analysis_seed'],f'{VERSION}:bootstrap'))
    own=set.union(*groups)
    if any(a&b for a,b in itertools.combinations(groups,2)) or own&(historical|prior): raise ValueError('raw seed overlap')
    if len(set(derived))!=len(derived) or set(derived)&(historical|prior|own): raise ValueError('derived seed overlap')
    return dict(status='PASS',raw_unique=len(own),derived_unique=len(derived),prior_derived_checked=len(prior),historical_registry_hashes=hashes)


def make_config(cfg): return old.make_config(cfg)


def cheap(c,state,tick,method):
    if method not in CHEAP: raise ValueError('cheap method')
    return old.myopic(c,state) if method=='deadline_matching' else dp.select(c,state,c.horizon-tick,method)


def continuation(c,states,tick,base):
    if tick>=48 or base=='deadline_matching': return reference_batch(c,states,c.horizon-tick,'deadline_matching')
    return dp.batch(c,states,c.horizon-tick,base)


def roots(c,state,tick,base,subset=False):
    first=cheap(c,state,tick,base)
    actions=(first,*(a for a in old.screen.action_space(c,state)[2] if a!=first))
    if not subset: return actions
    groups={}
    for a in actions:
        key=tuple(m for m,_ in a);duration=sum(c.service_time[m][j] for m,j in a)
        if key not in groups or duration<groups[key][0]: groups[key]=(duration,a)
    # Baseline must be in both pools for rollout improvement and score checks.
    # If its assignment is not duration-minimal it is retained as an extra root.
    return (first,*(a for _,a in groups.values() if a!=first))


def forecast(c,state,actions,tick,scenarios,seed,base,timeout=None):
    if tick not in range(48) or scenarios<2 or not actions: raise ValueError('forecast arguments')
    for a in actions: validate_matching(c,state,a)
    b=len(actions);h=c.horizon-tick;states=arrays(state,b*scenarios)
    start=np.zeros((b,c.machines,c.technicians),bool)
    for i,a in enumerate(actions):
        for m,j in a: start[i,m,j]=True
    shocks=np.random.default_rng(seed).random((h,scenarios,c.machines))<c.failure_probability
    total=np.zeros(b*scenarios);selected=np.repeat(start,scenarios,axis=0)
    for offset in range(h):
        if timeout: timeout()
        if offset: selected=continuation(c,states,tick+offset,base)
        states,cost=batch_tick(c,states,selected,np.tile(shocks[offset],(b,1)));total+=cost
    samples=total.reshape(b,scenarios)
    if not np.isfinite(samples).all(): raise RuntimeError('nonfinite forecast')
    return dict(samples=samples,means=samples.mean(1),transitions=b*scenarios*h)


def select(c,state,tick,method,base,cfg,seed,timeout=None):
    if method in CHEAP or tick>=48:
        action=cheap(c,state,tick,method if tick<48 else 'deadline_matching')
        return action,dict(planned=False,forecast_transitions=0)
    if method not in PLANNERS: raise ValueError('method')
    pool=roots(c,state,tick,base,method=='subset_rollout')
    if len(pool)==1: return pool[0],dict(planned=False,forecast_transitions=0)
    role=bank_seed(cfg,seed,tick)
    bank=forecast(c,state,pool,tick,cfg['scenarios'],role,base,timeout)
    chosen=old.minimum(bank['means'],tuple(range(len(pool))))
    return pool[chosen],dict(planned=True,candidate_count=len(pool),selection_seed=role,
        selected_mean=float(bank['means'][chosen]),baseline_mean=float(bank['means'][0]),forecast_transitions=bank['transitions'])


def episode(cfg,method,seed,base,phase,journal=None,timeout=None):
    c=make_config(cfg);env=WaitingEnv(c);env.reset(physical_seed(seed))
    counts=defaultdict(float);interior=defaultdict(float);spent=0;latency=0.;saved={};cutoff=None
    for tick in range(c.horizon):
        if timeout: timeout()
        state=env.state
        if phase=='confirmation' and method==base and tick in cfg['landmarks']: saved[tick]=state
        before=time.perf_counter();a,diag=select(c,state,tick,method,base,cfg,seed,timeout);latency+=time.perf_counter()-before
        spent+=diag['forecast_transitions'];validate_matching(c,state,a)
        if a not in old.screen.action_space(c,state)[2]: raise RuntimeError('root admissibility')
        metrics=None
        if tick<48:
            metrics=dict(**old.screen.state_metrics(c,state,a),**old.change_metrics(cheap(c,state,tick,base),a))
            for k,v in metrics.items(): counts[k]+=v
        events=env.events[tick];_,_,_,info=env.step(a)
        if tick<48:
            for k,v in info.items(): interior[k]+=v
        if tick==47: cutoff=dict(terminal_pending=old.screen.base.pending(c,env.state),terminal_failed=sum(env.state.failed))
        if journal:
            journal.add(dict(phase=phase,method=method,seed=seed,physical_seed=physical_seed(seed),tick=tick,
                state=old.compact(asdict(state)),action=old.compact(a),events=old.compact(events),following=old.compact(asdict(env.state)),
                metrics=old.compact(metrics),info=old.compact(info),diagnostic=old.compact(diag)))
    row=dict(phase=phase,method=method,seed=seed,physical_seed=physical_seed(seed),**{k:v/48 for k,v in counts.items()},
        availability=1-interior['unavailable_ticks']/(4*48),technician_utilization=interior['service_ticks']/(2*48),
        **cutoff,tail_pending=old.screen.base.pending(c,env.state),tail_failed=sum(env.state.failed),
        policy_cost=interior['objective'],tail_cost=env.metrics['objective']-interior['objective'],extended_cost=env.metrics['objective'],
        policy_failures=interior['failures'],policy_overdue_ticks=interior['overdue_waiting_ticks'],forecast_transitions=spent,planning_seconds=latency,
        **{f'extended_{k}':env.metrics[k] for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost','invalid_assignments')})
    if abs(sum(row[f'extended_{k}'] for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost'))-row['extended_cost'])>1e-8: raise RuntimeError('cost components')
    return row,saved


def choose_baseline(cfg,rows):
    expected=set(itertools.product(CHEAP,cfg['development_seeds']))
    if len(rows)!=len(expected) or {(r['method'],r['seed']) for r in rows}!=expected: raise RuntimeError('development grid')
    means={m:float(np.mean([r['extended_cost'] for r in rows if r['method']==m])) for m in CHEAP}
    return dict(base=min(CHEAP,key=lambda m:(means[m],m)),development_means=means,confirmation_outcomes_opened=False,phase='development_only')


def landmark(cfg,state,seed,tick,base,root_journal=None,timeout=None):
    c=make_config(cfg);pool=roots(c,state,tick,base);selection=forecast(c,state,pool,tick,cfg['scenarios'],bank_seed(cfg,seed,tick),base,timeout)
    held=forecast(c,state,pool,tick,cfg['scenarios'],bank_seed(cfg,seed,tick,'holdout'),base,timeout)
    subsets=set(roots(c,state,tick,base,True));eligible={m:tuple(i for i,a in enumerate(pool) if m=='joint_rollout' or a in subsets) for m in PLANNERS}
    chosen={m:old.minimum(selection['means'],eligible[m]) for m in PLANNERS}
    row=dict(seed=seed,tick=tick,forecast_transitions=selection['transitions']+held['transitions'])
    for m,i in chosen.items():
        values=held['samples'][0]-held['samples'][i]
        row.update({f'{m}_root':i,f'{m}_changed':int(i!=0),f'{m}_gain':float(values.mean()),f'{m}_mc_se':float(values.std(ddof=1)/np.sqrt(cfg['scenarios']))})
    if root_journal:
        for i,a in enumerate(pool):
            root_journal.add(dict(seed=seed,tick=tick,root=i,action=old.compact(a),selection_mean=float(selection['means'][i]),holdout_mean=float(held['means'][i]),
                **{f'selected_{m}':int(i==chosen[m]) for m in PLANNERS}))
    return row


def summarize(cfg,base,rows,local):
    expected=set(itertools.product(METHODS,cfg['confirmation_seeds']));lookup={(r['method'],r['seed']):r for r in rows}
    if len(rows)!=len(expected) or set(lookup)!=expected: raise RuntimeError('confirmation grid')
    if len(local)!=len(cfg['confirmation_seeds'])*len(cfg['landmarks']) or {(r['seed'],r['tick']) for r in local}!=set(itertools.product(cfg['confirmation_seeds'],cfg['landmarks'])): raise RuntimeError('landmark grid')
    fields=('extended_cost','availability','terminal_pending','policy_cost')
    values=np.asarray([[[lookup[m,s][k] for k in fields] for m in METHODS] for s in cfg['confirmation_seeds']])
    if not np.isfinite(values).all() or (values[:,:,0]<=0).any() or (values[:,:,3]<=0).any(): raise RuntimeError('finite positive cost denominators required')
    n=len(values);draws=np.random.default_rng(role_seed(cfg['analysis_seed'],f'{VERSION}:bootstrap')).integers(n,size=(cfg['bootstrap_samples'],n))
    boot=np.concatenate([values[d].mean(1) for d in np.array_split(draws,max(1,len(draws)//200))]);point=values.mean(0)
    a=(1-cfg['confidence'])/2;j=METHODS.index('joint_rollout');b=METHODS.index(base)
    result={}
    def add(key,estimate,samples): result[key]=dict(estimate=float(estimate),ci=np.quantile(samples,[a,1-a]).tolist(),confidence=cfg['confidence'])
    for method in CHEAP:
        i=METHODS.index(method);add(f'cost_vs_{method}',1-point[j,0]/point[i,0],1-boot[:,j,0]/boot[:,i,0])
    add('availability_difference',point[j,1]-point[b,1],boot[:,j,1]-boot[:,b,1])
    add('pending_difference',point[j,2]-point[b,2],boot[:,j,2]-boot[:,b,2])
    add('absolute_availability',point[j,1],boot[:,j,1]);add('absolute_pending',point[j,2],boot[:,j,2])
    add('interior_cost_gain',1-point[j,3]/point[b,3],1-boot[:,j,3]/boot[:,b,3])
    assert len(result)==8
    gates={f'cost_vs_{m}':result[f'cost_vs_{m}']['ci'][0]>=.01 for m in CHEAP}
    gates.update(availability_difference=result['availability_difference']['ci'][0]>=-.01,pending_difference=result['pending_difference']['ci'][1]<=.10,
        absolute_availability=result['absolute_availability']['ci'][0]>=.25,absolute_pending=result['absolute_pending']['ci'][1]<=1,
        interior_cost_gain=result['interior_cost_gain']['ci'][0]>0)
    means=[]
    for m in METHODS:
        group=[r for r in rows if r['method']==m]
        means.append(dict(method=m,**{k:float(np.mean([r[k] for r in group])) for k in group[0] if k not in ('phase','method','seed','physical_seed')}))
    i=METHODS.index('subset_rollout');subset=1-boot[:,i,0]/boot[:,b,0]
    mechanisms={}
    for method in PLANNERS:
        gains=np.asarray([np.mean([r[f'{method}_gain'] for r in local if r['seed']==s]) for s in cfg['confirmation_seeds']])
        mechanisms[method]=dict(mean_holdout_gain=float(gains.mean()),ci95=np.quantile(gains[draws].mean(1),[.025,.975]).tolist(),changed=sum(r[f'{method}_changed'] for r in local),landmarks=len(local))
    return dict(candidate=cfg['candidate'],base=base,comparisons=result,gates=gates,
        scientific_status=('JOINT_SETTING_CONFIRMED' if all(gates.values()) else 'NOT_CONFIRMED') if cfg['profile']=='full' else 'ENGINEERING_ONLY',
        means=means,subset_ablation=dict(estimate=float(1-point[i,0]/point[b,0]),ci95=np.quantile(subset,[.025,.975]).tolist(),scope='exploratory'),mechanism=mechanisms)


def read_csv(path):
    with Path(path).open() as f: return list(csv.DictReader(f))


def replay(path,cfg,base,rows,local,timeout=None,root_rows=None):
    c=make_config(cfg);envs={};count=0;paired={};totals=defaultdict(int);forecasts=0
    measured=defaultdict(lambda:defaultdict(float));metrics=defaultdict(lambda:defaultdict(float));spent=defaultdict(int)
    audit=set(cfg['confirmation_seeds'][:cfg['audit_seeds']]);lookup={(r['phase'],r['method'],r['seed']):r for r in rows}
    if any(r['physical_seed']!=physical_seed(r['seed']) for r in rows): raise RuntimeError('episode physical seed provenance')
    for record in tqdm(read_csv(path),desc='Replay all physical decisions',unit='tick'):
        if timeout: timeout()
        key=record['phase'],record['method'],int(record['seed']);tick=int(record['tick']);seed=key[-1]
        phase_base='deadline_matching' if key[0]=='development' else base
        if int(record['physical_seed'])!=physical_seed(seed): raise RuntimeError('decision physical seed provenance')
        if tick==0:
            if key in envs: raise RuntimeError('duplicate replay trajectory')
            e=WaitingEnv(c);e.reset(physical_seed(seed));envs[key]=e
            if paired.setdefault(seed,(e.state,e.events))!=(e.state,e.events): raise RuntimeError('paired physical tapes')
        e=envs[key];state=old.screen.base.unpack_state(json.loads(record['state']))
        if e.time!=tick or e.state!=state or tuple(json.loads(record['events']))!=e.events[tick]: raise RuntimeError('state/event replay')
        action=tuple(map(tuple,json.loads(record['action'])));diag=json.loads(record['diagnostic']);metric=json.loads(record['metrics'])
        if key[1] in CHEAP or tick>=48 or seed in audit:
            expected,expected_diag=select(c,state,tick,key[1],phase_base,cfg,seed,timeout)
            if expected!=action: raise RuntimeError('planner/action replay')
            old.close_enough(diag,expected_diag);forecasts+=expected_diag['forecast_transitions']
        else:
            pool=roots(c,state,tick,base,key[1]=='subset_rollout')
            if action not in pool: raise RuntimeError('planner restriction')
            if len(pool)==1:
                old.close_enough(diag,dict(planned=False,forecast_transitions=0))
            if len(pool)>1:
                if not diag['planned'] or diag['forecast_transitions']!=len(pool)*cfg['scenarios']*(60-tick) or diag['candidate_count']!=len(pool) or diag['selection_seed']!=bank_seed(cfg,seed,tick) or not np.isfinite([diag['selected_mean'],diag['baseline_mean']]).all() or diag['selected_mean']>diag['baseline_mean']+1e-10: raise RuntimeError('planner score contract')
        if action not in old.screen.action_space(c,state)[2]: raise RuntimeError('admissible action replay')
        expected_metrics=dict(**old.screen.state_metrics(c,state,action),**old.change_metrics(cheap(c,state,tick,phase_base),action)) if tick<48 else None
        old.close_enough(metric,expected_metrics)
        _,_,_,info=e.step(action);old.close_enough(info,json.loads(record['info']));old.close_enough(asdict(e.state),json.loads(record['following']))
        spent[key]+=diag['forecast_transitions']
        if tick<48:
            for k,v in info.items(): measured[key][k]+=v
            for k,v in expected_metrics.items(): metrics[key][k]+=v
        totals[key]+=1;count+=1
        if tick==47:
            for field,value in (('terminal_pending',old.screen.base.pending(c,e.state)),('terminal_failed',sum(e.state.failed)),('policy_cost',e.metrics['objective'])):
                old.close_enough(lookup[key][field],value)
        if tick==59:
            check=dict(**{k:v/48 for k,v in metrics[key].items()},availability=1-measured[key]['unavailable_ticks']/192,
                technician_utilization=measured[key]['service_ticks']/96,tail_pending=old.screen.base.pending(c,e.state),tail_failed=sum(e.state.failed),
                policy_cost=measured[key]['objective'],tail_cost=e.metrics['objective']-measured[key]['objective'],extended_cost=e.metrics['objective'],
                policy_failures=measured[key]['failures'],policy_overdue_ticks=measured[key]['overdue_waiting_ticks'],forecast_transitions=spent[key],
                **{f'extended_{k}':e.metrics[k] for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost','invalid_assignments')})
            for k,v in check.items(): old.close_enough(lookup[key][k],v)
    if set(envs)!=set(lookup) or any(v!=60 for v in totals.values()): raise RuntimeError('replay population')
    # Regenerate independent diagnostic banks on the same predeclared audit seeds.
    states={}
    for r in read_csv(path):
        if r['phase']=='confirmation' and r['method']==base and int(r['tick']) in cfg['landmarks']:
            states[int(r['seed']),int(r['tick'])]=old.screen.base.unpack_state(json.loads(r['state']))
    if root_rows is not None:
        validate_root_rows(c,cfg,base,states,local,root_rows)
    for r in local:
        if r['seed'] in audit:
            collector=[]
            class Sink:
                def add(self,row): collector.append(row)
            expected=landmark(cfg,states[r['seed'],r['tick']],r['seed'],r['tick'],base,Sink(),timeout)
            if root_rows is not None:
                for rr in collector:
                    record=next(x for x in root_rows if int(x['seed'])==rr['seed'] and int(x['tick'])==rr['tick'] and int(x['root'])==rr['root'])
                    for k,v in rr.items(): old.close_enough(v,record[k] if k=='action' else float(record[k]))
            old.close_enough(r,expected);forecasts+=expected['forecast_transitions']
    return dict(status='PASS',physical_transitions=count,episodes=len(envs),forecast_audit_seeds=sorted(audit),replayed_forecast_transitions=forecasts,invalid_assignments=0)


def validate_root_rows(c,cfg,base,states,local,rows):
    lookup={}
    for r in rows:
        key=int(r['seed']),int(r['tick']),int(r['root'])
        if key in lookup: raise RuntimeError('duplicate diagnostic root')
        lookup[key]=r
    expected=set()
    for row in local:
        seed,tick=row['seed'],row['tick'];state=states[seed,tick];pool=roots(c,state,tick,base)
        expected.update((seed,tick,i) for i in range(len(pool)))
        group=[]
        for i,action in enumerate(pool):
            record=lookup.get((seed,tick,i))
            if record is None or record['action']!=old.compact(action): raise RuntimeError('diagnostic root identity/grid')
            if not np.isfinite([float(record['selection_mean']),float(record['holdout_mean'])]).all(): raise RuntimeError('diagnostic finite scores')
            group.append(record)
        means=np.asarray([float(r['selection_mean']) for r in group]);subsets=set(roots(c,state,tick,base,True))
        for method in PLANNERS:
            eligible=tuple(i for i,a in enumerate(pool) if method=='joint_rollout' or a in subsets)
            selected=old.minimum(means,eligible)
            if row[f'{method}_root']!=selected: raise RuntimeError('diagnostic selected root')
            for i,record in enumerate(group):
                if int(record[f'selected_{method}'])!=int(i==selected): raise RuntimeError('diagnostic root flag')
            old.close_enough(row[f'{method}_gain'],float(group[0]['holdout_mean'])-float(group[selected]['holdout_mean']))
        if row['forecast_transitions']!=2*len(pool)*cfg['scenarios']*(60-tick): raise RuntimeError('diagnostic forecast count')
    if set(lookup)!=expected: raise RuntimeError('extra diagnostic roots')


def run(profile,output_dir,device='cpu'):
    if device!='cpu': raise ValueError('CPU only')
    cfg=settings(profile);seeds=seed_audit(cfg);git=old.screen.git_metadata();out=Path(output_dir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())): raise FileExistsError('refusing overwrite')
    if profile=='full' and git['dirty']: raise RuntimeError('full run requires clean committed source')
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    def timeout():
        if time.monotonic()-started>cfg['max_seconds']: raise TimeoutError('registered hard timeout')
    m=dict(protocol=VERSION,status='RUNNING',git=git,profile=profile,started_utc=datetime.now(timezone.utc).isoformat(),
        confirmation_outcomes_opened=False,runtime=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),device=device))
    write_json(out/'manifest.json',m)
    try:
        c=make_config(cfg);write_json(out/'resolved_config.json',dict(settings=cfg,config=asdict(c)))
        shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'seed_registry.json');snap=out/'source_snapshot';snap.mkdir()
        for name in SOURCES: shutil.copy2(Path(__file__).parent/name,snap/name)
        for module in (old,old.previous,old.screen,old.screen.previous,old.screen.base): shutil.copy2(module.SEED_PATH,snap/module.SEED_PATH.name)
        seals={str(p.relative_to(out)):old.screen.sha(p) for p in (out/'protocol.md',out/'seed_registry.json',*snap.iterdir())}
        m.update(source_seal=seals);write_json(out/'manifest.json',m)
        tables=dp.tables(c) # Build the deterministic reference once before evaluation.
        m['dp_table_fingerprints']=tables.fingerprints();write_json(out/'manifest.json',m)
        journal=Journal(out/'decisions.csv');dev=[];rows=[];saved={}
        try:
            for seed,method in tqdm(list(itertools.product(cfg['development_seeds'],CHEAP)),desc='Development cheap frontier',unit='episode'):
                row,_=episode(cfg,method,seed,'deadline_matching','development',journal,timeout);dev.append(row)
            write_csv(out/'development_episodes.csv',dev);selection=choose_baseline(cfg,dev);write_json(out/'selection.json',selection)
            seal=old.screen.sha(out/'selection.json');base=selection['base'];m.update(selection_sha256=seal,confirmation_outcomes_opened=profile=='full');write_json(out/'manifest.json',m)
            partial=Journal(out/'episodes.partial.csv')
            try:
                for seed,method in tqdm(list(itertools.product(cfg['confirmation_seeds'],METHODS)),desc='Confirm paired joint setting',unit='episode'):
                    row,states=episode(cfg,method,seed,base,'confirmation',journal,timeout);rows.append(row);partial.add(row)
                    saved.update({(seed,tick):state for tick,state in states.items()})
            finally: partial.close()
        finally: journal.close()
        write_csv(out/'episodes.csv',rows);locals_=[];root_journal=Journal(out/'landmark_roots.csv');local_journal=Journal(out/'landmarks.csv')
        try:
            for (seed,tick),state in tqdm(saved.items(),desc='Independent-bank joint diagnostics',unit='state'):
                row=landmark(cfg,state,seed,tick,base,root_journal,timeout);locals_.append(row);local_journal.add(row)
        finally: root_journal.close();local_journal.close()
        audit=replay(out/'decisions.csv',cfg,base,dev+rows,locals_,timeout,read_csv(out/'landmark_roots.csv'))
        n=len(cfg['confirmation_seeds']);cap=n*21*cfg['scenarios']*(2*sum(60-t for t in range(48))+2*sum(60-t for t in cfg['landmarks']))
        spent=sum(r['forecast_transitions'] for r in rows+locals_)
        expected=dict(development=len(CHEAP)*len(cfg['development_seeds']),confirmation=len(METHODS)*n,physical_transitions=(len(dev)+len(METHODS)*n)*60,landmarks=n*len(cfg['landmarks']))
        actual=dict(development=len(dev),confirmation=len(rows),physical_transitions=audit['physical_transitions'],landmarks=len(locals_))
        if actual!=expected or spent>cap or any(r['extended_invalid_assignments'] for r in dev+rows): raise RuntimeError('counts/feasibility/budget')
        if old.screen.sha(out/'selection.json')!=seal: raise RuntimeError('frozen development selection changed')
        for name,digest in seals.items():
            if old.screen.sha(out/name)!=digest: raise RuntimeError('artifact source seal')
        for name in SOURCES:
            if old.screen.sha(Path(__file__).parent/name)!=seals[f'source_snapshot/{name}']: raise RuntimeError('live source changed')
        if old.screen.sha(PLAN)!=seals['protocol.md'] or old.screen.sha(REGISTRY)!=seals['seed_registry.json']: raise RuntimeError('live protocol changed')
        for module in (old,old.previous,old.screen,old.screen.previous,old.screen.base):
            if old.screen.sha(module.SEED_PATH)!=seals[f'source_snapshot/{module.SEED_PATH.name}']: raise RuntimeError('live inherited registry changed')
        summary=summarize(cfg,base,rows,locals_);write_json(out/'summary.json',summary)
        audit.update(actual_counts=actual,expected_counts=expected,seed_audit=seeds,forecast_transitions=spent,forecast_cap=cap,selection_frozen=True,source_seal_unchanged=True);write_json(out/'audit.json',audit)
        files={str(p.relative_to(out)):dict(sha256=old.screen.sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        timeout();m.update(status='COMPLETED',audit='PASS',elapsed_seconds=time.monotonic()-started,actual_counts=actual,expected_counts=expected,scientific_status=summary['scientific_status'],files=files,artifact_bytes=sum(b['bytes'] for b in files.values()))
        write_json(out/'manifest.json',m);return m
    except Exception as error:
        m.update(status='FAILED',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started);write_json(out/'manifest.json',m);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',choices=('smoke','full'),required=True)
    parser.add_argument('--device',choices=('cpu',),default='cpu');parser.add_argument('--output-dir',type=Path,required=True);a=parser.parse_args()
    m=run(a.profile,a.output_dir,a.device);print(json.dumps({k:m[k] for k in ('status','audit','scientific_status','elapsed_seconds','actual_counts')},indent=2))

if __name__=='__main__': main()
