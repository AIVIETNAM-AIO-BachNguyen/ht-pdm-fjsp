"""Prospective cross-skill search; root assignment isolation and fresh confirmation."""
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
from ht_pdm_fjsp import maintenance_cross_skill_core as core
from ht_pdm_fjsp import maintenance_joint_frontier_mac as f
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_opportunity_cost import Journal, write_csv, write_json
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

VERSION='maintenance_cross_skill_mac_v1'
ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
CHEAP=f.CHEAP
PLANNERS=('assignment_rollout','joint_rollout')
METHODS=(*CHEAP,*PLANNERS)
REFERENCES=('baseline','quality','endpoint_dp')
LANDMARKS=(0,8,16,24,32,40,47)
SOURCES=tuple(dict.fromkeys(('maintenance_cross_skill_mac.py','maintenance_cross_skill_core.py',*f.SOURCES)))


def settings(profile):
    if profile not in ('smoke','full'):raise ValueError('unknown profile')
    raw=json.loads(REGISTRY.read_text())['panels'][profile]
    return dict(profile=profile,horizon=48,tail_ticks=12,scenarios=128 if profile=='full' else 8,
        **{f'{p}_seeds':list(range(raw[f'{p}_start'],raw[f'{p}_start']+raw[f'{p}_count'])) for p in ('development','screening','confirmation')},
        analysis_seed=raw['analysis_seed'],forecast_seed=raw['forecast_seed'],bootstrap_samples=20000 if profile=='full' else 200,
        confidence=1-.05/18,max_seconds=7200 if profile=='full' else 600,audit_seeds=2,landmarks=list(LANDMARKS),training='NOT_APPLICABLE',train_seeds=[])


def physical_seed(seed):return role_seed(seed,f'{VERSION}:physical')


def bank_seed(cfg,phase,seed,tick,held=False):
    if phase not in ('screening','confirmation') or tick not in range(48):raise ValueError('bank phase/tick')
    return role_seed(cfg['forecast_seed'],f'{VERSION}:{phase}:{"holdout" if held else "selection"}:{seed}:{tick}')


def seed_audit(cfg):
    if cfg!=settings(cfg['profile']):raise ValueError('locked settings')
    historical=set();hashes={}
    for path in sorted((ROOT/'configs').glob('*seed_registry.json')):
        if path!=REGISTRY:historical|=f.old.integers(json.loads(path.read_text()));hashes[path.name]=f.old.screen.sha(path)
    groups=[];derived=[]
    for profile in ('smoke','full'):
        c=settings(profile)
        groups.extend(set(c[f'{p}_seeds']) for p in ('development','screening','confirmation'))
        groups.extend(({c['analysis_seed']},{c['forecast_seed']}))
        derived.extend(physical_seed(s) for p in ('development','screening','confirmation') for s in c[f'{p}_seeds'])
        derived.extend(bank_seed(c,p,s,t,held) for p in ('screening','confirmation') for s in c[f'{p}_seeds'] for t in range(48) for held in (False,True))
        derived.append(role_seed(c['analysis_seed'],f'{VERSION}:bootstrap'))
    own=set.union(*groups)
    if any(a&b for a,b in itertools.combinations(groups,2)) or own&historical:raise ValueError('raw seed overlap')
    if len(derived)!=len(set(derived)) or set(derived)&(own|historical):raise ValueError('derived seed overlap')
    return dict(status='PASS',raw_unique=len(own),derived_unique=len(derived),historical_registry_hashes=hashes,
                intentional_pairing='same raw physical and forecast seeds across profiles/methods')


def root_pool(c,state,tick,method,base):
    if method=='assignment_rollout':return core.subset_pool(c,state,tick,base)
    if method=='joint_rollout':return f.roots(c,state,tick,base)
    raise ValueError('planner')


def select(c,state,tick,method,base,cfg,seed,phase,timeout=None):
    if method in CHEAP or tick>=48:
        return f.cheap(c,state,tick,method if tick<48 else 'deadline_matching'),dict(planned=False,forecast_transitions=0)
    pool=root_pool(c,state,tick,method,base)
    if len(pool)==1:return pool[0],dict(planned=False,forecast_transitions=0)
    role=bank_seed(cfg,phase,seed,tick)
    bank=f.forecast(c,state,pool,tick,cfg['scenarios'],role,base,timeout)
    i=f.old.minimum(bank['means'],tuple(range(len(pool))))
    return pool[i],dict(planned=True,candidate_count=len(pool),selected_root=i,selection_seed=role,
        selected_mean=float(bank['means'][i]),baseline_mean=float(bank['means'][0]),forecast_transitions=bank['transitions'])


def episode(cfg,spec,method,seed,base,phase,journal=None,timeout=None):
    c=core.make_config(spec);env=WaitingEnv(c);env.reset(physical_seed(seed));inside=defaultdict(float);changes=defaultdict(float);saved={};spent=0;elapsed=0
    for tick in range(60):
        if timeout:timeout()
        state=env.state
        if phase!='development' and method==base and tick in LANDMARKS:saved[tick]=state
        before=time.perf_counter();action,diag=select(c,state,tick,method,base,cfg,seed,phase,timeout);elapsed+=time.perf_counter()-before
        validate_matching(c,state,action)
        if action not in f.old.screen.action_space(c,state)[2]:raise RuntimeError('root admissibility')
        metric=f.old.change_metrics(f.cheap(c,state,tick,base),action) if tick<48 else None
        events=env.events[tick];_,_,_,info=env.step(action);spent+=diag['forecast_transitions']
        if tick<48:
            for k,v in info.items():inside[k]+=v
            for k,v in metric.items():changes[k]+=v
        if tick==47:pending=f.old.screen.base.pending(c,env.state);failed=sum(env.state.failed)
        if journal:journal.add(dict(phase=phase,candidate=spec['candidate'],method=method,seed=seed,physical_seed=physical_seed(seed),tick=tick,
            state=f.old.compact(asdict(state)),action=f.old.compact(action),events=f.old.compact(events),following=f.old.compact(asdict(env.state)),
            info=f.old.compact(info),metrics=f.old.compact(metric),diagnostic=f.old.compact(diag)))
    row=dict(phase=phase,candidate=spec['candidate'],method=method,seed=seed,physical_seed=physical_seed(seed),
        availability=1-inside['unavailable_ticks']/192,terminal_pending=pending,terminal_failed=failed,
        tail_pending=f.old.screen.base.pending(c,env.state),tail_failed=sum(env.state.failed),policy_cost=inside['objective'],
        extended_cost=env.metrics['objective'],tail_cost=env.metrics['objective']-inside['objective'],policy_failures=inside['failures'],
        policy_overdue_ticks=inside['overdue_waiting_ticks'],forecast_transitions=spent,planning_seconds=elapsed,
        **{k:v/48 for k,v in changes.items()},**{f'extended_{k}':env.metrics[k] for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost','invalid_assignments')})
    if method=='assignment_rollout' and row['subset_change_rate']!=0:raise RuntimeError('assignment root subset changed')
    return row,saved


def baseline_choices(cfg,specs,rows):
    expected=set(itertools.product([s['candidate'] for s in specs],CHEAP,cfg['development_seeds']))
    if len(rows)!=len(expected) or {(r['candidate'],r['method'],r['seed']) for r in rows}!=expected:raise RuntimeError('development grid')
    result={}
    for spec in specs:
        means={m:float(np.mean([r['extended_cost'] for r in rows if r['candidate']==spec['candidate'] and r['method']==m])) for m in CHEAP}
        result[spec['candidate']]=dict(base=min(CHEAP,key=lambda m:(means[m],m)),development_means=means)
    return result


def landmark(cfg,spec,state,seed,tick,base,phase,journal=None,timeout=None):
    c=core.make_config(spec);pool=core.subset_pool(c,state,tick,base);refs=core.reference_indices(c,state,tick,pool)
    bank=f.forecast(c,state,pool,tick,cfg['scenarios'],bank_seed(cfg,phase,seed,tick),base,timeout)
    held=f.forecast(c,state,pool,tick,cfg['scenarios'],bank_seed(cfg,phase,seed,tick,True),base,timeout)
    chosen=f.old.minimum(bank['means'],tuple(range(len(pool))))
    duration=[sum(c.service_time[m][j] for m,j in a) for a in pool]
    restoration=[sum(c.restoration[m][j] for m,j in a) for a in pool]
    row=dict(phase=phase,candidate=spec['candidate'],seed=seed,tick=tick,subset=f.old.compact(tuple(m for m,_ in pool[0])),
        selection_seed=bank_seed(cfg,phase,seed,tick),holdout_seed=bank_seed(cfg,phase,seed,tick,True),
        duration_span=max(duration)-min(duration),restoration_span=max(restoration)-min(restoration),chosen_root=chosen,choice=int(len(pool)>1),changed=int(chosen!=0),
        completion_age_span=core.completion_span(c,state,pool),forecast_transitions=bank['transitions']+held['transitions'],
        **{f'gain_{name}':float(held['means'][i]-held['means'][chosen]) for name,i in refs.items()})
    if journal:
        for i,a in enumerate(pool):journal.add(dict(phase=phase,candidate=spec['candidate'],seed=seed,tick=tick,root=i,action=f.old.compact(a),
            selection_mean=float(bank['means'][i]),holdout_mean=float(held['means'][i]),selected=int(i==chosen),
            **{f'reference_{name}':int(i==index) for name,index in refs.items()}))
    return row


def matrices(rows,local,seeds,candidate):
    rows=[r for r in rows if r['candidate']==candidate];local=[r for r in local if r['candidate']==candidate]
    lookup={(r['method'],r['seed']):r for r in rows}
    if len(rows)!=len(seeds)*5 or set(lookup)!=set(itertools.product(METHODS,seeds)):raise RuntimeError('episode grid')
    l={(r['seed'],r['tick']):r for r in local}
    if len(local)!=len(seeds)*7 or set(l)!=set(itertools.product(seeds,LANDMARKS)):raise RuntimeError('landmark grid')
    values=np.asarray([[[lookup[m,s][k] for k in ('extended_cost','availability','terminal_pending','policy_cost')] for m in METHODS] for s in seeds])
    labels=np.asarray([[np.mean([l[s,t][k] for t in LANDMARKS]) for k in (*[f'gain_{r}' for r in REFERENCES],'choice')] for s in seeds])
    if not np.isfinite(values).all() or not np.isfinite(labels).all() or (values[:,:,0]<=0).any() or (values[:,:,3]<=0).any():raise RuntimeError('finite costs/labels')
    return values,labels


def estimates(values,labels,base):
    b=METHODS.index(base);a=METHODS.index('assignment_rollout');j=METHODS.index('joint_rollout')
    result={f'local_{r}':labels[...,i] for i,r in enumerate(REFERENCES)}
    result['assignment_cost_gain']=1-values[...,a,0]/values[...,b,0]
    result.update({f'joint_cost_vs_{m}':1-values[...,j,0]/values[...,METHODS.index(m),0] for m in CHEAP})
    for name,i in (('assignment',a),('joint',j)):
        result.update({f'{name}_availability_difference':values[...,i,1]-values[...,b,1],f'{name}_pending_difference':values[...,i,2]-values[...,b,2],
                       f'{name}_absolute_availability':values[...,i,1],f'{name}_absolute_pending':values[...,i,2],
                       f'{name}_interior_gain':1-values[...,i,3]/values[...,b,3]})
    result['assignment_choice_fraction']=labels[...,3]
    if len(result)!=18:raise RuntimeError('registered CI family')
    return result


def screen_choice(cfg,specs,choices,rows,local):
    scores=[]
    for spec in specs:
        values,labels=matrices(rows,local,cfg['screening_seeds'],spec['candidate']);point=estimates(values.mean(0),labels.mean(0),choices[spec['candidate']]['base'])
        score=float(min(point['assignment_cost_gain'],*[point[f'joint_cost_vs_{m}'] for m in CHEAP]))
        eligible=spec['selectable'] and point['assignment_cost_gain']>=.015 and all(point[f'joint_cost_vs_{m}']>=.015 for m in CHEAP)
        eligible &= all(point[f'local_{r}']>=.75 for r in REFERENCES) and point['assignment_choice_fraction']>=.15
        eligible &= all(point[f'{p}_availability_difference']>=-.005 and point[f'{p}_pending_difference']<=.05 and point[f'{p}_absolute_availability']>=.4 and point[f'{p}_absolute_pending']<=.5 and point[f'{p}_interior_gain']>0 for p in ('assignment','joint'))
        scores.append(dict(candidate=spec['candidate'],selectable=spec['selectable'],eligible=bool(eligible),score=score,point={k:float(v) for k,v in point.items()}))
    eligible=[s for s in scores if s['eligible']]
    if cfg['profile']=='smoke':eligible=[s for s in scores if s['selectable']]
    winner=min(eligible,key=lambda s:(-s['score'],s['candidate']))['candidate'] if eligible else None
    return dict(winner=winner,scores=scores,status='SELECTED' if winner else 'NO_SCREEN_CANDIDATE',confirmation_outcomes_opened=False)


def summarize(cfg,spec,base,rows,local):
    values,labels=matrices(rows,local,cfg['confirmation_seeds'],spec['candidate']);n=len(values)
    draws=np.random.default_rng(role_seed(cfg['analysis_seed'],f'{VERSION}:bootstrap')).integers(n,size=(cfg['bootstrap_samples'],n))
    weights=np.stack([(draws==i).sum(1) for i in range(n)],axis=1)/n
    boot=np.einsum('bs,smk->bmk',weights,values,optimize=True);bl=weights@labels
    point=estimates(values.mean(0),labels.mean(0),base);samples=estimates(boot,bl,base);alpha=(1-cfg['confidence'])/2
    comparisons={k:dict(estimate=float(v),ci=np.quantile(samples[k],[alpha,1-alpha]).tolist(),confidence=cfg['confidence']) for k,v in point.items()}
    gates={}
    for key,item in comparisons.items():
        low,high=item['ci']
        gates[key]=(low>=.5 if key.startswith('local_') else high<=.10 if key.endswith('pending_difference') else high<=1 if key.endswith('absolute_pending')
                    else low>=-.01 if key.endswith('availability_difference') else low>=.4 if key.endswith('absolute_availability')
                    else low>=.10 if key=='assignment_choice_fraction' else low>0 if key.endswith('interior_gain') else low>=.01)
    means=[dict(method=m,**{k:float(np.mean([r[k] for r in rows if r['method']==m])) for k in rows[0] if k not in ('phase','candidate','method','seed','physical_seed')}) for m in METHODS]
    return dict(candidate=spec['candidate'],base=base,comparisons=comparisons,gates=gates,means=means,
        scientific_status=('CROSS_SKILL_SETTING_CONFIRMED' if all(gates.values()) else 'NOT_CONFIRMED') if cfg['profile']=='full' else 'ENGINEERING_ONLY',
        inference='fixed current state/tick/subset in local intervention; future decisions state-adaptive; full assignment trajectories can diverge')


def records(path):
    with Path(path).open() as stream:yield from csv.DictReader(stream)


def replay(cfg,specs,choices,path,rows,locals_,root_rows,timeout=None):
    specs={s['candidate']:s for s in specs};lookup={(r['phase'],r['candidate'],r['method'],r['seed']):r for r in rows}
    if len(lookup)!=len(rows):raise RuntimeError('duplicate episodes')
    active={};seen=set();counts=0;spent=0;saved={};paired={};forecast_checks=0
    for r in tqdm(records(path),desc='Replay physical episodes',unit='tick'):
        if timeout:timeout()
        phase,name,method,seed=r['phase'],r['candidate'],r['method'],int(r['seed']);tick=int(r['tick']);key=phase,name,method,seed
        if key not in lookup or int(r['physical_seed'])!=physical_seed(seed) or lookup[key]['physical_seed']!=physical_seed(seed):raise RuntimeError('episode/seed provenance')
        c=core.make_config(specs[name]);base='deadline_matching' if phase=='development' else choices[name]['base']
        if tick==0:
            if key in seen:raise RuntimeError('duplicate trajectory')
            seen.add(key);e=WaitingEnv(c);e.reset(physical_seed(seed));active[key]=(e,defaultdict(float),defaultdict(float),0)
            tape=(e.state,e.events)
            if paired.setdefault((name,seed),tape)!=tape:raise RuntimeError('physical pairing')
        if key not in active:raise RuntimeError('missing trajectory')
        e,inside,change,forecast_count=active[key]
        state=f.old.screen.base.unpack_state(json.loads(r['state']));action=tuple(map(tuple,json.loads(r['action'])));diag=json.loads(r['diagnostic'])
        if e.time!=tick or state!=e.state or tuple(json.loads(r['events']))!=e.events[tick]:raise RuntimeError('state/event replay')
        audit_seed=seed in cfg[f'{phase}_seeds'][:cfg['audit_seeds']]
        if method in CHEAP or tick>=48 or audit_seed:
            expected,expected_diag=select(c,state,tick,method,base,cfg,seed,phase,timeout)
            if action!=expected:raise RuntimeError('action/forecast replay')
            f.old.close_enough(diag,expected_diag);forecast_checks+=expected_diag['forecast_transitions']
        else:
            pool=root_pool(c,state,tick,method,base)
            if action not in pool:raise RuntimeError('planner pool')
            if len(pool)==1:f.old.close_enough(diag,dict(planned=False,forecast_transitions=0))
            else:
                expected_count=len(pool)*cfg['scenarios']*(60-tick)
                index=diag['selected_root']
                if type(index) is not int or not 0<=index<len(pool) or not diag['planned'] or diag['forecast_transitions']!=expected_count or diag['candidate_count']!=len(pool) or diag['selection_seed']!=bank_seed(cfg,phase,seed,tick) or pool[index]!=action or not np.isfinite([diag['selected_mean'],diag['baseline_mean']]).all() or diag['selected_mean']>diag['baseline_mean']+1e-10:raise RuntimeError('forecast diagnostic contract')
        if action not in f.old.screen.action_space(c,state)[2]:raise RuntimeError('legal root')
        metric=f.old.change_metrics(f.cheap(c,state,tick,base),action) if tick<48 else None
        f.old.close_enough(metric,json.loads(r['metrics']))
        if phase!='development' and method==base and tick in LANDMARKS:saved[phase,name,seed,tick]=state
        _,_,_,info=e.step(action);f.old.close_enough(info,json.loads(r['info']));f.old.close_enough(asdict(e.state),json.loads(r['following']))
        if tick<48:
            for k,v in info.items():inside[k]+=v
            for k,v in metric.items():change[k]+=v
        if tick==47:
            f.old.close_enough(lookup[key]['terminal_pending'],f.old.screen.base.pending(c,e.state));f.old.close_enough(lookup[key]['terminal_failed'],sum(e.state.failed))
        forecast_count+=diag['forecast_transitions'];active[key]=(e,inside,change,forecast_count);counts+=1;spent+=diag['forecast_transitions']
        if tick==59:
            check=dict(availability=1-inside['unavailable_ticks']/192,tail_pending=f.old.screen.base.pending(c,e.state),tail_failed=sum(e.state.failed),
                policy_cost=inside['objective'],extended_cost=e.metrics['objective'],tail_cost=e.metrics['objective']-inside['objective'],policy_failures=inside['failures'],policy_overdue_ticks=inside['overdue_waiting_ticks'],
                forecast_transitions=forecast_count,**{k:v/48 for k,v in change.items()},**{f'extended_{k}':e.metrics[k] for k in ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost','invalid_assignments')})
            for k,v in check.items():f.old.close_enough(lookup[key][k],v)
            if method=='assignment_rollout' and check['subset_change_rate']!=0:raise RuntimeError('assignment subset isolation')
            del active[key]
    if active or seen!=set(lookup) or counts!=60*len(rows):raise RuntimeError('physical population')
    groups=defaultdict(list)
    for r in root_rows:groups[r['phase'],r['candidate'],int(r['seed']),int(r['tick'])].append(r)
    expected={(r['phase'],r['candidate'],r['seed'],r['tick']) for r in locals_}
    if len(expected)!=len(locals_) or set(groups)!=expected or set(saved)!=expected:raise RuntimeError('landmark population')
    for r in tqdm(locals_,desc='Validate fixed-subset root grids',unit='state'):
        key=r['phase'],r['candidate'],r['seed'],r['tick'];phase,name,seed,tick=key;c=core.make_config(specs[name]);base=choices[name]['base'];state=saved[key]
        pool=core.subset_pool(c,state,tick,base);refs=core.reference_indices(c,state,tick,pool);rr=groups[key]
        ordered={int(x['root']):x for x in rr}
        if len(rr)!=len(pool) or set(ordered)!=set(range(len(pool))):raise RuntimeError('root grid')
        chosen=f.old.minimum(np.asarray([float(ordered[i]['selection_mean']) for i in range(len(pool))]),tuple(range(len(pool))))
        for i,action in enumerate(pool):
            record=ordered[i]
            if record['action']!=f.old.compact(action) or int(record['selected'])!=int(i==chosen) or not np.isfinite([float(record['selection_mean']),float(record['holdout_mean'])]).all():raise RuntimeError('root action/flag/score')
            for reference,index in refs.items():
                if int(record[f'reference_{reference}'])!=int(i==index):raise RuntimeError('reference root flag')
        duration=[sum(c.service_time[m][j] for m,j in a) for a in pool]
        restoration=[sum(c.restoration[m][j] for m,j in a) for a in pool]
        expected_row=dict(phase=phase,candidate=name,seed=seed,tick=tick,subset=f.old.compact(tuple(m for m,_ in pool[0])),
            selection_seed=bank_seed(cfg,phase,seed,tick),holdout_seed=bank_seed(cfg,phase,seed,tick,True),duration_span=max(duration)-min(duration),restoration_span=max(restoration)-min(restoration),
            chosen_root=chosen,choice=int(len(pool)>1),changed=int(chosen!=0),completion_age_span=core.completion_span(c,state,pool),forecast_transitions=2*len(pool)*cfg['scenarios']*(60-tick),
            **{f'gain_{ref}':float(ordered[index]['holdout_mean'])-float(ordered[chosen]['holdout_mean']) for ref,index in refs.items()})
        f.old.close_enough(r,expected_row)
        if name=='exchangeable' and any(abs(r[f'gain_{ref}'])>1e-10 for ref in REFERENCES):raise RuntimeError('exchangeability null')
        if seed in cfg[f'{phase}_seeds'][:cfg['audit_seeds']]:
            class Sink:
                def __init__(self):self.rows=[]
                def add(self,row):self.rows.append(row)
            sink=Sink();rebuild=landmark(cfg,specs[name],state,seed,tick,base,phase,sink,timeout);f.old.close_enough(r,rebuild)
            for row in sink.rows:
                record=ordered[row['root']]
                for k,v in row.items():f.old.close_enough(v,record[k] if isinstance(v,str) else float(record[k]))
            forecast_checks+=rebuild['forecast_transitions']
    return dict(status='PASS',episodes=len(rows),physical_transitions=counts,landmarks=len(locals_),forecast_transitions=spent+sum(r['forecast_transitions'] for r in locals_),replayed_forecast_transitions=forecast_checks,
        replay_scope='all physical ticks and root grids; regenerated planner/banks first two raw seeds per stage/config',invalid_assignments=0)


def run(profile,output_dir,device='cpu'):
    if device!='cpu':raise ValueError('CPU only')
    cfg=settings(profile);seed_info=seed_audit(cfg);git=f.old.screen.git_metadata();out=Path(output_dir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):raise FileExistsError('refusing overwrite')
    if profile=='full' and git['dirty']:raise RuntimeError('clean committed source required')
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    def timeout():
        if time.monotonic()-started>cfg['max_seconds']:raise TimeoutError('registered hard timeout')
    manifest=dict(protocol=VERSION,status='RUNNING',git=git,profile=profile,confirmation_outcomes_opened=False,started_utc=datetime.now(timezone.utc).isoformat(),runtime=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,device=device))
    write_json(out/'manifest.json',manifest)
    try:
        specs=core.specs(profile);write_json(out/'candidates.json',specs);write_json(out/'resolved_config.json',cfg)
        shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'registry.json');snap=out/'source_snapshot';snap.mkdir()
        for name in SOURCES:shutil.copy2(Path(__file__).parent/name,snap/name)
        rs=out/'registry_snapshot';rs.mkdir()
        for name in seed_info['historical_registry_hashes']:shutil.copy2(ROOT/'configs'/name,rs/name)
        seals={str(p.relative_to(out)):f.old.screen.sha(p) for p in (out/'protocol.md',out/'registry.json',*snap.iterdir(),*rs.iterdir())};manifest['source_seal']=seals;write_json(out/'manifest.json',manifest)
        for spec in specs:f.dp.tables(core.make_config(spec))
        journal=Journal(out/'decisions.csv');dev=[];screen=[];locals_=[];roots=[];rows=[];confirm_local=[]
        try:
            for spec,seed,method in tqdm(list(itertools.product(specs,cfg['development_seeds'],CHEAP)),desc='Development baseline selection',unit='episode'):
                row,_=episode(cfg,spec,method,seed,'deadline_matching','development',journal,timeout);dev.append(row)
            write_csv(out/'development_episodes.csv',dev);choices=baseline_choices(cfg,specs,dev);write_json(out/'baseline_choices.json',choices);choice_seal=f.old.screen.sha(out/'baseline_choices.json')
            root_journal=Journal(out/'screening_roots.csv');local_journal=Journal(out/'screening_landmarks.csv')
            try:
                for spec,seed,method in tqdm(list(itertools.product(specs,cfg['screening_seeds'],METHODS)),desc='Screen cross-skill family',unit='episode'):
                    base=choices[spec['candidate']]['base'];row,states=episode(cfg,spec,method,seed,base,'screening',journal,timeout);screen.append(row)
                    for tick,state in states.items():
                        loc=landmark(cfg,spec,state,seed,tick,base,'screening',root_journal,timeout);locals_.append(loc);local_journal.add(loc)
            finally:root_journal.close();local_journal.close()
            write_csv(out/'screening_episodes.csv',screen);selection=screen_choice(cfg,specs,choices,screen,locals_);write_json(out/'selection.json',selection);seal=f.old.screen.sha(out/'selection.json')
            manifest.update(selection_sha256=seal,baseline_choices_sha256=choice_seal,winner=selection['winner'],stage='screened');write_json(out/'manifest.json',manifest)
            if selection['winner']:
                winner=next(s for s in specs if s['candidate']==selection['winner']);base=choices[winner['candidate']]['base'];manifest.update(confirmation_outcomes_opened=profile=='full',stage='confirmation');write_json(out/'manifest.json',manifest)
                partial=Journal(out/'episodes.partial.csv');root_journal=Journal(out/'landmark_roots.csv');local_journal=Journal(out/'landmarks.csv')
                try:
                    for seed,method in tqdm(list(itertools.product(cfg['confirmation_seeds'],METHODS)),desc='Confirm sealed winner',unit='episode'):
                        row,states=episode(cfg,winner,method,seed,base,'confirmation',journal,timeout);rows.append(row);partial.add(row)
                        for tick,state in states.items():
                            loc=landmark(cfg,winner,state,seed,tick,base,'confirmation',root_journal,timeout);confirm_local.append(loc);local_journal.add(loc)
                finally:partial.close();root_journal.close();local_journal.close()
                write_csv(out/'episodes.csv',rows)
        finally:journal.close()
        all_roots=list(records(out/'screening_roots.csv'))+(list(records(out/'landmark_roots.csv')) if rows else [])
        audit=replay(cfg,specs,choices,out/'decisions.csv',dev+screen+rows,locals_+confirm_local,all_roots,timeout)
        if baseline_choices(cfg,specs,dev)!=choices or screen_choice(cfg,specs,choices,screen,locals_)!=selection:raise RuntimeError('selection reconstruction')
        if f.old.screen.sha(out/'selection.json')!=seal or f.old.screen.sha(out/'baseline_choices.json')!=choice_seal:raise RuntimeError('frozen choices')
        expected=dict(development=len(specs)*len(cfg['development_seeds'])*3,screening=len(specs)*len(cfg['screening_seeds'])*5,
                      confirmation=len(cfg['confirmation_seeds'])*5 if rows else 0,landmarks=(len(specs)*len(cfg['screening_seeds'])+(len(cfg['confirmation_seeds']) if rows else 0))*7)
        actual=dict(development=len(dev),screening=len(screen),confirmation=len(rows),landmarks=len(locals_)+len(confirm_local))
        # Registered loose cap includes every root at every planner/diagnostic tick.
        phase_seeds=len(specs)*len(cfg['screening_seeds'])+(len(cfg['confirmation_seeds']) if rows else 0)
        cap=phase_seeds*21*cfg['scenarios']*(2*sum(60-t for t in range(48))+2*sum(60-t for t in LANDMARKS))
        if actual!=expected or audit['forecast_transitions']>cap or any(r['extended_invalid_assignments'] for r in dev+screen+rows):raise RuntimeError('counts/feasibility/budget')
        for name,digest in seals.items():
            if f.old.screen.sha(out/name)!=digest:raise RuntimeError('source seal')
        for name in SOURCES:
            if f.old.screen.sha(Path(__file__).parent/name)!=seals[f'source_snapshot/{name}']:raise RuntimeError('live source')
        for name,digest in seed_info['historical_registry_hashes'].items():
            if f.old.screen.sha(ROOT/'configs'/name)!=digest:raise RuntimeError('registry drift')
        if f.old.screen.sha(PLAN)!=seals['protocol.md'] or f.old.screen.sha(REGISTRY)!=seals['registry.json']:raise RuntimeError('live protocol')
        summary=summarize(cfg,winner,base,rows,confirm_local) if rows else dict(scientific_status='NO_SCREEN_CANDIDATE' if profile=='full' else 'ENGINEERING_ONLY')
        summary['screening']=selection;write_json(out/'summary.json',summary)
        audit.update(expected_counts=expected,actual_counts=actual,forecast_cap=cap,seed_audit=seed_info,selection_frozen=True,source_seal_unchanged=True);write_json(out/'audit.json',audit)
        files={str(p.relative_to(out)):dict(sha256=f.old.screen.sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        timeout();manifest.update(status='COMPLETED',audit='PASS',scientific_status=summary['scientific_status'],elapsed_seconds=time.monotonic()-started,actual_counts=actual,files=files,artifact_bytes=sum(b['bytes'] for b in files.values()));write_json(out/'manifest.json',manifest);return manifest
    except Exception as error:
        manifest.update(status='FAILED',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started);write_json(out/'manifest.json',manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=('smoke','full'),required=True);p.add_argument('--device',choices=('cpu',),default='cpu');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();m=run(a.profile,a.output_dir,a.device)
    print(json.dumps({k:m[k] for k in ('status','audit','scientific_status','actual_counts','elapsed_seconds')},indent=2))

if __name__=='__main__':main()
