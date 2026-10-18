"""Independent C1 confirmation replay, resource audit and crossed inference."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors
from ht_pdm_fjsp.maintenance_dispatch import eligible_edges, role_seed, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, observation, transition
from ht_pdm_fjsp.technician_relational_prefix_policy import ReservationPolicy


VERSION='technician_diverse_maintenance_v1'
ARMS=('reservation_ppo','mask_only_ppo','independent_ppo','scratch_reservation_ppo')
FIELDS=('normalized_cost','availability','normalized_pending','normalized_failures')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def config_from_protocol(family,n):
    assert family in ('skill_bottleneck','long_service') and n in (8,16)
    k=3*n//8 if family=='skill_bottleneck' else n//2
    durations=[]
    for m in range(n):
        row=[]
        for j in range(k):
            if family=='skill_bottleneck':
                row.append(1 if m%3 in (j%3,(j+1)%3) else 0)
            else:
                row.append((2 if j%4==m%4 else 4)
                    if j%4 in (m%4,(m+1)%4) else 0)
        durations.append(tuple(row))
    c=WaitingConfig(n,k,60,4,.45,tuple(durations),
        tuple(tuple(.75 for _ in range(k)) for _ in range(n)),
        max_age=8,preventive_cost=1.,corrective_cost=2.,
        unavailable_cost=6.,failure_cost=15.,waiting_limit=4,waiting_price=12.)
    c.validate();return c


def proposal(c,state,tick):
    occupied={m for m in state.assigned if m>=0}
    if tick<48:
        return [m for m in range(c.machines) if m not in occupied and
            (state.failed[m] or state.ages[m]>=3)]
    return [m for m in range(c.machines) if state.failed[m] and m not in occupied]


def prepare(c,state,tick,pending,since):
    created=0
    if tick>=60:return created
    active={m for m in state.assigned if m>=0}
    for m in proposal(c,state,tick):
        if m not in pending and m not in active:
            pending.add(m);since[m]=tick;created+=1
    return created


def priority(c,state,m):
    return ((100 if state.failed[m] else 0)+12*state.pending_wait[m]+state.ages[m]
        +(10*c.failure_probability if state.ages[m]+1>=c.failure_age else 0))


def heuristic(c,state,pending,since,method):
    idle={j for j,rem in enumerate(state.remaining) if rem==0}
    order=sorted(pending,key=(lambda m:(since[m],m)) if method=='fcfs'
        else (lambda m:(-priority(c,state,m),since[m],m)))
    pairs=[]
    for m in order:
        choices=[j for j in idle if c.service_time[m][j]>0]
        if choices:
            j=min(choices,key=lambda j:(c.service_time[m][j],j))
            pairs.append((m,j));idle.remove(j)
    return tuple(sorted(pairs))


def decode(c,state,pending,arm,submitted):
    if arm in ('fcfs','risk'):
        assert submitted is None
        raise RuntimeError('heuristic dispatch needs request arrival times')
    token=np.asarray(submitted)
    assert token.shape==(c.technicians,) and token.dtype.kind in 'iu'
    assert ((token>=0)&(token<=c.machines)).all()
    mask=np.asarray(eligible_edges(c,state),bool)
    selected=[(int(m)-1,j) for j,m in enumerate(token) if m]
    assert all(m in pending and mask[m,j] for m,j in selected)
    if arm!='independent_ppo':assert len({m for m,_ in selected})==len(selected)
    pairs=[]
    for m in range(c.machines):
        pool=[pair for pair in selected if pair[0]==m]
        if pool:pairs.append(max(pool,key=lambda pair:(
            c.restoration[pair[0]][pair[1]]/c.service_time[pair[0]][pair[1]],-pair[1])))
    pairs=tuple(sorted(pairs));validate_matching(c,state,pairs)
    return pairs


def observation_hash(c,state,tick,pending,since):
    obs=observation(c,state,tick)
    machines=obs['machines'].copy()
    for m in range(c.machines):
        machines[m,2]=(tick-since[m]+1)/12 if m in pending else 0.
    obs['machines']=machines
    edges=obs['edges'].copy();edges[edges[...,0]<.5]=0
    obs['edges']=edges
    obs['pair_mask']=np.asarray(eligible_edges(c,state),bool)&np.asarray(
        [m in pending for m in range(c.machines)],bool)[:,None]
    flat=np.concatenate((machines.ravel(),obs['technicians'].ravel(),
        edges.ravel(),obs['global_features'])).astype(np.float32)
    obs['state']=flat
    return hashlib.sha256(flat.tobytes()).hexdigest(),obs


def liability(c,state,pending):
    total=0.
    for m in pending:
        duration=min(d for d in c.service_time[m] if d)
        total+=(c.corrective_cost if state.failed[m] else c.preventive_cost)
        total+=c.unavailable_cost*duration
        if state.failed[m]:total+=c.waiting_price*duration
    total+=c.unavailable_cost*sum(state.remaining)
    return total


def load_model(root,choices,f,n,arm,t):
    selected=choices[f'{f}:N{n}:{arm}:{t}']
    file=root/selected['path']
    assert digest(file)==selected['sha256']
    payload=torch.load(file,map_location='cpu',weights_only=True)
    internal='reservation_ppo' if arm=='scratch_reservation_ppo' else arm
    assert payload['protocol']==VERSION and payload['family']==f and payload['n']==n
    assert payload['arm']==arm and payload['algorithm']==internal
    assert payload['train_seed']==t and payload['steps']==selected['steps']
    model=ReservationPolicy(internal,payload['hidden'])
    model.load_state_dict(payload['state_dict']);model.eval()
    return model


def independent_inference(rows,cfg,selected_rule,select_ms,end_to_end_ms):
    configs=[tuple(c) for c in cfg['configs']]
    train=cfg['train_seeds'];seeds=cfg['test_seeds']
    lookup={(r['family'],int(r['n']),r['method'],int(r['train_seed']),int(r['seed'])):r for r in rows}
    arrays={}
    for f,n in configs:
        arrays[f,n]={}
        for arm in ARMS:
            arrays[f,n][arm]=np.asarray([[[float(lookup[f,n,arm,t,s][field]) for field in FIELDS]
                for s in seeds] for t in train])
        for method in ('fcfs','risk'):
            panel=np.asarray([[float(lookup[f,n,method,-1,s][field]) for field in FIELDS]
                for s in seeds])
            arrays[f,n][method]=np.repeat(panel[None],len(train),axis=0)
    families=sorted({f for f,_ in configs})
    def estimates(means):
        out={}
        cost=lambda arm,group:sum(means[c][arm][0] for c in group)/len(group)
        out['pooled_gain_full_vs_mask']=1-cost('reservation_ppo',configs)/cost('mask_only_ppo',configs)
        out['pooled_gain_full_vs_heuristic']=1-cost('reservation_ppo',configs)/cost(selected_rule,configs)
        for f in families:
            group=[c for c in configs if c[0]==f]
            out[f'{f}_gain_full_vs_mask']=1-cost('reservation_ppo',group)/cost('mask_only_ppo',group)
            for label,index in (('availability',1),('pending_per_machine',2),('failures_per_machine',3)):
                out[f'{f}_{label}_full_minus_mask']=sum(
                    means[c]['reservation_ppo'][index]-means[c]['mask_only_ppo'][index]
                    for c in group)/len(group)
        return out
    means={c:{a:v.mean((0,1)) for a,v in panel.items()} for c,panel in arrays.items()}
    points=estimates(means)
    samples=200 if cfg['profile']=='smoke' else 40000
    rng=np.random.default_rng(role_seed(cfg['analysis_seed'],VERSION+':independent_confirmation_bootstrap'))
    draws={name:[] for name in points}
    nt,ns=len(train),len(seeds)
    for start in tqdm(range(0,samples,128),desc='Independent C1 bootstrap',unit='batch'):
        batch=min(128,samples-start)
        # Keep each bootstrap draw's bundle weights; do not pool draws.
        ti=rng.integers(nt,size=(batch,nt))
        tw=(ti[:,:,None]==np.arange(nt)).mean(1)
        sampled={}
        for c,panel in arrays.items():
            ei=rng.integers(ns,size=(batch,ns))
            ew=(ei[:,:,None]==np.arange(ns)).mean(1)
            sampled[c]={a:np.einsum('bt,bs,tsd->db',tw,ew,v,optimize=True)
                for a,v in panel.items()}
        for name,value in estimates(sampled).items():draws[name].extend(value.tolist())
    alpha=(.05 if cfg['profile']=='smoke' else .025)/10
    bounds={name:dict(lower=float(np.quantile(draws[name],alpha)),
        upper=float(np.quantile(draws[name],1-alpha))) for name in points}
    guards=dict(pooled_full_mask_gain=points['pooled_gain_full_vs_mask']>=.01 and
        bounds['pooled_gain_full_vs_mask']['lower']>0,
        family_point_noninferiority=all(points[f'{f}_gain_full_vs_mask']>=-.01 for f in families),
        pooled_heuristic_noninferiority=bounds['pooled_gain_full_vs_heuristic']['lower']>=-.01)
    for f in families:
        guards[f'{f}_health']=(bounds[f'{f}_availability_full_minus_mask']['lower']>=-.01 and
            bounds[f'{f}_pending_per_machine_full_minus_mask']['upper']<=.025 and
            bounds[f'{f}_failures_per_machine_full_minus_mask']['upper']<=.25)
    guards['largest_select_latency']=float(np.quantile(select_ms,.95))<=50
    guards['zero_invalid_assignments']=all(float(r['invalid_assignments'])==0 for r in rows)
    guards={k:bool(v) for k,v in guards.items()}
    return dict(points={k:float(v) for k,v in points.items()},bounds=bounds,
        guards=guards,goal_pass=all(guards.values()),draws=samples,
        alpha_per_endpoint=alpha,
        model_select_p95_ms=float(np.quantile(select_ms,.95)),
        end_to_end_p95_ms=float(np.quantile(end_to_end_ms,.95)))


def verify(source,output):
    started=time.monotonic()
    manifest=read_json(source/'manifest.json')
    assert manifest['status']=='COMPLETED' and manifest['audit_status']=='PASS'
    assert manifest['protocol']==VERSION and manifest['test_opened']
    if manifest['profile']=='confirmation':
        prior=manifest['prior_development_gate']
        assert not prior['engineering_gate_bypass']
        assert read_json(prior['path']+'/summary.json')['development_screen_pass']
    for name,expected in tqdm(manifest['source_seal'].items(),desc='C1 source seals'):
        assert digest(source/name)==expected,name
    for name,meta in tqdm(manifest['output_bindings'].items(),desc='C1 output bindings'):
        path=source/name
        assert path.stat().st_size==meta['bytes'] and digest(path)==meta['sha256'],name
    cfg=read_json(source/'resolved_config.json')
    choices=read_json(source/'checkpoint_choices.json')
    summary=read_json(source/'summary.json')
    selected_rule=read_json(source/'frozen_reference_selection.json')['selected_rule']
    assert selected_rule==manifest['prior_development_gate']['selected_rule']
    episodes=read_rows(source/'episodes.csv')
    secondary=read_rows(source/'secondary_episodes.csv')
    configs=[tuple(c) for c in cfg['configs']]
    expected={(f,n,a,t,s) for f,n in configs for a in (*ARMS,'fcfs','risk')
        for t in (cfg['train_seeds'] if a in ARMS else [-1]) for s in cfg['test_seeds']}
    episode_key=lambda r:(r['family'],int(r['n']),r['method'],int(r['train_seed']),int(r['seed']))
    table={episode_key(r):r for r in episodes}
    assert len(episodes)==len(expected) and set(table)==expected
    torch.set_num_threads(1)
    model_cache={};paired={};select_ms=[];end_to_end_ms=[]
    mask_audit_seeds=set(cfg['test_seeds'][:8])
    counts=dict(episodes=0,physical_ticks=0,policy_ticks=0,
        request_checks=0,checkpoint_action_checks=0,matched_mask_stages=0,
        full_trajectory_mask_states=0,mask_trajectory_mask_states=0,
        invalid_assignments=0)
    with ((source/'decisions.csv').open(newline='') as decision_file,
          (source/'coordination.csv').open(newline='') as coordination_file):
        decisions=csv.DictReader(decision_file)
        coordination=csv.DictReader(coordination_file)
        for episode in tqdm(episodes,desc='Independent C1 physical replay',unit='episode'):
            f,n,arm,t,seed=episode_key(episode)
            c=config_from_protocol(f,n)
            physical_seed=role_seed(seed,f'{VERSION}:{f}:N{n}:physical')
            env=WaitingEnv(c);env.reset(physical_seed)
            state,events=env.state,env.events
            payload=json.dumps(dict(seed=physical_seed,state=asdict(state),events=events),sort_keys=True)
            population=hashlib.sha256(payload.encode()).hexdigest()
            assert int(episode['physical_seed'])==physical_seed
            assert episode['population_sha256']==population
            paired_key=f,n,seed
            if paired_key in paired:assert paired[paired_key]==population
            else:paired[paired_key]=population
            pending=set();since={};requests=prepare(c,state,0,pending,since)
            starts=completions=0
            components={k:0. for k in ('objective','maintenance_cost','unavailability_cost',
                'failure_cost','waiting_cost','unavailable_ticks','failures')}
            cutoff=None;terminal_pending=None
            for tick in range(60):
                d=next(decisions)
                assert episode_key(d)==(f,n,arm,t,seed) and int(d['tick'])==tick
                assert d['phase']==('policy' if tick<48 else 'tail')
                assert json.loads(d['state'])==json.loads(json.dumps(asdict(state)))
                assert json.loads(d['events'])==list(events[tick])
                recorded=tuple(tuple(pair) for pair in json.loads(d['action']))
                if tick<48:
                    r=next(coordination)
                    assert episode_key(r)==(f,n,arm,t,seed) and int(r['tick'])==tick
                    obs_hash,obs=observation_hash(c,state,tick,pending,since)
                    assert r['observation_sha256']==obs_hash
                    submitted=json.loads(r['submitted'])
                    if arm in ('fcfs','risk'):
                        assert submitted is None
                        selected=heuristic(c,state,pending,since,arm)
                    else:
                        selected=decode(c,state,pending,arm,submitted)
                    assert selected==recorded
                    if arm=='reservation_ppo' and n==max(nn for _,nn in configs):
                        select_ms.append(float(r['decision_ms']))
                        end_to_end_ms.append(float(r['decision_ms'])+float(r['environment_ms']))
                    if arm in ARMS and seed==cfg['test_seeds'][0]:
                        mk=f,n,arm,t
                        if mk not in model_cache:model_cache[mk]=load_model(source,choices,f,n,arm,t)
                        model=model_cache[mk]
                        assert np.array_equal(model.select(obs),np.asarray(submitted))
                        counts['checkpoint_action_checks']+=1
                    if arm in ('reservation_ppo','mask_only_ppo') and seed in mask_audit_seeds:
                        full_key=f,n,'reservation_ppo',t
                        mask_key=f,n,'mask_only_ppo',t
                        for model_key in (full_key,mask_key):
                            if model_key not in model_cache:
                                model_cache[model_key]=load_model(source,choices,*model_key)
                        batch=tensors({k:np.asarray(v)[None] for k,v in obs.items()})
                        forced=torch.as_tensor(np.asarray(submitted)[None],dtype=torch.long)
                        full_trace=[];mask_trace=[]
                        with torch.no_grad():
                            model_cache[full_key].trace(batch,recorded=forced,audit=full_trace)
                            model_cache[mask_key].trace(batch,recorded=forced,audit=mask_trace)
                        assert len(full_trace)==len(mask_trace)==c.technicians
                        reserved=set()
                        for j,(full,mask) in enumerate(zip(full_trace,mask_trace,strict=True)):
                            feasible=np.r_[True,[bool(obs['pair_mask'][m,j]) and m not in reserved
                                for m in range(n)]]
                            assert np.array_equal(full['legal'].numpy()[0],feasible)
                            for field in ('legal','resource','stats'):
                                assert torch.equal(full[field],mask[field])
                            assert not torch.count_nonzero(mask['prefix'])
                            counts['matched_mask_stages']+=1
                            if submitted[j]:reserved.add(submitted[j]-1)
                        key='full_trajectory_mask_states' if arm=='reservation_ppo' else 'mask_trajectory_mask_states'
                        counts[key]+=1
                    counts['policy_ticks']+=1
                else:
                    selected=heuristic(c,state,pending,since,'fcfs')
                    assert selected==recorded
                validate_matching(c,state,recorded)
                assert all(m in pending for m,_ in recorded)
                active_before={m for m in state.assigned if m>=0}|{m for m,_ in recorded}
                for m,_ in recorded:
                    pending.remove(m);since.pop(m);starts+=1
                following,cost,info=transition(c,state,recorded,events[tick])
                active_after={m for m in following.assigned if m>=0}
                completions+=len(active_before-active_after)
                for m,failed in enumerate(following.failed):
                    if failed and m not in pending and m not in active_after:
                        pending.add(m);since[m]=tick+1;requests+=1
                assert requests==len(pending)+len(active_after)+completions
                assert int(d['pending_after'])==len(pending)
                assert int(d['active_after'])==len(active_after)
                requests+=prepare(c,following,tick+1,pending,since)
                assert requests==len(pending)+len(active_after)+completions
                assert json.loads(d['following'])==json.loads(json.dumps(asdict(following)))
                assert abs(float(d['cost'])-cost)<1e-8
                for name in components:components[name]+=info[name]
                counts['invalid_assignments']+=info['invalid_assignments']
                counts['request_checks']+=1
                if tick==47:
                    cutoff=dict(components);terminal_pending=len(pending)
                state=following;counts['physical_ticks']+=1
            terminal=liability(c,state,pending)
            total=components['objective']+terminal
            assert abs(total-float(episode['extended_cost']))<1e-7
            assert abs(cutoff['objective']-float(episode['policy_cost']))<1e-7
            assert abs(1-cutoff['unavailable_ticks']/(n*48)-float(episode['availability']))<1e-10
            assert abs(cutoff['failures']/n-float(episode['normalized_failures']))<1e-10
            assert terminal_pending==int(episode['terminal_pending'])
            assert requests==int(episode['requests']) and starts==int(episode['starts'])
            assert completions==int(episode['completions'])
            assert len(pending)==int(episode['tail_pending'])
            assert len([m for m in state.assigned if m>=0])==int(episode['active_end'])
            assert abs(terminal-float(episode['terminal_liability']))<1e-7
            assert abs(total-sum(float(episode[name]) for name in
                ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost',
                 'terminal_liability')))<1e-7
            counts['episodes']+=1
        assert next(decisions,None) is None
        assert next(coordination,None) is None
    assert counts['physical_ticks']==60*len(episodes) and counts['invalid_assignments']==0
    expected_mask_states=len(configs)*len(cfg['train_seeds'])*len(mask_audit_seeds)*48
    assert counts['full_trajectory_mask_states']==expected_mask_states
    assert counts['mask_trajectory_mask_states']==expected_mask_states
    assert counts['matched_mask_stages']==2*len(cfg['train_seeds'])*len(mask_audit_seeds)*48*sum(
        config_from_protocol(f,n).technicians for f,n in configs)
    infer=independent_inference(episodes,cfg,selected_rule,select_ms,end_to_end_ms)
    for name,point in infer['points'].items():
        assert abs(point-summary['intervals'][name]['estimate'])<1e-12
    if cfg['profile']=='confirmation':
        assert infer['guards']==summary['guards'],'C1 independent inference gate disagreement'
    expected_secondary={(f,n,a,t,s,condition) for f,n in configs for a in ARMS
        for t in cfg['train_seeds'] for s in cfg['test_seeds']
        for condition in (('final','zero_shot') if a=='reservation_ppo' else ('final',))}
    secondary_key=lambda r:(r['family'],int(r['n']),r['method'],int(r['train_seed']),
                            int(r['seed']),r['condition'])
    assert len(secondary)==len(expected_secondary)
    assert {secondary_key(r) for r in secondary}==expected_secondary
    secondary_report=read_json(source/'secondary_summary.json')
    assert secondary_report['episodes']==len(secondary)
    for row in secondary:
        f,n,a,t,s,condition=secondary_key(row)
        assert row['population_sha256']==paired[f,n,s]
        assert int(float(row['invalid_assignments']))==0
        assert abs(float(row['extended_cost'])/n-float(row['normalized_cost']))<1e-10
    for f,n in configs:
        for a in ARMS:
            for condition in (('final','zero_shot') if a=='reservation_ppo' else ('final',)):
                panel=[r for r in secondary if (r['family'],int(r['n']),r['method'],r['condition'])==
                    (f,n,a,condition)]
                mean=statistics.fmean(float(r['normalized_cost']) for r in panel)
                reported=secondary_report['results'][f'{f}:N{n}'][a][condition]['mean_cost_per_machine']
                assert abs(mean-reported)<1e-12
    training_audit=read_json(source/'training_audit.json')
    initial=read_json(source/'initial_parameter_hashes.json')
    warm=read_json(source/'warm_choices.json')
    assert len(training_audit)==len(configs)*len(ARMS)*len(cfg['train_seeds'])
    for f,n in configs:
        for t in cfg['train_seeds']:
            assert len({initial[f'{f}:N{n}:{a}:{t}'] for a in ARMS})==1
    for t in cfg['train_seeds']:
        assert digest(source/'warm'/str(t)/'bank.npz')==warm[str(t)]['bank']['bank_sha256']
    for item in training_audit:
        assert item['learned_steps']==cfg['env_steps']
        assert item['training_episodes']==cfg['env_steps']//48
        assert item['physical_ticks']==cfg['env_steps']//48*60
        assert item['optimizer_steps']>0 and item['changed']
        if item['arm']=='scratch_reservation_ppo':
            assert item['warm_state_exposures']==item['warm_optimizer_steps']==0
            assert item['bank_sha256'] is None
        else:
            assert item['warm_state_exposures']==cfg['teacher_bank_steps']
            assert item['warm_optimizer_steps']>0 and item['bank_sha256']
    training_keys=set();training_tapes={}
    with (source/'training_episodes.csv').open(newline='') as stream:
        for row in tqdm(csv.DictReader(stream),desc='Independent C1 PPO tape audit',unit='episode'):
            f,n,a,t,w,e=(row['family'],int(row['n']),row['arm'],int(row['train_seed']),
                         int(row['worker']),int(row['episode_index']))
            identity=f,n,t,w,e,a
            assert identity not in training_keys
            training_keys.add(identity)
            key=f,n,t,w,e
            tape=row['physical_seed'],row['population_sha256']
            if key in training_tapes:assert training_tapes[key]==tape
            else:training_tapes[key]=tape
            assert abs(float(row['episode_reward'])*20+float(row['extended_cost']))<1e-5
            assert float(row['invalid_assignments'])==0
    expected_training={(f,n,t,w,e,a) for f,n in configs for t in cfg['train_seeds']
        for w in range(cfg['vector_envs'])
        for e in range(cfg['env_steps']//(48*cfg['vector_envs'])) for a in ARMS}
    assert training_keys==expected_training
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),output/Path(__file__).name)
    report=dict(status='PASS',source_artifact=str(source),
        source_manifest_sha256=digest(source/'manifest.json'),
        verifier_sha256=digest(output/Path(__file__).name),
        completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started,
        counts=counts,secondary_episodes=len(secondary),
        training_episodes=len(training_keys),paired_training_tapes=len(training_tapes),
        selected_rule=selected_rule,independent_inference=infer,
        source_summary_goal_pass=summary['goal_pass'],source_git=manifest['git'],
        invocation=dict(executable=sys.executable,argv=sys.argv,cwd=str(Path.cwd())))
    (output/'verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    report=verify(args.artifact.resolve(),args.output_dir.resolve())
    print(json.dumps(dict(status=report['status'],
        goal_pass=report['independent_inference']['goal_pass'],
        episodes=report['counts']['episodes'])))


if __name__=='__main__':main()
