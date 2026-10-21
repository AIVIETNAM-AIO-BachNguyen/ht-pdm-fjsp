"""Independent V19 source/Q/selected-episode and summary audit."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from itertools import permutations
from pathlib import Path
import random
import signal

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import CASES,HORIZON,FrameworkEnv,catalogue,configs,vector_observation,evaluate_episode
from ht_pdm_fjsp.maintenance_framework_v19_policy import TemporalMatchingActor,matching_profiles
from ht_pdm_fjsp.maintenance_framework_v14_campaign import V14Controller,load_checkpoint as load_v14_checkpoint
from ht_pdm_fjsp.maintenance_request_policies import request_action,risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings,state_from_dict,submit_requests,transition

ROOT=Path(__file__).resolve().parent.parent
VERSION='maintenance_framework_v19_temporal_capacity_dev_v1'
ACTIONS=catalogue()
ARMS=('temporal_deficit','snapshot_deficit')
V7=ROOT/'research_result/framework_v7_full_20261020T012531Z'
V14=ROOT/'research_result/framework_v14_dev_20261020T185303Z'
V18=ROOT/'research_result/framework_v18_dev_20261021T050413Z'
V14_ARMS=('competition','count_calibrated')
V18_ARMS=('bounded_disagreement','own_rank')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))

def close(a,b,label):
    if not np.isclose(float(a),float(b),atol=1e-7,rtol=0):
        raise AssertionError(f'{label}: {a} != {b}')

def actor_at(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    actor=TemporalMatchingActor(payload['metadata']['arm'])
    actor.load_state_dict(payload['state_dict']);actor.eval()
    return actor

def current(cfg,state,tick):
    subs=request_action(cfg,state,tick,'threshold')
    admitted=submit_requests(cfg,state,tick,subs)
    legal=sorted(ACTIONS.index(a) for a in legal_matchings(cfg,admitted))
    obs=vector_observation(cfg,admitted,tick)
    return subs,admitted,legal,obs

def actor_choice(actor,obs,legal,uniform=None):
    with torch.no_grad():
        logits=actor(torch.tensor(obs).reshape(1,56))[0]
        masked=torch.full_like(logits,-1e9)
        masked[legal]=logits[legal]
        probabilities=torch.softmax(masked,dim=0).numpy()
    if uniform is None:return int(np.argmax(probabilities))
    cumulative=0.
    for index in legal:
        cumulative+=float(probabilities[index])
        if uniform<cumulative:return index
    return legal[-1]

def independent_profile(obs,matching,tau):
    """Brute-force independent present-state matching deficit at one offset."""
    machine=obs[2:34].reshape(4,8);worker=obs[34:40].reshape(2,3)
    edges=obs[40:56].reshape(4,2,2)
    selected={m for m,_ in matching}
    demand=[m for m in range(4) if m not in selected and machine[m,5]<.5 and
            (machine[m,2]>.5 or round(float(machine[m,0])*8)+tau>=3)]
    free=[]
    for j in range(2):
        assigned=next((round(float(edges[m,j,1])*12) for m,k in matching if k==j),0)
        if max(round(float(worker[j,1])*12),assigned)<=tau:free.append(j)
    matched=0
    for size in range(1,min(len(demand),len(free))+1):
        if any(all(edges[m,j,0]>.5 for m,j in zip(choice,workers))
               for choice in permutations(demand,size)
               for workers in permutations(free,size)):
            matched=size
    return len(demand)/4,(len(demand)-matched)/4


def verify_profile(obs,legal,mode):
    actual=matching_profiles(obs.reshape(1,56),mode)[0]
    for action in legal:
        for tau in range(1,10):
            d,l=independent_profile(obs,ACTIONS[action],tau)
            close(actual[action,tau-1],d,'future demand profile')
            close(actual[action,9+tau-1],l if mode=='temporal_deficit' else
                  independent_profile(obs,ACTIONS[action],1)[1], 'matching deficit profile')


def physical_seed(train,iteration,case,index):
    return role_seed(train,f'{VERSION}:train:{iteration}:{case}:{index}')%(2**31)

def source_replay(row,actor,episodes_per_case):
    case=row['case'];seed=int(row['train_seed']);iteration=int(row['iteration']);tick=int(row['tick'])
    physical=int(row['physical_seed'])
    matches=[i for i in range(episodes_per_case) if physical_seed(seed,iteration,case,i)==physical]
    assert len(matches)==1
    index=matches[0];env=FrameworkEnv(configs()[case]);env.reset(seed=physical)
    rng=random.Random(role_seed(seed,f'{VERSION}:source_policy:{iteration}:{case}:{index}'))
    for t in range(tick+1):
        _,_,legal,obs=current(env.config,env.inner.state,t)
        choice=actor_choice(actor,obs,legal,rng.random())
        if t==tick:
            snapshot=json.loads(json.dumps(asdict(env.inner.state)))
            assert snapshot==json.loads(row['original_state'])
            assert choice==int(row['source_action'])
            assert legal==json.loads(row['legal'])
            rank=f'{VERSION}:rank:{seed}:{iteration}:{case}:{physical}:{tick}'
            assert hashlib.sha256(rank.encode()).hexdigest()==row['rank']
        env.step(choice)
    return env.inner.state

def root_values(row,actor,forecast_anchor,scenarios=4):
    case=row['case'];cfg=configs()[case];origin=int(row['tick'])
    seed=int(row['train_seed']);iteration=int(row['iteration']);physical=int(row['physical_seed'])
    role=f'{VERSION}:bank:{seed}:{iteration}:{case}:{physical}:{origin}'
    bank_seed=role_seed(forecast_anchor,role)
    assert int(row['bank_seed'])==bank_seed
    rng=random.Random(bank_seed);bank=[]
    for _ in range(scenarios):
        failures=[[rng.random()<cfg.failure_probability for m in range(cfg.machines)]
                  for t in range(origin,cfg.horizon)]
        choices=[rng.random() for t in range(origin,HORIZON)]
        bank.append((failures,choices))
    state0=state_from_dict(json.loads(row['original_state']))
    legal=json.loads(row['legal']);own={};transitions=0
    for root in legal:
        costs=[]
        for events,uniforms in bank:
            state=state0;cost_sum=0.
            for tick in range(origin,cfg.horizon):
                subs,admitted,available,obs=current(cfg,state,tick)
                if tick==origin:
                    chosen=root;assert chosen in available
                elif tick<HORIZON:
                    chosen=actor_choice(actor,obs,available,uniforms[tick-origin])
                else:chosen=ACTIONS.index(risk_matching(cfg,admitted,tick))
                state,cost,_=transition(cfg,state,tick,subs,ACTIONS[chosen],events[tick-origin])
                cost_sum+=cost;transitions+=1
            costs.append(cost_sum)
        own[root]=sum(costs)/scenarios
    assert transitions==int(row['forecast_transitions'])
    reported={int(a):float(v) for a,v in json.loads(row['scores_own']).items()}
    assert own.keys()==reported.keys()
    for a in own:close(own[a],reported[a],f'own root {a}')
    return transitions


def episode_replay(row,actor):
    cfg=configs()[row['case']];env=FrameworkEnv(cfg)
    env.reset(seed=int(row['seed']))
    legal_ticks=waits=0
    for tick in range(HORIZON):
        _,_,legal,obs=current(cfg,env.inner.state,tick)
        choice=actor_choice(actor,obs,legal)
        if any(len(ACTIONS[a])>0 for a in legal):
            legal_ticks+=1;waits+=int(len(ACTIONS[choice])==0)
        env.step(choice)
    metrics=env.episode_metrics()
    for key in ('objective','H_availability','H_failures','queued_cm_ticks',
                'queued_pm_ticks','H_pending','end_pending','service_starts',
                'maintenance_cost','unavailability_cost','failure_cost','tail_cost',
                'requests_created','requests_completed','end_in_service'):
        close(metrics[key],row[key],f'episode {key}')
    close(env.episode_return,row['episode_return'],'episode return')
    close(legal_ticks,row['legal_service_ticks'],'legal service ticks')
    close(waits,row['legal_service_waits'],'legal service waits')
    close(waits/legal_ticks if legal_ticks else 0,row['legal_service_wait_fraction'],'wait fraction')


def v14_episode_replay(row,actor):
    controller=V14Controller(actor,row['controller']);cfg=configs()[row['case']]
    env=FrameworkEnv(cfg);env.reset(seed=int(row['seed']))
    legal_ticks=waits=0
    for tick in range(HORIZON):
        subs,matching=controller.choose(cfg,env.inner.state,tick,case=row['case'],seed=int(row['seed']))
        admitted=submit_requests(cfg,env.inner.state,tick,subs)
        legal=legal_matchings(cfg,admitted)
        assert matching in legal
        if any(legal):legal_ticks+=1;waits+=int(len(matching)==0)
        env.step(ACTIONS.index(matching))
    metrics=env.episode_metrics()
    for key in ('objective','H_availability','H_failures','queued_cm_ticks',
                'queued_pm_ticks','H_pending','end_pending','service_starts',
                'maintenance_cost','unavailability_cost','failure_cost','tail_cost',
                'requests_created','requests_completed','end_in_service'):
        close(metrics[key],row[key],f'V14 episode {key}')
    close(env.episode_return,row['episode_return'],'V14 episode return')
    close(legal_ticks,row['legal_service_ticks'],'V14 legal service ticks')
    close(waits,row['legal_service_waits'],'V14 legal service waits')
    close(waits/legal_ticks if legal_ticks else 0,row['legal_service_wait_fraction'],'V14 wait fraction')


def audit(run):
    manifest=json.loads((run/'manifest.json').read_text())
    profile=manifest['profile']
    assert profile in ('smoke','dev') and manifest['status']=='COMPLETED'
    for name,digest in manifest['source_hashes'].items():assert sha(ROOT/name)==digest
    for name,digest in manifest['artifact_hashes'].items():assert sha(run/name)==digest
    registry=manifest['registry']
    assert registry['campaign_version']==VERSION
    assert json.loads((ROOT/'configs/maintenance_framework_v19_seed_registry.json').read_text())==registry
    assert sha(run/'selection.json')==manifest['selection_sha256']
    assert sha(V7/'selection.json')==manifest['v7_selection_sha256']
    assert sha(V14/'selection.json')==manifest['v14_selection_sha256']
    assert sha(V18/'selection.json')==manifest['v18_selection_sha256']
    v7=json.loads((V7/'selection.json').read_text())
    for arm in ('flat_free','shared_free','shared_canonical'):
        for seed in range(68000000,68000005):
            assert sha(V7/v7[arm][str(seed)]['checkpoint'])==manifest['prior_comparator_hashes'][f'{arm}/{seed}']
    v14_selection=json.loads((V14/'selection.json').read_text())
    for arm in V14_ARMS:
        for seed in range(72600000,72600005):
            item=v14_selection[arm][str(seed)]
            assert sha(V14/item['checkpoint'])==manifest['prior_comparator_hashes'][f'{arm}/{seed}']
    v18_selection=json.loads((V18/'selection.json').read_text())
    for arm in V18_ARMS:
        for seed in range(73300000,73300005):
            item=v18_selection[arm][str(seed)]
            assert sha(V18/item['checkpoint'])==manifest['prior_comparator_hashes'][f'{arm}/{seed}']
    if profile=='dev':
        train=tuple(range(73400000,73400005));select=tuple(range(73410000,73410008))
        final=tuple(range(73411000,73411016));updates=6;episodes_per_case=16;per_cell=64;scenarios=4
    else:
        train=(73420000,);select=final=(73420000,73420001)
        updates=1;episodes_per_case=2;per_cell=3;scenarios=2
    for key,expected in (('training',tuple(range(73400000,73400005))),
                         ('selection_development',tuple(range(73410000,73410008))),
                         ('evaluation_development',tuple(range(73411000,73411016))),
                         ('smoke',tuple(range(73420000,73420002)))):
        part=registry[key]
        assert tuple(range(part['start'],part['start']+part['count']))==expected
    generated=[physical_seed(s,u,c,i) for s in train for u in range(1,updates+1)
               for c in CASES for i in range(episodes_per_case)]
    assert len(generated)==len(set(generated))==manifest['training_physical_seed_count']
    assert not set(generated)&(set(select)|set(final))
    assert hashlib.sha256(json.dumps(generated).encode()).hexdigest()==manifest['training_physical_seed_sha256']
    tapes=json.loads((run/'training_tapes.json').read_text())
    for seed in train:
        expected=[[[case,physical_seed(seed,u,case,i)] for case in CASES for i in range(episodes_per_case)]
                  for u in range(1,updates+1)]
        for arm in ARMS:assert tapes[str(seed)][arm]==expected
    trace=rows(run/'training_trace.csv');qrows=rows(run/'counterfactual_states.csv')
    assert manifest['actor_parameters']==sum(p.numel() for p in TemporalMatchingActor('temporal_deficit').parameters())
    assert len(trace)==len(ARMS)*len(train)*updates
    assert len(qrows)==len(trace)*len(CASES)*per_cell
    assert manifest['counts']['training_steps']==len(ARMS)*len(train)*updates*episodes_per_case*len(CASES)*HORIZON
    trace_map={(r['arm'],int(r['train_seed']),int(r['iteration'])):r for r in trace}
    assert len(trace_map)==len(trace)
    cells={key:[] for key in trace_map}
    for r in qrows:cells[(r['arm'],int(r['train_seed']),int(r['iteration']))].append(r)
    for key,entry in trace_map.items():
        group=cells[key]
        assert len(group)==len(CASES)*per_cell==int(entry['selected_states'])
        assert int(entry['source_episodes'])==len(CASES)*episodes_per_case
        assert int(entry['source_steps'])==len(CASES)*episodes_per_case*HORIZON
        assert int(entry['forecast_roots'])==sum(len(json.loads(r['legal'])) for r in group)
        assert int(entry['forecast_transitions'])==sum(int(r['forecast_transitions']) for r in group)
        assert int(entry['forecast_scenarios'])==scenarios
        assert 0<=float(entry['kl'])<=.020001
        assert sha(run/entry['checkpoint'])==entry['checkpoint_sha256']
    for arm in ARMS:
        assert sum(int(r['forecast_transitions']) for r in trace if r['arm']==arm)==manifest['counts']['training_forecast_transitions'][arm]
    initial_active={}
    for seed in train:
        models={arm:actor_at(run/'checkpoints'/arm/str(seed)/'0.pt') for arm in ARMS}
        left,right=(models[a] for a in ARMS)
        assert all(torch.equal(left.state_dict()[name],right.state_dict()[name]) for name in left.state_dict())
        sample=next(r for r in qrows if r['arm']==ARMS[0] and int(r['train_seed'])==seed)
        state=state_from_dict(json.loads(sample['original_state']))
        _,_,legal,obs=current(configs()[sample['case']],state,int(sample['tick']))
        assert len(legal)>1
        active=[]
        for model in (left,right):
            model.zero_grad();model(torch.tensor(obs).reshape(1,56))[:,legal].sum().backward()
            active.append(sum(p.numel() for p in model.parameters() if p.grad is not None and torch.any(p.grad!=0)))
        assert active[0]==active[1] and active[0]>0
        initial_active[str(seed)]=active[0]
    sampled=[]
    for key,group in sorted(cells.items()):
        for case in CASES:
            options=sorted((r for r in group if r['case']==case),key=lambda r:r['rank'])
            assert len(options)==per_cell
            sampled.append(options[0])
    transitions=0
    for r in tqdm(sampled,desc='Independent V19 source/full Q replay',unit='state'):
        old=actor_at(run/'checkpoints'/r['arm']/r['train_seed']/f'{int(r["iteration"])-1}.pt')
        source_replay(r,old,episodes_per_case)
        state=state_from_dict(json.loads(r['original_state']))
        _,_,legal,obs=current(configs()[r['case']],state,int(r['tick']))
        assert legal==json.loads(r['legal'])
        verify_profile(obs,legal,r['arm'])
        with torch.no_grad():
            logits=old(torch.tensor(obs).reshape(1,56))
            assert logits.shape==(1,len(ACTIONS)) and torch.isfinite(logits).all()
        transitions+=root_values(r,old,int(registry['forecast_seed']),scenarios)
    development=rows(run/'development.csv')
    assert len(development)==len(ARMS)*len(train)*(updates+1)*len(CASES)*len(select)==manifest['counts']['development']
    selection=json.loads((run/'selection.json').read_text())
    for arm in ARMS:
        for seed in train:
            options=[]
            for update in range(updates+1):
                slot=[r for r in development if r['controller']==arm and int(r['train_seed'])==seed
                      and int(r['checkpoint_iteration'])==update]
                assert len(slot)==len(CASES)*len(select)
                assert {int(r['seed']) for r in slot}==set(select)
                options.append((float(np.mean([float(r['objective']) for r in slot])),update))
            best=min(options)
            item=selection[arm][str(seed)]
            assert item['iteration']==best[1] and sha(run/item['checkpoint'])==item['sha256']
            close(item['development_mean_cost'],best[0],'checkpoint selection')
    episodes=rows(run/'episodes.csv')
    expected=len(CASES)*len(final)*(len(ARMS)*len(train)+5*7+3)
    assert len(episodes)==expected==manifest['counts']['evaluation']
    assert len({(r['case'],r['controller'],r['train_seed'],r['seed']) for r in episodes})==expected
    for r in episodes:
        assert int(r['seed']) in final
        close(float(r['episode_return'])+float(r['objective']),0,'reward/cost')
        close(float(r['objective']),sum(float(r[k]) for k in ('maintenance_cost','unavailability_cost','failure_cost')),'cost components')
        close(float(r['requests_created'])-float(r['requests_completed']),float(r['end_pending'])+float(r['end_in_service']),'request conservation')
        assert float(r['invalid_assignments'])==0
    cache={};learned=[r for r in episodes if r['controller'] in ARMS]
    for r in tqdm(learned,desc='Independent V19 selected episode replay',unit='episode'):
        k=(r['controller'],r['train_seed'])
        if k not in cache:cache[k]=actor_at(run/selection[k[0]][k[1]]['checkpoint'])
        episode_replay(r,cache[k])
    frozen=[r for r in episodes if r['controller'] in V14_ARMS]
    v14_cache={}
    for r in tqdm(frozen,desc='Independent V19 frozen V14 replay',unit='episode'):
        k=(r['controller'],r['train_seed'])
        if k not in v14_cache:
            item=v14_selection[k[0]][k[1]]
            v14_cache[k]=load_v14_checkpoint(V14/item['checkpoint'])
        v14_episode_replay(r,v14_cache[k])
    frozen_v18=[r for r in episodes if r['controller'] in V18_ARMS]
    v18_cache={}
    from ht_pdm_fjsp.maintenance_framework_v18_campaign import load_checkpoint as load_v18_checkpoint
    for r in tqdm(frozen_v18,desc='Independent V19 frozen V18 replay',unit='episode'):
        k=(r['controller'],r['train_seed'])
        if k not in v18_cache:
            item=v18_selection[k[0]][k[1]]
            v18_cache[k]=load_v18_checkpoint(V18/item['checkpoint'])
        episode_replay(r,v18_cache[k])
    summary=json.loads((run/'summary.json').read_text())
    controllers=(*ARMS,*V14_ARMS,*V18_ARMS,'flat_free','shared_free','shared_canonical','fcfs','risk_matching','rollout')
    for name in controllers:
        case_means=[]
        for case in CASES:
            cell=[r for r in episodes if r['case']==case and r['controller']==name]
            assert len(cell)==summary['by_case'][case][name]['count']
            for metric,want in summary['by_case'][case][name]['means'].items():
                close(np.mean([float(r[metric]) for r in cell]),want,'case '+metric)
            case_means.append(np.mean([float(r['objective']) for r in cell]))
        close(np.mean(case_means),summary['pooled'][name],'pooled')
    lookup={(r['case'],r['controller'],r['train_seed'],int(r['seed'])):float(r['objective']) for r in episodes}
    matrix=np.asarray([[np.mean([lookup[(case,ARMS[0],str(seed),physical)]-
        lookup[(case,ARMS[1],str(seed),physical)] for case in CASES])
        for physical in final] for seed in train])
    close(matrix.mean(),summary['ablation']['mean'],'ablation')
    for i,seed in enumerate(train):close(matrix[i].mean(),summary['ablation']['train_seed_means'][str(seed)],'seed mean')
    rng=np.random.default_rng(registry['bootstrap_seed'])
    si=rng.integers(0,len(train),size=(20000,len(train)))
    pi=rng.integers(0,len(final),size=(20000,len(train),len(final)))
    ci=np.quantile(matrix[si[:,:,None],pi].mean((1,2)),[.025,.975])
    for got,want in zip(summary['ablation']['hierarchical_ci95'],ci):close(got,want,'bootstrap')
    control_cost={seed:float(np.mean([lookup[(case,ARMS[1],str(seed),physical)]
                  for case in CASES for physical in final])) for seed in train}
    strong=sorted(train,key=lambda seed:(control_cost[seed],seed))[:3]
    strong_wins=sum(matrix[train.index(seed)].mean()<0 for seed in strong)
    simple_dominators=[name for name in ('fcfs','risk_matching') if
        summary['pooled'][name]<=summary['pooled'][ARMS[0]] and
        summary['pooled_selection_seconds'][name]<=summary['pooled_selection_seconds'][ARMS[0]]]
    assert summary['dev_gate_details']==dict(strong_control_seeds=strong,strong_wins=strong_wins,simple_dominators=simple_dominators)
    expected_gate=bool(matrix.mean()<=-10 and sum(matrix.mean(axis=1)<0)>=4 and strong_wins>=2 and ci[1]<0 and
        summary['pooled'][ARMS[0]]<=1.02*summary['pooled']['rollout'] and
        summary['pooled'][ARMS[0]]<=summary['pooled']['competition'] and
        summary['pooled_selection_seconds']['rollout']>=10*summary['pooled_selection_seconds'][ARMS[0]] and
        not simple_dominators)
    assert summary['dev_go_gate']==expected_gate
    return dict(status='PASS',run=str(run),profile=profile,counterfactual_rows=len(qrows),
        sampled_full_Q_replays=len(sampled),sampled_forecast_transitions=transitions,
        selected_episode_replays=len(learned),v14_episode_replays=len(frozen),v18_episode_replays=len(frozen_v18),selection_episodes=len(development),evaluation=len(episodes),
        equal_initial_active_parameters=initial_active,
        checks=['source/artifact/V7/V14/V18 hashes','paired initial actor and training role seeds',
                'independent feature matching, source-policy and full common-bank Q sample replay','all selected RL and frozen V14/V18 episode cost/request replay',
                'disjoint selection/final grids','summary/bootstrap/gate'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    def timeout(*_):raise TimeoutError('V19 independent audit exceeded four hours')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(14400)
    try:report=audit(args.run_dir.resolve())
    finally:signal.alarm(0)
    (out/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
