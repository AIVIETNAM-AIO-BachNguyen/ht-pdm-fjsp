"""Independent V11 DEV source/counterfactual/episode and aggregation audit."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import CASES,HORIZON,FrameworkEnv,catalogue,configs,vector_observation
from ht_pdm_fjsp.maintenance_framework_v6_policy import EdgeSharingActor
from ht_pdm_fjsp.maintenance_request_policies import request_action,risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings,state_from_dict,submit_requests,transition

ROOT=Path(__file__).resolve().parent.parent
VERSION='maintenance_framework_v11_reference_continuation_dev_v1'
ACTIONS=catalogue()
ARMS=('reference_credit','self_credit')
V7=ROOT/'research_result/framework_v7_full_20261020T012531Z'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))

def close(a,b,label):
    if not np.isclose(float(a),float(b),atol=1e-7,rtol=0):
        raise AssertionError(f'{label}: {a} != {b}')

def actor_at(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    actor=EdgeSharingActor('shared_edge')
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

def physical_seed(train,iteration,case,index):
    return role_seed(train,f'{VERSION}:train:{iteration}:{case}:{index}')%(2**31)

def source_replay(row,actor):
    case=row['case'];seed=int(row['train_seed']);iteration=int(row['iteration']);tick=int(row['tick'])
    physical=int(row['physical_seed'])
    matches=[i for i in range(16) if physical_seed(seed,iteration,case,i)==physical]
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
    rng=random.Random(bank_seed)
    bank=[]
    for _ in range(scenarios):
        failures=[[rng.random()<cfg.failure_probability for m in range(cfg.machines)]
                  for t in range(origin,cfg.horizon)]
        choices=[rng.random() for t in range(origin,HORIZON)]
        bank.append((failures,choices))
    state0=state_from_dict(json.loads(row['original_state']))
    legal=json.loads(row['legal']);scores={'self':{},'reference':{}}
    transitions={'self':0,'reference':0}
    for root in legal:
        for continuation in ('self','reference'):
            costs=[]
            for events,uniforms in bank:
                state=state0;cost_sum=0.
                for tick in range(origin,cfg.horizon):
                    subs,admitted,available,obs=current(cfg,state,tick)
                    if tick==origin:
                        chosen=root
                        assert chosen in available
                    elif continuation=='self' and tick<HORIZON:
                        chosen=actor_choice(actor,obs,available,uniforms[tick-origin])
                    else:
                        chosen=ACTIONS.index(risk_matching(cfg,admitted,tick))
                    state,cost,_=transition(cfg,state,tick,subs,ACTIONS[chosen],events[tick-origin])
                    cost_sum+=cost;transitions[continuation]+=1
                costs.append(cost_sum)
            scores[continuation][root]=sum(costs)/len(costs)
    for continuation in ('self','reference'):
        assert transitions[continuation]==int(row[f'forecast_transitions_{continuation}'])
        reported={int(a):float(v) for a,v in json.loads(row[f'scores_{continuation}']).items()}
        assert scores[continuation].keys()==reported.keys()
        for action in scores[continuation]:
            close(scores[continuation][action],reported[action],f'{continuation} root {action}')
    return transitions['self']+transitions['reference']

def episode_replay(row,actor):
    cfg=configs()[row['case']];env=FrameworkEnv(cfg)
    env.reset(seed=int(row['seed']))
    for tick in range(HORIZON):
        _,_,legal,obs=current(cfg,env.inner.state,tick)
        choice=actor_choice(actor,obs,legal)
        env.step(choice)
    metrics=env.episode_metrics()
    for key in ('objective','H_availability','H_failures','queued_cm_ticks',
                'queued_pm_ticks','H_pending','end_pending','service_starts',
                'maintenance_cost','unavailability_cost','failure_cost','tail_cost',
                'requests_created','requests_completed','end_in_service'):
        close(metrics[key],row[key],f'episode {key}')
    close(env.episode_return,row['episode_return'],'episode return')


def audit(run):
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='COMPLETED' and manifest['profile']=='dev'
    for name,digest in manifest['source_hashes'].items():assert sha(ROOT/name)==digest
    for name,digest in manifest['artifact_hashes'].items():assert sha(run/name)==digest
    selection=json.loads((run/'selection.json').read_text())
    assert sha(run/'selection.json')==manifest['selection_sha256']
    assert sha(V7/'selection.json')==manifest['v7_selection_sha256']
    prior=json.loads((V7/'selection.json').read_text())
    for arm in ('flat_free','shared_free','shared_canonical'):
        for seed in range(68000000,68000005):
            assert sha(V7/prior[arm][str(seed)]['checkpoint'])==manifest['prior_comparator_hashes'][f'{arm}/{seed}']
    train_seeds=tuple(range(70500000,70500005));dev_seeds=tuple(range(70510000,70510016))
    physical=[physical_seed(seed,iteration,case,index) for seed in train_seeds
              for iteration in range(1,7) for case in CASES for index in range(16)]
    assert len(physical)==len(set(physical))==manifest['training_physical_seed_count']
    assert hashlib.sha256(json.dumps(physical).encode()).hexdigest()==manifest['training_physical_seed_sha256']
    tapes=json.loads((run/'training_tapes.json').read_text())
    for train in train_seeds:
        expected=[[[case,physical_seed(train,iteration,case,index)] for case in CASES for index in range(16)]
                  for iteration in range(1,7)]
        for arm in ARMS:assert tapes[str(train)][arm]==expected
    trace=rows(run/'training_trace.csv')
    qrows=rows(run/'counterfactual_states.csv')
    assert len(trace)==len(ARMS)*len(train_seeds)*6
    assert manifest['counts']['training_steps']==len(ARMS)*len(train_seeds)*6*64*HORIZON
    for arm in ARMS:
        assert sum(int(r['forecast_transitions']) for r in trace if r['arm']==arm)==(
            manifest['counts']['training_forecast_transitions'][arm])
    trace_map={(r['arm'],int(r['train_seed']),int(r['iteration'])):r for r in trace}
    assert len(trace_map)==len(trace)
    by_cell={key:[] for key in trace_map}
    for r in qrows:
        key=(r['arm'],int(r['train_seed']),int(r['iteration']))
        by_cell[key].append(r)
    for key,trace_row in trace_map.items():
        cell=by_cell[key]
        assert len(cell)==int(trace_row['selected_states'])
        for continuation in ('self','reference'):
            assert sum(int(r[f'forecast_transitions_{continuation}']) for r in cell)==int(
                trace_row[f'forecast_transitions_{continuation}'])
        assert sum(int(r['forecast_transitions_self'])+int(r['forecast_transitions_reference'])
                   for r in cell)==int(trace_row['forecast_transitions'])
        assert sum(len(json.loads(r['legal'])) for r in cell)==int(trace_row['forecast_roots'])
        assert int(trace_row['source_episodes'])==64 and int(trace_row['source_steps'])==64*HORIZON
        assert int(trace_row['forecast_scenarios'])==4
        coverage=json.loads(trace_row['coverage'])
        for case in CASES:
            for bin_index,quota in enumerate((16,8,8)):
                key=f'{case}/{bin_index}'
                selected=sum(r['case']==case and min(int(r['tick'])//16,2)==bin_index for r in cell)
                assert selected==coverage[key]['selected']<=quota
                assert selected<=coverage[key]['eligible']
        assert 0<=float(trace_row['kl'])<=.020001
        assert sha(run/trace_row['checkpoint'])==trace_row['checkpoint_sha256']
    for seed in train_seeds:
        left={(r['case'],r['physical_seed'],r['tick']):
              (r['bank_seed'],r['scores_self'],r['scores_reference'])
              for r in by_cell[('reference_credit',seed,1)]}
        right={(r['case'],r['physical_seed'],r['tick']):
               (r['bank_seed'],r['scores_self'],r['scores_reference'])
               for r in by_cell[('self_credit',seed,1)]}
        assert left==right
    sampled=[]
    for key,cell in sorted(by_cell.items()):
        for case in CASES:
            options=sorted((r for r in cell if r['case']==case),key=lambda r:r['rank'])
            if options:sampled.append(options[0])
    transitions=0
    for r in tqdm(sampled,desc='Independent V11 source/Q replay',unit='state'):
        old=actor_at(run/'checkpoints'/r['arm']/r['train_seed']/f'{int(r["iteration"])-1}.pt')
        source_replay(r,old)
        transitions+=root_values(r,old,int(manifest['registry']['forecast_seed']))
    development=rows(run/'development.csv')
    expected_dev=len(ARMS)*len(train_seeds)*7*len(CASES)*len(dev_seeds)
    assert len(development)==expected_dev==manifest['counts']['development']
    for arm in ARMS:
        for train in train_seeds:
            candidates=[]
            for iteration in range(7):
                cell=[r for r in development if r['controller']==arm and
                      int(r['train_seed'])==train and int(r['checkpoint_iteration'])==iteration]
                assert len(cell)==len(CASES)*len(dev_seeds)
                candidates.append((float(np.mean([float(r['objective']) for r in cell])),iteration))
            best=min(candidates)
            item=selection[arm][str(train)]
            assert item['iteration']==best[1]
            close(item['development_mean_cost'],best[0],'checkpoint selection')
            assert sha(run/item['checkpoint'])==item['sha256']
    episodes=rows(run/'episodes.csv')
    expected_eval=len(CASES)*len(dev_seeds)*(len(ARMS)*len(train_seeds)+3*5+3)
    assert len(episodes)==expected_eval==manifest['counts']['evaluation']
    keys={(r['case'],r['controller'],r['train_seed'],int(r['seed'])) for r in episodes}
    assert len(keys)==expected_eval
    for r in episodes:
        close(float(r['episode_return'])+float(r['objective']),0,'reward/cost')
        close(float(r['objective']),sum(float(r[k]) for k in
              ('maintenance_cost','unavailability_cost','failure_cost')),'cost components')
        close(float(r['requests_created'])-float(r['requests_completed']),
              float(r['end_pending'])+float(r['end_in_service']),'request conservation')
        assert float(r['invalid_assignments'])==0
    model_cache={}
    learned=[r for r in episodes if r['controller'] in ARMS]
    for r in tqdm(learned,desc='Independent V11 selected episode replay',unit='episode'):
        key=(r['controller'],r['train_seed'])
        if key not in model_cache:model_cache[key]=actor_at(run/selection[key[0]][key[1]]['checkpoint'])
        episode_replay(r,model_cache[key])
    summary=json.loads((run/'summary.json').read_text())
    for name in (*ARMS,'flat_free','shared_free','shared_canonical','fcfs','risk_matching','rollout'):
        case_means=[]
        for case in CASES:
            cell=[float(r['objective']) for r in episodes if r['case']==case and r['controller']==name]
            close(np.mean(cell),summary['by_case'][case][name]['means']['objective'],'case mean')
            case_means.append(np.mean(cell))
        close(np.mean(case_means),summary['pooled'][name],'pooled mean')
    lookup={(r['case'],r['controller'],r['train_seed'],int(r['seed'])):float(r['objective']) for r in episodes}
    matrix=np.asarray([[np.mean([lookup[(case,'reference_credit',str(train),seed)]-
        lookup[(case,'self_credit',str(train),seed)] for case in CASES])
        for seed in dev_seeds] for train in train_seeds])
    close(np.mean(matrix),summary['ablation']['mean'],'ablation mean')
    rng=np.random.default_rng(int(manifest['registry']['bootstrap_seed']))
    ti=rng.integers(0,5,size=(20000,5));pi=rng.integers(0,16,size=(20000,16))
    ci=np.quantile(matrix[ti[:,:,None],pi[:,None,:]].mean(axis=(1,2)),[.025,.975])
    for got,want in zip(ci,summary['ablation']['hierarchical_ci95']):close(got,want,'bootstrap')
    seed_means={seed:float(matrix[i].mean()) for i,seed in enumerate(train_seeds)}
    control_cost={seed:float(np.mean([lookup[(case,'self_credit',str(seed),physical)]
                  for case in CASES for physical in dev_seeds])) for seed in train_seeds}
    strong=sorted(train_seeds,key=lambda seed:(control_cost[seed],seed))[:3]
    strong_wins=sum(seed_means[seed]<0 for seed in strong)
    dominance=(summary['pooled']['risk_matching']<=summary['pooled']['reference_credit'] and
               summary['pooled_selection_seconds']['risk_matching']<=
               summary['pooled_selection_seconds']['reference_credit'])
    assert summary['dev_gate_details']==dict(strong_control_seeds=strong,
                                              strong_wins=strong_wins,risk_dominates=dominance)
    expected_gate=(matrix.mean()<=-10 and sum(value<0 for value in seed_means.values())>=4 and
                   strong_wins>=3 and
                   summary['pooled']['reference_credit']<=1.02*summary['pooled']['rollout'] and
                   summary['pooled_selection_seconds']['rollout']>=
                   10*summary['pooled_selection_seconds']['reference_credit'] and not dominance)
    assert summary['dev_go_gate']==bool(expected_gate)
    return dict(status='PASS',run=str(run),counterfactual_rows=len(qrows),
        sampled_full_Q_replays=len(sampled),sampled_forecast_transitions=transitions,
        selected_episode_replays=len(learned),development=len(development),evaluation=len(episodes),
        checks=['source/artifact/V7 comparator hashes','physical/source-policy stream replay',
                'full-horizon common-bank root-score sample replay','complete checkpoint and episode grids',
                'selected V11 full-episode cost/health/request replay','selection and hierarchical bootstrap'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    result=audit(args.run_dir.resolve())
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
