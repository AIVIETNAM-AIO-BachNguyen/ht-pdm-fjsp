"""Independent selected-policy trajectory, sample and bounded-root audit for V9."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import CASES,HORIZON,FrameworkEnv,catalogue,configs,vector_observation
from ht_pdm_fjsp.maintenance_framework_v6_policy import EdgeSharingActor
from ht_pdm_fjsp.maintenance_request_policies import choose,request_action,risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings,state_from_dict,submit_requests,validate_state

ROOT=Path(__file__).resolve().parent.parent
V9=ROOT/'research_result/framework_v9_dev_20261020T062502Z'
VERSION='v9_selected_action_failure_dev_v1'
ARMS=('all_action','chosen_action')
TRAIN=range(70300000,70300005)
DEV=range(70310000,70310016)
ACTIONS=catalogue()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def csv_rows(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))

def close(x,y,label):
    if not np.isclose(float(x),float(y),atol=1e-8,rtol=0):
        raise AssertionError(f'{label}: {x} != {y}')

def actor_load(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    actor=EdgeSharingActor('shared_edge');actor.load_state_dict(payload['state_dict']);actor.eval()
    return actor

def present(cfg,state,tick):
    subs=request_action(cfg,state,tick,'threshold')
    admitted=submit_requests(cfg,state,tick,subs)
    legal=sorted(ACTIONS.index(a) for a in legal_matchings(cfg,admitted))
    obs=vector_observation(cfg,admitted,tick)
    return admitted,legal,obs

def choose_actor(actor,obs,legal):
    with torch.no_grad():
        scores=actor(torch.tensor(obs).reshape(1,56))[0]
        masked=torch.full_like(scores,-1e9);masked[legal]=scores[legal]
        return int(np.argmax(torch.softmax(masked,dim=0).numpy()))

def classify(left,right):
    a,b=ACTIONS[left],ACTIONS[right]
    if a==b:return 'match'
    if not a and b:return 'wait_vs_service'
    if a and not b:return 'service_vs_wait'
    if len(a)!=len(b):return 'lower_cardinality' if len(a)<len(b) else 'higher_cardinality'
    return ('machine_subset' if {m for m,_ in a}!={m for m,_ in b}
            else 'technician_assignment')

def rank(case,arm,seed,physical,tick):
    text=f'{VERSION}:{case}:{arm}:{seed}:{physical}:{tick}'
    return hashlib.sha256(text.encode()).hexdigest()

def audit(run):
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='COMPLETED' and manifest['profile']=='dev'
    for name,digest in manifest['source_hashes'].items():assert sha(ROOT/name)==digest
    for name,digest in manifest['artifact_hashes'].items():assert sha(run/name)==digest
    for name,key in [('manifest.json','v9_manifest_sha256'),('selection.json','v9_selection_sha256'),
                     ('episodes.csv','v9_episodes_sha256')]:assert sha(V9/name)==manifest[key]
    original_manifest=json.loads((V9/'manifest.json').read_text())
    selection=json.loads((V9/'selection.json').read_text())
    models={}
    for arm in ARMS:
        for seed in TRAIN:
            path=V9/selection[arm][str(seed)]['checkpoint']
            digest=sha(path)
            assert digest==manifest['selected_checkpoint_hashes'][f'{arm}/{seed}']
            models[(arm,seed)]=actor_load(path)
    ref={(r['case'],r['controller'],int(r['train_seed']),int(r['seed'])):r
         for r in csv_rows(V9/'episodes.csv') if r['controller'] in ARMS}
    actions=csv_rows(run/'actions.csv');episodes=csv_rows(run/'episodes.csv')
    assert len(actions)==4*2*5*16*48==manifest['counts']['decisions']
    assert len(episodes)==4*2*5*16==manifest['counts']['episodes']
    action_map={(r['case'],r['source_arm'],int(r['train_seed']),
                 int(r['physical_seed']),int(r['tick'])):r for r in actions}
    episode_map={(r['case'],r['source_arm'],int(r['train_seed']),
                  int(r['physical_seed'])):r for r in episodes}
    assert len(action_map)==len(actions) and len(episode_map)==len(episodes)
    candidates=defaultdict(list);snapshots={};by_source={};by_bin={}
    grid=[(case,arm,seed,physical) for case in CASES for arm in ARMS
          for seed in TRAIN for physical in DEV]
    for case,arm,seed,physical in tqdm(grid,desc='Independent V9 action trajectory audit',unit='episode'):
        cfg=configs()[case];env=FrameworkEnv(cfg);env.reset(seed=physical)
        hstarts=0
        for tick in range(HORIZON):
            old=env.inner.state
            admitted,legal,obs=present(cfg,old,tick)
            chosen=choose_actor(models[(arm,seed)],obs,legal)
            risk=ACTIONS.index(risk_matching(cfg,admitted,tick))
            key=(case,arm,seed,physical,tick);row=action_map[key]
            assert int(row['chosen_action'])==chosen and int(row['risk_action'])==risk
            assert int(row['legal_count'])==len(legal)
            assert int(row['legal_nonempty'])==int(any(ACTIONS[a] for a in legal))
            assert row['category']==classify(chosen,risk)
            assert row['rank']==rank(*key)
            assert row['clock_bin']==('early','middle','late')[tick//16]
            for field,value in [('chosen_cardinality',len(ACTIONS[chosen])),
                                ('risk_cardinality',len(ACTIONS[risk])),
                                ('queued_cm',sum(r is not None and r.started_at is None and
                                                 r.kind=='corrective' for r in admitted.requests)),
                                ('queued_pm',sum(r is not None and r.started_at is None and
                                                 r.kind=='preventive' for r in admitted.requests)),
                                ('free_technicians',sum(x==0 for x in old.physical.remaining)),
                                ('chosen_duration',sum(cfg.service_time[m][j] for m,j in ACTIONS[chosen])),
                                ('risk_duration',sum(cfg.service_time[m][j] for m,j in ACTIONS[risk]))]:
                close(row[field],value,field)
            hstarts+=len(ACTIONS[chosen])
            source=f'{case}/{arm}/{seed}'
            cell=by_source.setdefault(source,dict(decisions=0,wait_with_legal_service=0,
                wait_without_legal_service=0,service_starts=0,risk_service_starts=0,
                queued_cm_ticks_preaction=0,categories=Counter()))
            cell['decisions']+=1
            cell['wait_with_legal_service']+=chosen==0 and any(ACTIONS[a] for a in legal)
            cell['wait_without_legal_service']+=chosen==0 and not any(ACTIONS[a] for a in legal)
            cell['service_starts']+=len(ACTIONS[chosen]);cell['risk_service_starts']+=len(ACTIONS[risk])
            cell['queued_cm_ticks_preaction']+=sum(r is not None and r.started_at is None and
                                                  r.kind=='corrective' for r in admitted.requests)
            cell['categories'][classify(chosen,risk)]+=1
            b=f'{source}/{row["clock_bin"]}'
            bucket=by_bin.setdefault(b,dict(decisions=0,wait_with_legal_service=0,
                lower_cardinality=0,machine_subset=0,technician_assignment=0))
            bucket['decisions']+=1
            bucket['wait_with_legal_service']+=chosen==0 and any(ACTIONS[a] for a in legal)
            for name in ('lower_cardinality','machine_subset','technician_assignment'):
                bucket[name]+=classify(chosen,risk)==name
            if len(legal)>=2:
                candidates[(case,arm,seed,row['clock_bin'])].append(key)
                snapshots[key]=json.loads(json.dumps(asdict(old)))
            env.step(chosen)
        metrics=env.episode_metrics()
        item=episode_map[(case,arm,seed,physical)]
        prior=ref[(case,arm,seed,physical)]
        for field in ('objective','H_availability','H_failures','queued_cm_ticks',
                      'H_pending','end_pending','service_starts'):
            close(item[field],metrics[field],f'episode {field}')
            close(item[field],prior[field],f'prior episode {field}')
        assert hstarts==int(item['H_service_starts'])
    summary=json.loads((run/'summary.json').read_text())
    assert summary['episodes']==len(episodes) and summary['decisions']==len(actions)
    assert summary['candidate_states']==len(snapshots)
    assert summary['by_source']==by_source
    assert summary['by_case_arm_bin']==by_bin
    states=[json.loads(line) for line in (run/'states.jsonl').open()]
    choices=csv_rows(run/'choices.csv')
    selected=[];sampling={}
    for case in CASES:
        for arm in ARMS:
            for seed in TRAIN:
                for b in ('early','middle','late'):
                    key=(case,arm,seed,b)
                    options=sorted(candidates[key],key=lambda x:rank(*x))
                    picked=options[:2];selected+=picked
                    sampling['/'.join(map(str,key))]=dict(eligible=len(options),selected=len(picked))
    selected.sort(key=lambda x:rank(*x))
    assert summary['sampling']==sampling and len(states)==len(selected)==240
    assert len(choices)==len(states)*len(ARMS)
    choice_map={(int(r['state_id']),r['policy_arm']):r for r in choices}
    assert len(choice_map)==len(choices)
    root_summary={};transition_total=0
    for index,(raw,key) in enumerate(tqdm(zip(states,selected),total=len(states),
                                           desc='Independent V9 bounded-root audit',unit='state')):
        case,arm,seed,physical,tick=key
        assert raw['state_id']==index
        assert (raw['case'],raw['source_arm'],raw['train_seed'],raw['physical_seed'],raw['tick'])==key
        assert raw['original']==snapshots[key]
        cfg=configs()[case];state=state_from_dict(raw['original']);validate_state(cfg,state,tick)
        admitted,legal,obs=present(cfg,state,tick)
        assert raw['legal_actions']==legal
        forecast=role_seed(70360000,f'{VERSION}:{rank(*key)}')
        assert raw['forecast_seed']==forecast
        _,best_matching,diagnostics=choose(cfg,state,tick,'threshold','rollout',
            policy_horizon=HORIZON,forecast_seed=forecast,scenarios=16,forecast_horizon=6)
        scores={ACTIONS.index(tuple(x['matching'])):float(x['expected_cost'])
                for x in diagnostics['root_scores']}
        assert scores.keys()=={int(a) for a in raw['scores']}
        for a,v in scores.items():close(v,raw['scores'][str(a)],'root score')
        risk=ACTIONS.index(risk_matching(cfg,admitted,tick));planner=ACTIONS.index(best_matching)
        assert raw['risk_action']==risk and raw['planner_action']==planner
        assert raw['forecast_transitions']==diagnostics['forecast_transitions']
        transition_total+=diagnostics['forecast_transitions']
        src=f'{case}/{arm}/{seed}'
        cell=root_summary.setdefault(src,dict(states=0,source_regret_sum=0.,risk_regret_sum=0.,
            source_gap_ge_1=0,source_gap_ge_5=0,source_gap_ge_10=0,categories=Counter()))
        cell['states']+=1
        for target in ARMS:
            row=choice_map[(index,target)]
            action=choose_actor(models[(target,seed)],obs,legal)
            assert int(row['policy_action'])==action
            assert int(row['risk_action'])==risk and int(row['planner_action'])==planner
            close(row['full_regret'],scores[action]-scores[planner],'root regret')
            close(row['risk_regret'],scores[risk]-scores[planner],'risk regret')
            assert row['versus_risk']==classify(action,risk)
            assert row['versus_planner']==classify(action,planner)
            assert int(row['cardinality_delta_vs_planner'])==len(ACTIONS[action])-len(ACTIONS[planner])
            if target==arm:
                cell['source_regret_sum']+=scores[action]-scores[planner]
                cell['risk_regret_sum']+=scores[risk]-scores[planner]
                cell['categories'][classify(action,planner)]+=1
                for threshold in (1,5,10):cell[f'source_gap_ge_{threshold}']+=scores[action]-scores[planner]>=threshold
    assert summary['root_by_source']==root_summary
    assert summary['forecast_transitions']==transition_total==manifest['counts']['forecast_transitions']
    return dict(status='PASS',run=str(run),episodes=len(episodes),decisions=len(actions),
        root_states=len(states),forecast_transitions=transition_total,
        checks=['source/input/artifact/checkpoint hashes','complete selected-policy episode and action replay',
                'exact hash-stratified state sample','common-bank root score and choice replay',
                'taxonomy and summary accounting'])


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
