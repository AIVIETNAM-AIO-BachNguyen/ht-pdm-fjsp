"""First-divergence V17/rollout replay with common-bank own and risk continuation."""
from __future__ import annotations
import argparse,csv,hashlib,json,random,subprocess,time
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import CASES,HORIZON,FrameworkEnv,catalogue,configs
from ht_pdm_fjsp.maintenance_framework_v17_campaign import load_checkpoint,legal_present,policy_action
from ht_pdm_fjsp.maintenance_framework_v7_campaign import V7RuleController
from ht_pdm_fjsp.maintenance_request_policies import request_action,risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings,state_from_dict,submit_requests,transition

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'research_result/framework_v17_dev_20261021T022840Z'
AUDIT=ROOT/'research_result/framework_v17_audit_20261021T031626Z/independent_audit.json'
PLAN=ROOT/'research_result/v17_first_divergence_continuation_plan_vi.md'
TARGET=('interchangeable_scale1','reciprocal_scale2')
TRAIN=tuple(range(73000000,73000005));PHYSICAL=tuple(range(73011000,73011016))
ACTIONS=catalogue();ANCHOR=73070000;BOOTSTRAP_SEED=73070001
METRICS=('objective','maintenance_cost','unavailability_cost','failure_cost','tail_cost',
         'H_availability','H_failures','queued_cm_ticks','queued_pm_ticks','H_pending',
         'end_pending','service_starts','requests_created','requests_completed')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write(p,rows):
    with Path(p).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def verify_metrics(got,want):
    for name in METRICS:
        assert np.isclose(float(got[name]),float(want[name]),atol=1e-8,rtol=0),(name,got[name],want[name])

def kind(left,right):
    a,b=ACTIONS[left],ACTIONS[right]
    if a==b:return 'same'
    if a and not b:return 'actor_service_rollout_wait'
    if b and not a:return 'actor_wait_rollout_service'
    if len(a)!=len(b):return 'actor_higher_cardinality' if len(a)>len(b) else 'actor_lower_cardinality'
    return 'machine_set' if {m for m,_ in a}!={m for m,_ in b} else 'assignment'

def episode(case,physical,actor,forecast_anchor,deadline):
    cfg=configs()[case];env=FrameworkEnv(cfg);env.reset(seed=physical)
    rule=V7RuleController('rollout',forecast_anchor,deadline) if actor is None else None
    actions=[];states=[];contexts=[]
    for tick in range(HORIZON):
        deadline();state=env.inner.state
        submissions,admitted,legal,obs=legal_present(cfg,state,tick)
        if actor is None:
            supplied,matching=rule.choose(cfg,state,tick,case=case,seed=physical)
            assert supplied==submissions
            action=ACTIONS.index(matching)
        else:action=policy_action(actor,obs,legal)
        assert action in legal
        actions.append(action);states.append(asdict(state))
        contexts.append(dict(tick=tick,ages=list(state.physical.ages),remaining=list(state.physical.remaining),
                             assigned=list(state.physical.assigned),failed=list(state.physical.failed),
                             queued=[i for i,r in enumerate(admitted.requests) if r and r.started_at is None],
                             legal=legal))
        env.step(action)
    return actions,states,contexts,env.episode_metrics()

def bank(cfg,case,train,physical,tick,count):
    seed=role_seed(ANCHOR,f'v17_first_divergence:{case}:{train}:{physical}:{tick}')
    rng=random.Random(seed)
    tapes=[]
    for _ in range(count):
        events=[[rng.random()<cfg.failure_probability for _ in range(cfg.machines)]
                for _ in range(tick,cfg.horizon)]
        uniforms=[rng.random() for _ in range(tick,HORIZON)]
        tapes.append(dict(events=events,uniforms=uniforms))
    return seed,tapes

def root_cost(cfg,state,tick,action,tape,actor,mode,deadline):
    current=state;total=0.;short=0.;steps=0
    for offset,t in enumerate(range(tick,cfg.horizon)):
        deadline()
        submissions,admitted,legal,obs=legal_present(cfg,current,t)
        if offset==0:chosen=action
        elif t>=HORIZON or mode=='risk':chosen=ACTIONS.index(risk_matching(cfg,admitted,t))
        else:chosen=policy_action(actor,obs,legal,tape['uniforms'][offset])
        assert chosen in legal
        current,cost,_=transition(cfg,current,t,submissions,ACTIONS[chosen],tape['events'][offset])
        total+=cost;steps+=1
        if offset<6:short+=cost
    return short,total,steps

def bootstrap(cube):
    rng=np.random.default_rng(BOOTSTRAP_SEED);values=np.empty(20000)
    for left in tqdm(range(0,20000,1000),desc='V17 first-divergence bootstrap',unit='chunk'):
        seeds=rng.integers(0,5,size=(1000,5))
        physical=rng.integers(0,16,size=(1000,5,16))
        values[left:left+1000]=np.take_along_axis(cube[seeds],physical,axis=2).mean((1,2))
    return np.quantile(values,[.025,.975]).tolist()

def run(profile,out):
    start=time.monotonic();limit=1800 if profile=='dev' else 300
    def deadline():
        if time.monotonic()-start>limit:raise TimeoutError('V17 first-divergence deadline')
    report=json.loads(AUDIT.read_text());source=json.loads((SOURCE/'manifest.json').read_text())
    assert report['status']=='PASS' and Path(report['run']).resolve()==SOURCE
    assert source['status']=='COMPLETED' and source['profile']=='dev'
    for name,digest in source['artifact_hashes'].items():assert sha(SOURCE/name)==digest
    selection=json.loads((SOURCE/'selection.json').read_text())
    assert sha(SOURCE/'selection.json')==source['selection_sha256']
    seedset=TRAIN if profile=='dev' else TRAIN[:1]
    physicals=PHYSICAL if profile=='dev' else PHYSICAL[:1]
    scenarios=16 if profile=='dev' else 1
    models={};checkpoint_hashes={}
    for train in seedset:
        item=selection['rank_projection'][str(train)];cp=SOURCE/item['checkpoint']
        assert sha(cp)==item['sha256']
        models[train]=load_checkpoint(cp);checkpoint_hashes[str(train)]=item['sha256']
    original={(r['controller'],r['case'],r['train_seed'],int(r['seed'])):r for r in read(SOURCE/'episodes.csv')
              if r['case'] in TARGET and r['controller'] in ('rank_projection','rollout')}
    rollout={};roll_rows=[];actor_rows=[];pairs=[];q_rows=[];actions=[];transitions=0
    for case in tqdm(TARGET,desc='V17 unique rollout replay',unit='case'):
        for physical in physicals:
            seq,states,contexts,metrics=episode(case,physical,None,source['registry']['forecast_seed'],deadline)
            verify_metrics(metrics,original[('rollout',case,'',physical)])
            rollout[(case,physical)]=(seq,states,contexts,metrics)
            roll_rows.append(dict(case=case,physical_seed=physical,**{k:metrics[k] for k in METRICS}))
            actions.extend(dict(controller='rollout',case=case,train_seed='',physical_seed=physical,
                                tick=t,action=a) for t,a in enumerate(seq))
    grid=[(case,train,physical) for case in TARGET for train in seedset for physical in physicals]
    for case,train,physical in tqdm(grid,desc='V17 first-divergence pairs',unit='pair'):
        deadline();cfg=configs()[case];actor=models[train]
        seq,states,contexts,metrics=episode(case,physical,actor,source['registry']['forecast_seed'],deadline)
        verify_metrics(metrics,original[('rank_projection',case,str(train),physical)])
        actions.extend(dict(controller='rank_projection',case=case,train_seed=train,physical_seed=physical,
                            tick=t,action=a) for t,a in enumerate(seq))
        other,other_states,_,baseline=rollout[(case,physical)]
        first=next((t for t,(a,b) in enumerate(zip(seq,other)) if a!=b),None)
        row=dict(case=case,train_seed=train,physical_seed=physical,first_tick=-1 if first is None else first,
                 category='same' if first is None else kind(seq[first],other[first]),
                 actor_action=-1 if first is None else seq[first],
                 rollout_action=-1 if first is None else other[first],
                 actor_duration=0 if first is None else sum(cfg.service_time[m][j] for m,j in ACTIONS[seq[first]]),
                 rollout_duration=0 if first is None else sum(cfg.service_time[m][j] for m,j in ACTIONS[other[first]]),
                 actual_cost_delta=metrics['objective']-baseline['objective'],
                 actual_failure_delta=metrics['H_failures']-baseline['H_failures'],
                 actual_pm_queue_delta=metrics['queued_pm_ticks']-baseline['queued_pm_ticks'],
                 actual_cm_queue_delta=metrics['queued_cm_ticks']-baseline['queued_cm_ticks'],
                 own_q6_delta=0.,own_qfull_delta=0.,risk_q6_delta=0.,risk_qfull_delta=0.,
                 actor_probability=0.,rollout_probability=0.,logit_margin=0.,forecast_seed=0,
                 forecast_transitions=0)
        if first is not None:
            assert states[first]==other_states[first]
            assert all(seq[t]==other[t] and states[t]==other_states[t] for t in range(first))
            state=state_from_dict(states[first]);_,_,legal,obs=legal_present(cfg,state,first)
            with torch.no_grad():logits=actor(torch.as_tensor(obs).reshape(1,56))[0].numpy()
            allowed=np.full_like(logits,-1e9);allowed[list(legal)]=logits[list(legal)]
            logits_exp=np.exp(allowed-np.max(allowed));probs=logits_exp/logits_exp.sum()
            row.update(actor_probability=float(probs[seq[first]]),rollout_probability=float(probs[other[first]]),
                       logit_margin=float(logits[seq[first]]-logits[other[first]]))
            bank_seed,tapes=bank(cfg,case,train,physical,first,scenarios)
            details=[];scores={mode:[] for mode in ('own','risk')}
            for tape in tapes:
                one={}
                for mode in ('own','risk'):
                    for label,action in (('actor',seq[first]),('rollout',other[first])):
                        result=root_cost(cfg,state,first,action,tape,actor,mode,deadline)
                        one[f'{mode}_{label}']=result[:2];transitions+=result[2]
                    scores[mode].append((one[f'{mode}_actor'][0]-one[f'{mode}_rollout'][0],
                                         one[f'{mode}_actor'][1]-one[f'{mode}_rollout'][1]))
                details.append(one)
            for mode in ('own','risk'):
                row[f'{mode}_q6_delta']=float(np.mean([x[0] for x in scores[mode]]))
                row[f'{mode}_qfull_delta']=float(np.mean([x[1] for x in scores[mode]]))
            row.update(forecast_seed=bank_seed,forecast_transitions=scenarios*4*(cfg.horizon-first))
            q_rows.append(dict(case=case,train_seed=train,physical_seed=physical,tick=first,
                               original_state=states[first],context=contexts[first],forecast_seed=bank_seed,
                               bank_sha256=hashlib.sha256(json.dumps(tapes,sort_keys=True).encode()).hexdigest(),
                               tapes=tapes,scenario_scores=details))
        pairs.append(row)
    write(out/'rollout_episodes.csv',roll_rows);write(out/'actions.csv',actions);write(out/'pairs.csv',pairs)
    with (out/'q_banks.jsonl').open('w') as f:
        for record in q_rows:f.write(json.dumps(record,sort_keys=True)+'\n')
    summary=dict(status='COMPLETED',profile=profile,pairs=len(pairs),divergent=len(q_rows),
                 forecast_transitions=transitions,cases={})
    for case in TARGET:
        cell=[r for r in pairs if r['case']==case]
        cube=np.asarray([[next(r['own_qfull_delta'] for r in cell if r['train_seed']==seed and r['physical_seed']==physical)
                          for physical in physicals] for seed in seedset])
        by_seed={str(seed):float(cube[i].mean()) for i,seed in enumerate(seedset)}
        ci=bootstrap(cube) if profile=='dev' else None
        summary['cases'][case]=dict(pairs=len(cell),categories={k:sum(r['category']==k for r in cell)
                                     for k in ('same','actor_service_rollout_wait','actor_wait_rollout_service',
                                               'actor_higher_cardinality','actor_lower_cardinality','machine_set','assignment')},
            first_tick_mean=float(np.mean([r['first_tick'] for r in cell if r['first_tick']>=0])) if any(r['first_tick']>=0 for r in cell) else None,
            actual_cost_delta=float(np.mean([r['actual_cost_delta'] for r in cell])),
            actual_failure_delta=float(np.mean([r['actual_failure_delta'] for r in cell])),
            own_qfull_delta=float(cube.mean()),own_qfull_ci95=ci,own_qfull_by_seed=by_seed,
            own_q6_delta=float(np.mean([r['own_q6_delta'] for r in cell])),
            risk_qfull_delta=float(np.mean([r['risk_qfull_delta'] for r in cell])),
            own_qfull_ge5=sum(r['own_qfull_delta']>=5 for r in cell),
            own_qfull_le_minus5=sum(r['own_qfull_delta']<=-5 for r in cell),
            conditional_own_qfull=float(np.mean([r['own_qfull_delta'] for r in cell if r['first_tick']>=0]))
                if any(r['first_tick']>=0 for r in cell) else None)
    summary['directional_gate']=bool(profile=='dev' and all(
        summary['cases'][case]['own_qfull_delta']>=5 and
        summary['cases'][case]['own_qfull_ci95'][0]>0 and
        sum(x>0 for x in summary['cases'][case]['own_qfull_by_seed'].values())>=4
        for case in TARGET))
    save(out/'summary.json',summary)
    manifest=dict(status='COMPLETED',profile=profile,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_run=str(SOURCE),
                  source_manifest_sha256=sha(SOURCE/'manifest.json'),source_audit_sha256=sha(AUDIT),
                  source_selection_sha256=sha(SOURCE/'selection.json'),
                  source_episodes_sha256=sha(SOURCE/'episodes.csv'),
                  plan_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),checkpoint_hashes=checkpoint_hashes,
                  scenarios=scenarios,forecast_anchor=ANCHOR,bootstrap_seed=BOOTSTRAP_SEED,
                  wall_seconds=time.monotonic()-start,
                  artifact_hashes={name:sha(out/name) for name in ('rollout_episodes.csv','actions.csv','pairs.csv',
                                                                   'q_banks.jsonl','summary.json')})
    save(out/'manifest.json',manifest)
    print(json.dumps(summary))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('smoke','dev'),required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    run(args.profile,out)
if __name__=='__main__':main()
