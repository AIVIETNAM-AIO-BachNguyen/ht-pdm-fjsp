"""Independent exact V17/rollout first-divergence, own/risk Q and aggregate audit."""
from __future__ import annotations
import argparse,csv,hashlib,json,random,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON,FrameworkEnv,catalogue,configs,vector_observation
from ht_pdm_fjsp.maintenance_framework_v17_policy import FullMatchingActor
from ht_pdm_fjsp.maintenance_framework_v7_campaign import V7RuleController
from ht_pdm_fjsp.maintenance_request_policies import request_action,risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings,state_from_dict,submit_requests,transition

ROOT=Path(__file__).resolve().parent.parent
TARGET=('interchangeable_scale1','reciprocal_scale2')
ACTIONS=catalogue();METRICS=('objective','maintenance_cost','unavailability_cost','failure_cost','tail_cost',
'H_availability','H_failures','queued_cm_ticks','queued_pm_ticks','H_pending','end_pending','service_starts','requests_created','requests_completed')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert np.isclose(float(a),float(b),rtol=0,atol=1e-8),(a,b)
def legal_state(cfg,state,tick):
    subs=request_action(cfg,state,tick,'threshold')
    pending=submit_requests(cfg,state,tick,subs)
    legal=sorted(ACTIONS.index(a) for a in legal_matchings(cfg,pending))
    return subs,pending,legal,vector_observation(cfg,pending,tick)
def action(model,obs,legal,uniform=None):
    with torch.no_grad():
        logits=model(torch.as_tensor(obs).reshape(1,56))[0]
        full=torch.full_like(logits,-1e9);full[legal]=logits[legal]
        p=torch.softmax(full,dim=0).numpy()
    if uniform is None:return int(np.argmax(p))
    total=0.
    for index in legal:
        total+=float(p[index])
        if uniform<total:return index
    return legal[-1]
def classify(x,y):
    a,b=ACTIONS[x],ACTIONS[y]
    if a==b:return 'same'
    if a and not b:return 'actor_service_rollout_wait'
    if b and not a:return 'actor_wait_rollout_service'
    if len(a)!=len(b):return 'actor_higher_cardinality' if len(a)>len(b) else 'actor_lower_cardinality'
    return 'machine_set' if {m for m,_ in a}!={m for m,_ in b} else 'assignment'
def replay(case,physical,model,anchor,deadline):
    cfg=configs()[case];env=FrameworkEnv(cfg);env.reset(seed=physical)
    rule=V7RuleController('rollout',anchor,deadline) if model is None else None
    actions=[];states=[]
    for tick in range(HORIZON):
        deadline();state=env.inner.state;states.append(asdict(state))
        subs,_,legal,obs=legal_state(cfg,state,tick)
        if model is None:
            supplied,matching=rule.choose(cfg,state,tick,case=case,seed=physical)
            assert supplied==subs;chosen=ACTIONS.index(matching)
        else:chosen=action(model,obs,legal)
        assert chosen in legal
        actions.append(chosen);env.step(chosen)
    return actions,states,env.episode_metrics()
def scenario(seed,cfg,tick,n):
    rng=random.Random(seed);tapes=[]
    for _ in range(n):
        events=[[rng.random()<cfg.failure_probability for _ in range(cfg.machines)]
                for _ in range(tick,cfg.horizon)]
        uniforms=[rng.random() for _ in range(tick,HORIZON)]
        tapes.append(dict(events=events,uniforms=uniforms))
    return tapes
def cost(cfg,original,tick,root,tape,model,mode,deadline):
    state=original;short=total=0.;steps=0
    for offset,t in enumerate(range(tick,cfg.horizon)):
        deadline();subs,pending,legal,obs=legal_state(cfg,state,t)
        if offset==0:chosen=root
        elif mode=='risk' or t>=HORIZON:chosen=ACTIONS.index(risk_matching(cfg,pending,t))
        else:chosen=action(model,obs,legal,tape['uniforms'][offset])
        assert chosen in legal
        state,c,_=transition(cfg,state,t,subs,ACTIONS[chosen],tape['events'][offset])
        total+=c;steps+=1
        if offset<6:short+=c
    return (short,total),steps
def bootstrap(cube):
    rng=np.random.default_rng(73070001);out=np.empty(20000)
    for offset in tqdm(range(0,20000,1000),desc='Independent V17 divergence CI',unit='chunk'):
        seed=rng.integers(0,5,size=(1000,5));physical=rng.integers(0,16,size=(1000,5,16))
        out[offset:offset+1000]=np.take_along_axis(cube[seed],physical,axis=2).mean((1,2))
    return np.quantile(out,[.025,.975])
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();start=time.monotonic();limit=1800
    def deadline():
        if time.monotonic()-start>limit:raise TimeoutError('independent divergence audit')
    run=args.run_dir.resolve();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest=json.loads((run/'manifest.json').read_text());profile=manifest['profile']
    assert manifest['status']=='COMPLETED' and profile in ('smoke','dev')
    source=Path(manifest['source_run']);source_manifest=json.loads((source/'manifest.json').read_text())
    assert source_manifest['status']=='COMPLETED' and source_manifest['profile']=='dev'
    assert sha(source/'manifest.json')==manifest['source_manifest_sha256']
    assert sha(source/'selection.json')==manifest['source_selection_sha256']
    assert sha(source/'episodes.csv')==manifest['source_episodes_sha256']
    main_audit=ROOT/'research_result/framework_v17_audit_20261021T031626Z/independent_audit.json'
    assert sha(main_audit)==manifest['source_audit_sha256']
    assert json.loads(main_audit.read_text())['status']=='PASS'
    assert sha(ROOT/'research_result/v17_first_divergence_continuation_plan_vi.md')==manifest['plan_sha256']
    assert sha(ROOT/'research_result/diagnose_v17_first_divergence.py')==manifest['script_sha256']
    for name,digest in source_manifest['artifact_hashes'].items():assert sha(source/name)==digest
    for name,digest in manifest['artifact_hashes'].items():assert sha(run/name)==digest
    seeds=tuple(range(73000000,73000005)) if profile=='dev' else (73000000,)
    physicals=tuple(range(73011000,73011016)) if profile=='dev' else (73011000,)
    count=16 if profile=='dev' else 1
    assert manifest['scenarios']==count and manifest['forecast_anchor']==73070000
    selection=json.loads((source/'selection.json').read_text());models={}
    for train in seeds:
        item=selection['rank_projection'][str(train)];cp=source/item['checkpoint']
        assert sha(cp)==item['sha256']==manifest['checkpoint_hashes'][str(train)]
        model=FullMatchingActor();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['state_dict']);model.eval();models[train]=model
    reference={(r['controller'],r['case'],r['train_seed'],int(r['seed'])):r for r in read(source/'episodes.csv')}
    action_rows=read(run/'actions.csv');actions={}
    for r in action_rows:actions.setdefault((r['controller'],r['case'],r['train_seed'],int(r['physical_seed'])),[]).append((int(r['tick']),int(r['action'])))
    assert len(actions)==len(TARGET)*len(physicals)*(len(seeds)+1)
    for sequence in actions.values():assert [t for t,a in sequence]==list(range(HORIZON))
    rollout_rows={(r['case'],int(r['physical_seed'])):r for r in read(run/'rollout_episodes.csv')}
    pairs={(r['case'],int(r['train_seed']),int(r['physical_seed'])):r for r in read(run/'pairs.csv')}
    banks={(r['case'],int(r['train_seed']),int(r['physical_seed'])):r for r in map(json.loads,(run/'q_banks.jsonl').open())}
    assert len(rollout_rows)==len(TARGET)*len(physicals)
    assert len(pairs)==len(TARGET)*len(physicals)*len(seeds)
    replayed_roll={};forecast_steps=0
    for case in TARGET:
        for physical in physicals:
            seq,states,metrics=replay(case,physical,None,source_manifest['registry']['forecast_seed'],deadline)
            assert seq==[a for t,a in actions[('rollout',case,'',physical)]]
            for name in METRICS:
                close(metrics[name],rollout_rows[(case,physical)][name])
                close(metrics[name],reference[('rollout',case,'',physical)][name])
            replayed_roll[(case,physical)]=(seq,states,metrics)
    for case,train,physical in tqdm(pairs,desc='Independent V17 divergence and Q bank',unit='pair'):
        deadline();r=pairs[(case,train,physical)];cfg=configs()[case];model=models[train]
        seq,states,metrics=replay(case,physical,model,source_manifest['registry']['forecast_seed'],deadline)
        assert seq==[a for t,a in actions[('rank_projection',case,str(train),physical)]]
        for name in METRICS:close(metrics[name],reference[('rank_projection',case,str(train),physical)][name])
        other,other_states,baseline=replayed_roll[(case,physical)]
        first=next((t for t,(a,b) in enumerate(zip(seq,other)) if a!=b),None)
        assert int(r['first_tick'])==(-1 if first is None else first)
        assert r['category']==('same' if first is None else classify(seq[first],other[first]))
        assert int(r['actor_duration'])==(0 if first is None else sum(cfg.service_time[m][j] for m,j in ACTIONS[seq[first]]))
        assert int(r['rollout_duration'])==(0 if first is None else sum(cfg.service_time[m][j] for m,j in ACTIONS[other[first]]))
        for key,name in (('actual_cost_delta','objective'),('actual_failure_delta','H_failures'),
                         ('actual_pm_queue_delta','queued_pm_ticks'),('actual_cm_queue_delta','queued_cm_ticks')):
            close(r[key],metrics[name]-baseline[name])
        if first is None:
            assert (case,train,physical) not in banks
            assert int(r['forecast_transitions'])==0
            continue
        assert all(seq[t]==other[t] and states[t]==other_states[t] for t in range(first))
        assert states[first]==other_states[first]
        b=banks[(case,train,physical)];assert b['original_state']==json.loads(json.dumps(states[first])) and int(b['tick'])==first
        seed=role_seed(73070000,f'v17_first_divergence:{case}:{train}:{physical}:{first}')
        assert seed==int(r['forecast_seed'])==int(b['forecast_seed'])
        tapes=scenario(seed,cfg,first,count);assert tapes==b['tapes']
        assert hashlib.sha256(json.dumps(tapes,sort_keys=True).encode()).hexdigest()==b['bank_sha256']
        scores={'own':[],'risk':[]};steps=0
        for tape,reported in zip(tapes,b['scenario_scores']):
            for mode in ('own','risk'):
                results={}
                for label,root in (('actor',seq[first]),('rollout',other[first])):
                    x,n=cost(cfg,state_from_dict(states[first]),first,root,tape,model,mode,deadline)
                    results[label]=x;steps+=n
                    for got,want in zip(x,reported[f'{mode}_{label}']):close(got,want)
                scores[mode].append((results['actor'][0]-results['rollout'][0],
                                      results['actor'][1]-results['rollout'][1]))
        assert steps==int(r['forecast_transitions']);forecast_steps+=steps
        for mode in ('own','risk'):
            close(r[f'{mode}_q6_delta'],np.mean([x[0] for x in scores[mode]]))
            close(r[f'{mode}_qfull_delta'],np.mean([x[1] for x in scores[mode]]))
        _,_,legal,obs=legal_state(cfg,state_from_dict(states[first]),first)
        with torch.no_grad():logits=model(torch.as_tensor(obs).reshape(1,56))[0].numpy()
        mask=np.full_like(logits,-1e9);mask[legal]=logits[legal];p=np.exp(mask-np.max(mask));p/=p.sum()
        close(r['actor_probability'],p[seq[first]]);close(r['rollout_probability'],p[other[first]])
        close(r['logit_margin'],logits[seq[first]]-logits[other[first]])
    summary=json.loads((run/'summary.json').read_text())
    assert summary['pairs']==len(pairs) and summary['divergent']==len(banks)
    assert summary['forecast_transitions']==forecast_steps
    for case in TARGET:
        cell=[r for (c,t,s),r in pairs.items() if c==case];saved=summary['cases'][case]
        assert saved['pairs']==len(cell)
        for name,want in saved['categories'].items():assert want==sum(r['category']==name for r in cell)
        for key,value in (('actual_cost_delta','actual_cost_delta'),('actual_failure_delta','actual_failure_delta'),
                          ('own_qfull_delta','own_qfull_delta'),('own_q6_delta','own_q6_delta'),
                          ('risk_qfull_delta','risk_qfull_delta')):
            close(saved[key],np.mean([float(r[value]) for r in cell]))
        if profile=='dev':
            cube=np.asarray([[float(pairs[(case,seed,physical)]['own_qfull_delta'])
                              for physical in physicals] for seed in seeds])
            for got,want in zip(saved['own_qfull_ci95'],bootstrap(cube)):close(got,want)
            for i,seed in enumerate(seeds):close(saved['own_qfull_by_seed'][str(seed)],cube[i].mean())
        else:assert saved['own_qfull_ci95'] is None
    gate=bool(profile=='dev' and all(summary['cases'][case]['own_qfull_delta']>=5 and
        summary['cases'][case]['own_qfull_ci95'][0]>0 and
        sum(x>0 for x in summary['cases'][case]['own_qfull_by_seed'].values())>=4 for case in TARGET))
    assert summary['directional_gate']==gate
    report=dict(status='PASS',run=str(run),profile=profile,pairs=len(pairs),banks=len(banks),
                forecast_transitions=forecast_steps,episode_replays=len(pairs)+len(rollout_rows),
                wall_seconds=time.monotonic()-start)
    (out/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
