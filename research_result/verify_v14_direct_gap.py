"""Independent replay of frozen V14 direct rollout-gap diagnostic."""
from __future__ import annotations
import argparse,csv,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON,configs
from ht_pdm_fjsp.maintenance_framework_v14_campaign import load_checkpoint
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
from ht_pdm_fjsp.maintenance_request_policies import risk_matching
from ht_pdm_fjsp.maintenance_requests import state_from_dict,transition
from research_result import verify_v19_first_divergence as independent

ROOT=Path(__file__).resolve().parent.parent
SOURCE=independent.SOURCE;SOURCE_AUDIT=independent.SOURCE_AUDIT
V14=independent.V14;V14_AUDIT=independent.V14_AUDIT
PLAN=ROOT/'research_result/v14_direct_gap_plan_vi.md'
CASES=('interchangeable_scale1','reciprocal_scale2');TRAIN=tuple(range(72600000,72600005))
PHYSICAL=tuple(range(73411000,73411016));ANCHOR=73670000;BOOTSTRAP=73670001
ACTIONS=independent.ACTIONS;METRICS=independent.METRICS
CATEGORIES=('same','v14_service_other_wait','v14_wait_other_service',
            'v14_higher_cardinality','v14_lower_cardinality','machine_set','assignment')
sha=independent.sha;read=independent.read;close=independent.close;current=independent.current

def classify(left,right):return independent.classify(left,right).replace('temporal_','v14_')
def scenario(seed,cfg,tick,count):
    rng=random.Random(seed)
    return [[[rng.random()<cfg.failure_probability for _ in range(cfg.machines)]
             for _ in range(tick,cfg.horizon)] for _ in range(count)]
def root_cost(cfg,original,tick,root,tape,actor,mode,deadline):
    state=original;short=total=0.;steps=0
    for offset,t in enumerate(range(tick,cfg.horizon)):
        deadline();subs,admitted,legal,obs=current(cfg,state,t)
        if offset==0:action=root
        elif t>=HORIZON or mode=='risk':action=ACTIONS.index(risk_matching(cfg,admitted,t))
        else:action=independent.choice(actor,obs,legal)
        assert action in legal
        state,cost,_=transition(cfg,state,t,subs,ACTIONS[action],tape[offset])
        total+=cost;steps+=1
        if offset<6:short+=cost
    return (short,total),steps
def bootstrap(cube):
    rng=np.random.default_rng(BOOTSTRAP);draws=np.empty(20000)
    for first in range(0,20000,1000):
        seed=rng.integers(0,5,size=(1000,5));physical=rng.integers(0,16,size=(1000,5,16))
        draws[first:first+1000]=np.take_along_axis(cube[seed],physical,axis=2).mean((1,2))
    return np.quantile(draws,[.025,.975]).tolist()
def audit(run):
    start=time.monotonic();limit=1800
    def deadline():
        if time.monotonic()-start>limit:raise TimeoutError('V14 direct-gap independent audit')
    torch.set_num_threads(1)
    m=json.loads((run/'manifest.json').read_text());profile=m['profile']
    assert m['status']=='COMPLETED' and profile in ('smoke','dev')
    assert sha(PLAN)==m['plan_sha256']
    assert sha(ROOT/'research_result/diagnose_v14_direct_gap.py')==m['script_sha256']
    assert sha(Path(__file__))==m['verifier_sha256']
    assert sha(ROOT/'research_result/diagnose_v19_first_divergence.py')==m['base_script_sha256']
    assert Path(m['source_run']).resolve()==SOURCE and Path(m['v14_run']).resolve()==V14
    assert sha(SOURCE/'manifest.json')==m['source_manifest_sha256']
    assert sha(SOURCE_AUDIT)==m['source_audit_sha256']
    assert sha(SOURCE/'episodes.csv')==m['source_episodes_sha256']
    assert sha(V14/'manifest.json')==m['v14_manifest_sha256']
    assert sha(V14_AUDIT)==m['v14_audit_sha256']
    assert sha(V14/'selection.json')==m['v14_selection_sha256']
    assert json.loads(SOURCE_AUDIT.read_text())['status']==json.loads(V14_AUDIT.read_text())['status']=='PASS'
    source=json.loads((SOURCE/'manifest.json').read_text());assert source['status']=='COMPLETED'
    for name,digest in source['artifact_hashes'].items():assert sha(SOURCE/name)==digest
    for name,digest in m['artifact_hashes'].items():assert sha(run/name)==digest
    used=set()
    for registry in (ROOT/'configs').glob('*seed_registry.json'):
        used|=registered_integers(json.loads(registry.read_text()))
    from ht_pdm_fjsp.maintenance_framework_v19_campaign import episode_seed as v19_seed
    from ht_pdm_fjsp.maintenance_framework_v14_campaign import episode_seed as v14_seed
    used|={v19_seed(s,i,c,k) for s in range(73400000,73400005) for i in range(1,7) for c in configs() for k in range(16)}
    used|={v14_seed(s,i,c,k) for s in TRAIN for i in range(1,7) for c in configs() for k in range(16)}
    assert not {ANCHOR,BOOTSTRAP}&used and ANCHOR!=BOOTSTRAP
    train=TRAIN if profile=='dev' else TRAIN[:1]
    physicals=PHYSICAL if profile=='dev' else PHYSICAL[2:3]
    count=16 if profile=='dev' else 1
    assert m['forecast_anchor']==ANCHOR and m['bootstrap_seed']==BOOTSTRAP and m['scenarios']==count
    selection=json.loads((V14/'selection.json').read_text());models={}
    for seed in train:
        item=selection['competition'][str(seed)];cp=V14/item['checkpoint']
        assert sha(cp)==item['sha256']==m['checkpoint_hashes'][str(seed)]
        models[seed]=load_checkpoint(cp)
    originals={(r['controller'],r['case'],r['train_seed'],int(r['seed'])):r for r in read(SOURCE/'episodes.csv')}
    recorded={(r['controller'],r['case'],r['train_seed'],int(r['physical_seed'])):r for r in read(run/'episodes.csv')}
    actions={}
    for row in read(run/'actions.csv'):
        key=(row['controller'],row['case'],row['train_seed'],int(row['physical_seed']))
        actions.setdefault(key,[]).append((int(row['tick']),int(row['action'])))
    specs=[('rollout',case,'',physical,None) for case in CASES for physical in physicals]
    specs += [('competition',case,str(seed),physical,models[seed]) for case in CASES for seed in train for physical in physicals]
    assert len(specs)==len(recorded)==len(actions)==m['counts']['episodes']
    traces={}
    for arm,case,seed,physical,model in tqdm(specs,desc='Independent V14 direct episode replay',unit='episode'):
        deadline();key=(arm,case,seed,physical)
        seq,states,metrics=independent.replay(case,physical,arm,model,source['registry']['forecast_seed'],deadline)
        assert actions[key]==list(enumerate(seq))
        for name in METRICS:
            close(metrics[name],recorded[key][name],f'run {name}')
            close(metrics[name],originals[key][name],f'source {name}')
        traces[key]=(seq,states,metrics)
    pairs={(r['case'],int(r['train_seed']),int(r['physical_seed'])):r for r in read(run/'q_pairs.csv')}
    banks={(r['case'],int(r['train_seed']),int(r['physical_seed'])):r for r in map(json.loads,(run/'q_banks.jsonl').open())}
    assert len(pairs)==len(CASES)*len(train)*len(physicals)==m['counts']['q_pairs']
    assert len(banks)==m['counts']['q_banks']
    steps=0
    for case,seed,physical in tqdm(pairs,desc='Independent V14 first roots/full Q',unit='pair'):
        deadline();r=pairs[(case,seed,physical)];cfg=configs()[case]
        assert case in CASES and seed in train and physical in physicals
        left,lstates,lm=traces[('competition',case,str(seed),physical)]
        right,rstates,rm=traces[('rollout',case,'',physical)]
        first=next((t for t,(a,b) in enumerate(zip(left,right)) if a!=b),None)
        assert int(r['first_tick'])==(-1 if first is None else first)
        assert r['first_state']==('' if first is None else json.dumps(lstates[first],sort_keys=True))
        assert r['category']==('same' if first is None else classify(left[first],right[first]))
        assert int(r['v14_action'])==(-1 if first is None else left[first])
        assert int(r['rollout_action'])==(-1 if first is None else right[first])
        for key,name in (('actual_cost_delta','objective'),('actual_failure_delta','H_failures'),
                         ('actual_pm_queue_delta','queued_pm_ticks'),('actual_cm_queue_delta','queued_cm_ticks'),
                         ('actual_service_start_delta','service_starts')):
            close(r[key],lm[name]-rm[name],key)
        if first is None:
            assert (case,seed,physical) not in banks and int(r['forecast_transitions'])==0
            continue
        assert lstates[first]==rstates[first]
        assert all(left[t]==right[t] and lstates[t]==rstates[t] for t in range(first))
        b=banks[(case,seed,physical)];assert int(b['tick'])==first and b['original_state']==json.loads(json.dumps(lstates[first]))
        state=state_from_dict(lstates[first]);_,admitted,legal,_=current(cfg,state,first)
        context=dict(tick=first,ages=list(state.physical.ages),remaining=list(state.physical.remaining),
                     assigned=list(state.physical.assigned),failed=list(state.physical.failed),
                     queued=[i for i,x in enumerate(admitted.requests) if x and x.started_at is None],legal=legal)
        assert b['context']==json.loads(json.dumps(context))
        bank_seed=role_seed(ANCHOR,f'v14_direct_gap:{case}:{seed}:{physical}:{first}')
        assert bank_seed==int(r['forecast_seed'])==int(b['forecast_seed'])
        tapes=scenario(bank_seed,cfg,first,count)
        assert tapes==b['tapes'] and b['bank_sha256']==hashlib.sha256(json.dumps(tapes,sort_keys=True).encode()).hexdigest()
        values={mode:[] for mode in ('own','risk')};local_steps=0
        assert len(b['scenario_scores'])==count
        for tape,reported in zip(tapes,b['scenario_scores']):
            for mode in ('own','risk'):
                found={}
                for label,root in (('v14',left[first]),('rollout',right[first])):
                    value,n=root_cost(cfg,state,first,root,tape,models[seed],mode,deadline)
                    found[label]=value;local_steps+=n
                    for got,want in zip(value,reported[f'{mode}_{label}']):close(got,want,'scenario score')
                values[mode].append((found['v14'][0]-found['rollout'][0],found['v14'][1]-found['rollout'][1]))
        assert local_steps==int(r['forecast_transitions'])==count*4*(cfg.horizon-first)
        steps+=local_steps
        for mode in ('own','risk'):
            close(r[f'{mode}_q6_delta'],np.mean([v[0] for v in values[mode]]),'Q6')
            close(r[f'{mode}_qfull_delta'],np.mean([v[1] for v in values[mode]]),'Qfull')
    assert steps==m['counts']['forecast_transitions']
    summary=json.loads((run/'summary.json').read_text())
    assert summary['q_pairs']==len(pairs) and summary['q_banks']==len(banks)
    assert summary['forecast_transitions']==steps
    for case in CASES:
        cell=[r for (c,s,p),r in pairs.items() if c==case];item=summary['cases'][case]
        assert item['pairs']==len(cell)
        for category in CATEGORIES:assert item['categories'][category]==sum(r['category']==category for r in cell)
        ticks=[int(r['first_tick']) for r in cell if int(r['first_tick'])>=0]
        if ticks:close(item['first_tick_mean'],np.mean(ticks),'first tick')
        else:assert item['first_tick_mean'] is None
        for key in ('actual_cost_delta','actual_failure_delta','actual_pm_queue_delta','actual_cm_queue_delta',
                    'actual_service_start_delta','own_q6_delta','own_qfull_delta','risk_q6_delta','risk_qfull_delta'):
            close(item[key],np.mean([float(r[key]) for r in cell]),key)
        for mode in ('own','risk'):
            divergent=[float(r[f'{mode}_qfull_delta']) for r in cell if int(r['first_tick'])>=0]
            if divergent:close(item[f'conditional_{mode}_qfull'],np.mean(divergent),'conditional Q')
            else:assert item[f'conditional_{mode}_qfull'] is None
            cube=np.asarray([[float(pairs[(case,seed,physical)][f'{mode}_qfull_delta'])
                              for physical in physicals] for seed in train])
            for i,seed in enumerate(train):close(item[f'{mode}_qfull_by_seed'][str(seed)],cube[i].mean(),'seed Q')
            if profile=='dev':
                for got,want in zip(item[f'{mode}_qfull_ci95'],bootstrap(cube)):close(got,want,'bootstrap')
            else:assert item[f'{mode}_qfull_ci95'] is None
        if profile=='dev':
            assert item['material_root_failure']==(item['own_qfull_delta']>=5 and item['own_qfull_ci95'][0]>0)
            assert item['continuation_mismatch']==(item['risk_qfull_ci95'][0]>0 and item['own_qfull_delta']<=0)
        else:assert item['material_root_failure'] is item['continuation_mismatch'] is None
    return dict(status='PASS',run=str(run),profile=profile,episode_replays=len(specs),
                q_pairs=len(pairs),full_q_banks=len(banks),forecast_transitions=steps,
                wall_seconds=time.monotonic()-start)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    report=audit(a.run_dir.resolve())
    (out/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
