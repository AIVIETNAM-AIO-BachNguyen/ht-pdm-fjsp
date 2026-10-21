"""Independent exact V19 first-divergence trajectory and dual-continuation audit."""
from __future__ import annotations
import argparse,csv,hashlib,json,random,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON,FrameworkEnv,catalogue,configs,vector_observation
from ht_pdm_fjsp.maintenance_framework_v19_policy import TemporalMatchingActor
from ht_pdm_fjsp.maintenance_framework_v14_campaign import V14Controller,load_checkpoint as load_v14
from ht_pdm_fjsp.maintenance_framework_v7_campaign import V7RuleController
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
from ht_pdm_fjsp.maintenance_request_policies import request_action,risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings,state_from_dict,submit_requests,transition

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'research_result/framework_v19_dev_20261021T075021Z'
SOURCE_AUDIT=ROOT/'research_result/framework_v19_audit_20261021T085652Z/independent_audit.json'
V14=ROOT/'research_result/framework_v14_dev_20261020T185303Z'
V14_AUDIT=ROOT/'research_result/framework_v14_audit_20261020T200552Z/independent_audit.json'
PLAN=ROOT/'research_result/v19_first_divergence_plan_vi.md'
TARGET=('interchangeable_scale1','reciprocal_scale2','reciprocal_scale3')
TRAIN=tuple(range(73400000,73400005));V14_TRAIN=tuple(range(72600000,72600005))
PHYSICAL=tuple(range(73411000,73411016));ANCHOR=73570000;BOOTSTRAP=73570001
ACTIONS=catalogue()
METRICS=('objective','maintenance_cost','unavailability_cost','failure_cost','tail_cost',
'H_availability','H_failures','queued_cm_ticks','queued_pm_ticks','H_pending','end_pending',
'service_starts','requests_created','requests_completed')
CATEGORIES=('same','temporal_service_other_wait','temporal_wait_other_service',
'temporal_higher_cardinality','temporal_lower_cardinality','machine_set','assignment')
COMPARISONS=tuple((case,other) for case in ('interchangeable_scale1','reciprocal_scale2')
                  for other in ('snapshot_deficit','rollout'))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def close(a,b,label):
    if not np.isclose(float(a),float(b),rtol=0,atol=1e-8):raise AssertionError(f'{label}: {a} != {b}')
def current(cfg,state,tick):
    subs=request_action(cfg,state,tick,'threshold')
    admitted=submit_requests(cfg,state,tick,subs)
    legal=sorted(ACTIONS.index(a) for a in legal_matchings(cfg,admitted))
    obs=vector_observation(cfg,admitted,tick)
    return subs,admitted,legal,obs
def choice(model,obs,legal,uniform=None):
    with torch.no_grad():
        logits=model(torch.as_tensor(obs).reshape(1,56))[0]
        masked=torch.full_like(logits,-1e9);masked[legal]=logits[legal]
        p=torch.softmax(masked,dim=0).numpy()
    if uniform is None:return int(np.argmax(p))
    acc=0.
    for action in legal:
        acc+=float(p[action])
        if uniform<acc:return action
    return legal[-1]
def classify(left,right):
    a,b=ACTIONS[left],ACTIONS[right]
    if a==b:return 'same'
    if a and not b:return 'temporal_service_other_wait'
    if b and not a:return 'temporal_wait_other_service'
    if len(a)!=len(b):return 'temporal_higher_cardinality' if len(a)>len(b) else 'temporal_lower_cardinality'
    return 'machine_set' if {m for m,_ in a}!={m for m,_ in b} else 'assignment'
def replay(case,physical,arm,model,anchor,deadline):
    cfg=configs()[case];env=FrameworkEnv(cfg);env.reset(seed=physical)
    controller=V7RuleController('rollout',anchor,deadline) if arm=='rollout' else (V14Controller(model,'competition') if arm=='competition' else None)
    seq=[];states=[]
    for tick in range(HORIZON):
        deadline();state=env.inner.state;states.append(asdict(state))
        subs,_,legal,obs=current(cfg,state,tick)
        if controller is None:action=choice(model,obs,legal)
        else:
            supplied,matching=controller.choose(cfg,state,tick,case=case,seed=physical)
            assert supplied==subs;action=ACTIONS.index(matching)
        assert action in legal
        seq.append(action);env.step(action)
    return seq,states,env.episode_metrics()
def scenario(seed,cfg,tick,count):
    rng=random.Random(seed);tapes=[]
    for _ in range(count):
        events=[[rng.random()<cfg.failure_probability for _ in range(cfg.machines)] for _ in range(tick,cfg.horizon)]
        uniforms=[rng.random() for _ in range(tick,HORIZON)]
        tapes.append(dict(events=events,uniforms=uniforms))
    return tapes
def root_cost(cfg,original,tick,root,tape,actor,mode,deadline):
    state=original;short=total=0.;steps=0
    for offset,t in enumerate(range(tick,cfg.horizon)):
        deadline();subs,admitted,legal,obs=current(cfg,state,t)
        if offset==0:action=root
        elif t>=HORIZON or mode=='risk':action=ACTIONS.index(risk_matching(cfg,admitted,t))
        else:action=choice(actor,obs,legal,tape['uniforms'][offset])
        assert action in legal
        state,cost,_=transition(cfg,state,t,subs,ACTIONS[action],tape['events'][offset])
        total+=cost;steps+=1
        if offset<6:short+=cost
    return (short,total),steps
def bootstrap(cube):
    rng=np.random.default_rng(BOOTSTRAP);draws=np.empty(20000)
    for offset in range(0,20000,1000):
        seed=rng.integers(0,5,size=(1000,5));physical=rng.integers(0,16,size=(1000,5,16))
        draws[offset:offset+1000]=np.take_along_axis(cube[seed],physical,axis=2).mean((1,2))
    return np.quantile(draws,[.025,.975]).tolist()
def audit(run):
    start=time.monotonic();limit=1800
    def deadline():
        if time.monotonic()-start>limit:raise TimeoutError('independent V19 divergence audit')
    torch.set_num_threads(1)
    manifest=json.loads((run/'manifest.json').read_text());profile=manifest['profile']
    assert manifest['status']=='COMPLETED' and profile in ('smoke','dev')
    assert sha(PLAN)==manifest['plan_sha256']
    assert sha(ROOT/'research_result/diagnose_v19_first_divergence.py')==manifest['script_sha256']
    assert sha(Path(__file__))==manifest['verifier_sha256']
    assert Path(manifest['source_run']).resolve()==SOURCE and Path(manifest['v14_run']).resolve()==V14
    assert sha(SOURCE/'manifest.json')==manifest['source_manifest_sha256']
    assert sha(SOURCE_AUDIT)==manifest['source_audit_sha256']
    assert sha(SOURCE/'selection.json')==manifest['source_selection_sha256']
    assert sha(SOURCE/'episodes.csv')==manifest['source_episodes_sha256']
    assert sha(V14/'manifest.json')==manifest['v14_manifest_sha256']
    assert sha(V14_AUDIT)==manifest['v14_audit_sha256']
    assert sha(V14/'selection.json')==manifest['v14_selection_sha256']
    assert json.loads(SOURCE_AUDIT.read_text())['status']=='PASS'
    assert json.loads(V14_AUDIT.read_text())['status']=='PASS'
    source=json.loads((SOURCE/'manifest.json').read_text());assert source['status']=='COMPLETED'
    for name,digest in source['artifact_hashes'].items():assert sha(SOURCE/name)==digest
    for name,digest in manifest['artifact_hashes'].items():assert sha(run/name)==digest
    used=set()
    for registry in (ROOT/'configs').glob('*seed_registry.json'):
        used|=registered_integers(json.loads(registry.read_text()))
    from ht_pdm_fjsp.maintenance_framework_v19_campaign import episode_seed
    used|={episode_seed(seed,iteration,case,index) for seed in TRAIN for iteration in range(1,7)
           for case in configs() for index in range(16)}
    assert not {ANCHOR,BOOTSTRAP}&used and ANCHOR!=BOOTSTRAP
    train=TRAIN if profile=='dev' else TRAIN[:1]
    v14_train=V14_TRAIN if profile=='dev' else V14_TRAIN[:1]
    physicals=PHYSICAL if profile=='dev' else PHYSICAL[:1]
    count=16 if profile=='dev' else 1
    assert manifest['forecast_anchor']==ANCHOR and manifest['bootstrap_seed']==BOOTSTRAP and manifest['scenarios']==count
    selection=json.loads((SOURCE/'selection.json').read_text());v14_selection=json.loads((V14/'selection.json').read_text())
    models={};v14_models={}
    for arm in ('temporal_deficit','snapshot_deficit'):
        for seed in train:
            item=selection[arm][str(seed)];cp=SOURCE/item['checkpoint']
            assert sha(cp)==item['sha256']==manifest['checkpoint_hashes'][f'{arm}/{seed}']
            actor=TemporalMatchingActor(arm);actor.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['state_dict']);actor.eval()
            models[(arm,seed)]=actor
    for seed in v14_train:
        item=v14_selection['competition'][str(seed)];cp=V14/item['checkpoint']
        assert sha(cp)==item['sha256']==manifest['checkpoint_hashes'][f'competition/{seed}']
        v14_models[seed]=load_v14(cp)
    originals={(r['controller'],r['case'],r['train_seed'],int(r['seed'])):r for r in read(SOURCE/'episodes.csv')}
    actions={}
    for row in read(run/'actions.csv'):
        key=(row['controller'],row['case'],row['train_seed'],int(row['physical_seed']))
        actions.setdefault(key,[]).append((int(row['tick']),int(row['action'])))
    specs=[('rollout',case,'',physical,None) for case in ('interchangeable_scale1','reciprocal_scale2') for physical in physicals]
    specs += [('competition','reciprocal_scale3',str(seed),physical,v14_models[seed])
              for seed in v14_train for physical in physicals]
    specs += [(arm,case,str(seed),physical,models[(arm,seed)])
              for arm in ('temporal_deficit','snapshot_deficit') for case in TARGET
              for seed in train for physical in physicals]
    assert len(actions)==len(specs)==manifest['counts']['episodes']
    recorded={(r['controller'],r['case'],r['train_seed'],int(r['physical_seed'])):r for r in read(run/'episodes.csv')}
    assert len(recorded)==len(specs)
    traces={}
    for arm,case,seed,physical,model in tqdm(specs,desc='Independent V19 diagnostic episode replay',unit='episode'):
        deadline();key=(arm,case,seed,physical)
        seq,states,metrics=replay(case,physical,arm,model,source['registry']['forecast_seed'],deadline)
        assert actions[key]==list(enumerate(seq))
        for name in METRICS:
            close(metrics[name],recorded[key][name],f'run episode {name}')
            close(metrics[name],originals[(arm,case,seed,physical)][name],f'source episode {name}')
        traces[key]=(seq,states,metrics)
    q_pairs={(r['case'],r['other'],int(r['train_seed']),int(r['physical_seed'])):r for r in read(run/'q_pairs.csv')}
    banks={(r['case'],r['other'],int(r['train_seed']),int(r['physical_seed'])):r
           for r in map(json.loads,(run/'q_banks.jsonl').open())}
    expected=len(COMPARISONS)*len(train)*len(physicals)
    assert len(q_pairs)==expected==manifest['counts']['q_pairs']
    assert len(banks)==manifest['counts']['q_banks']
    steps=0
    for case,other,seed,physical in tqdm(q_pairs,desc='Independent V19 divergence/full Q',unit='pair'):
        deadline();r=q_pairs[(case,other,seed,physical)];cfg=configs()[case]
        assert (case,other) in COMPARISONS and r['comparison']==f'{case}/{other}'
        actor=models[('temporal_deficit',seed)]
        left,left_states,left_metrics=traces[('temporal_deficit',case,str(seed),physical)]
        right,right_states,right_metrics=traces[(other,case,str(seed) if other=='snapshot_deficit' else '',physical)]
        first=next((t for t,(a,b) in enumerate(zip(left,right)) if a!=b),None)
        assert int(r['first_tick'])==(-1 if first is None else first)
        assert r['first_state']==('' if first is None else json.dumps(left_states[first],sort_keys=True))
        assert r['category']==('same' if first is None else classify(left[first],right[first]))
        assert int(r['temporal_action'])==(-1 if first is None else left[first])
        assert int(r['other_action'])==(-1 if first is None else right[first])
        assert int(r['temporal_duration'])==(0 if first is None else sum(cfg.service_time[m][j] for m,j in ACTIONS[left[first]]))
        assert int(r['other_duration'])==(0 if first is None else sum(cfg.service_time[m][j] for m,j in ACTIONS[right[first]]))
        for key,name in (('actual_cost_delta','objective'),('actual_failure_delta','H_failures'),
                         ('actual_pm_queue_delta','queued_pm_ticks'),('actual_cm_queue_delta','queued_cm_ticks'),('actual_service_start_delta','service_starts')):
            close(r[key],left_metrics[name]-right_metrics[name],key)
        if first is None:
            assert (case,other,seed,physical) not in banks and int(r['forecast_transitions'])==0
            continue
        assert all(left[t]==right[t] and left_states[t]==right_states[t] for t in range(first))
        assert left_states[first]==right_states[first]
        b=banks[(case,other,seed,physical)];assert int(b['tick'])==first and b['original_state']==json.loads(json.dumps(left_states[first]))
        current_state=state_from_dict(left_states[first]);_,admitted,legal,obs=current(cfg,current_state,first)
        context=dict(tick=first,ages=list(current_state.physical.ages),remaining=list(current_state.physical.remaining),
                     assigned=list(current_state.physical.assigned),failed=list(current_state.physical.failed),
                     queued=[i for i,x in enumerate(admitted.requests) if x and x.started_at is None],legal=legal)
        assert b['context']==json.loads(json.dumps(context))
        with torch.no_grad():logits=actor(torch.as_tensor(obs).reshape(1,56))[0].numpy()
        masked=np.full_like(logits,-1e9);masked[legal]=logits[legal];p=np.exp(masked-np.max(masked));p/=p.sum()
        close(r['temporal_probability'],p[left[first]],'temporal probability')
        close(r['other_probability'],p[right[first]],'other probability')
        close(r['logit_margin'],logits[left[first]]-logits[right[first]],'logit margin')
        seedbank=role_seed(ANCHOR,f'v19_first_divergence:{case}:{other}:{seed}:{physical}:{first}')
        assert seedbank==int(r['forecast_seed'])==int(b['forecast_seed'])
        tapes=scenario(seedbank,cfg,first,count)
        assert b['tapes']==tapes and b['bank_sha256']==hashlib.sha256(json.dumps(tapes,sort_keys=True).encode()).hexdigest()
        values={'own':[],'risk':[]};local_steps=0
        assert len(b['scenario_scores'])==count
        for tape,reported in zip(tapes,b['scenario_scores']):
            for mode in ('own','risk'):
                found={}
                for label,root in (('temporal',left[first]),('other',right[first])):
                    result,n=root_cost(cfg,current_state,first,root,tape,actor,mode,deadline)
                    found[label]=result;local_steps+=n
                    for got,want in zip(result,reported[f'{mode}_{label}']):close(got,want,'scenario root')
                values[mode].append((found['temporal'][0]-found['other'][0],found['temporal'][1]-found['other'][1]))
        assert local_steps==int(r['forecast_transitions'])==count*4*(cfg.horizon-first)
        steps+=local_steps
        for mode in ('own','risk'):
            close(r[f'{mode}_q6_delta'],np.mean([v[0] for v in values[mode]]),'Q6')
            close(r[f'{mode}_qfull_delta'],np.mean([v[1] for v in values[mode]]),'Qfull')
    v14_pairs={(int(r['train_seed']),int(r['v14_train_seed']),int(r['physical_seed'])):r for r in read(run/'v14_pairs.csv')}
    assert len(v14_pairs)==len(train)*len(v14_train)*len(physicals)==manifest['counts']['v14_pairs']
    for seed,v14_seed,physical in tqdm(v14_pairs,desc='Independent V19/V14 cross-product',unit='pair'):
        r=v14_pairs[(seed,v14_seed,physical)]
        left,left_states,left_metrics=traces[('temporal_deficit','reciprocal_scale3',str(seed),physical)]
        right,right_states,right_metrics=traces[('competition','reciprocal_scale3',str(v14_seed),physical)]
        first=next((t for t,(a,b) in enumerate(zip(left,right)) if a!=b),None)
        assert int(r['first_tick'])==(-1 if first is None else first)
        assert r['first_state']==('' if first is None else json.dumps(left_states[first],sort_keys=True))
        assert r['category']==('same' if first is None else classify(left[first],right[first]))
        assert int(r['temporal_action'])==(-1 if first is None else left[first])
        assert int(r['other_action'])==(-1 if first is None else right[first])
        if first is not None:
            assert left_states[first]==right_states[first]
            assert all(left[t]==right[t] and left_states[t]==right_states[t] for t in range(first))
        close(r['actual_cost_delta'],left_metrics['objective']-right_metrics['objective'],'V14 cost')
        close(r['actual_failure_delta'],left_metrics['H_failures']-right_metrics['H_failures'],'V14 failures')
    assert steps==manifest['counts']['forecast_transitions']
    summary=json.loads((run/'summary.json').read_text())
    assert summary['q_pairs']==len(q_pairs) and summary['q_banks']==len(banks)
    assert summary['v14_pairs']==len(v14_pairs) and summary['forecast_transitions']==steps
    for case,other in COMPARISONS:
        label=f'{case}/{other}';cell=[r for (c,o,s,p),r in q_pairs.items() if (c,o)==(case,other)]
        item=summary['comparisons'][label]
        assert item['pairs']==len(cell)
        for category in CATEGORIES:assert item['categories'][category]==sum(r['category']==category for r in cell)
        ticks=[int(r['first_tick']) for r in cell if int(r['first_tick'])>=0]
        if ticks:close(item['first_tick_mean'],np.mean(ticks),'first tick')
        else:assert item['first_tick_mean'] is None
        for key in ('actual_cost_delta','actual_failure_delta','actual_pm_queue_delta','actual_cm_queue_delta','actual_service_start_delta',
                    'own_q6_delta','risk_q6_delta','own_qfull_delta','risk_qfull_delta'):
            close(item[key],np.mean([float(r[key]) for r in cell]),key)
        for mode in ('own','risk'):
            values=[float(r[f'{mode}_qfull_delta']) for r in cell if int(r['first_tick'])>=0]
            if values:close(item[f'conditional_{mode}_qfull'],np.mean(values),'conditional Q')
            else:assert item[f'conditional_{mode}_qfull'] is None
            cube=np.asarray([[float(q_pairs[(case,other,seed,physical)][f'{mode}_qfull_delta'])
                              for physical in physicals] for seed in train])
            for i,seed in enumerate(train):close(item[f'{mode}_qfull_by_seed'][str(seed)],cube[i].mean(),'seed Q')
            if profile=='dev':
                for got,want in zip(item[f'{mode}_qfull_ci95'],bootstrap(cube)):close(got,want,'Q bootstrap')
            else:assert item[f'{mode}_qfull_ci95'] is None
    v=summary['v14_reciprocal3'];assert v['pairs']==len(v14_pairs)
    for category in CATEGORIES:assert v['categories'][category]==sum(r['category']==category for r in v14_pairs.values())
    close(v['actual_cost_delta'],np.mean([float(r['actual_cost_delta']) for r in v14_pairs.values()]),'V14 aggregate')
    close(v['actual_failure_delta'],np.mean([float(r['actual_failure_delta']) for r in v14_pairs.values()]),'V14 failure aggregate')
    if profile=='dev':
        flags={}
        for case in ('interchangeable_scale1','reciprocal_scale2'):
            x=summary['comparisons'][f'{case}/rollout']
            y=summary['comparisons'][f'{case}/snapshot_deficit']
            flags[f'{case}_continuation_mismatch']=(
                x['risk_qfull_delta']>=5 and x['risk_qfull_ci95'][0]>0 and x['own_qfull_delta']<=0)
            flags[f'{case}_own_root_failure']=(
                x['own_qfull_delta']>=5 and x['own_qfull_ci95'][0]>0)
            flags[f'{case}_temporal_root_alignment']=(
                y['own_qfull_delta']<=-5 and y['own_qfull_ci95'][1]<0)
        assert summary['interpretation_flags']==flags
    else:assert summary['interpretation_flags'] is None
    return dict(status='PASS',run=str(run),profile=profile,episode_replays=len(specs),
                q_pairs=len(q_pairs),full_q_banks=len(banks),forecast_transitions=steps,
                v14_cross_pairs=len(v14_pairs),wall_seconds=time.monotonic()-start)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    report=audit(args.run_dir.resolve())
    (out/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
