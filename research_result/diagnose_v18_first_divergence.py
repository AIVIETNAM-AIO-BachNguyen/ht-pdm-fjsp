"""V18 opened-DEV first-divergence and dual-continuation diagnostic."""
from __future__ import annotations
import argparse,csv,hashlib,json,random,subprocess,time
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON,FrameworkEnv,catalogue,configs
from ht_pdm_fjsp.maintenance_framework_v18_campaign import load_checkpoint,legal_present,policy_action
from ht_pdm_fjsp.maintenance_framework_v14_campaign import V14Controller,load_checkpoint as load_v14
from ht_pdm_fjsp.maintenance_framework_v7_campaign import V7RuleController
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
from ht_pdm_fjsp.maintenance_request_policies import risk_matching
from ht_pdm_fjsp.maintenance_requests import state_from_dict,transition

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'research_result/framework_v18_dev_20261021T050413Z'
SOURCE_AUDIT=ROOT/'research_result/framework_v18_audit_20261021T061005Z/independent_audit.json'
V14=ROOT/'research_result/framework_v14_dev_20261020T185303Z'
V14_AUDIT=ROOT/'research_result/framework_v14_audit_20261020T200552Z/independent_audit.json'
PLAN=ROOT/'research_result/v18_first_divergence_plan_vi.md'
TARGET=('interchangeable_scale1','reciprocal_scale3')
TRAIN=tuple(range(73300000,73300005));V14_TRAIN=tuple(range(72600000,72600005))
PHYSICAL=tuple(range(73311000,73311016))
ACTIONS=catalogue();ANCHOR=73370000;BOOTSTRAP=73370001
METRICS=('objective','maintenance_cost','unavailability_cost','failure_cost','tail_cost',
         'H_availability','H_failures','queued_cm_ticks','queued_pm_ticks','H_pending',
         'end_pending','service_starts','requests_created','requests_completed')
CATEGORIES=('same','bounded_service_other_wait','bounded_wait_other_service',
            'bounded_higher_cardinality','bounded_lower_cardinality','machine_set','assignment')
COMPARISONS=(('interchangeable_scale1','own_rank'),('interchangeable_scale1','rollout'),
             ('reciprocal_scale3','own_rank'))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def close_metrics(got,want):
    for key in METRICS:assert np.isclose(float(got[key]),float(want[key]),rtol=0,atol=1e-8),(key,got[key],want[key])
def classify(left,right):
    a,b=ACTIONS[left],ACTIONS[right]
    if a==b:return 'same'
    if a and not b:return 'bounded_service_other_wait'
    if b and not a:return 'bounded_wait_other_service'
    if len(a)!=len(b):return 'bounded_higher_cardinality' if len(a)>len(b) else 'bounded_lower_cardinality'
    return 'machine_set' if {m for m,_ in a}!={m for m,_ in b} else 'assignment'
def episode(case,physical,mode,model,forecast_anchor,deadline):
    cfg=configs()[case];env=FrameworkEnv(cfg);env.reset(seed=physical)
    controller=V7RuleController('rollout',forecast_anchor,deadline) if mode=='rollout' else (V14Controller(model,'competition') if mode=='competition' else None)
    actions=[];states=[];contexts=[]
    for tick in range(HORIZON):
        deadline();state=env.inner.state
        submissions,admitted,legal,obs=legal_present(cfg,state,tick)
        if controller is None:choice=policy_action(model,obs,legal)
        else:
            supplied,matching=controller.choose(cfg,state,tick,case=case,seed=physical)
            assert supplied==submissions;choice=ACTIONS.index(matching)
        assert choice in legal
        actions.append(choice);states.append(asdict(state))
        contexts.append(dict(tick=tick,ages=list(state.physical.ages),remaining=list(state.physical.remaining),
                             assigned=list(state.physical.assigned),failed=list(state.physical.failed),
                             queued=[i for i,r in enumerate(admitted.requests) if r and r.started_at is None],legal=legal))
        env.step(choice)
    return actions,states,contexts,env.episode_metrics()
def bank(cfg,case,comparison,train,physical,tick,count):
    seed=role_seed(ANCHOR,f'v18_first_divergence:{case}:{comparison}:{train}:{physical}:{tick}')
    rng=random.Random(seed);tapes=[]
    for _ in range(count):
        events=[[rng.random()<cfg.failure_probability for _ in range(cfg.machines)] for _ in range(tick,cfg.horizon)]
        uniforms=[rng.random() for _ in range(tick,HORIZON)]
        tapes.append(dict(events=events,uniforms=uniforms))
    return seed,tapes
def root_cost(cfg,state,tick,root,tape,actor,mode,deadline):
    short=total=0.;steps=0
    for offset,t in enumerate(range(tick,cfg.horizon)):
        deadline();submissions,admitted,legal,obs=legal_present(cfg,state,t)
        if offset==0:choice=root
        elif t>=HORIZON or mode=='risk':choice=ACTIONS.index(risk_matching(cfg,admitted,t))
        else:choice=policy_action(actor,obs,legal,tape['uniforms'][offset])
        assert choice in legal
        state,cost,_=transition(cfg,state,t,submissions,ACTIONS[choice],tape['events'][offset])
        total+=cost;steps+=1
        if offset<6:short+=cost
    return (short,total),steps
def bootstrap(cube):
    rng=np.random.default_rng(BOOTSTRAP);draws=np.empty(20000)
    for first in range(0,20000,1000):
        seed=rng.integers(0,5,size=(1000,5));physical=rng.integers(0,16,size=(1000,5,16))
        draws[first:first+1000]=np.take_along_axis(cube[seed],physical,axis=2).mean((1,2))
    return np.quantile(draws,[.025,.975]).tolist()
def run(profile,out):
    if profile not in ('smoke','dev'):raise ValueError(profile)
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(1);start=time.monotonic();limit=300 if profile=='smoke' else 1800
    def deadline():
        if time.monotonic()-start>limit:raise TimeoutError('V18 divergence deadline')
    manifest=dict(status='RUNNING',profile=profile,started_utc=datetime.now(timezone.utc).isoformat())
    save(out/'manifest.json',manifest)
    try:
        if profile=='dev':
            files=['research_result/diagnose_v18_first_divergence.py','research_result/verify_v18_first_divergence.py',
                   'research_result/v18_first_divergence_plan_vi.md']
            dirty=subprocess.check_output(['git','diff','--name-only','HEAD','--',*files],cwd=ROOT,text=True).splitlines()
            untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*files],cwd=ROOT,text=True).splitlines()
            if dirty or untracked:raise RuntimeError(f'uncommitted diagnostic source: {dirty+untracked}')
        source=json.loads((SOURCE/'manifest.json').read_text());source_audit=json.loads(SOURCE_AUDIT.read_text())
        v14_manifest=json.loads((V14/'manifest.json').read_text());v14_audit=json.loads(V14_AUDIT.read_text())
        assert source['status']==v14_manifest['status']=='COMPLETED'
        assert source_audit['status']==v14_audit['status']=='PASS'
        assert Path(source_audit['run']).resolve()==SOURCE and Path(v14_audit['run']).resolve()==V14
        assert sha(SOURCE/'selection.json')==source['selection_sha256']
        assert sha(V14/'selection.json')==v14_manifest['selection_sha256']
        for name,digest in source['artifact_hashes'].items():assert sha(SOURCE/name)==digest
        used=set()
        for registry in (ROOT/'configs').glob('*seed_registry.json'):
            used|=registered_integers(json.loads(registry.read_text()))
        from ht_pdm_fjsp.maintenance_framework_v18_campaign import episode_seed
        used|={episode_seed(seed,iteration,case,index) for seed in TRAIN for iteration in range(1,7)
               for case in configs() for index in range(16)}
        assert not {ANCHOR,BOOTSTRAP}&used and ANCHOR!=BOOTSTRAP
        train=TRAIN if profile=='dev' else TRAIN[:1]
        v14_train=V14_TRAIN if profile=='dev' else V14_TRAIN[:1]
        physicals=PHYSICAL if profile=='dev' else PHYSICAL[:1]
        count=16 if profile=='dev' else 1
        selection=json.loads((SOURCE/'selection.json').read_text())
        v14_selection=json.loads((V14/'selection.json').read_text())
        models={};checkpoint_hashes={}
        for arm in ('bounded_disagreement','own_rank'):
            for seed in train:
                item=selection[arm][str(seed)];cp=SOURCE/item['checkpoint']
                assert sha(cp)==item['sha256']
                models[(arm,seed)]=load_checkpoint(cp);checkpoint_hashes[f'{arm}/{seed}']=item['sha256']
        v14_models={}
        for seed in v14_train:
            item=v14_selection['competition'][str(seed)];cp=V14/item['checkpoint']
            assert sha(cp)==item['sha256']
            v14_models[seed]=load_v14(cp);checkpoint_hashes[f'competition/{seed}']=item['sha256']
        original={(r['controller'],r['case'],r['train_seed'],int(r['seed'])):r for r in read(SOURCE/'episodes.csv')}
        traces={};episode_rows=[];actions=[]
        specs=[('rollout','interchangeable_scale1','',physical,None) for physical in physicals]
        specs += [('competition','reciprocal_scale3',str(seed),physical,v14_models[seed])
                  for seed in v14_train for physical in physicals]
        specs += [(arm,case,str(seed),physical,models[(arm,seed)])
                  for arm in ('bounded_disagreement','own_rank') for case in TARGET
                  for seed in train for physical in physicals]
        for arm,case,seed,physical,model in tqdm(specs,desc='V18 diagnostic episode replay',unit='episode'):
            seq,states,contexts,metrics=episode(case,physical,arm,model,source['registry']['forecast_seed'],deadline)
            close_metrics(metrics,original[(arm,case,seed,physical)])
            traces[(arm,case,seed,physical)]=(seq,states,contexts,metrics)
            episode_rows.append(dict(controller=arm,case=case,train_seed=seed,physical_seed=physical,
                                     **{k:metrics[k] for k in METRICS}))
            actions.extend(dict(controller=arm,case=case,train_seed=seed,physical_seed=physical,
                                tick=t,action=a) for t,a in enumerate(seq))
        q_pairs=[];q_banks=[];v14_pairs=[];transitions=0
        grid=[(case,other,seed,physical) for case,other in COMPARISONS
              for seed in train for physical in physicals]
        for case,other,seed,physical in tqdm(grid,desc='V18 first-divergence Q',unit='pair'):
            deadline();cfg=configs()[case];actor=models[('bounded_disagreement',seed)]
            left,left_states,left_context,left_metrics=traces[('bounded_disagreement',case,str(seed),physical)]
            right,right_states,_,right_metrics=traces[(other,case,str(seed) if other=='own_rank' else '',physical)]
            first=next((t for t,(a,b) in enumerate(zip(left,right)) if a!=b),None)
            row=dict(comparison=f'{case}/{other}',case=case,other=other,train_seed=seed,physical_seed=physical,
                     first_tick=-1 if first is None else first,first_state='' if first is None else json.dumps(left_states[first],sort_keys=True),
                     category='same' if first is None else classify(left[first],right[first]),
                     bounded_action=-1 if first is None else left[first],other_action=-1 if first is None else right[first],
                     bounded_duration=0 if first is None else sum(cfg.service_time[m][j] for m,j in ACTIONS[left[first]]),
                     other_duration=0 if first is None else sum(cfg.service_time[m][j] for m,j in ACTIONS[right[first]]),
                     actual_cost_delta=left_metrics['objective']-right_metrics['objective'],
                     actual_failure_delta=left_metrics['H_failures']-right_metrics['H_failures'],
                     actual_pm_queue_delta=left_metrics['queued_pm_ticks']-right_metrics['queued_pm_ticks'],
                     actual_cm_queue_delta=left_metrics['queued_cm_ticks']-right_metrics['queued_cm_ticks'],
                     own_q6_delta=0.,own_qfull_delta=0.,risk_q6_delta=0.,risk_qfull_delta=0.,
                     bounded_probability=0.,other_probability=0.,logit_margin=0.,forecast_seed=0,forecast_transitions=0)
            if first is not None:
                assert left_states[first]==right_states[first]
                assert all(left[t]==right[t] and left_states[t]==right_states[t] for t in range(first))
                state=state_from_dict(left_states[first]);_,_,legal,obs=legal_present(cfg,state,first)
                with torch.no_grad():logits=actor(torch.as_tensor(obs).reshape(1,56))[0].numpy()
                masked=np.full_like(logits,-1e9);masked[list(legal)]=logits[list(legal)]
                p=np.exp(masked-np.max(masked));p/=p.sum()
                row.update(bounded_probability=float(p[left[first]]),other_probability=float(p[right[first]]),
                           logit_margin=float(logits[left[first]]-logits[right[first]]))
                bank_seed,tapes=bank(cfg,case,other,seed,physical,first,count)
                scores={mode:[] for mode in ('own','risk')};details=[]
                for tape in tqdm(tapes,desc=f'V18 bank {case}/{other}',unit='scenario',leave=False):
                    detail={}
                    for mode in ('own','risk'):
                        for label,root in (('bounded',left[first]),('other',right[first])):
                            value,steps=root_cost(cfg,state,first,root,tape,actor,mode,deadline)
                            detail[f'{mode}_{label}']=value;transitions+=steps
                        scores[mode].append((detail[f'{mode}_bounded'][0]-detail[f'{mode}_other'][0],
                                             detail[f'{mode}_bounded'][1]-detail[f'{mode}_other'][1]))
                    details.append(detail)
                for mode in ('own','risk'):
                    row[f'{mode}_q6_delta']=float(np.mean([x[0] for x in scores[mode]]))
                    row[f'{mode}_qfull_delta']=float(np.mean([x[1] for x in scores[mode]]))
                row.update(forecast_seed=bank_seed,forecast_transitions=count*4*(cfg.horizon-first))
                q_banks.append(dict(comparison=row['comparison'],case=case,other=other,train_seed=seed,
                    physical_seed=physical,tick=first,original_state=left_states[first],context=left_context[first],
                    forecast_seed=bank_seed,bank_sha256=hashlib.sha256(json.dumps(tapes,sort_keys=True).encode()).hexdigest(),
                    tapes=tapes,scenario_scores=details))
            q_pairs.append(row)
        v14_grid=[(seed,v14_seed,physical) for seed in train for v14_seed in v14_train for physical in physicals]
        for seed,v14_seed,physical in tqdm(v14_grid,desc='V18/V14 cross-product actions',unit='pair'):
            left,left_states,_,left_metrics=traces[('bounded_disagreement','reciprocal_scale3',str(seed),physical)]
            right,right_states,_,right_metrics=traces[('competition','reciprocal_scale3',str(v14_seed),physical)]
            first=next((t for t,(a,b) in enumerate(zip(left,right)) if a!=b),None)
            if first is not None:
                assert left_states[first]==right_states[first]
                assert all(left[t]==right[t] and left_states[t]==right_states[t] for t in range(first))
            v14_pairs.append(dict(train_seed=seed,v14_train_seed=v14_seed,physical_seed=physical,
                first_tick=-1 if first is None else first,first_state='' if first is None else json.dumps(left_states[first],sort_keys=True),
                category='same' if first is None else classify(left[first],right[first]),
                bounded_action=-1 if first is None else left[first],other_action=-1 if first is None else right[first],
                actual_cost_delta=left_metrics['objective']-right_metrics['objective'],
                actual_failure_delta=left_metrics['H_failures']-right_metrics['H_failures']))
        write(out/'episodes.csv',episode_rows);write(out/'actions.csv',actions)
        write(out/'q_pairs.csv',q_pairs);write(out/'v14_pairs.csv',v14_pairs)
        with (out/'q_banks.jsonl').open('w') as f:
            for row in q_banks:f.write(json.dumps(row,sort_keys=True)+'\n')
        summary=dict(status='COMPLETED',profile=profile,q_pairs=len(q_pairs),q_banks=len(q_banks),
                     v14_pairs=len(v14_pairs),forecast_transitions=transitions,comparisons={})
        for case,other in COMPARISONS:
            label=f'{case}/{other}';cell=[r for r in q_pairs if r['comparison']==label]
            item=dict(pairs=len(cell),categories={k:sum(r['category']==k for r in cell) for k in CATEGORIES},
                first_tick_mean=float(np.mean([r['first_tick'] for r in cell if r['first_tick']>=0])) if any(r['first_tick']>=0 for r in cell) else None)
            for key in ('actual_cost_delta','actual_failure_delta','actual_pm_queue_delta','actual_cm_queue_delta',
                        'own_q6_delta','risk_q6_delta','own_qfull_delta','risk_qfull_delta'):
                item[key]=float(np.mean([r[key] for r in cell]))
            for mode in ('own','risk'):
                cube=np.asarray([[next(r[f'{mode}_qfull_delta'] for r in cell if r['train_seed']==seed and r['physical_seed']==physical)
                                  for physical in physicals] for seed in train])
                item[f'{mode}_qfull_ci95']=bootstrap(cube) if profile=='dev' else None
                item[f'conditional_{mode}_qfull']=float(np.mean([r[f'{mode}_qfull_delta'] for r in cell if r['first_tick']>=0])) if any(r['first_tick']>=0 for r in cell) else None
                item[f'{mode}_qfull_by_seed']={str(seed):float(cube[i].mean()) for i,seed in enumerate(train)}
            summary['comparisons'][label]=item
        summary['v14_reciprocal3']=dict(pairs=len(v14_pairs),categories={k:sum(r['category']==k for r in v14_pairs) for k in CATEGORIES},
            actual_cost_delta=float(np.mean([r['actual_cost_delta'] for r in v14_pairs])),
            actual_failure_delta=float(np.mean([r['actual_failure_delta'] for r in v14_pairs])))
        if profile=='dev':
            x=summary['comparisons']['interchangeable_scale1/rollout'];y=summary['comparisons']['reciprocal_scale3/own_rank']
            summary['interpretation_flags']=dict(
                interchange_continuation_mismatch=(x['risk_qfull_delta']>=5 and x['risk_qfull_ci95'][0]>0 and x['own_qfull_delta']<=0),
                interchange_own_rank_failure=(x['own_qfull_delta']>=5 and x['own_qfull_ci95'][0]>0 and x['risk_qfull_delta']>=5 and x['risk_qfull_ci95'][0]>0),
                reciprocal3_root_alignment=(y['own_qfull_delta']<=-5 and y['own_qfull_ci95'][1]<0))
        else:summary['interpretation_flags']=None
        save(out/'summary.json',summary)
        manifest.update(status='COMPLETED',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_run=str(SOURCE),source_manifest_sha256=sha(SOURCE/'manifest.json'),source_audit_sha256=sha(SOURCE_AUDIT),
            source_selection_sha256=sha(SOURCE/'selection.json'),source_episodes_sha256=sha(SOURCE/'episodes.csv'),
            v14_run=str(V14),v14_manifest_sha256=sha(V14/'manifest.json'),v14_audit_sha256=sha(V14_AUDIT),
            v14_selection_sha256=sha(V14/'selection.json'),plan_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),
            verifier_sha256=sha(ROOT/'research_result/verify_v18_first_divergence.py'),
            checkpoint_hashes=checkpoint_hashes,scenarios=count,forecast_anchor=ANCHOR,bootstrap_seed=BOOTSTRAP,
            counts=dict(episodes=len(episode_rows),actions=len(actions),q_pairs=len(q_pairs),q_banks=len(q_banks),
                        v14_pairs=len(v14_pairs),forecast_transitions=transitions),
            artifact_hashes={name:sha(out/name) for name in ('episodes.csv','actions.csv','q_pairs.csv','v14_pairs.csv','q_banks.jsonl','summary.json')})
        save(out/'manifest.json',manifest)
        return out
    except Exception as exc:
        manifest.update(status='INCOMPLETE' if isinstance(exc,TimeoutError) else 'FAILED',
                        error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        save(out/'manifest.json',manifest);raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=('smoke','dev'),required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();print(json.dumps(dict(status='COMPLETED',output_dir=str(run(args.profile,args.output_dir)))))
if __name__=='__main__':main()
