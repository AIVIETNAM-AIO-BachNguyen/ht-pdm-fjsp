"""Frozen V14 versus rollout first-divergence diagnostic on opened V19 DEV."""
from __future__ import annotations
import argparse,csv,hashlib,json,random,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import HORIZON,configs
from ht_pdm_fjsp.maintenance_framework_v14_campaign import load_checkpoint,legal_present,policy_action
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
from ht_pdm_fjsp.maintenance_request_policies import risk_matching
from ht_pdm_fjsp.maintenance_requests import state_from_dict,transition
from research_result import diagnose_v19_first_divergence as base

ROOT=Path(__file__).resolve().parent.parent
SOURCE=base.SOURCE;SOURCE_AUDIT=base.SOURCE_AUDIT
V14=base.V14;V14_AUDIT=base.V14_AUDIT
PLAN=ROOT/'research_result/v14_direct_gap_plan_vi.md'
CASES=('interchangeable_scale1','reciprocal_scale2')
TRAIN=tuple(range(72600000,72600005));PHYSICAL=tuple(range(73411000,73411016))
ANCHOR=73670000;BOOTSTRAP=73670001;ACTIONS=base.ACTIONS;METRICS=base.METRICS
CATEGORIES=('same','v14_service_other_wait','v14_wait_other_service',
            'v14_higher_cardinality','v14_lower_cardinality','machine_set','assignment')
sha=base.sha;save=base.save;read=base.read;write=base.write;close_metrics=base.close_metrics

def classify(left,right):return base.classify(left,right).replace('temporal_','v14_')
def bank(cfg,case,seed,physical,tick,count):
    bank_seed=role_seed(ANCHOR,f'v14_direct_gap:{case}:{seed}:{physical}:{tick}')
    rng=random.Random(bank_seed)
    tapes=[[[rng.random()<cfg.failure_probability for _ in range(cfg.machines)]
            for _ in range(tick,cfg.horizon)] for _ in range(count)]
    return bank_seed,tapes

def root_cost(cfg,state,tick,root,tape,actor,mode,deadline):
    short=total=0.;steps=0
    for offset,t in enumerate(range(tick,cfg.horizon)):
        deadline();subs,admitted,legal,obs=legal_present(cfg,state,t)
        if offset==0:action=root
        elif t>=HORIZON or mode=='risk':action=ACTIONS.index(risk_matching(cfg,admitted,t))
        else:action=policy_action(actor,obs,legal)
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

def run(profile,out):
    assert profile in ('smoke','dev')
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(1);start=time.monotonic();limit=300 if profile=='smoke' else 1800
    def deadline():
        if time.monotonic()-start>limit:raise TimeoutError('V14 direct-gap diagnostic deadline')
    manifest=dict(status='RUNNING',profile=profile,started_utc=datetime.now(timezone.utc).isoformat())
    save(out/'manifest.json',manifest)
    try:
        if profile=='dev':
            files=['research_result/diagnose_v14_direct_gap.py','research_result/verify_v14_direct_gap.py',
                   'research_result/v14_direct_gap_plan_vi.md']
            dirty=subprocess.check_output(['git','diff','--name-only','HEAD','--',*files],cwd=ROOT,text=True).splitlines()
            untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*files],cwd=ROOT,text=True).splitlines()
            if dirty or untracked:raise RuntimeError(f'uncommitted diagnostic source: {dirty+untracked}')
        source=json.loads((SOURCE/'manifest.json').read_text());audit=json.loads(SOURCE_AUDIT.read_text())
        v14_manifest=json.loads((V14/'manifest.json').read_text());v14_audit=json.loads(V14_AUDIT.read_text())
        assert source['status']==v14_manifest['status']=='COMPLETED'
        assert audit['status']==v14_audit['status']=='PASS'
        assert Path(audit['run']).resolve()==SOURCE and Path(v14_audit['run']).resolve()==V14
        for name,digest in source['artifact_hashes'].items():assert sha(SOURCE/name)==digest
        assert sha(V14/'selection.json')==v14_manifest['selection_sha256']
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
        selection=json.loads((V14/'selection.json').read_text())
        models={};checkpoint_hashes={}
        for seed in train:
            item=selection['competition'][str(seed)];cp=V14/item['checkpoint']
            assert sha(cp)==item['sha256']
            models[seed]=load_checkpoint(cp);checkpoint_hashes[str(seed)]=item['sha256']
        originals={(r['controller'],r['case'],r['train_seed'],int(r['seed'])):r for r in read(SOURCE/'episodes.csv')}
        traces={};episode_rows=[];action_rows=[]
        specs=[('rollout',case,'',physical,None) for case in CASES for physical in physicals]
        specs += [('competition',case,str(seed),physical,models[seed]) for case in CASES for seed in train for physical in physicals]
        for arm,case,seed,physical,model in tqdm(specs,desc='V14 direct-gap episode replay',unit='episode'):
            seq,states,contexts,metrics=base.episode(case,physical,arm,model,source['registry']['forecast_seed'],deadline)
            close_metrics(metrics,originals[(arm,case,seed,physical)])
            traces[(arm,case,seed,physical)]=(seq,states,contexts,metrics)
            episode_rows.append(dict(controller=arm,case=case,train_seed=seed,physical_seed=physical,
                                     **{k:metrics[k] for k in METRICS}))
            action_rows.extend(dict(controller=arm,case=case,train_seed=seed,physical_seed=physical,
                                    tick=t,action=a) for t,a in enumerate(seq))
        pairs=[];banks=[];steps=0
        for case in CASES:
            cfg=configs()[case]
            for seed in tqdm(train,desc=f'V14 first-divergence {case}',unit='seed'):
                for physical in physicals:
                    deadline();left,states,contexts,lm=traces[('competition',case,str(seed),physical)]
                    right,rstates,_,rm=traces[('rollout',case,'',physical)]
                    first=next((t for t,(a,b) in enumerate(zip(left,right)) if a!=b),None)
                    row=dict(case=case,train_seed=seed,physical_seed=physical,
                             first_tick=-1 if first is None else first,
                             first_state='' if first is None else json.dumps(states[first],sort_keys=True),
                             category='same' if first is None else classify(left[first],right[first]),
                             v14_action=-1 if first is None else left[first],
                             rollout_action=-1 if first is None else right[first],
                             actual_cost_delta=lm['objective']-rm['objective'],
                             actual_failure_delta=lm['H_failures']-rm['H_failures'],
                             actual_pm_queue_delta=lm['queued_pm_ticks']-rm['queued_pm_ticks'],
                             actual_cm_queue_delta=lm['queued_cm_ticks']-rm['queued_cm_ticks'],
                             actual_service_start_delta=lm['service_starts']-rm['service_starts'],
                             own_q6_delta=0.,own_qfull_delta=0.,risk_q6_delta=0.,risk_qfull_delta=0.,
                             forecast_seed=0,forecast_transitions=0)
                    if first is not None:
                        assert states[first]==rstates[first]
                        assert all(left[t]==right[t] and states[t]==rstates[t] for t in range(first))
                        current=state_from_dict(states[first]);bank_seed,tapes=bank(cfg,case,seed,physical,first,count)
                        scores={mode:[] for mode in ('own','risk')};details=[]
                        for tape in tqdm(tapes,desc=f'V14 bank {case}',unit='scenario',leave=False):
                            detail={}
                            for mode in ('own','risk'):
                                for label,root in (('v14',left[first]),('rollout',right[first])):
                                    value,n=root_cost(cfg,current,first,root,tape,models[seed],mode,deadline)
                                    detail[f'{mode}_{label}']=value;steps+=n
                                scores[mode].append((detail[f'{mode}_v14'][0]-detail[f'{mode}_rollout'][0],
                                                     detail[f'{mode}_v14'][1]-detail[f'{mode}_rollout'][1]))
                            details.append(detail)
                        for mode in ('own','risk'):
                            row[f'{mode}_q6_delta']=float(np.mean([x[0] for x in scores[mode]]))
                            row[f'{mode}_qfull_delta']=float(np.mean([x[1] for x in scores[mode]]))
                        row.update(forecast_seed=bank_seed,forecast_transitions=count*4*(cfg.horizon-first))
                        banks.append(dict(case=case,train_seed=seed,physical_seed=physical,tick=first,
                                          original_state=states[first],context=contexts[first],forecast_seed=bank_seed,
                                          bank_sha256=hashlib.sha256(json.dumps(tapes,sort_keys=True).encode()).hexdigest(),
                                          tapes=tapes,scenario_scores=details))
                    pairs.append(row)
        write(out/'episodes.csv',episode_rows);write(out/'actions.csv',action_rows);write(out/'q_pairs.csv',pairs)
        with (out/'q_banks.jsonl').open('w') as f:
            for row in banks:f.write(json.dumps(row,sort_keys=True)+'\n')
        summary=dict(status='COMPLETED',profile=profile,q_pairs=len(pairs),q_banks=len(banks),
                     forecast_transitions=steps,cases={})
        for case in CASES:
            cell=[r for r in pairs if r['case']==case]
            item=dict(pairs=len(cell),categories={k:sum(r['category']==k for r in cell) for k in CATEGORIES},
                      first_tick_mean=float(np.mean([r['first_tick'] for r in cell if r['first_tick']>=0]))
                      if any(r['first_tick']>=0 for r in cell) else None)
            for key in ('actual_cost_delta','actual_failure_delta','actual_pm_queue_delta','actual_cm_queue_delta',
                        'actual_service_start_delta','own_q6_delta','own_qfull_delta','risk_q6_delta','risk_qfull_delta'):
                item[key]=float(np.mean([r[key] for r in cell]))
            for mode in ('own','risk'):
                cube=np.asarray([[next(r[f'{mode}_qfull_delta'] for r in cell if r['train_seed']==seed and r['physical_seed']==physical)
                                  for physical in physicals] for seed in train])
                item[f'{mode}_qfull_by_seed']={str(seed):float(cube[i].mean()) for i,seed in enumerate(train)}
                item[f'{mode}_qfull_ci95']=bootstrap(cube) if profile=='dev' else None
                divergent=[r[f'{mode}_qfull_delta'] for r in cell if r['first_tick']>=0]
                item[f'conditional_{mode}_qfull']=float(np.mean(divergent)) if divergent else None
            if profile=='dev':
                item['material_root_failure']=item['own_qfull_delta']>=5 and item['own_qfull_ci95'][0]>0
                item['continuation_mismatch']=item['risk_qfull_ci95'][0]>0 and item['own_qfull_delta']<=0
            else:item['material_root_failure']=item['continuation_mismatch']=None
            summary['cases'][case]=item
        save(out/'summary.json',summary)
        manifest.update(status='COMPLETED',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
                        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                        source_run=str(SOURCE),source_manifest_sha256=sha(SOURCE/'manifest.json'),source_audit_sha256=sha(SOURCE_AUDIT),
                        source_episodes_sha256=sha(SOURCE/'episodes.csv'),v14_run=str(V14),
                        v14_manifest_sha256=sha(V14/'manifest.json'),v14_audit_sha256=sha(V14_AUDIT),
                        v14_selection_sha256=sha(V14/'selection.json'),plan_sha256=sha(PLAN),
                        script_sha256=sha(Path(__file__)),verifier_sha256=sha(ROOT/'research_result/verify_v14_direct_gap.py'),
                        base_script_sha256=sha(base.__file__),checkpoint_hashes=checkpoint_hashes,
                        forecast_anchor=ANCHOR,bootstrap_seed=BOOTSTRAP,scenarios=count,
                        counts=dict(episodes=len(episode_rows),actions=len(action_rows),q_pairs=len(pairs),
                                    q_banks=len(banks),forecast_transitions=steps),
                        artifact_hashes={name:sha(out/name) for name in ('episodes.csv','actions.csv','q_pairs.csv','q_banks.jsonl','summary.json')})
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
    a=p.parse_args();print(json.dumps(dict(status='COMPLETED',output_dir=str(run(a.profile,a.output_dir)))))
if __name__=='__main__':main()
