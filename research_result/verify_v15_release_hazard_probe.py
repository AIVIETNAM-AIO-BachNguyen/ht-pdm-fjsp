"""Independent all-row reconstruction of the V15 release-hazard Q-bank probe."""
from __future__ import annotations

import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES,catalogue,configs,vector_observation
from ht_pdm_fjsp.maintenance_framework_v14_policy import OpportunityCostActor
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings,state_from_dict,submit_requests

ROOT=Path(__file__).resolve().parent.parent
ACTIONS=catalogue()
ALPHAS=(0.,.25,.5,1.,2.,4.,8.,16.,32.)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def close(a,b,label):
    if not np.isclose(float(a),float(b),rtol=0,atol=1e-8):raise AssertionError((label,a,b))

def features(cfg,state,weights,legal):
    present={m for m in state.physical.assigned if m>=0}
    def value(action):
        remaining=list(state.physical.remaining)
        served={m for m,j in action}
        for m,j in action:remaining[j]=cfg.service_time[m][j]
        totals=[0.,0.]
        for n in range(cfg.machines):
            if n in present or state.physical.failed[n] or n in served:continue
            d=max(1,min(remaining[j] for j in range(cfg.technicians) if cfg.service_time[n][j]>0))
            safe=max(0,cfg.failure_age-state.physical.ages[n]-1)
            h=max(0,d-safe)
            totals[0]+=weights[n]*(-np.expm1(h*np.log1p(-cfg.failure_probability)))
            totals[1]+=weights[n]*(-np.expm1(d*np.log1p(-cfg.failure_probability)))
        return totals
    baseline=value(())
    return {a:tuple(v-baseline[k] for k,v in enumerate(value(ACTIONS[a]))) for a in legal}

def argmax(scores,legal,feature,alpha):
    ranked=sorted(legal,key=lambda a:(-(scores[a]-alpha*feature[a]),a))
    return ranked[0]

def classification(a,b):
    if a==b:return 'same'
    if len(ACTIONS[a])==0 or len(ACTIONS[b])==0:return 'wait'
    ma=tuple(sorted(m for m,_ in ACTIONS[a]));mb=tuple(sorted(m for m,_ in ACTIONS[b]))
    return 'assignment' if ma==mb else 'machine_set'

def audit(source,diagnostic):
    deadline=time.monotonic()+1200
    manifest=json.loads((diagnostic/'manifest.json').read_text())
    source_manifest=json.loads((source/'manifest.json').read_text())
    profile=manifest['profile']
    assert profile in ('smoke','dev') and manifest['status']==source_manifest['status']=='COMPLETED'
    assert Path(manifest['source_run']).resolve()==source
    assert manifest['source_manifest_sha256']==sha(source/'manifest.json')
    assert manifest['source_q_sha256']==sha(source/'counterfactual_states.csv')
    assert manifest['source_trace_sha256']==sha(source/'training_trace.csv')
    assert manifest['protocol_sha256']==sha(ROOT/'research_result/v15_release_hazard_probe_protocol_vi.md')
    assert manifest['proposal_sha256']==sha(ROOT/'research_result/v15_release_hazard_proposal_vi.md')
    assert manifest['script_sha256']==sha(ROOT/'research_result/run_v15_release_hazard_probe.py')
    for name,digest in manifest['artifact_hashes'].items():assert sha(diagnostic/name)==digest
    for name,digest in source_manifest['artifact_hashes'].items():assert sha(source/name)==digest
    if profile=='dev':
        main=Path(manifest['main_audit_dir']);report=json.loads((main/'independent_audit.json').read_text())
        assert manifest['main_audit_sha256']==sha(main/'independent_audit.json')
        assert report['status']=='PASS' and Path(report['run']).resolve()==source
        seeds=tuple(range(72600000,72600005));updates=6;states=64;draws=20000
    else:
        assert manifest['main_audit_dir'] is None and manifest['main_audit_sha256'] is None
        seeds=(72620000,);updates=1;states=3;draws=100
    raw=[r for r in read_rows(source/'counterfactual_states.csv') if r['arm']=='count_calibrated']
    key=lambda r:(int(r['train_seed']),int(r['iteration']),r['case'],int(r['physical_seed']),int(r['tick']),r['rank'])
    raw_map={key(r):r for r in raw};rows=read_rows(diagnostic/'rows.csv')
    assert len(raw_map)==len(raw)==len(rows)==len(seeds)*updates*len(CASES)*states
    assert {key(r) for r in rows}==set(raw_map)
    rows_by_cell={}
    for r in rows:rows_by_cell.setdefault((int(r['train_seed']),int(r['iteration']),r['case']),[]).append(r)
    assert set(rows_by_cell)=={(s,u,c) for s in seeds for u in range(1,updates+1) for c in CASES}
    for cell in rows_by_cell.values():cell.sort(key=lambda r:r['rank']);assert len(cell)==states
    qz=np.empty((len(seeds),updates,len(CASES),states,len(ALPHAS)))
    qb=np.empty_like(qz)
    models={};torch.set_num_threads(1)
    for si,seed in enumerate(seeds):
        for ui in range(updates):
            cp=source/'checkpoints'/'count_calibrated'/str(seed)/f'{ui}.pt'
            assert manifest['checkpoint_hashes'][f'{seed}/{ui}']==sha(cp)
            payload=torch.load(cp,map_location='cpu',weights_only=True)
            assert payload['metadata']['arm']=='count_calibrated'
            actor=OpportunityCostActor(mode='count_calibrated');actor.load_state_dict(payload['state_dict']);actor.eval()
            models[(seed,ui)]=actor
            for ci,case in enumerate(CASES):
                cfg=configs()[case]
                for ri,row in enumerate(tqdm(rows_by_cell[(seed,ui+1,case)],
                    desc=f'Independent V15 feature {seed}/{ui+1}/{case}',unit='state',leave=False)):
                    if time.monotonic()>deadline:raise TimeoutError('V15 audit deadline')
                    original=raw_map[key(row)]
                    state=state_from_dict(json.loads(original['original_state']))
                    tick=int(row['tick'])
                    admitted=submit_requests(cfg,state,tick,request_action(cfg,state,tick,'threshold'))
                    legal=sorted(ACTIONS.index(a) for a in legal_matchings(cfg,admitted))
                    assert legal==json.loads(row['legal'])==json.loads(original['legal'])
                    observation=vector_observation(cfg,admitted,tick)
                    with torch.no_grad():logits,_,demand=actor.components(torch.tensor(observation).reshape(1,56))
                    logits={a:float(logits[0,a]) for a in legal}
                    for a,v in json.loads(row['logits']).items():close(v,logits[int(a)],'logit')
                    weights=[]
                    for n in range(cfg.machines):
                        eligible=np.asarray([j for j in range(cfg.technicians) if cfg.service_time[n][j]>0])
                        weights.append(float(demand[0,n,eligible].mean()))
                    for got,want in zip(json.loads(row['weights']),weights):close(got,want,'weight')
                    feat=features(cfg,state,weights,legal)
                    for a in legal:
                        close(json.loads(row['z_feature'])[str(a)],feat[a][0],'hazard feature')
                        close(json.loads(row['b_feature'])[str(a)],feat[a][1],'control feature')
                    q={int(a):float(v) for a,v in json.loads(original['scores']).items()}
                    assert q=={int(a):float(v) for a,v in json.loads(row['q']).items()}
                    azs=[];abs_=[];qzs=[];qbs=[]
                    for ai,alpha in enumerate(ALPHAS):
                        z=argmax(logits,legal,{a:feat[a][0] for a in legal},alpha)
                        b=argmax(logits,legal,{a:feat[a][1] for a in legal},alpha)
                        azs.append(z);abs_.append(b);qzs.append(q[z]);qbs.append(q[b])
                        qz[si,ui,ci,ri,ai]=q[z];qb[si,ui,ci,ri,ai]=q[b]
                    assert json.loads(row['z_actions'])==azs and json.loads(row['b_actions'])==abs_
                    for got,want in zip(json.loads(row['z_q']),qzs):close(got,want,'z Q')
                    for got,want in zip(json.loads(row['b_q']),qbs):close(got,want,'b Q')
                    zv=[feat[a][0] for a in legal];bv=[feat[a][1] for a in legal]
                    close(row['z_range'],max(zv)-min(zv),'z range')
                    close(row['b_range'],max(bv)-min(bv),'b range')
                    opposite=any((zv[i]-zv[j])*(bv[i]-bv[j])<0 and
                                 abs(zv[i]-zv[j])>=.05 and abs(bv[i]-bv[j])>=.05
                                 for i in range(len(legal)) for j in range(i+1,len(legal)))
                    assert row['ranking_discordance']==str(bool(opposite))
    summary=json.loads((diagnostic/'summary.json').read_text())
    assert tuple(summary['alpha_grid'])==ALPHAS and summary['rows']==len(rows)
    assert summary['scenario_count']==(4 if profile=='dev' else 2)
    assert manifest['bootstrap_seed']==72640003 and manifest['bootstrap_draws']==draws
    fitted={'z':{},'b':{}}
    for si,seed in enumerate(seeds):
        if profile=='dev':
            others=[j for j in range(len(seeds)) if j!=si]
            fitted['z'][str(seed)]=int(np.argmin(qz[others].mean(axis=(0,1,2,3))))
            fitted['b'][str(seed)]=int(np.argmin(qb[others].mean(axis=(0,1,2,3))))
        else:fitted['z'][str(seed)]=fitted['b'][str(seed)]=ALPHAS.index(1.)
    assert summary['fit_indices']==fitted
    for proxy in fitted:
        for seed,index in fitted[proxy].items():close(summary['fit_alpha'][proxy][seed],ALPHAS[index],'alpha')
    cube=np.empty((len(seeds),updates,len(CASES),states))
    for si,seed in enumerate(seeds):
        for ui in range(updates):
            for ci,case in enumerate(CASES):
                for ri,row in enumerate(rows_by_cell[(seed,ui+1,case)]):
                    zi=fitted['z'][str(seed)];bi=fitted['b'][str(seed)]
                    az=json.loads(row['z_actions'])[zi];ab=json.loads(row['b_actions'])[bi]
                    assert int(row['chosen_z'])==az and int(row['chosen_b'])==ab
                    close(row['alpha_z'],ALPHAS[zi],'selected alpha z')
                    close(row['alpha_b'],ALPHAS[bi],'selected alpha b')
                    qza=qz[si,ui,ci,ri,zi];qba=qb[si,ui,ci,ri,bi]
                    close(row['selected_q_z'],qza,'selected Q z')
                    close(row['selected_q_b'],qba,'selected Q b')
                    close(row['delta'],qza-qba,'delta')
                    assert row['decision_class']==classification(az,ab)
                    cube[si,ui,ci,ri]=qza-qba
                    legal=json.loads(row['legal']);score={int(a):float(v) for a,v in json.loads(row['logits']).items()}
                    for name,action,alpha,feat_name in (('z',az,ALPHAS[zi],'z_feature'),('b',ab,ALPHAS[bi],'b_feature')):
                        feat={int(a):float(v) for a,v in json.loads(row[feat_name]).items()}
                        adjusted=sorted((score[a]-alpha*feat[a] for a in legal),reverse=True)
                        near=len(adjusted)>1 and adjusted[0]-adjusted[1]<=1e-6
                        assert row[f'{name}_near_tie']==str(bool(near))
    close(summary['mean_delta'],cube.mean(),'mean')
    close(summary['baseline_q'],qz[...,0].mean(),'base Q')
    close(summary['selected_q_z'],np.mean([float(r['selected_q_z']) for r in rows]),'z selected Q')
    close(summary['selected_q_b'],np.mean([float(r['selected_q_b']) for r in rows]),'b selected Q')
    for si,seed in enumerate(seeds):close(summary['by_train_seed'][str(seed)],cube[si].mean(),'seed')
    for ci,case in enumerate(CASES):close(summary['by_case'][case],cube[:,:,ci,:].mean(),'case')
    for ui in range(updates):close(summary['by_iteration'][str(ui+1)],cube[:,ui].mean(),'update')
    changed=[r for r in rows if r['decision_class']!='same']
    assert summary['changed_actions']==len(changed)
    close(summary['changed_fraction'],len(changed)/len(rows),'change fraction')
    for name in ('same','wait','machine_set','assignment'):
        assert summary['decision_classes'][name]==sum(r['decision_class']==name for r in rows)
    assert summary['improved']==int((cube<0).sum())
    assert summary['worsened']==int((cube>0).sum())
    assert summary['tied']==int((cube==0).sum())
    close(summary['changed_mean_delta'],np.mean([float(r['delta']) for r in changed]) if changed else 0.,'changed mean')
    close(summary['z_material_fraction'],np.mean([float(r['z_range'])>=.05 for r in rows]),'z material')
    close(summary['b_material_fraction'],np.mean([float(r['b_range'])>=.05 for r in rows]),'b material')
    close(summary['ranking_discordance_fraction'],np.mean([r['ranking_discordance']=='True' for r in rows]),'discordance')
    assert summary['near_tie_z']==sum(r['z_near_tie']=='True' for r in rows)
    assert summary['near_tie_b']==sum(r['b_near_tie']=='True' for r in rows)
    if profile=='dev':
        rng=np.random.default_rng(72640003);draws_out=np.empty(draws)
        for left in range(0,draws,1000):
            count=min(1000,draws-left)
            chosen=rng.integers(0,5,size=(count,5))
            ranks=rng.integers(0,64,size=(count,5,updates,len(CASES),states))
            draws_out[left:left+count]=np.take_along_axis(cube[chosen],ranks,axis=-1).mean(axis=(1,2,3,4))
        for got,want in zip(summary['hierarchical_ci95'],np.quantile(draws_out,[.025,.975])):close(got,want,'CI')
    else:assert summary['hierarchical_ci95'] is None
    means=cube.mean(axis=(1,2,3))
    gate=bool(profile=='dev' and cube.mean()<=-1 and sum(means<0)>=4 and
              summary['hierarchical_ci95'][1]<0 and len(changed)/len(rows)>=.01)
    assert summary['diagnostic_gate']==gate
    return dict(status='PASS',source_run=str(source),diagnostic=str(diagnostic),rows=len(rows),
                checkpoints=len(models),checks=['source/artifact/checkpoint hashes','all current legal actions and hazard/control features',
                'all fixed-alpha actions and Q values','leave-one-seed-out alpha selection','all heldout metrics/bootstrap/gate'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--diagnostic-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    result=audit(args.run_dir.resolve(),args.diagnostic_dir.resolve())
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
