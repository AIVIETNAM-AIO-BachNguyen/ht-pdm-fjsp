"""Prelocked V15 release-hazard versus age-agnostic feature probe on V14 Q banks."""
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
PROTOCOL=ROOT/'research_result/v15_release_hazard_probe_protocol_vi.md'
PROPOSAL=ROOT/'research_result/v15_release_hazard_proposal_vi.md'
ACTIONS=catalogue()
ALPHAS=(0.,.25,.5,1.,2.,4.,8.,16.,32.)
BOOTSTRAP_SEED=72640003

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def exposure_features(cfg,state,weights,legal):
    """Incremental probability of failure before next compatible service opportunity."""
    assert len(weights)==cfg.machines and all(np.isfinite(weights))
    occupied=set(m for m in state.physical.assigned if m>=0)
    def totals(matching):
        release=list(state.physical.remaining)
        started={m for m,_ in matching}
        for m,j in matching:
            assert release[j]==0 and cfg.service_time[m][j]>0
            release[j]=cfg.service_time[m][j]
        z=b=0.
        for n in range(cfg.machines):
            if n in occupied or state.physical.failed[n] or n in started:continue
            first=max(1,min(release[j] for j in range(cfg.technicians)
                            if cfg.service_time[n][j]>0))
            safe=max(0,cfg.failure_age-state.physical.ages[n]-1)
            h=max(0,first-safe)
            z+=float(weights[n])*(1-(1-cfg.failure_probability)**h)
            b+=float(weights[n])*(1-(1-cfg.failure_probability)**first)
        return z,b
    wait=totals(())
    features={}
    for index in legal:
        z,b=totals(ACTIONS[index])
        features[index]=(z-wait[0],b-wait[1])
    assert ACTIONS.index(()) in features
    assert max(abs(v) for v in features[ACTIONS.index(())])<1e-12
    return features

def choose(logits,legal,feature,alpha):
    return max(legal,key=lambda i:(float(logits[i])-alpha*feature[i],-i))

def category(a,b):
    if a==b:return 'same'
    if not ACTIONS[a] or not ACTIONS[b]:return 'wait'
    if {m for m,_ in ACTIONS[a]}!={m for m,_ in ACTIONS[b]}:return 'machine_set'
    return 'assignment'

def bootstrap(cube,draws):
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    out=np.empty(draws)
    nseeds,updates,cases,states=cube.shape
    for left in tqdm(range(0,draws,1000),desc='V15 hierarchical bootstrap',unit='chunk',leave=False):
        count=min(1000,draws-left)
        selected=rng.integers(0,nseeds,size=(count,nseeds))
        state=rng.integers(0,states,size=(count,nseeds,updates,cases,states))
        out[left:left+count]=np.take_along_axis(cube[selected],state,axis=-1).mean(axis=(1,2,3,4))
    return tuple(float(x) for x in np.quantile(out,[.025,.975]))

def analyze(run,out,profile,main_audit):
    start=time.monotonic();deadline=start+(1200 if profile=='dev' else 300)
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='COMPLETED' and manifest['profile']==profile
    if profile=='dev':
        report=json.loads((main_audit/'independent_audit.json').read_text())
        assert report['status']=='PASS' and Path(report['run']).resolve()==run
    else:assert main_audit is None
    assert all(sha(ROOT/name)==digest for name,digest in manifest['source_hashes'].items())
    for name,digest in manifest['artifact_hashes'].items():assert sha(run/name)==digest
    source=[r for r in read_csv(run/'counterfactual_states.csv') if r['arm']=='count_calibrated']
    if profile=='dev':seeds=tuple(range(72600000,72600005));updates=6;states=64;draws=20000
    else:seeds=(72620000,);updates=1;states=3;draws=100
    assert len(source)==len(seeds)*updates*len(CASES)*states
    groups={}
    for row in source:groups.setdefault((int(row['train_seed']),int(row['iteration']),row['case']),[]).append(row)
    assert set(groups)=={(seed,u,case) for seed in seeds for u in range(1,updates+1) for case in CASES}
    assert all(len(v)==states for v in groups.values())
    for key in groups:groups[key].sort(key=lambda r:r['rank'])
    qz=np.empty((len(seeds),updates,len(CASES),states,len(ALPHAS)))
    qb=np.empty_like(qz)
    rows=[];checkpoint_hashes={};torch.set_num_threads(1)
    trace={(int(r['train_seed']),int(r['iteration'])):r['checkpoint_sha256']
           for r in read_csv(run/'training_trace.csv') if r['arm']=='count_calibrated'}
    for si,seed in enumerate(seeds):
        for ui in range(updates):
            cp=run/'checkpoints'/'count_calibrated'/str(seed)/f'{ui}.pt'
            checkpoint_hashes[f'{seed}/{ui}']=sha(cp)
            if ui>0:assert checkpoint_hashes[f'{seed}/{ui}']==trace[(seed,ui)]
            payload=torch.load(cp,map_location='cpu',weights_only=True)
            assert payload['metadata']['arm']=='count_calibrated'
            actor=OpportunityCostActor(mode='count_calibrated');actor.load_state_dict(payload['state_dict']);actor.eval()
            for ci,case in enumerate(CASES):
                cfg=configs()[case]
                for ri,row in enumerate(tqdm(groups[(seed,ui+1,case)],desc=f'V15 feature {seed}/{ui+1}/{case}',unit='state',leave=False)):
                    if time.monotonic()>deadline:raise TimeoutError('V15 probe exceeded deadline')
                    state=state_from_dict(json.loads(row['original_state']))
                    tick=int(row['tick'])
                    admitted=submit_requests(cfg,state,tick,request_action(cfg,state,tick,'threshold'))
                    legal=[ACTIONS.index(a) for a in legal_matchings(cfg,admitted)]
                    assert legal==json.loads(row['legal']) and len(legal)>1
                    obs=torch.as_tensor(vector_observation(cfg,admitted,tick)).reshape(1,56)
                    with torch.no_grad():logits,_,demand=actor.components(obs)
                    logit=logits[0].numpy();demand=demand[0].numpy()
                    weight=np.asarray([float(np.mean([demand[n,j] for j in range(cfg.technicians)
                                      if cfg.service_time[n][j]>0])) for n in range(cfg.machines)])
                    feature=exposure_features(cfg,state,weight,legal)
                    q={int(a):float(v) for a,v in json.loads(row['scores']).items()}
                    assert set(q)==set(legal)
                    z_actions=[];b_actions=[];z_q=[];b_q=[]
                    for ai,alpha in enumerate(ALPHAS):
                        az=choose(logit,legal,{a:v[0] for a,v in feature.items()},alpha)
                        ab=choose(logit,legal,{a:v[1] for a,v in feature.items()},alpha)
                        z_actions.append(az);b_actions.append(ab)
                        z_q.append(q[az]);b_q.append(q[ab])
                        qz[si,ui,ci,ri,ai]=q[az];qb[si,ui,ci,ri,ai]=q[ab]
                    zvalues=[feature[a][0] for a in legal]
                    bvalues=[feature[a][1] for a in legal]
                    discordant=any((zvalues[i]-zvalues[j])*(bvalues[i]-bvalues[j])<0 and
                                   abs(zvalues[i]-zvalues[j])>=.05 and abs(bvalues[i]-bvalues[j])>=.05
                                   for i in range(len(legal)) for j in range(i+1,len(legal)))
                    rows.append(dict(train_seed=seed,iteration=ui+1,case=case,
                        physical_seed=row['physical_seed'],tick=tick,rank=row['rank'],
                        legal=json.dumps(legal),logits=json.dumps({a:float(logit[a]) for a in legal},sort_keys=True),
                        weights=json.dumps(weight.tolist()),
                        z_feature=json.dumps({a:feature[a][0] for a in legal},sort_keys=True),
                        b_feature=json.dumps({a:feature[a][1] for a in legal},sort_keys=True),
                        q=json.dumps(q,sort_keys=True),z_actions=json.dumps(z_actions),
                        b_actions=json.dumps(b_actions),z_q=json.dumps(z_q),b_q=json.dumps(b_q),
                        z_range=max(zvalues)-min(zvalues),b_range=max(bvalues)-min(bvalues),
                        ranking_discordance=bool(discordant)))
    fit={'z':{},'b':{}}
    for si,seed in enumerate(seeds):
        if profile=='dev':
            others=[i for i in range(len(seeds)) if i!=si]
            fit['z'][str(seed)]=int(np.argmin(qz[others].mean(axis=(0,1,2,3))))
            fit['b'][str(seed)]=int(np.argmin(qb[others].mean(axis=(0,1,2,3))))
        else:fit['z'][str(seed)]=fit['b'][str(seed)]=ALPHAS.index(1.)
    cube=np.empty((len(seeds),updates,len(CASES),states))
    for r in rows:
        si=seeds.index(int(r['train_seed']));ui=int(r['iteration'])-1;ci=CASES.index(r['case'])
        ri=next(i for i,x in enumerate(groups[(int(r['train_seed']),ui+1,r['case'])]) if x['rank']==r['rank'])
        zi=fit['z'][str(seeds[si])];bi=fit['b'][str(seeds[si])]
        az=json.loads(r['z_actions'])[zi];ab=json.loads(r['b_actions'])[bi]
        qza=float(json.loads(r['z_q'])[zi]);qba=float(json.loads(r['b_q'])[bi])
        delta=qza-qba;cube[si,ui,ci,ri]=delta
        r.update(alpha_z=ALPHAS[zi],alpha_b=ALPHAS[bi],chosen_z=az,chosen_b=ab,
                 selected_q_z=qza,selected_q_b=qba,delta=delta,decision_class=category(az,ab))
        for arm,action,alpha,field in (('z',az,ALPHAS[zi],'z_feature'),('b',ab,ALPHAS[bi],'b_feature')):
            score=json.loads(r['logits']);feature=json.loads(r[field]);values=sorted(
                (float(score[str(i)])-alpha*float(feature[str(i)]) for i in json.loads(r['legal'])),reverse=True)
            r[f'{arm}_near_tie']=len(values)>1 and values[0]-values[1]<=1e-6
    changed=[r for r in rows if r['decision_class']!='same']
    ci=bootstrap(cube,draws) if profile=='dev' else None
    mean=cube.mean(axis=(1,2,3))
    summary=dict(status='COMPLETED',profile=profile,rows=len(rows),scenario_count=4 if profile=='dev' else 2,
        alpha_grid=ALPHAS,fit_indices=fit,
        fit_alpha={proxy:{seed:ALPHAS[index] for seed,index in values.items()} for proxy,values in fit.items()},
        mean_delta=float(cube.mean()),hierarchical_ci95=ci,
        by_train_seed={str(seed):float(mean[i]) for i,seed in enumerate(seeds)},
        by_case={case:float(cube[:,:,i,:].mean()) for i,case in enumerate(CASES)},
        by_iteration={str(u+1):float(cube[:,u,:,:].mean()) for u in range(updates)},
        baseline_q=float(qz[...,0].mean()),selected_q_z=float(np.mean([r['selected_q_z'] for r in rows])),
        selected_q_b=float(np.mean([r['selected_q_b'] for r in rows])),
        changed_actions=len(changed),changed_fraction=len(changed)/len(rows),
        decision_classes={name:sum(r['decision_class']==name for r in rows)
                          for name in ('same','wait','machine_set','assignment')},
        improved=sum(r['delta']<0 for r in rows),worsened=sum(r['delta']>0 for r in rows),
        tied=sum(r['delta']==0 for r in rows),
        changed_mean_delta=float(np.mean([r['delta'] for r in changed])) if changed else 0.,
        z_material_fraction=float(np.mean([float(r['z_range'])>=.05 for r in rows])),
        b_material_fraction=float(np.mean([float(r['b_range'])>=.05 for r in rows])),
        ranking_discordance_fraction=float(np.mean([r['ranking_discordance'] for r in rows])),
        near_tie_z=sum(r['z_near_tie'] for r in rows),near_tie_b=sum(r['b_near_tie'] for r in rows),
        diagnostic_gate=bool(profile=='dev' and cube.mean()<=-1 and sum(mean<0)>=4 and
                             ci[1]<0 and len(changed)/len(rows)>=.01))
    write_csv(out/'rows.csv',rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    provenance=dict(status='COMPLETED',profile=profile,source_run=str(run),
        source_manifest_sha256=sha(run/'manifest.json'),source_q_sha256=sha(run/'counterfactual_states.csv'),
        source_trace_sha256=sha(run/'training_trace.csv'),
        main_audit_dir=str(main_audit) if main_audit else None,
        main_audit_sha256=sha(main_audit/'independent_audit.json') if main_audit else None,
        protocol_sha256=sha(PROTOCOL),proposal_sha256=sha(PROPOSAL),script_sha256=sha(Path(__file__)),
        checkpoint_hashes=checkpoint_hashes,bootstrap_seed=BOOTSTRAP_SEED,bootstrap_draws=draws,
        wall_seconds=time.monotonic()-start,
        artifact_hashes={name:sha(out/name) for name in ('rows.csv','summary.json')})
    (out/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(summary))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--profile',choices=('smoke','dev'),default='dev')
    parser.add_argument('--main-audit-dir',type=Path)
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    analyze(args.run_dir.resolve(),out,args.profile,args.main_audit_dir.resolve() if args.main_audit_dir else None)

if __name__=='__main__':main()
