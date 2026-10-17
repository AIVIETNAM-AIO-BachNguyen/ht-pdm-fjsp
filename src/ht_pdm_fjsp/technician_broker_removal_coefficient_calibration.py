"""Fresh fit/heldout scalar coefficient diagnostic; no policy/critic updates."""
import argparse,hashlib,json,multiprocessing as mp,shutil,subprocess,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_broker_removal_gradient_diagnostic as d

e=d.e;ROOT=Path(__file__).resolve().parents[2];VERSION='technician_broker_removal_coefficient_calibration_v1';PLAN=ROOT/'docs'/f'{VERSION}_plan.md';REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'


def coefficient(state,conditional):
    x=np.asarray(state,float);h=x-np.asarray(conditional,float);e.check(x.shape==h.shape and x.ndim==2 and len(x)>1 and np.isfinite(x).all() and np.isfinite(h).all(),'finite fit vectors');xc=x-x.mean(0);hc=h-h.mean(0);den=len(x)-1;v=float(np.square(xc).sum()/den);c=float((xc*hc).sum()/den);varh=float(np.square(hc).sum()/den);value=c/varh if varh>1e-20 else 0.;e.check(np.isfinite(value) and v>1e-20,'finite fit coefficient');return dict(coefficient=value,fit_state_variance=v,fit_control_variance=varh,fit_covariance=c,fit_optimistic_gain=c*c/(varh*v) if varh>1e-20 else 0.)


def evaluate_coefficient(state,conditional,value):
    x=np.asarray(state,float);b=np.asarray(conditional,float);h=x-b;mixed=x-value*h;v=d.trace_covariance(x);w=d.trace_covariance(mixed);hc=h-h.mean(0);xc=x-x.mean(0);quadratic=v-2*value*float((xc*hc).sum()/(len(x)-1))+value*value*d.trace_covariance(h);e.check(v>1e-20 and np.isclose(w,quadratic,atol=1e-9,rtol=1e-10),'heldout quadratic identity');return dict(gain=1-w/v,fixed_one_gain=1-d.trace_covariance(b)/v,state_variance=v,mixed_variance=w,control_variance=d.trace_covariance(h),covariance=float((xc*hc).sum()/(len(x)-1)))


def primary(records):
    e.check(len(records)==15 and {(r['train_seed'],r['steps']) for r in records}=={(t,s) for t in d.TRAINS for s in d.STEPS},'all15calibration nodes');replicas={};checks={}
    for t in d.TRAINS:
        rr=[r for r in records if r['train_seed']==t];gain=float(np.mean([r['evaluation']['gain'] for r in rr]));fixed=float(np.mean([r['evaluation']['fixed_one_gain'] for r in rr]));positive=sum(r['evaluation']['gain']>0 for r in rr);replicas[str(t)]=dict(mean_gain=gain,mean_fixed_one_gain=fixed,positive_checkpoints=positive);checks[str(t)]=dict(material_mean_gain=gain>=.05,majority_positive=positive>=3,better_than_fixed_one=gain>fixed)
    return replicas,checks


def node(job):
    source,outname,t,step,index,registry,start=job;torch.set_num_threads(1);directory=Path(outname)/f'{t}_{step}';directory.mkdir();r,h=d.reference()
    def guard():e.check(time.monotonic()-start<3600,'60minute hardstop');e.check(not (Path(outname)/'abort.json').exists(),'sibling technical abort')
    checkpoint=Path(source)/f'N4/{e.CORE[0]}/{t}/checkpoints/step_{step:09d}.pt';model=e.load_model(checkpoint,e.CORE[0],e.settings(),t,step).double();model.critic.requires_grad_(False);model.removal.requires_grad_(False);fit_seed=h.seed(registry['action_bases'][index],VERSION+':fit:actions');eval_seed=h.seed(registry['action_bases'][index],VERSION+':evaluation:actions');fit=d.collect(model,registry['fit_physical_seeds'],fit_seed,directory/'fit',guard,VERSION)
    with np.load(directory/'fit/gradients.npz',allow_pickle=False) as f:calibration=coefficient(f['state'],f['conditional'])
    if step==0:e.check(calibration['fit_control_variance']<=1e-20 and calibration['coefficient']==0,'initial control identically zero')
    calibration.update(fit_gradients_sha256=h.sha(directory/'fit/gradients.npz'),phase='sealed_before_evaluation_collection',source_checkpoint_sha256=h.sha(checkpoint));e.check(not (directory/'evaluation').exists(),'no heldout collection before fit seal');e.write_json(directory/'coefficient.json',calibration);seal=h.sha(directory/'coefficient.json');evaluation=d.collect(model,registry['evaluation_physical_seeds'],eval_seed,directory/'evaluation',guard,VERSION);e.check(h.sha(directory/'coefficient.json')==seal,'fit coefficient never refitted')
    with np.load(directory/'evaluation/gradients.npz',allow_pickle=False) as f:heldout=evaluate_coefficient(f['state'],f['conditional'],calibration['coefficient'])
    if step==0:
        heldout['raw_step0_gain']=heldout['gain'];e.check(abs(heldout['gain'])<1e-10,'initial heldout invariance');heldout['gain']=0.
    record=dict(train_seed=t,steps=step,coefficient=calibration,evaluation=heldout,fit=fit,evaluation_audit=evaluation,fit_action_seed=fit_seed,evaluation_action_seed=eval_seed,coefficient_sha256=seal,optimizer_updates=0);e.write_json(directory/'summary.json',record);return record


def seed_audit(registry,h):
    raw=[*registry['fit_physical_seeds'],*registry['evaluation_physical_seeds'],*registry['action_bases'],registry['analysis_seed'],*registry['engineering'].values()];historical=set();files={};own=h.sha(REGISTRY)
    def ints(x):
        if isinstance(x,dict):return set.union(set(),*(ints(v) for v in x.values()))
        if isinstance(x,list):return set.union(set(),*(ints(v) for v in x))
        return {x} if isinstance(x,int) and not isinstance(x,bool) else set()
    for p in [*(ROOT/'configs').glob('*seed_registry.json'),*Path('/home/bachng/ht-pdm-fjsp-worktrees').glob('*/configs/*seed_registry.json'),*Path('/home/bachng/ht-pdm-fjsp/configs').glob('*seed_registry.json')]:
        digest=h.sha(p)
        if digest!=own and digest not in files:historical|=ints(h.read(p));files[digest]=str(p)
    derived=[h.seed(s,f'{VERSION}:N4:physical') for s in (*registry['fit_physical_seeds'],*registry['evaluation_physical_seeds'],registry['engineering']['physical_seed'])]+[h.seed(s,f'{VERSION}:{role}:actions') for s in registry['action_bases'] for role in ('fit','evaluation')]+[h.seed(registry['engineering']['action_seed'],VERSION+':engineering:actions'),h.seed(registry['analysis_seed'],VERSION+':analysis')]
    e.check(len(raw)==len(set(raw))==402 and len(derived)==len(set(derived))==417 and not set(raw)&historical and not set(derived)&(historical|set(raw)),'new calibration seeds/streams disjoint');return dict(status='PASS',raw=402,derived=417,historical_registries=files,confirmation_states_constructed=False,limitation='Historical raw registries checked; all historical derived namespaces not exhaustively reconstructed.')


def run(source,verification,output):
    source=Path(source).resolve();verification=Path(verification).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);r,h=d.reference();start=time.monotonic();sm=h.read(source/'manifest.json');vm=h.read(verification/'manifest.json');v=h.read(verification/'verification.json');manifest=dict(status='RUNNING',protocol=VERSION,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source=str(source),source_manifest_sha256=h.sha(source/'manifest.json'),verification_manifest_sha256=h.sha(verification/'manifest.json'),goal_pass=False,test_opened=False,optimizer_updates=0);e.write_json(out/'manifest.json',manifest)
    def bindings():
        e.check(h.sha(source/'manifest.json')==manifest['source_manifest_sha256'] and h.sha(verification/'manifest.json')==manifest['verification_manifest_sha256'],'source manifests unchanged')
        for root,entries in ((source,sm['output_bindings']),(verification,vm['outputs'])):
            for name,meta in entries.items():e.check(h.sha(root/name)==meta['sha256'],'bound source '+name)
    try:
        e.check(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean committed source');e.check(sm['status']=='COMPLETED' and sm['git']['commit']==r.COMMIT and vm['status']=='COMPLETED' and v['status']=='PASS' and v['source_manifest_sha256']==manifest['source_manifest_sha256'] and not sm['development_screen_pass'] and not sm['test_opened'],'verified negative cost source');bindings();registry=h.read(REGISTRY);audit=seed_audit(registry,h);e.write_json(out/'seed_audit.json',audit);shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'seed_registry.json');shutil.copy2(__file__,out/'calibration.py');shutil.copy2(d.__file__,out/'frozen_diagnostic.py');e.write_json(out/'resolved_config.json',dict(nodes=[dict(train_seed=t,steps=s) for t in d.TRAINS for s in d.STEPS],fit_episodes=128,evaluation_episodes=256,workers=3,max_seconds=3600,dtype='float64',coefficient_refits=0,optimizer_updates=0,material_gain=.05,majority_positive=3));seals={str(p):h.sha(p) for p in (Path(__file__),Path(d.__file__),PLAN,REGISTRY,ROOT/'scripts/verify_broker_removal_development.py',ROOT/'scripts/verify_broker_marginal_development.py')};records=[]
        for name,digest in sm['source_seal'].items():
            if name.startswith('source_snapshot/'):
                path=Path(__file__).parent/Path(name).name;e.check(h.sha(path)==digest,'frozen production module '+name);seals[str(path)]=digest
        e.write_json(out/'code_seals.json',seals)
        with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
            jobs=[(str(source),str(out),t,s,i,registry,start) for i,(t,s) in enumerate((t,s) for t in d.TRAINS for s in d.STEPS)];futures=[pool.submit(node,j) for j in jobs]
            try:
                for future in tqdm(as_completed(futures),total=15,desc='15FULL fresh fit/heldout nodes'):e.check(time.monotonic()-start<3600,'60minute hardstop');records.append(future.result());e.write_json(out/'records.partial.json',records)
            except Exception as exc:
                e.write_json(out/'abort.json',dict(error=str(exc)))
                for future in futures:future.cancel()
                raise
        records.sort(key=lambda r:(r['train_seed'],r['steps']));replicas,checks=primary(records);e.check(sum(r[k]['episodes'] for r in records for k in ('fit','evaluation_audit'))==5760 and sum(r[k]['policy_decisions'] for r in records for k in ('fit','evaluation_audit'))==276480 and sum(r[k]['physical_transitions'] for r in records for k in ('fit','evaluation_audit'))==345600,'exact calibration budget');summary=dict(classification='FROZEN_FRESH_FIT_HELDOUT_CALIBRATION_ONLY',records=records,replicas=replicas,checks=checks,mechanism_screen_pass=all(v for row in checks.values() for v in row.values()),source_cost_verdict_overridden=False,goal_pass=False,test_opened=False,optimizer_updates=0,episodes=5760,policy_decisions=276480,physical_transitions=345600,limitation='Point diagnostic, no scientific CI/cost improvement or finite PPO update guarantee; no actor/critic weights changed. Source negative verdict retained.');bindings()
        for path,digest in seals.items():e.check(h.sha(Path(path))==digest,'live code unchanged')
        e.write_json(out/'summary.json',summary);manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,mechanism_screen_pass=summary['mechanism_screen_pass'],output_bindings={str(p.relative_to(out)):dict(sha256=h.sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'});e.write_json(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),mechanism_screen_pass=summary['mechanism_screen_pass'])));return summary
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);e.write_json(out/'manifest.json',manifest);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--verification',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.verification,a.output)
