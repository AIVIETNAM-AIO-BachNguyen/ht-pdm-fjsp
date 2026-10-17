"""Read-only fit coefficient and independent dense gradient heldout verification."""
import argparse,hashlib,importlib.util,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1];VERSION='technician_broker_removal_coefficient_calibration_v1';COMMIT='a2ce6366aaab8d97a6c201052ad2224ec3d6bb1c';TRAINS=tuple(range(330000000,330000003));STEPS=(0,122880,245760,368640,491520)
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def covariance(x,y):
    a=x-x.mean(0);b=y-y.mean(0);return float(np.einsum('ij,ij->',a,b)/(len(x)-1))
def fit_and_evaluate(fit_state,fit_conditional,evaluation_state,evaluation_conditional):
    h=fit_state-fit_conditional;den=covariance(h,h);value=covariance(fit_state,h)/den if den>1e-20 else 0.;v=covariance(evaluation_state,evaluation_state);mixed=evaluation_state-value*(evaluation_state-evaluation_conditional);w=covariance(mixed,mixed);assert v>1e-20 and np.isfinite(value) and np.isfinite(w);return dict(coefficient=value,gain=1-w/v,fixed_one_gain=1-covariance(evaluation_conditional,evaluation_conditional)/v,state_variance=v,mixed_variance=w,control_variance=covariance(evaluation_state-evaluation_conditional,evaluation_state-evaluation_conditional),covariance=covariance(evaluation_state,evaluation_state-evaluation_conditional))


def verify_case(directory,seeds,action_seed,weights,record,h,r,g,manual_logp,guard):
    arrays={};observations={};gradients={};size=len(seeds)
    for filename,target in (('arrays.npz',arrays),('observations.npz',observations),('gradients.npz',gradients)):
        with np.load(directory/filename,allow_pickle=False) as f:target.update({k:f[k] for k in f.files})
    assert set(gradients)==set(g.KINDS) and all(v.shape==(size,11141) for v in gradients.values());count=g.group_physics(directory,observations,arrays,seeds,h,guard,VERSION);obs=[{k:v[i] for k,v in observations.items()} for i in range(size*48)];dense=r.dense_baselines(obs,arrays['actions'],weights,h);value=h.mlp(observations['state'].astype(float),weights,'critic').ravel()*115.2;g.close(arrays['value'],value,'V');g.close(arrays['actual'],dense['actual'],'actualb');g.close(arrays['conditional'],dense['mixed'],'mixedB');g.close(arrays['logs'],dense['factors'],'logs');lp=dense['logp'].reshape(size,48,2,5);rng=torch.Generator().manual_seed(action_seed)
    for tick in range(48):
        expected=torch.stack([torch.multinomial(torch.tensor(np.exp(lp[:,tick,j])),1,generator=rng).squeeze(-1) for j in range(2)],-1).numpy();assert np.array_equal(expected,arrays['actions'].reshape(size,48,2)[:,tick])
    errors={k:0. for k in g.KINDS}
    for episode in tqdm(range(size),desc='Independent full gradient replay',leave=False):
        guard();sl=slice(episode*48,(episode+1)*48);reference=g.dense_gradients(obs[sl],arrays['actions'][sl],arrays['returns'][sl],value[sl],dense['actual'][sl],dense['mixed'][sl],weights,h,manual_logp)
        for kind in g.KINDS:errors[kind]=max(errors[kind],g.close(gradients[kind][episode],reference[kind],kind,True))
    assert record['episodes']==size and record['physical_transitions']==count and record['policy_decisions']==size*48 and record['optimizer_updates']==0
    for kind,vectors in gradients.items():h.near(covariance(vectors,vectors),record['variance'][kind],'raw covariance')
    h.near(1-covariance(gradients['conditional'],gradients['conditional'])/covariance(gradients['state'],gradients['state']),record['gain_vs_state'],'raw baseline gain')
    return gradients,dict(episodes=size,physical_transitions=count,gradient_vectors=size*3,gradient_max_errors=errors)


def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);g=module('calibration_dense_gradient','verify_broker_removal_gradient_diagnostic.py');r=module('calibration_dense_baseline','verify_broker_removal_development.py');h=r.helpers();entropy=module('calibration_manual_actor','verify_broker_entropy_alignment.py');torch.set_num_threads(1);start=time.monotonic();m=h.read(source/'manifest.json');manifest=dict(status='RUNNING',source=str(source),source_manifest_sha256=h.sha(source/'manifest.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),goal_pass=False,test_opened=False);(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def guard():assert time.monotonic()-start<1800,'30minute hardstop'
    def bindings():
        assert h.sha(source/'manifest.json')==manifest['source_manifest_sha256']
        for name,meta in m['output_bindings'].items():guard();assert h.sha(source/name)==meta['sha256'] and (source/name).stat().st_size==meta['bytes'],name
    try:
        assert m['status']=='COMPLETED' and m['audit_status']=='PASS' and m['protocol']==VERSION and m['git_commit']==COMMIT and m['optimizer_updates']==0 and not m['test_opened'];bindings();s=h.read(source/'summary.json');registry=h.read(source/'seed_registry.json');assert registry['fit_physical_seeds']==list(range(350100000,350100128)) and registry['evaluation_physical_seeds']==list(range(350200000,350200256)) and registry['action_bases']==list(range(350000000,350000015));assert len(s['records'])==15 and {(r['train_seed'],r['steps']) for r in s['records']}=={(t,step) for t in TRAINS for step in STEPS};records=[];budget=dict(episodes=0,physical_transitions=0,gradient_vectors=0)
        for index,(t,step) in enumerate(tqdm([(t,step) for t in TRAINS for step in STEPS],desc='15independent fit/heldout node audits')):
            guard();record=next(v for v in s['records'] if v['train_seed']==t and v['steps']==step);directory=source/f'{t}_{step}';checkpoint=Path(m['source'])/f'N4/removal_rb_ppo/{t}/checkpoints/step_{step:09d}.pt';assert h.sha(checkpoint)==record['coefficient']['source_checkpoint_sha256'];payload=torch.load(checkpoint,map_location='cpu',weights_only=True);assert payload['steps']==step and payload['train_seed']==t and payload['algorithm']=='removal_rb_ppo';weights={k:v.numpy().astype(float) for k,v in payload['state_dict'].items()};digest=hashlib.sha256()
            for k,v in weights.items():digest.update(k.encode());digest.update(v.tobytes())
            assert digest.hexdigest()==record['fit']['parameter_sha256']==record['evaluation_audit']['parameter_sha256'];data={};cases={}
            for phase,seeds,key in (('fit',registry['fit_physical_seeds'],'fit'),('evaluation',registry['evaluation_physical_seeds'],'evaluation_audit')):
                action_seed=h.seed(registry['action_bases'][index],f'{VERSION}:{phase}:actions');assert action_seed==record[phase+'_action_seed'];data[phase],cases[phase]=verify_case(directory/phase,seeds,action_seed,weights,record[key],h,r,g,entropy.manual_logp,guard)
                for k in budget:budget[k]+=cases[phase][k]
            coefficient=h.read(directory/'coefficient.json');assert coefficient==record['coefficient'] and h.sha(directory/'coefficient.json')==record['coefficient_sha256'] and coefficient['fit_gradients_sha256']==h.sha(directory/'fit/gradients.npz') and coefficient['phase']=='sealed_before_evaluation_collection';values=fit_and_evaluate(data['fit']['state'],data['fit']['conditional'],data['evaluation']['state'],data['evaluation']['conditional']);h.near(values['coefficient'],coefficient['coefficient'],'fit-only coefficient');e=record['evaluation']
            if step==0:assert values['coefficient']==0 and abs(values['gain'])<1e-10;values['gain']=0.
            for k in ('gain','fixed_one_gain','state_variance','mixed_variance','control_variance','covariance'):h.near(values[k],e[k],'heldout '+k)
            records.append(dict(train_seed=t,steps=step,**values,cases=cases))
        checks={};replicas={}
        for t in TRAINS:
            rr=[r for r in records if r['train_seed']==t];gain=float(np.mean([r['gain'] for r in rr]));fixed=float(np.mean([r['fixed_one_gain'] for r in rr]));positive=sum(r['gain']>0 for r in rr);replicas[str(t)]=dict(mean_gain=gain,mean_fixed_one_gain=fixed,positive_checkpoints=positive);checks[str(t)]=dict(material_mean_gain=gain>=.05,majority_positive=positive>=3,better_than_fixed_one=gain>fixed)
            for key,value in replicas[str(t)].items():h.near(value,s['replicas'][str(t)][key],'replica '+key)
        passed=all(v for row in checks.values() for v in row.values());assert budget==dict(episodes=5760,physical_transitions=345600,gradient_vectors=17280) and checks==s['checks'] and passed==s['mechanism_screen_pass']==m['mechanism_screen_pass'] and s['optimizer_updates']==0 and not s['source_cost_verdict_overridden'] and not s['goal_pass'];bindings();result=dict(status='PASS',source=str(source),source_manifest_sha256=manifest['source_manifest_sha256'],records=records,replicas=replicas,checks=checks,mechanism_screen_pass=passed,budget=budget,gradient_coordinates=192516480,optimizer_updates=0,goal_pass=False,test_opened=False);(out/'verification.json').write_text(json.dumps(result,indent=2));shutil.copy2(__file__,out/'verifier.py');shutil.copy2(ROOT/'scripts/verify_broker_removal_gradient_diagnostic.py',out/'gradient_reference.py');shutil.copy2(ROOT/'docs/technician_broker_removal_calibration_verification_v1_plan.md',out/'protocol.md');manifest.update(status='COMPLETED',elapsed_seconds=time.monotonic()-start,outputs={p.name:dict(sha256=h.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(output=str(out),status='PASS',mechanism_screen_pass=passed)))
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.output)
