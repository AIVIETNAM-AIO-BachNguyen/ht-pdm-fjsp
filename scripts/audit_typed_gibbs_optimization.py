"""Frozen TRAIN/CFDEV distribution audit; no TEST performance or new physics."""
import argparse,csv,hashlib,json,shutil,subprocess,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
import typed_gibbs_reference as reference
from verify_typed_gibbs_development import helpers

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/bachng/ht-pdm-fjsp/artifacts/technician_typed_gibbs_confirmation_v1_20261017T122835996955Z')
AUDIT=Path('/home/bachng/ht-pdm-fjsp/artifacts/technician_typed_gibbs_confirmation_verification_v1_20261017T125736209944Z')
PLAN=ROOT/'docs/technician_typed_gibbs_optimization_audit_v1_plan.md'
CORE=reference.CORE;TRAINS=tuple(range(380300000,380300005));STEPS=(0,122880,245760,368640,491520)


def policy_statistics(probability,eligible):
    p=np.asarray(probability);eligible=np.asarray(eligible);assert p.shape==eligible.shape and eligible.dtype==bool and np.isfinite(p).all() and (p>=0).all() and (eligible.sum(-1)>0).all();np.testing.assert_allclose(p.sum(-1),1.,atol=1e-10,rtol=0);assert (p[~eligible]==0).all();count=eligible.sum(-1);inform=count>1;logp=np.zeros_like(p);np.log(p,out=logp,where=p>0);entropy=-(p*logp).sum(-1);ratio=np.zeros_like(entropy);ratio[inform]=entropy[inform]/np.log(count[inform]);assert (ratio>=-1e-10).all() and (ratio<=1+1e-10).all();ordered=np.sort(p,-1);gap=ordered[:,-1]-ordered[:,-2]
    return dict(states=len(p),informative_states=int(inform.sum()),mean_normalized_entropy=float(ratio[inform].mean()) if inform.any() else 0.,mean_mode_probability=float(p.max(-1)[inform].mean()) if inform.any() else 1.,near_deterministic_fraction=float((p.max(-1)[inform]>=.99).mean()) if inform.any() else 0.,ambiguous_mode_states=int((inform&(gap<=1e-10)).sum()))


def movement(before,after,eligible):
    assert before.shape==after.shape==eligible.shape;inform=eligible.sum(-1)>1;tv=np.abs(before-after).sum(-1)/2;one=np.sort(before,-1);two=np.sort(after,-1);unique=inform&(one[:,-1]-one[:,-2]>1e-10)&(two[:,-1]-two[:,-2]>1e-10);changed=before.argmax(-1)!=after.argmax(-1)
    return dict(late_mean_tv=float(tv[inform].mean()) if inform.any() else 0.,late_max_tv=float(tv.max()),unique_mode_comparisons=int(unique.sum()),unique_mode_changes=int((unique&changed).sum()),unique_mode_change_fraction=float(changed[unique].mean()) if unique.any() else 0.)


def decisions(late):
    candidate=[r for r in late if r['method']==CORE[0]];assert len(candidate)==5
    previous=np.mean([r['previous_cfdev_cost'] for r in candidate]);final=np.mean([r['final_cfdev_cost'] for r in candidate]);gain=float(1-final/previous);positive=sum(r['late_cfdev_cost_gain']>0 for r in candidate);tv=float(np.mean([r['late_mean_tv'] for r in candidate]));collapse={f"{r['method']}:{r['train_seed']}":bool(r['informative_states']>0 and r['near_deterministic_fraction']>=.9 and r['late_mean_tv']<=.01 and r['late_cfdev_cost_gain']<=.001) for r in late}
    return dict(gibbs_pooled_late_cfdev_gain=gain,gibbs_positive_late_replicas=positive,gibbs_mean_late_tv=tv,possible_probability_collapse=collapse,permit_designing_one_2x_budget_followup=bool(gain>=.005 and positive>=4 and tv>=.01),training_launch_authorized_by_this_artifact=False,interpretation='Exploratory engineering heuristic only; cannot establish optimization cause, improve a failed TEST verdict, or open N>4.')


def run(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);h=helpers();torch.set_num_threads(1);started=time.monotonic();manifest=dict(status='RUNNING',training_steps=0,new_physical_states=0,test_outcomes_used=False,test_performance_decoded=False,test_verdict_entry_only=True,goal_pass=False,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());inputs={}
    def write(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def guard():assert time.monotonic()-started<300,'5minute hardstop'
    write()
    try:
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source';assert manifest['git_commit']==subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=ROOT,text=True).strip(),'pushed source'
        m=h.read(SOURCE/'manifest.json');am=h.read(AUDIT/'manifest.json');assert m['status']=='COMPLETED' and m['audit_status']=='PASS' and not m['confirmation_screen_pass'] and am['status']=='COMPLETED' and not am['original_confirmation_pass'] and am['source_manifest_sha256']==h.sha(SOURCE/'manifest.json'),'verified negative entry verdict'
        manifest['source_manifest_sha256']=h.sha(SOURCE/'manifest.json');manifest['audit_manifest_sha256']=h.sha(AUDIT/'manifest.json')
        def bound(name):
            p=SOURCE/name;meta=m['output_bindings'][name];assert h.sha(p)==meta['sha256'] and p.stat().st_size==meta['bytes'],'input binding';inputs[name]=meta;return p
        choices=h.read(bound('checkpoint_choices.json'));blocks=[];origins=[]
        for t in TRAINS:
            meta=choices[f'{CORE[0]}:{t}']['observations'];assert meta['states']==48 and meta['seed']==380400000
            path=bound(meta['path']);assert h.sha(path)==meta['sha256']
            with np.load(path,allow_pickle=False) as f:block={k:f[k].copy() for k in f.files}
            assert all(len(v)==48 for v in block.values()) and block['state'].shape==(48,71);blocks.append(block);origins.extend(dict(train_seed=t,state_index=i,physical_cfdev_seed=380400000) for i in range(48))
        keys=set(blocks[0]);assert all(set(b)==keys for b in blocks);pool={k:np.concatenate([b[k] for b in blocks]) for k in keys};np.savez_compressed(out/'frozen_cfdev_probe_pool.npz',**pool);obs=[{k:v[i] for k,v in pool.items()} for i in range(240)];features=reference.features(obs).reshape(-1,89);legal=pool['pair_mask'];canonical=reference.tokens(h);all_p=np.empty((3,5,5,240,21));node=[];statistics={};eligibility=None
        dev=defaultdict(list)
        for r in h.rows(bound('development_episodes.csv')):
            if r['method'] in CORE:dev[r['method'],int(r['train_seed']),int(r['checkpoint_steps'])].append(r)
        progress=defaultdict(list)
        for r in h.rows(bound('training_progress.csv')):
            if r['algorithm'] in CORE:progress[r['algorithm'],int(r['train_seed'])].append(r)
        sourcefiles=[Path(__file__),ROOT/'scripts/typed_gibbs_reference.py',ROOT/'scripts/verify_broker_marginal_development.py',ROOT/'scripts/verify_typed_gibbs_development.py',PLAN];manifest['code_seal']={str(p.relative_to(ROOT)):h.sha(p) for p in sourcefiles};write()
        for ai,a in enumerate(tqdm(CORE,desc='Frozen policy families')):
            for ti,t in enumerate(TRAINS):
                for ci,step in enumerate(STEPS):
                    guard();name=f'N4/{a}/{t}/checkpoints/step_{step:09d}.pt';payload=torch.load(bound(name),map_location='cpu',weights_only=True);assert all(payload[k]==v for k,v in dict(protocol='technician_typed_gibbs_ppo_v1',algorithm=a,train_seed=t,steps=step).items());digest=hashlib.sha256()
                    for key,v in payload['state_dict'].items():digest.update(key.encode());digest.update(v.numpy().tobytes())
                    assert digest.hexdigest()==payload['parameter_sha256'];weights={key:v.numpy().astype(float) for key,v in payload['state_dict'].items()};utility=h.mlp(features,weights,'actor').reshape(240,4,2);r=reference.distribution(utility,legal,a,canonical);p=np.exp(r['logp']);all_p[ai,ti,ci]=p
                    if eligibility is None:eligibility=r['eligible']
                    else:assert np.array_equal(eligibility,r['eligible'])
                    if step==0:np.testing.assert_allclose(p,eligibility/eligibility.sum(-1)[:,None],atol=1e-10,rtol=0)
                    description=policy_statistics(p,eligibility);rows=dev[a,t,step];assert len(rows)==64 and {int(row['seed']) for row in rows}==set(range(380400000,380400064))
                    description.update(method=a,train_seed=t,checkpoint_steps=step,utility_min=float(utility.min()),utility_max=float(utility.max()),**{'cfdev_'+field:float(np.mean([float(row[field]) for row in rows])) for field in ('normalized_cost','availability','normalized_pending','normalized_failures')});node.append(description);statistics[a,t,step]=description
        late=[]
        for ai,a in enumerate(CORE):
            for ti,t in enumerate(TRAINS):
                before=statistics[a,t,368640];after=statistics[a,t,491520];rows=progress[a,t];assert len(rows)==160 and [int(r['learned_steps']) for r in rows]==list(range(3072,491521,3072));last=rows[-40:];late.append(dict(method=a,train_seed=t,previous_cfdev_cost=before['cfdev_normalized_cost'],final_cfdev_cost=after['cfdev_normalized_cost'],late_cfdev_cost_gain=1-after['cfdev_normalized_cost']/before['cfdev_normalized_cost'],informative_states=after['informative_states'],near_deterministic_fraction=after['near_deterministic_fraction'],final_normalized_entropy=after['mean_normalized_entropy'],**movement(all_p[ai,ti,3],all_p[ai,ti,4],eligibility),**{'last40_'+field+'_mean':float(np.mean([float(r[field]) for r in last])) for field in ('actor_update_norm','critic_update_norm','preclip_global_gradient_norm')}))
        np.savez_compressed(out/'matching_probabilities.npz',probabilities=all_p,eligible=eligibility,canonical_tokens=canonical,train_seeds=np.array(TRAINS),checkpoint_steps=np.array(STEPS));result=dict(classification='EXPLORATORY_FROZEN_OPTIMIZATION_AUDIT',policy_state_evaluations=18000,probe_states=240,probe_cfdev_physical_seeds=1,origins=origins,node_statistics=node,late_comparisons=late,readout=decisions(late),training_steps=0,new_physical_states=0,test_performance_decoded=False,goal_pass=False,limitation='Only one exogenous CFDEV seed and five candidate trajectories; fixed-state distribution descriptors do not prove expected cost, gradient cause or convergence.')
        (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');(out/'input_bindings.json').write_text(json.dumps(inputs,indent=2)+'\n')
        for name,meta in inputs.items():assert h.sha(SOURCE/name)==meta['sha256'],'input changed'
        for name,digest in manifest['code_seal'].items():assert h.sha(ROOT/name)==digest,'code/protocol changed'
        assert h.sha(SOURCE/'manifest.json')==manifest['source_manifest_sha256'] and h.sha(AUDIT/'manifest.json')==manifest['audit_manifest_sha256']
        for p in sourcefiles:
            target=out/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-started,readout=result['readout'],output_bindings={str(p.relative_to(out)):dict(sha256=h.sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'});write();print(json.dumps(dict(output=str(out),status='PASS',readout=result['readout'],goal_pass=False)));return result
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-started);write();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);run(parser.parse_args().output)
