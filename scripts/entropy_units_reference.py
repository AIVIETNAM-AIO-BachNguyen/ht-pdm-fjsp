"""Independent NumPy matching reference for six entropy arms; verifier only."""
import numpy as np
import torch
import typed_gibbs_reference as base

CORE=('gibbs_joint_entropy_ppo','myopic_joint_entropy_ppo','reverse_joint_entropy_ppo',
      'gibbs_machine_entropy_ppo','myopic_machine_entropy_ppo','reverse_machine_entropy_ppo')
BASE={a:base.CORE[i%3] for i,a in enumerate(CORE)}


def enrich_observation(observation,h,arm=None):
    o=dict(observation);workers=o['technicians'].copy();workers[:,2]=0
    o['local']=np.concatenate([o['machines'],np.tile(workers.ravel(),(4,1)),o['edges'].reshape(4,-1),np.tile(o['global_features'],(4,1)),np.eye(4,dtype=np.float32)],-1).astype(np.float32)
    o['local_mask']=np.c_[np.ones(4,bool),o['pair_mask']]
    if arm in ('ppo','normalized_ppo'):o['joint_mask']=np.array([all(o['pair_mask'][m,j] for m,j in a) for a in h.baseline_catalogue()])
    if arm=='full_mappo':o['full_local']=np.concatenate([np.tile(o['state'],(4,1)),np.eye(4,dtype=np.float32)],-1)
    return o



def baseline_diagnostics(action,arm,pairs):
    counts=np.bincount(np.asarray(action,dtype=int),minlength=3)[1:] if arm in ('mappo','full_mappo') else np.zeros(2,dtype=int)
    return dict(conflicts=int((counts>1).sum()),rejected_proposals=int(np.maximum(counts-1,0).sum()),executed_starts=len(pairs))


def physical_reference(observations,weights,arm,h):
    utility=h.mlp(base.features(observations).reshape(-1,89),weights,'actor').reshape(-1,4,2)
    legal=np.stack([o['pair_mask'] for o in observations]);canonical=base.tokens(h)
    return utility,legal,canonical,base.distribution(utility,legal,BASE[arm],canonical)


def probe_metrics(observations,reference,actions,arm):
    entropy=reference['entropy'];count=reference['eligible'].sum(-1);inform=count>1
    mode=reference['logp'].max(-1)[inform];normalized=entropy[inform]/np.log(count[inform])
    available=np.array([o['technicians'][:,0].sum() for o in observations],dtype=float)
    remaining=available.copy();before=[]
    for i in (range(3,-1,-1) if BASE[arm]==base.CORE[2] else range(4)):
        before.append(remaining.copy());remaining-=np.asarray(actions)[:,i]>0
    return dict(states=len(observations),informative_states=int(inform.sum()),mean_normalized_entropy=float(normalized.mean()) if inform.any() else 0.,mean_mode_probability=float(np.exp(mode).mean()) if inform.any() else 1.,near_deterministic_fraction=float(np.mean(np.exp(mode)>=.99)) if inform.any() else 0.,mean_initial_available_workers=float(available.mean()),mean_available_workers_before_map_stage=float(np.stack(before,-1).mean()),entropy_aggregation='joint' if arm in CORE[:3] else 'machine',base_density=BASE[arm])


def verify_neural(data,arm,train,source,choices,config,h):
    observations=[r[0] for r in data];actions=np.asarray([r[1] for r in data]);w=h.weights(source,choices[f'{arm}:{train}'])
    if arm not in CORE:
        deficit=h.verify_actor(observations,actions,arm,w)
        from ht_pdm_fjsp import technician_typed_gibbs_entropy_units_experiment as production
        model=production.load_model(source/choices[f'{arm}:{train}']['path'],arm,config,train,491520)
        with torch.no_grad():value=model.evaluate(production.learner.tensors([enrich_observation(o,h,arm) for o in observations]),torch.tensor(actions,dtype=torch.long))[2].numpy()
        # PPO's literal raw critic has unit1; other baselines preserve original115.2.
        prefix='critic' if arm=='ppo' else 'critic.net';unit=1 if arm=='ppo' else 115.2
        expected=h.mlp(np.stack([o['state'] for o in observations]).astype(float),w,prefix).ravel()*unit
        np.testing.assert_allclose(value,expected,atol=2e-5*unit,rtol=1e-6)
        return dict(states=len(data),greedy_deficit=deficit,value_max_error=float(abs(value-expected).max()))
    utility,legal,canonical,reference=physical_reference(observations,w,arm,h)
    same=(canonical[None]==actions[:,None,:]).all(-1);h.check(same.sum(-1).min()==1,'canonical physical actions');ix=same.argmax(-1);lp=reference['logp'][np.arange(len(data)),ix];deficit=reference['logp'].max(-1)-lp;h.check((deficit/(5e-6+1e-6*np.abs(lp))).max()<=1,'manual physical MAP')
    from ht_pdm_fjsp import technician_typed_gibbs_entropy_units_experiment as production
    model=production.load_model(source/choices[f'{arm}:{train}']['path'],arm,config,train,491520);batch=production.learner.tensors(observations)
    with torch.no_grad():logs,entropy,value=model.evaluate(batch,torch.tensor(actions,dtype=torch.long))
    scale=1 if arm in CORE[:3] else 4
    np.testing.assert_allclose(logs.numpy(),lp,atol=5e-6,rtol=1e-6);np.testing.assert_allclose(entropy.numpy()*scale,reference['entropy'],atol=5e-6,rtol=1e-6)
    if len(data[0])>2:
        records=[r[2] for r in data];np.testing.assert_allclose(np.array([r['utilities'] for r in records]),utility,atol=5e-6,rtol=1e-6);np.testing.assert_allclose([r['physical_logp'] for r in records],lp,atol=5e-6,rtol=1e-6);np.testing.assert_allclose([r['physical_entropy'] for r in records],reference['entropy'],atol=5e-6,rtol=1e-6)
    expected_value=h.mlp(np.stack([o['state'] for o in observations]).astype(float),w,'critic').ravel()*115.2;np.testing.assert_allclose(value.numpy(),expected_value,atol=2e-5*115.2,rtol=1e-6)
    return dict(states=len(data),greedy_deficit=float(deficit.max()),log_max_error=float(np.abs(logs.numpy()-lp).max()),entropy_max_error=float(np.abs(entropy.numpy()*scale-reference['entropy']).max()),value_max_error=float(abs(value.numpy()-expected_value).max()),mechanism=base.mechanism(utility,legal,actions,canonical) if BASE[arm]==base.CORE[0] else None)
