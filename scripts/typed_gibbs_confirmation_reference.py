"""Independent confirmation statistics and chunked raw-MLP matching reference."""
import numpy as np
import torch
import typed_gibbs_reference as matching
from verify_typed_gibbs_development import independent_initial,seed_contract,helpers,endpoint_gates as point_gates

FAMILY='technician_typed_gibbs_ppo_v1'
CORE=('typed_gibbs_ppo','typed_myopic_ppo','reverse_typed_myopic_ppo')
ARMS=(*CORE,'ppo','normalized_ppo','mappo','full_mappo')
REFS=('ppo','normalized_ppo')
FIELDS=('normalized_cost','availability','normalized_pending','normalized_failures')
ALPHA=.05/(7*8)/12
NAMES=tuple('gain_vs_'+a for a in ARMS[1:])+tuple(f'{metric}_delta_vs_{a}' for a in REFS for metric in FIELDS[1:])


def cartesian_means(values,train_indices,seed_indices):
    """Materialize each resampled Cartesian product; no count-weighted einsum."""
    v=np.asarray(values,dtype=np.float64);ti=np.asarray(train_indices);si=np.asarray(seed_indices)
    if v.ndim!=4 or v.shape[0]!=7 or v.shape[-1]!=4 or ti.ndim!=2 or si.ndim!=2 or len(ti)!=len(si):raise ValueError('reference shapes')
    if ti.shape[1]!=v.shape[1] or si.shape[1]!=v.shape[2] or (ti<0).any() or (ti>=v.shape[1]).any() or (si<0).any() or (si>=v.shape[2]).any():raise ValueError('reference index ranges')
    selected=v[:,ti[:,:,None],si[:,None,:],:]
    return selected.mean(axis=(2,3)).transpose(1,0,2)


def effects(means):
    result=[]
    for row in np.asarray(means):
        c=row[0];r=[1-c[0]/row[i,0] for i in range(1,7)]
        for a in (3,4):r.extend((c[1]-row[a,1],c[2]-row[a,2],c[3]-row[a,3]))
        result.append(r)
    return np.asarray(result,dtype=np.float64)


def bounds(points,resamples,alpha=ALPHA):
    result={}
    for i,name in enumerate(NAMES):
        upper=name.startswith('normalized_pending') or name.startswith('normalized_failures')
        threshold=.025 if name.startswith('normalized_pending') else .25 if name.startswith('normalized_failures') else .005 if i==0 else 0. if i in (4,5) else -.01
        strict=i in (4,5);margin=1e-12 if strict else 0.;value=float(np.quantile(resamples[:,i],1-alpha if upper else alpha,method='linear'))
        passed=value<=threshold if upper else value>threshold+margin if strict else value>=threshold
        result[name]=dict(point=float(points[i]),bound=value,orientation='upper' if upper else 'lower',threshold=threshold,strict=strict,numeric_margin=margin,passed=bool(passed),one_sided_alpha=alpha)
    return result


def regenerate_indices(h,analysis_seed,t,s,draws=200000,chunk=256):
    seed=h.seed(analysis_seed,f'maintenance_balanced_algorithm_benchmark_v1:{FAMILY}:bootstrap');rng=np.random.default_rng(seed)
    if type(rng.bit_generator).__name__!='PCG64':raise AssertionError('registered PCG64')
    for start in range(0,draws,chunk):
        size=min(chunk,draws-start);ti=rng.integers(0,t,(size,t),dtype=np.int64);si=rng.integers(0,s,(size,s),dtype=np.int64)
        yield start,ti,si


def neural(data,arm,train,source,choices,config,h,guard=lambda:None,chunk=512):
    """Check every recorded actor outcome/value with separate mathematical code."""
    from ht_pdm_fjsp import technician_typed_gibbs_experiment as production
    w=h.weights(source,choices[f'{arm}:{train}']);model=production.load_model(source/choices[f'{arm}:{train}']['path'],arm,config,train,config['env_steps']);result=dict(states=len(data),greedy_deficit=0.,value_max_error=0.,log_max_error=0.,entropy_max_error=0.,utility_max_error=0.)
    for start in range(0,len(data),chunk):
        guard();part=data[start:start+chunk];obs=[r[0] for r in part];actions=np.array([r[1] for r in part]);state=np.stack([o['state'] for o in obs]).astype(float);unit=1. if arm=='ppo' else 115.2;prefix='critic' if arm in CORE or arm=='ppo' else 'critic.net';expected_v=h.mlp(state,w,prefix).ravel()*unit
        if arm not in CORE:
            result['greedy_deficit']=max(result['greedy_deficit'],h.verify_actor(obs,actions,arm,w))
            with torch.no_grad():actual_v=model.critic(torch.tensor(state,dtype=torch.float32)).squeeze(-1).numpy()
        else:
            canonical=matching.tokens(h);utility=h.mlp(matching.features(obs).reshape(-1,89),w,'actor').reshape(-1,4,2);legal=np.stack([o['pair_mask'] for o in obs]);reference=matching.distribution(utility,legal,arm,canonical);same=(canonical[None]==actions[:,None]).all(-1);h.check((same.sum(-1)==1).all(),'canonical action in independent support');ix=same.argmax(-1);lp=reference['logp'][np.arange(len(part)),ix];deficit=reference['logp'].max(-1)-lp;h.check((deficit/(5e-6+1e-6*np.abs(lp))).max()<=1,'independent physical MAP');result['greedy_deficit']=max(result['greedy_deficit'],float(deficit.max()))
            with torch.no_grad():actual_log,actual_h,v=model.evaluate(production.learner.tensors(obs),torch.tensor(actions,dtype=torch.long))
            actual_v=v.numpy();np.testing.assert_allclose(actual_log.numpy(),lp,atol=5e-6,rtol=1e-6);np.testing.assert_allclose(actual_h.numpy()*4,reference['entropy'],atol=5e-6,rtol=1e-6)
            records=[r[2] for r in part];recorded_u=np.array([r['utilities'] for r in records]);np.testing.assert_allclose(recorded_u,utility,atol=5e-6,rtol=1e-6);np.testing.assert_allclose([r['physical_logp'] for r in records],lp,atol=5e-6,rtol=1e-6);np.testing.assert_allclose([r['physical_entropy'] for r in records],reference['entropy'],atol=5e-6,rtol=1e-6)
            result['utility_max_error']=max(result['utility_max_error'],float(np.abs(recorded_u-utility).max()));result['log_max_error']=max(result['log_max_error'],float(np.abs(actual_log.numpy()-lp).max()));result['entropy_max_error']=max(result['entropy_max_error'],float(np.abs(actual_h.numpy()*4-reference['entropy']).max()))
        np.testing.assert_allclose(actual_v,expected_v,atol=2e-5*115.2,rtol=1e-6);result['value_max_error']=max(result['value_max_error'],float(abs(actual_v-expected_v).max()))
    return result
