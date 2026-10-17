"""Independent NumPy exhaustive labeled matching reference, verifier ONLY."""
import numpy as np

CORE=('typed_gibbs_ppo','typed_myopic_ppo','reverse_typed_myopic_ppo')


def tokens(h):
    result=np.zeros((21,4),int)
    for ix,pairs in enumerate(h.baseline_catalogue()):
        for m,j in pairs:result[ix,m]=j+1
    return result


def decode(action,legal):
    u=np.asarray(action);assert u.shape==(4,) and u.dtype.kind in 'iu' and ((u>=0)&(u<=2)).all()
    pairs=tuple((m,int(j)-1) for m,j in enumerate(u) if j)
    assert len({j for m,j in pairs})==len(pairs) and all(legal[m,j] for m,j in pairs)
    return pairs,dict(conflicts=0,rejected_proposals=0,executed_starts=len(pairs))


def features(observations):
    b=len(observations);state=np.stack([o['state'] for o in observations]).astype(float);machine=np.stack([o['machines'] for o in observations]).astype(float);profiles=np.stack([o['edges'][...,:3].reshape(4,6) for o in observations]).astype(float);free=np.stack([o['technicians'][:,0] for o in observations]).astype(float)
    return np.concatenate([np.broadcast_to(state[:,None,:],(b,4,71)),machine,profiles,np.broadcast_to(np.eye(4)[None],(b,4,4)),np.broadcast_to(free[:,None,:],(b,4,2))],-1)


def distribution(utility,legal,arm,canonical):
    """Every physical density. Myopic counts explicit legal prefix completions."""
    b=len(utility);eligible=np.ones((b,21),bool);energy=np.zeros((b,21))
    for m in range(4):
        t=canonical[:,m];selected=t>0;eligible[:,selected]&=legal[:,m,t[selected]-1];energy[:,selected]+=utility[:,m,t[selected]-1]
    weighted=np.where(eligible,energy,-np.inf)
    if arm==CORE[0]:lp=weighted-np.logaddexp.reduce(weighted,-1)[:,None]
    else:
        assert arm in CORE[1:];lp=np.zeros((b,21));consistent=np.broadcast_to(eligible[:,None,:],(b,21,21)).copy()
        for m in (range(3,-1,-1) if arm==CORE[2] else range(4)):
            counts=np.stack([(consistent&(canonical[None,None,:,m]==t)).sum(-1) for t in range(3)],-1)
            current=np.concatenate([np.zeros((b,1)),utility[:,m]],-1);logits=np.where(counts>0,np.log(np.maximum(counts,1))+current[:,None,:],-np.inf);norm=np.logaddexp.reduce(logits,-1);chosen=np.broadcast_to(canonical[None,:,m,None],(b,21,1));selected=np.take_along_axis(logits,chosen,-1).squeeze(-1)
            # Invalid target traces can exhaust their prefix; final support masks them.
            lp+=np.where(np.isfinite(norm),np.where(np.isfinite(selected),selected,0)-np.where(np.isfinite(norm),norm,0),0)
            consistent&=canonical[None,None,:,m]==canonical[None,:,None,m]
        lp=np.where(eligible,lp,-np.inf)
    assert np.isfinite(lp[eligible]).all();np.testing.assert_allclose(np.exp(lp).sum(-1),1.,atol=1e-10,rtol=0)
    safe=np.where(eligible,lp,0);entropy=-(np.exp(lp)*safe).sum(-1)
    return dict(logp=lp,entropy=entropy,eligible=eligible,energy=energy)


def mechanism(utility,legal,actions,canonical):
    g=distribution(utility,legal,CORE[0],canonical);a=distribution(utility,legal,CORE[1],canonical);r=distribution(utility,legal,CORE[2],canonical);index=(canonical[None]==actions[:,None,:]).all(-1).argmax(-1);rows=np.arange(len(utility));deficit=a['logp'].max(-1)-a['logp'][rows,index];reverse=r['logp'].max(-1)-r['logp'][rows,index];consistent=g['eligible'].copy();past_energy=np.zeros(len(utility));tvs=[];spans=[];choice_stages=0
    for m in range(4):
        pg=[];pm=[];messages=[];counts=[]
        for t in range(3):
            mask=consistent&(canonical[None,:,m]==t);count=mask.sum(-1);counts.append(count)
            pg.append(np.where(mask,np.exp(g['logp']),0).sum(-1));pm.append(np.where(mask,np.exp(a['logp']),0).sum(-1))
            current=np.zeros(len(utility)) if t==0 else utility[:,m,t-1];future=np.where(mask,g['energy']-past_energy[:,None]-current[:,None],-np.inf);messages.append(np.logaddexp.reduce(future,-1)-np.log(np.maximum(count,1)))
        pg=np.stack(pg,-1);pm=np.stack(pm,-1);pg/=pg.sum(-1)[:,None];pm/=pm.sum(-1)[:,None];tvs.extend((np.abs(pg-pm).sum(-1)/2).tolist());messages=np.stack(messages,-1);allowed=np.stack(counts,-1)>0;span=np.where(allowed,messages,-np.inf).max(-1)-np.where(allowed,messages,np.inf).min(-1);spans.extend(span.tolist());choice_stages+=int((allowed.sum(-1)>=2).sum());chosen=actions[:,m];past_energy+=np.where(chosen>0,utility[np.arange(len(utility)),m,np.maximum(chosen-1,0)],0);consistent&=canonical[None,:,m]==chosen[:,None]
    tolerance=5e-6+1e-6*np.abs(a['logp'][rows,index]);rtol=5e-6+1e-6*np.abs(r['logp'][rows,index])
    return dict(states=len(utility),candidate_action_outside_myopic_map=int((deficit>tolerance).sum()),candidate_action_outside_reverse_map=int((reverse>rtol).sum()),mean_myopic_map_log_deficit=float(deficit.mean()),conditional_stages=len(tvs),stages_with_multiple_choices=choice_stages,mean_conditional_tv=float(np.mean(tvs)),max_conditional_tv=float(max(tvs)),mean_future_message_span=float(np.mean(spans)),max_future_message_span=float(max(spans)))


def verify_neural(data,arm,train,source,choices,config,h):
    observations=[r[0] for r in data];actions=[r[1] for r in data];w=h.weights(source,choices[f'{arm}:{train}'])
    if arm not in CORE:return dict(states=len(data),greedy_deficit=h.verify_actor(observations,actions,arm,w))
    canonical=tokens(h);x=features(observations);utility=h.mlp(x.reshape(-1,89),w,'actor').reshape(-1,4,2);legal=np.stack([o['pair_mask'] for o in observations]);reference=distribution(utility,legal,arm,canonical);u=np.asarray(actions);same=(canonical[None]==u[:,None,:]).all(-1);assert same.sum(-1).min()==1;ix=same.argmax(-1);rows=np.arange(len(data));lp=reference['logp'][rows,ix];deficit=reference['logp'].max(-1)-lp;assert (deficit/(5e-6+1e-6*np.abs(lp))).max()<=1,'manual physical MAP'
    from ht_pdm_fjsp import technician_typed_gibbs_experiment as production
    import torch
    model=production.load_model(source/choices[f'{arm}:{train}']['path'],arm,config,train,491520);batch=production.learner.tensors(observations)
    with torch.no_grad():logs,entropy,value=model.evaluate(batch,torch.tensor(u,dtype=torch.long))
    np.testing.assert_allclose(logs.numpy(),lp,atol=5e-6,rtol=1e-6);np.testing.assert_allclose(entropy.numpy()*4,reference['entropy'],atol=5e-6,rtol=1e-6)
    if len(data[0])>2:
        records=[r[2] for r in data];np.testing.assert_allclose(np.array([r['utilities'] for r in records]),utility,atol=5e-6,rtol=1e-6);np.testing.assert_allclose([r['physical_logp'] for r in records],lp,atol=5e-6,rtol=1e-6);np.testing.assert_allclose([r['physical_entropy'] for r in records],reference['entropy'],atol=5e-6,rtol=1e-6)
    expected_value=h.mlp(np.stack([o['state'] for o in observations]).astype(float),w,'critic').ravel()*115.2;np.testing.assert_allclose(value.numpy(),expected_value,atol=2e-5*115.2,rtol=1e-6)
    return dict(states=len(data),greedy_deficit=float(deficit.max()),log_max_error=float(np.abs(logs.numpy()-lp).max()),entropy_max_error=float(np.abs(entropy.numpy()*4-reference['entropy']).max()),value_max_error=float(abs(value.numpy()-expected_value).max()),mechanism=mechanism(utility,legal,u,canonical) if arm==CORE[0] else None)
