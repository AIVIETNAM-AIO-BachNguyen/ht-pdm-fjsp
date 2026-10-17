"""Independent NumPy conditional matching counts and raw actor replay."""
import itertools
import numpy as np

CORE=('capacity_order_ppo','fixed_order_ppo','reverse_capacity_ppo','mask_only_order_ppo','independent_machine_q_ppo','independent_machine_p_ppo')


def decode(u,legal,edges,arm):
    independent=arm in CORE[-2:];u=np.asarray(u);assert u.shape==(4,) and u.dtype.kind in 'iu' and ((u>=0)&(u<=2)).all();requests=[(m,int(t)-1) for m,t in enumerate(u) if t];assert all(legal[m,j] for m,j in requests);pairs=[];conflicts=rejected=0
    for j in range(2):
        group=[p for p in requests if p[1]==j]
        if not group:continue
        assert independent or len(group)==1;conflicts+=int(len(group)>1);rejected+=len(group)-1;pairs.append(max(group,key=lambda p:(float(edges[p[0],j,2])/round(float(edges[p[0],j,1])*12),-p[0])))
    return tuple(sorted(pairs)),dict(conflicts=conflicts,rejected_proposals=rejected,executed_starts=len(pairs),order_forward=int(arm!='reverse_capacity_ppo'),order_reverse=int(arm=='reverse_capacity_ppo'),prefix_enabled=int(not independent))


def catalogue_tokens(h):
    table=np.zeros((21,4),int)
    for index,pairs in enumerate(h.baseline_catalogue()):
        for m,j in pairs:table[index,m]=j+1
    return table


def machine_reference(observations,actions,arm,weights,h,verify_greedy=True):
    b=len(actions);u=np.asarray(actions);assert u.shape==(b,4);legal=np.stack([o['pair_mask'] for o in observations]);active=legal.any(-1);free=legal.any(-2);assert np.array_equal(legal,active[:,:,None]&free[:,None]);state=np.stack([o['state'] for o in observations]).astype(float);machines=np.stack([o['machines'] for o in observations]).astype(float);edges=np.stack([o['edges'] for o in observations]).astype(float);duration=np.rint(edges[...,1]*12);capacity=(legal/np.maximum(duration,1)).sum(-1)
    if arm=='fixed_order_ppo':order=np.broadcast_to(np.arange(4),(b,4))
    else:order=np.argsort(np.where(active,-capacity if arm=='reverse_capacity_ppo' else capacity,np.inf),axis=1,kind='stable')
    canonical=catalogue_tokens(h);eligible=np.array([[all(o['pair_mask'][m,j] for m,j in pairs) for pairs in h.baseline_catalogue()] for o in observations]);consistent=eligible.copy();plan=np.zeros((b,4,2));reserved=np.zeros((b,2),bool);processed=np.zeros((b,4),bool);logp=np.zeros((b,4,3));maximum_deficit=0.;rows=np.arange(b);independent=arm in CORE[-2:]
    for position in range(4):
        m=order[:,position];tokens=canonical[:,m].T;prior_counts=np.stack([((eligible if independent else consistent)&(tokens==t)).sum(-1) for t in range(3)],-1);support=prior_counts>0;prior=np.log(np.maximum(prior_counts,1))-np.log(prior_counts.sum(-1))[:,None];explicit=not independent and arm!='mask_only_order_ppo';features=np.c_[state,machines[rows,m],edges[rows,m].reshape(b,-1),plan.reshape(b,-1) if explicit else np.zeros((b,8)),reserved.astype(float) if explicit else np.zeros((b,2)),processed.astype(float) if explicit else np.zeros((b,4)),np.eye(4)[m],np.full((b,1),position/4)];z=h.mlp(features,weights,'actor')+prior;masked=np.where(support,z,-np.inf);lp=masked-np.logaddexp.reduce(masked,axis=-1)[:,None];chosen=u[rows,m];assert support[rows,chosen].all();deficit=masked.max(-1)-z[rows,chosen]
        if verify_greedy:assert (deficit/(5e-6+1e-6*np.abs(z[rows,chosen]))).max()<=1,'manual conditional greedy'
        maximum_deficit=max(maximum_deficit,float(deficit.max()));logp[rows,m]=lp;processed[rows,m]=True
        if not independent:
            consistent &= tokens==chosen[:,None];selected=rows[chosen>0];worker=chosen[selected]-1;reserved[selected,worker]=True;plan[selected,m[selected],worker]=1
    proposal=logp[np.arange(b)[:,None],np.arange(4)[None],u].sum(-1)
    if arm!='independent_machine_q_ppo':return dict(log_probability=proposal,greedy_deficit=maximum_deficit,proposal_logp=logp)
    latent=np.array(list(itertools.product(range(3),repeat=4)));table=np.array([catalogue_index(decode(v,np.ones((4,2),bool),observations[0]['edges'],arm)[0]) for v in latent]);physical=np.array([catalogue_index(decode(a,o['pair_mask'],o['edges'],arm)[0]) for a,o in zip(u,observations)]);latent_lp=sum(logp[:,m,latent[:,m]] for m in range(4));same=(table[None]==physical[:,None]).all(-1);mass=np.logaddexp.reduce(np.where(same,latent_lp,-np.inf),axis=-1);assert np.isfinite(mass).all();return dict(log_probability=mass,greedy_deficit=maximum_deficit,proposal_logp=logp)


def catalogue_index(pairs):
    u=np.zeros(4,int)
    for m,j in pairs:u[m]=j+1
    return u


def technician_reference(observations,actions,weights,h):
    b=len(actions);actions=np.asarray(actions);state=np.stack([o['state'] for o in observations]).astype(float);tech=np.stack([o['technicians'] for o in observations]).astype(float);edges=np.stack([o['edges'] for o in observations]).astype(float);legal=np.stack([o['pair_mask'] for o in observations]);plan=np.zeros((b,4,2));reserved=np.zeros((b,4),bool);rows=np.arange(b);result=np.zeros(b);maximum=0.
    for j in range(2):
        inputs=np.c_[state,tech[:,j],edges[:,:,j].reshape(b,-1),plan.reshape(b,-1),reserved.astype(float),np.full((b,1),j/2)];logits=h.mlp(inputs,weights,'actor');support=np.c_[np.ones(b,bool),legal[:,:,j]&~reserved];z=np.where(support,logits,-np.inf);u=actions[:,j];assert support[rows,u].all();deficit=z.max(-1)-logits[rows,u];assert (deficit/(5e-6+1e-6*np.abs(logits[rows,u]))).max()<=1,'raw technician greedy';maximum=max(maximum,float(deficit.max()));result+=z[rows,u]-np.logaddexp.reduce(z,-1);selected=rows[u>0];machine=u[selected]-1;plan[selected,machine,j]=1;reserved[selected,machine]=True
    return dict(log_probability=result,greedy_deficit=maximum)


def verify_neural(data,arm,train,source,choices,config,h):
    observations,actions=map(list,zip(*data));w=h.weights(source,choices[f'{arm}:{train}'])
    if arm in CORE:reference=machine_reference(observations,np.asarray(actions),arm,w,h)
    elif arm=='raw_tech_ppo':reference=technician_reference(observations,np.asarray(actions),w,h)
    else:return dict(states=len(data),greedy_deficit=h.verify_actor(observations,actions,arm,w))
    from ht_pdm_fjsp import technician_capacity_order_experiment as production
    import torch
    model=production.load_model(source/choices[f'{arm}:{train}']['path'],arm,config,train,491520);batch=production.learner.tensors(observations)
    with torch.no_grad():logs,_,value=model.evaluate(batch,torch.tensor(np.asarray(actions),dtype=torch.long))
    expected=reference['log_probability'];error=np.abs(logs.numpy()-expected);assert (error/(5e-6+1e-6*np.abs(expected))).max()<=1,'manual joint density';expected_value=h.mlp(np.stack([o['state'] for o in observations]).astype(float),w,'critic').ravel()*115.2;np.testing.assert_allclose(value.numpy(),expected_value,atol=2e-5*115.2,rtol=1e-6)
    return dict(states=len(data),greedy_deficit=reference['greedy_deficit'],log_max_error=float(error.max()),value_max_error=float(abs(value.numpy()-expected_value).max()))
