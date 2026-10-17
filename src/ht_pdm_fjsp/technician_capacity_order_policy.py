"""Machine-agent capacity ordering with exact scalar matching likelihood."""
from functools import lru_cache
import math
import numpy as np
import torch
from torch import nn
from ht_pdm_fjsp.maintenance_balanced_algorithms import mlp,masked,tensors
from ht_pdm_fjsp.technician_broker_marginal_policy import physical_tokens
from ht_pdm_fjsp.technician_broker_removal_credit import factor_log_probs

CORE=('capacity_order_ppo','fixed_order_ppo','reverse_capacity_ppo','mask_only_order_ppo','independent_machine_q_ppo','independent_machine_p_ppo')
INDEPENDENT=CORE[-2:]


@lru_cache(maxsize=32)
def log_counts(n,k):
    counts=[[1]*(k+1) for _ in range(n+1)]
    for a in range(1,n+1):
        for b in range(1,k+1):counts[a][b]=counts[a-1][b]+b*counts[a-1][b-1]
    return tuple(tuple(math.log(v) for v in row) for row in counts)


def structural_inputs(batch,algorithm):
    legal=batch['pair_mask'];active=legal.any(-1);free=legal.any(-2)
    if not torch.equal(legal,active[:,:,None]&free[:,None,:]):raise ValueError('count prior requires complete bipartite active/free support')
    duration=torch.round(batch['edges'][...,1].double()*12)
    if (legal&(duration<1)).any():raise ValueError('positive eligible service duration')
    capacity=(legal.double()/duration.clamp_min(1)).sum(-1)
    if algorithm=='fixed_order_ppo':order=torch.arange(legal.shape[1],device=legal.device).expand(len(legal),-1)
    else:
        score=-capacity if algorithm=='reverse_capacity_ppo' else capacity
        order=score.masked_fill(~active,torch.inf).argsort(dim=-1,stable=True)
    return active,free,order


def decode_machine_tokens(tokens,legal,edges,algorithm):
    tokens=np.asarray(tokens);legal=np.asarray(legal);n,k=legal.shape
    if algorithm not in CORE or tokens.shape!=(n,) or tokens.dtype.kind not in 'iu' or ((tokens<0)|(tokens>k)).any():raise ValueError('canonical machine tokens')
    requests=[(m,int(t)-1) for m,t in enumerate(tokens) if t]
    if any(not legal[m,j] for m,j in requests):raise ValueError('masked machine request')
    pairs=[];conflicts=rejected=0
    for j in range(k):
        group=[p for p in requests if p[1]==j]
        if not group:continue
        if len(group)>1 and algorithm not in INDEPENDENT:raise ValueError('duplicate reserved technician')
        conflicts+=int(len(group)>1);rejected+=len(group)-1
        pairs.append(max(group,key=lambda p:(float(edges[p[0],j,2])/round(float(edges[p[0],j,1])*12),-p[0])))
    return tuple(sorted(pairs)),dict(conflicts=conflicts,rejected_proposals=rejected,executed_starts=len(pairs),order_forward=int(algorithm!='reverse_capacity_ppo'),order_reverse=int(algorithm=='reverse_capacity_ppo'),prefix_enabled=int(algorithm not in INDEPENDENT))


class CapacityOrderPolicy(nn.Module):
    is_actor=True
    joint_policy=True
    def __init__(self,algorithm,n=4,k=2,hidden=64):
        super().__init__()
        if algorithm not in CORE:raise ValueError('capacity order algorithm')
        self.algorithm=algorithm;self.n=n;self.k=k;self.state_dim=6*n+3*k+4*n*k+9;self.input_dim=self.state_dim+6+4*k+n*k+k+2*n+1;self.unit=28.8*n
        self.actor=mlp(self.input_dim,hidden,k+1);self.critic=mlp(self.state_dim,hidden,1)
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)

    def trace(self,batch,recorded=None,generator=None,deterministic=False):
        active,free,order=structural_inputs(batch,self.algorithm);state=batch['state'];b=len(state);rows=torch.arange(b,device=state.device);dtype=state.dtype;independent=self.algorithm in INDEPENDENT
        if state.shape!=(b,self.state_dim):raise ValueError('raw state dimensions')
        if recorded is not None and (recorded.shape!=(b,self.n) or recorded.dtype!=torch.long):raise ValueError('canonical recorded vector')
        if recorded is not None and ((recorded<0)|(recorded>self.k)).any():raise ValueError('token range')
        processed=torch.zeros((b,self.n),dtype=torch.bool,device=state.device);reserved=torch.zeros((b,self.k),dtype=torch.bool,device=state.device);plan=torch.zeros((b,self.n,self.k),dtype=dtype,device=state.device);actions=torch.zeros((b,self.n),dtype=torch.long,device=state.device);logs=[];entropies=[];proposal_logs=torch.zeros((b,self.n,self.k+1),dtype=dtype,device=state.device);table=torch.tensor(log_counts(self.n,self.k),dtype=dtype,device=state.device)
        for position in range(self.n):
            m=order[:,position];explicit=not independent and self.algorithm!='mask_only_order_ppo';p=plan if explicit else torch.zeros_like(plan);r=reserved.to(dtype) if explicit else torch.zeros_like(reserved,dtype=dtype);done=processed.to(dtype) if explicit else torch.zeros_like(processed,dtype=dtype)
            inputs=torch.cat([state,batch['machines'][rows,m],batch['edges'][rows,m].flatten(1),p.flatten(1),r,done,torch.nn.functional.one_hot(m,self.n).to(dtype),torch.full((b,1),position/self.n,dtype=dtype,device=state.device)],-1)
            legal=batch['pair_mask'][rows,m] & (torch.ones_like(reserved) if independent else ~reserved);support=torch.cat([torch.ones((b,1),dtype=torch.bool,device=state.device),legal],-1)
            remaining=active.sum(-1) if independent else (active&~processed).sum(-1);available=free.sum(-1) if independent else (free&~reserved).sum(-1);before=(remaining-1).clamp_min(0);normalizer=table[remaining,available];wait=table[before,available]-normalizer;serve=table[before,(available-1).clamp_min(0)]-normalizer;prior=torch.cat([wait[:,None],serve[:,None].expand(-1,self.k)],-1)
            log=torch.log_softmax(masked(self.actor(inputs)+prior,support),-1)
            if recorded is not None:chosen=recorded[rows,m]
            elif deterministic:chosen=log.argmax(-1)
            else:
                if generator is None:raise ValueError('explicit action RNG')
                chosen=torch.multinomial(log.exp(),1,generator=generator).squeeze(-1)
            if not support.gather(-1,chosen[:,None]).all():raise ValueError('illegal replayed machine token')
            actions[rows,m]=chosen;proposal_logs=proposal_logs.index_put((rows,m),log);logs.append(log.gather(-1,chosen[:,None]).squeeze(-1));entropies.append(-(log.exp()*log).sum(-1));processed=processed.clone();processed[rows,m]=True
            if not independent:
                selected=rows[chosen>0];workers=chosen[selected]-1;reserved=reserved.clone();plan=plan.clone();reserved[selected,workers]=True;plan[selected,m[selected],workers]=1
        probability=torch.stack(logs,-1).sum(-1)
        if self.algorithm=='independent_machine_q_ppo':
            priority=(batch['edges'][...,2].double()/torch.round(batch['edges'][...,1].double()*12).clamp_min(1)).transpose(1,2);legal=batch['pair_mask'].transpose(1,2);physical=physical_tokens(actions,legal,priority);probability=factor_log_probs(proposal_logs,physical,legal,priority).sum(-1)
        return actions,probability,torch.stack(entropies,-1).mean(-1)

    def evaluate(self,batch,actions):
        _,log,entropy=self.trace(batch,recorded=actions);return log,entropy,self.critic(batch['state']).squeeze(-1)*self.unit

    @torch.no_grad()
    def sample(self,batch,generator=None,deterministic=False,epsilon=0.):
        if epsilon!=0:raise ValueError('epsilon0 categorical actor')
        action,log,_=self.trace(batch,generator=generator,deterministic=deterministic);return action,log,self.critic(batch['state']).squeeze(-1)*self.unit

    @staticmethod
    def environment_action(action):return action.numpy().copy()

    @torch.no_grad()
    def select(self,obs):return self.trace(tensors([obs]),deterministic=True)[0][0].numpy().copy()
