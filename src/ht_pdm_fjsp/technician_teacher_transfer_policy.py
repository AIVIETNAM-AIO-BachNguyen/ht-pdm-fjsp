"""Matched teacher-assisted and random-initialized reservation policies."""
import numpy as np
import torch
from torch import nn
from ht_pdm_fjsp.maintenance_balanced_algorithms import mlp, masked, tensors


class ReservationPolicy(nn.Module):
    is_actor=True
    joint_policy=True
    def __init__(self,algorithm,hidden=64):
        super().__init__()
        if algorithm not in ('teacher_reservation_ppo','scratch_reservation_ppo'): raise ValueError(algorithm)
        if hidden%4:raise ValueError('hidden width must divide into four attention heads')
        self.algorithm=algorithm;self.hidden=hidden
        self.edge_encoder=mlp(24,hidden,hidden)
        self.value_edge_encoder=mlp(24,hidden,hidden)
        self.query=nn.Linear(hidden,hidden,bias=False)
        self.key=nn.Linear(hidden,hidden,bias=False)
        self.attention_value=nn.Linear(hidden,hidden,bias=False)
        self.actor=mlp(4*hidden+3,hidden,1)
        self.wait=mlp(3*hidden+3,hidden,1)
        self.critic=mlp(2*hidden+9,hidden,1)

    def encode(self,batch):
        m,t,e,g=(batch[k] for k in ('machines','technicians','edges','global_features'))
        b,n,k,_=e.shape
        machine_id=torch.linspace(0.,1.,n)[None,:,None,None].expand(b,n,k,1)
        technician_id=torch.linspace(0.,1.,k)[None,None,:,None].expand(b,n,k,1)
        raw=torch.cat([m[:,:,None].expand(-1,-1,k,-1),t[:,None].expand(-1,n,-1,-1),e,
            g[:,None,None].expand(-1,n,k,-1),machine_id,technician_id],-1)
        edge=self.edge_encoder(raw)
        machines=edge.mean(2)
        heads=4;channels=self.hidden//heads
        query=self.query(edge).reshape(b,n,k,heads,channels)
        key=self.key(machines).reshape(b,n,heads,channels)
        attention_value=self.attention_value(machines).reshape(b,n,heads,channels)
        weights=torch.softmax(torch.einsum('bnkhd,bmhd->bnkhm',query,key)/channels**.5,-1)
        context=torch.einsum('bnkhm,bmhd->bnkhd',weights,attention_value).reshape(b,n,k,self.hidden)
        edge=edge+machines[:,:,None]+context
        # Full raw graph information is broadcast in both arms; no future shocks.
        global_pool=edge.mean((1,2));tech_pool=edge.mean(1)
        value_edge=self.value_edge_encoder(raw)
        value=self.critic(torch.cat([value_edge.mean((1,2)),value_edge.mean(1).mean(1),g],-1)).squeeze(-1)*(n/4)
        return edge,global_pool,tech_pool,value

    def trace(self,batch,recorded=None,generator=None,deterministic=False,audit=None):
        edge,g,tech,value=self.encode(batch);b,n,k,_=edge.shape
        if recorded is not None and (recorded.shape!=(b,k) or recorded.dtype!=torch.long): raise ValueError('recorded action contract')
        initial=batch['pair_mask'];reserved=torch.zeros((b,n),dtype=torch.bool)
        prefix=torch.zeros((b,self.hidden))
        logs=torch.zeros(b);entropies=torch.zeros(b);actions=[]
        for j in range(k):
            live=initial[:,:,j] & ~reserved
            future=torch.arange(k)[None,None,:]>=j
            resource=(initial & ~reserved[:,:,None] & future).float()
            pool=(edge*resource[...,None]).sum((1,2))/resource.sum((1,2)).clamp_min(1)[:,None]
            stats=torch.stack([live.float().mean(1),reserved.float().mean(1),
                torch.full((b,),j/max(1,k-1))],-1)
            context=torch.cat([edge[:,:,j],g[:,None].expand(-1,n,-1),pool[:,None].expand(-1,n,-1),prefix[:,None].expand(-1,n,-1),
                stats[:,None].expand(-1,n,-1)],-1)
            logits=torch.cat([self.wait(torch.cat([tech[:,j],pool,prefix,stats],-1)),self.actor(context).squeeze(-1)],-1)
            legal=torch.cat([torch.ones((b,1),dtype=torch.bool),live],-1)
            if audit is not None:
                audit.append(dict(legal=legal.detach().clone(),resource=resource.detach().clone(),
                    stats=stats.detach().clone(),prefix=prefix.detach().clone()))
            logits=masked(logits,legal);log=torch.log_softmax(logits,-1);prob=log.exp()
            if recorded is None:
                chosen=logits.argmax(-1) if deterministic else torch.multinomial(prob,1,generator=generator).squeeze(-1)
            else: chosen=recorded[:,j]
            if ((chosen<0)|(chosen>n)).any() or not legal.gather(1,chosen[:,None]).all(): raise ValueError('illegal recorded reservation')
            logs=logs+log.gather(1,chosen[:,None]).squeeze(1)
            entropies=entropies-(prob*log).sum(1)
            actions.append(chosen)
            message=edge[torch.arange(b),(chosen-1).clamp_min(0),j]*(chosen>0)[:,None]
            prefix=prefix+message/k
            following=reserved.clone();rows=torch.arange(b)[chosen>0];following[rows,chosen[chosen>0]-1]=True
            reserved=following
        return torch.stack(actions,1),logs,entropies,value

    def evaluate(self,batch,actions):
        _,log,entropy,value=self.trace(batch,recorded=actions)
        return log,entropy,value

    @torch.no_grad()
    def sample(self,batch,generator=None,deterministic=False,epsilon=0):
        if not deterministic and generator is None: raise ValueError('explicit action RNG required')
        action,log,_,value=self.trace(batch,generator=generator,deterministic=deterministic)
        return action,log,value

    def environment_action(self,action): return action.numpy().copy()

    @torch.no_grad()
    def select(self,obs):
        batch=tensors({k:np.asarray(v)[None] for k,v in obs.items()})
        return self.sample(batch,deterministic=True)[0][0].numpy().copy()
