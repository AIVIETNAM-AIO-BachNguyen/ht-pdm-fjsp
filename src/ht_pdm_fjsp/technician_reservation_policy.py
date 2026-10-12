"""Shared bids, pooled graph messages and technician reservation PPO."""
import numpy as np
import torch
from torch import nn
from ht_pdm_fjsp.maintenance_balanced_algorithms import mlp, masked, tensors


class ReservationPolicy(nn.Module):
    is_actor=True
    joint_policy=True
    def __init__(self,algorithm,hidden=64):
        super().__init__()
        if algorithm not in ('reservation_ppo','independent_ppo'): raise ValueError(algorithm)
        self.algorithm=algorithm;self.hidden=hidden
        self.edge_encoder=mlp(22,hidden,hidden)
        self.actor=mlp(3*hidden+3,hidden,1)
        self.wait=mlp(2*hidden+3,hidden,1)
        self.critic=mlp(2*hidden+9,hidden,1)

    def encode(self,batch):
        m,t,e,g=(batch[k] for k in ('machines','technicians','edges','global_features'))
        b,n,k,_=e.shape
        raw=torch.cat([m[:,:,None].expand(-1,-1,k,-1),t[:,None].expand(-1,n,-1,-1),e,
            g[:,None,None].expand(-1,n,k,-1)],-1)
        edge=self.edge_encoder(raw)
        # Full raw graph information is broadcast in both arms; no future shocks.
        global_pool=edge.mean((1,2));tech_pool=edge.mean(1)
        value=self.critic(torch.cat([global_pool,tech_pool.mean(1),g],-1)).squeeze(-1)
        return edge,global_pool,tech_pool,value

    def trace(self,batch,recorded=None,generator=None,deterministic=False):
        edge,g,tech,value=self.encode(batch);b,n,k,_=edge.shape
        if recorded is not None and (recorded.shape!=(b,k) or recorded.dtype!=torch.long): raise ValueError('recorded action contract')
        initial=batch['pair_mask'];reserved=torch.zeros((b,n),dtype=torch.bool)
        logs=torch.zeros(b);entropies=torch.zeros(b);actions=[]
        for j in range(k):
            live=initial[:,:,j] & ~reserved if self.algorithm=='reservation_ppo' else initial[:,:,j]
            resource=(initial & ~reserved[:,:,None]).float() if self.algorithm=='reservation_ppo' else initial.float()
            pool=(edge*resource[...,None]).sum((1,2))/resource.sum((1,2)).clamp_min(1)[:,None]
            stats=torch.stack([live.float().mean(1),reserved.float().mean(1) if self.algorithm=='reservation_ppo' else torch.zeros(b),
                torch.full((b,),j/max(1,k-1))],-1)
            context=torch.cat([edge[:,:,j],g[:,None].expand(-1,n,-1),pool[:,None].expand(-1,n,-1),
                stats[:,None].expand(-1,n,-1)],-1)
            logits=torch.cat([self.wait(torch.cat([tech[:,j],pool,stats],-1)),self.actor(context).squeeze(-1)],-1)
            legal=torch.cat([torch.ones((b,1),dtype=torch.bool),live],-1)
            logits=masked(logits,legal);log=torch.log_softmax(logits,-1);prob=log.exp()
            if recorded is None:
                chosen=logits.argmax(-1) if deterministic else torch.multinomial(prob,1,generator=generator).squeeze(-1)
            else: chosen=recorded[:,j]
            if ((chosen<0)|(chosen>n)).any() or not legal.gather(1,chosen[:,None]).all(): raise ValueError('illegal recorded reservation')
            logs=logs+log.gather(1,chosen[:,None]).squeeze(1)
            entropies=entropies-(prob*log).sum(1)
            actions.append(chosen)
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
