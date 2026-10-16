"""Independent technician proposals with exact fixed-broker physical likelihood.

PPO objective/actor/critic are unchanged; the learning ratio may use the broker's
pushforward mass. Entropy remains an explicitly shared proposal regularizer.
"""
import numpy as np
import torch
from ht_pdm_fjsp.maintenance_balanced_algorithms import masked,tensors
from ht_pdm_fjsp.technician_raw_state_reservation_policy import RawStateReservationPolicy

CORE_ARMS=('broker_marginal_ppo','broker_proposal_ppo','id_marginal_ppo','id_proposal_ppo')


def priorities(batch,algorithm):
    if algorithm not in CORE_ARMS:raise ValueError('algorithm')
    legal=batch['pair_mask'];edges=batch['edges'];b,n,k=legal.shape
    if edges.shape!=(b,n,k,4):raise ValueError('edge shape')
    if algorithm.startswith('id_'):return -torch.arange(k,device=edges.device,dtype=torch.float64).expand(b,n,k)
    duration=torch.round(edges[...,1].double()*12)
    if (legal & (duration<1)).any():raise ValueError('eligible positive service duration')
    return edges[...,2].double()/duration.clamp_min(1)


def physical_tokens(proposals,legal,priority):
    b,n,k=legal.shape
    if proposals.shape!=(b,k) or proposals.dtype!=torch.long or ((proposals<0)|(proposals>n)).any():raise ValueError('canonical proposal vector')
    machines=torch.arange(1,n+1,device=proposals.device)[None,:,None]
    requests=proposals[:,None,:]==machines
    if (requests & ~legal).any():raise ValueError('masked proposal')
    owner=priority.masked_fill(~requests,-torch.inf).argmax(-1)
    winners=requests & (torch.arange(k,device=proposals.device)[None,None,:]==owner[...,None])
    return (winners.to(torch.long)*machines).sum(1)


def physical_logprob(log_probs,physical,legal,priority):
    b,n,k=legal.shape
    if log_probs.shape!=(b,k,n+1) or physical.shape!=(b,k) or physical.dtype!=torch.long or ((physical<0)|(physical>n)).any():raise ValueError('physical action/density shape')
    requests=physical[:,None,:]==torch.arange(1,n+1,device=physical.device)[None,:,None]
    if (requests & ~legal).any() or (requests.sum(-1)>1).any():raise ValueError('feasible physical matching')
    occupied=requests.any(-1);owner=requests.long().argmax(-1);owner_priority=priority.gather(-1,owner[...,None]).squeeze(-1);terms=[]
    for j in range(k):
        loser=(owner_priority>priority[:,:,j]) | ((owner_priority==priority[:,:,j]) & (owner<j))
        allowed=torch.cat([torch.ones((b,1),device=physical.device,dtype=torch.bool),occupied & loser & legal[:,:,j]],-1)
        idle_factor=torch.logsumexp(log_probs[:,j].masked_fill(~allowed,-torch.inf),-1)
        winner_factor=log_probs[:,j].gather(1,physical[:,j,None]).squeeze(-1)
        terms.append(torch.where(physical[:,j]>0,winner_factor,idle_factor))
    return torch.stack(terms,-1).sum(-1)


def decode_proposals(tokens,legal,algorithm,edges):
    tokens=np.asarray(tokens);legal=np.asarray(legal);edges=np.asarray(edges);n,k=legal.shape
    if algorithm not in CORE_ARMS or tokens.shape!=(k,) or tokens.dtype.kind not in 'iu' or ((tokens<0)|(tokens>n)).any() or edges.shape!=(n,k,4):raise ValueError('canonical proposal shape/type')
    requests=[(int(t)-1,j) for j,t in enumerate(tokens) if t]
    if any(not legal[m,j] for m,j in requests):raise ValueError('masked proposal')
    selected=[];conflicts=rejected=0
    for m in range(n):
        group=[p for p in requests if p[0]==m]
        if not group:continue
        conflicts+=int(len(group)>1);rejected+=len(group)-1
        if algorithm.startswith('id_'):winner=min(group,key=lambda p:p[1])
        else:
            if any(round(float(edges[m,j,1])*12)<1 for _,j in group):raise ValueError('positive service duration')
            winner=max(group,key=lambda p:(float(edges[p[0],p[1],2])/round(float(edges[p[0],p[1],1])*12),-p[1]))
        selected.append(winner)
    return tuple(sorted(selected)),dict(conflicts=conflicts,rejected_proposals=rejected,executed_starts=len(selected),order_forward=1,order_reverse=0,prefix_enabled=0)


class BrokerMarginalPolicy(RawStateReservationPolicy):
    def __init__(self,algorithm,n=4,k=2,hidden=64):
        if algorithm not in CORE_ARMS:raise ValueError('algorithm')
        super().__init__('independent_raw_ppo',n,k,hidden);self.algorithm=algorithm

    def conditional(self,batch,planned,reserved,j):
        state=batch['state'];b=len(state)
        if state.shape!=(b,self.state_dim) or batch['edges'].shape!=(b,self.n,self.k,4) or batch['pair_mask'].shape!=(b,self.n,self.k):raise ValueError('physical state shape')
        zero=torch.zeros((b,self.n*self.k+self.n),dtype=state.dtype,device=state.device)
        index=torch.full((b,1),j/self.k,dtype=state.dtype,device=state.device)
        inputs=torch.cat([state,batch['technicians'][:,j],batch['edges'][:,:,j].flatten(1),zero,index],-1)
        support=torch.cat([torch.ones((b,1),dtype=torch.bool,device=state.device),batch['pair_mask'][:,:,j]],-1)
        return torch.log_softmax(masked(self.actor(inputs),support),-1),support

    def proposal_log_probs(self,batch):
        pairs=[self.conditional(batch,None,None,j) for j in range(self.k)]
        return torch.stack([p for p,s in pairs],1),torch.stack([s for p,s in pairs],1)

    def trace(self,batch,recorded=None,generator=None,deterministic=False):
        log,support=self.proposal_log_probs(batch);b=len(log)
        if recorded is not None:
            if recorded.shape!=(b,self.k) or recorded.dtype!=torch.long:raise ValueError('canonical recorded proposals')
            action=recorded
        elif deterministic:action=log.argmax(-1)
        else:
            if generator is None:raise ValueError('explicit action RNG')
            action=torch.stack([torch.multinomial(log[:,j].exp(),1,generator=generator).squeeze(-1) for j in range(self.k)],-1)
        if ((action<0)|(action>self.n)).any() or not support.gather(-1,action[...,None]).all():raise ValueError('masked/invalid recorded proposal')
        proposal=log.gather(-1,action[...,None]).squeeze(-1).sum(-1)
        priority=priorities(batch,self.algorithm);physical=physical_tokens(action,batch['pair_mask'],priority)
        marginal=physical_logprob(log,physical,batch['pair_mask'],priority)
        entropy=-(log.exp()*log).sum(-1).mean(-1)
        return action,marginal if self.algorithm in (CORE_ARMS[0],CORE_ARMS[2]) else proposal,entropy

    @torch.no_grad()
    def select(self,obs):
        # Deployment requires no critic, matching likelihood or Jacobian.
        return self.proposal_log_probs(tensors([obs]))[0].argmax(-1)[0].numpy().copy()
