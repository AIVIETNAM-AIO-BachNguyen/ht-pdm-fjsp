"""Scratch machine utilities coordinated by exact two-type physical matching DP."""
import numpy as np
import torch
from torch import nn
from ht_pdm_fjsp.maintenance_balanced_algorithms import mlp, tensors
from ht_pdm_fjsp import technician_typed_gibbs as algebra

CORE=('typed_gibbs_ppo','typed_myopic_ppo','reverse_typed_myopic_ppo')


def decode_tokens(action,legal):
    action=np.asarray(action);legal=np.asarray(legal);n,k=legal.shape
    if action.shape!=(n,) or action.dtype.kind not in 'iu' or ((action<0)|(action>k)).any():raise ValueError('canonical machine tokens')
    pairs=tuple((m,int(j)-1) for m,j in enumerate(action) if j)
    if len({j for m,j in pairs})!=len(pairs) or any(not legal[m,j] for m,j in pairs):raise ValueError('illegal typed matching')
    return pairs,dict(conflicts=0,rejected_proposals=0,executed_starts=len(pairs))


@torch.no_grad()
def sample_matching(tables,types,available,generator):
    if generator is None:raise ValueError('explicit action RNG')
    counts=algebra._availability(tables,types,available);b,n,_=tables['utilities'].shape;rows=torch.arange(b,device=available.device);free=available.clone();actions=torch.zeros((b,n),dtype=torch.long,device=available.device)
    for i in range(n):
        probability=tables['group_log_probabilities'][i][rows,counts[:,0],counts[:,1]].exp()
        group=torch.multinomial(probability,1,generator=generator).squeeze(-1)
        for t in (0,1):
            selected=rows[group==t+1]
            if not len(selected):continue
            mask=free[selected]&(types==t)[None];j=torch.multinomial(mask.to(probability.dtype),1,generator=generator).squeeze(-1)
            actions[selected,i]=j+1;free[selected,j]=False;counts[selected,t]-=1
    return actions


class TypedGibbsPolicy(nn.Module):
    is_actor=True
    joint_policy=True
    def __init__(self,algorithm,n=4,k=2,hidden=64):
        super().__init__()
        if algorithm not in CORE or k<2 or n<1:raise ValueError('typed policy shape/mode')
        self.algorithm=algorithm;self.n=n;self.k=k;self.state_dim=6*n+3*k+4*n*k+9;self.input_dim=self.state_dim+6+6+n+2;self.unit=28.8*n
        self.actor=mlp(self.input_dim,hidden,2);self.critic=mlp(self.state_dim,hidden,1)
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)

    def inputs(self,batch):
        state=batch['state'];b=len(state);device=state.device;types=torch.arange(self.k,device=device)%2;available=batch['technicians'][...,0]>.5;eligible=[];profiles=[];counts=[];capacities=tuple(int((types==t).sum()) for t in (0,1))
        if state.shape!=(b,self.state_dim) or batch['pair_mask'].shape!=(b,self.n,self.k):raise ValueError('raw physical observation shapes')
        for t in (0,1):
            ix=types==t;edges=batch['edges'][:,:,ix,:3]
            if not torch.equal(edges,edges[:,:,:1].expand_as(edges)):raise ValueError('within-type physical service profile differs')
            mask=batch['pair_mask'][:,:,ix];free=available[:,ix]
            allowed=(mask&free[:,None,:]).any(-1)
            if not torch.equal(mask,allowed[:,:,None]&free[:,None,:]):raise ValueError('within-type free-worker eligibility differs')
            eligible.append(allowed);profiles.append(edges[:,:,0]);counts.append(free.sum(-1).to(state.dtype)/capacities[t])
        features=torch.cat([state[:,None,:].expand(-1,self.n,-1),batch['machines'],torch.cat(profiles,-1),torch.eye(self.n,dtype=state.dtype,device=device)[None].expand(b,-1,-1),torch.stack(counts,-1)[:,None,:].expand(-1,self.n,-1)],-1)
        return features,types,available,torch.stack(eligible,-1),capacities

    def tables(self,batch):
        features,types,available,eligible,caps=self.inputs(batch);utility=self.actor(features);order=torch.arange(self.n-1,-1,-1,device=utility.device) if self.algorithm==CORE[2] else torch.arange(self.n,device=utility.device)
        table=algebra.policy_tables(utility[:,order],caps,eligible[:,order],mode='gibbs' if self.algorithm==CORE[0] else 'myopic')
        return table,types,available,order

    def evaluate(self,batch,actions):
        table,types,available,order=self.tables(batch);log=algebra.matching_log_probability(table,types,available,actions[:,order]);root=algebra.root_values(table,types,available)
        return log,root['entropy']/self.n,self.critic(batch['state']).squeeze(-1)*self.unit

    @torch.no_grad()
    def trace(self,batch,generator=None,deterministic=False):
        table,types,available,order=self.tables(batch);ordered=algebra.maximum_matching(table,types,available) if deterministic else sample_matching(table,types,available,generator);action=torch.zeros_like(ordered);action[:,order]=ordered;log=algebra.matching_log_probability(table,types,available,ordered)
        return action,log,table,types,available,order

    @torch.no_grad()
    def sample(self,batch,generator=None,deterministic=False,epsilon=0.):
        if epsilon!=0:raise ValueError('categorical epsilon0')
        action,log,*_=self.trace(batch,generator,deterministic);return action,log,self.critic(batch['state']).squeeze(-1)*self.unit

    @staticmethod
    def environment_action(action):return action.numpy().copy()

    @torch.no_grad()
    def select(self,obs):return self.trace(tensors([obs]),deterministic=True)[0][0].numpy().copy()

    @torch.no_grad()
    def describe(self,obs,action):
        table,types,available,order=self.tables(tensors([obs]));canonical=torch.zeros_like(table['utilities']);canonical[:,order]=table['utilities'];root=algebra.root_values(table,types,available);log=algebra.matching_log_probability(table,types,available,torch.as_tensor(action,dtype=torch.long)[None,order])
        return dict(utilities=canonical[0].tolist(),physical_logp=float(log[0]),physical_entropy=float(root['entropy'][0]))
