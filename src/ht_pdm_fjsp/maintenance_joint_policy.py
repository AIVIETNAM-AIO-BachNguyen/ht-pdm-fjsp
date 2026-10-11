"""Centralized N4/K2 matching policy; joint assignment or restricted ablation."""
from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ht_pdm_fjsp.maintenance_dispatch import validate_matching
from ht_pdm_fjsp.maintenance_waiting import observation
from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue, membership, feasible, fixed_candidates

JOINT, FIXED = "joint_ppo", "fixed_ppo"
ARMS = (JOINT, FIXED)
CATALOGUE = catalogue(4, 2)
INDEX = {p:i for i,p in enumerate(CATALOGUE)}
FEATURE_CONTRACT = dict(machine=6,technician=3,edge=4,global_features=9,
    machines=4,technicians=2,actions=21,entity_ids=False,future_events=False,
    time_feature="remaining learned horizon",absent_edge_features="zero",
    constraint="all feasible partial matchings, including WAIT")


def feature_batch(configs, states, remaining, learned_horizon):
    if not len(configs)==len(states)==len(remaining) or not configs:
        raise ValueError("feature dimensions")
    data={k:[] for k in ("machines","technicians","edges","global_features","joint_mask","fixed_mask")}
    for c,s,h in zip(configs,states,remaining,strict=True):
        if c.machines!=4 or c.technicians!=2 or not 1<=h<=learned_horizon:
            raise ValueError("N4/K2/positive remaining horizon required")
        obs=observation(replace(c,horizon=learned_horizon),s,learned_horizon-h)
        obs["edges"][obs["edges"][...,0]<.5]=0.
        for k,v in obs.items():data[k].append(v)
        for k,actions in (("joint_mask",feasible(c,s)),("fixed_mask",fixed_candidates(c,s))):
            mask=np.zeros(21,bool)
            for pairs in actions:mask[INDEX[pairs]]=True
            data[k].append(mask)
    return {k:torch.from_numpy(np.stack(v)) for k,v in data.items()}


class MatchingPolicy(nn.Module):
    def __init__(self, algorithm=JOINT, hidden=32):
        super().__init__()
        if algorithm not in ARMS:raise ValueError(algorithm)
        self.algorithm,self.hidden=algorithm,hidden
        enc=lambda d:nn.Sequential(nn.Linear(d,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.machine_encoder=enc(6);self.technician_encoder=enc(3)
        self.edge_encoder=enc(4);self.global_encoder=enc(9);self.pair_encoder=enc(3*hidden)
        self.critic=nn.Sequential(nn.Linear(3*hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.actor=nn.Sequential(nn.Linear(7*hidden+2,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.register_buffer("members",torch.from_numpy(membership(4,2).copy()))

    def distribution(self, batch):
        edge=self.edge_encoder(batch["edges"])
        valid=batch["edges"][...,0]
        weighted=edge*valid[...,None]
        machine=self.machine_encoder(batch["machines"])+weighted.sum(2)/valid.sum(2).clamp_min(1)[...,None]
        tech=self.technician_encoder(batch["technicians"])+weighted.sum(1)/valid.sum(1).clamp_min(1)[...,None]
        g=self.global_encoder(batch["global_features"])
        all_m,all_t=machine.mean(1),tech.mean(1)
        pair=self.pair_encoder(torch.cat([machine[:,:,None].expand(-1,4,2,-1),
                                         tech[:,None].expand(-1,4,2,-1),edge],-1))
        value=self.critic(torch.cat([g,all_m,all_t],-1)).squeeze(-1)
        mem=self.members;count=mem.sum((1,2));chosen_m=mem.sum(2);chosen_t=mem.sum(1)
        mean=lambda weights,nodes,denominator:torch.einsum("ae,beh->bah",weights,nodes)/denominator.clamp_min(1)[None,:,None]
        selected_pair=torch.einsum("amj,bmjh->bah",mem,pair)/count.clamp_min(1)[None,:,None]
        b=len(g)
        context=torch.cat([g[:,None].expand(-1,21,-1),all_m[:,None].expand(-1,21,-1),
            all_t[:,None].expand(-1,21,-1),selected_pair,mean(chosen_m,machine,count),
            mean(1-chosen_m,machine,4-count),mean(1-chosen_t,tech,2-count),
            (count/4)[None,:,None].expand(b,-1,-1),
            (batch["technicians"][:,:,0].sum(1)/2)[:,None,None].expand(-1,21,-1)],-1)
        mask=batch["fixed_mask"] if self.algorithm==FIXED else batch["joint_mask"]
        if not mask[:,0].all():raise ValueError("WAIT mask")
        return Categorical(logits=self.actor(context).squeeze(-1).masked_fill(~mask,-torch.inf)),value

    def evaluate(self,batch,action):
        dist,value=self.distribution(batch)
        mask=batch["fixed_mask"] if self.algorithm==FIXED else batch["joint_mask"]
        if not mask.gather(1,action[:,None]).all():raise ValueError("infeasible recorded action")
        return dist.log_prob(action),dist.entropy(),value

    @torch.no_grad()
    def sample(self,batch,generator=None,deterministic=False):
        dist,value=self.distribution(batch)
        action=dist.logits.argmax(1) if deterministic else torch.multinomial(dist.probs,1,generator=generator).squeeze(1)
        return action,dist.log_prob(action),value

    def decode(self,c,state,action):
        pairs=CATALOGUE[int(action)]
        validate_matching(c,state,pairs)
        if self.algorithm==FIXED and pairs not in fixed_candidates(c,state):raise ValueError("fixed action")
        return pairs

    @torch.no_grad()
    def select(self,c,state,remaining,horizon):
        action,prob,value=self.sample(feature_batch([c],[state],[remaining],horizon),deterministic=True)
        return self.decode(c,state,action[0]),dict(action_index=int(action[0]),log_probability=float(prob[0]),value=float(value[0]))
