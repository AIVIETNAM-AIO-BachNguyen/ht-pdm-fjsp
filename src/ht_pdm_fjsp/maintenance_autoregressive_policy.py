"""Paired N4/K2 PPO heads: catalogue or sequential pair/STOP matching."""
from dataclasses import replace
import numpy as np
import torch
from torch.distributions import Categorical
from ht_pdm_fjsp.maintenance_dispatch import eligible_edges, validate_matching
from ht_pdm_fjsp.maintenance_waiting import observation
from ht_pdm_fjsp.maintenance_joint_policy import MatchingPolicy, JOINT, feature_batch as catalogue_features

CATALOGUE_ARM = "catalogue_ppo"
CONTEXT = "ar_context_ppo"
FROZEN = "ar_frozen_context_ppo"
ARMS = (CATALOGUE_ARM, CONTEXT, FROZEN)
STOP = 8
FEATURE_CONTRACT = dict(machine=6,technician=3,edge=4,global_features=9,
    machines=4,technicians=2,entity_ids=False,future_events=False,
    time_feature="remaining learned horizon",absent_edge_features="zero",
    tokens="pair=2*machine+technician;STOP=8;padding=-1;length=3",
    sequence_likelihood="sum conditional token log probabilities including STOP",
    context="reaggregate remaining feasible graph; frozen variant uses initial graph")


def feature_batch(configs, states, remaining, learned_horizon, algorithm=CONTEXT):
    if algorithm not in ARMS or not configs or not len(configs)==len(states)==len(remaining):
        raise ValueError("feature dimensions/algorithm")
    if algorithm == CATALOGUE_ARM:
        return catalogue_features(configs,states,remaining,learned_horizon)
    data={k:[] for k in ("machines","technicians","edges","global_features","pair_mask")}
    for c,s,h in zip(configs,states,remaining,strict=True):
        if c.machines!=4 or c.technicians!=2 or not 1<=h<=learned_horizon:
            raise ValueError("N4/K2/positive horizon required")
        obs=observation(replace(c,horizon=learned_horizon),s,learned_horizon-h)
        obs["edges"][obs["edges"][...,0]<.5]=0.
        for k,v in obs.items():data[k].append(v)
        data["pair_mask"].append(np.asarray(eligible_edges(c,s),dtype=bool))
    return {k:torch.from_numpy(np.stack(v)) for k,v in data.items()}


class CataloguePolicy(MatchingPolicy):
    def __init__(self,hidden=32):
        super().__init__(JOINT,hidden)
        self.algorithm=CATALOGUE_ARM

    @torch.no_grad()
    def select(self,c,state,remaining,horizon):
        action,prob,value=self.sample(feature_batch([c],[state],[remaining],horizon,self.algorithm),deterministic=True)
        return self.decode(c,state,action[0]),dict(action_index=int(action[0]),log_probability=float(prob[0]),value=float(value[0]))


class AutoregressivePolicy(MatchingPolicy):
    """Same parameter shapes as catalogue, with nine token scores per prefix."""
    def __init__(self,algorithm=CONTEXT,hidden=32):
        if algorithm not in (CONTEXT,FROZEN):raise ValueError(algorithm)
        super().__init__(JOINT,hidden)
        self.algorithm=algorithm

    def encode(self,batch):
        edge=self.edge_encoder(batch["edges"])
        valid=batch["edges"][...,0]
        weighted=edge*valid[...,None]
        raw_m=self.machine_encoder(batch["machines"])
        raw_t=self.technician_encoder(batch["technicians"])
        machine=raw_m+weighted.sum(2)/valid.sum(2).clamp_min(1)[...,None]
        tech=raw_t+weighted.sum(1)/valid.sum(1).clamp_min(1)[...,None]
        g=self.global_encoder(batch["global_features"])
        pair=self.pair_encoder(torch.cat([machine[:,:,None].expand(-1,4,2,-1),
            tech[:,None].expand(-1,4,2,-1),edge],-1))
        value=self.critic(torch.cat([g,machine.mean(1),tech.mean(1)],-1)).squeeze(-1)
        return dict(edge=edge,raw_m=raw_m,raw_t=raw_t,machine=machine,tech=tech,g=g,pair=pair,value=value)

    def resource_context(self,encoded,live,initial):
        graph=live if self.algorithm==CONTEXT else initial
        weights=graph.float()
        edges=encoded["edge"]*weights[...,None]
        machines=encoded["raw_m"]+edges.sum(2)/weights.sum(2).clamp_min(1)[...,None]
        techs=encoded["raw_t"]+edges.sum(1)/weights.sum(1).clamp_min(1)[...,None]
        wm,wt=weights.sum(2)>0,weights.sum(1)>0
        pool=lambda x,w:(x*w[...,None]).sum(1)/w.sum(1).clamp_min(1)[...,None]
        return pool(machines,wm),pool(techs,wt)

    def token_distribution(self,batch,encoded,live,count):
        rem_m,rem_t=self.resource_context(encoded,live,batch["pair_mask"])
        b=len(live);h=self.hidden
        expand=lambda x:x[:,None].expand(-1,9,-1)
        pair=torch.cat([encoded["pair"].reshape(b,8,h),torch.zeros(b,1,h)],1)
        machines=encoded["machine"][:,:,None].expand(-1,4,2,-1).reshape(b,8,h)
        machines=torch.cat([machines,torch.zeros(b,1,h)],1)
        context=torch.cat([expand(encoded["g"]),expand(encoded["machine"].mean(1)),
            expand(encoded["tech"].mean(1)),pair,machines,expand(rem_m),expand(rem_t),
            (count.float()/4)[:,None,None].expand(-1,9,-1),
            (batch["technicians"][:,:,0].sum(1)/2)[:,None,None].expand(-1,9,-1)],-1)
        mask=torch.cat([live.reshape(b,8),torch.ones(b,1,dtype=torch.bool)],1)
        return Categorical(logits=self.actor(context).squeeze(-1).masked_fill(~mask,-torch.inf)),mask

    def trace(self,batch,recorded=None,generator=None,deterministic=False):
        b=len(batch["machines"])
        if recorded is not None and (recorded.shape!=(b,3) or recorded.dtype!=torch.long):
            raise ValueError("recorded trace dimensions/dtype")
        encoded=self.encode(batch);live=batch["pair_mask"].clone()
        active=torch.ones(b,dtype=torch.bool);count=torch.zeros(b,dtype=torch.long)
        logprob=torch.zeros(b);entropy=torch.zeros(b);tokens=[]
        for step in range(3):
            dist,mask=self.token_distribution(batch,encoded,live,count)
            if recorded is None:
                action=dist.logits.argmax(1) if deterministic else torch.multinomial(dist.probs,1,generator=generator).squeeze(1)
                stored=torch.where(active,action,-1)
            else:
                stored=recorded[:,step]
                if ((~active)&(stored!=-1)).any() or (active&((stored<0)|(stored>STOP))).any():
                    raise ValueError("invalid STOP/padding trace")
                action=torch.where(active,stored,STOP)
            if not mask.gather(1,action[:,None]).all():raise ValueError("infeasible recorded token")
            logprob=logprob+torch.where(active,dist.log_prob(action),0.)
            entropy=entropy+torch.where(active,dist.entropy(),0.)
            tokens.append(stored)
            chosen=active&(action<STOP)
            rows=torch.arange(b)[chosen];m=action[chosen]//2;j=action[chosen]%2
            # A new mask tensor avoids modifying a mask saved by autograd.
            following=live.clone();following[rows,m,:]=False;following[rows,:,j]=False
            live=following;count=count+chosen.long();active=chosen
        if active.any():raise ValueError("trace must end in STOP")
        return torch.stack(tokens,1),logprob,entropy,encoded["value"]

    def evaluate(self,batch,action):
        _,prob,entropy,value=self.trace(batch,recorded=action)
        return prob,entropy,value

    @torch.no_grad()
    def sample(self,batch,generator=None,deterministic=False):
        tokens,prob,_,value=self.trace(batch,generator=generator,deterministic=deterministic)
        return tokens,prob,value

    def decode(self,c,state,action):
        values=action.tolist()
        if len(values)!=3 or STOP not in values:raise ValueError("STOP required")
        end=values.index(STOP)
        if any(x!=-1 for x in values[end+1:]) or any(not 0<=x<STOP for x in values[:end]):
            raise ValueError("invalid trace padding")
        pairs=tuple(sorted((int(x)//2,int(x)%2) for x in values[:end]))
        validate_matching(c,state,pairs);return pairs

    @torch.no_grad()
    def select(self,c,state,remaining,horizon):
        action,prob,value=self.sample(feature_batch([c],[state],[remaining],horizon,self.algorithm),deterministic=True)
        return self.decode(c,state,action[0]),dict(token_trace=action[0].tolist(),log_probability=float(prob[0]),value=float(value[0]))
