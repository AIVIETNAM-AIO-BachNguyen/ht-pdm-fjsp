"""Relative continuation-value head; frozen PPO controls machines and CM edges."""
import torch
from torch import nn
from ht_pdm_fjsp.maintenance_autoregressive_policy import CONTEXT, STOP, feature_batch
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_heterogeneous_model import membership

CONTRACT=dict(version="assignment_delta_value_v1",edge_features=13,state_context="detached frozen PPO global/machine/technician mean embeddings",
    pooling="selected pair mean/max",target="paired (cost_right-cost_left)/20 under frozen greedy PPO",
    timing="same selected PM machine set and exact CM edges",future_events=False,entity_ids=False,tie_tolerance=1e-5)


def inputs(batch,base):
    with torch.no_grad():
        e=base.encode(batch)
        context=torch.cat([e["g"],e["machine"].mean(1),e["tech"].mean(1)],-1).detach()
    raw=torch.cat([batch["machines"][:,:,None].expand(-1,4,2,-1),
        batch["technicians"][:,None].expand(-1,4,2,-1),batch["edges"]],-1)
    return dict(raw_pairs=raw,context=context)


class RelativeValueHead(nn.Module):
    def __init__(self,hidden=32):
        super().__init__();self.hidden=hidden
        self.pair=nn.Sequential(nn.Linear(13,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.score=nn.Sequential(nn.Linear(5*hidden,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.register_buffer("members",torch.from_numpy(membership(4,2)).float())

    def forward(self,data):
        pair=self.pair(data["raw_pairs"])
        masks=self.members[None,...,None];count=masks.sum((2,3)).clamp_min(1)
        mean=(pair[:,None]*masks).sum((2,3))/count
        maximum=pair[:,None].masked_fill(~masks.bool(),-torch.inf).amax((2,3))
        maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
        context=data["context"][:,None].expand(-1,len(CATALOGUE),-1)
        return self.score(torch.cat([context,mean,maximum],-1)).squeeze(-1)


def allowed_mask(batch,original_ids):
    members=torch.from_numpy(membership(4,2)).bool()
    chosen=members[original_ids]
    same_m=(members.any(2)[None]==chosen.any(2)[:,None]).all(2)
    cm=chosen & (batch["machines"][:,:,1]>.5)[:,:,None]
    fixed_cm=(members[None] & (batch["machines"][:,:,1]>.5)[:,None,:,None])
    exact_cm=(fixed_cm==cm[:,None]).flatten(2).all(2)
    legal=(~members[None] | batch["pair_mask"][:,None]).flatten(2).all(2)
    return same_m & exact_cm & legal


def canonical_tokens(action):
    return [2*m+j for m,j in action]+[STOP]+[-1]*(2-len(action))


class ValueController:
    assignment_value_controller=True
    algorithm=CONTEXT
    def __init__(self,base,head,name="true_value"):
        self.base,self.head,self.name=base,head.eval(),name
        self.learned_horizon,self.tail_ticks=base.learned_horizon,base.tail_ticks
        self.last_decode_audit=None

    @torch.no_grad()
    def sample(self,batch,generator=None,deterministic=False):
        if not deterministic:raise ValueError("value head is a deterministic policy improvement, not a PPO actor")
        tokens=self.base.greedy_tokens(batch)
        # Scalar stable greedy choices for tolerance-sensitive PPO rows.
        encoded=self.base.encode(batch);live=batch["pair_mask"].clone();count=torch.zeros(len(tokens),dtype=torch.long)
        active=torch.ones(len(tokens),dtype=torch.bool);near=torch.zeros(len(tokens),dtype=torch.bool)
        for step in range(3):
            dist,_=self.base.token_distribution(batch,encoded,live,count)
            top=dist.logits.topk(2,1).values
            near |= active & (top[:,0]-top[:,1] <= 1.1e-5)
            tok=tokens[:,step];selected=active & (tok>=0)&(tok<STOP);rows=torch.arange(len(tokens))[selected]
            live[rows,tok[selected]//2,:]=False;live[rows,:,tok[selected]%2]=False
            count+=selected.long();active=selected
        for i in near.nonzero().flatten().tolist():
            tokens[i]=self.base.greedy_tokens({k:v[i:i+1] for k,v in batch.items()})[0]
        originals=[tuple(sorted((t//2,t%2) for t in row if 0<=t<STOP)) for row in tokens.tolist()]
        original_ids=torch.tensor([CATALOGUE.index(a) for a in originals])
        allowed=allowed_mask(batch,original_ids)
        scores=self.head(inputs(batch,self.base))
        if not torch.isfinite(scores).all():raise RuntimeError("nonfinite relative value predictions")
        masked=scores.masked_fill(~allowed,torch.inf)
        order=masked.topk(2,1,largest=False).values
        value_near=(allowed.sum(1)>1)&(order[:,1]-order[:,0]<=1.1e-5)
        for i in value_near.nonzero().flatten().tolist():
            sub={k:v[i:i+1] for k,v in batch.items()}
            scores[i]=self.head(inputs(sub,self.base))[0]
        result=[];audit=[]
        for i,original in enumerate(originals):
            ids=allowed[i].nonzero().flatten().tolist();minimum=min(float(scores[i,j]) for j in ids)
            ties=[j for j in ids if float(scores[i,j])<=minimum+CONTRACT["tie_tolerance"]]
            orig=int(original_ids[i]);best=orig if orig in ties else min(ties)
            action=CATALOGUE[best];result.append(canonical_tokens(action))
            audit.append(dict(original=original,selected=action,candidates=[CATALOGUE[j] for j in ids],
                relative_cost_predictions=[float(scores[i,j]) for j in ids],changed=best!=orig,
                near_tie=bool(near[i] or value_near[i]),
                prediction_semantics="normalized relative continuation cost; PPO sequence likelihood below is descriptive only"))
        chosen=torch.tensor(result,dtype=torch.long)
        probability,_,value=self.base.evaluate(batch,chosen)
        self.last_decode_audit=audit
        return chosen,probability,value

    def decode(self,c,state,tokens):return self.base.decode(c,state,tokens)

    @torch.no_grad()
    def select(self,c,state,remaining,horizon):
        if horizon!=self.learned_horizon:raise ValueError("original learned horizon required")
        a,p,v=self.sample(feature_batch([c],[state],[remaining],horizon,CONTEXT),deterministic=True)
        return self.decode(c,state,a[0]),dict(token_trace=a[0].tolist(),log_probability=float(p[0]),value=float(v[0]),
            assignment_decode=self.last_decode_audit[0])
