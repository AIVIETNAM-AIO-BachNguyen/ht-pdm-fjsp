"""Gaussian latent utility PPO with fixed polynomial assignment decoders."""
import math
import numpy as np
import torch
from torch import nn
from ortools.graph.python import min_cost_flow
from ht_pdm_fjsp.maintenance_balanced_algorithms import mlp,tensors
from ht_pdm_fjsp.technician_marginal_matching_policy import MarginalMatchingPolicy

CORE_ARMS=('joint_clearing_ppo','forward_clearing_ppo','reverse_clearing_ppo','independent_clearing_ppo')
QUANTIZATION=1_000_000


def quantized_utilities(utility,legal):
    utility=np.asarray(utility,np.float64);legal=np.asarray(legal)
    if utility.ndim!=2 or utility.shape!=legal.shape or legal.dtype!=bool or not np.isfinite(utility).all() or (np.abs(utility)>1e6).any():
        raise ValueError('finite utility matrix and bool support required')
    if (utility[~legal]!=0).any():raise ValueError('masked utility must be deterministic zero')
    return np.rint(utility*QUANTIZATION).astype(np.int64)


def decode_utility(utility,legal,algorithm):
    if algorithm not in CORE_ARMS:raise ValueError('decoder arm')
    legal=np.asarray(legal);q=quantized_utilities(utility,legal);n,k=q.shape
    proposals=[]
    for j in range(k):
        options=np.flatnonzero(legal[:,j] & (q[:,j]>0))
        if len(options):proposals.append((int(max(options,key=lambda m:(int(q[m,j]),-int(m)))),j))
    conflicts=sum(max(0,sum(m==a for m,_ in proposals)-1) for a in range(n))
    if algorithm=='joint_clearing_ppo':
        flow=min_cost_flow.SimpleMinCostFlow();sink=n+k
        for j in range(k):
            flow.set_node_supply(n+j,1);flow.add_arc_with_capacity_and_unit_cost(n+j,sink,1,0)
            for m in range(n):
                if legal[m,j] and q[m,j]>0:flow.add_arc_with_capacity_and_unit_cost(n+j,m,1,-int(q[m,j]))
        for m in range(n):flow.add_arc_with_capacity_and_unit_cost(m,sink,1,0)
        flow.set_node_supply(sink,-k)
        if flow.solve()!=flow.OPTIMAL:raise RuntimeError('utility flow not optimal')
        pairs=[(flow.head(a),flow.tail(a)-n) for a in range(flow.num_arcs())
               if flow.flow(a) and flow.head(a)<n and flow.tail(a)>=n]
    elif algorithm=='independent_clearing_ppo':
        pairs=[]
        for m in range(n):
            group=[p for p in proposals if p[0]==m]
            if group:pairs.append(max(group,key=lambda p:(int(q[p]),-p[1])))
    else:
        pairs=[];reserved=set()
        for j in (range(k) if algorithm=='forward_clearing_ppo' else reversed(range(k))):
            options=[m for m in range(n) if legal[m,j] and q[m,j]>0 and m not in reserved]
            if options:
                m=max(options,key=lambda m:(int(q[m,j]),-m));pairs.append((m,j));reserved.add(m)
    pairs=tuple(sorted(pairs))
    if len({m for m,j in pairs})!=len(pairs) or len({j for m,j in pairs})!=len(pairs) or any(not legal[m,j] for m,j in pairs):raise RuntimeError('invalid decoded matching')
    return pairs,dict(conflicts=int(conflicts>0) if algorithm=='independent_clearing_ppo' else 0,
        rejected_proposals=conflicts if algorithm=='independent_clearing_ppo' else 0,
        uncoordinated_proposal_rejections=conflicts,executed_starts=len(pairs),
        quantized_utility=sum(int(q[m,j]) for m,j in pairs),order_forward=int(algorithm=='forward_clearing_ppo'),order_reverse=int(algorithm=='reverse_clearing_ppo'))


class JointClearingPolicy(nn.Module):
    is_actor=True
    joint_policy=True
    raw_pairs=staticmethod(MarginalMatchingPolicy.raw_pairs)
    physical_value=MarginalMatchingPolicy.physical_value

    def __init__(self,algorithm,hidden=32):
        super().__init__()
        if algorithm not in CORE_ARMS:raise ValueError('clearing policy arm')
        self.algorithm=algorithm;self.hidden=hidden
        self.encoder=mlp(22,hidden,hidden);self.bid=mlp(4*hidden,hidden,1)
        self.log_scale=nn.Parameter(torch.zeros(()))
        self.value_encoder=mlp(22,hidden,hidden);self.value_calendar_encoder=mlp(4,hidden,hidden)
        self.value_context=mlp(2*hidden+1,hidden,hidden);self.value_head=mlp(hidden+9,hidden,1)

    def gradient_groups(self):
        return {label:[p for name,p in self.named_parameters() if name.startswith('value_')==(label=='critic')]
                for label in ('actor','critic')}

    def distribution(self,batch):
        edge=self.encoder(self.raw_pairs(batch));b,n,k,h=edge.shape
        context=torch.cat([edge,edge.mean(2)[:,:,None].expand(-1,-1,k,-1),
            edge.mean(1)[:,None].expand(-1,n,-1,-1),edge.mean((1,2))[:,None,None].expand(-1,n,k,-1)],-1)
        mean=self.bid(context).squeeze(-1);legal=batch['pair_mask']
        return torch.where(legal,mean,torch.zeros_like(mean)),2*torch.tanh(self.log_scale/2),legal

    def evaluate(self,batch,actions):
        mean,log_std,legal=self.distribution(batch)
        if actions.shape!=mean.shape or not actions.is_floating_point() or not torch.isfinite(actions).all() or (actions[~legal]!=0).any():
            raise ValueError('recorded latent matrix/support contract')
        density=-.5*((actions-mean)/log_std.exp()).square()-log_std-.5*math.log(2*math.pi)
        log=torch.where(legal,density,torch.zeros_like(density)).sum((1,2))
        count=legal.sum((1,2));entropy=(log_std+.5*math.log(2*math.pi*math.e))*(count>0)
        return log,entropy,self.physical_value(batch)

    @torch.no_grad()
    def sample(self,batch,generator=None,deterministic=False,epsilon=0):
        if epsilon!=0 or (not deterministic and generator is None):raise ValueError('explicit Gaussian RNG and epsilon0')
        mean,log_std,legal=self.distribution(batch)
        action=mean if deterministic else torch.where(legal,mean+log_std.exp()*torch.randn(mean.shape,generator=generator,device=mean.device),torch.zeros_like(mean))
        log,_,value=self.evaluate(batch,action)
        return action,log,value

    @staticmethod
    def environment_action(action):return action.numpy().copy()

    @torch.no_grad()
    def select(self,obs):return self.distribution(tensors([obs]))[0][0].numpy().copy()
