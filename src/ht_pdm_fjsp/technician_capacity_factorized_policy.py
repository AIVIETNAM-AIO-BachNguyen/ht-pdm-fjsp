"""Per-technician service mass and capacity-sensitive conditional ranking."""
import torch
from ht_pdm_fjsp.maintenance_balanced_algorithms import masked
from ht_pdm_fjsp.technician_completion_policy import initial_calendar,calendar_embeddings
from ht_pdm_fjsp.technician_residual_capacity_policy import ResidualCapacityPolicy,capacity_summaries

CORE_ARMS=('residual_capacity_ppo','no_capacity_ppo','initial_capacity_ppo',
           'permuted_capacity_ppo','independent_capacity_ppo','flat_capacity_ppo')


def token_log_probabilities(wait,bids,legal,factored=True):
    """Exact categorical scores; differentiable ranking normalization, no detach.

    Center machine log mass to log(|L|). This is equivalent to a binary WAIT/GO
    sigmoid with z=wait-log(|L|), followed by conditional softmax over legal bids.
    Common zero heads yield bit-identical flat/factored initial token logits.
    """
    if wait.shape!=(*bids.shape[:-1],1) or legal.shape!=bids.shape or legal.dtype!=torch.bool:
        raise ValueError('token head/support shape or type')
    count=legal.sum(-1,keepdim=True)
    support=torch.cat([torch.ones_like(wait,dtype=torch.bool),legal],-1)
    if factored:
        safe=legal.clone();safe[:,0]|=count[:,0]==0
        log_mass=torch.logsumexp(masked(bids,safe),-1,keepdim=True)
        bids=bids-log_mass+count.clamp_min(1).to(bids.dtype).log()
    log=torch.log_softmax(masked(torch.cat([wait,bids],-1),support),-1)
    forced=torch.full_like(log,-1e9);forced[:,0]=0
    return torch.where(count>0,log,forced),support


class FactorizedCapacityPolicy(ResidualCapacityPolicy):
    def __init__(self,algorithm,hidden=27):
        if algorithm not in CORE_ARMS:raise ValueError('factorized capacity arm')
        super().__init__('residual_capacity_ppo' if algorithm=='flat_capacity_ppo' else algorithm,hidden)
        self.algorithm=algorithm
        for head in (self.bid,self.wait):
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)

    def conditionals(self,batch,base,calendar,planned,technician,position,reserved):
        b,n,k,_=base.shape;rows=torch.arange(b,device=base.device)
        if self.algorithm=='independent_capacity_ppo':
            calendar,planned=initial_calendar(batch);reserved=torch.zeros_like(reserved)
        true,static=capacity_summaries(batch,reserved,technician)
        permuted=torch.roll(true,shifts=1,dims=1)
        message=(torch.zeros_like(true) if self.algorithm=='no_capacity_ppo' else static
                 if self.algorithm=='initial_capacity_ppo' else permuted
                 if self.algorithm=='permuted_capacity_ppo' else true)
        edge=(base+self.capacity_encoder(message)[:,:,None,:]
              +calendar_embeddings(self.calendar_encoder,calendar,planned))
        machine,tech,graph=edge.mean(2),edge.mean(1),edge.mean((1,2))
        own=tech[rows,technician]
        pair=edge[rows[:,None],torch.arange(n,device=base.device)[None],technician[:,None]]
        context=torch.cat([pair,machine,own[:,None].expand(-1,n,-1),graph[:,None].expand(-1,n,-1)],-1)
        bids=self.bid(context).squeeze(-1)
        stage=torch.full((b,1),position/k,dtype=base.dtype,device=base.device)
        wait=self.wait(torch.cat([own,graph,stage],-1))
        legal=batch['pair_mask'][rows[:,None],torch.arange(n,device=base.device)[None],technician[:,None]] & ~reserved
        return token_log_probabilities(wait,bids,legal,self.algorithm!='flat_capacity_ppo')
