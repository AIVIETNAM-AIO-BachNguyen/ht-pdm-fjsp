"""Raw-state conditional technician policy; ordinary joint categorical PPO."""
import numpy as np
import torch
from torch import nn
from ht_pdm_fjsp.maintenance_balanced_algorithms import mlp,masked,tensors

CORE_ARMS=('raw_reservation_ppo','mask_only_ppo','reverse_reservation_ppo','independent_raw_ppo')


class RawStateReservationPolicy(nn.Module):
    is_actor=True
    joint_policy=True

    def __init__(self,algorithm,n=4,k=2,hidden=64):
        super().__init__()
        if algorithm not in CORE_ARMS or min(n,k,hidden)<1:raise ValueError('policy/shape')
        self.algorithm=algorithm;self.n=n;self.k=k;self.hidden=hidden
        self.state_dim=6*n+3*k+4*n*k+9;self.input_dim=self.state_dim+3+4*n+n*k+n+1
        self.actor=mlp(self.input_dim,hidden,n+1);self.critic=mlp(self.state_dim,hidden,1)
        self.unit=28.8*n

    def gradient_groups(self):return dict(actor=list(self.actor.parameters()),critic=list(self.critic.parameters()))

    def conditional(self,batch,planned,reserved,j):
        b=len(planned);state=batch['state']
        if state.shape!=(b,self.state_dim) or batch['edges'].shape!=(b,self.n,self.k,4):raise ValueError('physical state shape')
        zero=self.algorithm in ('mask_only_ppo','independent_raw_ppo')
        plan=torch.zeros_like(planned) if zero else planned
        mask=torch.zeros_like(reserved) if zero else reserved
        index=torch.full((b,1),j/self.k,dtype=state.dtype,device=state.device)
        inputs=torch.cat([state,batch['technicians'][:,j],batch['edges'][:,:,j].flatten(1),plan.flatten(1),mask.to(state.dtype),index],-1)
        legal=batch['pair_mask'][:,:,j] & (torch.ones_like(reserved) if self.algorithm=='independent_raw_ppo' else ~reserved)
        support=torch.cat([torch.ones((b,1),dtype=torch.bool,device=state.device),legal],-1)
        return torch.log_softmax(masked(self.actor(inputs),support),-1),support

    def trace(self,batch,recorded=None,generator=None,deterministic=False):
        b=len(batch['state']);device=batch['state'].device;dtype=batch['state'].dtype
        if recorded is not None and (recorded.shape!=(b,self.k) or recorded.dtype!=torch.long):raise ValueError('canonical technician action vector')
        planned=torch.zeros((b,self.n,self.k),dtype=dtype,device=device);reserved=torch.zeros((b,self.n),dtype=torch.bool,device=device)
        action=torch.zeros((b,self.k),dtype=torch.long,device=device);logs=[];entropies=[]
        order=range(self.k-1,-1,-1) if self.algorithm=='reverse_reservation_ppo' else range(self.k)
        for j in order:
            log,support=self.conditional(batch,planned,reserved,j)
            if recorded is not None:chosen=recorded[:,j]
            elif deterministic:chosen=log.argmax(-1)
            else:
                if generator is None:raise ValueError('explicit action RNG')
                chosen=torch.multinomial(log.exp(),1,generator=generator).squeeze(-1)
            if ((chosen<0)|(chosen>self.n)).any() or not support.gather(1,chosen[:,None]).all():raise ValueError('masked/invalid recorded technician token')
            action[:,j]=chosen;logs.append(log.gather(1,chosen[:,None]).squeeze(-1));entropies.append(-(log.exp()*log).sum(-1))
            rows=torch.arange(b,device=device)[chosen>0];machines=chosen[rows]-1
            planned=planned.clone();reserved=reserved.clone();planned[rows,machines,j]=1;reserved[rows,machines]=True
        return action,torch.stack(logs,1).sum(1),torch.stack(entropies,1).mean(1)

    def evaluate(self,batch,actions):
        _,log,entropy=self.trace(batch,recorded=actions)
        return log,entropy,self.critic(batch['state']).squeeze(-1)*self.unit

    @torch.no_grad()
    def sample(self,batch,generator=None,deterministic=False,epsilon=0):
        if epsilon!=0 or (not deterministic and generator is None):raise ValueError('explicit RNG and epsilon0')
        action,log,_=self.trace(batch,generator=generator,deterministic=deterministic)
        return action,log,self.critic(batch['state']).squeeze(-1)*self.unit

    @staticmethod
    def environment_action(action):return action.numpy().copy()

    @torch.no_grad()
    def select(self,obs):return self.trace(tensors([obs]),deterministic=True)[0][0].numpy().copy()


def decode_tokens(tokens,legal,algorithm,edges):
    tokens=np.asarray(tokens);legal=np.asarray(legal);n,k=legal.shape
    if algorithm not in CORE_ARMS or tokens.shape!=(k,) or tokens.dtype.kind not in 'iu' or ((tokens<0)|(tokens>n)).any():raise ValueError('canonical token shape/type/range')
    requests=[(int(m)-1,j) for j,m in enumerate(tokens) if m]
    if any(not legal[m,j] for m,j in requests):raise ValueError('masked request')
    conflicts=rejected=0
    if algorithm=='independent_raw_ppo':
        pairs=[]
        for m in range(n):
            group=[p for p in requests if p[0]==m]
            if group:
                conflicts+=int(len(group)>1);rejected+=len(group)-1
                pairs.append(max(group,key=lambda p:(float(edges[p[0],p[1],2])/round(float(edges[p[0],p[1],1])*12),-p[1])))
    else:
        if len({m for m,j in requests})!=len(requests):raise ValueError('duplicate reservation')
        pairs=requests
    return tuple(sorted(pairs)),dict(conflicts=conflicts,rejected_proposals=rejected,executed_starts=len(pairs),
        order_forward=int(algorithm!='reverse_reservation_ppo'),order_reverse=int(algorithm=='reverse_reservation_ppo'),
        prefix_enabled=int(algorithm in ('raw_reservation_ppo','reverse_reservation_ppo')))
