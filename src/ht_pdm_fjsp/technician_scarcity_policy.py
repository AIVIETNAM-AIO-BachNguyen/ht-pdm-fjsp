"""Learned bids, bounded demand prices, and exact joint PPO trace likelihoods."""
import numpy as np
import torch
from torch import nn
from ht_pdm_fjsp.maintenance_balanced_algorithms import mlp, masked, tensors
from ht_pdm_fjsp.technician_scarcity_env import GRAPH_ARMS

def scarcity_prices(bids,wait,legal,rounds=3,step=.5):
    """Nonnegative dual-demand messages; all probabilities include WAIT."""
    available=legal.any(1);price=torch.zeros_like(bids[:,0])
    for _ in range(rounds):
        logits=masked(torch.cat([bids-price[:,None],wait[:,:,None]],-1),
            torch.cat([legal,torch.ones_like(legal[...,:1])],-1))
        demand=torch.softmax(logits,-1)[...,:bids.shape[-1]].sum(1)
        price=torch.relu(price+step*(demand-1))*available
    center=(price*available).sum(1)/available.sum(1).clamp_min(1)
    return (price-center[:,None])*available

class ScarcityPolicy(nn.Module):
    is_actor=True
    joint_policy=True
    def __init__(self,algorithm,hidden=32):
        super().__init__()
        if algorithm not in GRAPH_ARMS:raise ValueError('graph arm')
        self.algorithm=algorithm;self.hidden=hidden
        self.encoder=mlp(22,hidden,hidden)
        self.value_encoder=mlp(22,hidden,hidden)
        self.bid=mlp(4*hidden,hidden,1)
        self.machine_wait=mlp(2*hidden,hidden,1)
        self.stop=mlp(2*hidden+3,hidden,1)
        self.critic=mlp(hidden+9,hidden,1)

    def encode(self,batch):
        m,t,e,g=(batch[k] for k in ('machines','technicians','edges','global_features'))
        b,n,k,_=e.shape
        raw=torch.cat([m[:,:,None].expand(-1,-1,k,-1),t[:,None].expand(-1,n,-1,-1),e,g[:,None,None].expand(-1,n,k,-1)],-1)
        edge=self.encoder(raw);global_pool=edge.mean((1,2));machines=edge.mean(2);tech=edge.mean(1)
        context=torch.cat([edge,machines[:,:,None].expand(-1,-1,k,-1),tech[:,None].expand(-1,n,-1,-1),global_pool[:,None,None].expand(-1,n,k,-1)],-1)
        bids=self.bid(context).squeeze(-1)
        wait=self.machine_wait(torch.cat([machines,global_pool[:,None].expand(-1,n,-1)],-1)).squeeze(-1)
        value=self.critic(torch.cat([self.value_encoder(raw).mean((1,2)),g],-1)).squeeze(-1)*(n/4)
        return edge,global_pool,bids,wait,value

    def trace(self,batch,recorded=None,generator=None,deterministic=False,diagnostics=False):
        edge,g,bids,wait,value=self.encode(batch);b,n,k=bids.shape
        if self.algorithm=='independent_ppo':
            if diagnostics:self.last_diagnostics=dict(price_mean_abs=0.,price_active_fraction=0.)
            logits=masked(torch.cat([wait[:,:,None],bids],-1),torch.cat([torch.ones((b,n,1),dtype=torch.bool),batch['pair_mask']],-1))
            dist=torch.distributions.Categorical(logits=logits)
            chosen=(logits.argmax(-1) if deterministic else torch.multinomial(dist.probs.reshape(-1,k+1),1,generator=generator).reshape(b,n)) if recorded is None else recorded
            if chosen.shape!=(b,n) or chosen.dtype!=torch.long or ((chosen<0)|(chosen>k)).any():raise ValueError('recorded proposal shape/range')
            legal=torch.cat([torch.ones((b,n,1),dtype=torch.bool),batch['pair_mask']],-1)
            if not legal.gather(-1,chosen[...,None]).all():raise ValueError('masked recorded proposal')
            return chosen,dist.log_prob(chosen).sum(1),dist.entropy().sum(1)/k,value
        if recorded is not None and (recorded.shape!=(b,k) or recorded.dtype!=torch.long):raise ValueError('recorded edge shape')
        legal=batch['pair_mask'].clone();stopped=torch.zeros(b,dtype=torch.bool);prefix=torch.zeros_like(g)
        logs=torch.zeros(b);entropy=torch.zeros(b);actions=[];amounts=[];active=[]
        for stage in range(k):
            stats=torch.stack([legal.float().mean((1,2)),legal.any(1).float().mean(1),torch.full((b,),stage/max(1,k))],-1)
            stop=self.stop(torch.cat([g,prefix,stats],-1)).squeeze(-1)+wait.mean(1)
            prices=scarcity_prices(bids,wait,legal) if self.algorithm=='scarcity_ppo' else torch.zeros((b,k))
            if diagnostics:
                relevant=(~stopped)&legal.any((1,2));amount=prices.detach().abs().mean(1)*relevant
                amounts.append(float(amount.mean()));active.append(float(((amount>1e-6)&relevant).float().mean()))
            logits=torch.cat([stop[:,None],(bids-prices[:,None]).reshape(b,-1)],-1)
            support=torch.cat([torch.ones((b,1),dtype=torch.bool),legal.reshape(b,-1)&~stopped[:,None]],-1)
            logits=masked(logits,support);log=torch.log_softmax(logits,-1);prob=log.exp()
            chosen=(logits.argmax(-1) if deterministic else torch.multinomial(prob,1,generator=generator).squeeze(-1)) if recorded is None else recorded[:,stage]
            if ((chosen<0)|(chosen>n*k)).any() or not support.gather(1,chosen[:,None]).all():raise ValueError('illegal edge trace')
            # Canonical STOP suffix has probability one, not repeated stop costs.
            logs+=log.gather(1,chosen[:,None]).squeeze(1);entropy-=(prob*log).sum(1);actions.append(chosen)
            rows=torch.arange(b)[chosen>0];mi=(chosen[rows]-1)//k;ti=(chosen[rows]-1)%k
            prefix=prefix.clone();prefix[rows]=prefix[rows]+edge[rows,mi,ti]/k
            following=legal.clone();following[rows,mi,:]=False;following[rows,:,ti]=False;legal=following
            stopped=stopped|(chosen==0)
        if diagnostics:self.last_diagnostics=dict(price_mean_abs=sum(amounts)/k,price_active_fraction=sum(active)/k)
        return torch.stack(actions,1),logs,entropy/k,value

    def evaluate(self,batch,actions):
        _,log,entropy,value=self.trace(batch,recorded=actions);return log,entropy,value

    @torch.no_grad()
    def sample(self,batch,generator=None,deterministic=False,epsilon=0):
        if not deterministic and generator is None:raise ValueError('explicit action RNG')
        action,log,_,value=self.trace(batch,generator=generator,deterministic=deterministic)
        return action,log,value

    def environment_action(self,action):return action.numpy().copy()

    @torch.no_grad()
    def select(self,obs):
        return self.trace(tensors({k:np.asarray(v)[None] for k,v in obs.items()}),deterministic=True,diagnostics=True)[0][0].numpy().copy()
