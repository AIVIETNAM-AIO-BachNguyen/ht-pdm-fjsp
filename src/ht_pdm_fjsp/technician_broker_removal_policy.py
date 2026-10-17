"""PPO-derived factor credit, cached at behavior weights; no matching catalogue in core."""
import torch
from ht_pdm_fjsp import technician_broker_marginal_experiment as old
from ht_pdm_fjsp.technician_broker_marginal_policy import BrokerMarginalPolicy,priorities,physical_tokens
from ht_pdm_fjsp.technician_broker_removal_credit import factor_log_probs,conditional_removals,expected_removal_baseline

CORE=('removal_rb_ppo','factor_state_ppo','blind_removal_ppo','proposal_removal_ppo','proposal_state_ppo','joint_state_ppo')
ARM=old.CORE_ARMS[0]


def removal_features(state,matchings,n,k):
    """B,removed_i[,alternative],tech -> rawstate + tech-major matching + i/K."""
    if matchings.shape[0]!=len(state) or matchings.shape[1]!=k or matchings.shape[-1]!=k:raise ValueError('removal-context shape')
    prefix=matchings.shape[:-1];physical=(matchings[...,None]==torch.arange(1,n+1,device=state.device)).to(state.dtype).flatten(-2)
    s=state.reshape(len(state),*([1]*(matchings.ndim-2)),state.shape[-1]).expand(*prefix,state.shape[-1]);role=(torch.arange(k,device=state.device,dtype=state.dtype)/k).reshape(1,k,*([1]*(matchings.ndim-3)),1).expand(*prefix,1)
    return torch.cat([s,physical,role],-1)


def predict_removal(model,features):
    if model.algorithm=='blind_removal_ppo':features=torch.cat([features[...,:model.state_dim],torch.zeros_like(features[...,model.state_dim:-1]),features[...,-1:]],-1)
    return model.removal(features).squeeze(-1)*model.removal_unit


class RemovalPolicy(BrokerMarginalPolicy):
    def __init__(self,algorithm,n=4,k=2,hidden=64):
        if algorithm not in CORE:raise ValueError('removal policy mode')
        super().__init__(ARM,n,k,hidden);self.algorithm=algorithm;self.removal=old.learner.mlp(self.state_dim+n*k+1,hidden,1);self.removal_unit=self.unit;self.joint_policy=algorithm=='joint_state_ppo'

    def trace(self,batch,recorded=None,generator=None,deterministic=False):
        logp,support=self.proposal_log_probs(batch)
        if recorded is not None:action=recorded
        elif deterministic:action=logp.argmax(-1)
        else:
            if generator is None:raise ValueError('explicit action RNG')
            action=torch.stack([torch.multinomial(logp[:,j].exp(),1,generator=generator).squeeze(-1) for j in range(self.k)],-1)
        if action.shape!=(len(logp),self.k) or action.dtype!=torch.long or ((action<0)|(action>self.n)).any() or not support.gather(-1,action[...,None]).all():raise ValueError('legal canonical proposals')
        physical=physical_tokens(action,batch['pair_mask'],priorities(batch,ARM));factors=factor_log_probs(logp,physical,batch['pair_mask'],priorities(batch,ARM))
        logs=logp.gather(-1,action[...,None]).squeeze(-1) if self.algorithm.startswith('proposal_') else factors
        if self.joint_policy:logs=logs.sum(-1)
        return action,logs,-(logp.exp()*logp).sum(-1).mean(-1)


class CapacityPPO(old.reference.NormalizedPPO):
    def __init__(self,hidden=64):
        super().__init__(hidden);self.algorithm='capacity_ppo';self.state_dim=71;self.n=4;self.k=2;self.removal_unit=old.value_unit(4);self.removal=old.learner.mlp(80,hidden,1)


def initialize_removal_from_fresh_state(model):
    """Fresh common V copied into private b, added context/index columns zero."""
    state=model.critic.net if isinstance(model.critic,old.reference.ValueUnits) else model.critic
    with torch.no_grad():
        model.removal[0].weight.zero_();model.removal[0].weight[:,:model.state_dim].copy_(state[0].weight);model.removal[0].bias.copy_(state[0].bias)
        for layer in (2,4):model.removal[layer].load_state_dict(state[layer].state_dict())


@torch.no_grad()
def baseline_cache(model,batch,actions,old_value):
    if not hasattr(model,'removal'):return None
    b=len(actions);k=model.k;n=model.n
    if model.algorithm in CORE:
        logp,_=model.proposal_log_probs(batch);rank=priorities(batch,ARM);physical=physical_tokens(actions,batch['pair_mask'],rank);actual=[]
        for i in range(k):
            removed=actions.clone();removed[:,i]=0;actual.append(physical_tokens(removed,batch['pair_mask'],rank))
        actual=torch.stack(actual,1);weights,alternatives=conditional_removals(logp,physical,batch['pair_mask'],rank)
    else:
        # Existing joint-PPO catalogue ONLY in the external capacity baseline.
        table=torch.zeros((21,2),dtype=torch.long,device=actions.device)
        from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue
        for token,pairs in enumerate(catalogue(4,2)):
            for m,j in pairs:table[token,j]=m+1
        physical=table[actions];actual=physical[:,None,:].expand(b,k,k).clone()
        for i in range(k):actual[:,i,i]=0
        alternatives=actual[:,:,None,:].expand(b,k,k+1,k);weights=torch.zeros((b,k,k+1),device=actions.device);weights[:,:,0]=1
    contexts=removal_features(batch['state'],actual,n,k);actual_values=predict_removal(model,contexts);mixed=expected_removal_baseline(weights,predict_removal(model,removal_features(batch['state'],alternatives,n,k)));empty=contexts.clone();empty[...,model.state_dim:-1]=0;empty_values=predict_removal(model,empty)
    baseline=mixed if model.algorithm in ('removal_rb_ppo','blind_removal_ppo') else actual_values if model.algorithm=='proposal_removal_ppo' else old_value[:,None].expand(b,k)
    return dict(contexts=contexts.clone(),baseline=baseline.clone(),actual_values=actual_values.clone(),mixed_values=mixed.clone(),context_sensitivity=(actual_values-empty_values).abs().mean(),fallback_mass=weights[:,:,1:].sum(-1).mean(),weight_error=(weights.sum(-1)-1).abs().max())


def prepare_actor_advantage(returns,old_value,cache,algorithm):
    base=returns-old_value;center=base.mean();scale=base.std(unbiased=False)+1e-8
    if algorithm in CORE and algorithm!='joint_state_ppo':result=(returns[:,None]-cache['baseline']-center)/scale
    else:result=(base-center)/scale
    return result.detach(),scale.detach()


def actor_objective(current_log,old_log,advantage,algorithm,clip=.2,agents=2):
    ratio=(current_log-old_log).exp();a=advantage if current_log.ndim==advantage.ndim else advantage[:,None]
    objective=torch.minimum(ratio*a,ratio.clamp(1-clip,1+clip)*a).mean()
    return objective/agents if algorithm=='joint_state_ppo' else objective


def critic_loss(model,batch,returns,current_value,cache,unit):
    loss=((current_value-returns)/unit).square().mean()
    if cache is not None:
        extra=((predict_removal(model,cache['contexts'])-returns[:,None])/model.removal_unit).square();loss=(loss+extra.mean(0).sum())/(model.k+1)
    return loss
