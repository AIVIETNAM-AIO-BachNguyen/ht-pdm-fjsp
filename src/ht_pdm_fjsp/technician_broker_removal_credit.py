"""Polynomial removal-baseline algebra for independent winner-only proposals.

Not a learner. Actor callers MUST detach expected baseline values and weights.
No enumeration of proposal tuples, technician permutations or matching catalogue.
"""
import torch


def alias_support(physical,legal,priority):
    if legal.ndim!=3:raise ValueError('B,N,K legal mask')
    b,n,k=legal.shape
    if physical.shape!=(b,k) or physical.dtype!=torch.long or priority.shape!=(b,n,k) or ((physical<0)|(physical>n)).any():raise ValueError('physical tokens/priority shape')
    requests=physical[:,None,:]==torch.arange(1,n+1,device=physical.device)[None,:,None]
    if (requests & ~legal).any() or (requests.sum(-1)>1).any():raise ValueError('legal physical matching required')
    occupied=requests.any(-1);owner=requests.long().argmax(-1);winner_priority=priority.gather(-1,owner[...,None]).squeeze(-1);rows=[]
    for j in range(k):
        loses=(winner_priority>priority[:,:,j])|((winner_priority==priority[:,:,j])&(owner<j))
        rows.append(torch.cat([torch.ones((b,1),dtype=torch.bool,device=physical.device),occupied & loses & legal[:,:,j]],-1))
    return torch.stack(rows,1)


def factor_log_probs(logp,physical,legal,priority):
    support=alias_support(physical,legal,priority)
    if logp.shape!=support.shape:raise ValueError('B,K,N+1 proposal log probabilities')
    idle=torch.logsumexp(logp.masked_fill(~support,-torch.inf),-1);winner=logp.gather(-1,physical[...,None]).squeeze(-1)
    return torch.where(physical>0,winner,idle)


def conditional_removals(logp,physical,legal,priority):
    """Return (weights, matchings) indexed B,removed_i,fallback_j+1[,tech].

    Slot0 means no fallback; slotj+1 promotes idle j when removing winner i.
    Impossible slots have zero weight and repeat the valid no-fallback matching.
    For an idle i, slot0 is the original physical matching with probability1.
    Weights are differentiable for algebra checks; detach the complete expected
    baseline when used in an actor objective.
    """
    support=alias_support(physical,legal,priority)
    if logp.shape!=support.shape:raise ValueError('B,K,N+1 proposal log probabilities')
    b,n,k=legal.shape;normalizer=torch.logsumexp(logp.masked_fill(~support,-torch.inf),-1);all_weights=[];all_matchings=[];ids=torch.arange(k,device=physical.device)[None]
    for i in range(k):
        token=physical[:,i];index=token[:,None,None].expand(b,k,1);candidate=(physical==0)&(token[:,None]>0)&support.gather(-1,index).squeeze(-1)
        log_alpha=logp.gather(-1,index).squeeze(-1)-normalizer;alpha=torch.where(candidate,log_alpha,torch.full_like(log_alpha,-torch.inf)).exp()
        rank=priority.gather(1,(token-1).clamp_min(0)[:,None,None].expand(b,1,k)).squeeze(1);weights=[(1-alpha).prod(-1)];base=physical.clone();base[:,i]=0;matchings=[base]
        for j in range(k):
            outranks=(rank>rank[:,j,None])|((rank==rank[:,j,None])&(ids<j));weights.append(alpha[:,j]*torch.where(outranks,1-alpha,torch.ones_like(alpha)).prod(-1));replacement=base.clone();replacement[:,j]=torch.where(candidate[:,j],token,replacement[:,j]);matchings.append(replacement)
        all_weights.append(torch.stack(weights,1));all_matchings.append(torch.stack(matchings,1))
    return torch.stack(all_weights,1),torch.stack(all_matchings,1)


def expected_removal_baseline(weights,values):
    """Conditional E[b_i(s,T(U^(-i)))|S]; NOT automatically stop-gradient."""
    if weights.shape!=values.shape:raise ValueError('baseline values must match B,K,K+1 weights')
    return (weights*values).sum(-1)
