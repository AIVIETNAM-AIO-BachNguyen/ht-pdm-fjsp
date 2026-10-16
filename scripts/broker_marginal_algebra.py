"""Finite fixed-broker algebra helper; no policy, simulator or training runner.

Derivation and strict assumptions: docs/technician_broker_marginal_algebra_note.md.
This observed-matching likelihood never enumerates proposals or matchings.
"""
import torch


def physical_logprob(log_probs,matching,ranks,legal):
    """Single-state exact pushforward log mass, canonical K-vector token encoding.

    log_probs: K x (N+1), masked categorical log masses, WAIT token0.
    matching: lengthK integer vector, 0 for physically idle and m+1 for winners.
    ranks: N x K strict per-machine ranks (0 best), with ties already resolved.
    legal: N x K boolean mask. Callers compute policy logits separately.
    """
    if log_probs.ndim!=2 or not log_probs.is_floating_point():raise ValueError('categorical log mass matrix')
    k,width=log_probs.shape;n=width-1
    ranks=torch.as_tensor(ranks,device=log_probs.device);legal=torch.as_tensor(legal,device=log_probs.device)
    action=torch.as_tensor(matching,device=log_probs.device)
    if min(n,k)<1 or ranks.shape!=(n,k) or legal.shape!=(n,k) or legal.dtype!=torch.bool:raise ValueError('shape/mask')
    if action.shape!=(k,) or action.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64) or ((action<0)|(action>n)).any():raise ValueError('canonical physical action')
    expected=torch.arange(k,device=log_probs.device).expand(n,k)
    if ranks.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64) or not torch.equal(ranks.sort(1).values,expected):raise ValueError('strict priority ranks')
    if not torch.isfinite(log_probs[:,0]).all() or torch.isnan(log_probs).any() or torch.isposinf(log_probs).any():raise ValueError('finite WAIT/log mass')
    if not torch.allclose(torch.logsumexp(log_probs,1),torch.zeros(k,dtype=log_probs.dtype,device=log_probs.device),atol=1e-6,rtol=0):raise ValueError('normalized categorical log masses')
    if (log_probs[:,1:].T[~legal].exp()>0).any():raise ValueError('masked categorical log masses')
    winners=[(int(action[j])-1,j) for j in range(k) if action[j]]
    if len({m for m,j in winners})!=len(winners) or any(not legal[m,j] for m,j in winners):raise ValueError('feasible physical matching')
    terms=[]
    for j in range(k):
        token=int(action[j])
        if token:terms.append(log_probs[j,token])
        else:
            allowed=[0]+[m+1 for m,w in winners if legal[m,j] and ranks[m,w]<ranks[m,j]]
            terms.append(torch.logsumexp(log_probs[j,allowed],dim=0))
    return torch.stack(terms).sum()
