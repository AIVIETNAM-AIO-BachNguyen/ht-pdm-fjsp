"""Exact two-type labeled matching distributions; algebra helper, no RL learner."""
import torch


def _shift(table,axis,fill):
    shape=list(table.shape);shape[axis]=1;pad=table.new_full(shape,fill)
    return torch.cat([pad,table[:,:-1,:] if axis==1 else table[:,:,:-1]],dim=axis)


def _branches(following,utility,eligible,log_a,log_b):
    first=_shift(following,1,-torch.inf)+utility[:,0,None,None]+log_a
    second=_shift(following,2,-torch.inf)+utility[:,1,None,None]+log_b
    first=first.masked_fill(~eligible[:,0,None,None],-torch.inf);second=second.masked_fill(~eligible[:,1,None,None],-torch.inf)
    return torch.stack([following,first,second],-1)


def _partitions(utilities,eligible,capacities,log_a,log_b):
    batch,n,_=utilities.shape;a,b=capacities;result=[None]*(n+1);result[n]=utilities.new_zeros((batch,a+1,b+1))
    for i in range(n-1,-1,-1):result[i]=torch.logsumexp(_branches(result[i+1],utilities[:,i],eligible[:,i],log_a,log_b),-1)
    return result


def policy_tables(utilities,capacities,eligible=None,mode='gibbs'):
    """B,N,2 utilities; all availability count states, log density/entropy/MAP.

    Workers are labeled within each of two identical physical types. Computing
    all count states costs O(B*N*(K0+1)*(K1+1)); it is not polynomial in the
    number of arbitrary technician types. `myopic` uses unweighted future counts.
    """
    if utilities.ndim!=3 or utilities.shape[-1]!=2 or not utilities.is_floating_point() or not torch.isfinite(utilities).all():raise ValueError('finite B,N,2 utilities')
    if len(capacities)!=2 or any(not isinstance(k,int) or isinstance(k,bool) or k<0 for k in capacities):raise ValueError('two nonnegative integer capacities')
    if mode not in ('gibbs','myopic'):raise ValueError('policy mode')
    if eligible is None:eligible=torch.ones_like(utilities,dtype=torch.bool)
    if eligible.shape!=utilities.shape or eligible.dtype!=torch.bool:raise ValueError('type eligibility shape')
    a,b=capacities;log_a=torch.arange(a+1,device=utilities.device,dtype=utilities.dtype).clamp_min(1).log()[None,:,None];log_b=torch.arange(b+1,device=utilities.device,dtype=utilities.dtype).clamp_min(1).log()[None,None,:]
    weighted=_partitions(utilities,eligible,capacities,log_a,log_b)
    future=weighted if mode=='gibbs' else _partitions(torch.zeros_like(utilities),eligible,capacities,log_a,log_b)
    batch,n,_=utilities.shape;entropy=utilities.new_zeros((batch,a+1,b+1));maximum=entropy.clone();conditionals=[None]*n;choices=[None]*n;label_multiplicity=torch.stack([torch.zeros_like(entropy),log_a.expand_as(entropy),log_b.expand_as(entropy)],-1)
    for i in range(n-1,-1,-1):
        branches=_branches(future[i+1],utilities[:,i],eligible[:,i],log_a,log_b);group=branches-torch.logsumexp(branches,-1)[...,None];conditionals[i]=group;concrete=group-label_multiplicity;probability=group.exp();safe=torch.where(torch.isfinite(concrete),concrete,torch.zeros_like(concrete));next_entropy=torch.stack([entropy,_shift(entropy,1,0.),_shift(entropy,2,0.)],-1);entropy=(probability*(next_entropy-safe)).sum(-1)
        with torch.no_grad():
            next_max=torch.stack([maximum,_shift(maximum,1,0.),_shift(maximum,2,0.)],-1);maximum,choices[i]=(concrete.detach()+next_max).max(-1)
    return dict(utilities=utilities,eligible=eligible,capacities=capacities,mode=mode,log_partition=weighted[0],entropy=entropy,max_log_probability=maximum,group_log_probabilities=tuple(conditionals),best_group=tuple(choices))


def _availability(tables,worker_types,available):
    utility=tables['utilities'];batch=len(utility)
    if worker_types.ndim!=1 or worker_types.dtype!=torch.long or ((worker_types<0)|(worker_types>1)).any():raise ValueError('labeled worker types0/1')
    if available.shape!=(batch,len(worker_types)) or available.dtype!=torch.bool:raise ValueError('labeled availability')
    if tuple(int((worker_types==t).sum()) for t in (0,1))!=tables['capacities']:raise ValueError('worker capacity/type contract')
    return torch.stack([available[:,worker_types==t].sum(-1) for t in (0,1)],-1)


def matching_log_probability(tables,worker_types,available,actions):
    utility=tables['utilities'];batch,n,_=utility.shape;counts=_availability(tables,worker_types,available);workers=len(worker_types)
    if actions.shape!=(batch,n) or actions.dtype!=torch.long or ((actions<0)|(actions>workers)).any():raise ValueError('canonical machine->worker tokens')
    rows=torch.arange(batch,device=utility.device);free=available.clone();value=utility.new_zeros(batch)
    for i in range(n):
        chosen=actions[:,i];selected=rows[chosen>0];j=chosen[selected]-1;group=torch.zeros(batch,dtype=torch.long,device=utility.device)
        if len(selected):
            t=worker_types[j]
            if not free[selected,j].all() or not tables['eligible'][selected,i,t].all():raise ValueError('unavailable/duplicate/ineligible worker')
            group[selected]=t+1
        log=tables['group_log_probabilities'][i][rows,counts[:,0],counts[:,1],group];multiplicity=torch.ones(batch,dtype=utility.dtype,device=utility.device)
        if len(selected):multiplicity[selected]=counts[selected,t].to(utility.dtype)
        value=value+log-multiplicity.log()
        if len(selected):free[selected,j]=False;counts=counts.clone();counts[selected,t]-=1
    return value


def root_values(tables,worker_types,available):
    counts=_availability(tables,worker_types,available);rows=torch.arange(len(available),device=available.device)
    return {key:tables[key][rows,counts[:,0],counts[:,1]] for key in ('log_partition','entropy','max_log_probability')}


@torch.no_grad()
def maximum_matching(tables,worker_types,available):
    counts=_availability(tables,worker_types,available);batch,n,_=tables['utilities'].shape;rows=torch.arange(batch,device=available.device);actions=torch.zeros((batch,n),dtype=torch.long,device=available.device);free=available.clone()
    for i in range(n):
        group=tables['best_group'][i][rows,counts[:,0],counts[:,1]]
        for t in (0,1):
            selected=rows[group==t+1]
            if not len(selected):continue
            mask=free[selected]&(worker_types==t)[None]
            if not mask.any(-1).all():raise RuntimeError('MAP chose empty type')
            j=mask.long().argmax(-1);actions[selected,i]=j+1;free[selected,j]=False;counts[selected,t]-=1
    return actions
