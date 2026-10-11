"""Same-root forks to the original H48 + L12 terminal boundary."""
from dataclasses import asdict
import math
import numpy as np
import torch

from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_matching, validate_state
from ht_pdm_fjsp.maintenance_waiting import transition
from ht_pdm_fjsp.maintenance_joint_policy import JOINT, CATALOGUE
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays, batch_tick, membership, reference_batch
from ht_pdm_fjsp.maintenance_timing_forks import assert_batch_feasible, without_preventive

ARMS = ('rl','deadline')
LABELS = ('rl_root','wait','no_pm','reference')
COMPONENTS = ('maintenance_cost','unavailability_cost','failure_cost','waiting_cost')


def state_at(batch,i):
    return DispatchState(**{k:tuple(bool(v) if k=='failed' else int(v) for v in values[i]) for k,values in batch.items()})


def batch_features(c,s,remaining,horizon=48):
    """Exact observation tensors and feasible catalogue mask for JOINT inference."""
    if c.machines!=4 or c.technicians!=2 or not 1<=remaining<=horizon:
        raise ValueError('N4/K2 and positive learned time required')
    age,failed,wait=s['ages'],s['failed'],s['pending_wait'];b=len(age)
    owners=s['assigned'][:,None,:]==np.arange(4)[None,:,None]
    occupied=owners.any(2);rem=(owners*s['remaining'][:,None,:]).sum(2)
    hazard=(~failed & ~occupied & (age+1>=c.failure_age))*c.failure_probability
    machine=np.stack([age/8.,failed,wait/12.,occupied,rem/12.,hazard],-1).astype(np.float32)
    worker_failed=np.take_along_axis(failed,np.maximum(s['assigned'],0),axis=1)&(s['assigned']>=0)
    tech=np.stack([s['remaining']==0,s['remaining']/12.,worker_failed],-1).astype(np.float32)
    durations=np.asarray(c.service_time);effects=np.asarray(c.restoration);present=durations>0
    edges=np.empty((b,4,2,4),np.float32)
    edges[...,0]=present;edges[...,1]=durations/12.;edges[...,2]=effects;edges[...,3]=owners
    edges*=present[None,:,:,None]
    global_row=np.asarray([remaining/12.,c.failure_age/8.,c.failure_probability,c.preventive_cost/20.,
        c.corrective_cost/20.,c.unavailable_cost/20.,c.failure_cost/20.,c.waiting_limit/12.,c.waiting_price/20.],np.float32)
    valid=present[None]&~occupied[:,:,None]&(s['remaining']==0)[:,None,:]
    mask=np.einsum('amj,bmj->ba',membership(4,2),~valid)==0
    data=dict(machines=machine,technicians=tech,edges=edges,
        global_features=np.repeat(global_row[None],b,axis=0),joint_mask=mask)
    if not mask[:,0].all() or any(not np.isfinite(v).all() for v in data.values()):
        raise RuntimeError('invalid inference features/mask')
    return {k:torch.from_numpy(v) for k,v in data.items()}


@torch.no_grad()
def rl_batch(c,s,remaining,model,horizon=48):
    if model.algorithm!=JOINT:
        raise ValueError('frozen JOINT policy required')
    features=batch_features(c,s,remaining,horizon)
    distribution,_=model.distribution(features)
    logits=distribution.logits
    chosen=logits.argmax(1).numpy()
    top=logits.topk(2,dim=1).values
    near=(top[:,0]-top[:,1]<=1e-5).numpy()
    # Use original scalar inference when batch arithmetic might alter a tie.
    for i in np.flatnonzero(near):
        pairs,_=model.select(c,state_at(s,int(i)),remaining,horizon)
        chosen[i]=CATALOGUE.index(pairs)
    selected=membership(4,2)[chosen].astype(bool)
    assert_batch_feasible(c,s,selected)
    return selected,dict(batched_decision_rows=len(chosen),near_tie_scalar_rows=int(near.sum()),
        rl_action_histogram=np.bincount(chosen,minlength=21).tolist())


def root_actions(state,rl,reference):
    roots=[rl,(),without_preventive(state,rl),reference]
    unique=list(dict.fromkeys(roots))
    return tuple(unique),tuple(unique.index(a) for a in roots)


def fork_values(c,state,rl,reference,model,seed,scenarios,tick,horizon=48,tail=12,record_trace=False):
    if scenarios<2 or horizon!=48 or tail!=12 or not 0<=tick<horizon:
        raise ValueError('two scenarios and original H48/L12 clock required')
    validate_state(c,state);actions,label_indices=root_actions(state,rl,reference)
    for action in actions:validate_matching(c,state,action)
    a=len(actions);physical=horizon+tail-tick;b=a*scenarios
    roots=np.zeros((a,4,2),bool)
    for i,pairs in enumerate(actions):
        for m,j in pairs:roots[i,m,j]=True
    shocks=np.random.default_rng(seed).random((physical,scenarios,4))<c.failure_probability
    totals=[];base_totals=[];tail_totals=[];component_values=[];cutoffs=[];terminals=[];first_values=[];traces=[]
    counts=dict(batched_decision_rows=0,near_tie_scalar_rows=0,rl_action_histogram=[0]*21,
        rl_continuation_transition_rows=0,tail_transition_rows=0)
    for arm in ARMS:
        s=arrays(state,b);base=np.zeros(b);extra=np.zeros(b)
        comp={k:np.zeros(b) for k in COMPONENTS};first=None;cutoff=None;trace=[]
        for offset in range(physical):
            global_time=tick+offset
            if offset==0:selected=np.repeat(roots,scenarios,0)
            elif arm=='rl' and global_time<horizon:
                selected,inference=rl_batch(c,s,horizon-global_time,model,horizon)
                for k,value in inference.items():
                    counts[k]=(np.asarray(counts[k])+value).tolist() if k=='rl_action_histogram' else counts[k]+value
                counts['rl_continuation_transition_rows']+=b
            else:
                selected=reference_batch(c,s,horizon+tail-global_time,'deadline_matching')
            assert_batch_feasible(c,s,selected)
            if record_trace:trace.append(selected.reshape(a,scenarios,4,2).copy())
            servicing=selected.any(2).copy()
            for m in range(4):servicing[:,m]|=(s['assigned']==m).any(1)
            start=selected.any(2)
            maintenance=(start & ~s['failed']).sum(1)*c.preventive_cost+(start&s['failed']).sum(1)*c.corrective_cost
            unavailable=(s['failed']|servicing).sum(1)*c.unavailable_cost
            waiting=(s['failed']&~servicing&(s['pending_wait']+1>c.waiting_limit)).sum(1)*c.waiting_price
            following,cost=batch_tick(c,s,selected,np.tile(shocks[offset],(a,1)))
            failure=(following['failed']&~s['failed']).sum(1)*c.failure_cost
            if not np.array_equal(maintenance+unavailable+waiting+failure,cost):
                raise RuntimeError('physical cost component reconciliation')
            for name,value in zip(COMPONENTS,[maintenance,unavailable,failure,waiting],strict=True):comp[name]+=value
            if offset==0:first=cost.reshape(a,scenarios)
            if global_time<horizon:base+=cost
            else:extra+=cost;counts['tail_transition_rows']+=b
            s=following
            if global_time==horizon-1:cutoff={k:v.reshape(a,scenarios,-1).copy() for k,v in s.items()}
        if cutoff is None:raise RuntimeError('cutoff48 missing')
        base_totals.append(base.reshape(a,scenarios));tail_totals.append(extra.reshape(a,scenarios))
        totals.append((base+extra).reshape(a,scenarios));first_values.append(first)
        component_values.append({k:v.reshape(a,scenarios) for k,v in comp.items()})
        cutoffs.append(cutoff);terminals.append({k:v.reshape(a,scenarios,-1).copy() for k,v in s.items()})
        if record_trace:traces.append(np.stack(trace))
    costs=np.stack(totals)
    if not np.isfinite(costs).all() or not np.array_equal(first_values[0],first_values[1]):
        raise RuntimeError('nonfinite/unpaired root costs')
    if counts['tail_transition_rows']!=len(ARMS)*a*scenarios*tail:
        raise RuntimeError('tail clock mismatch')
    return dict(actions=actions,label_indices=label_indices,costs=costs,base=np.stack(base_totals),tail=np.stack(tail_totals),
        component_costs=component_values,cutoff=cutoffs,terminal=terminals,first=np.stack(first_values),shocks=shocks,
        transitions=len(ARMS)*a*scenarios*physical,physical_ticks=physical,counters=counts,
        action_trace=np.stack(traces) if record_trace else None)


def paired_delta(values,arm,left,right):
    i=values['label_indices'][LABELS.index(left)];j=values['label_indices'][LABELS.index(right)]
    samples=values['costs'][ARMS.index(arm),i]-values['costs'][ARMS.index(arm),j]
    return float(samples.mean()),float(samples.std(ddof=1)/math.sqrt(len(samples)))


def scalar_replay(c,state,values,model,tick):
    from ht_pdm_fjsp.maintenance_opportunity_cost import rule_action
    for arm_i,arm in enumerate(ARMS):
        for i,root in enumerate(values['actions']):
            for scenario in range(values['shocks'].shape[1]):
                current=state;base=tail=0.;components={k:0. for k in COMPONENTS}
                for offset in range(60-tick):
                    t=tick+offset
                    action=root if offset==0 else model.select(c,current,48-t,48)[0] if arm=='rl' and t<48 else rule_action(c,current,60-t,'deadline_matching')
                    if values['action_trace'] is not None:
                        selected=membership(4,2)[CATALOGUE.index(action)].astype(bool)
                        np.testing.assert_array_equal(selected,values['action_trace'][arm_i,offset,i,scenario])
                    current,cost,info=transition(c,current,action,values['shocks'][offset,scenario])
                    if t<48:base+=cost
                    else:tail+=cost
                    for k in COMPONENTS:components[k]+=info[k]
                    if t==47:
                        for k,v in asdict(current).items():np.testing.assert_array_equal(v,values['cutoff'][arm_i][k][i,scenario])
                if base!=values['base'][arm_i,i,scenario] or tail!=values['tail'][arm_i,i,scenario]:
                    raise RuntimeError('scalar base/tail replay mismatch')
                for k in COMPONENTS:
                    if components[k]!=values['component_costs'][arm_i][k][i,scenario]:raise RuntimeError('scalar component mismatch')
                for k,v in asdict(current).items():np.testing.assert_array_equal(v,values['terminal'][arm_i][k][i,scenario])
    return True
