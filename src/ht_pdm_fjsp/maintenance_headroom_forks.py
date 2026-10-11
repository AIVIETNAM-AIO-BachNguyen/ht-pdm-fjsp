"""Generic root interventions, continuing the frozen policy on its original clock."""
from dataclasses import asdict
import numpy as np
from ht_pdm_fjsp.maintenance_dispatch import validate_matching, validate_state
from ht_pdm_fjsp.maintenance_waiting import transition
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays, batch_tick, membership, reference_batch
from ht_pdm_fjsp.maintenance_timing_forks import assert_batch_feasible
from ht_pdm_fjsp.maintenance_policy_forks import rl_batch, COMPONENTS
ARMS = ('rl',)

def fork_roots(c,state,actions,model,seed,scenarios,tick,horizon=48,tail=12,record_trace=False):
    if scenarios<2 or horizon!=48 or tail!=12 or not 0<=tick<horizon:
        raise ValueError('two scenarios and original H48/L12 clock required')
    validate_state(c,state)
    actions=tuple(actions)
    if not actions or len(actions)>3 or len(set(actions))!=len(actions):
        raise ValueError("one to three unique canonical roots required")
    for action in actions:
        validate_matching(c,state,action)
        if action not in CATALOGUE:raise ValueError("canonical catalogue action required")
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
    if not np.isfinite(costs).all():
        raise RuntimeError('nonfinite/unpaired root costs')
    if counts['tail_transition_rows']!=len(ARMS)*a*scenarios*tail:
        raise RuntimeError('tail clock mismatch')
    return dict(actions=actions,costs=costs,base=np.stack(base_totals),tail=np.stack(tail_totals),
        component_costs=component_values,cutoff=cutoffs,terminal=terminals,first=np.stack(first_values),shocks=shocks,
        transitions=len(ARMS)*a*scenarios*physical,physical_ticks=physical,counters=counts,
        action_trace=np.stack(traces) if record_trace else None)



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
