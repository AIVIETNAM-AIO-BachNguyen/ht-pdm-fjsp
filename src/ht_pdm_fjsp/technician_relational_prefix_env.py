"""Current-state resource graph and polynomial assignment, with scalar physics."""
from dataclasses import asdict, replace
import hashlib
import json
import numpy as np
from ortools.graph.python import min_cost_flow
from ht_pdm_fjsp.maintenance_dispatch import eligible_edges, role_seed, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, observation
from ht_pdm_fjsp.maintenance_balanced_benchmark_env import fixed_config
from ht_pdm_fjsp.maintenance_skill_contention_screen import choose, urgency, quality, pending

VERSION = 'technician_relational_prefix_v2'


def scaled_config(n):
    if n not in (4,8,16,32): raise ValueError('registered machine scale required')
    original=fixed_config(); k=n//2
    return replace(original,machines=n,technicians=k,
        service_time=tuple(tuple(original.service_time[m%4][j%2] for j in range(k)) for m in range(n)),
        restoration=tuple(tuple(original.restoration[m%4][j%2] for j in range(k)) for m in range(n)))


def graph_observation(c,state,tick):
    obs=observation(replace(c,horizon=48),state,tick)
    obs['edges'][obs['edges'][...,0]<.5]=0
    mask=np.asarray(eligible_edges(c,state),dtype=bool)
    urgent=np.asarray([state.failed[m] or state.ages[m]>=c.failure_age-1 for m in range(c.machines)])
    obs['pair_mask']=mask & urgent[:,None]
    # Raw state is identical to original centralized benchmark at N4.
    obs['state']=np.concatenate([obs['machines'].ravel(),obs['technicians'].ravel(),
        obs['edges'].ravel(),obs['global_features']]).astype(np.float32)
    workers=obs['technicians'].copy();workers[:,2]=0
    obs['local']=np.concatenate([obs['machines'],np.tile(workers.ravel(),(c.machines,1)),
        obs['edges'].reshape(c.machines,-1),np.tile(obs['global_features'],(c.machines,1)),
        np.eye(c.machines,dtype=np.float32)],axis=1).astype(np.float32)
    obs['local_mask']=np.concatenate([np.ones((c.machines,1),bool),obs['pair_mask']],axis=1)
    return obs


def deadline(c,state):
    if c.machines==4: return choose(c,state,'deadline_matching')
    mask=graph_observation(c,state,0)['pair_mask'];n,k=mask.shape
    # k units, each worker sends one unit to an urgent machine or directly WAIT.
    flow=min_cost_flow.SimpleMinCostFlow();sink=n+k
    for j in range(k):
        flow.add_arc_with_capacity_and_unit_cost(n+j,sink,1,0)
        flow.set_node_supply(n+j,1)
        for m in range(n):
            if mask[m,j]:
                cost=-round(8*(urgency(c,state,m)+quality(c,m,j)))
                flow.add_arc_with_capacity_and_unit_cost(n+j,m,1,cost)
    for m in range(n): flow.add_arc_with_capacity_and_unit_cost(m,sink,1,0)
    flow.set_node_supply(sink,-k)
    if flow.solve()!=flow.OPTIMAL: raise RuntimeError('deadline matching solver failed')
    return tuple(sorted((flow.head(a),flow.tail(a)-n) for a in range(flow.num_arcs())
        if flow.flow(a) and flow.head(a)<n and flow.tail(a)>=n))


class ResourceEnv:
    def __init__(self,seed,n,method,worker=0,training=False,journal=None):
        self.config=scaled_config(n);self.env=WaitingEnv(self.config)
        self.seed=int(seed);self.method=method;self.worker=worker;self.training=training
        self.episode_index=-1;self.journal=journal;self.active=False
        self.actions=None
        if method=='ppo':
            if n!=4: raise ValueError('catalogue PPO only permitted at N4')
            from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue
            self.actions=catalogue(4,2)

    def observe(self):
        obs=graph_observation(self.config,self.env.state,self.env.time)
        if self.actions is not None:
            obs['joint_mask']=np.asarray([all(obs['pair_mask'][m,j] for m,j in a) for a in self.actions])
        return obs

    def reset(self):
        self.episode_index+=1
        namespace=f'{VERSION}:N{self.config.machines}'
        role=(f'{namespace}:training_physical:worker:{self.worker}:episode:{self.episode_index}'
            if self.training else f'{namespace}:physical')
        self.physical_seed=role_seed(self.seed,role);self.env.reset(self.physical_seed);self.active=True
        payload=json.dumps(dict(seed=self.physical_seed,state=asdict(self.env.state),events=self.env.events),sort_keys=True)
        self.population=hashlib.sha256(payload.encode()).hexdigest()
        return self.observe()

    def resolve(self,submitted):
        c=self.config;s=self.env.state;mask=self.observe()['pair_mask']
        conflicts=rejected=0
        if self.method=='ppo':
            if not isinstance(submitted,(int,np.integer)) or not 0<=submitted<len(self.actions): raise ValueError('bad catalogue action')
            pairs=self.actions[int(submitted)]
        elif self.method=='deadline': pairs=deadline(c,s)
        else:
            proposals=np.asarray(submitted)
            technician=self.method in ('reservation_ppo','mask_only_ppo','independent_ppo')
            count=c.technicians if technician else c.machines
            bound=c.machines if technician else c.technicians
            if proposals.shape!=(count,) or proposals.dtype.kind not in 'iu' or ((proposals<0)|(proposals>bound)).any():
                raise ValueError('bad proposals')
            requests=[(int(m)-1,j) for j,m in enumerate(proposals) if m] if technician else [(m,int(j)-1) for m,j in enumerate(proposals) if j]
            if any(not mask[m,j] for m,j in requests): raise ValueError('masked proposal')
            pairs=[]
            for index in range(c.machines if technician else c.technicians):
                group=[p for p in requests if p[0 if technician else 1]==index]
                if group:
                    conflicts+=int(len(group)>1);rejected+=len(group)-1
                    winner=max(group,key=lambda p:(quality(c,*p),-p[1])) if technician else max(group,key=lambda p:(urgency(c,s,p[0]),-p[0]))
                    pairs.append(winner)
            pairs=tuple(sorted(pairs))
        if any(not mask[m,j] for m,j in pairs): raise ValueError('nonurgent/busy assignment')
        validate_matching(c,s,pairs)
        return pairs,dict(conflicts=conflicts,rejected_proposals=rejected,executed_starts=len(pairs))

    def physical_step(self,pairs,phase):
        c=self.config;state=self.env.state;tick=self.env.time
        validate_matching(c,state,pairs)
        _,_,_,info=self.env.step(pairs)
        if self.journal:
            self.journal.add(dict(seed=self.seed,physical_seed=self.physical_seed,tick=tick,phase=phase,
                state=json.dumps(asdict(state)),action=json.dumps(pairs),events=json.dumps(self.env.events[tick]),
                following=json.dumps(asdict(self.env.state)),cost=info['objective']))
        return info['objective']

    def step(self,submitted):
        if not self.active: raise RuntimeError('reset before finished episode')
        pairs,coord=self.resolve(submitted);cost=self.physical_step(pairs,'policy');done=self.env.time==48
        info=dict(coordination=coord)
        if done:
            cutoff=dict(self.env.metrics);s=self.env.state;c=self.config
            metrics=dict(terminal_pending=pending(c,s),terminal_failed=sum(s.failed))
            for _ in range(12): cost+=self.physical_step(deadline(c,self.env.state),'tail')
            metrics.update(self.env.metrics,availability=1-cutoff['unavailable_ticks']/(c.machines*48),
                technician_utilization=cutoff['service_ticks']/(c.technicians*48),
                tail_pending=pending(c,self.env.state),tail_failed=sum(self.env.state.failed),
                policy_cost=cutoff['objective'],tail_cost=self.env.metrics['objective']-cutoff['objective'],
                extended_cost=self.env.metrics['objective'],physical_ticks=60,policy_failures=cutoff['failures'],
                policy_overdue_ticks=cutoff['overdue_waiting_ticks'],population_sha256=self.population,
                physical_seed=self.physical_seed,episode_index=self.episode_index)
            info['episode_metrics']=metrics;self.active=False
        return self.observe(),-cost/20,done,info
