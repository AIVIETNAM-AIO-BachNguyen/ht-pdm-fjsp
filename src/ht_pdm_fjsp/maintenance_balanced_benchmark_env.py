"""Frozen balanced 2–2 physics and simultaneous-proposal arbitration."""
from dataclasses import asdict
import hashlib
import json
import numpy as np

from ht_pdm_fjsp.maintenance_joint_learning_env import JointLearningEnv
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig

VERSION = 'maintenance_balanced_algorithm_benchmark_v1'


def fixed_config():
    return WaitingConfig(machines=4, technicians=2, horizon=60, failure_age=4,
        failure_probability=.45, service_time=((1,2),(1,4),(2,1),(4,1)),
        restoration=((.5,1.),(.5,1.),(1.,.5),(1.,.5)), max_age=8,
        preventive_cost=1., corrective_cost=2., unavailable_cost=12., failure_cost=15.,
        waiting_limit=4, waiting_price=12.)


def observation(inner):
    state = inner._observation()
    machines = state[:24].reshape(4,6)
    public_workers = state[24:30].reshape(2,3).copy()
    public_workers[:,2] = 0.  # Never reveal another machine's failure status.
    edges = state[30:62].reshape(4,8)
    local = np.concatenate((machines, np.tile(public_workers.ravel(),(4,1)), edges,
        np.tile(state[62:],(4,1)), np.eye(4,dtype=np.float32)),axis=1).astype(np.float32)
    joint = inner.action_masks()
    masks = np.zeros((4,3),dtype=bool); masks[:,0] = True
    for action, legal in zip(inner.actions,joint):
        if legal:
            for machine,worker in action:
                masks[machine,worker+1] = True
    return dict(state=state, local=local, joint_mask=joint, local_mask=masks)


def arbitrate(inner, proposals):
    proposed = np.asarray(proposals)
    if proposed.shape != (4,) or proposed.dtype.kind not in 'iu' or ((proposed<0)|(proposed>2)).any():
        raise ValueError('four integer WAIT/A/B proposals required')
    masks = observation(inner)['local_mask']
    if not masks[np.arange(4),proposed].all():
        raise ValueError('masked local proposal')
    state=inner.env.state
    priority = [100*state.failed[m]+20*max(state.pending_wait[m]-3,0)
        +10*(inner.config.failure_probability if state.ages[m]+1>=inner.config.failure_age and not state.failed[m] else 0)
        +state.ages[m]/8 for m in range(4)]
    matching=[]; contested=0; rejected=0
    for worker in range(2):
        pool=[m for m in range(4) if proposed[m]==worker+1]
        if pool:
            matching.append((max(pool,key=lambda m:(priority[m],-m)),worker))
            contested += int(len(pool)>1); rejected += len(pool)-1
    action=tuple(sorted(matching))
    if action not in inner.actions or not inner.action_masks()[inner.actions.index(action)]:
        raise RuntimeError('arbitration produced an illegal matching')
    feasible_capacity=max(map(len,(a for a,ok in zip(inner.actions,inner.action_masks()) if ok)))
    return action,dict(proposals=proposed.tolist(), conflict_workers=contested,
        rejected_proposals=rejected, proposed_starts=int((proposed!=0).sum()),
        executed_starts=len(action), unused_feasible_capacity=feasible_capacity-len(action))


class BenchmarkEnv:
    def __init__(self, seed, worker=0, training=False, journal=None):
        self.seed=int(seed); self.first=True
        self.inner=JointLearningEnv(fixed_config(),namespace=VERSION,worker=worker,
            training=training,journal=journal)
        self.initial_population=None

    def reset(self):
        self.inner.reset(seed=self.seed if self.first or not self.inner.training else None)
        self.first=False
        payload=json.dumps(dict(seed=self.inner.physical_seed,state=asdict(self.inner.env.state),
            events=self.inner.env.events),sort_keys=True,separators=(',',':'))
        self.initial_population=hashlib.sha256(payload.encode()).hexdigest()
        return observation(self.inner)

    def step(self, action):
        if np.isscalar(action):
            index=action
            matching=self.inner.actions[int(action)] if isinstance(action,(int,np.integer)) and 0<=action<len(self.inner.actions) else ()
            diagnostic=dict(proposals=None,conflict_workers=0,rejected_proposals=0,
                proposed_starts=len(matching),executed_starts=len(matching),
                unused_feasible_capacity=max(map(len,(a for a,ok in zip(self.inner.actions,self.inner.action_masks()) if ok)))-len(matching))
        else:
            matching,diagnostic=arbitrate(self.inner,action)
            index=self.inner.actions.index(matching)
        _,reward,done,truncated,info=self.inner.step(index)
        if truncated: raise RuntimeError('unexpected truncation')
        if done:
            info['episode_metrics']['population_sha256']=self.initial_population
            self.inner.episode_metrics['population_sha256']=self.initial_population
        return observation(self.inner),reward,done,dict(info,coordination=diagnostic)
