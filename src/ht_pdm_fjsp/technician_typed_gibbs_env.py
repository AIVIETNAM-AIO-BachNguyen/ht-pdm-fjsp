"""Unchanged N4 maintenance physics and independent fresh typed-Gibbs RNG roles."""
from dataclasses import asdict
import hashlib,json
import numpy as np
from ht_pdm_fjsp.technician_reservation_env import ResourceEnv as Base,scaled_config,role_seed
from ht_pdm_fjsp.technician_completion_v2_env import value_unit
from ht_pdm_fjsp.technician_typed_gibbs_policy import CORE,decode_tokens

VERSION='technician_typed_gibbs_ppo_v1'


class ResourceEnv(Base):
    def __init__(self,seed,n,method,worker=0,training=False,journal=None):
        if n!=4:raise ValueError('N4 first; larger physical settings closed')
        self.arm=method;super().__init__(seed,n,'ppo' if method=='normalized_ppo' else method,worker,training,journal)
        c=self.config
        for t in (0,1):
            ix=list(range(t,c.technicians,2))
            if any(len({c.service_time[m][j] for j in ix})!=1 or len({c.restoration[m][j] for j in ix})!=1 for m in range(n)):raise ValueError('two repeated technician profiles required')

    def observe(self):
        obs=super().observe()
        if self.arm=='full_mappo':obs['full_local']=np.concatenate([np.tile(obs['state'],(4,1)),np.eye(4,dtype=np.float32)],-1)
        return obs

    def reset(self):
        self.episode_index+=1;role=f'{VERSION}:N4:training_physical:worker:{self.worker}:episode:{self.episode_index}' if self.training else f'{VERSION}:N4:physical';self.physical_seed=role_seed(self.seed,role);self.env.reset(self.physical_seed);self.active=True;self.broker_totals=dict.fromkeys(('conflicts','rejected_proposals'),0);self.population=hashlib.sha256(json.dumps(dict(seed=self.physical_seed,state=asdict(self.env.state),events=self.env.events),sort_keys=True).encode()).hexdigest();return self.observe()

    def resolve(self,action):
        if self.arm in CORE:return decode_tokens(action,self.observe()['pair_mask'])
        return super().resolve(action)

    def step(self,action):
        obs,reward,done,info=super().step(action)
        for key in self.broker_totals:self.broker_totals[key]+=info['coordination'].get(key,0)
        if done:info['episode_metrics'].update({'broker_'+k:v for k,v in self.broker_totals.items()})
        return obs,reward,done,info
