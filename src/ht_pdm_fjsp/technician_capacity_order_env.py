"""Unchanged N4 physics, fresh streams and machine-agent canonical starts."""
from dataclasses import asdict
import hashlib,json
import numpy as np
from ht_pdm_fjsp.technician_reservation_env import ResourceEnv as Base,scaled_config,role_seed
from ht_pdm_fjsp.technician_completion_v2_env import value_unit
from ht_pdm_fjsp.technician_capacity_order_policy import CORE,decode_machine_tokens
from ht_pdm_fjsp.technician_raw_state_reservation_policy import decode_tokens

VERSION='technician_capacity_order_ppo_v1'


class ResourceEnv(Base):
    def __init__(self,seed,n,method,worker=0,training=False,journal=None):
        if n!=4:raise ValueError('N4 first; larger physical states closed')
        self.arm=method;super().__init__(seed,n,'ppo' if method=='normalized_ppo' else method,worker,training,journal)

    def observe(self):
        obs=super().observe()
        if self.arm=='full_mappo':obs['full_local']=np.concatenate([np.tile(obs['state'],(4,1)),np.eye(4,dtype=np.float32)],-1)
        return obs

    def reset(self):
        self.episode_index+=1;role=f'{VERSION}:N4:training_physical:worker:{self.worker}:episode:{self.episode_index}' if self.training else f'{VERSION}:N4:physical';self.physical_seed=role_seed(self.seed,role);self.env.reset(self.physical_seed);self.active=True;self.broker_totals=dict.fromkeys(('order_forward','order_reverse','conflicts','rejected_proposals'),0);self.population=hashlib.sha256(json.dumps(dict(seed=self.physical_seed,state=asdict(self.env.state),events=self.env.events),sort_keys=True).encode()).hexdigest();return self.observe()

    def resolve(self,action):
        if self.arm in CORE:
            o=self.observe();return decode_machine_tokens(action,o['pair_mask'],o['edges'],self.arm)
        if self.arm=='raw_tech_ppo':
            o=self.observe();return decode_tokens(action,o['pair_mask'],'raw_reservation_ppo',o['edges'])
        pairs,diag=super().resolve(action);return pairs,dict(diag,order_forward=0,order_reverse=0,prefix_enabled=0)

    def step(self,action):
        obs,reward,done,info=super().step(action)
        for key in self.broker_totals:self.broker_totals[key]+=info['coordination'].get(key,0)
        if done:info['episode_metrics'].update({'broker_'+k:v for k,v in self.broker_totals.items()})
        return obs,reward,done,info
