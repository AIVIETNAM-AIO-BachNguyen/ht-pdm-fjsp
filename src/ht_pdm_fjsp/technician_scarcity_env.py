"""Teacher-free allocation traces on unchanged scalar maintenance physics."""
from dataclasses import asdict
import hashlib
import json
import numpy as np
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.technician_reservation_env import ResourceEnv as BaseEnv, scaled_config, deadline

VERSION = 'technician_scarcity_ppo_v1'
GRAPH_ARMS = ('scarcity_ppo', 'reservation_ppo', 'independent_ppo')

class ResourceEnv(BaseEnv):
    def reset(self):
        self.episode_index += 1
        namespace=f'{VERSION}:N{self.config.machines}'
        role=(f'{namespace}:training_physical:worker:{self.worker}:episode:{self.episode_index}'
              if self.training else f'{namespace}:physical')
        self.physical_seed=role_seed(self.seed,role)
        self.env.reset(self.physical_seed);self.active=True
        payload=json.dumps(dict(seed=self.physical_seed,state=asdict(self.env.state),events=self.env.events),sort_keys=True)
        self.population=hashlib.sha256(payload.encode()).hexdigest()
        return self.observe()

    def observe(self):
        obs=super().observe()
        if self.method=='full_mappo':
            n=self.config.machines
            obs['full_local']=np.concatenate([np.tile(obs['state'],(n,1)),np.eye(n,dtype=np.float32)],axis=1)
        return obs

    def resolve(self,submitted):
        if self.method not in ('scarcity_ppo','reservation_ppo'):
            # All non-broker arms are machine proposals, including independent.
            if self.method=='independent_ppo':
                previous=self.method;self.method='machine_independent'
                try:return super().resolve(submitted)
                finally:self.method=previous
            return super().resolve(submitted)
        c=self.config;mask=self.observe()['pair_mask'].copy();tokens=np.asarray(submitted)
        if tokens.shape!=(c.technicians,) or tokens.dtype.kind not in 'iu' or ((tokens<0)|(tokens>c.machines*c.technicians)).any():
            raise ValueError('bad edge trace')
        pairs=[];stopped=False
        for token in tokens:
            if token==0:stopped=True;continue
            if stopped:raise ValueError('noncanonical STOP suffix')
            m,j=divmod(int(token)-1,c.technicians)
            if not mask[m,j]:raise ValueError('masked/repeated resource')
            pairs.append((m,j));mask[m,:]=False;mask[:,j]=False
        pairs=tuple(sorted(pairs));validate_matching(c,self.env.state,pairs)
        return pairs,dict(conflicts=0,rejected_proposals=0,executed_starts=len(pairs))
