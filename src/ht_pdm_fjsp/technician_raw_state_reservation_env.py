"""Fresh paired N4/K2 physics; within-tick canonical reservation tokens."""
from dataclasses import asdict
import hashlib,json
from ht_pdm_fjsp.maintenance_dispatch import role_seed,validate_matching
from ht_pdm_fjsp.technician_joint_clearing_env import ResourceEnv as BaseEnv,scaled_config,value_unit
from ht_pdm_fjsp.technician_raw_state_reservation_policy import CORE_ARMS,decode_tokens

VERSION='technician_raw_state_reservation_ppo_v1'


class ResourceEnv(BaseEnv):
    def __init__(self,seed,n,method,worker=0,training=False,journal=None):
        super().__init__(seed,n,'joint_clearing_ppo' if method in CORE_ARMS else method,worker,training,journal)
        self.method=method

    def reset(self):
        self.episode_index+=1
        role=(f'{VERSION}:N4:training_physical:worker:{self.worker}:episode:{self.episode_index}'
              if self.training else f'{VERSION}:N4:physical')
        self.physical_seed=role_seed(self.seed,role);self.env.reset(self.physical_seed);self.active=True
        self.broker_totals=dict.fromkeys(('order_forward','order_reverse','conflicts','rejected_proposals'),0)
        payload=json.dumps(dict(seed=self.physical_seed,state=asdict(self.env.state),events=self.env.events),sort_keys=True)
        self.population=hashlib.sha256(payload.encode()).hexdigest()
        return self.observe()

    def resolve(self,submitted):
        if self.method not in CORE_ARMS:return super().resolve(submitted)
        obs=self.observe();pairs,info=decode_tokens(submitted,obs['pair_mask'],self.method,obs['edges'])
        validate_matching(self.config,self.env.state,pairs);return pairs,info
