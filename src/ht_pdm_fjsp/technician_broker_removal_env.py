"""Fresh N4 physical streams, unchanged broker/action semantics."""
from dataclasses import asdict
import hashlib,json
from ht_pdm_fjsp.technician_broker_marginal_env import ResourceEnv as Base,scaled_config,value_unit,role_seed
from ht_pdm_fjsp.technician_broker_marginal_policy import decode_proposals
from ht_pdm_fjsp.technician_broker_removal_policy import CORE,ARM

VERSION='technician_broker_removal_ppo_v1'


class ResourceEnv(Base):
    def __init__(self,seed,n,method,worker=0,training=False,journal=None):
        if n!=4:raise ValueError('N4 only until registered gates pass')
        self.arm=method;super().__init__(seed,n,ARM if method in CORE else 'normalized_ppo' if method=='capacity_ppo' else method,worker,training,journal)

    def reset(self):
        self.episode_index+=1;role=f'{VERSION}:N4:training_physical:worker:{self.worker}:episode:{self.episode_index}' if self.training else f'{VERSION}:N4:physical';self.physical_seed=role_seed(self.seed,role);self.env.reset(self.physical_seed);self.active=True;self.broker_totals=dict.fromkeys(('order_forward','order_reverse','conflicts','rejected_proposals'),0);self.population=hashlib.sha256(json.dumps(dict(seed=self.physical_seed,state=asdict(self.env.state),events=self.env.events),sort_keys=True).encode()).hexdigest();return self.observe()

    def resolve(self,submitted):
        if self.arm not in CORE:return super().resolve(submitted)
        o=self.observe();return decode_proposals(submitted,o['pair_mask'],ARM,o['edges'])
