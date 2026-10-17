"""Same N4 physics/support, explicit experiment-to-density mapping and new streams."""
from dataclasses import asdict
import hashlib,json
from ht_pdm_fjsp.technician_typed_gibbs_env import ResourceEnv as Base,scaled_config,value_unit,role_seed
from ht_pdm_fjsp.technician_typed_gibbs_entropy_units_policy import BASE

VERSION='technician_typed_gibbs_entropy_units_v1'


class ResourceEnv(Base):
    def __init__(self,seed,n,method,worker=0,training=False,journal=None):
        self.experiment_arm=method
        super().__init__(seed,n,BASE.get(method,method),worker,training,journal)

    def reset(self):
        self.episode_index+=1;role=f'{VERSION}:N4:training_physical:worker:{self.worker}:episode:{self.episode_index}' if self.training else f'{VERSION}:N4:physical'
        self.physical_seed=role_seed(self.seed,role);self.env.reset(self.physical_seed);self.active=True;self.broker_totals=dict.fromkeys(('conflicts','rejected_proposals'),0)
        self.population=hashlib.sha256(json.dumps(dict(seed=self.physical_seed,state=asdict(self.env.state),events=self.env.events),sort_keys=True).encode()).hexdigest()
        return self.observe()
