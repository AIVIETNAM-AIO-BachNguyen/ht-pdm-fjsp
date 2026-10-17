"""Factorial entropy aggregation; original labeled matching densities unchanged."""
import torch
from ht_pdm_fjsp.technician_typed_gibbs_policy import TypedGibbsPolicy,CORE as BASE_CORE
from ht_pdm_fjsp import technician_typed_gibbs as algebra

CORE=('gibbs_joint_entropy_ppo','myopic_joint_entropy_ppo','reverse_joint_entropy_ppo',
      'gibbs_machine_entropy_ppo','myopic_machine_entropy_ppo','reverse_machine_entropy_ppo')
BASE={name:BASE_CORE[i%3] for i,name in enumerate(CORE)}


class EntropyUnitsPolicy(TypedGibbsPolicy):
    def __init__(self,algorithm,n=4,k=2,hidden=64):
        if algorithm not in CORE:raise ValueError('factorial entropy arm')
        super().__init__(BASE[algorithm],n,k,hidden)
        self.base_density=BASE[algorithm];self.algorithm=algorithm;self.entropy_aggregation='joint' if algorithm in CORE[:3] else 'machine'

    def tables(self,batch):
        # Experiment labels are NEVER used as density identifiers.
        features,types,available,eligible,caps=self.inputs(batch);utility=self.actor(features)
        order=torch.arange(self.n-1,-1,-1,device=utility.device) if self.base_density==BASE_CORE[2] else torch.arange(self.n,device=utility.device)
        table=algebra.policy_tables(utility[:,order],caps,eligible[:,order],mode='gibbs' if self.base_density==BASE_CORE[0] else 'myopic')
        return table,types,available,order

    def evaluate(self,batch,actions):
        log,per_machine_entropy,value=super().evaluate(batch,actions)
        return log,per_machine_entropy*(self.n if self.entropy_aggregation=='joint' else 1),value

    @torch.no_grad()
    def entropy_probe(self,batch):
        table,types,available,order=self.tables(batch);root=algebra.root_values(table,types,available)
        zero=algebra.policy_tables(torch.zeros_like(table['utilities']),table['capacities'],table['eligible'])
        log_count=algebra.root_values(zero,types,available)['log_partition'];inform=log_count>0
        ratio=root['entropy'][inform]/log_count[inform];mode=root['max_log_probability'][inform].exp();actions=algebra.maximum_matching(table,types,available)
        remaining=available.sum(-1).to(table['utilities'].dtype);before=[]
        for i in range(self.n):before.append(remaining.clone());remaining=remaining-(actions[:,i]>0).to(remaining.dtype)
        return dict(states=len(available),informative_states=int(inform.sum()),mean_normalized_entropy=float(ratio.mean()) if len(ratio) else 0.,mean_mode_probability=float(mode.mean()) if len(mode) else 1.,near_deterministic_fraction=float((mode>=.99).double().mean()) if len(mode) else 0.,mean_initial_available_workers=float(available.sum(-1).float().mean()),mean_available_workers_before_map_stage=float(torch.stack(before,-1).mean()),entropy_aggregation=self.entropy_aggregation,base_density=self.base_density)
