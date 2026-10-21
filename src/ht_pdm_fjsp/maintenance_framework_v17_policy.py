"""Full-matching pre-decision value policy shared by both V17 credit objectives."""
from __future__ import annotations
import torch
from torch import nn
from ht_pdm_fjsp.maintenance_framework_v1 import catalogue

ACTIONS=catalogue()

class FullMatchingActor(nn.Module):
    def __init__(self):
        super().__init__()
        incidence=torch.zeros((len(ACTIONS),4,2),dtype=torch.float32)
        for a,matching in enumerate(ACTIONS):
            for m,j in matching:incidence[a,m,j]=1
        self.register_buffer('incidence',incidence.flatten(1))
        self.register_buffer('served',incidence.sum(dim=2).clamp(max=1))
        self.net=nn.Sequential(nn.Linear(74,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))

    def feature_tensor(self,observation):
        if observation.ndim!=2 or observation.shape[1]!=56:
            raise ValueError('expected [batch,56] current observation')
        batch=observation.shape[0]
        rem=observation[:,[35,38]]
        comp=observation[:,40:56].reshape(batch,4,2,2)[...,0]>0
        earliest=rem[:,None,:].expand(-1,4,-1).masked_fill(~comp,float('inf')).min(dim=2).values
        if not torch.isfinite(earliest).all():raise ValueError('machine without compatible technician')
        release=torch.cat((rem,earliest.clamp(min=1/12)),dim=1)
        features=torch.cat((
            observation[:,None,:].expand(-1,len(ACTIONS),-1),
            self.incidence[None,:,:].expand(batch,-1,-1),
            self.served[None,:,:].expand(batch,-1,-1),
            release[:,None,:].expand(-1,len(ACTIONS),-1)),dim=2)
        assert features.shape==(batch,len(ACTIONS),74)
        return features

    def forward(self,observation):
        return -self.net(self.feature_tensor(observation)).squeeze(-1)
