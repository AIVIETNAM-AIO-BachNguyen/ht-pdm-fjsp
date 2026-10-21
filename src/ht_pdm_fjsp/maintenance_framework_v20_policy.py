"""V20 count-calibrated V14 actor with explicit WAIT/start probability."""
from __future__ import annotations

import torch
from torch import nn

from ht_pdm_fjsp.maintenance_framework_v14_policy import ACTIONS,OpportunityCostActor

WAIT=ACTIONS.index(())


class ActivationMarginActor(nn.Module):
    """Identical V14 parameters in both V20 arms; action map is shared."""

    def __init__(self):
        super().__init__()
        self.core=OpportunityCostActor('count_calibrated')

    def forward(self,observation:torch.Tensor)->torch.Tensor:
        return self.core(observation)


def factorized_probabilities(logits:torch.Tensor,legal:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Return full policy, WAIT probability and conditional service policy."""
    if logits.ndim!=2 or legal.shape!=logits.shape or legal.dtype!=torch.bool:
        raise ValueError('expected [batch,actions] logits and boolean legal mask')
    if not bool(torch.all(legal[:,WAIT])):
        raise ValueError('WAIT must be legal')
    service=legal.clone();service[:,WAIT]=False
    count=service.sum(dim=1)
    service_logits=logits.masked_fill(~service,-1e9)
    pool=torch.logsumexp(service_logits,dim=1)-count.clamp(min=1).to(logits.dtype).log()
    p_start=torch.where(count>0,torch.sigmoid(pool-logits[:,WAIT]),torch.zeros_like(pool))
    conditional=torch.softmax(service_logits,dim=1).masked_fill(~service,0.)
    full=p_start[:,None]*conditional
    full=full.clone();full[:,WAIT]=1.-p_start
    return full,1.-p_start,conditional


def greedy_action(actor:ActivationMarginActor,observation,legal_actions)->int:
    with torch.no_grad():
        obs=torch.as_tensor(observation).reshape(1,56)
        mask=torch.zeros((1,len(ACTIONS)),dtype=torch.bool)
        mask[0,list(legal_actions)]=True
        probabilities,_,_=factorized_probabilities(actor(obs),mask)
    return int(torch.argmax(probabilities[0]))


def activation_targets(q:torch.Tensor,legal:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    """Fixed soft WAIT target and bounded absolute-margin weight from root costs."""
    if q.shape!=legal.shape or q.ndim!=2 or legal.dtype!=torch.bool:
        raise ValueError('Q/mask shape')
    service=legal.clone();service[:,WAIT]=False
    best=q.masked_fill(~service,float('inf')).min(dim=1).values
    margin=torch.where(service.any(dim=1),best-q[:,WAIT],torch.zeros_like(best))
    target=torch.sigmoid(margin/10.)
    weight=(margin.abs()/10.).clamp(max=2.)
    return target,weight


def conditional_service_gain(conditional:torch.Tensor,q:torch.Tensor,legal:torch.Tensor)->torch.Tensor:
    """Per-state centered conditional service cost improvement, zero if no service."""
    service=legal.clone();service[:,WAIT]=False
    count=service.sum(dim=1).clamp(min=1)
    mean=(q.masked_fill(~service,0.)).sum(dim=1)/count
    advantage=((mean[:,None]-q)/10.).masked_fill(~service,0.)
    return (conditional*advantage).sum(dim=1)
