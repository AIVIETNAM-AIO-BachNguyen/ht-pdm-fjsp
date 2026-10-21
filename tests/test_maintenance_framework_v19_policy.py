"""Independent combinatorial checks for V19 present-information exposure features."""
from __future__ import annotations

from itertools import permutations
import numpy as np
import pytest
import torch
from ht_pdm_fjsp.maintenance_framework_v1 import FrameworkEnv,catalogue,configs
from ht_pdm_fjsp.maintenance_framework_v19_policy import TemporalMatchingActor,matching_profiles

ACTIONS=catalogue()


def independent_profile(obs,action,tau):
    machines=obs[2:34].reshape(4,8)
    workers=obs[34:40].reshape(2,3)
    edges=obs[40:56].reshape(4,2,2)
    selected={m for m,_ in action}
    demand=[m for m in range(4) if m not in selected and machines[m,5]<.5 and
            (machines[m,2]>.5 or round(float(machines[m,0])*8)+tau>=3)]
    worker_available=[]
    for j in range(2):
        assigned_duration=next((round(float(edges[m,j,1])*12) for m,k in action if k==j),0)
        remaining=round(float(workers[j,1])*12)
        if max(assigned_duration,remaining)<=tau:worker_available.append(j)
    feasible=0
    for size in range(1,min(len(demand),len(worker_available))+1):
        if any(all(edges[m,j,0]>.5 for m,j in zip(choice,perm))
               for choice in permutations(demand,size)
               for perm in permutations(worker_available,size)):
            feasible=size
    return len(demand)/4,(len(demand)-feasible)/4


@pytest.mark.parametrize('case',tuple(configs()))
@pytest.mark.parametrize('seed', (73420000,73420001))
def test_profile_matches_independent_enumeration(case,seed):
    env=FrameworkEnv(configs()[case]);obs,_=env.reset(seed=seed)
    observations=[]
    for tick in range(12):
        observations.append(obs)
        legal=np.flatnonzero(env.action_masks())
        action=int(legal[(tick*3)%len(legal)])
        obs,_,done,_,_=env.step(action)
        assert not done
    x=np.stack(observations)
    result=matching_profiles(x,'temporal_deficit')
    snapshot=matching_profiles(x,'snapshot_deficit')
    assert result.shape==(12,len(ACTIONS),18)
    np.testing.assert_array_equal(result[:,:,:9],snapshot[:,:,:9])
    np.testing.assert_array_equal(snapshot[:,:,9:],np.broadcast_to(snapshot[:,:,9:10],snapshot[:,:,9:].shape))
    for i in (0,3,7,11):
        for a in range(len(ACTIONS)):
            for tau in (1,2,3,6,9):
                d,l=independent_profile(x[i],ACTIONS[a],tau)
                assert result[i,a,tau-1]==d
                assert result[i,a,9+tau-1]==l
                assert snapshot[i,a,9+tau-1]==result[i,a,9]


def test_equal_capacity_and_only_deficit_inputs_change():
    obs,_=FrameworkEnv(configs()['reciprocal_scale3']).reset(seed=73420000)
    torch.manual_seed(73400000)
    a=TemporalMatchingActor('temporal_deficit')
    torch.manual_seed(73400000)
    b=TemporalMatchingActor('snapshot_deficit')
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    left=a.feature_tensor(torch.tensor(obs).reshape(1,56))
    right=b.feature_tensor(torch.tensor(obs).reshape(1,56))
    assert left.shape==(1,len(ACTIONS),92)
    assert torch.equal(left[:,:,:83],right[:,:,:83])
    assert torch.equal(left[:,:,74:83],right[:,:,74:83])
    assert torch.equal(right[:,:,83:],right[:,:,83:84].expand_as(right[:,:,83:]))


def test_invalid_inputs():
    with pytest.raises(ValueError):matching_profiles(np.zeros((56,),np.float32),'temporal_deficit')
    with pytest.raises(ValueError):matching_profiles(np.zeros((1,56),np.float32),'other')
