import numpy as np
import pytest

from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.technician_diverse_maintenance_env import (
    DiverseEnv, FAMILIES, SCALES, family_config,
)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('n', SCALES)
def test_preregistered_config_and_request_conservation(family,n):
    c=family_config(family,n)
    assert c.technicians==(3*n//8 if family=='skill_bottleneck' else n//2)
    assert c.unavailable_cost==6 and c.waiting_price==12
    for m in range(n):
        assert any(c.service_time[m])
    for j in range(c.technicians):
        assert any(c.service_time[m][j] for m in range(n))
    if family=='skill_bottleneck':
        assert set(d for row in c.service_time for d in row)=={0,1}
    else:
        assert set(d for row in c.service_time for d in row)=={0,2,4}
    env=DiverseEnv(65800000,n,family,'fcfs')
    obs=env.reset();rewards=0.
    for _ in range(48):
        assert obs['pair_mask'].shape==(n,c.technicians)
        assert not np.any(obs['pair_mask'] &
            (np.asarray(env.env.state.remaining)[None,:]>0))
        obs,reward,done,info=env.step(None);rewards+=reward
    assert done and info['episode_metrics']['physical_ticks']==60
    m=info['episode_metrics']
    assert abs(-20*rewards-m['extended_cost'])<1e-7
    assert m['invalid_assignments']==0
    assert m['requests']==m['completions']+m['tail_pending']+m['active_end']
    assert m['starts']==m['completions']+m['active_end']


def test_same_tape_all_arms_and_pending_age_adapter():
    arms=('reservation_ppo','mask_only_ppo','independent_ppo','fcfs','risk')
    envs=[DiverseEnv(65800000,8,'skill_bottleneck',arm) for arm in arms]
    for env in envs:env.reset()
    assert len({env.physical_seed for env in envs})==1
    assert len({env.population for env in envs})==1
    env=envs[0]
    c=env.config
    env.env.state=DispatchState((3,)*8,(False,)*8,(0,)*8,
        (0,)*c.technicians,(-1,)*c.technicians)
    env.pending={0,1};env.since={0:0,1:0}
    envs[1].env.state=env.env.state
    envs[1].pending=set(env.pending);envs[1].since=dict(env.since)
    obs=env.observe()
    assert obs['machines'][0,2]==pytest.approx(1/12)
    assert obs['machines'][2,2]==0
    assert not obs['pair_mask'][2].any()
    assert np.array_equal(obs['pair_mask'],envs[1].observe()['pair_mask'])
    assert np.array_equal(obs['machines'],envs[1].observe()['machines'])


def test_coordinated_duplicate_and_unrequested_proposal_rejected():
    env=DiverseEnv(65800000,8,'skill_bottleneck','reservation_ppo');env.reset()
    c=env.config
    env.env.state=DispatchState((3,)*8,(False,)*8,(0,)*8,
        (0,)*c.technicians,(-1,)*c.technicians)
    env.pending={0,1};env.since={0:0,1:0}
    with pytest.raises(ValueError,match='duplicate'):
        env.resolve(np.asarray([1,0,1]))
    with pytest.raises(ValueError,match='unrequested'):
        env.resolve(np.asarray([4,0,0]))
