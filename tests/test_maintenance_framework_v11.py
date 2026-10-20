"""V11 paired continuation banks and equal-compute action-credit update."""
import numpy as np

from ht_pdm_fjsp import maintenance_framework_v11_campaign as v11


def first_multi_action_record():
    cfg=v11.configs()['reciprocal_scale2']
    env=v11.FrameworkEnv(cfg);env.reset(seed=70590000)
    for tick in range(v11.HORIZON):
        state=env.inner.state
        _,_,legal,obs=v11.legal_present(cfg,state,tick)
        if len(legal)>=2:
            return cfg,dict(case='reciprocal_scale2',physical_seed=70590000,tick=tick,
                            original_state=state,observation=obs,legal=legal,source_action=legal[0])
        env.step(legal[0])
    raise AssertionError('no multi-action state')


def test_paired_full_horizon_bank_and_reference_actor_independence():
    cfg,record=first_multi_action_record()
    actor=v11.actor_new(70500000);other=v11.actor_new(70500001)
    seed,bank=v11.scenario_bank(70520000,70500000,1,record,2)
    assert (seed,bank)==v11.scenario_bank(70520000,70500000,1,record,2)
    root=record['legal'][0];events,uniforms=bank[0]
    left,steps=v11.forecast_cost(actor,cfg,record,root,events,uniforms,'reference',lambda:None)
    right,_=v11.forecast_cost(other,cfg,record,root,events,uniforms,'reference',lambda:None)
    assert left==right and steps==cfg.horizon-record['tick']
    self_q,ref_q,n_self,n_ref,bank_seed=v11.counterfactual_values(
        actor,record,70520000,70500000,1,2,lambda:None)
    assert bank_seed==seed
    assert set(self_q)==set(ref_q)==set(record['legal'])
    assert n_self==n_ref==len(record['legal'])*2*(cfg.horizon-record['tick'])
    assert all(np.isfinite(x) for x in (*self_q.values(),*ref_q.values()))


def test_same_update_code_uses_selected_q_values():
    _,record=first_multi_action_record()
    legal=record['legal'];left=v11.actor_new(70500000);right=v11.actor_new(70500000)
    assert v11.actor_digest(left)==v11.actor_digest(right)
    q_self={a:100.+i for i,a in enumerate(legal)}
    q_ref={a:100.-i for i,a in enumerate(legal)}
    first=v11.update_actor(left,[record],[q_self],'self_credit',70500000,1,lambda:None,epochs=1)
    second=v11.update_actor(right,[record],[q_ref],'reference_credit',70500000,1,lambda:None,epochs=1)
    assert first['updates']>=1 and second['updates']>=1
    assert v11.actor_digest(left)!=v11.actor_digest(right)


def test_v11_registry_is_disjoint():
    reg,groups=v11.seed_registry()
    assert reg['campaign_version']==v11.VERSION
    assert tuple(groups[0])==tuple(range(70500000,70500005))
    assert not set(groups[0])&set(groups[1])
