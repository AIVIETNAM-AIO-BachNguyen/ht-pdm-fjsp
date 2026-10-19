import torch
from ht_pdm_fjsp import technician_prefix_activation_v5_experiment as x
from ht_pdm_fjsp import technician_prefix_activation_v5_env as e
from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from scripts.diagnose_prefix_activation_v5 import residual_off


def test_residual_off_preserves_encoder_and_only_removes_direct_logit_path():
    torch.set_num_threads(1)
    cfg=x.settings('smoke')
    full=x.make_model('reservation_ppo',4,cfg,69800000)
    off=residual_off(full)
    env=e.ResourceEnv(69900000,4,'reservation_ppo');env.reset()
    env.env.state=DispatchState((3,)*4,(False,)*4,(0,)*4,(0,)*2,(-1,)*2)
    batch=tensors([env.observe()])
    with torch.no_grad():
        original=full.encode(batch)
        off.residual_intervention_state['calls']=0
        intervened=off.encode(batch)
        assert off.residual_intervention_state['calls']==1
        assert all(torch.equal(a,b) for a,b in zip(original,intervened,strict=True))
        actions=torch.tensor([[1,2]])
        traces=[[],[]]
        full.trace(batch,recorded=actions,audit=traces[0])
        off.trace(batch,recorded=actions,audit=traces[1])
    assert len(traces[0])==len(traces[1])==2
    for a,b in zip(*traces,strict=True):
        for field in ('legal','resource','stats','prefix','candidate_capacity','wait_capacity','base_logits'):
            assert torch.equal(a[field],b[field])
        assert not torch.count_nonzero(b['prefix_residual'])
        assert torch.equal(a['full_logits'],b['full_logits']+a['prefix_residual'])
    assert torch.count_nonzero(traces[0][1]['prefix_residual'])
    assert off.residual_intervention_state['suppressed_nonzero']>0


def test_residual_off_same_weights_and_actions_stay_legal():
    cfg=x.settings('smoke')
    full=x.make_model('reservation_ppo',8,cfg,69800000)
    off=residual_off(full)
    assert all(torch.equal(a,b) for a,b in zip(full.state_dict().values(),off.state_dict().values(),strict=True))
    env=e.ResourceEnv(69900000,8,'reservation_ppo')
    obs=env.reset()
    for _ in range(4):
        a=full.select(obs)
        b=off.select(obs)
        assert a.shape==b.shape
        env.resolve(a)
        env.resolve(b)
        obs,_,_,_=env.step(a)
