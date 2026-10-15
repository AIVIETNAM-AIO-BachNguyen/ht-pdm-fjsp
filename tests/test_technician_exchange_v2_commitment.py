"""Commitment, immediate-inverse history and equivalent neural replay."""
from copy import deepcopy
import itertools

import numpy as np
import pytest
import torch

from ht_pdm_fjsp.technician_exchange_v2_env import CORE_ARMS, decode_trace, ResourceEnv, value_unit
from ht_pdm_fjsp.technician_exchange_v2_policy import ExchangePolicy, calendar_table
from ht_pdm_fjsp.technician_exchange_policy import ExchangePolicy as LiteralV1
from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors
from ht_pdm_fjsp.maintenance_dispatch import DispatchState


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def batch_at(n, count=5):
    observations=[]
    for seed in range(count):
        env=ResourceEnv(160100000+seed%2,n,'exchange_ppo');env.reset()
        env.env.state=DispatchState(tuple(3+(i+seed)%3 for i in range(n)),
            (False,)*n,(0,)*n,(0,)*(n//2),(-1,)*(n//2))
        observations.append(env.observe())
    return tensors(observations)


def test_commitment_and_inverse_control_have_distinct_support():
    mask=np.ones((4,2),bool)
    trace=np.array([1,2,0,1])  # Late first exchange revises a previously KEEP-committed job.
    for arm in ('unrestricted_exchange_ppo','inverse_mask_ppo'):
        assert decode_trace(mask,trace,arm)[0].tolist()==[2,1]
    with pytest.raises(ValueError,match='committed'):decode_trace(mask,trace,'exchange_ppo')


@pytest.mark.parametrize('trace', [[1,2,2,2],[0,2,2,2]])
def test_immediate_inverse_blocks_swap_and_transfer_reversal(trace):
    mask=np.ones((4,2),bool);trace=np.array(trace)
    _,diag=decode_trace(mask,trace,'unrestricted_exchange_ppo')
    assert diag['repair_cancellations']==1
    with pytest.raises(ValueError,match='inverse'):decode_trace(mask,trace,'inverse_mask_ppo')
    with pytest.raises(ValueError,match='committed'):decode_trace(mask,trace,'exchange_ppo')


@pytest.mark.parametrize('middle', [0,4,7])
def test_inverse_control_resets_after_keep_replacement_or_drop(middle):
    mask=np.ones((6,3),bool);trace=np.array([1,2,3,3,middle,3])
    plan,_=decode_trace(mask,trace,'inverse_mask_ppo')
    assert plan[0]==1 and plan[2]==3
    with pytest.raises(ValueError,match='committed'):decode_trace(mask,trace,'exchange_ppo')


def test_all_small_ordered_repairs_preserve_committed_prefix_and_cannot_cancel():
    mask=np.ones((4,2),bool);valid=0
    for trace in itertools.product(range(6),repeat=4):
        try:final,diag=decode_trace(mask,np.array(trace),'exchange_ppo')
        except ValueError:continue
        valid+=1
        prefix=decode_trace(mask,np.array([*trace[:3],0]),'exchange_ppo')[0]
        assert final[0]==prefix[0]
        assert not diag['repair_cancellations']
        assert bool(diag['revisions'])==bool(diag['net_plan_changes'])
    assert valid>21


@pytest.mark.parametrize('arm', CORE_ARMS)
@pytest.mark.parametrize('n', [4,32])
def test_batched_replay_matches_literal_probabilities_values_and_all_parameter_gradients(arm,n):
    torch.manual_seed(160000000);model=ExchangePolicy(arm);batch=batch_at(n)
    actions=model.sample(batch,torch.Generator().manual_seed(5))[0]
    coefficient=torch.linspace(-1.,1.,len(actions))[:,None]
    gradients=[];outputs=[]
    for mode in ('fast','sequential','literal'):
        model.zero_grad(set_to_none=True)
        result=model.trace(batch,recorded=actions,sequential=mode=='sequential',literal=mode=='literal')[1:]
        log,entropy,value=result
        loss=(log*coefficient).sum(1).mean()-.01*entropy.mean()+.5*(value/value_unit(n)).square().mean()
        loss.backward()
        gradients.append({k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None})
        outputs.append(tuple(t.detach().clone() for t in result))
    for other in outputs[1:]:
        for a,b in zip(outputs[0],other):assert torch.allclose(a,b,atol=2e-5,rtol=0)
    for other in gradients[1:]:
        assert set(other)==set(gradients[0])
        for key,a in gradients[0].items():assert torch.allclose(a,other[key],atol=2e-5,rtol=2e-5),key


def test_unrestricted_control_retains_v1_neural_policy_with_fresh_equal_parameters():
    torch.manual_seed(160000000);new=ExchangePolicy('unrestricted_exchange_ppo')
    old=LiteralV1('exchange_ppo');old.load_state_dict(deepcopy(new.state_dict()))
    batch=batch_at(8);actions=new.sample(batch,torch.Generator().manual_seed(7))[0]
    for a,b in zip(new.evaluate(batch,actions),old.evaluate(batch,actions)):
        assert torch.allclose(a,b,atol=2e-5,rtol=0)


def test_calendar_memoization_preserves_input_bits_and_parameter_gradients():
    torch.manual_seed(17);encoder=ExchangePolicy('exchange_ppo').calendar_encoder
    raw=torch.tensor([[1.,1/12,3/8,0.],[1.,1/12,3/8,1.],
                      [0.,0.,5/8,0.],[0.,0.,5/8,1.]]).repeat(1024,1)
    expected=encoder(raw);expected.mean().backward()
    grads=[p.grad.clone() for p in encoder.parameters()]
    encoder.zero_grad(set_to_none=True);actual=calendar_table(encoder,raw)
    actual.mean().backward()
    assert torch.allclose(actual,expected,atol=2e-6,rtol=0)
    for p,g in zip(encoder.parameters(),grads):assert torch.allclose(p.grad,g,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('learned_input',[False,True])
def test_calendar_noncanonical_or_differentiable_inputs_use_literal_path(learned_input):
    torch.manual_seed(19);encoder=ExchangePolicy('exchange_ppo').calendar_encoder
    raw=torch.tensor([[1.,.101,3/8,0.],[1.,1/12,.375001,1.]],requires_grad=learned_input)
    expected=encoder(raw);actual=calendar_table(encoder,raw)
    assert torch.equal(expected,actual)
    if learned_input:
        left=torch.autograd.grad(expected.sum(),raw,retain_graph=True)[0]
        right=torch.autograd.grad(actual.sum(),raw)[0]
        assert torch.equal(left,right)
