from dataclasses import replace
import itertools
import numpy as np
import pytest
import torch

from ht_pdm_fjsp.technician_exchange_v2_env import CORE_ARMS, ResourceEnv, decode_trace, value_unit, VERSION
from ht_pdm_fjsp.technician_exchange_v2_policy import ExchangePolicy, apply_move, move_support, provisional_calendar, preceding_owner
from ht_pdm_fjsp.maintenance_balanced_algorithms import tensors, parameter_digest, fit
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_matching


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def urgent(n=4, arm='exchange_ppo'):
    env = ResourceEnv(160100000, n, arm)
    env.reset()
    env.env.state = DispatchState((3,)*n, (False,)*n, (0,)*n, (0,)*(n//2), (-1,)*(n//2))
    return env


def model(arm):
    with torch.random.fork_rng():
        torch.manual_seed(160000000)
        return ExchangePolicy(arm)


def test_scalar_bilateral_swap_transfer_drop_and_reciprocal_legality():
    mask = np.ones((4, 2), bool)
    plan, diag = decode_trace(mask, np.array([1, 2, 2, 0]), 'exchange_ppo')
    assert plan.tolist() == [2, 1] and diag['exchanges'] == 1 and diag['transfers'] == 0
    plan, diag = decode_trace(mask, np.array([0, 2, 2, 0]), 'exchange_ppo')
    assert plan.tolist() == [2, 0] and diag['transfers'] == 1
    assert decode_trace(mask, np.array([1, 2, 5, 1]), 'exchange_ppo')[0].tolist() == [0, 1]
    for arm in ('self_repair_ppo', 'construct_only_ppo'):
        with pytest.raises(ValueError): decode_trace(mask, np.array([1, 2, 2, 0]), arm)
    mask[0, 1] = False
    with pytest.raises(ValueError): decode_trace(mask, np.array([1, 2, 2, 0]), 'exchange_ppo')
    for trace in ([1, 2, 1, 0], [0, 2, 5, 0], [5, 0, 0, 0], [1, 1, 0, 0]):
        with pytest.raises(ValueError): decode_trace(mask, np.array(trace), 'exchange_ppo')


@pytest.mark.parametrize('arm', CORE_ARMS)
@pytest.mark.parametrize('sparse', [False, True])
def test_tensor_masks_updates_equal_independent_scalar_broker_for_every_reachable_prefix(arm, sparse):
    env = urgent(arm=arm)
    obs = env.observe()
    if sparse: obs['pair_mask'][[0, 2], 1] = False
    mask = obs['pair_mask']
    batch = tensors([obs])
    prefixes = [()]
    for slot in range(4):
        following = []
        for prefix in prefixes:
            state = decode_trace(mask, np.array([*prefix, *([0]*(4-len(prefix)))]), arm)[0]
            matching = torch.tensor(state[None])
            previous = torch.tensor([-1])
            if len(prefix) > 2 and arm in ('exchange_ppo','unrestricted_exchange_ppo','inverse_mask_ppo'):
                before = decode_trace(mask, np.array([*prefix[:-1], *([0]*(5-len(prefix)))]), arm)[0]
                token = prefix[-1]
                owner = np.flatnonzero(before == token) if 1 <= token <= 4 else []
                if len(owner): previous = torch.tensor([int(owner[0])])
            support = move_support(batch, matching, slot % 2, slot >= 2, arm, previous)[0]
            for token in range(6):
                trace = np.array([*prefix, token, *([0]*(3-slot))])
                try:
                    expected, _ = decode_trace(mask, trace, arm)
                except ValueError:
                    assert not support[token]
                    continue
                assert support[token]
                actual = apply_move(matching, torch.tensor([token]), slot % 2, slot >= 2, arm, 4)
                assert np.array_equal(actual[0].numpy(), expected)
                following.append((*prefix, token))
        prefixes = following


@pytest.mark.parametrize('arm', CORE_ARMS)
def test_exact_latent_mass_and_full_physical_support_with_duplicate_traces(arm):
    env, m = urgent(arm=arm), model(arm)
    traces, physical = [], set()
    for trace in itertools.product(range(6), repeat=4):
        try: pairs, _ = env.resolve(np.array(trace))
        except ValueError: continue
        traces.append(trace); physical.add(pairs)
    obs = env.observe()
    batch = tensors({key: np.repeat(value[None], len(traces), axis=0) for key, value in obs.items()})
    logs, _, _ = m.evaluate(batch, torch.tensor(traces))
    assert float(logs.detach().sum(1).exp().sum()) == pytest.approx(1., abs=2e-6)
    assert len(physical) == 21
    if arm != 'construct_only_ppo': assert len(traces) > len(physical)
    assert env.env.time == 0


@pytest.mark.parametrize('n', [4, 8, 16, 32])
def test_matched_core_capacity_sample_replay_and_finite_gradients(n):
    models = [model(a) for a in CORE_ARMS]
    assert len({parameter_digest(m) for m in models}) == 1
    assert {sum(p.numel() for p in m.parameters()) for m in models} == {19255}
    for m in models:
        env = urgent(n, m.algorithm)
        batch = tensors([env.observe()])
        action, log, value = m.sample(batch, torch.Generator().manual_seed(2))
        other, entropy, new_value = m.evaluate(batch, action)
        assert torch.allclose(log, other, atol=2e-5, rtol=0) and torch.equal(value, new_value)
        pairs, _ = env.resolve(action[0].numpy())
        validate_matching(env.config, env.env.state, pairs)
        (-other.sum(1).mean()-.01*entropy.mean()+(new_value/value_unit(n)).square().mean()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_existing_services_immutable_forced_slots_and_calendar_release():
    env = urgent()
    env.env.state = replace(env.env.state, remaining=(2, 0), assigned=(0, -1))
    before = env.env.state
    batch = tensors([env.observe()]);m = model('exchange_ppo')
    action, log, _ = m.sample(batch, torch.Generator().manual_seed(5))
    assert action[0, 0] == action[0, 2] == log[0, 0] == log[0, 2] == 0
    env.resolve(action[0].numpy())
    assert env.env.state == before and env.env.time == 0
    cal, _, _ = provisional_calendar(batch, torch.tensor([[0, 2]]))
    assert cal[0, 0].tolist() == pytest.approx([1., 2/12, 1/8])
    empty, _, _ = provisional_calendar(batch, torch.zeros((1, 2), dtype=torch.long))
    assert empty[0, 1].tolist() == pytest.approx([0., 0., 3/8])


def test_state_only_critic_called_once_and_inference_skips_it(monkeypatch):
    env, m = urgent(), model('exchange_ppo')
    batch = tensors([env.observe()]);calls = []
    handle = m.value_head.register_forward_hook(lambda module, inputs, output: calls.append(output))
    left = m.evaluate(batch, torch.tensor([[1, 2, 2, 0]]))[2]
    assert len(calls) == 1 and torch.equal(left[:, :1].expand_as(left), left)
    right = m.evaluate(batch, torch.tensor([[0, 0, 3, 1]]))[2]
    assert len(calls) == 2 and torch.equal(left, right)
    handle.remove()
    expected = m.sample(batch, deterministic=True)[0][0].numpy()
    monkeypatch.setattr(m.value_encoder, 'forward', lambda *args: (_ for _ in ()).throw(AssertionError('critic inference')))
    assert np.array_equal(m.select(env.observe()), expected)


@pytest.mark.parametrize('arm', CORE_ARMS)
def test_machine_permutation_through_construction_and_repair(arm):
    env, m = urgent(arm=arm), model(arm)
    obs = env.observe();order = np.array([2, 0, 3, 1])
    other = {k: v.copy() for k, v in obs.items()}
    for key in ('machines', 'edges', 'pair_mask'): other[key] = obs[key][order]
    trace = m.sample(tensors([obs]), torch.Generator().manual_seed(7))[0]
    mapped = torch.tensor([[int(np.where(order == token-1)[0][0])+1 if 1 <= token <= 4 else int(token) for token in trace[0]]])
    left, right = m.evaluate(tensors([obs]), trace), m.evaluate(tensors([other]), mapped)
    for i, (a, b) in enumerate(zip(left, right)):
        assert torch.allclose(a/(value_unit(4) if i == 2 else 1), b/(value_unit(4) if i == 2 else 1), atol=2e-6, rtol=0)


def test_independent_actor_has_no_other_intents_and_arbitrates():
    env, m = urgent(arm='independent_ppo'), model('independent_ppo')
    batch = tensors([env.observe()]);base = m.encode_static(batch, False)[0]
    left = m.logits(batch, base, torch.tensor([[1, 2]]), 0, True)
    right = m.logits(batch, base, torch.tensor([[1, 3]]), 0, True)
    assert torch.equal(left, right)
    pairs, diag = env.resolve(np.array([1, 1, 0, 0]))
    assert len(pairs) == 1 and diag['conflicts'] == diag['rejected_proposals'] == 1


def test_no_future_event_inputs_or_large_matching_catalogue(monkeypatch):
    import ht_pdm_fjsp.maintenance_heterogeneous_model as module
    monkeypatch.setattr(module, 'catalogue', lambda *args: (_ for _ in ()).throw(AssertionError('catalogue')))
    for n in (8, 16, 32):
        env = urgent(n);obs = env.observe();env.env.events = ((True,)*n,)*60
        assert all(np.array_equal(v, env.observe()[k]) for k, v in obs.items())
        env.resolve(model('exchange_ppo').select(obs));assert env.actions is None


@pytest.mark.parametrize('arm', CORE_ARMS)
def test_exact_physical_training_budget_and_joint_trace_learning(arm):
    unit = value_unit(4)
    cfg = dict(env_steps=384, vector_envs=2, rollout=192, epochs=2, minibatch=48,
        learning_rate=.0003, gradient_clip=.5, gamma=1., checkpoints=[192, 384],
        clip=.2, entropy_weight=.01, value_weight=.5, torch_threads=1,
        rng_namespace=VERSION+':engineering:', group_gradient_clipping=True,
        value_loss_normalizer=unit, behavior_log_atol=2e-5, behavior_log_rtol=0.,
        behavior_value_atol=unit*2e-5, behavior_value_rtol=1e-6)
    m = model(arm);before = parameter_digest(m)
    audit = fit(m, cfg, 160000000, lambda s, w: ResourceEnv(s, 4, arm, w, True))
    assert audit['learned_steps'] == 384 and audit['physical_ticks'] == 480
    assert audit['training_episodes'] == 8 and audit['optimizer_steps'] == 16
    assert audit['behavior_normalized_max_error'] <= 1 and parameter_digest(m) != before
