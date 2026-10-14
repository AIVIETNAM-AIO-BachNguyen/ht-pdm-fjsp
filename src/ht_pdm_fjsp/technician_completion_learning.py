"""Matched prefix/state baselines and component/joint PPO surrogates."""
import torch


def prepare_advantage(returns, old_values, mode):
    if returns.ndim != 1 or old_values.ndim != 2 or len(returns) != len(old_values):
        raise ValueError('allocation return/value shapes')
    if mode not in ('prefix', 'state', 'joint'):
        raise ValueError('allocation credit mode')
    returns, old_values = returns.detach(), old_values.detach()
    raw = returns[:, None] - old_values if mode == 'prefix' else returns - old_values[:, 0]
    advantage = (raw - raw.mean()) / (raw.std(unbiased=False) + 1e-8)
    if mode == 'state':
        advantage = advantage[:, None].expand_as(old_values)
    return advantage, dict(allocation_advantage_std=float(raw.std(unbiased=False)),
                          prefix_baseline_spread=float(old_values.std(1, unbiased=False).mean()))


def allocation_objective(current_log, old_log, advantage, mode, clip):
    if current_log.shape != old_log.shape or current_log.ndim != 2:
        raise ValueError('allocation log shapes')
    if mode == 'joint':
        ratio = (current_log.sum(1) - old_log.sum(1)).exp()
    elif mode in ('prefix', 'state'):
        ratio = (current_log - old_log).exp()
    else:
        raise ValueError('allocation surrogate mode')
    if ratio.shape != advantage.shape:
        raise ValueError('allocation advantage/ratio shapes')
    return torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()


def value_loss(current_values, returns, normalizer):
    if current_values.ndim != 2 or returns.ndim != 1 or len(current_values) != len(returns):
        raise ValueError('allocation critic target shapes')
    return torch.nn.functional.mse_loss(current_values / normalizer,
        returns[:, None].expand_as(current_values) / normalizer)
