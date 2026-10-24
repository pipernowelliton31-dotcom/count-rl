"""Group-relative PPO-clip GRPO on a single SDE transition."""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, bool]:
    """Return standardized advantages and whether the group has usable variance."""
    rewards = rewards.float().reshape(-1)
    if rewards.numel() < 2:
        return torch.zeros_like(rewards), False
    std = rewards.std(unbiased=False)
    if float(std) < eps:
        return torch.zeros_like(rewards), False
    adv = (rewards - rewards.mean()) / (std + eps)
    return adv, True


def importance_ratio(log_prob_new: torch.Tensor, log_prob_old: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_prob_new.float() - log_prob_old.float().detach())


def grpo_clip_loss(
    log_prob_new: torch.Tensor,
    log_prob_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.1,
) -> torch.Tensor:
    """L = -mean min(ρ A, clip(ρ, 1-ε, 1+ε) A). Teacher actions must not be in this batch."""
    ratio = importance_ratio(log_prob_new, log_prob_old)
    advantages = advantages.float().reshape(-1)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return -torch.minimum(unclipped, clipped).mean()
