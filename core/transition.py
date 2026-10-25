"""ODE Euler and coefficient-preserving DDIM-style SDE kernels.

Sample and log-prob share the same mean/std. ODE tails stay Euler.
SDE uses the CPS/Stepwise mean with a conservative Stepwise noise schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def _broadcast_std(std: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    while std.ndim < sample.ndim:
        std = std.unsqueeze(-1)
    return std


def _broadcast_time(t: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    t = t.float()
    while t.ndim < sample.ndim:
        t = t.unsqueeze(-1)
    return t


def ode_mean(sample: torch.Tensor, velocity: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Flow-matching Euler: x_{t+dt} = x_t + dt * v. dt = sigma_next - sigma < 0."""
    return sample.float() + dt.float() * velocity.float()


def ddim_sde_std(
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    eta: float = 0.3,
    next_cap: float = 0.95,
) -> torch.Tensor:
    """s = min(η · t' · √(1-t), cap · t'). Guarantees s ≤ t' so the DDIM sqrt is real."""
    t = sigma.float().clamp(0.0, 1.0)
    t_next = sigma_next.float().abs()
    interior = (1.0 - t).clamp(min=0.0).sqrt()
    unclamped = float(eta) * t_next * interior
    capped = float(next_cap) * t_next
    return torch.minimum(unclamped, capped)


def plan_sde_std(
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    noise_coeff: float = 0.3,
    next_cap: float = 0.95,
    eta: float | None = None,
) -> torch.Tensor:
    """Backward-compatible alias. Prefer ddim_sde_std; eta overrides noise_coeff."""
    return ddim_sde_std(sigma, sigma_next, eta=float(eta if eta is not None else noise_coeff), next_cap=next_cap)


def ddim_mean(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    std: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Coefficient-preserving DDIM mean. std=0 recovers Euler exactly.

    x0 = x - t v,  ε = x + (1-t) v
    μ  = (1-t') x0 + sqrt(t'^2 - s^2) ε
    """
    t = _broadcast_time(sigma, sample)
    t_next = _broadcast_time(sigma_next, sample)
    x = sample.float()
    v = velocity.float()
    x0 = x - t * v
    eps = x + (1.0 - t) * v
    if std is None:
        s = t.new_zeros(())
    elif torch.is_tensor(std):
        s = _broadcast_std(std.float(), sample)
    else:
        s = x.new_full((), float(std))
        s = _broadcast_std(s, sample)
    old_coeff = (t_next.square() - s.square()).clamp(min=0.0).sqrt()
    return (1.0 - t_next) * x0 + old_coeff * eps


def gaussian_log_prob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, min_std: float = 1e-6) -> torch.Tensor:
    """Mean log-prob over non-batch dims. Same reduction as Flow-GRPO."""
    std = _broadcast_std(std.float().clamp(min=min_std), sample)
    diff = sample.float() - mean.float()
    log_prob = -(diff ** 2) / (2.0 * std ** 2) - torch.log(std) - 0.5 * math.log(2.0 * math.pi)
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def gaussian_mean_kl(mu_teacher: torch.Tensor, mu_student: torch.Tensor, std: torch.Tensor, min_std: float = 1e-6) -> torch.Tensor:
    """Per-dimension average KL of two equal-variance Gaussians: ||μT-μ||² / (2 d σ²)."""
    std = std.float().clamp(min=min_std)
    diff = mu_teacher.float() - mu_student.float()
    mse = diff.pow(2).mean()
    return mse / (2.0 * std.mean() ** 2)


@dataclass
class Transition:
    next_latents: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    noise: torch.Tensor | None
    log_prob: torch.Tensor | None
    velocity: torch.Tensor
    dt: torch.Tensor


def apply_transition(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    *,
    use_sde: bool,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    noise_coeff: float = 0.3,
    next_cap: float = 0.95,
    min_std: float = 1e-6,
    eta: float | None = None,
) -> Transition:
    dt = sigma_next.float() - sigma.float()
    while dt.ndim < sample.ndim:
        dt = dt.unsqueeze(-1)
    if not use_sde:
        mean = ode_mean(sample, velocity, dt)
        next_latents = mean.to(dtype=sample.dtype)
        zero = sigma.float() * 0.0
        return Transition(next_latents, mean, zero, None, None, velocity, dt)

    eta_used = float(eta if eta is not None else noise_coeff)
    std = ddim_sde_std(sigma, sigma_next, eta=eta_used, next_cap=next_cap)
    mean = ddim_mean(sample, velocity, sigma, sigma_next, std=std)
    std_b = _broadcast_std(std, sample)
    if noise is None:
        noise = torch.randn(sample.shape, device=sample.device, dtype=torch.float32, generator=generator)
    else:
        noise = noise.float()
    next_latents = (mean + std_b * noise).to(dtype=sample.dtype)
    log_prob = gaussian_log_prob(next_latents, mean, std, min_std=min_std)
    return Transition(next_latents, mean, std, noise, log_prob, velocity, dt)


def sde_mean_from_velocity(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    std: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """SDE mean used at train time. Pass the rollout std so μ_φ matches sampling."""
    if std is None:
        dt = sigma_next.float() - sigma.float()
        while dt.ndim < sample.ndim:
            dt = dt.unsqueeze(-1)
        return ode_mean(sample, velocity, dt)
    if torch.is_tensor(std) and float(std.detach().float().reshape(-1)[0].abs()) == 0.0:
        dt = sigma_next.float() - sigma.float()
        while dt.ndim < sample.ndim:
            dt = dt.unsqueeze(-1)
        return ode_mean(sample, velocity, dt)
    return ddim_mean(sample, velocity, sigma, sigma_next, std=std)
