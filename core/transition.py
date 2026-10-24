"""ODE Euler and plan DDIM-style SDE kernels. Sample and log-prob share the mean/std."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def _broadcast_std(std: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    while std.ndim < sample.ndim:
        std = std.unsqueeze(-1)
    return std


def ode_mean(sample: torch.Tensor, velocity: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Flow-matching Euler: x_{t+dt} = x_t + dt * v. dt = sigma_next - sigma < 0."""
    return sample.float() + dt.float() * velocity.float()


def plan_sde_std(sigma: torch.Tensor, sigma_next: torch.Tensor, noise_coeff: float = 0.3, next_cap: float = 0.95) -> torch.Tensor:
    """σ = min(c · t √(1-t), cap · t_next). t is the current flow sigma in [0, 1]."""
    t = sigma.float().clamp(0.0, 1.0)
    t_next = sigma_next.float().abs()
    interior = (1.0 - t).clamp(min=0.0).sqrt()
    unclamped = float(noise_coeff) * t * interior
    capped = float(next_cap) * t_next
    return torch.minimum(unclamped, capped)


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
) -> Transition:
    dt = sigma_next.float() - sigma.float()
    while dt.ndim < sample.ndim:
        dt = dt.unsqueeze(-1)
    mean = ode_mean(sample, velocity, dt)
    if not use_sde:
        next_latents = mean.to(dtype=sample.dtype)
        zero = sigma.float() * 0.0
        return Transition(next_latents, mean, zero, None, None, velocity, dt)

    std = plan_sde_std(sigma, sigma_next, noise_coeff=noise_coeff, next_cap=next_cap)
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
) -> torch.Tensor:
    dt = sigma_next.float() - sigma.float()
    while dt.ndim < sample.ndim:
        dt = dt.unsqueeze(-1)
    return ode_mean(sample, velocity, dt)
