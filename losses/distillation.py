"""Reward-gated velocity MSE and optional behavior-keep MSE. FP32 loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.transition import gaussian_mean_kl


def weighted_velocity_mse(
    student_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean over latent elements, then weight-normalize. Zero-weight rows do not dilute."""
    student = student_velocity.float()
    teacher = teacher_velocity.float().detach()
    per = F.mse_loss(student, teacher, reduction="none")
    per = per.reshape(per.shape[0], -1).mean(dim=1)
    w = weight.float().reshape(-1)
    return (w * per).sum() / (w.sum() + eps)


def behavior_keep_loss(student_velocity: torch.Tensor, snapshot_velocity: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_velocity.float(), snapshot_velocity.float().detach())


def teacher_kl_loss(
    mu_teacher: torch.Tensor,
    mu_student: torch.Tensor,
    std: torch.Tensor,
    weight: torch.Tensor | float,
    min_std: float = 1e-6,
) -> torch.Tensor:
    kl = gaussian_mean_kl(mu_teacher.detach(), mu_student, std, min_std=min_std)
    if torch.is_tensor(weight):
        w = weight.float().reshape(-1).mean()
    else:
        w = float(weight)
    return w * kl
