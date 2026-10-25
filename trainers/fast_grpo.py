"""Version B: Flow-GRPO-Fast on one SDE step + reward-gated teacher KL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..core.sampler import GRPOGroup, collect_grpo_group
from ..core.transition import gaussian_log_prob, sde_mean_from_velocity
from ..losses.distillation import teacher_kl_loss
from ..losses.grpo import grpo_clip_loss
from .common import (
    build_evaluator,
    build_runtime,
    clip_and_step,
    log_metrics,
    make_optimizer,
    output_dir,
    prepare_cases,
    round_slices,
    set_warmup_lr,
)


class FastGRPOTrainer:
    def __init__(self, cfg: dict[str, Any], mock_reward=None):
        if int(cfg.get("group_size") or 0) != 4:
            raise ValueError("Version B is pinned to group_size=4")
        self.cfg = cfg
        self.train_cases, self.eval_cases = prepare_cases(cfg)
        self.root = output_dir(cfg)
        self.runtime = build_runtime(cfg, self.train_cases + self.eval_cases)
        self.evaluator = build_evaluator(cfg, mock_fn=mock_reward)
        self.optimizer = make_optimizer(self.runtime, cfg)
        self.update = 0

    def _loss_group(self, group: GRPOGroup) -> tuple[torch.Tensor, dict[str, float]]:
        runtime = self.runtime
        device = runtime.device
        stats = {
            "grpo": 0.0,
            "kl": 0.0,
            "teacher_weight": float(group.teacher_weight),
            "usable_grpo": group.usable_grpo,
            "invalid_reward": group.invalid_reward,
            "teacher_anchor": group.teacher_anchor,
        }
        if group.invalid_reward or (not group.usable_grpo and group.teacher_weight <= 0):
            return None, stats
        x = group.x_s.to(device)
        v_phi = runtime.predict_guided(x, group.step, group.case["plain_prompt"], "student", grad=True)
        _, sigma, sigma_next = runtime.sigma_pair(group.step)
        std = group.std.to(device)
        mu_phi = sde_mean_from_velocity(x, v_phi, sigma, sigma_next, std=std)
        losses = []
        loss_cfg = self.cfg["loss"]
        if group.usable_grpo:
            log_new = []
            log_old = []
            for member in group.members:
                action = member.action.to(device)
                log_new.append(gaussian_log_prob(action, mu_phi, std, min_std=runtime.min_std).reshape(-1))
                log_old.append(member.log_prob_old.to(device).reshape(-1))
            grpo = grpo_clip_loss(
                torch.cat(log_new),
                torch.cat(log_old),
                group.advantages.to(device),
                clip_epsilon=float(loss_cfg.get("ppo_clip_epsilon", 0.1)),
            )
            losses.append(grpo)
            stats["grpo"] = float(grpo.detach().cpu())
        if group.teacher_weight > 0 and group.teacher_mean is not None:
            kl = teacher_kl_loss(
                group.teacher_mean.to(device),
                mu_phi,
                std,
                weight=group.teacher_weight * float(loss_cfg.get("teacher_weight", 0.05)),
                min_std=runtime.min_std,
            )
            losses.append(kl)
            stats["kl"] = float(kl.detach().cpu())
        ref_w = float(loss_cfg.get("reference_kl_weight", 0.0))
        if ref_w > 0 and losses:
            with torch.no_grad():
                v_base = runtime.predict_guided(x, group.step, group.case["plain_prompt"], "base", grad=False)
            mu_base = sde_mean_from_velocity(x, v_base, sigma, sigma_next, std=std)
            ref = teacher_kl_loss(mu_base, mu_phi, std, weight=ref_w, min_std=runtime.min_std)
            losses.append(ref)
            stats["ref"] = float(ref.detach().cpu())
        if not losses:
            return None, stats
        return torch.stack(losses).sum(), stats

    def fit(self) -> Path:
        budget = self.cfg["budget"]
        slices = round_slices(self.train_cases, int(budget["n_rounds"]), int(budget["groups_per_round"]))
        epochs = max(1, int(self.cfg["optimizer"].get("ppo_epochs_per_rollout", 1)))
        for round_id, batch in enumerate(slices):
            self.runtime.snapshot_old()
            groups: list[GRPOGroup] = []
            for case in batch:
                group = collect_grpo_group(self.runtime, self.evaluator, case, self.cfg, self.root)
                groups.append(group)
                log_metrics(
                    self.root,
                    {
                        "round": round_id,
                        "event": "collect",
                        "case": case.id,
                        "target": case.target,
                        "rewards": [m.reward for m in group.members],
                        "counts": [m.count for m in group.members],
                        "reliable": [m.reliable for m in group.members],
                        "reasons": [m.reason for m in group.members],
                        "attempts": [m.attempts for m in group.members],
                        "usable_grpo": group.usable_grpo,
                        "invalid_reward": group.invalid_reward,
                        "teacher_weight": group.teacher_weight,
                        "teacher_count": group.teacher_count,
                        "teacher_anchor": group.teacher_anchor,
                        "skipped_teacher": group.skipped_teacher,
                        **group.timing,
                    },
                )
            for group in groups:
                for _ in range(epochs):
                    self.optimizer.zero_grad(set_to_none=True)
                    loss, stats = self._loss_group(group)
                    if loss is None:
                        log_metrics(
                            self.root,
                            {
                                "round": round_id,
                                "event": "skip_update",
                                "case": group.case["id"],
                                **stats,
                            },
                        )
                        continue
                    loss.backward()
                    lr = set_warmup_lr(self.optimizer, self.update, self.cfg)
                    grad_norm = clip_and_step(self.runtime, self.optimizer, self.cfg)
                    self.update += 1
                    log_metrics(
                        self.root,
                        {
                            "round": round_id,
                            "update": self.update,
                            "lr": lr,
                            "grad_norm": grad_norm,
                            "case": group.case["id"],
                            **stats,
                        },
                    )
            self.runtime.lora.save(self.root / "checkpoints" / f"lora_round{round_id}.pt")
        self.runtime.lora.save(self.root / "checkpoints" / "lora_final.pt")
        return self.root
