"""Version B: Flow-GRPO-Fast on one SDE step + reward-gated teacher KL."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from ..core.sampler import GRPOGroup, collect_grpo_group
from ..core.transition import gaussian_log_prob, sde_mean_from_velocity
from ..losses.distillation import teacher_kl_loss, weighted_velocity_mse
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


def backward_multistep_distillation(runtime, group: GRPOGroup, cfg: dict[str, Any], stats: dict[str, Any]) -> None:
    """Backpropagate later teacher-trajectory steps one at a time.

    The branch-step loss is part of _loss_group; this helper handles only later
    steps so the branch forward is reused and no step graph is retained.
    """
    mode = str(cfg.get("loss", {}).get("distillation", "gaussian_mean_kl"))
    if mode != "multi_step_velocity_mse" or group.teacher_weight <= 0:
        return
    trace = [r for r in group.teacher_trace if int(r["step"]) != group.step]
    if not trace:
        return
    device = runtime.device
    distill_cfg = cfg.get("distillation", {})
    scale = (group.teacher_weight * float(distill_cfg.get("weight", 0.05))
             / max(len(group.teacher_trace), 1))
    for record in trace:
        x_t = record["latents"].to(device)
        step_t = int(record["step"])
        v_t = runtime.predict_guided(
            x_t, step_t, group.case["plain_prompt"], "student", grad=True
        )
        step_loss = torch.nn.functional.mse_loss(
            v_t.float(), record["velocity"].to(device).float().detach()
        ) * scale
        step_loss.backward()
        stats["distill"] += float(step_loss.detach().cpu())



class FastGRPOTrainer:
    def __init__(self, cfg: dict[str, Any], mock_reward=None):
        if int(cfg.get("group_size") or 0) != 4:
            raise ValueError("Version B is pinned to group_size=4")
        self.cfg = cfg
        distill_mode = str(cfg.get("loss", {}).get("distillation", "gaussian_mean_kl"))
        if distill_mode not in {"gaussian_mean_kl", "multi_step_velocity_mse"}:
            raise ValueError(f"Unsupported Version B loss.distillation={distill_mode!r}")
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
            "distill": 0.0,
            "distill_steps": 0,
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
        distill_mode = str(loss_cfg.get("distillation", "gaussian_mean_kl"))
        if distill_mode == "multi_step_velocity_mse":
            if group.teacher_weight > 0 and group.teacher_trace:
                # Reuse branch velocity computed above; subsequent steps are
                # backpropagated sequentially by backward_multistep_distillation.
                branch_target = next(
                    (r["velocity"] for r in group.teacher_trace if int(r["step"]) == group.step), None
                )
                if branch_target is not None:
                    dloss = weighted_velocity_mse(
                        v_phi, branch_target.to(device),
                        torch.tensor([group.teacher_weight], device=device),
                    ) * (float(self.cfg.get("distillation", {}).get("weight", 0.05))
                         / max(len(group.teacher_trace), 1))
                    losses.append(dloss)
                    stats["distill"] = float(dloss.detach().cpu())
                    stats["distill_steps"] = len(group.teacher_trace)
        elif distill_mode == "gaussian_mean_kl":
            # Backward-compatible behavior for legacy Version B configs.
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
        runtime = self.runtime
        device = runtime.device
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
                    update_t0 = time.perf_counter()
                    if runtime_cuda := getattr(self.runtime, "torch", None):
                        if runtime_cuda.cuda.is_available():
                            runtime_cuda.cuda.reset_peak_memory_stats(self.runtime.device)
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
                    # Low-memory multi-step ODE distillation: one Student forward
                    # and backward at a time; never retain all step graphs together.
                    backward_multistep_distillation(runtime, group, self.cfg, stats)
                    lr = set_warmup_lr(self.optimizer, self.update, self.cfg)
                    grad_norm = clip_and_step(self.runtime, self.optimizer, self.cfg)
                    self.update += 1
                    stats["update_sec"] = time.perf_counter() - update_t0
                    if runtime_cuda is not None and runtime_cuda.cuda.is_available():
                        stats["gpu_update_peak_allocated_mb"] = float(runtime_cuda.cuda.max_memory_allocated(self.runtime.device) / (1024 ** 2))
                        stats["gpu_update_peak_reserved_mb"] = float(runtime_cuda.cuda.max_memory_reserved(self.runtime.device) / (1024 ** 2))
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
