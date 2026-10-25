"""Version A: reward-gated velocity distillation. No SDE, no GRPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..core.sampler import DistillRecord, collect_distill_round
from ..losses.distillation import behavior_keep_loss, weighted_velocity_mse
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


class DistillationTrainer:
    def __init__(self, cfg: dict[str, Any], mock_reward=None):
        self.cfg = cfg
        self.train_cases, self.eval_cases = prepare_cases(cfg)
        self.root = output_dir(cfg)
        self.runtime = build_runtime(cfg, self.train_cases + self.eval_cases)
        self.evaluator = build_evaluator(cfg, mock_fn=mock_reward)
        self.optimizer = make_optimizer(self.runtime, cfg)
        self.update = 0

    def _forward_record(self, record: DistillRecord) -> tuple[torch.Tensor, dict[str, float]]:
        runtime = self.runtime
        device = runtime.device
        x = record.x_s.to(device)
        v_student = runtime.predict_guided(x, record.step, record.case["plain_prompt"], "student", grad=True)
        stats = {"weight": float(record.weight), "distill": 0.0, "keep": 0.0}
        losses = []
        if record.weight > 0 and record.v_teacher is not None:
            weight = torch.tensor([record.weight], device=device)
            distill = weighted_velocity_mse(v_student.unsqueeze(0), record.v_teacher.to(device).unsqueeze(0), weight)
            losses.append(distill)
            stats["distill"] = float(distill.detach().cpu())
        keep_w = float(self.cfg["loss"].get("behavior_keep_weight", 0.1))
        if record.weight > 0 and keep_w > 0 and record.keep:
            keep_losses = []
            for item in record.keep:
                kx = item["latents"].to(device)
                kv = runtime.predict_guided(kx, int(item["step"]), record.case["plain_prompt"], "student", grad=True)
                keep_losses.append(behavior_keep_loss(kv, item["velocity"].to(device)))
            keep = torch.stack(keep_losses).mean()
            losses.append(keep_w * keep)
            stats["keep"] = float(keep.detach().cpu())
        if not losses:
            return None, stats
        return torch.stack([loss if loss.ndim == 0 else loss.mean() for loss in losses]).sum(), stats

    def _optimize_cache(self, records: list[DistillRecord], round_id: int) -> None:
        opt_cfg = self.cfg["optimizer"]
        accum = max(1, int(opt_cfg.get("grad_accum", 8)))
        max_passes = int(opt_cfg.get("max_cache_passes", 2))
        max_updates = int(self.cfg["budget"].get("max_updates_per_round", 32))
        usable = [r for r in records if r.weight > 0 and r.v_teacher is not None]
        if not usable:
            log_metrics(self.root, {"round": round_id, "event": "no_supervision"})
            return
        self.optimizer.zero_grad(set_to_none=True)
        updates_this_round = 0
        for pass_id in range(max_passes):
            pending = 0
            for record in usable:
                if updates_this_round >= max_updates:
                    return
                loss, stats = self._forward_record(record)
                if loss is None:
                    log_metrics(self.root, {"round": round_id, "event": "skip_update", "case": record.case["id"], **stats})
                    continue
                (loss / accum).backward()
                pending += 1
                if pending < accum:
                    continue
                lr = set_warmup_lr(self.optimizer, self.update, self.cfg)
                grad_norm = clip_and_step(self.runtime, self.optimizer, self.cfg)
                self.update += 1
                updates_this_round += 1
                pending = 0
                log_metrics(
                    self.root,
                    {
                        "round": round_id,
                        "pass": pass_id,
                        "update": self.update,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        **stats,
                        "case": record.case["id"],
                        "count_plain": record.count_plain,
                        "count_teacher": record.count_teacher,
                        "gate": record.weight,
                    },
                )
            if pending:
                lr = set_warmup_lr(self.optimizer, self.update, self.cfg)
                grad_norm = clip_and_step(self.runtime, self.optimizer, self.cfg)
                self.update += 1
                updates_this_round += 1
                log_metrics(self.root, {"round": round_id, "pass": pass_id, "update": self.update, "lr": lr, "grad_norm": grad_norm, "flush": True})

    def fit(self) -> Path:
        budget = self.cfg["budget"]
        slices = round_slices(self.train_cases, int(budget["n_rounds"]), int(budget["cases_per_round"]))
        for round_id, batch in enumerate(slices):
            self.runtime.snapshot_old()
            records = collect_distill_round(self.runtime, self.evaluator, batch, self.cfg, self.root)
            accepted = 0
            for record in records:
                accepted += int(record.weight > 0)
                log_metrics(
                    self.root,
                    {
                        "round": round_id,
                        "event": "collect",
                        "case": record.case["id"],
                        "target": record.case["target"],
                        "count_plain": record.count_plain,
                        "count_teacher": record.count_teacher,
                        "weight": record.weight,
                        "skipped_teacher": record.skipped_teacher,
                        "reason_plain": record.reason_plain,
                        "reason_teacher": record.reason_teacher,
                        "reliable_plain": record.reliable_plain,
                        "attempts_plain": record.attempts_plain,
                        "attempts_teacher": record.attempts_teacher,
                        "scores_plain": (record.extra_plain or {}).get("scores"),
                        "scores_teacher": (record.extra_teacher or {}).get("scores"),
                        **record.timing,
                    },
                )
            log_metrics(self.root, {"round": round_id, "event": "gate_rate", "accepted": accepted, "n": len(records)})
            self._optimize_cache(records, round_id)
            ckpt = self.root / "checkpoints" / f"lora_round{round_id}.pt"
            self.runtime.lora.save(ckpt)
        self.runtime.lora.save(self.root / "checkpoints" / "lora_final.pt")
        return self.root
