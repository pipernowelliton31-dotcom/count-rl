"""Version B: Flow-GRPO-Fast on one SDE step + reward-gated teacher KL."""

from __future__ import annotations

from pathlib import Path
import os
import signal
import uuid
import time
from typing import Any

import torch

from ..core.distributed import (
    assert_launch_budget,
    average_gradients,
    barrier,
    broadcast_object,
    init_distributed,
    max_int,
    shard,
)
from ..core.io_utils import atomic_torch_save
from ..core.sampler import GRPOGroup, collect_grpo_group
from .rollout_rendezvous import before_collective
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
        gate_mode = str(cfg.get("loss", {}).get("reward_gate", "positive_reward_gain"))
        if gate_mode not in ("strict_exact", "positive_reward_gain"):
            raise ValueError(f"Unsupported loss.reward_gate={gate_mode!r}")
        distill_mode = str(cfg.get("loss", {}).get("distillation", "gaussian_mean_kl"))
        if distill_mode not in {"gaussian_mean_kl", "multi_step_velocity_mse"}:
            raise ValueError(f"Unsupported Version B loss.distillation={distill_mode!r}")
        self.dist = init_distributed()
        assert_launch_budget(cfg)
        self.train_cases, self.eval_cases = prepare_cases(cfg)
        self.root = self._run_dir()
        # A unique per-run token shared before expensive trajectory collection.
        # Prevent the 2026-11-21 crash at before_collective after ~92 rollouts.
        current_token = os.environ.get("COUNT_RL_FILE_RENDEZVOUS_ID")
        proposed = (current_token or f"auto_{uuid.uuid4().hex}") if self.dist.is_main else None
        token = broadcast_object(proposed)
        if current_token and current_token != token:
            raise RuntimeError("Inconsistent rendezvous token across ranks")
        if not token or not token.replace("_", "").replace("-", "").isalnum():
            raise RuntimeError("Invalid rendezvous token before model initialization")
        os.environ["COUNT_RL_FILE_RENDEZVOUS_ID"] = token
        self.runtime = build_runtime(cfg, self.train_cases + self.eval_cases)
        self.evaluator = build_evaluator(cfg, mock_fn=mock_reward)
        self.optimizer = make_optimizer(self.runtime, cfg)
        self.update = 0
        self.start_round = 0
        self._stop = False
        self._install_stop_handlers()
        self._maybe_resume()
        # Detect divergent recovered models before the first generation.
        self.runtime.lora.assert_rank_consistency()

    def _install_stop_handlers(self) -> None:
        def _request_stop(signum, _frame) -> None:
            self._stop = True
            print(f"rank {self.dist.rank} received signal {signum}; will save at the next round boundary", flush=True)

        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGUSR1, _request_stop)

    def _run_dir(self) -> Path:
        resume = os.environ.get("RESUME_FROM") or self.cfg.get("resume_from")
        experiment = self.cfg.get("experiment", {})
        if resume:
            source_root = Path(resume).resolve().parent.parent
            if not source_root.is_dir():
                raise FileNotFoundError(f"Resume run directory does not exist: {source_root}")
            # A legacy RESUME_FROM normally resumes *in place*, so saving a
            # compact train_state would overwrite the original 12 GiB run.
            # External runs require an explicit new-output opt-in.
            workspace = Path(__file__).resolve().parents[2]
            if not experiment.get("resume_into_new_run", False):
                if source_root != workspace and workspace not in source_root.parents:
                    raise RuntimeError(
                        "Refusing to overwrite a checkpoint outside the isolated FP32 workspace. "
                        "Set experiment.resume_into_new_run: true to load an external checkpoint "
                        "and write all new outputs to a separate run directory."
                    )
                return source_root
        if self.dist.is_main:
            root = output_dir(self.cfg)
            payload = str(root)
        else:
            payload = None
        payload = broadcast_object(payload)
        barrier()
        return Path(payload)

    def _state_path(self) -> Path:
        return self.root / "checkpoints" / "train_state.pt"

    def _save_resume(self, next_round: int) -> None:
        if self.dist.is_main:
            self._state_path().parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(
                self._state_path(),
                {
                    "lora_state": self.runtime.lora.state_dict(),
                    "lora_state_format": "lora_compact_v1",
                    "optimizer": self.optimizer.state_dict(),
                    "lora_master_dtype": "float32",
                    "adam_moment_dtype": "float32",
                    "update": self.update,
                    "next_round": next_round,
                    "world_size": self.dist.world,
                    "rollouts_reusable": False,
                },
            )
        barrier()

    def _maybe_resume(self) -> None:
        resume = os.environ.get("RESUME_FROM") or self.cfg.get("resume_from")
        if not resume:
            return
        payload = torch.load(Path(resume), map_location="cpu", weights_only=False, mmap=True)
        # A completed 2-GPU round can resume as 4 GPU ranks without changing
        # the 4-group global optimizer batch. Never migrate in-progress rollouts.
        source_world = int(payload["world_size"])
        if source_world != self.dist.world:
            migration = self.cfg.get("migration", {})
            n_rounds = int(self.cfg["budget"]["n_rounds"])
            n_groups = int(self.cfg["budget"]["groups_per_round"])
            valid_boundary = (
                payload.get("rollouts_reusable") is False
                and 1 <= int(payload.get("next_round", -1)) < n_rounds
                and int(payload.get("update", -1)) >= 0
            )
            allowed = (
                source_world == 2
                and self.dist.world == 4
                and migration.get("allow_world_size_2_to_4") is True
                and os.environ.get("ALLOW_2_TO_4_GPU_MIGRATION") == "1"
                and int(self.cfg.get("backup", {}).get("groups_per_physical_rank_per_update", 0)) == 1
                and n_groups % 4 == 0
                and valid_boundary
            )
            if not allowed:
                raise SystemExit(
                    f"Checkpoint world_size={source_world} differs from WORLD_SIZE={self.dist.world}. "
                    "Only explicitly approved 2->4 migration at a completed round boundary is supported."
                )
            if self.dist.is_main:
                print(
                    f"[WORLD_SIZE_MIGRATION] 2->4 rank migration; "
                    f"resume round={payload['next_round']} update={payload['update']}; "
                    "global effective batch remains 4 groups, old rollouts discarded.",
                    flush=True,
                )
        # PyTorch casts old BF16 optimizer moments to the FP32 Parameter dtype
        # when loading a legacy checkpoint. Check this instead of assuming it.
        self.runtime.lora.load_state_dict(payload["lora_state"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.runtime.lora.assert_master_fp32()
        for param in self.runtime.lora.trainable_parameters():
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = self.optimizer.state[param].get(key)
                if value is not None and value.dtype != torch.float32:
                    raise RuntimeError(
                        f"Optimizer {key} was loaded as {value.dtype}, expected float32"
                    )
        self.update = int(payload["update"])
        self.start_round = int(payload["next_round"])
        self.runtime.snapshot_old()
        log_metrics(self.root, {
            "event": "resume",
            "next_round": self.start_round,
            "update": self.update,
            "rollouts_reusable": False,
        })

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
            grpo_scale = float(loss_cfg.get("grpo_scale", 1.0))
            if not 0 <= grpo_scale <= 1e6:
                raise ValueError(f"Invalid loss.grpo_scale={grpo_scale}")
            losses.append(grpo * grpo_scale)
            stats["grpo"] = float(grpo.detach().cpu())
            stats["grpo_scaled"] = float((grpo * grpo_scale).detach().cpu())
            stats["grpo_scale"] = grpo_scale
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

    def _sync_group_update(self, group_batch: list[GRPOGroup], round_id: int) -> None:
        """Normalize gradients over four effective groups per optimizer step.

        On two ranks each rank accumulates two groups; on four ranks it uses
        one group. The denominator counts groups actually contributing loss.
        """
        runtime = self.runtime
        params = runtime.lora.trainable_parameters()
        self.optimizer.zero_grad(set_to_none=True)
        if runtime.torch.cuda.is_available():
            runtime.torch.cuda.reset_peak_memory_stats(runtime.device)
        update_t0 = time.perf_counter()
        participants = 0
        records: list[dict[str, Any]] = []
        for group in group_batch:
            loss, stats = self._loss_group(group)
            stats["case"] = group.case["id"]
            if loss is not None:
                loss.backward()
                backward_multistep_distillation(runtime, group, self.cfg, stats)
                participants += 1
                stats["participated"] = True
            else:
                stats["participated"] = False
            records.append(stats)
        n_part = average_gradients(params, participants)
        if n_part <= 0:
            for stats in records:
                log_metrics(self.root, {"round": round_id, "event": "skip_update", **stats})
            return
        lr = set_warmup_lr(self.optimizer, self.update, self.cfg)
        # Record FP32 master gradients and *actual AdamW parameter movement*.
        # CPU copies avoid an additional ~122 MiB peak GPU allocation.
        before_cpu = [p.detach().cpu().clone() for p in params] if self.cfg.get("record_step_delta", False) else []
        if before_cpu:
            sums = [torch.zeros((), device=runtime.device, dtype=torch.float32),
                    torch.zeros((), device=runtime.device, dtype=torch.float32)]
            for idx,p in enumerate(params):
                if p.grad is not None:
                    if p.grad.dtype != torch.float32 or p.dtype != torch.float32:
                        raise RuntimeError("Expected FP32 LoRA master and FP32 gradient")
                    sums[idx % 2] += p.grad.detach().float().square().sum()
            grad_A,grad_B = [float(v.sqrt().item()) for v in sums]
        grad_norm = clip_and_step(runtime, self.optimizer, self.cfg)
        step_delta = {}
        if before_cpu:
            delta_sq = [0.0, 0.0]
            changed = [0, 0]
            bf16_changed = [0, 0]
            totals = [0, 0]
            for idx,(param,old) in enumerate(zip(params,before_cpu)):
                kind = idx % 2
                new = param.detach().cpu()
                delta = new - old
                delta_sq[kind] += float(delta.double().square().sum())
                changed[kind] += int((new != old).sum())
                bf16_changed[kind] += int((new.to(torch.bfloat16) != old.to(torch.bfloat16)).sum())
                totals[kind] += old.numel()
            step_delta = {
                "grad_A_norm":grad_A,
                "grad_B_norm":grad_B,
                "delta_A_l2":delta_sq[0]**0.5,
                "delta_B_l2":delta_sq[1]**0.5,
                "fp32_changed_A_pct":100*changed[0]/totals[0],
                "fp32_changed_B_pct":100*changed[1]/totals[1],
                "bf16_shadow_changed_A_pct":100*bf16_changed[0]/totals[0],
                "bf16_shadow_changed_B_pct":100*bf16_changed[1]/totals[1],
            }
            del before_cpu
        self.update += 1
        duration = time.perf_counter() - update_t0
        stats = {
            "cases": [r["case"] for r in records],
            "per_group_stats": records,
            "update_sec": duration,
            "participating_groups": n_part,
            "participating_local_groups": participants,
            "world_size": self.dist.world,
            "virtual_groups_per_rank": int(self.cfg.get("backup", {}).get("groups_per_physical_rank_per_update", 1)),
            "grpo": sum(float(r.get("grpo", 0)) for r in records) / max(len(records), 1),
            "grpo_scaled": sum(float(r.get("grpo_scaled", 0)) for r in records) / max(len(records), 1),
            "kl": sum(float(r.get("kl", 0)) for r in records) / max(len(records), 1),
            "distill": sum(float(r.get("distill", 0)) for r in records) / max(len(records), 1),
            "teacher_weight": sum(float(r.get("teacher_weight", 0)) for r in records) / max(len(records), 1),
        }
        if runtime.torch.cuda.is_available():
            stats["gpu_update_peak_allocated_mb"] = float(
                runtime.torch.cuda.max_memory_allocated(runtime.device) / (1024 ** 2)
            )
            stats["gpu_update_peak_reserved_mb"] = float(
                runtime.torch.cuda.max_memory_reserved(runtime.device) / (1024 ** 2)
            )
        if records:
            log_metrics(self.root, {
                "round": round_id, "update": self.update,
                "lr": lr, "grad_norm": grad_norm, **step_delta, **stats,
            })

    def fit(self) -> Path:
        budget = self.cfg["budget"]
        slices = round_slices(self.train_cases, int(budget["n_rounds"]), int(budget["groups_per_round"]))
        epochs = max(1, int(self.cfg["optimizer"].get("ppo_epochs_per_rollout", 1)))
        stepped = False
        for round_id, batch in enumerate(slices):
            if round_id < self.start_round:
                continue
            if self._stop:
                self._save_resume(round_id)
                log_metrics(self.root, {"event": "stopped", "next_round": round_id})
                break
            self.runtime.snapshot_old()
            local_cases = shard(batch, self.dist.rank, self.dist.world)
            groups: list[GRPOGroup] = []
            for case in local_cases:
                group = collect_grpo_group(self.runtime, self.evaluator, case, self.cfg, self.root)
                groups.append(group)
                log_metrics(self.root, {
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
                    "world_size": self.dist.world,
                    **group.timing,
                })
            before_collective(self.root, round_id, self.dist.rank, self.dist.world, len(groups))
            barrier()
            slots = max_int(len(groups))
            accum = int(self.cfg.get("backup", {}).get("groups_per_physical_rank_per_update", 1))
            if accum < 1 or (self.dist.world * accum != 4):
                raise RuntimeError(f"Effective group batch must be 4: world={self.dist.world}, accum={accum}")
            for _ in range(epochs):
                for slot in range(0, slots, accum):
                    group_batch = groups[slot: slot + accum]
                    before = self.update
                    self._sync_group_update(group_batch, round_id)
                    stepped = stepped or self.update > before
            if self.dist.is_main:
                self.runtime.lora.save(self.root / "checkpoints" / f"lora_round{round_id}.pt")
            self._save_resume(round_id + 1)
            barrier()
        if self.dist.is_main:
            self.runtime.lora.save(self.root / "checkpoints" / "lora_final.pt")
        barrier()
        smoke = os.environ.get("COUNT_RL_EULER_SMOKE", "")
        if self.dist.is_main and smoke and stepped:
            label = "EULER_4GPU_SMOKE_TEST" if smoke == "4gpu" else "EULER_1GPU_SMOKE_TEST"
            print(f"{label} = PASS", flush=True)
        elif self.dist.is_main and smoke:
            print(f"EULER smoke requested ({smoke}) but no optimizer step ran", flush=True)
        return self.root
