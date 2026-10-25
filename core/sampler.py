"""Collect student / teacher trajectories. Generation has no gradient."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .cases import TrainingCase
from .io_utils import save_png
from .reward import RewardEvaluator
from .runtime import Flux2Runtime


@dataclass
class DistillRecord:
    case: dict[str, Any]
    step: int
    x_s: torch.Tensor
    v_old: torch.Tensor
    v_teacher: torch.Tensor
    weight: float
    reward_plain: float
    count_plain: int
    exact_plain: bool
    reward_teacher: float | None = None
    count_teacher: int | None = None
    skipped_teacher: bool = False
    reason_plain: str = ""
    reason_teacher: str = ""
    extra_plain: dict[str, Any] | None = None
    extra_teacher: dict[str, Any] | None = None
    keep: list[dict[str, Any]] = field(default_factory=list)
    images: dict[str, str] = field(default_factory=dict)


@dataclass
class GRPOMember:
    action: torch.Tensor
    noise: torch.Tensor
    mean_old: torch.Tensor
    log_prob_old: torch.Tensor
    reward: float
    count: int
    exact: bool
    image: str | None = None


@dataclass
class GRPOGroup:
    case: dict[str, Any]
    step: int
    x_s: torch.Tensor
    std: torch.Tensor
    members: list[GRPOMember]
    advantages: torch.Tensor
    usable_grpo: bool
    teacher_mean: torch.Tensor | None
    teacher_weight: float
    teacher_reward: float | None = None
    teacher_count: int | None = None
    skipped_teacher: bool = False
    images: dict[str, str] = field(default_factory=dict)


def _save(root: Path, case_id: str, tag: str, image) -> str:
    path = root / "images" / f"{case_id}_{tag}.png"
    save_png(path, image)
    return str(path)


def collect_distill_case(
    runtime: Flux2Runtime,
    evaluator: RewardEvaluator,
    case: TrainingCase,
    cfg: dict[str, Any],
    image_root: Path,
) -> DistillRecord:
    branch = int(cfg["branch_after_completed_steps"])
    keep_steps = set(int(s) for s in cfg.get("keep_steps", ()))
    state = runtime.prefix_to_branch(case.seed, case.plain_prompt, branch, "old")
    x_s = state["latents"]
    step = int(state["step"])
    proposal = runtime.teacher_guided_velocity(x_s, step, case.plain_prompt, case.structured_prompt)
    student = runtime.apply_kernel(x_s, proposal.v_old, step, use_sde=False)
    image_s, keep = runtime.finish_and_decode(
        student.next_latents, case.plain_prompt, step + 1, "old", keep_steps=keep_steps
    )
    path_s = _save(image_root, case.id, "student", image_s)
    result_s = evaluator.evaluate(image_path=path_s, target=case.target, object_text=case.countgd_text, case_id=case.id)
    keep = [
        {"step": step, "latents": x_s.detach().to("cpu"), "velocity": proposal.v_old.detach().to("cpu")}
    ] + keep
    record = DistillRecord(
        case=case.as_dict(),
        step=step,
        x_s=x_s.detach().to("cpu"),
        v_old=proposal.v_old.detach().to("cpu"),
        v_teacher=proposal.v_teacher.detach().to("cpu"),
        weight=0.0,
        reward_plain=result_s.reward,
        count_plain=result_s.count,
        exact_plain=result_s.exact,
        skipped_teacher=False,
        reason_plain=result_s.reason,
        extra_plain=result_s.extra,
        keep=keep,
        images={"student": path_s},
    )
    if cfg.get("skip_teacher_if_student_exact", True) and result_s.exact and result_s.reliable:
        record.skipped_teacher = True
        return record
    if not result_s.reliable:
        record.skipped_teacher = True
        return record
    teacher = runtime.apply_kernel(x_s, proposal.v_teacher, step, use_sde=False)
    image_t, _ = runtime.finish_and_decode(teacher.next_latents, case.plain_prompt, step + 1, "old")
    path_t = _save(image_root, case.id, "teacher", image_t)
    result_t = evaluator.evaluate(image_path=path_t, target=case.target, object_text=case.countgd_text, case_id=case.id)
    record.images["teacher"] = path_t
    record.reward_teacher = result_t.reward
    record.count_teacher = result_t.count
    record.reason_teacher = result_t.reason
    record.extra_teacher = result_t.extra
    if result_t.reliable:
        record.weight = evaluator.gate(result_t.reward, result_s.reward)
    else:
        record.weight = 0.0
        record.skipped_teacher = True
    return record


def collect_grpo_group(
    runtime: Flux2Runtime,
    evaluator: RewardEvaluator,
    case: TrainingCase,
    cfg: dict[str, Any],
    image_root: Path,
) -> GRPOGroup:
    from ..losses.grpo import group_advantages

    branch = int(cfg["branch_after_completed_steps"])
    group_size = int(cfg.get("group_size") or 4)
    state = runtime.prefix_to_branch(case.seed, case.plain_prompt, branch, "old")
    x_s = state["latents"]
    step = int(state["step"])
    proposal = runtime.teacher_guided_velocity(x_s, step, case.plain_prompt, case.structured_prompt)
    members: list[GRPOMember] = []
    images = {}
    std = None
    for index in range(group_size):
        transition = runtime.apply_kernel(x_s, proposal.v_old, step, use_sde=True)
        if std is None:
            std = transition.std.detach().to("cpu")
        image, _ = runtime.finish_and_decode(transition.next_latents, case.plain_prompt, step + 1, "old")
        path = _save(image_root, case.id, f"student{index}", image)
        images[f"student{index}"] = path
        result = evaluator.evaluate(image_path=path, target=case.target, object_text=case.countgd_text, case_id=case.id)
        members.append(
            GRPOMember(
                action=transition.next_latents.detach().to("cpu"),
                noise=transition.noise.detach().to("cpu"),
                mean_old=transition.mean.detach().to("cpu"),
                log_prob_old=transition.log_prob.detach().to("cpu"),
                reward=result.reward,
                count=result.count,
                exact=bool(result.exact and result.reliable),
                image=path,
            )
        )
    rewards = torch.tensor([m.reward for m in members], dtype=torch.float32)
    advantages, usable = group_advantages(rewards, eps=float(cfg.get("loss", {}).get("advantage_std_eps", 1e-8)))
    teacher_mean = None
    teacher_weight = 0.0
    teacher_reward = None
    teacher_count = None
    skipped = False
    trigger = str(cfg.get("teacher_trigger", "student_all_inexact"))
    need_teacher = True
    if trigger == "student_all_inexact":
        need_teacher = not any(m.exact for m in members)
    if not need_teacher:
        skipped = True
    else:
        noise0 = members[0].noise.to(device=x_s.device, dtype=torch.float32)
        teacher = runtime.apply_kernel(x_s, proposal.v_teacher, step, use_sde=True, noise=noise0)
        image_t, _ = runtime.finish_and_decode(teacher.next_latents, case.plain_prompt, step + 1, "old")
        path_t = _save(image_root, case.id, "teacher", image_t)
        images["teacher"] = path_t
        result_t = evaluator.evaluate(image_path=path_t, target=case.target, object_text=case.countgd_text, case_id=case.id)
        teacher_reward = result_t.reward
        teacher_count = result_t.count
        if result_t.reliable:
            teacher_weight = evaluator.gate(result_t.reward, members[0].reward)
            teacher_mean = teacher.mean.detach().to("cpu")
        else:
            skipped = True
    return GRPOGroup(
        case=case.as_dict(),
        step=step,
        x_s=x_s.detach().to("cpu"),
        std=std if std is not None else torch.zeros(()),
        members=members,
        advantages=advantages.detach().cpu(),
        usable_grpo=usable,
        teacher_mean=teacher_mean,
        teacher_weight=float(teacher_weight),
        teacher_reward=teacher_reward,
        teacher_count=teacher_count,
        skipped_teacher=skipped,
        images=images,
    )
