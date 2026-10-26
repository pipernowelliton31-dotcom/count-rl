"""Collect student / teacher trajectories. Generation has no gradient."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .cases import TrainingCase
from .io_utils import save_png
from .reward import RewardEvaluator
from .runtime import Flux2Runtime

KNOWN_TRIGGERS = ("student_all_inexact", "half_inexact", "never", "always")


def _sync_cuda(runtime: Flux2Runtime) -> None:
    torch_mod = getattr(runtime, "torch", None)
    if torch_mod is not None and hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()


class StageTimer:
    def __init__(self, runtime: Flux2Runtime):
        self.runtime = runtime
        self.times: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        self.times[name] = self.times.get(name, 0.0) + float(seconds)

    def stage(self, name: str):
        runtime = self.runtime

        class _Ctx:
            def __enter__(_self):
                _sync_cuda(runtime)
                _self.t0 = time.perf_counter()
                return self

            def __exit__(_self, *_exc):
                _sync_cuda(runtime)
                self.add(name, time.perf_counter() - _self.t0)
                return False

        return _Ctx()

    def merge_decode(self, timing: dict[str, float]) -> None:
        self.add("offload_sec", timing.get("offload_sec", 0.0))
        self.add("decode_sec", timing.get("decode_sec", 0.0))


def _save(root: Path, case_id: str, tag: str, image) -> str:
    path = root / "images" / f"{case_id}_{tag}.png"
    save_png(path, image)
    return str(path)


def _need_teacher(trigger: str, members: list["GRPOMember"]) -> bool:
    if trigger not in KNOWN_TRIGGERS:
        raise ValueError(f"Unknown teacher_trigger={trigger!r}. Expected one of {KNOWN_TRIGGERS}")
    if trigger == "never":
        return False
    if trigger == "always":
        return True
    wrong = sum(1 for m in members if not m.exact)
    n = len(members)
    if trigger == "half_inexact":
        return 2 * wrong >= n
    return wrong == n


@dataclass
class DistillRecord:
    case: dict[str, Any]
    step: int
    x_s: torch.Tensor
    v_old: torch.Tensor
    v_teacher: torch.Tensor | None
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
    reliable_plain: bool = True
    attempts_plain: int = 1
    attempts_teacher: int = 0
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class GRPOMember:
    action: torch.Tensor
    noise: torch.Tensor
    mean_old: torch.Tensor
    log_prob_old: torch.Tensor
    reward: float
    count: int
    exact: bool
    reliable: bool = True
    reason: str = ""
    attempts: int = 1
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
    invalid_reward: bool = False
    teacher_anchor: int | None = None
    images: dict[str, str] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)


def collect_distill_round(
    runtime: Flux2Runtime,
    evaluator: RewardEvaluator,
    cases: list[TrainingCase],
    cfg: dict[str, Any],
    image_root: Path,
) -> list[DistillRecord]:
    """Two-phase A collect: all students decode once, then only needed teachers."""
    branch = int(cfg["branch_after_completed_steps"])
    keep_steps = set(int(s) for s in cfg.get("keep_steps", ()))
    skip_if_exact = bool(cfg.get("skip_teacher_if_student_exact", True))
    pending: list[dict[str, Any]] = []
    for case in cases:
        timer = StageTimer(runtime)
        with timer.stage("prefix_sec"):
            state = runtime.prefix_to_branch(case.seed, case.plain_prompt, branch, "old")
        x_s = state["latents"]
        step = int(state["step"])
        with timer.stage("v_old_sec"):
            v_old = runtime.predict_guided(x_s, step, case.plain_prompt, "old", grad=False)
        student = runtime.apply_kernel(x_s, v_old, step, use_sde=False)
        with timer.stage("tail_sec"):
            latent, keep = runtime.finish_tail(student.next_latents, case.plain_prompt, step + 1, "old", keep_steps=keep_steps)
        pending.append(
            {
                "case": case,
                "timer": timer,
                "x_s": x_s,
                "step": step,
                "v_old": v_old,
                "latent": latent,
                "keep": [
                    {"step": step, "latents": x_s.detach().to("cpu"), "velocity": v_old.detach().to("cpu")}
                ]
                + keep,
            }
        )
    images, decode_timing = runtime.decode_batch([p["latent"] for p in pending])
    per = (decode_timing["offload_sec"] / max(len(pending), 1), decode_timing["decode_sec"] / max(len(pending), 1))
    records: list[DistillRecord] = []
    teacher_jobs: list[tuple[int, Any]] = []
    for index, (item, image) in enumerate(zip(pending, images)):
        item["timer"].add("offload_sec", per[0])
        item["timer"].add("decode_sec", per[1])
        case = item["case"]
        path_s = _save(image_root, case.id, "student", image)
        with item["timer"].stage("countgd_sec"):
            result_s = evaluator.evaluate_with_retry(
                image_path=path_s, target=case.target, object_text=case.countgd_text, case_id=case.id
            )
        record = DistillRecord(
            case=case.as_dict(),
            step=item["step"],
            x_s=item["x_s"].detach().to("cpu"),
            v_old=item["v_old"].detach().to("cpu"),
            v_teacher=None,
            weight=0.0,
            reward_plain=result_s.reward,
            count_plain=result_s.count,
            exact_plain=result_s.exact,
            skipped_teacher=True,
            reason_plain=result_s.reason,
            extra_plain=result_s.extra,
            keep=item["keep"],
            images={"student": path_s},
            reliable_plain=result_s.reliable,
            attempts_plain=result_s.attempts,
            timing=item["timer"].times,
        )
        records.append(record)
        if not result_s.reliable:
            continue
        if skip_if_exact and result_s.exact:
            continue
        teacher_jobs.append((index, item))

    if not teacher_jobs:
        return records

    teacher_latents = []
    for index, item in teacher_jobs:
        case = item["case"]
        with item["timer"].stage("teacher_sec"):
            v_teacher = runtime.teacher_velocity_from_old(
                item["v_old"], item["x_s"], item["step"], case.plain_prompt, case.structured_prompt
            )
            teacher = runtime.apply_kernel(item["x_s"], v_teacher, item["step"], use_sde=False)
            latent_t, _ = runtime.finish_tail(teacher.next_latents, case.plain_prompt, item["step"] + 1, "old")
        item["v_teacher"] = v_teacher
        teacher_latents.append(latent_t)
    images_t, t_timing = runtime.decode_batch(teacher_latents)
    t_off = t_timing["offload_sec"] / max(len(teacher_jobs), 1)
    t_dec = t_timing["decode_sec"] / max(len(teacher_jobs), 1)
    for (index, item), image_t in zip(teacher_jobs, images_t):
        item["timer"].add("offload_sec", t_off)
        item["timer"].add("decode_sec", t_dec)
        case = item["case"]
        path_t = _save(image_root, case.id, "teacher", image_t)
        with item["timer"].stage("countgd_sec"):
            result_t = evaluator.evaluate_with_retry(
                image_path=path_t, target=case.target, object_text=case.countgd_text, case_id=case.id
            )
        record = records[index]
        record.images["teacher"] = path_t
        record.v_teacher = item["v_teacher"].detach().to("cpu")
        record.reward_teacher = result_t.reward
        record.count_teacher = result_t.count
        record.reason_teacher = result_t.reason
        record.extra_teacher = result_t.extra
        record.attempts_teacher = result_t.attempts
        record.timing = item["timer"].times
        if result_t.reliable:
            record.weight = evaluator.gate(result_t.reward, record.reward_plain)
            record.skipped_teacher = False
        else:
            record.weight = 0.0
            record.skipped_teacher = True
    return records


def collect_distill_case(
    runtime: Flux2Runtime,
    evaluator: RewardEvaluator,
    case: TrainingCase,
    cfg: dict[str, Any],
    image_root: Path,
) -> DistillRecord:
    return collect_distill_round(runtime, evaluator, [case], cfg, image_root)[0]


def collect_grpo_group(
    runtime: Flux2Runtime,
    evaluator: RewardEvaluator,
    case: TrainingCase,
    cfg: dict[str, Any],
    image_root: Path,
) -> GRPOGroup:
    from ..losses.grpo import group_advantages

    timer = StageTimer(runtime)
    branch = int(cfg["branch_after_completed_steps"])
    group_size = int(cfg.get("group_size") or 4)
    trigger = str(cfg.get("teacher_trigger", "half_inexact"))
    with timer.stage("prefix_sec"):
        state = runtime.prefix_to_branch(case.seed, case.plain_prompt, branch, "old")
    x_s = state["latents"]
    step = int(state["step"])
    with timer.stage("v_old_sec"):
        v_old = runtime.predict_guided(x_s, step, case.plain_prompt, "old", grad=False)
    drafts: list[dict[str, Any]] = []
    std = None
    with timer.stage("tail_sec"):
        for index in range(group_size):
            transition = runtime.apply_kernel(x_s, v_old, step, use_sde=True)
            if std is None:
                std = transition.std.detach().to("cpu")
            latent, _ = runtime.finish_tail(transition.next_latents, case.plain_prompt, step + 1, "old")
            drafts.append(
                {
                    "action": transition.next_latents.detach().to("cpu"),
                    "noise": transition.noise.detach().to("cpu") if transition.noise is not None else None,
                    "mean_old": transition.mean.detach().to("cpu"),
                    "log_prob_old": transition.log_prob.detach().to("cpu") if transition.log_prob is not None else None,
                    "latent": latent,
                }
            )
    images, decode_timing = runtime.decode_batch([d["latent"] for d in drafts])
    timer.merge_decode(decode_timing)
    members: list[GRPOMember] = []
    saved: dict[str, str] = {}
    with timer.stage("countgd_sec"):
        for index, (draft, image) in enumerate(zip(drafts, images)):
            path = _save(image_root, case.id, f"student{index}", image)
            saved[f"student{index}"] = path
            result = evaluator.evaluate_with_retry(
                image_path=path, target=case.target, object_text=case.countgd_text, case_id=case.id
            )
            members.append(
                GRPOMember(
                    action=draft["action"],
                    noise=draft["noise"],
                    mean_old=draft["mean_old"],
                    log_prob_old=draft["log_prob_old"],
                    reward=result.reward,
                    count=result.count,
                    exact=bool(result.exact and result.reliable),
                    reliable=result.reliable,
                    reason=result.reason,
                    attempts=result.attempts,
                    image=path,
                )
            )
    invalid = not all(m.reliable for m in members)
    if invalid:
        advantages = torch.zeros(len(members), dtype=torch.float32)
        usable = False
    else:
        rewards = torch.tensor([m.reward for m in members], dtype=torch.float32)
        advantages, usable = group_advantages(rewards, eps=float(cfg.get("loss", {}).get("advantage_std_eps", 1e-8)))
    teacher_mean = None
    teacher_weight = 0.0
    teacher_reward = None
    teacher_count = None
    teacher_anchor = None
    skipped = True
    need_teacher = (not invalid) and _need_teacher(trigger, members)
    if need_teacher:
        wrong_ids = [i for i, m in enumerate(members) if not m.exact]
        if not wrong_ids:
            skipped = True
        else:
            rng = random.Random(int(case.seed))
            teacher_anchor = int(rng.choice(wrong_ids))
            anchor = members[teacher_anchor]
            with timer.stage("teacher_sec"):
                teacher, _v_teacher, latent_t = runtime.run_teacher_structured_ode(
                    x_s, v_old, step, case.plain_prompt, case.structured_prompt
                )
            images_t, t_timing = runtime.decode_batch([latent_t])
            timer.merge_decode(t_timing)
            path_t = _save(image_root, case.id, "teacher", images_t[0])
            saved["teacher"] = path_t
            with timer.stage("countgd_sec"):
                result_t = evaluator.evaluate_with_retry(
                    image_path=path_t, target=case.target, object_text=case.countgd_text, case_id=case.id
                )
            teacher_reward = result_t.reward
            teacher_count = result_t.count
            if result_t.reliable:
                teacher_weight = evaluator.gate(result_t.reward, anchor.reward)
                teacher_mean = teacher.mean.detach().to("cpu")
                skipped = False
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
        invalid_reward=invalid,
        teacher_anchor=teacher_anchor,
        images=saved,
        timing=timer.times,
    )
