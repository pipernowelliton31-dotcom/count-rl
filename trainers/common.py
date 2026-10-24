"""Shared optimizer, logging, and student-velocity training forwards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.cases import TrainingCase, build_eval_cases, build_train_cases, unique_prompts
from ..core.io_utils import append_jsonl, atomic_json, now_stamp
from ..core.reward import RewardEvaluator
from ..core.runtime import Flux2Runtime


def build_runtime(cfg: dict[str, Any], cases: list[TrainingCase]) -> Flux2Runtime:
    runtime = Flux2Runtime(cfg)
    runtime.precompute_prompts(unique_prompts(cases))
    runtime.snapshot_old()
    return runtime


def build_evaluator(cfg: dict[str, Any], mock_fn=None) -> RewardEvaluator:
    return RewardEvaluator(cfg, mock_fn=mock_fn)


def make_optimizer(runtime: Flux2Runtime, cfg: dict[str, Any]):
    import torch

    opt_cfg = cfg["optimizer"]
    params = runtime.lora.trainable_parameters()
    return torch.optim.AdamW(
        params,
        lr=float(opt_cfg["lr"]),
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
    )


def set_warmup_lr(optimizer, step: int, cfg: dict[str, Any]) -> float:
    opt_cfg = cfg["optimizer"]
    base = float(opt_cfg["lr"])
    warmup = int(opt_cfg.get("warmup_updates", 0))
    lr = base if warmup <= 0 or step >= warmup else base * float(step + 1) / float(warmup)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def clip_and_step(runtime: Flux2Runtime, optimizer, cfg: dict[str, Any]) -> float:
    import torch

    params = runtime.lora.trainable_parameters()
    clip = float(cfg["optimizer"].get("grad_clip", 1.0))
    grad_norm = torch.nn.utils.clip_grad_norm_(params, clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm)


def output_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["experiment"]["output_root"]) / str(cfg["method"]) / now_stamp().replace(":", "")
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    atomic_json(root / "config.json", cfg)
    return root


def log_metrics(root: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["time"] = now_stamp()
    append_jsonl(root / "logs" / "metrics.jsonl", payload)
    print(payload, flush=True)


def round_slices(items: list, n_rounds: int, per_round: int) -> list[list]:
    out = []
    cursor = 0
    for _ in range(n_rounds):
        out.append(items[cursor : cursor + per_round])
        cursor += per_round
    return out


def prepare_cases(cfg: dict[str, Any]) -> tuple[list[TrainingCase], list[TrainingCase]]:
    train = build_train_cases(cfg)
    if cfg.get("_limit"):
        train = train[: int(cfg["_limit"])]
    return train, build_eval_cases(cfg)
