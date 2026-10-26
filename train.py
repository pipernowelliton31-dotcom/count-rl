#!/usr/bin/env python3
"""Train Version A (distill) or Version B (GRPO-Fast + KL)."""

from __future__ import annotations

import argparse
import json

from .core.cases import build_eval_cases, build_train_cases
from .core.config import is_version_b, load_experiment_config


def comparison_table(cfg_a: dict, cfg_b: dict) -> list[dict[str, str]]:
    def g(cfg, *keys, default=""):
        cur = cfg
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    rows = [
        ("method", cfg_a.get("method"), cfg_b.get("method")),
        ("use_sde", cfg_a.get("use_sde"), cfg_b.get("use_sde")),
        ("use_grpo", cfg_a.get("use_grpo"), cfg_b.get("use_grpo")),
        ("group_size", cfg_a.get("group_size"), cfg_b.get("group_size")),
        ("branch_after_completed_steps", cfg_a.get("branch_after_completed_steps"), cfg_b.get("branch_after_completed_steps")),
        ("teacher", "1 ODE action", "1 Euler residual + structured ODE tail"),
        ("student_trajectories", 1, 4),
        ("max_images_per_case", 2, 5),
        ("core_loss", "w * MSE(v_φ, v_T) + 0.1 L_keep", "L_GRPO + λ w KL_dim + β L_ref"),
        ("lr", g(cfg_a, "optimizer", "lr"), g(cfg_b, "optimizer", "lr")),
        ("lora_rank", g(cfg_a, "lora", "rank"), g(cfg_b, "lora", "rank")),
        ("guidance", g(cfg_a, "model", "guidance"), g(cfg_b, "model", "guidance")),
        ("steps", g(cfg_a, "model", "steps"), g(cfg_b, "model", "steps")),
        ("train_units", g(cfg_a, "budget", "n_train_cases"), g(cfg_b, "budget", "n_train_groups")),
        ("quota_9_10", f"{g(cfg_a,'budget','quota',9)}/{g(cfg_a,'budget','quota',10)}", f"{g(cfg_b,'budget','quota',9)}/{g(cfg_b,'budget','quota',10)}"),
        ("teacher_trigger", "student not exact", g(cfg_b, "teacher_trigger")),
        ("eval", "plain prompt, 50-step ODE, no teacher", "plain prompt, 50-step ODE, no teacher"),
    ]
    return [{"field": a, "version_a": str(b), "version_b": str(c)} for a, b, c in rows]


def print_comparison() -> None:
    cfg_a = load_experiment_config("distill")
    cfg_b = load_experiment_config("fast_grpo")
    rows = comparison_table(cfg_a, cfg_b)
    print(f"{'field':<32} {'Version A':<48} {'Version B'}")
    print("-" * 120)
    for row in rows:
        print(f"{row['field']:<32} {row['version_a']:<48} {row['version_b']}")
    print()
    print(f"A train cases: {len(build_train_cases(cfg_a))}  eval: {len(build_eval_cases(cfg_a))}")
    print(f"B train groups: {len(build_train_cases(cfg_b))}  eval: {len(build_eval_cases(cfg_b))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=None, help="distill | fast_grpo")
    parser.add_argument("--config", default=None, help="YAML path; overrides --method")
    parser.add_argument("--dry-run", action="store_true", help="Print cases and comparison; do not load FLUX.2")
    parser.add_argument("--compare", action="store_true", help="Print Version A vs B config table and exit")
    parser.add_argument("--selftest", action="store_true", help="Run CPU unit checks and exit")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on train units for a debug run")
    parser.add_argument("--reward-backend", default=None, help="Override reward.backend (countgd|mock)")
    parser.add_argument(
        "--probe-teacher",
        nargs="+",
        metavar="CASE_ID",
        help="Regenerate teacher-only structured ODE images for these case ids and exit",
    )
    parser.add_argument("--lora", default=None, help="Optional LoRA checkpoint for --probe-teacher")
    args = parser.parse_args(argv)

    if args.selftest:
        from .selftest import run_selftest

        run_selftest()
        return 0
    if args.compare or (args.dry_run and args.method is None and args.config is None):
        print_comparison()
        return 0

    cfg = load_experiment_config(args.method, args.config)
    if args.reward_backend:
        cfg.setdefault("reward", {})["backend"] = args.reward_backend
    if args.probe_teacher:
        from .probe_teacher import probe_teacher_cases

        root = probe_teacher_cases(cfg, args.probe_teacher, lora_path=args.lora)
        print(f"done {root}", flush=True)
        return 0
    if args.limit:
        cfg["_limit"] = int(args.limit)
        cfg["budget"]["n_rounds"] = 1
        per = "groups_per_round" if is_version_b(cfg) else "cases_per_round"
        cfg["budget"][per] = int(args.limit)

    if args.dry_run:
        print(json.dumps({k: cfg[k] for k in ("method", "use_sde", "use_grpo", "group_size") if k in cfg}, indent=2))
        cases = build_train_cases(cfg)
        if args.limit:
            cases = cases[: args.limit]
        evals = build_eval_cases(cfg)
        print(f"train_units={len(build_train_cases(cfg))} showing={len(cases)} eval={len(evals)}")
        print(f"first={cases[0].id if cases else None}")
        print(f"plain={cases[0].plain_prompt if cases else None}")
        print(f"structured={cases[0].structured_prompt if cases else None}")
        print_comparison()
        return 0

    if is_version_b(cfg):
        from .trainers.fast_grpo import FastGRPOTrainer

        trainer = FastGRPOTrainer(cfg)
    else:
        from .trainers.distillation import DistillationTrainer

        trainer = DistillationTrainer(cfg)
    root = trainer.fit()
    print(f"done {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
