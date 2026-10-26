"""Teacher-only probe: shared plain prefix, Euler residual, structured ODE tail."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core.cases import parse_case_id
from .core.sampler import StageTimer, _save
from .core.reward import count_reward
from .trainers.common import build_evaluator, build_runtime, log_metrics, output_dir


B50_METRICS = Path(
    "/data/storage22t/lsj/flux2-count/outputs/count_rl/"
    "structure_assisted_flow_grpo_fast/2026-10-26T005737+0800/logs/metrics.jsonl"
)


def _old_teacher_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "collect" or not row.get("case"):
            continue
        out[str(row["case"])] = row
    return out


def probe_teacher_cases(cfg: dict[str, Any], case_ids: list[str], lora_path: str | None = None) -> Path:
    cfg = dict(cfg)
    cfg["method"] = f"{cfg.get('method', 'structure_assisted_flow_grpo_fast')}_teacher_probe"
    cases = [parse_case_id(case_id) for case_id in case_ids]
    runtime = build_runtime(cfg, cases)
    if lora_path:
        runtime.lora.load(lora_path)
        runtime.snapshot_old()
    evaluator = build_evaluator(cfg)
    root = output_dir(cfg)
    old_rows = _old_teacher_rows(B50_METRICS)
    branch = int(cfg["branch_after_completed_steps"])
    print(
        json.dumps(
            {
                "event": "probe_start",
                "branch": branch,
                "lora": lora_path,
                "cases": [c.id for c in cases],
                "root": str(root),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for case in cases:
        old = old_rows.get(case.id, {})
        timer = StageTimer(runtime)
        with timer.stage("prefix_sec"):
            state = runtime.prefix_to_branch(case.seed, case.plain_prompt, branch, "old")
        x_s = state["latents"]
        step = int(state["step"])
        with timer.stage("v_old_sec"):
            v_old = runtime.predict_guided(x_s, step, case.plain_prompt, "old", grad=False)
        with timer.stage("teacher_sec"):
            _teacher, _v_teacher, latent_t = runtime.run_teacher_structured_ode(
                x_s, v_old, step, case.plain_prompt, case.structured_prompt
            )
        images, decode_timing = runtime.decode_batch([latent_t])
        timer.merge_decode(decode_timing)
        path_t = _save(root, case.id, "teacher", images[0])
        with timer.stage("countgd_sec"):
            result = evaluator.evaluate_with_retry(
                image_path=path_t, target=case.target, object_text=case.countgd_text, case_id=case.id
            )
        old_count = old.get("teacher_count")
        student_counts = old.get("counts") or []
        modal = max(set(student_counts), key=student_counts.count) if student_counts else None
        payload = {
            "event": "probe_teacher",
            "case": case.id,
            "target": case.target,
            "step": step,
            "new_teacher_count": result.count,
            "new_teacher_reward": result.reward,
            "new_teacher_exact": result.exact,
            "new_reliable": result.reliable,
            "new_reason": result.reason,
            "old_teacher_count": old_count,
            "old_teacher_weight": old.get("teacher_weight"),
            "old_student_counts": student_counts,
            "old_modal_student": modal,
            "image": path_t,
            **timer.times,
        }
        log_metrics(root, payload)
        print(
            f"{case.id} target={case.target} old_T={old_count} new_T={result.count} "
            f"students={student_counts} exact={result.exact} w_vs_modal="
            f"{evaluator.gate(result.reward, count_reward(modal, case.target)) if modal is not None else None}",
            flush=True,
        )
    return root
