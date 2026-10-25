"""Held-out plain-prompt ODE eval. No teacher, no best-of-G."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..core.cases import TrainingCase
from ..core.io_utils import atomic_json, save_png
from ..core.reward import RewardEvaluator, is_exact
from ..core.runtime import Flux2Runtime


def evaluate_plain_ode(
    runtime: Flux2Runtime,
    evaluator: RewardEvaluator,
    cases: list[TrainingCase],
    root: Path,
    tag: str = "eval",
) -> dict[str, Any]:
    out = root / tag
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    by_target: dict[int, list[bool]] = defaultdict(list)
    mae = []
    batch_size = int(getattr(runtime, "cfg", {}).get("eval", {}).get("decode_batch_size", 16) or 16)
    pending_cases: list[TrainingCase] = []
    pending_latents = []

    def _flush() -> None:
        if not pending_cases:
            return
        images, _timing = runtime.decode_batch(pending_latents)
        for case, image in zip(list(pending_cases), images):
            path = out / "images" / f"{case.id}.png"
            save_png(path, image)
            result = evaluator.evaluate_with_retry(
                image_path=path, target=case.target, object_text=case.countgd_text, case_id=case.id
            )
            exact = is_exact(result.count, case.target) if result.reliable else False
            by_target[case.target].append(exact)
            if result.reliable:
                mae.append(abs(result.count - case.target))
            rows.append(
                {
                    **case.as_dict(),
                    "count": result.count,
                    "reward": result.reward,
                    "exact": exact,
                    "reliable": result.reliable,
                    "reason": result.reason,
                    "attempts": result.attempts,
                    "image": str(path),
                }
            )
        pending_cases.clear()
        pending_latents.clear()

    for case in cases:
        state = runtime.prefix_to_branch(case.seed, case.plain_prompt, 0, "student")
        latent, _ = runtime.finish_tail(state["latents"], case.plain_prompt, 0, "student")
        pending_cases.append(case)
        pending_latents.append(latent)
        if len(pending_cases) >= batch_size:
            _flush()
    _flush()
    summary = {
        "n": len(rows),
        "exact_accuracy": (sum(r["exact"] for r in rows) / len(rows)) if rows else 0.0,
        "mae": (sum(mae) / len(mae)) if mae else None,
        "by_target": {
            str(t): (sum(flags) / len(flags) if flags else 0.0) for t, flags in sorted(by_target.items())
        },
        "rows": rows,
    }
    atomic_json(out / "summary.json", summary)
    return summary
