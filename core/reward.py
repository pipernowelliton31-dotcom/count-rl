"""Final-image counting reward and teacher gate. CountGD is not differentiable."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


def count_reward(hat_n: int | float, target: int, exact_bonus: float = 0.3) -> float:
    err = abs(float(hat_n) - float(target))
    return (1.0 / (1.0 + err)) + (exact_bonus if err == 0 else 0.0)


def reward_gate(reward_teacher: float, reward_student: float, gain_scale: float = 0.5) -> float:
    """Clip positive reward gain into [0, 1]. Zero if the teacher is not better."""
    if gain_scale <= 0:
        raise ValueError("gain_scale must be positive")
    delta = float(reward_teacher) - float(reward_student)
    if delta <= 0:
        return 0.0
    weight = delta / float(gain_scale)
    return 1.0 if weight > 1.0 else weight


def is_exact(hat_n: int | float, target: int) -> bool:
    return int(round(float(hat_n))) == int(target)


@dataclass
class RewardResult:
    count: int
    reward: float
    exact: bool
    reliable: bool
    reason: str = ""
    extra: dict[str, Any] | None = None


class RewardEvaluator:
    """CountGD-backed evaluator with a mock backend for CPU tests."""

    def __init__(self, cfg: dict[str, Any], mock_fn: Callable[..., RewardResult] | None = None):
        reward_cfg = cfg.get("reward", {})
        self.backend = str(reward_cfg.get("backend", "countgd"))
        self.exact_bonus = float(reward_cfg.get("exact_bonus", 0.3))
        self.gain_scale = float(reward_cfg.get("gain_scale", 0.5))
        self.max_plausible = int(reward_cfg.get("max_plausible_count", 40))
        self.threshold = float(reward_cfg.get("confidence_threshold", 0.26))
        self.device = str(reward_cfg.get("device", "cuda"))
        self.countgd_root = Path(reward_cfg.get("countgd_root", "/data/storage22t/lsj/repos/CountGD"))
        self.layout_root = Path(reward_cfg.get("layout_v51_root", "/home/lsj/layout_v51_conservative"))
        self.mock_fn = mock_fn
        self._runner = None

    def gate(self, reward_teacher: float, reward_student: float) -> float:
        return reward_gate(reward_teacher, reward_student, self.gain_scale)

    def evaluate(
        self,
        *,
        image_path: str | Path | None,
        target: int,
        object_text: str,
        case_id: str = "",
    ) -> RewardResult:
        if self.backend == "mock":
            if self.mock_fn is None:
                raise RuntimeError("reward.backend=mock requires a mock_fn")
            return self.mock_fn(image_path=image_path, target=target, object_text=object_text, case_id=case_id)
        return self._evaluate_countgd(image_path=Path(image_path), target=target, object_text=object_text)

    def _load_countgd(self):
        if self._runner is not None:
            return self._runner
        import sys

        root = str(self.layout_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from lib.detector import CountGDRunner, discover_countgd_assets

        assets = discover_countgd_assets(self.countgd_root)
        self._runner = CountGDRunner(assets, device=self.device, trust_checkpoint=True)
        return self._runner

    def _evaluate_countgd(self, *, image_path: Path, target: int, object_text: str) -> RewardResult:
        try:
            detections = self._load_countgd().detect(image_path=image_path, object_text=object_text)
        except Exception as exc:
            return RewardResult(count=-1, reward=0.0, exact=False, reliable=False, reason=f"detect_failed:{exc!r}")
        count = int(len(detections))
        if count < 0 or count > self.max_plausible:
            return RewardResult(
                count=count,
                reward=count_reward(count, target, self.exact_bonus),
                exact=False,
                reliable=False,
                reason="implausible_count",
                extra={"n_boxes": count},
            )
        exact = is_exact(count, target)
        return RewardResult(
            count=count,
            reward=count_reward(count, target, self.exact_bonus),
            exact=exact,
            reliable=True,
            extra={"n_boxes": count, "threshold": self.threshold},
        )
