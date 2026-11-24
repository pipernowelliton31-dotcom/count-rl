"""Final-image counting reward and teacher gate. CountGD is not differentiable."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO


def count_reward(
    hat_n: int | float, target: int, exact_bonus: float = 0.3,
    overcount_penalty: float = 0.0,
) -> float:
    """Bounded count reward. Extra count is penalized asymmetrically when enabled.

    Default overcount_penalty=0 reproduces all historical baseline rewards.
    """
    import math
    penalty = float(overcount_penalty)
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("overcount_penalty must be finite and nonnegative")
    signed_err = float(hat_n) - float(target)
    err = abs(signed_err)
    denom = 1.0 + err + penalty * max(0.0, signed_err)
    return (1.0 / denom) + (exact_bonus if err == 0 else 0.0)


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
    attempts: int = 1


class RewardEvaluator:
    """CountGD-backed evaluator with a mock backend for CPU tests."""

    def __init__(self, cfg: dict[str, Any], mock_fn: Callable[..., RewardResult] | None = None):
        reward_cfg = cfg.get("reward", {})
        self.backend = str(reward_cfg.get("backend", "countgd"))
        self.exact_bonus = float(reward_cfg.get("exact_bonus", 0.3))
        self.overcount_penalty = float(reward_cfg.get("overcount_penalty", 0.0))
        if not __import__("math").isfinite(self.overcount_penalty) or self.overcount_penalty < 0:
            raise ValueError("reward.overcount_penalty must be finite and >= 0")
        self.gain_scale = float(reward_cfg.get("gain_scale", 0.5))
        self.max_plausible = int(reward_cfg.get("max_plausible_count", 40))
        self.threshold = float(reward_cfg.get("confidence_threshold", 0.26))
        self.device = str(reward_cfg.get("device", "cuda"))
        self.countgd_root = Path(reward_cfg.get("countgd_root", "/data/storage22t/lsj/repos/CountGD"))
        self.layout_root = Path(reward_cfg.get("layout_v51_root", "/home/lsj/layout_v51_conservative"))
        default_py = Path("/data/storage22t/lsj/conda/envs/countgd/bin/python")
        configured_py = reward_cfg.get("countgd_python")
        if configured_py:
            self.countgd_python = Path(configured_py)
        elif default_py.is_file():
            self.countgd_python = default_py
        else:
            self.countgd_python = Path(sys.executable)
        self.mock_fn = mock_fn
        self._runner = None
        self._worker: subprocess.Popen | None = None
        self.max_retries = int(reward_cfg.get("max_retries", 2))
        self.deform_attn_backend = str(reward_cfg.get("deform_attn_backend", "cuda")).lower()
        if self.deform_attn_backend not in {"cuda", "pytorch"}:
            raise ValueError(f"Invalid reward.deform_attn_backend={self.deform_attn_backend!r}")

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

    def evaluate_with_retry(
        self,
        *,
        image_path: str | Path | None,
        target: int,
        object_text: str,
        case_id: str = "",
    ) -> RewardResult:
        """Retry only transient CountGD failures. implausible_count is not retried."""
        last: RewardResult | None = None
        attempts = 0
        max_attempts = 1 + max(0, self.max_retries)
        while attempts < max_attempts:
            attempts += 1
            last = self.evaluate(
                image_path=image_path, target=target, object_text=object_text, case_id=case_id
            )
            last.attempts = attempts
            if last.reliable:
                return last
            if not str(last.reason).startswith("detect_failed"):
                return last
            if self._worker is not None:
                self._stop_worker()
        return last if last is not None else RewardResult(
            count=-1, reward=0.0, exact=False, reliable=False, reason="detect_failed", attempts=attempts
        )

    def _use_subprocess(self) -> bool:
        return Path(self.countgd_python).is_file() and Path(self.countgd_python).resolve() != Path(sys.executable).resolve()

    def _load_countgd(self):
        if self._use_subprocess():
            return self._ensure_worker()
        if self._runner is not None:
            return self._runner
        import types

        layout_root = str(self.layout_root)
        if layout_root not in sys.path:
            sys.path.insert(0, layout_root)
        countgd_root = str(self.countgd_root)
        if countgd_root not in sys.path:
            sys.path.insert(0, countgd_root)
        # CountGD's datasets_inference/__init__.py imports pycocotools. The t2i env
        # has a numpy/pycocotools ABI mismatch; detect() only needs .transforms.
        if "datasets_inference" not in sys.modules:
            pkg = types.ModuleType("datasets_inference")
            pkg.__path__ = [str(self.countgd_root / "datasets_inference")]
            pkg.__package__ = "datasets_inference"
            pkg.__file__ = str(self.countgd_root / "datasets_inference" / "__init__.py")
            sys.modules["datasets_inference"] = pkg
        from lib.detector import CountGDRunner, discover_countgd_assets

        assets = discover_countgd_assets(countgd_root=self.countgd_root)
        self._runner = CountGDRunner(assets, device=self.device, trust_checkpoint=True)
        return self._runner

    def _ensure_worker(self) -> subprocess.Popen:
        if self._worker is not None and self._worker.poll() is None:
            return self._worker
        log_dir = Path(os.environ.get("COUNTGD_WORKER_LOG_DIR", "/tmp/count_rl_smoke"))
        log_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["COUNTGD_DEFORM_ATTN_BACKEND"] = self.deform_attn_backend
        rank = os.environ.get("LOCAL_RANK", "standalone")
        worker_log = log_dir / f"countgd_worker_rank{rank}.err"
        proc = subprocess.Popen(
            [
                str(self.countgd_python),
                "-u",
                "-c",
                _COUNTGD_WORKER,
                str(self.layout_root),
                str(self.countgd_root),
                str(self.device),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(worker_log, "a"),
            env=env,
            text=True,
            bufsize=1,
        )
        ready = self._read_worker_message(proc, timeout_label="init")
        if not ready.get("ok"):
            raise RuntimeError(f"CountGD worker init failed: {ready}")
        self._worker = proc
        atexit.register(self._stop_worker)
        return proc

    def _stop_worker(self) -> None:
        proc = self._worker
        self._worker = None
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass

    def _detect_subprocess(self, image_path: Path, object_text: str) -> dict[str, Any]:
        proc = self._ensure_worker()
        stdin: TextIO | None = proc.stdin
        if stdin is None:
            raise RuntimeError("CountGD worker stdin is missing")
        stdin.write(json.dumps({"cmd": "detect", "image_path": str(image_path), "object_text": object_text}) + "\n")
        stdin.flush()
        payload = self._read_worker_message(proc, timeout_label="detect")
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error") or payload)
        return payload

    def _read_worker_message(self, proc: subprocess.Popen, timeout_label: str) -> dict[str, Any]:
        stdout: TextIO | None = proc.stdout
        if stdout is None:
            raise RuntimeError("CountGD worker stdout is missing")
        while True:
            line = stdout.readline()
            if not line:
                raise RuntimeError(f"CountGD worker died during {timeout_label}; inspect COUNTGD_WORKER_LOG_DIR")
            line = line.strip()
            if not line.startswith("COUNTGD_MSG "):
                continue
            return json.loads(line[len("COUNTGD_MSG ") :])

    def _detections_extra(self, detections, count: int) -> dict[str, Any]:
        extra: dict[str, Any] = {"n_boxes": count, "threshold": self.threshold}
        if isinstance(detections, dict):
            if detections.get("scores") is not None:
                extra["scores"] = [float(x) for x in detections["scores"]]
            if detections.get("boxes"):
                extra["boxes"] = detections["boxes"]
            extra["via"] = detections.get("via", "subprocess")
            return extra
        if detections is None or not hasattr(detections, "columns"):
            return extra
        if "confidence" in detections.columns:
            extra["scores"] = [float(x) for x in detections["confidence"].tolist()]
        records = []
        for row in detections.to_dict(orient="records"):
            item = {}
            for key in (
                "confidence",
                "cx_norm",
                "cy_norm",
                "w_norm",
                "h_norm",
                "query_idx",
            ):
                if key in row and row[key] is not None:
                    try:
                        item[key] = float(row[key]) if key != "query_idx" else int(row[key])
                    except (TypeError, ValueError):
                        item[key] = row[key]
            if item:
                records.append(item)
        if records:
            extra["boxes"] = records
        extra["via"] = "inprocess"
        return extra

    def _evaluate_countgd(self, *, image_path: Path, target: int, object_text: str) -> RewardResult:
        try:
            if self._use_subprocess():
                detections = self._detect_subprocess(image_path, object_text)
                count = int(detections.get("count", -1))
            else:
                detections = self._load_countgd().detect(image_path=image_path, object_text=object_text)
                count = int(len(detections))
        except Exception as exc:
            cause = f" cause={exc.__cause__!r}" if exc.__cause__ is not None else ""
            return RewardResult(
                count=-1,
                reward=0.0,
                exact=False,
                reliable=False,
                reason=f"detect_failed:{exc!r}{cause}",
            )
        extra = self._detections_extra(detections, count)
        if count < 0 or count > self.max_plausible:
            return RewardResult(
                count=count,
                reward=count_reward(count, target, self.exact_bonus, self.overcount_penalty),
                exact=False,
                reliable=False,
                reason="implausible_count",
                extra=extra,
            )
        exact = is_exact(count, target)
        return RewardResult(
            count=count,
            reward=count_reward(count, target, self.exact_bonus, self.overcount_penalty),
            exact=exact,
            reliable=True,
            extra=extra,
        )


_COUNTGD_WORKER = r"""
import json, os, sys, traceback
from pathlib import Path

layout_root, countgd_root, device = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, layout_root)
from lib.detector import CountGDRunner, discover_countgd_assets

def emit(payload):
    sys.stdout.write("COUNTGD_MSG " + json.dumps(payload) + "\n")
    sys.stdout.flush()

try:
    assets = discover_countgd_assets(countgd_root=countgd_root)
    backend = os.environ.get("COUNTGD_DEFORM_ATTN_BACKEND", "cuda").lower()
    if backend == "pytorch":
        # Inference-only alternative to the native C++/CUDA kernel, which crashes
        # with illegal memory access intermittently on physical GPU 4.
        # Patch only THIS isolated reward subprocess, never shared CountGD files.
        sys.path.insert(0, str(countgd_root))
        from models.GroundingDINO import ms_deform_attn as _deform
        def _safe_deform_attn(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
            return _deform.multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
        _deform.MultiScaleDeformableAttnFunction.apply = staticmethod(_safe_deform_attn)
    elif backend != "cuda":
        raise ValueError(f"Invalid COUNTGD_DEFORM_ATTN_BACKEND={backend!r}")
    runner = CountGDRunner(assets, device=device, trust_checkpoint=True)
    emit({"ok": True, "event": "ready", "backend": backend})
except Exception:
    emit({"ok": False, "error": traceback.format_exc()})
    raise SystemExit(1)

for line in sys.stdin:
    req = json.loads(line)
    if req.get("cmd") == "quit":
        break
    try:
        det = runner.detect(image_path=req["image_path"], object_text=req["object_text"])
        boxes = []
        scores = []
        if hasattr(det, "to_dict"):
            scores = [float(x) for x in det["confidence"].tolist()] if "confidence" in det.columns else []
            cols = [c for c in ("confidence", "cx_norm", "cy_norm", "w_norm", "h_norm") if c in det.columns]
            boxes = det[cols].to_dict(orient="records") if cols else []
        emit({"ok": True, "count": int(len(det)), "scores": scores, "boxes": boxes, "via": "subprocess"})
    except Exception:
        emit({"ok": False, "error": traceback.format_exc()})
"""
