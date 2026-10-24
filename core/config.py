"""Load and merge Version A / Version B YAML configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

METHOD_FILES = {
    "reward_gated_distillation": "distill.yaml",
    "distill": "distill.yaml",
    "a": "distill.yaml",
    "structure_assisted_flow_grpo_fast": "fast_grpo_distill.yaml",
    "fast_grpo": "fast_grpo_distill.yaml",
    "grpo": "fast_grpo_distill.yaml",
    "b": "fast_grpo_distill.yaml",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        candidate = CONFIG_DIR / path.name
        if candidate.is_file():
            path = candidate
        else:
            raise FileNotFoundError(path)
    raw = _read_yaml(path)
    parent_name = raw.get("extends")
    if parent_name:
        parent = load_yaml_config(path.parent / parent_name)
        merged = deep_merge(parent, raw)
    else:
        merged = deepcopy(raw)
    merged["_config_path"] = str(path.resolve())
    return merged


def load_experiment_config(method: str | None = None, path: str | Path | None = None) -> dict[str, Any]:
    if path is not None:
        cfg = load_yaml_config(path)
    elif method is None:
        raise ValueError("Provide --method or --config")
    else:
        key = method.strip().lower()
        if key not in METHOD_FILES:
            raise ValueError(f"Unknown method {method!r}. Expected one of {sorted(METHOD_FILES)}")
        cfg = load_yaml_config(CONFIG_DIR / METHOD_FILES[key])
    cfg.setdefault("method", "reward_gated_distillation")
    cfg.setdefault("use_grpo", False)
    cfg.setdefault("use_sde", False)
    return cfg


def is_version_b(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("use_grpo") or cfg.get("use_sde"))
