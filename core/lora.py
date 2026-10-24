"""Switchable LoRA on FLUX.2 dual-stream attention + single-stream fused projections.

Modes:
  base     — frozen Base 9B (teacher / CFG uncond)
  student  — trainable LoRA (current policy)
  old      — frozen LoRA snapshot used for rollout
"""

from __future__ import annotations

from typing import Iterable, Literal

import torch
import torch.nn as nn

LoraMode = Literal["base", "student", "old"]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if base.weight is None:
            raise ValueError("LoRA requires a Linear with weight")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        in_features = base.in_features
        out_features = base.out_features
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.register_buffer("old_A", self.lora_A.weight.detach().clone())
        self.register_buffer("old_B", self.lora_B.weight.detach().clone())
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.mode: LoraMode = "student"
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def set_mode(self, mode: LoraMode) -> None:
        if mode not in ("base", "student", "old"):
            raise ValueError(mode)
        self.mode = mode

    def snapshot_old(self) -> None:
        self.old_A.copy_(self.lora_A.weight.detach())
        self.old_B.copy_(self.lora_B.weight.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if self.mode == "base":
            return result
        if self.mode == "old":
            delta = nn.functional.linear(nn.functional.linear(x, self.old_A), self.old_B)
            return result + delta * self.scaling
        return result + self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


def _is_target(name: str, module: nn.Module, dual_suffixes: Iterable[str], single_suffixes: Iterable[str]) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if name.startswith("transformer_blocks."):
        return any(name.endswith(suffix) for suffix in dual_suffixes)
    if name.startswith("single_transformer_blocks."):
        return any(name.endswith(suffix) for suffix in single_suffixes)
    return False


def _assign(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


class LoRAManager:
    def __init__(self, transformer: nn.Module, cfg: dict):
        lora_cfg = cfg.get("lora", cfg)
        self.rank = int(lora_cfg.get("rank", 16))
        self.alpha = float(lora_cfg.get("alpha", 32))
        self.dropout = float(lora_cfg.get("dropout", 0.0))
        dual = tuple(lora_cfg.get("dual_suffixes") or ())
        single = tuple(lora_cfg.get("single_suffixes") or ())
        self.layers: list[LoRALinear] = []
        self.names: list[str] = []
        for name, module in list(transformer.named_modules()):
            if _is_target(name, module, dual, single):
                wrapped = LoRALinear(module, self.rank, self.alpha, self.dropout)
                _assign(transformer, name, wrapped)
                self.layers.append(wrapped)
                self.names.append(name)
        if not self.layers:
            raise RuntimeError(
                "No LoRA targets matched. Check Diffusers FLUX.2 module names against lora.dual_suffixes / single_suffixes."
            )
        for name, param in transformer.named_parameters():
            param.requires_grad = ("lora_A" in name) or ("lora_B" in name)
        self.set_mode("student")

    def set_mode(self, mode: LoraMode) -> None:
        for layer in self.layers:
            layer.set_mode(mode)

    def snapshot_old(self) -> None:
        for layer in self.layers:
            layer.snapshot_old()

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.layers:
            params.extend(layer.lora_A.parameters())
            params.extend(layer.lora_B.parameters())
        return params

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: layer.state_dict() for name, layer in zip(self.names, self.layers)}

    def load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        for name, layer in zip(self.names, self.layers):
            layer.load_state_dict(state[name], strict=True)

    def save(self, path) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "names": self.names,
                "rank": self.rank,
                "alpha": self.alpha,
                "state": self.state_dict(),
            },
            path,
        )
