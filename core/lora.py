"""Switchable LoRA on FLUX.2 dual-stream attention + single-stream fused projections.

Modes:
  base     — frozen Base 9B (teacher / CFG uncond)
  student  — trainable LoRA (current policy)
  old      — frozen LoRA snapshot used for rollout
"""

from __future__ import annotations

from typing import Iterable, Literal
import hashlib

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
        # Wrap happens after pipe.to("cuda"); default Linear/buffer construction
        # would otherwise leave adapters on CPU while activations are on CUDA.
        device = base.weight.device
        # Reproduce the legacy BF16 initialization for identical initial
        # sampled policies with the same seed; promote immediately afterward.
        # Later optimizer updates accumulate in FP32 master Parameters.
        init_dtype = base.weight.dtype
        self.lora_A = nn.Linear(in_features, rank, bias=False, device=device, dtype=init_dtype)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=device, dtype=init_dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(dtype=torch.float32)
        self.lora_B.to(dtype=torch.float32)
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
        a = self.lora_A.weight.detach()
        b = self.lora_B.weight.detach()
        if self.old_A.device != a.device or self.old_A.dtype != a.dtype:
            self.register_buffer("old_A", a.clone())
            self.register_buffer("old_B", b.clone())
        else:
            self.old_A.copy_(a)
            self.old_B.copy_(b)

    def _align_adapters(self, x: torch.Tensor) -> None:
        """Validate placement without silently quantizing trainable FP32 parameters."""
        for name, tensor in (
            ("lora_A", self.lora_A.weight),
            ("lora_B", self.lora_B.weight),
            ("old_A", self.old_A),
            ("old_B", self.old_B),
        ):
            if tensor.dtype != torch.float32 or tensor.device != x.device:
                raise RuntimeError(
                    f"{name}: expected FP32 on {x.device}, got {tensor.dtype} on {tensor.device}. "
                    "Move the module by device only (do not cast LoRA to BF16)."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if self.mode == "base":
            return result
        self._align_adapters(x)
        if self.mode == "old":
            # A frozen old policy snapshot must not track the student updates.
            old_A = self.old_A.to(dtype=x.dtype)
            old_B = self.old_B.to(dtype=x.dtype)
            delta = nn.functional.linear(nn.functional.linear(x, old_A), old_B)
        else:
            # Autograd propagates through these temporary BF16 copies to FP32
            # lora_A/B Parameters; the backward and AdamW states stay FP32.
            a_compute = self.lora_A.weight.to(dtype=x.dtype)
            b_compute = self.lora_B.weight.to(dtype=x.dtype)
            hidden = nn.functional.linear(x, a_compute)
            delta = nn.functional.linear(self.dropout(hidden), b_compute)
        return result + delta * self.scaling


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

    def assert_master_fp32(self) -> None:
        """Fail fast rather than silently losing FP32 LoRA precision."""
        for name, layer in zip(self.names, self.layers):
            for field, tensor in (
                ("lora_A", layer.lora_A.weight),
                ("lora_B", layer.lora_B.weight),
                ("old_A", layer.old_A),
                ("old_B", layer.old_B),
            ):
                if tensor.dtype != torch.float32:
                    raise RuntimeError(
                        f"{name}.{field} is {tensor.dtype}; expected FP32 master/snapshot."
                    )

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def _trainable_sha256(self) -> str:
        """Bit-for-bit fingerprint of LoRA master weights, in stable module order."""
        digest = hashlib.sha256()
        for name, layer in zip(self.names, self.layers):
            digest.update(name.encode("utf-8"))
            for tensor in (layer.lora_A.weight, layer.lora_B.weight):
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def assert_rank_consistency(self) -> None:
        """Fail on rank divergence before either rollout or optimizer updates."""
        dist = torch.distributed
        if not dist.is_available() or not dist.is_initialized():
            return
        digests = [None] * dist.get_world_size()
        dist.all_gather_object(digests, self._trainable_sha256())
        if len(set(digests)) != 1:
            raise RuntimeError(f"LoRA rank mismatch after initialization/resume: {digests}")

    def synchronize_initial_parameters(self) -> None:
        """Broadcast rank-0 A/B AFTER staged transformer GPU placement, once."""
        dist = torch.distributed
        if not dist.is_available() or not dist.is_initialized():
            return
        self.assert_master_fp32()
        # Validate names/shapes collectively before tensor broadcasts to avoid
        # silently synchronizing different LoRA submodules on different ranks.
        local_signature = [
            (name, tuple(layer.lora_A.weight.shape), tuple(layer.lora_B.weight.shape))
            for name, layer in zip(self.names, self.layers)
        ]
        signature = [local_signature if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(signature, src=0)
        matches = torch.tensor(
            [int(local_signature == signature[0])],
            dtype=torch.int32,
            device=self.layers[0].lora_A.weight.device,
        )
        dist.all_reduce(matches, op=dist.ReduceOp.MIN)
        if not int(matches.item()):
            raise RuntimeError("Different LoRA module names/shapes across ranks; refusing to broadcast")
        with torch.no_grad():
            for layer in self.layers:
                dist.broadcast(layer.lora_A.weight.detach(), src=0)
                dist.broadcast(layer.lora_B.weight.detach(), src=0)
        # A/B snapshots must be taken AFTER synchronization, not before it.
        self.snapshot_old()
        self.assert_rank_consistency()

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Compact LoRA-only state. Do not serialize frozen base.weight."""
        return {
            name: {
                "lora_A.weight": layer.lora_A.weight.detach(),
                "lora_B.weight": layer.lora_B.weight.detach(),
            }
            for name, layer in zip(self.names, self.layers)
        }

    def load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        """Read both legacy full-layer and compact A/B-only checkpoint states."""
        if set(state) != set(self.names):
            missing = sorted(set(self.names) - set(state))
            extra = sorted(set(state) - set(self.names))
            raise ValueError(f"LoRA module mismatch: missing={missing[:3]}, extra={extra[:3]}")
        with torch.no_grad():
            for name, layer in zip(self.names, self.layers):
                tensors = state[name]
                for key, param in (
                    ("lora_A.weight", layer.lora_A.weight),
                    ("lora_B.weight", layer.lora_B.weight),
                ):
                    if key not in tensors:
                        raise KeyError(f"Missing {name}.{key} in LoRA checkpoint")
                    value = tensors[key]
                    if not isinstance(value, torch.Tensor) or value.shape != param.shape:
                        raise ValueError(
                            f"{name}.{key}: expected shape {tuple(param.shape)}, "
                            f"got {getattr(value, 'shape', None)}"
                        )
                    param.copy_(value.to(device=param.device, dtype=param.dtype))
        # An old-policy snapshot at the start of a new round matches the
        # restored Student. The trainer already refreshes old on each round.
        self.snapshot_old()
        self.assert_master_fp32()

    def save(self, path) -> None:
        from pathlib import Path
        from .io_utils import atomic_torch_save

        path = Path(path)
        atomic_torch_save(
            path,
            {
                "names": self.names,
                "rank": self.rank,
                "alpha": self.alpha,
                "format": "lora_compact_v1",
                "state": self.state_dict(),
            },
        )

    def load(self, path) -> None:
        from pathlib import Path

        payload = torch.load(Path(path), map_location="cpu", weights_only=False, mmap=True)
        if "names" in payload and list(payload["names"]) != self.names:
            raise ValueError("LoRA layer names in checkpoint differ from the active transformer")
        if "rank" in payload and int(payload["rank"]) != self.rank:
            raise ValueError(f"LoRA rank mismatch: checkpoint={payload['rank']}, model={self.rank}")
        self.load_state_dict(payload["state"])
