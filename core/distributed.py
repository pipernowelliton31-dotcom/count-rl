"""Single-node process group for sharded rollout and synchronized LoRA updates.

Each rank owns one GPU. A GRPO group stays on one rank. After the frozen-policy
rollout, ranks average the LoRA gradients of the groups that actually produced a
loss, then take the same optimizer step. Ranks that skip a slot still enter the
collective, so a missing teacher or a zero advantage cannot deadlock the others.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

_STATE: "Dist | None" = None


@dataclass
class Dist:
    rank: int
    local_rank: int
    world: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world > 1


def current() -> Dist:
    if _STATE is None:
        return Dist(rank=0, local_rank=0, world=1)
    return _STATE


def current_rank() -> int:
    return current().rank


def metrics_filename() -> str:
    state = current()
    if state.world > 1:
        return f"metrics_rank{state.rank}.jsonl"
    return "metrics.jsonl"


def shard(items: Sequence, rank: int, world: int) -> list:
    if world <= 1:
        return list(items)
    return list(items)[rank::world]


def memory_is_enough(total_bytes: int, min_gb: float, slack_gb: float) -> bool:
    """Nominal GB is not GiB. A 40GB A100 reports about 39.6 GiB."""
    floor = (float(min_gb) - float(slack_gb)) * (1024 ** 3)
    return int(total_bytes) >= floor


def init_distributed() -> Dist:
    """Bind this process to LOCAL_RANK. Initialize NCCL only when WORLD_SIZE > 1."""
    global _STATE
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world < 1:
        raise RuntimeError(f"WORLD_SIZE must be positive, got {world}")
    if rank < 0 or rank >= world:
        raise RuntimeError(f"RANK {rank} is outside WORLD_SIZE {world}")
    if world > 1:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU training requires CUDA, but torch.cuda.is_available() is false")
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK {local_rank} but this process sees {torch.cuda.device_count()} GPU(s). "
                "Launch with torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE and do not "
                "also set Slurm --ntasks equal to the GPU count."
            )
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=__import__("datetime").timedelta(minutes=180))
    elif os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            local_rank = 0
    _STATE = Dist(rank=rank, local_rank=local_rank, world=world)
    return _STATE


def assert_launch_budget(cfg: dict) -> Dist:
    state = current()
    runtime_cfg = cfg.get("runtime") or {}
    max_gpus = int(runtime_cfg.get("max_gpus", 4))
    if state.world > max_gpus:
        raise SystemExit(
            f"Refusing to start: WORLD_SIZE={state.world} exceeds runtime.max_gpus={max_gpus}. "
            "This project is capped at one 4-GPU job."
        )
    expected = runtime_cfg.get("num_gpus")
    if expected is not None and int(expected) != state.world:
        raise SystemExit(
            f"Refusing to start: this config requires runtime.num_gpus={int(expected)}, "
            f"but WORLD_SIZE={state.world}. Launch with torchrun --nproc_per_node={int(expected)}, "
            "or use the 1-GPU smoke config. The run will not silently change the GPU count."
        )
    min_gb = runtime_cfg.get("min_gpu_memory_gb")
    if min_gb is None or state.world < 1:
        return state
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("This config requires a GPU, but CUDA is not available.")
    device_index = state.local_rank if state.world > 1 else 0
    props = torch.cuda.get_device_properties(device_index)
    slack = float(runtime_cfg.get("memory_slack_gb", 4))
    if not memory_is_enough(props.total_memory, float(min_gb), slack):
        gib = props.total_memory / (1024 ** 3)
        raise SystemExit(
            f"GPU {props.name} has {gib:.2f} GiB. Config runtime.min_gpu_memory_gb={min_gb} "
            f"allows {slack:g} GB of nominal-vs-GiB slack, and this card is below that floor. "
            "Not switching to another GPU size or a lower resolution."
        )
    return state


def device_string() -> str:
    state = current()
    if state.world > 1:
        return f"cuda:{state.local_rank}"
    return "cuda:0"


def barrier() -> None:
    state = current()
    if state.world <= 1:
        return
    import torch

    torch.distributed.barrier()


def broadcast_object(value):
    state = current()
    if state.world <= 1:
        return value
    import torch

    payload = [value if state.is_main else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def max_int(value: int) -> int:
    state = current()
    if state.world <= 1:
        return int(value)
    import torch

    tensor = torch.tensor([int(value)], device=device_string(), dtype=torch.int64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return int(tensor.item())


def average_gradients(params, participated: bool) -> int:
    """Mean of the ranks that ran backward. Every rank must call this."""
    state = current()
    import torch

    # participated may be a boolean (standard trainer) or the exact number of
    # accumulated groups on this rank (2-GPU virtual 4-group backup).
    if state.world <= 1:
        return int(participated)
    flag = torch.tensor([float(participated)], device=device_string())
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.SUM)
    count = int(flag.item())
    for param in params:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
        if count > 0:
            param.grad.div_(count)
    return count
