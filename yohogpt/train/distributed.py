"""Distributed setup helpers (torchrun / SLURM / single process)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def setup_distributed() -> DistInfo:
    """Initialise torch.distributed from torchrun (or SLURM) environment variables.

    Multi-node runs are launched with ``scripts/launch_multinode.sh`` which sets
    MASTER_ADDR/MASTER_PORT and uses the c10d rendezvous; NCCL picks the
    InfiniBand / RoCE fabric if present (see docs/NETWORKING.md).
    """
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        backend = "nccl"
    elif torch.backends.mps.is_available() and world == 1:
        device, backend = torch.device("mps"), "gloo"
    else:
        device, backend = torch.device("cpu"), "gloo"

    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, device_id=device if backend == "nccl" else None)
    return DistInfo(rank, local, world, device)


def cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_mean(value: torch.Tensor, info: DistInfo) -> torch.Tensor:
    if info.enabled:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= info.world_size
    return value
