"""Checkpoint save / load / rotation.

Layout::

    out_dir/
      latest                 -> text file with the newest step dir name
      step-00001000/
        model.pt             full (unsharded) state dict, loadable on any world size
        optim.pt             optimiser state (full)
        model_config.json
        train_state.json     step, tokens seen, train config
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..model import YohoGPT


def _unwrap(model: nn.Module) -> nn.Module:
    model = getattr(model, "_orig_mod", model)  # torch.compile
    return getattr(model, "module", model)  # DDP


def is_fsdp(model: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:  # pragma: no cover
        return False
    return isinstance(getattr(model, "_orig_mod", model), FSDP)


def gather_state(model: nn.Module, optimizer: torch.optim.Optimizer | None):
    """Return full (model_sd, optim_sd) on rank 0 for DDP or FSDP models."""
    if is_fsdp(model):
        from torch.distributed.fsdp import FullStateDictConfig, FullOptimStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        m = getattr(model, "_orig_mod", model)
        with FSDP.state_dict_type(
            m, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            msd = m.state_dict()
            osd = FSDP.optim_state_dict(m, optimizer) if optimizer is not None else None
        return msd, osd
    msd = _unwrap(model).state_dict()
    return msd, optimizer.state_dict() if optimizer is not None else None


def save_checkpoint(out_dir: str | Path, step: int, model: nn.Module, optimizer, model_cfg: ModelConfig,
                    train_state: dict, is_main: bool, keep: int = 3) -> Path | None:
    msd, osd = gather_state(model, optimizer)
    if not is_main:
        return None
    out_dir = Path(out_dir)
    ckpt = out_dir / f"step-{step:08d}"
    tmp = out_dir / f".tmp-step-{step:08d}"
    tmp.mkdir(parents=True, exist_ok=True)
    torch.save(msd, tmp / "model.pt")
    if osd is not None:
        torch.save(osd, tmp / "optim.pt")
    (tmp / "model_config.json").write_text(json.dumps(model_cfg.to_dict(), indent=2))
    (tmp / "train_state.json").write_text(json.dumps({"step": step, **train_state}, indent=2, default=str))
    if ckpt.exists():
        shutil.rmtree(ckpt)
    tmp.rename(ckpt)  # atomic publish: a crash never leaves a half-written "latest"
    (out_dir / "latest").write_text(ckpt.name)

    steps = sorted(p for p in out_dir.glob("step-*") if p.is_dir())
    for old in steps[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
    return ckpt


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    out_dir = Path(out_dir)
    marker = out_dir / "latest"
    if marker.exists():
        p = out_dir / marker.read_text().strip()
        if p.exists():
            return p
    return None


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept either a run dir (uses ``latest``) or a specific step dir."""
    path = Path(path)
    if (path / "model.pt").exists():
        return path
    latest = latest_checkpoint(path)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint found at {path}")
    return latest


def load_model(path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype | None = None) -> YohoGPT:
    ckpt = resolve_checkpoint(path)
    cfg = ModelConfig(**json.loads((ckpt / "model_config.json").read_text()))
    model = YohoGPT(cfg)
    sd = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    if dtype is not None:
        model = model.to(dtype)
    return model.to(device).eval()
