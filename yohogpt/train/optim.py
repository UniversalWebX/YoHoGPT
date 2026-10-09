"""Optimisers, chosen by memory budget.

Bytes of optimiser state per parameter (fp32 params):
    adamw      8   best quality, default
    adamw8bit  2   needs `pip install bitsandbytes`
    lion       4   use ~3-10x lower lr and ~3x higher weight decay than AdamW
    adafactor  ~0  factored second moments; good when memory is the hard limit
"""

from __future__ import annotations

import torch
import torch.nn as nn

OPTIMIZERS = ("adamw", "adamw8bit", "lion", "adafactor")
STATE_BYTES_PER_PARAM = {"adamw": 8.0, "adamw8bit": 2.0, "lion": 4.0, "adafactor": 0.1}


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al., 2023): sign of an interpolated momentum, one state tensor."""

    def __init__(self, params, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                update = m.mul(beta1).add_(p.grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-group["lr"])
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)
        return loss


def param_groups(model: nn.Module, weight_decay: float, grouped: bool = True) -> list[dict]:
    """Decay matrices/embeddings, not norms. Tied weights are counted once."""
    params, seen = [], set()
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    if not grouped:  # FSDP flat params: dimensionality is not meaningful
        return [{"params": params, "weight_decay": weight_decay}]
    return [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]


def make_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float, betas: tuple[float, float],
                   device_type: str, grouped: bool = True) -> torch.optim.Optimizer:
    groups = param_groups(model, weight_decay, grouped)
    if name == "adamw":
        kwargs = {"fused": True} if device_type == "cuda" else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **kwargs)
    if name == "lion":
        return Lion(groups, lr=lr, betas=betas)
    if name == "adafactor":
        if not hasattr(torch.optim, "Adafactor"):
            raise RuntimeError("optimizer=adafactor needs torch>=2.5")
        return torch.optim.Adafactor(groups, lr=lr)
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise RuntimeError("optimizer=adamw8bit needs `pip install bitsandbytes`") from e
        return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas)
    raise ValueError(f"unknown optimizer {name!r}; choose from {OPTIMIZERS}")
