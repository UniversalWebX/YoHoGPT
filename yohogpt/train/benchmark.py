"""Measure real training speed and memory for a config on this machine.

    yoho benchmark configs/train/pretrain-4gb.yaml                # tokens/s, peak memory, ETA
    yoho benchmark configs/train/pretrain-4gb.yaml --find-batch   # largest micro_batch that fits
    yoho benchmark configs/train/pretrain-4gb.yaml optimizer=lion grad_checkpointing=false

Runs a few full optimiser steps on random tokens (no dataset needed) with exactly
the model, precision, optimiser and checkpointing settings of the config.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
import time

import torch

from ..config import ModelConfig, TrainConfig, parse_overrides
from ..model import YohoGPT
from .distributed import DistInfo
from .optim import make_optimizer
from .trainer import peak_flops, resolve_runtime


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_benchmark(cfg: TrainConfig, steps: int = 10, warmup: int = 3, micro_batch: int | None = None) -> dict:
    info = DistInfo(device=_device())
    dtype = resolve_runtime(cfg, info, log=False)
    mb = micro_batch or cfg.micro_batch
    mcfg = ModelConfig.load(cfg.model)
    if cfg.seq_len > mcfg.max_seq_len:
        mcfg.max_seq_len = cfg.seq_len
    model = YohoGPT(mcfg).to(info.device)
    model.grad_checkpointing = cfg.grad_checkpointing
    fwd = torch.compile(model) if cfg.compile else model
    opt = make_optimizer(cfg.optimizer, model, cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2), info.device.type)
    use_amp = dtype != torch.float32
    autocast = (lambda: torch.autocast(info.device.type, dtype=dtype)) if use_amp else contextlib.nullcontext
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16 and info.device.type == "cuda")
    cuda = info.device.type == "cuda"
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(info.device)

    x = torch.randint(0, mcfg.vocab_size, (mb, cfg.seq_len), device=info.device)

    def step() -> float:
        with autocast():
            _, loss = fwd(x, x, return_logits=False, loss_chunk=cfg.loss_chunk_tokens)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        return loss.item()

    for _ in range(warmup):
        step()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps

    tps = mb * cfg.seq_len / dt
    total_tokens = cfg.max_steps * cfg.global_batch_tokens
    peak = peak_flops(info, cfg.precision)
    fpt = model.flops_per_token(cfg.seq_len) + (2 * model.num_params() if cfg.grad_checkpointing else 0)
    out = {
        "device": torch.cuda.get_device_name(info.device) if cuda else str(info.device),
        "params_M": round(model.num_params() / 1e6, 1),
        "precision": cfg.precision,
        "optimizer": cfg.optimizer,
        "grad_checkpointing": cfg.grad_checkpointing,
        "micro_batch": mb,
        "seq_len": cfg.seq_len,
        "step_time_s": round(dt, 4),
        "tokens_per_s": round(tps, 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated(info.device) / 1e9, 3) if cuda else None,
        "mfu": round(fpt * tps / peak, 3) if peak else None,
        "run_tokens": total_tokens,
        "eta_hours": round(total_tokens / tps / 3600, 3),
        "eta_days": round(total_tokens / tps / 86400, 2),
    }
    del model, opt, fwd
    if cuda:
        torch.cuda.empty_cache()
    return out


def find_batch(cfg: TrainConfig, limit: int = 256) -> dict:
    """Double micro_batch until CUDA OOM; return the best fitting result."""
    if not torch.cuda.is_available():
        raise SystemExit("--find-batch needs a CUDA GPU (CPU memory is not the bottleneck)")
    best, mb = None, 1
    while mb <= limit:
        try:
            best = run_benchmark(copy.deepcopy(cfg), steps=3, warmup=1, micro_batch=mb)
            print(json.dumps(best), file=sys.stderr)
            mb *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    if best is None:
        raise SystemExit("micro_batch=1 does not fit: enable grad_checkpointing, use optimizer=adafactor, "
                         "lower seq_len or pick a smaller model (`yoho plan`)")
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="yoho benchmark", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--find-batch", action="store_true")
    args = ap.parse_args(argv)
    cfg = TrainConfig.load(args.config, parse_overrides(args.overrides))
    res = find_batch(cfg) if args.find_batch else run_benchmark(cfg, args.steps)
    print(json.dumps(res, indent=2))
    if args.find_batch:
        print(f"\nset micro_batch: {res['micro_batch']} in {args.config} (keep global_batch_tokens; "
              "gradient accumulation makes up the difference)")


if __name__ == "__main__":
    main()
