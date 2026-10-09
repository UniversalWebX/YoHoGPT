"""One training loop for every stage: pretrain, long-context, SFT and DPO.

Single GPU / CPU:
    yoho train configs/train/pretrain-1b.yaml
Multi-GPU / multi-node:
    torchrun --nproc_per_node 8 -m yohogpt.train.trainer configs/train/pretrain-1b.yaml
    scripts/launch_multinode.sh configs/train/pretrain-7b.yaml   (see docs/NETWORKING.md)

Any config field can be overridden: ``... max_steps=100 micro_batch=2``.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import queue
import sys
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import hardware
from ..config import ModelConfig, TrainConfig, parse_overrides
from ..model import Block, YohoGPT
from .checkpoint import latest_checkpoint, resolve_checkpoint, save_checkpoint
from .distributed import DistInfo, all_reduce_mean, cleanup, setup_distributed
from .optim import make_optimizer



def resolve_runtime(cfg: TrainConfig, info: DistInfo, log: bool = True) -> torch.dtype:
    """Adapt the config to the hardware: precision, torch.compile, FSDP availability."""
    say = (lambda msg: print(f"[yoho] {msg}", file=sys.stderr)) if log and info.is_main else (lambda msg: None)
    if cfg.precision == "auto":
        cfg.precision = hardware.best_precision(info.device)
        say(f"precision=auto -> {cfg.precision}")
    if info.device.type != "cuda" and cfg.precision == "fp16":
        cfg.precision = "fp32"
    if info.device.type == "cuda" and cfg.precision == "bf16" and torch.cuda.get_device_capability(info.device) < (8, 0):
        say("bf16 is not supported in hardware on this GPU (needs sm_80+); using "
            f"{hardware.best_precision(info.device)}")
        cfg.precision = hardware.best_precision(info.device)
    if cfg.compile and not hardware.supports_compile(info.device):
        say("torch.compile needs a GPU with compute capability >= 7.0; disabling")
        cfg.compile = False
    if cfg.parallelism == "fsdp" and info.device.type != "cuda":
        say("FSDP needs CUDA; falling back to DDP on this device")
        cfg.parallelism = "ddp"
    if info.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return hardware.dtype_of(cfg.precision)


def peak_flops(info: DistInfo, precision: str) -> float | None:
    env = os.environ.get("YOHO_PEAK_FLOPS")
    return float(env) if env else hardware.peak_flops(info.device, precision)


class Prefetcher:
    """Builds the next batches on a background thread (pinned memory on CUDA) so the
    accelerator never waits on memmap reads / numpy sampling."""

    def __init__(self, get_batch, start_step: int, max_steps: int, grad_accum: int, device: torch.device,
                 depth: int) -> None:
        self.q: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
        pin = device.type == "cuda"

        def work() -> None:
            try:
                for step in range(start_step, max_steps):
                    for micro in range(grad_accum):
                        x, y = get_batch(step, micro)
                        if pin:
                            x, y = x.pin_memory(), y.pin_memory()
                        self.q.put((step, micro, x, y))
            except BaseException as e:  # surface data errors in the training thread
                self.q.put(e)

        threading.Thread(target=work, daemon=True, name="yoho-prefetch").start()

    def __call__(self, step: int, micro: int):
        item = self.q.get()
        if isinstance(item, BaseException):
            raise item
        s, m, x, y = item
        assert (s, m) == (step, micro), f"prefetcher out of sync: got {(s, m)}, expected {(step, micro)}"
        return x, y


# ------------------------------------------------------------------ schedule
def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.decay == "wsd":  # warmup-stable-decay: flat, then linear anneal in the last fraction
        decay_start = int(cfg.max_steps * (1 - cfg.wsd_decay_fraction))
        if step < decay_start:
            return cfg.lr
        frac = (step - decay_start) / max(1, cfg.max_steps - decay_start)
        return cfg.lr + (cfg.min_lr - cfg.lr) * min(1.0, frac)
    frac = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    if cfg.decay == "linear":
        return cfg.lr + (cfg.min_lr - cfg.lr) * frac
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * frac))


# ------------------------------------------------------------------ model setup
def build_model(cfg: TrainConfig, info: DistInfo) -> tuple[YohoGPT, ModelConfig, int, dict]:
    """Create the model, loading weights from ``init_from`` (fine-tuning) and
    returning resume state if ``out_dir`` already has a checkpoint."""
    resume = latest_checkpoint(cfg.out_dir)
    src = resume or (resolve_checkpoint(cfg.init_from) if cfg.init_from else None)
    if src is not None:
        model_cfg = ModelConfig(**json.loads((src / "model_config.json").read_text()))
        if cfg.seq_len > model_cfg.max_seq_len:  # long-context extension stage
            model_cfg.max_seq_len = cfg.seq_len
    else:
        model_cfg = ModelConfig.load(cfg.model)
    model = YohoGPT(model_cfg)
    step, state = 0, {}
    if src is not None:
        sd = torch.load(src / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        if info.is_main:
            print(f"loaded weights from {src}", file=sys.stderr)
    if resume is not None:
        state = json.loads((resume / "train_state.json").read_text())
        step = state["step"]
    return model, model_cfg, step, state


def wrap_model(model: YohoGPT, cfg: TrainConfig, info: DistInfo, dtype: torch.dtype):
    model.grad_checkpointing = cfg.grad_checkpointing
    if cfg.parallelism == "fsdp" and info.enabled:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        mp = MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32, buffer_dtype=dtype) \
            if dtype != torch.float32 else None
        model = FSDP(
            model,
            auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            device_id=info.device if info.device.type == "cuda" else None,
            use_orig_params=True,
            limit_all_gathers=True,
        )
    else:
        model = model.to(info.device)
        if info.enabled:
            from torch.nn.parallel import DistributedDataParallel as DDP

            model = DDP(model, device_ids=[info.local_rank] if info.device.type == "cuda" else None)
    if cfg.compile:
        model = torch.compile(model)
    return model


def _raw(model) -> YohoGPT:
    m = getattr(model, "_orig_mod", model)
    return getattr(m, "module", m)


# ------------------------------------------------------------------ data
def build_data(cfg: TrainConfig, info: DistInfo):
    if cfg.stage in ("pretrain", "long_context"):
        from ..data.dataset import MixtureSampler, PackedTokens

        sources = {m.name: (PackedTokens(m.path), m.weight) for m in cfg.mixture}
        sampler = MixtureSampler(sources, cfg.seq_len, cfg.micro_batch, cfg.seed, info.rank, cfg.val_fraction)

        def get(step: int, micro: int, val: bool = False):
            return sampler.batch(step * 1000 + micro, val)

        return get
    from ..tokenizer import YohoTokenizer

    tok = YohoTokenizer.load(cfg.tokenizer)
    if cfg.stage == "sft":
        from ..data.dataset import SFTDataset

        ds = SFTDataset(cfg.sft_files, tok, cfg.seq_len, cfg.seed, max(cfg.val_fraction, 0.01))

        def get(step: int, micro: int, val: bool = False):
            rng = np.random.default_rng([cfg.seed, info.rank, step, micro, int(val)])
            return ds.batch(rng, cfg.micro_batch, val)

        return get
    if cfg.stage == "dpo":
        from ..data.dataset import PreferenceDataset

        ds = PreferenceDataset(cfg.sft_files, tok, cfg.seq_len)

        def get(step: int, micro: int, val: bool = False):
            rng = np.random.default_rng([cfg.seed, info.rank, step, micro, int(val)])
            return ds.batch(rng, cfg.micro_batch)

        return get
    raise ValueError(f"unknown stage {cfg.stage}")


# ------------------------------------------------------------------ losses
def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels != -100
    safe = labels.clamp(min=0)
    lp = torch.gather(F.log_softmax(logits.float(), dim=-1), 2, safe.unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1)


def dpo_loss(policy, ref, x, y, beta: float) -> tuple[torch.Tensor, dict]:
    logits, _ = policy(x)
    pl = sequence_logps(logits, y)
    with torch.no_grad():
        rlogits, _ = ref(x)
        rl = sequence_logps(rlogits, y)
    b = x.size(0) // 2
    margin = (pl[:b] - rl[:b]) - (pl[b:] - rl[b:])
    loss = -F.logsigmoid(beta * margin).mean()
    return loss, {"reward_acc": (margin > 0).float().mean().item(), "margin": margin.mean().item()}


# ------------------------------------------------------------------ main loop
def train(cfg: TrainConfig) -> dict:
    info = setup_distributed()
    torch.manual_seed(cfg.seed + info.rank)
    dtype = resolve_runtime(cfg, info)

    model, model_cfg, start_step, state = build_model(cfg, info)
    ref_model = None
    if cfg.stage == "dpo":
        ref_model = copy.deepcopy(model).to(info.device, dtype=dtype).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    n_params = model.num_params()
    flops_per_token = model.flops_per_token(cfg.seq_len)
    model = wrap_model(model, cfg, info, dtype)
    fsdp = cfg.parallelism == "fsdp" and info.enabled
    optimizer = make_optimizer(cfg.optimizer, model, cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2),
                               info.device.type, grouped=not fsdp)
    peak = peak_flops(info, cfg.precision)

    resume = latest_checkpoint(cfg.out_dir)
    if resume is not None and (resume / "optim.pt").exists():
        osd = torch.load(resume / "optim.pt", map_location="cpu", weights_only=False)
        if cfg.parallelism == "fsdp" and info.enabled:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            osd = FSDP.optim_state_dict_to_load(model, optimizer, osd)
        optimizer.load_state_dict(osd)

    get_batch = build_data(cfg, info)
    tokens_per_micro = cfg.micro_batch * cfg.seq_len
    grad_accum = max(1, cfg.global_batch_tokens // (tokens_per_micro * info.world_size))
    next_batch = Prefetcher(get_batch, start_step, cfg.max_steps, grad_accum, info.device, cfg.prefetch_batches) \
        if cfg.prefetch_batches > 0 else get_batch
    use_amp = dtype != torch.float32 and not (cfg.parallelism == "fsdp" and info.enabled)
    autocast = (lambda: torch.autocast(info.device.type, dtype=dtype)) if use_amp else contextlib.nullcontext
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16 and info.device.type == "cuda")

    out_dir = Path(cfg.out_dir)
    log_file = None
    wandb_run = None
    if info.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        log_file = open(out_dir / "log.jsonl", "a")
        print(f"[yoho] stage={cfg.stage} params={n_params / 1e6:.1f}M world={info.world_size} "
              f"grad_accum={grad_accum} tokens/step={tokens_per_micro * grad_accum * info.world_size:,} "
              f"device={info.device} precision={cfg.precision} optimizer={cfg.optimizer} "
              f"start_step={start_step}", file=sys.stderr)
        if cfg.wandb_project:
            import wandb

            wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg.to_dict(), resume="allow")

    def compute_loss(x, y):
        if ref_model is not None:
            return dpo_loss(model, ref_model, x, y, cfg.dpo_beta)
        _, loss = model(x, y, return_logits=False, loss_chunk=cfg.loss_chunk_tokens)
        return loss, {}

    @torch.no_grad()
    def evaluate(step: int) -> float:
        model.eval()
        losses = []
        for i in range(cfg.eval_batches):
            x, y = get_batch(step, i, val=True)
            with autocast():
                loss, _ = compute_loss(x.to(info.device), y.to(info.device))
            losses.append(loss.float())
        model.train()
        return all_reduce_mean(torch.stack(losses).mean(), info).item()

    tokens_seen = state.get("tokens_seen", 0)
    last = {"step": start_step}
    model.train()
    t0 = time.time()
    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        total_loss = torch.zeros((), device=info.device)
        extras: dict = {}
        for micro in range(grad_accum):
            x, y = next_batch(step, micro)
            x, y = x.to(info.device, non_blocking=True), y.to(info.device, non_blocking=True)
            sync = micro == grad_accum - 1 or not info.enabled or not hasattr(model, "no_sync")
            ctx = contextlib.nullcontext() if sync else model.no_sync()
            with ctx, autocast():
                loss, extras = compute_loss(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            total_loss += loss.detach()

        scaler.unscale_(optimizer)
        if hasattr(model, "clip_grad_norm_"):  # FSDP
            grad_norm = model.clip_grad_norm_(cfg.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_micro * grad_accum * info.world_size

        if step % cfg.log_interval == 0 or step == cfg.max_steps - 1:
            loss_val = all_reduce_mean(total_loss, info).item()
            dt = time.time() - t0
            t0 = time.time()
            steps_done = cfg.log_interval if step > start_step else 1
            tps = tokens_per_micro * grad_accum * info.world_size * steps_done / max(dt, 1e-9)
            mfu = flops_per_token * tps / (peak * info.world_size) if peak else 0.0
            last = {"step": step, "loss": round(loss_val, 4), "lr": lr, "grad_norm": round(float(grad_norm), 3),
                    "tokens_seen": tokens_seen, "tok_per_s": int(tps), "mfu": round(mfu, 3), **extras}
            if info.is_main:
                print(json.dumps(last), file=sys.stderr)
                log_file.write(json.dumps(last) + "\n")
                log_file.flush()
                if wandb_run:
                    wandb_run.log(last, step=step)

        if cfg.eval_interval and (step + 1) % cfg.eval_interval == 0:
            val = evaluate(step)
            last["val_loss"] = val
            if info.is_main:
                print(json.dumps({"step": step, "val_loss": round(val, 4)}), file=sys.stderr)
                log_file.write(json.dumps({"step": step, "val_loss": val}) + "\n")
                if wandb_run:
                    wandb_run.log({"val_loss": val}, step=step)

        if (step + 1) % cfg.ckpt_interval == 0 or step == cfg.max_steps - 1:
            save_checkpoint(cfg.out_dir, step + 1, model, optimizer, model_cfg,
                            {"tokens_seen": tokens_seen, "config": cfg.to_dict()}, info.is_main, cfg.keep_checkpoints)

    if log_file:
        log_file.close()
    if wandb_run:
        wandb_run.finish()
    cleanup()
    return last


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cfg = TrainConfig.load(argv[0], parse_overrides(argv[1:]))
    train(cfg)


if __name__ == "__main__":
    main()
