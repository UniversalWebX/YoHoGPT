"""Compute planner: what YoHoGPT can you train on *your* hardware in *your* time?

    yoho plan --gpu "GTX 1050 Ti" --days 30
    yoho plan --gpu auto --hours 48 --write configs/generated     # also writes model+train YAML
    yoho plan --gpu h100 --gpus 256 --days 10

For every candidate model size it estimates peak training memory (weights,
gradients, optimiser state, activations, chunked logits) and throughput
(FLOPs/token / (peak FLOPS x MFU)), then recommends the largest model that fits
in memory and can see ~20 tokens/parameter (Chinchilla compute-optimal) within
the budget. Estimates only - confirm with ``yoho benchmark``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import hardware
from .config import ModelConfig
from .train.optim import STATE_BYTES_PER_PARAM

# Candidate shapes: (label, d_model, n_layers, n_heads, n_kv_heads)
LADDER = [
    ("20M", 384, 6, 6, 2),
    ("50M", 512, 8, 8, 4),
    ("100M", 768, 12, 12, 4),
    ("200M", 1024, 16, 16, 4),
    ("350M", 1024, 24, 16, 4),
    ("750M", 1536, 24, 16, 4),
    ("1.2B", 2048, 24, 16, 8),
    ("3B", 2560, 32, 32, 8),
    ("7B", 4096, 32, 32, 8),
    ("34B", 7168, 60, 56, 8),
]
CUDA_OVERHEAD_GB = 0.5  # context + allocator slack
MEMORY_HEADROOM = 0.9  # plan to 90% of the card; desktop GPUs also drive the display
MIN_TOKENS_WITHOUT_CKPT = 4096
TOKENS_PER_PARAM = 20


@dataclass
class Estimate:
    label: str
    cfg: ModelConfig
    params: int
    memory_gb: float
    micro_batch: int
    grad_ckpt: bool
    fits: bool
    tokens_per_s: float
    budget_tokens: float
    days_compute_optimal: float


def candidate(label: str, d: int, layers: int, heads: int, kv: int, vocab: int, seq_len: int) -> ModelConfig:
    small = d <= 1024
    return ModelConfig(name=f"yoho-{label.lower()}", vocab_size=vocab, d_model=d, n_layers=layers, n_heads=heads,
                       n_kv_heads=kv, max_seq_len=seq_len, multiple_of=64 if small else 256,
                       tie_embeddings=small)


def memory_gb(cfg: ModelConfig, micro_batch: int, seq_len: int, precision: str, optimizer: str,
              grad_ckpt: bool, loss_chunk: int = 2048, shards: int = 1) -> float:
    """Peak training memory per GPU. ``shards`` > 1 models FSDP FULL_SHARD."""
    n = cfg.num_params()
    act_bytes = 4 if precision == "fp32" else 2
    states = n * (4 + 4 + STATE_BYTES_PER_PARAM[optimizer]) / shards  # fp32 weights + grads + optimiser
    tokens = micro_batch * seq_len
    # per-layer activations saved for backward (memory-efficient attention: no T^2 term)
    layer_full = tokens * (10 * cfg.d_model + 4 * cfg.ffn_hidden) * act_bytes
    if grad_ckpt:
        acts = cfg.n_layers * tokens * cfg.d_model * act_bytes + layer_full
    else:
        acts = cfg.n_layers * layer_full
    chunk = min(loss_chunk or tokens, tokens)
    logits = 3 * chunk * cfg.vocab_size * 4  # fp32 logits, softmax, grad
    return (states + acts + logits) / 1e9 * 1.1 + CUDA_OVERHEAD_GB


def flops_per_token(cfg: ModelConfig, seq_len: int, grad_ckpt: bool) -> float:
    n = cfg.num_params()
    attn = 12 * cfg.n_layers * cfg.d_model * seq_len
    return (8 if grad_ckpt else 6) * n + attn  # checkpointing re-runs the forward (+2N)


def plan(spec: hardware.GPUSpec, n_gpus: int, hours: float, precision: str, optimizer: str, vocab: int,
         seq_len: int, mfu: float, memory_gb_override: float | None = None) -> list[Estimate]:
    mem_limit = (memory_gb_override or spec.memory_gb) * MEMORY_HEADROOM
    flops = spec.peak_flops(precision) * mfu * n_gpus
    out = []
    for label, d, layers, heads, kv in LADDER:
        cfg = candidate(label, d, layers, heads, kv, vocab, seq_len)

        def fit(ckpt: bool, min_tokens: int = 0):
            for mb in (32, 16, 8, 4, 2, 1):
                if mb * seq_len < min_tokens:
                    return None
                m = memory_gb(cfg, mb, seq_len, precision, optimizer, ckpt, shards=n_gpus)
                if m <= mem_limit:
                    return mb, ckpt, m
            return None

        # Skip recompute only if a reasonably large micro-batch still fits; tiny batches
        # leave the GPU idle, so checkpointing with a bigger batch is usually faster.
        best = fit(False, MIN_TOKENS_WITHOUT_CKPT) or fit(True) or fit(False)
        fits = best is not None
        mb, ckpt, m = best if fits else (1, True, memory_gb(cfg, 1, seq_len, precision, optimizer, True,
                                                            shards=n_gpus))
        tps = flops / flops_per_token(cfg, seq_len, ckpt)
        n = cfg.num_params()
        out.append(Estimate(label, cfg, n, m, mb, ckpt, fits, tps, tps * hours * 3600,
                            TOKENS_PER_PARAM * n / tps / 86400))
    return out


def recommend(estimates: list[Estimate]) -> Estimate:
    fitting = [e for e in estimates if e.fits]
    if not fitting:
        raise SystemExit("no candidate model fits in memory on this GPU")
    optimal = [e for e in fitting if e.budget_tokens >= TOKENS_PER_PARAM * e.params]
    return optimal[-1] if optimal else fitting[0]


def _fmt_tokens(t: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if t >= div:
            return f"{t / div:.1f}{unit}"
    return f"{t:.0f}"


def _fmt_days(d: float) -> str:
    if d < 1:
        return f"{d * 24:.1f} h"
    if d < 365:
        return f"{d:.0f} d"
    return f"{d / 365:,.0f} yr"


def write_configs(rec: Estimate, out_dir: Path, seq_len: int, precision: str, optimizer: str, hours: float,
                  n_gpus: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{rec.cfg.name}.yaml"
    model_path.write_text(yaml.safe_dump(rec.cfg.to_dict(), sort_keys=False))
    tokens = rec.budget_tokens  # use the whole budget: over-training small models improves them further
    global_batch = max(rec.micro_batch * seq_len * n_gpus, min(262_144 if rec.params < 5e8 else 1_048_576,
                                                               int(tokens / 2000)))
    max_steps = max(100, int(tokens // global_batch))
    from .data.sources import load_catalogue

    weights = load_catalogue().mixture_weights("pretrain_small")
    train = {
        "run_name": f"{rec.cfg.name}-pretrain",
        "stage": "pretrain",
        "model": str(model_path),
        "out_dir": f"checkpoints/{rec.cfg.name}-pretrain",
        "seq_len": seq_len,
        "micro_batch": rec.micro_batch,
        "global_batch_tokens": global_batch,
        "max_steps": max_steps,
        "lr": 3e-4 if optimizer != "lion" else 5e-5,
        "min_lr": 3e-5 if optimizer != "lion" else 5e-6,
        "warmup_steps": min(2000, max(50, max_steps // 50)),
        "decay": "wsd",
        "weight_decay": 0.1 if optimizer != "lion" else 0.3,
        "optimizer": optimizer,
        "precision": precision,
        "grad_checkpointing": rec.grad_ckpt,
        "eval_interval": max(100, max_steps // 50),
        "ckpt_interval": max(100, max_steps // 100),
        "mixture": [{"name": k, "path": f"data/processed/{k}", "weight": round(v, 4)} for k, v in weights.items()],
    }
    train_path = out_dir / f"{rec.cfg.name}-pretrain.yaml"
    header = (f"# Generated by `yoho plan` for {n_gpus} GPU(s), {hours:.0f}h budget.\n"
              f"# ~{_fmt_tokens(max_steps * global_batch)} tokens. Verify speed/memory with `yoho benchmark` first.\n")
    train_path.write_text(header + yaml.safe_dump(train, sort_keys=False))
    return model_path, train_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="yoho plan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="auto", help=f"GPU name or 'auto'. Known: {', '.join(hardware.GPUS)}")
    ap.add_argument("--gpus", type=int, default=1, help="number of GPUs")
    budget = ap.add_mutually_exclusive_group()
    budget.add_argument("--hours", type=float)
    budget.add_argument("--days", type=float)
    ap.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--optimizer", default="adamw", choices=list(STATE_BYTES_PER_PARAM))
    ap.add_argument("--vocab", type=int, default=32768, help="tokenizer size (smaller vocab saves memory)")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--mfu", type=float, default=0.4, help="assumed model-FLOPs utilisation")
    ap.add_argument("--memory-gb", type=float, help="override GPU memory")
    ap.add_argument("--write", help="directory to write the recommended model + train configs")
    args = ap.parse_args(argv)

    if args.gpu == "auto":
        spec = hardware.detect()
        if spec is None:
            raise SystemExit("could not detect a known CUDA GPU; pass --gpu NAME")
    else:
        try:
            spec = hardware.lookup(args.gpu)
        except KeyError as e:
            raise SystemExit(str(e)) from e
    hours = args.hours if args.hours is not None else (args.days or 30) * 24
    precision = args.precision if args.precision != "auto" else hardware.precision_for_capability(spec.capability)
    if precision == "bf16" and spec.bf16_tflops is None:
        raise SystemExit(f"{spec.name} has no bf16 support; use --precision fp16 or fp32")

    ests = plan(spec, args.gpus, hours, precision, args.optimizer, args.vocab, args.seq_len, args.mfu, args.memory_gb)
    rec = recommend(ests)
    peak = spec.peak_flops(precision) / 1e12
    print(f"\n{args.gpus} x {spec.name}: {args.memory_gb or spec.memory_gb:g} GB, {peak:g} TFLOPS {precision} peak, "
          f"MFU {args.mfu:.0%}, budget {_fmt_days(hours / 24)}, optimizer {args.optimizer}, seq {args.seq_len}\n")
    print(f"{'model':>6} {'params':>8} {'mem GB':>7} {'fits':>5} {'batch':>5} {'ckpt':>5} {'tok/s':>9} "
          f"{'tokens in budget':>17} {'time for 20 tok/param':>22}")
    for e in ests:
        mark = "  <== recommended" if e is rec else ""
        print(f"{e.label:>6} {e.params / 1e6:>7.0f}M {e.memory_gb:>7.1f} {'yes' if e.fits else 'NO':>5} "
              f"{e.micro_batch if e.fits else '-':>5} {('on' if e.grad_ckpt else 'off') if e.fits else '-':>5} "
              f"{e.tokens_per_s:>9,.0f} {_fmt_tokens(e.budget_tokens):>17} {_fmt_days(e.days_compute_optimal):>22}{mark}")
    print(f"\nRecommended: {rec.label} ({rec.params / 1e6:.0f}M params), micro_batch={rec.micro_batch}, "
          f"grad_checkpointing={rec.grad_ckpt}. Compute-optimal is ~{_fmt_tokens(TOKENS_PER_PARAM * rec.params)} "
          f"tokens; your budget allows ~{_fmt_tokens(rec.budget_tokens)} (extra tokens still help a small model).")
    if rec.params < 1e9:
        print("Note: models under ~1B parameters can autocomplete code but will not hold strong conversations\n"
              "or work reliably as an agent. See docs/SMALL_GPU.md.")
    if args.write:
        m, t = write_configs(rec, Path(args.write), args.seq_len, precision, args.optimizer, hours, args.gpus)
        print(f"\nwrote {m}\nwrote {t}\nnext: yoho benchmark {t} --find-batch")


if __name__ == "__main__":
    main()
