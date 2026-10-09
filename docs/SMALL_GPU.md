# Training YoHoGPT on a small GPU (GTX 1050 Ti, GTX 1650, RTX 3060, ...)

## What to expect

The training budget is set by the hardware, and the hardware limits what you can get out of it:

| GPU | Memory | Best precision | Model `yoho plan` picks for a 30-day budget | Time to train it compute-optimally (20 tokens/param) |
|---|---|---|---|---|
| GTX 1050 Ti | 4 GB | fp32 (Pascal fp16 runs at 1/64 speed) | ~100M | ~25 days (2B tokens) |
| GTX 1650 Super | 4 GB | fp16 | ~100M | ~6 days (2B tokens) |
| RTX 3060 | 12 GB | bf16 | ~300M | ~18 days (6B tokens) |
| RTX 4090 | 24 GB | bf16 | ~700M | ~11 days (14B tokens) |

These are `yoho plan` estimates at 40% MFU. Run `yoho benchmark` to measure your own card.

A 100M model trained on 2B tokens is roughly GPT-2-small class, trained on code:

- It **can** autocomplete lines and short functions, with fill-in-the-middle in an IDE.
- It **cannot** reliably hold a technical conversation or run as an autonomous agent. Those abilities show up at several billion parameters and hundreds of billions to trillions of tokens. See `RELEASE_PLAN.md`.

## Steps

```bash
# 1. Ask the planner (or let it detect your GPU with --gpu auto)
yoho plan --gpu "GTX 1050 Ti" --days 30 --write configs/generated

# 2. Get a few billion tokens of high-value data instead of everything
yoho download --stage pretrain_small --out data/raw --limit 2000000
yoho train-tokenizer --input data/raw/pretrain_small --out artifacts/tokenizer-32k --vocab-size 32768 \
    --max-chars 2000000000
for src in data/raw/pretrain_small/*; do
  yoho prepare --raw "$src" --out "data/processed/$(basename $src)" \
      --tokenizer artifacts/tokenizer-32k/tokenizer.json --fim-rate 0.5
done

# 3. Measure on the real card and find the largest micro-batch that fits
yoho benchmark configs/train/pretrain-4gb.yaml --find-batch

# 4. Train. Ctrl-C or a reboot is fine: rerun the same command and it resumes from the last checkpoint.
yoho train configs/train/pretrain-4gb.yaml micro_batch=<from step 3>
```

## Optimizations the trainer applies

| Optimization | Effect | Setting |
|---|---|---|
| Precision picked automatically for the GPU | Pascal cards get fp32 instead of slow fp16 or unsupported bf16. Turing gets fp16 with loss scaling. Ampere and newer get bf16 | `precision: auto` |
| Chunked cross-entropy | The full tokens × vocab logits never exist at once. Measured: peak 2.3 GB → 0.8 GB at 4k tokens × 32k vocab, with the same loss | `loss_chunk_tokens: 1024` |
| Gradient checkpointing | Recomputes activations in backward. Costs about 25% more compute and frees most activation memory | `grad_checkpointing: true` |
| Low-memory optimizers | Optimizer state per parameter: AdamW 8 B, Lion 4 B, 8-bit AdamW 2 B, Adafactor ~0 B | `optimizer: adamw\|lion\|adamw8bit\|adafactor` |
| Smaller vocabulary | A 32k vocab instead of 49k cuts embedding and logit memory by a third | `--vocab-size 32768` + `yoho-small.yaml` |
| Tied embeddings | Input and output embeddings share weights: −25M params at 100M scale | `tie_embeddings: true` |
| Background batch prefetch | The GPU never waits on data loading | `prefetch_batches: 4` |
| Gradient accumulation | A large effective batch (256k tokens) from tiny micro-batches, which keeps training stable | `global_batch_tokens` |
| `torch.compile` guard | Turned off automatically on GPUs older than sm_70, where Triton can't run | `compile: true` on RTX cards |
| Frequent, atomic checkpoints | Safe to stop and resume on a desktop | `ckpt_interval: 250` |

If you hit out-of-memory, try these in order: lower `micro_batch` (gradient accumulation keeps the effective batch the same), then `grad_checkpointing: true`, then `optimizer: adafactor`, then `seq_len: 512`, then a smaller model from `yoho plan`.

## Getting more for the same compute

1. **Data quality beats data quantity at this scale.** The `pretrain_small` mix is weighted toward Python, Q&A and educational text, and it is deduplicated and filtered.
2. **Train past "compute-optimal" if you have the time.** A small model keeps improving well beyond 20 tokens per parameter, and a small model is cheap to run afterwards.
3. **Use the rest for SFT on your own use case.** A small model fine-tuned on a narrow domain, such as one framework or your codebase's style, goes further than a general one.
4. **Rent for the big run, develop locally.** Use the local GPU for debugging and ablations, then run the same configs on rented A100s or H100s. Nothing in the code changes.
