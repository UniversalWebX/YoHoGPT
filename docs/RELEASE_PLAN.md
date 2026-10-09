# YoHoGPT-1 release plan

## Goal

The first release should do three things:

1. **Talk.** Hold a fluent, multi-turn technical conversation: explain code, discuss designs, ask clarifying questions.
2. **Code well.** Write correct code across the major languages and do fill-in-the-middle completion in an IDE.
3. **Act as an agent.** Work through a repository on its own with the tool loop: read, edit, run tests, debug, finish.

The release model is **YoHoGPT-1 (7B)**. A 1B model is trained first to run ablations cheaply and to test the infrastructure.

## Model sizes

| Config | Params | Layers | d_model | Heads (KV) | Context | Role |
|---|---|---|---|---|---|---|
| `yoho-tiny` | 30M | 4 | 256 | 8 (4) | 2k | CI and smoke tests |
| `yoho-1b` | 1.2B | 24 | 2048 | 16 (8) | 8k | Ablations of the data mix and hyperparameters |
| **`yoho-7b`** | **7.4B** | 32 | 4096 | 32 (8) | 16k | **YoHoGPT-1** |
| `yoho-34b` | 34B | 60 | 7168 | 56 (8) | 16k | YoHoGPT-2 flagship |

## Training stages for YoHoGPT-1

| # | Stage | Data | Tokens | Seq len | LR | Config |
|---|---|---|---|---|---|---|
| 1 | Pretrain | About 85% code, 15% NL and math (`sources.yaml: pretrain`), FIM on 50% of code docs | 2T | 4k | 3e-4, WSD schedule | `pretrain-7b.yaml` |
| 2 | Long context | Repository-level packing (`<|repo_name|>`, `<|file_sep|>`) | 50B | 16k | 1e-4 | `long-context-7b.yaml` |
| 3 | SFT | Code instructions, human chat, agent trajectories, self-play | about 2B × 3 epochs | 16k | 2e-5 | `sft-7b.yaml` |
| 4 | DPO | Execution-verified (pass, fail) pairs plus general preferences | about 0.4B | 4k | 5e-7, β=0.1 | `dpo-7b.yaml` |

Self-improvement loop, which runs between stages 3 and 4 and again for every release after that:

`yoho eval mbpp --problems <train problems> -n 8 --emit-sft ... --emit-pairs ...`

This samples many solutions per problem and runs them against the tests. Passing solutions become SFT data (rejection sampling). (pass, fail) pairs become DPO data. Successful agent trajectories from `yoho agent` runs on synthetic tasks feed `self_play.jsonl`.

## Compute budget

Training FLOPs ≈ 6 · N · D. The estimates assume H100s at about 40% MFU (~400 TFLOP/s effective per GPU):

| Run | N | D | FLOPs | GPU-hours | Wall-clock |
|---|---|---|---|---|---|
| 1B ablation | 1.2B | 1T | 7.4e21 | ~5k | ~3.5 days on 64 GPUs |
| 7B pretrain | 7.4B | 2T | 8.9e22 | ~62k | ~10 days on 256 GPUs |
| 7B long context | 7.4B | 50B | ~3e21 | ~3k | ~1 day on 128 GPUs |
| 7B SFT + DPO | 7.4B | ~7B | ~3e20 | ~0.5k | hours |

**Total for release 1 is about 70k H100-hours**, or roughly $150–250k at current cloud prices, with headroom for ablations and restarts. Storage: about 30 TB raw, about 8 TB tokenized.

## Release gates

These are targets, not guarantees. They depend on data quality and on the ablations run with the 1B model.

| Eval | What it measures | Gate for YoHoGPT-1 |
|---|---|---|
| HumanEval+ / MBPP+ pass@1 | Function-level correctness | Competitive with open 7B code models |
| MultiPL-E | Correctness across languages | No language far below the Python result |
| SWE-bench Verified | Fixing real repository issues as an agent | Non-trivial resolve rate; tracked release over release |
| Held-out chat (LLM-judged) | Conversation quality | Preferred over the SFT-only checkpoint |
| Safety red-team | Secret exfiltration, destructive commands, prompt injection from files | Zero critical findings |

`yoho eval` and `yoho eval-swebench` produce the numbers. Every training set is n-gram decontaminated against all eval sets in `sources.yaml: eval`.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Data licensing | Per-file license allowlist, gated-dataset terms, the license recorded on every record, and legal review of `sources.yaml` |
| Memorized secrets or PII | Regex redaction in `filters.redact`, plus near-dedup, which also lowers memorization |
| Benchmark contamination | 13-gram decontamination against every eval set before tokenization |
| Training instability at 7B | bf16 with fp32 reductions, grad clipping, scaled residual init, WSD schedule (roll back to any stable checkpoint and re-anneal), and loss spikes watched in W&B |
| Node failures | Atomic checkpoints every 2k steps, `--requeue` and `torchrun --max_restarts`, deterministic data order on resume |
| Agent misuse | Sandbox per run with `--network none`, dropped capabilities, resource limits; API keys; per-tenant isolation of runs |

## Faster alternative

If time to market matters more than training from scratch, stages 2–4 of this pipeline could run on top of a permissively licensed open-weights Llama-architecture base instead of a YoHoGPT pretrain. That cuts compute by about 10×. It needs two extra steps: an import script, which is the reverse of `export_hf.py` because the parameter names already match, and adopting the base model's tokenizer with YoHoGPT's special tokens added. YoHoGPT-1 as planned here is trained from scratch.
