# YoHoGPT

YoHoGPT is a large language model built only for software engineering, in the spirit of Devin. This repository holds the whole system needed to build and run it:

| Layer | What's here | Where |
|---|---|---|
| **Model** | Llama-style decoder (RMSNorm, RoPE, GQA, SwiGLU, KV cache, FlashAttention via SDPA), sizes from 30M to 34B | `yohogpt/model.py`, `configs/model/` |
| **Tokenizer** | Byte-level BPE with code-aware splitting, plus special tokens for ChatML, fill-in-the-middle (FIM) and tool calls | `yohogpt/tokenizer/` |
| **Training data** | A list of every source with its license and mixture weight, covering pretraining, long context, SFT, DPO and eval decontamination | [`data/sources.yaml`](data/sources.yaml) |
| **Data pipeline** | Download, license filter, quality filters, secret/PII redaction, exact and MinHash dedup, benchmark decontamination, FIM, then tokenize and pack | `yohogpt/data/` |
| **Training** | One trainer for pretraining, long-context, SFT and DPO, with DDP/FSDP, grad accumulation and checkpointing, and resume on multi-node runs. Precision is picked per GPU, the loss is computed in chunks, and there are low-memory optimizers. Includes a planner (`yoho plan`) and a benchmark (`yoho benchmark`) | `yohogpt/train/`, `yohogpt/plan.py`, `configs/train/` |
| **Inference** | KV-cached sampling (temperature, top-p, top-k, repetition penalty, stop strings), streaming, and export to HF format for vLLM | `yohogpt/inference/`, `yohogpt/export_hf.py` |
| **Agent** | A Devin-style loop that plans, calls tools and observes the results. Tools: bash, read/write/edit files, search, finish. Runs in a Docker sandbox | `yohogpt/agent/` |
| **Networking** | OpenAI-compatible REST API with SSE streaming, tool calling and FIM completions. Agent runs over REST, SSE and WebSocket. API keys, rate limiting, Prometheus metrics. nginx, docker-compose and Kubernetes configs. NCCL multi-node training | `yohogpt/server/`, `deploy/`, `scripts/`, [`docs/NETWORKING.md`](docs/NETWORKING.md) |
| **Evaluation** | HumanEval and MBPP pass@k with sandboxed execution, a SWE-bench predictions runner, and generation of execution-verified SFT/DPO data | `yohogpt/eval/` |

The release plan, model sizes and compute budget are in [`docs/RELEASE_PLAN.md`](docs/RELEASE_PLAN.md). The system design is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**Training on a consumer GPU** (GTX 1050 Ti, GTX 1650, RTX 3060, ...): see [`docs/SMALL_GPU.md`](docs/SMALL_GPU.md). `yoho plan --gpu "GTX 1050 Ti" --days 30` tells you what fits on a card and how long it will take. `yoho benchmark` measures the real numbers on your machine.

> **A note on expectations.** The code here is complete and tested end to end. It is not a trained model. To get a YoHoGPT that talks fluently and codes well, you have to run this pipeline at scale. For the 7B release that means about 2T tokens on roughly 256 H100s for about 10 days. Everything you need for that run is in this repo. The smoke test below runs the same pipeline on a CPU at toy scale.

## Quickstart (CPU, ~2 minutes, no downloads)

```bash
pip install -e ".[serve,dev]"
pytest                       # 55 tests: model, tokenizer, data, training, optimizations, server, agent, HF export
scripts/smoke_test.sh        # data -> tokenizer -> pretrain -> SFT -> API server -> chat/FIM/agent
```

## The full pipeline

```bash
pip install -e ".[all]"
yoho sources                                   # print every training source, weight and license

# 1. Data
huggingface-cli login                          # several code datasets are gated: accept their terms first
yoho download --stage all --out data/raw
yoho train-tokenizer --input data/raw/pretrain --out artifacts/tokenizer --vocab-size 49152
yoho prepare --raw data/raw/pretrain/starcoderdata --out data/processed/starcoderdata \
    --tokenizer artifacts/tokenizer/tokenizer.json --eval-files data/eval/humaneval.jsonl --fim-rate 0.5
# ...one `prepare` per source (scripts/release_pipeline.sh loops over all of them)

# 2. Train (each stage resumes automatically from out_dir/latest)
torchrun --nproc_per_node 8 -m yohogpt.train.trainer configs/train/pretrain-1b.yaml     # one node
NNODES=32 MASTER_ADDR=10.0.0.1 scripts/launch_multinode.sh configs/train/pretrain-7b.yaml  # cluster
sbatch --nodes=32 scripts/slurm_train.sbatch configs/train/pretrain-7b.yaml              # SLURM
yoho train configs/train/long-context-7b.yaml
yoho train configs/train/sft-7b.yaml
yoho train configs/train/dpo-7b.yaml
# any field can be overridden: yoho train configs/train/sft-7b.yaml lr=1e-5 max_steps=3000

# 3. Serve
yoho export-hf --checkpoint checkpoints/yoho-7b-dpo --tokenizer artifacts/tokenizer --out artifacts/hf/yoho-7b
YOHO_API_KEYS=$(openssl rand -hex 24) docker compose -f deploy/docker-compose.yml up -d
# or, without vLLM:  yoho serve --backend local --checkpoint checkpoints/yoho-7b-dpo

# 4. Use it
yoho chat --url https://api.yohogpt.example
yoho agent --url https://api.yohogpt.example --repo-url https://github.com/you/project \
    "The /login endpoint returns 500 when the password is empty. Fix it and add a test."

# 5. Evaluate
yoho eval humaneval --upstream http://localhost:8080 -n 20 -k 1 10
yoho eval-swebench --upstream http://localhost:8080 --limit 500
```

## API

Any OpenAI SDK works against the gateway:

```python
from openai import OpenAI
client = OpenAI(base_url="https://api.yohogpt.example/v1", api_key="...")
for chunk in client.chat.completions.create(model="yohogpt", stream=True,
        messages=[{"role": "user", "content": "Write a thread-safe LRU cache in Rust"}]):
    print(chunk.choices[0].delta.content or "", end="")
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat. Supports `stream`, `stop`, `seed` and OpenAI `tools`/`tool_calls` |
| `POST` | `/v1/completions` | Completion. `suffix` turns on fill-in-the-middle for IDE autocomplete |
| `GET` | `/v1/models` | List models |
| `POST` | `/v1/agent/runs` | Start an autonomous run: `{task, repo_url?, files?, max_steps?}` |
| `GET` | `/v1/agent/runs/{id}` | Status, summary, final `git diff`, events |
| `GET` | `/v1/agent/runs/{id}/events` | Live event stream (SSE) |
| `WS` | `/v1/agent/runs/{id}/ws` | Live events out; `{"type":"message"}` or `{"type":"cancel"}` in |
| `POST` | `/v1/agent/runs/{id}/messages` | Send guidance mid-run (like talking to Devin while it works) |
| `POST` | `/v1/agent/runs/{id}/cancel` | Stop a run |
| `GET` | `/health`, `/metrics` | Probes and Prometheus metrics |

## Repository layout

```
configs/model/       yoho-tiny | yoho-small (4 GB GPUs) | yoho-1b | yoho-7b (release 1) | yoho-34b
configs/train/       smoke | pretrain-4gb | pretrain-1b | pretrain-7b | long-context-7b | sft-7b | dpo-7b
data/sources.yaml    the training-data catalogue
deploy/              Dockerfile, sandbox image, docker-compose, nginx, Kubernetes
docs/                ARCHITECTURE, RELEASE_PLAN, NETWORKING, SMALL_GPU
scripts/             smoke test, multi-node launcher, SLURM job, release pipeline
yohogpt/             the Python package (CLI: `yoho`)
tests/               pytest suite (CPU only)
```

## License

MIT, see [LICENSE](LICENSE). Each training dataset has its own license and terms. Review `data/sources.yaml` before you train.
