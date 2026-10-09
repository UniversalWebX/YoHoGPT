# YoHoGPT networking

YoHoGPT has two networks. The **serving** network carries user traffic. The **training** network carries GPU-to-GPU gradient traffic.

## 1. Serving

```
 clients (SDKs, IDE plugins, `yoho chat`, `yoho agent`)
      │  HTTPS / HTTP2, SSE token streams, WebSocket agent sessions
      ▼
 ┌──────────────┐   TLS termination, per-IP limits, no buffering for SSE,
 │ nginx/Ingress│   WebSocket upgrade, 1h timeouts for agent sessions
 └──────┬───────┘
        │ HTTP/1.1 keep-alive
   ┌────┴─────────────────────────┐
   ▼                              ▼
 ┌──────────────┐          ┌──────────────┐  per-run sandbox containers
 │ yoho-gateway │          │  yoho-agent  │──► (docker, --network none,
 │ (stateless,  │          │  (stateful)  │     cap-drop ALL, cpu/mem/pid limits)
 │  autoscaled) │          └──────┬───────┘
 └──────┬───────┘                 │
        │   OpenAI-compatible HTTP, internal API key
        ▼                         ▼
 ┌──────────────────────────────────────┐
 │  vLLM pool (GPU): continuous batching,│  ClusterIP only; NetworkPolicy allows
 │  PagedAttention, prefix caching       │  ingress only from gateway/agent pods
 └──────────────────────────────────────┘
```

**Gateway** (`yohogpt/server/app.py`, FastAPI + uvicorn):

- **Auth.** Bearer API keys from `YOHO_API_KEYS`. Only SHA-256 hashes are kept in memory, and comparison is constant-time. WebSocket clients may pass `?api_key=` because browsers can't set headers on a WebSocket.
- **Rate limits.** A token bucket per API key (`YOHO_RATE_LIMIT` requests/min), plus a per-IP limit in nginx. `429` responses carry `Retry-After`.
- **Backpressure.** `YOHO_MAX_GENERATIONS` caps concurrent generations per replica. `YOHO_MAX_RUNS` caps concurrent agent runs.
- **Streaming.** OpenAI-format SSE (`data: {...}\n\n`, ending in `data: [DONE]`). Responses carry `X-Accel-Buffering: no` so proxies don't buffer tokens.
- **Tenancy.** Each agent run belongs to the API key that created it. Other keys get `404`.
- **Observability.** `/metrics` (Prometheus), an `X-Request-Id` on every response, and JSON access logs in nginx.

**Backends** (`yohogpt/inference/backends.py`):

- `openai` (production). The gateway forwards to vLLM, TGI or SGLang serving the exported HF checkpoint (`yoho export-hf`). That gives continuous batching and high throughput. Scale out by adding vLLM replicas behind the `yoho-vllm` Service.
- `local`. In-process PyTorch, one sequence at a time per replica. For development, or for small deployments without vLLM.

**Ports.** 443 is the only port exposed publicly (nginx/Ingress). 8080 is the gateway and is internal. 8000 is vLLM and is reachable only from inside the backend network.

**Agent sandboxes.** Containers start with `--network none` by default, so code the model runs can't exfiltrate data or reach internal services. When a task needs dependencies, pre-install them in the sandbox image (`deploy/Dockerfile.sandbox`). Alternatively, run a sandbox with network access on an egress-filtered network that only allows your package mirrors.

## 2. Training

Multi-node training uses `torchrun` with a c10d rendezvous (`scripts/launch_multinode.sh`, `scripts/slurm_train.sbatch`):

- **Rendezvous and control.** TCP to `MASTER_ADDR:MASTER_PORT` (default 29500). Low bandwidth.
- **Collectives.** NCCL. DDP uses all-reduce of gradients. FSDP uses all-gather of parameters plus reduce-scatter of gradients.
  - Inside a node, NVLink/NVSwitch is used automatically.
  - Across nodes, InfiniBand or RoCE v2 with GPUDirect RDMA. Set `NCCL_IB_HCA` (e.g. `mlx5`), `NCCL_IB_GID_INDEX=3` for RoCE v2, and `NCCL_SOCKET_IFNAME` for the bootstrap interface.
  - Plan for at least 400 Gb/s per node (8×H100 nodes usually have 8×400 Gb/s). FSDP at 7B moves about 3× the model size per step per GPU.
- **Fault tolerance.** `--max_restarts 3` in torchrun and `--requeue` in SLURM. Checkpoints are published atomically (`.tmp-step-*`, then a rename, then `latest`). Data order is deterministic on `(seed, rank, step)`, so a restarted job carries on exactly where it stopped.
- **Data loading.** Token shards are memory-mapped from shared storage (Lustre, NFS or object-store FUSE). Random window reads are small, and the OS page cache absorbs most of them.

Before a long run, check the fabric:

```bash
# all_reduce_perf from nccl-tests on 2 nodes; expect ~bus bandwidth close to NIC line rate
mpirun -np 16 -H node1:8,node2:8 ./build/all_reduce_perf -b 8M -e 4G -f 2 -g 1
NCCL_DEBUG=INFO scripts/launch_multinode.sh configs/train/pretrain-1b.yaml max_steps=20   # look for "NET/IB"
```
