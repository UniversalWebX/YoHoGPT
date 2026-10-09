#!/usr/bin/env bash
# Launch a multi-node training run with torchrun's c10d rendezvous.
# Run this same command on every node:
#
#   NNODES=32 MASTER_ADDR=10.0.0.1 scripts/launch_multinode.sh configs/train/pretrain-7b.yaml
#
# Node ranks are assigned by the rendezvous, so the command is identical on every node.
# Networking notes (see docs/NETWORKING.md):
#   * MASTER_ADDR:MASTER_PORT must be reachable from all nodes (TCP, rendezvous only).
#   * Gradient traffic goes over NCCL. With InfiniBand/RoCE, NCCL uses GPUDirect RDMA
#     automatically; set NCCL_IB_HCA to the HCAs to use and NCCL_SOCKET_IFNAME to the
#     interface for NCCL's bootstrap/TCP fallback.
set -euo pipefail
CONFIG=${1:?usage: launch_multinode.sh CONFIG [overrides...]}
shift
: "${NNODES:?set NNODES}" "${MASTER_ADDR:?set MASTER_ADDR}"
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}       # RoCE v2
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-PHB}
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

exec torchrun \
  --nnodes "$NNODES" --nproc_per_node "$GPUS_PER_NODE" \
  --rdzv_backend c10d --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" --rdzv_id "${RUN_ID:-yoho}" \
  --max_restarts 3 \
  -m yohogpt.train.trainer "$CONFIG" "$@"
