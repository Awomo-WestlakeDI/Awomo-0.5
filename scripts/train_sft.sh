#!/usr/bin/env bash
# Copyright (c) 2026 Awomo-WAM Team. Apache-2.0.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${NNODES:?Set NNODES=4}"
: "${NODE_RANK:?Set NODE_RANK=0,1,2,3}"
: "${MASTER_ADDR:?Set a reachable training rendezvous address}"
: "${MASTER_PORT:?Set a free rendezvous port}"
if [[ "$NNODES" != 4 || "${NPROC_PER_NODE:-8}" != 8 ]]; then
  echo 'The released reference uses 4 nodes x 8 GPUs; do not silently change its batch or sampler.' >&2
  exit 2
fi
export TOKENIZERS_PARALLELISM=false
exec torchrun --nnodes="$NNODES" --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  "$ROOT/training/robodojo/train.py" "$@"
