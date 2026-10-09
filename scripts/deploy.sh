#!/usr/bin/env bash
# Start N policy servers (ports 18000..18000+N-1, server i on GPU i % NUM_GPUS) in the background.
# Usage: bash scripts/deploy.sh <policy module> <checkpoint> <num_servers> [num_gpus]
#   e.g. bash scripts/deploy.sh eval_robocasa365.awomo_05i.policy checkpoints/Awomo-0.5I-RoboCasa365 8
# Extra server flags (e.g. --qwen-path / --vae-path) can be passed via SERVER_ARGS. Stop with: pkill -f deploy/server.py
set -euo pipefail
POLICY=${1:?policy module}
CKPT=${2:?checkpoint dir}
NUM=${3:?num servers}
GPUS=${4:-$NUM}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/logs"
for ((i = 0; i < NUM; i++)); do
  CUDA_VISIBLE_DEVICES=$((i % GPUS)) nohup python -u "$ROOT/deploy/server.py" --policy "$POLICY" --checkpoint "$CKPT" \
    --port $((18000 + i)) ${SERVER_ARGS:-} > "$ROOT/logs/server_$((18000 + i)).log" 2>&1 &
  echo "server $i -> GPU $((i % GPUS)), port $((18000 + i)), log logs/server_$((18000 + i)).log"
done
