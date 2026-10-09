#!/usr/bin/env bash
# RoboCasa365 evaluation with N workers (worker i -> policy server port 18000+i, rendering on GPU i % NUM_GPUS).
# Usage: bash scripts/launch_robocasa365.sh <num_workers> <output_dir> [num_gpus]
# Env: TASK_SET (target50), NUM_TRIALS (30), TASKS (optional, space separated). Re-running resumes.
set -euo pipefail
NUM=${1:?num workers}
OUT=${2:?output dir}
GPUS=${3:-$NUM}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EVAL="$ROOT/eval_robocasa365/dynamic_eval.py"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
python "$EVAL" init --out "$OUT" --task-set "${TASK_SET:-target50}" --trials "${NUM_TRIALS:-30}" ${TASKS:+--tasks $TASKS}
mkdir -p "$OUT/logs"
PIDS=()
stop_workers() {  # on exit or Ctrl-C, do not leave workers running in the background
  if [ ${#PIDS[@]} -gt 0 ]; then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap stop_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for ((i = 0; i < NUM; i++)); do
  MUJOCO_EGL_DEVICE_ID=$((i % GPUS)) python -u "$EVAL" worker --out "$OUT" --port $((18000 + i)) \
    > "$OUT/logs/worker_$i.log" 2>&1 &
  PIDS+=($!)
done
FAILED=0
for ((i = 0; i < NUM; i++)); do
  if ! wait "${PIDS[$i]}"; then
    echo "worker $i failed: $(tail -1 "$OUT/logs/worker_$i.log")" >&2
    FAILED=1
  fi
done
if [ "$FAILED" -ne 0 ]; then
  exit 2
fi
python "$EVAL" merge --out "$OUT"
