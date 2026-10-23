#!/usr/bin/env bash
# Thread-count sweep with tuned malloc (see docs: candle allocates multi-MB
# tensors every timestep; the default glibc mmap threshold causes millions of
# page faults and ~40% system time).
set -euo pipefail
cd "$HOME/pilserlabs/wmm"
mkdir -p runs logs

export MALLOC_MMAP_THRESHOLD_=33554432
export MALLOC_TRIM_THRESHOLD_=2000000000
export MALLOC_TOP_PAD_=67108864

STEPS="${STEPS:-30}"
BATCH="${BATCH:-32}"
FILES="${FILES:-60}"

for T in ${THREADS:-1 4 6 11}; do
  echo "=== threads $T ==="
  RAYON_NUM_THREADS="$T" CANDLE_NUM_THREADS="$T" \
    /usr/bin/time -f "RESULT threads=$T wall=%e user=%U sys=%S cpu=%P maxrss_kb=%M" \
    taskset -c 0-10 nice -n 10 ./bin/wmm train \
      --data data/maestro-v3.0.0 --files "$FILES" --epochs 10 --seq 256 \
      --batch "$BATCH" --max-steps "$STEPS" --log-every "$STEPS" \
      --out "runs/thr-$T.safetensors" 2>&1 | grep -E "wall=|events/s"
done
echo ALL_DONE
