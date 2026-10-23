#!/usr/bin/env bash
# Scaling test: run N single-threaded workers pinned to CPUs 0..N-1 and report
# aggregate events/s. Used to choose the worker count on this machine.
set -euo pipefail
cd "$HOME/pilserlabs/wmm"
mkdir -p runs logs

N="${N:-4}"
STEPS="${STEPS:-20}"
BATCH="${BATCH:-32}"
FILES="${FILES:-40}"

export RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=2000000000
export MALLOC_TOP_PAD_=67108864

START=$(date +%s)
for i in $(seq 0 $((N - 1))); do
  (
    taskset -c "$i" nice -n 10 ./bin/wmm train --data data/maestro-v3.0.0 \
      --files "$FILES" --epochs 10 --seq 256 --batch "$BATCH" \
      --max-steps "$STEPS" --log-every "$STEPS" --seed "$((i + 1))" \
      --eval-windows 16 --out "runs/par-$i.safetensors"
  ) >"logs/par-$i.log" 2>&1 &
done
wait
END=$(date +%s)
WALL=$((END - START))
EVENTS=$((N * STEPS * BATCH * 256))
echo "SCALE workers=$N steps=$STEPS wall=${WALL}s aggregate_eps=$((EVENTS / WALL)) per_worker_eps=$((EVENTS / WALL / N))"
