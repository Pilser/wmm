#!/usr/bin/env bash
# Benchmark throughput vs. batch size on IICO.
#
# Leaves ~1 core free: taskset 0-10 = 11 of 12 logical CPUs, nice 10 so the
# other services (mailcow, Supabase, n8n) win ties.
set -euo pipefail
cd "$HOME/pilserlabs/wmm"
mkdir -p runs logs

export RAYON_NUM_THREADS="${THREADS:-11}"
export CANDLE_NUM_THREADS="$RAYON_NUM_THREADS"
CPUS="${CPUS:-0-10}"

echo "threads=$RAYON_NUM_THREADS cpus=$CPUS host=$(hostname) nproc=$(nproc)"

for B in ${BATCHES:-32 128 256}; do
  echo "=== batch $B ==="
  /usr/bin/time -f "RESULT batch=$B wall=%e user=%U sys=%S cpu=%P maxrss_kb=%M" \
    taskset -c "$CPUS" nice -n 10 ./bin/wmm train \
      --data data/maestro-v3.0.0 --files 60 --epochs 10 --seq 256 \
      --batch "$B" --max-steps 20 --log-every 5 \
      --out "runs/bench-b$B.safetensors" 2>&1 | tail -8
done
