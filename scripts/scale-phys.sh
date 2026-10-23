#!/usr/bin/env bash
# Correct scaling test: one worker per *physical* core.
#
# i5-12500: 6 physical cores, 2 SMT threads each. Sibling pairs are
# (0,1),(2,3),(4,5),(6,7),(8,9),(10,11). We pin each worker to the first
# thread of a physical core (0,2,4,6,8) and leave the last physical core
# (10,11) completely idle for mailcow / Supabase / n8n.
set -euo pipefail
cd "$HOME/pilserlabs/wmm"
mkdir -p runs logs

STEPS="${STEPS:-60}"
BATCH="${BATCH:-32}"
FILES="${FILES:-60}"

export RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=2000000000
export MALLOC_TOP_PAD_=67108864

# Physical-core first-thread IDs, in order. Workers take the first $N.
PHYS=(0 2 4 6 8 10)

for N in "${@:-1 2 4 5}"; do
  START=$(date +%s)
  for i in $(seq 0 $((N - 1))); do
    CPU="${PHYS[$i]}"
    (
      taskset -c "$CPU" nice -n 10 ./bin/wmm train --data data/maestro-v3.0.0 \
        --files "$FILES" --epochs 10 --seq 256 --batch "$BATCH" \
        --max-steps "$STEPS" --log-every "$STEPS" --seed "$((i + 1))" \
        --eval-windows 16 --out "runs/phys-$N-$i.safetensors"
    ) >"logs/phys-$N-$i.log" 2>&1 &
  done
  wait
  END=$(date +%s)
  WALL=$((END - START))
  EVENTS=$((N * STEPS * BATCH * 256))
  echo "SCALE_PHYS workers=$N cpus=${PHYS[*]:0:N} wall=${WALL}s aggregate_eps=$((EVENTS / WALL)) per_worker_eps=$((EVENTS / WALL / N))"
done
echo ALL_DONE
