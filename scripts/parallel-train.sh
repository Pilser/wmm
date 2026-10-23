#!/usr/bin/env bash
# Data-parallel training on IICO: N single-threaded workers, one CPU each,
# then average their weights.
#
# Why: one process is CPU-bound at ~10.7k events/s and Rayon gives no speedup
# (the GRU's per-step matrix products are too small). Independent processes do
# scale, so we use local SGD with one synchronisation point (weight averaging).
#
# CPU budget: workers are pinned to CPUs 0..N-1. With N=10 on a 12-thread
# machine, two CPUs stay free for mailcow / Supabase / n8n (~17%). Use N=11 to
# leave exactly one CPU (~8%).
set -euo pipefail
cd "$HOME/pilserlabs/wmm"
mkdir -p runs logs

N="${N:-10}"
FILES="${FILES:-40}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-32}"
OUT="${OUT:-runs/small.safetensors}"
BASE="runs/base.safetensors"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/parallel-$STAMP.log"

# One thread per worker; tuned allocator (see scripts/thread-sweep.sh).
export RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=2000000000
export MALLOC_TOP_PAD_=67108864

echo "workers=$N files=$FILES epochs=$EPOCHS batch=$BATCH out=$OUT" | tee "$LOG"

# 1. One shared initial checkpoint so that averaging is meaningful.
./bin/wmm train --data data/maestro-v3.0.0 --files 5 --epochs 1 --seq 256 \
    --batch 32 --max-steps 1 --eval-windows 16 --log-every 0 \
    --out "$BASE" >>"$LOG" 2>&1
echo "base checkpoint ready" | tee -a "$LOG"

# 2. Workers, one CPU each.
START=$(date +%s)
for i in $(seq 0 $((N - 1))); do
  taskset -c "$i" ./bin/wmm train --data data/maestro-v3.0.0 \
      --files "$FILES" --epochs "$EPOCHS" --seq 256 --batch "$BATCH" \
      --seed "$((i + 1))" --init "$BASE" --log-every 0 --eval-windows 64 \
      --out "runs/w-$i.safetensors" >"logs/w-$i.log" 2>&1 &
done
wait
END=$(date +%s)
echo "all $N workers finished in $((END - START)) s" | tee -a "$LOG"

# 3. Average the weights of exactly these N workers (never stale w-* files).
INPUTS=()
for i in $(seq 0 $((N - 1))); do
  INPUTS+=("runs/w-$i.safetensors")
done
./bin/wmm merge --inputs "${INPUTS[@]}" --out "$OUT" | tee -a "$LOG"

# 4. Per-worker summaries.
for i in $(seq 0 $((N - 1))); do
  echo "--- worker $i ---" >>"$LOG"
  grep -E "steps |loss " "logs/w-$i.log" >>"$LOG" || true
done
tail -n 8 "$LOG"
