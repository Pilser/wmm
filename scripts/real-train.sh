#!/usr/bin/env bash
# Real training run on IICO.
#
# - 5 workers, one per *physical* core (CPUs 0,2,4,6,8); the 6th core
#   (CPUs 10,11) stays fully idle for mailcow / Supabase / n8n.
# - Single-threaded workers: Rayon gives no speedup on this tiny GRU, but
#   separate processes do scale.
# - All workers start from one shared checkpoint, then their weights are
#   averaged at the end (local SGD with one sync point).
set -euo pipefail
cd "$HOME/pilserlabs/wmm"
mkdir -p runs logs

N="${N:-5}"
SPLIT="${SPLIT:-train}"
FILES="${FILES:-0}"          # 0 = all files in the split
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-32}"
SEQ="${SEQ:-256}"
OUT="${OUT:-runs/real.safetensors}"
BASE="runs/base.safetensors"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/train-$STAMP.log"

export RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=2000000000
export MALLOC_TOP_PAD_=67108864

PHYS=(0 2 4 6 8 10)   # first thread of each physical core
CPUS="${PHYS[*]:0:N}"

echo "workers=$N cpus=$CPUS split=$SPLIT files=$FILES epochs=$EPOCHS batch=$BATCH out=$OUT" | tee "$LOG"

BASE_ARGS=("--data" "data/maestro-v3.0.0" "--split" "$SPLIT" "--seq" "$SEQ"
           "--batch" "$BATCH" "--log-every" "0" "--eval-windows" "64")

# 1. One shared seed checkpoint so averaging is meaningful.
./bin/wmm train "${BASE_ARGS[@]}" --files 1 --epochs 1 --max-steps 1 \
    --out "$BASE" >>"$LOG" 2>&1
echo "base checkpoint ready" | tee -a "$LOG"

# 2. Workers, one per physical core.
START=$(date +%s)
i=0
for CPU in "${PHYS[@]:0:N}"; do
  taskset -c "$CPU" ./bin/wmm train "${BASE_ARGS[@]}" \
      --files "$FILES" --epochs "$EPOCHS" --seed "$((i + 1))" \
      --init "$BASE" --save-every 500 \
      --out "runs/w-$i.safetensors" >"logs/w-$i.log" 2>&1 &
  i=$((i + 1))
done
wait
END=$(date +%s)
echo "all $N workers finished in $((END - START)) s" | tee -a "$LOG"

# 3. Average the weights of exactly these N workers (never the stale w-* files
#    left by older runs).
INPUTS=()
for i in $(seq 0 $((N - 1))); do
  INPUTS+=("runs/w-$i.safetensors")
done
./bin/wmm merge --inputs "${INPUTS[@]}" --out "$OUT" | tee -a "$LOG"
