#!/usr/bin/env bash
# Continue training from an existing checkpoint on IICO.
#
# Same physical-core pinning as real-train.sh, but starts from the previous
# run's merged weights (runs/real.safetensors) instead of a fresh seed, and
# bounds the run to ~15 minutes (max-minutes). Weights are averaged at the end.
set -euo pipefail
cd "$HOME/pilserlabs/wmm"
mkdir -p runs logs

N="${N:-5}"
SPLIT="${SPLIT:-train}"
FILES="${FILES:-0}"
EPOCHS="${EPOCHS:-4}"          # cap; max-minutes will end it first
MINUTES="${MINUTES:-14}"
BATCH="${BATCH:-32}"
SEQ="${SEQ:-256}"
INIT="${INIT:-runs/real.safetensors}"
OUT="${OUT:-runs/continue.safetensors}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/continue-$STAMP.log"

export RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=2000000000
export MALLOC_TOP_PAD_=67108864

PHYS=(0 2 4 6 8 10)
CPUS="${PHYS[*]:0:N}"
echo "workers=$N cpus=$CPUS init=$INIT minutes=$MINUTES out=$OUT" | tee "$LOG"

BASE_ARGS=("--data" "data/maestro-v3.0.0" "--split" "$SPLIT" "--seq" "$SEQ"
           "--batch" "$BATCH" "--max-minutes" "$MINUTES" "--log-every" "25"
           "--eval-windows" "64")

# Workers all continue from the same merged checkpoint.
START=$(date +%s)
i=0
for CPU in "${PHYS[@]:0:N}"; do
  taskset -c "$CPU" ./bin/wmm train "${BASE_ARGS[@]}" \
      --files "$FILES" --epochs "$EPOCHS" --seed "$((i + 1))" \
      --init "$INIT" --save-every 200 \
      --out "runs/c-$i.safetensors" >"logs/c-$i.log" 2>&1 &
  i=$((i + 1))
done
wait
END=$(date +%s)
echo "all $N workers finished in $((END - START)) s" | tee -a "$LOG"

INPUTS=()
for i in $(seq 0 $((N - 1))); do
  INPUTS+=("runs/c-$i.safetensors")
done
./bin/wmm merge --inputs "${INPUTS[@]}" --out "$OUT" | tee -a "$LOG"
