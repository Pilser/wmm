#!/usr/bin/env bash
# Hourly training loop on IICO.
#
# Each hour:
#   1. Train 5 workers (one per physical core, CPUs 0,2,4,6,8) for 60 min,
#      starting from the previous hour's merged weights.
#   2. Merge the workers into runs/hour{N}.safetensors.
#   3. Sample 2 MIDI files (t=0.6, seeds 1 and 2) into runs/midi/.
#   4. Log everything to logs/hourly-progress.log and continue to the next
#      hour. The loop is resumable: if it is restarted, it finds the newest
#      hour checkpoint and continues from there.
#
# Set MAX_HOURS=0 to run forever (default) or a positive number to cap it.
set -euo pipefail
cd "$HOME/pilserlabs/wmm"
mkdir -p runs/midi logs

export RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1
export MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=2000000000
export MALLOC_TOP_PAD_=67108864

N="${N:-5}"
MINUTES="${MINUTES:-60}"
FIRST_INIT="${FIRST_INIT:-runs/hour.safetensors}"
MAX_HOURS="${MAX_HOURS:-0}"
TEMP="${TEMP:-0.6}"
SEEDS="${SEEDS:-1 2}"

PHYS=(0 2 4 6 8 10)   # first thread of each physical core

latest_hour() {
  ls runs/hour[0-9]*.safetensors 2>/dev/null | sort -V | tail -1 || true
}

LAST=$(latest_hour)
INIT="${LAST:-$FIRST_INIT}"
HOUR=1
if [ -n "$LAST" ]; then
  HOUR=$(( $(basename "$LAST" .safetensors | tr -dc '0-9') + 1 ))
fi

echo "[loop] resume-init=$INIT start-hour=$HOUR max-hours=$MAX_HOURS workers=$N minutes=$MINUTES $(date -u +%FT%TZ)" | tee -a logs/hourly-progress.log

while true; do
  OUT="runs/hour${HOUR}.safetensors"
  echo "[hour $HOUR] START $(date -u +%FT%TZ) init=$INIT" | tee -a logs/hourly-progress.log

  # --- train N workers for ~MINUTES minutes, one per physical core ---
  START=$(date +%s)
  i=0
  for CPU in "${PHYS[@]:0:N}"; do
    taskset -c "$CPU" ./bin/wmm train \
        --data data/maestro-v3.0.0 --split train --files 0 --seq 256 --batch 32 \
        --epochs 8 --max-minutes "$MINUTES" --log-every 25 --eval-windows 64 \
        --seed "$((i + 1))" --init "$INIT" --save-every 500 \
        --out "runs/h$HOUR-w$i.safetensors" >"logs/h$HOUR-w$i.log" 2>&1 &
    i=$((i + 1))
  done
  wait
  END=$(date +%s)

  # --- merge this hour's workers ---
  INPUTS=()
  for i in $(seq 0 $((N - 1))); do
    INPUTS+=("runs/h$HOUR-w$i.safetensors")
  done
  ./bin/wmm merge --inputs "${INPUTS[@]}" --out "$OUT" >>logs/hourly-progress.log 2>&1
  echo "[hour $HOUR] merged in $((END - START)) s -> $OUT" | tee -a logs/hourly-progress.log

  # --- sample a couple of MIDI files from this hour's checkpoint ---
  for s in $SEEDS; do
    ./bin/wmm sample --checkpoint "$OUT" --events 800 --temperature "$TEMP" \
        --seed "$s" --out "runs/midi/hour${HOUR}_s${s}.mid" \
        >>logs/hourly-progress.log 2>&1
  done
  echo "[hour $HOUR] samples: runs/midi/hour${HOUR}_s*.mid" | tee -a logs/hourly-progress.log

  INIT="$OUT"
  HOUR=$((HOUR + 1))
  if [ "$MAX_HOURS" -ne 0 ] && [ "$HOUR" -gt "$MAX_HOURS" ]; then
    echo "[loop] reached MAX_HOURS=$MAX_HOURS, stopping" | tee -a logs/hourly-progress.log
    break
  fi
done
