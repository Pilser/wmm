#!/usr/bin/env bash
# Small but real training run on IICO (smoke test at scale).
#
# CPU budget: 11 of 12 logical CPUs, nice 10. One core stays free for
# mailcow / Supabase / n8n.
set -euo pipefail
cd "$HOME/pilserlabs/wmm"
mkdir -p runs logs

export RAYON_NUM_THREADS="${THREADS:-11}"
export CANDLE_NUM_THREADS="$RAYON_NUM_THREADS"
CPUS="${CPUS:-0-10}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/train-small-$STAMP.log"
echo "threads=$RAYON_NUM_THREADS cpus=$CPUS log=$LOG"

taskset -c "$CPUS" nice -n 10 ./bin/wmm train \
  --data data/maestro-v3.0.0 \
  --files "${FILES:-40}" \
  --epochs "${EPOCHS:-2}" \
  --batch "${BATCH:-256}" \
  --seq 256 \
  --save-every "${SAVE_EVERY:-100}" \
  --log-every 10 \
  --max-minutes "${MAX_MINUTES:-30}" \
  --out runs/small.safetensors 2>&1 | tee "$LOG"
