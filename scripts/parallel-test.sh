#!/usr/bin/env bash
# Throughput test: N independent workers, one CPU each.
# Each worker is deliberately single-threaded (RAYON_NUM_THREADS=1) so the
# measurement is "one core per process", not SMT sharing.
set -euo pipefail
cd "$HOME/pilserlabs/wmm"
mkdir -p runs logs

N="${N:-4}"
STEPS="${STEPS:-20}"

for i in $(seq 0 $((N - 1))); do
  (
    export RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1
    /usr/bin/time -f "W$i wall=%e user=%U sys=%S cpu=%P maxrss_kb=%M" \
      taskset -c "$i" ./bin/wmm train \
        --data data/maestro-v3.0.0 --files 60 --epochs 10 --seq 256 \
        --batch 32 --max-steps "$STEPS" --log-every "$STEPS" --seed "$((i + 1))" \
        --out "runs/par-$i.safetensors"
  ) 2>&1 | grep -E "wall=|events/s" &
done
wait
echo "ALL_DONE workers=$N steps=$STEPS"
