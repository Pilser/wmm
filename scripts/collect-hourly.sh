#!/usr/bin/env bash
# Local collector: watch IICO's hourly loop and pull everything back as each
# hour finishes.
#
# For every hour checkpoint that exists remotely and is not yet collected:
#   1. Copy checkpoint + JSON + MIDI samples + worker logs into artifacts/iico/.
#   2. Render the first sample to WAV (local SoundFont).
#   3. Append a progress row to artifacts/iico/hourly-progress.md.
#
# Idempotent: completed hours are remembered in artifacts/iico/.collected.
set -euo pipefail
cd "$HOME/AAA-PROJECTS/wmm"

mkdir -p artifacts/iico/runs artifacts/iico/logs artifacts/iico/midi/hourly
MARK=artifacts/iico/.collected
touch "$MARK"

remote_hours() {
  timeout 40 ssh iico 'ls ~/pilserlabs/wmm/runs/hour[0-9]*.json 2>/dev/null | sed -E "s#.*/hour([0-9]+)\.json#\1#" | sort -n' 2>/dev/null || true
}

while :; do
  for H in $(remote_hours); do
    if grep -qx "$H" "$MARK"; then
      continue
    fi
    echo "[collect] hour $H finished remotely; pulling"
    timeout 300 scp -o ConnectTimeout=30 iico:~/pilserlabs/wmm/runs/hour${H}.safetensors iico:~/pilserlabs/wmm/runs/hour${H}.json artifacts/iico/runs/ 2>/dev/null || { echo "[collect] scp checkpoint failed for hour $H"; continue; }
    timeout 120 scp -o ConnectTimeout=30 "iico:~/pilserlabs/wmm/runs/midi/hour${H}_s*.mid" artifacts/iico/midi/hourly/ 2>/dev/null || true
    timeout 120 scp -o ConnectTimeout=30 "iico:~/pilserlabs/wmm/logs/h${H}-w*.log" artifacts/iico/logs/ 2>/dev/null || true

    # Render the seed-1 sample locally.
    MID=$(ls artifacts/iico/midi/hourly/hour${H}_s1.mid 2>/dev/null | head -1)
    if [ -n "$MID" ]; then
      ./target/release/wmm render -i "$MID" -o "${MID%.mid}.wav" >/dev/null 2>&1 || true
      STAT=$(./target/release/wmm stats "$MID" 2>/dev/null | tail -2 | tr '\n' ' ')
      LOSS=$(grep -o '"final_loss": [0-9.e-]*' artifacts/iico/runs/hour${H}.json | head -1)
      echo "| hour $H | $(date -u +%FT%TZ) | $LOSS | $STAT |" >> artifacts/iico/hourly-progress.md
    else
      echo "| hour $H | $(date -u +%FT%TZ) | no sample |" >> artifacts/iico/hourly-progress.md
    fi
    echo "$H" >> "$MARK"
    echo "[collect] hour $H collected"
  done
  sleep 120
done
