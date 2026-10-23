#!/usr/bin/env bash
# Download and unpack MAESTRO v3.0.0 (MIDI only) on the IICO machine.
set -euo pipefail
cd "$HOME/pilserlabs/wmm"
mkdir -p data logs

URL="https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
ZIP="data/maestro-v3.0.0-midi.zip"

echo "[$(date -u +%FT%TZ)] downloading $URL"
rm -f "$ZIP"
curl -L --fail --retry 5 --retry-delay 5 --progress-bar -o "$ZIP" "$URL"
ls -lh "$ZIP"

echo "[$(date -u +%FT%TZ)] unzipping"
unzip -q -o "$ZIP" -d data/

echo "[$(date -u +%FT%TZ)] verifying"
COUNT=$(find data/maestro-v3.0.0 -type f \( -name '*.midi' -o -name '*.mid' \) | wc -l)
echo "midi files: $COUNT"
du -sh data/maestro-v3.0.0
ls data/maestro-v3.0.0 | head

if [ "$COUNT" -ne 1276 ]; then
  echo "ERROR: expected 1276 MIDI files, found $COUNT" >&2
  exit 1
fi
echo "[$(date -u +%FT%TZ)] DONE: dataset ready"
