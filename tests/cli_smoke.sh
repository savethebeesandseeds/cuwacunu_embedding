#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
device="${1:-cpu}"
[[ "$device" == cpu || "$device" == cuda ]] || { echo 'Usage: cli_smoke.sh [cpu|cuda]' >&2; exit 1; }
output="output/smoke-$device"
.build/embedding synthetic --config config/default.conf --output "$output/data.pt" --samples 9
.build/embedding train --config config/default.conf --input "$output/data.pt" \
  --checkpoint "$output/model.pt" --steps 8 --device "$device"
.build/embedding train --resume "$output/model.pt" --input "$output/data.pt" \
  --checkpoint "$output/resumed.pt" --steps 2 --device "$device"
.build/embedding embed --checkpoint "$output/resumed.pt" --input "$output/data.pt" \
  --output "$output/embeddings.pt" --batch-size 4 --device "$device"
if [[ "$device" == cuda ]]; then
  # A CUDA-trained checkpoint must also load and serve on CPU.
  .build/embedding embed --checkpoint "$output/resumed.pt" --input "$output/data.pt" \
    --output "$output/embeddings-cpu.pt" --batch-size 4 --device cpu
fi
printf 'PASS: %s CLI train, resume, and embedding export (%s)\n' "$device" "$output"
