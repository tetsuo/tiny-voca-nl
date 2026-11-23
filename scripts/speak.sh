#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
export PATH="$root/.tools/espeak-ng/bin:$PATH"
export ESPEAK_DATA_PATH="$root/.tools/espeak-ng/share"
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
if [[ ! -x "$root/.venv/bin/python" ]]; then
  echo 'Create .venv first; see README.md.' >&2
  exit 1
fi
exec "$root/.venv/bin/python" -m tiny_voca.infer \
  --checkpoint "$root/checkpoints/calibrated.pt" --device cpu --voice alex --frontend espeak "$@"
