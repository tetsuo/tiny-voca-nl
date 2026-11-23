#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ ]]; then
  echo 'Usage: scripts/resume.sh TOTAL_STEPS (outputs to NL_OUT or runs/alex)' >&2
  exit 2
fi
cd "$(dirname "$0")/.."
out=${NL_OUT:-runs/alex}
if [[ ! -x .venv/bin/python ]]; then
  echo 'Create .venv first; see README.md.' >&2
  exit 1
fi
checkpoint=checkpoints/step-00030000.pt
if [[ -f "$out/last.pt" ]]; then
  checkpoint="$out/last.pt"
fi
export PATH="$PWD/.tools/espeak-ng/bin:$PATH" ESPEAK_DATA_PATH="$PWD/.tools/espeak-ng/share"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
exec .venv/bin/python -u -m tiny_voca.train \
  --resume "$checkpoint" --steps "$1" --output "$out" --device mps \
  --config checkpoints/config.json --train data/annotations/train.jsonl \
  --validation data/annotations/validation.jsonl \
  --batch-frames 2400 --max-batch 10 --learning-rate .00018 --warmup 500 --lr-half-life 25000 \
  --teacher-hold 3000 --teacher-ramp 7000 --teacher-final 0 \
  --gan-start 8000 --gan-ramp 3000 --gan-ratio .7 --crop-samples 8192 --crops 2 \
  --validation-size 24 --seed 20261113 --reuse-wave-gradient \
  --log-every 100 --save-every 1000 --keep-every 5000 --max-seconds 86400
