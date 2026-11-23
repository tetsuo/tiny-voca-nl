#!/usr/bin/env bash
set -euo pipefail

workspace=$(cd "$(dirname "$0")/.." && pwd)
cd "$workspace"
run=${NL_RUN:-runs/alex}
dir="$run/scaled/corpus"
cfg="$workspace/config.json"
py=${NL_PYTHON:-.venv/bin/python}
if [[ -x "$workspace/.tools/espeak-ng/bin/espeak-ng" ]]; then
  export PATH="$workspace/.tools/espeak-ng/bin:$PATH"
  export ESPEAK_DATA_PATH="${ESPEAK_DATA_PATH:-$workspace/.tools/espeak-ng/share}"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1

case "${1:-}" in
  prepare)
    for split in train validation test; do
      "$py" -m tiny_voca.prepare --inputs "$dir/source/$split.jsonl" --split "$split" \
        --config "$cfg" --output "$dir/features" --resume --workers "${NL_PREPARE_WORKERS:-5}"
    done
    "$py" -m tiny_voca.dutch_corpus audit --inputs "$dir/source" --asr-model "${NL_ASR:-$workspace/assets/faster-whisper-small}" \
      --output "$dir/reading-check.json"
    "$py" -m tiny_voca.dutch_corpus annotate --inputs "$dir/source" --features "$dir/features" \
      --output "$dir/annotations" --audit "$dir/reading-check.json"
    ;;
  train)
    shift
    exec "$py" -u -m tiny_voca.train --config "$cfg" --train "$dir/annotations/train.jsonl" \
      --validation "$dir/annotations/validation.jsonl" --output "$run/scaled/student" --device mps \
      --batch-frames 2400 --max-batch 10 --learning-rate 0.0003 --warmup 1000 --lr-half-life 60000 \
      --teacher-hold 3000 --teacher-ramp 7000 --teacher-final 0 --gan-start 8000 --gan-ramp 3000 \
      --gan-ratio 0.7 --crop-samples 8192 --crops 2 --validation-size 24 --log-every 100 \
      --save-every 1000 --keep-every 5000 --seed 20261113 --max-seconds 86400 --steps 25000 "$@"
    ;;
  *)
    echo 'Usage: scale.sh prepare|train' >&2
    exit 2
    ;;
esac
