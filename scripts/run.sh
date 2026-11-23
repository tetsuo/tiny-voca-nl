#!/usr/bin/env bash
set -euo pipefail

scripts=$(cd "$(dirname "$0")" && pwd)
workspace=$(cd "$scripts/.." && pwd)
cd "$workspace"
run=${NL_RUN:-runs/alex}
asr=${NL_ASR:-$workspace/assets/faster-whisper-small}
py=${NL_PYTHON:-.venv/bin/python}
config="$workspace/config.json"
if [[ -x "$workspace/.tools/espeak-ng/bin/espeak-ng" ]]; then
  export PATH="$workspace/.tools/espeak-ng/bin:$PATH"
  export ESPEAK_DATA_PATH="${ESPEAK_DATA_PATH:-$workspace/.tools/espeak-ng/share}"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1

generate_data() {
  local dir=$1
  shift
  "$py" -m tiny_voca.dutch_corpus generate \
    --archive assets/voice-datasets/nl_NL-alex.zip \
    --model assets/piper-voices/nl/nl_NL/alex/medium/nl_NL-alex-medium.onnx \
    --prompts "$workspace/prompts.txt" --output "$dir/source" "$@"
}

prepare_data() {
  local dir=$1
  shift
  generate_data "$dir" "$@"
  for split in train validation test; do
    "$py" -m tiny_voca.prepare --inputs "$dir/source/$split.jsonl" --split "$split" \
      --config "$config" --output "$dir/features"
  done
}

annotate() {
  local dir=$1
  shift
  "$py" -m tiny_voca.dutch_corpus annotate --inputs "$dir/source" --features "$dir/features" \
    --output "$dir/annotations" "$@"
}

case "${1:-}" in
  reference)
    if [[ $# -ne 3 || -z $2 ]]; then
      echo 'Usage: run.sh reference "Dutch text" path/to/new.wav' >&2
      exit 2
    fi
    # Same local Alex teacher as the listening references, at its native rate.
    printf '%s\n' "$2" | "$py" -m piper --output-raw --volume 0.85 \
      -m assets/piper-voices/nl/nl_NL/alex/medium/nl_NL-alex-medium.onnx \
      | ffmpeg -nostdin -hide_banner -loglevel error -f s16le -ar 22050 -ac 1 -i pipe:0 \
          -ar 24000 -ac 1 -c:a pcm_s16le "$3"
    ;;
  pilot)
    dir="$run/pilot"
    prepare_data "$dir" --pilot
    annotate "$dir"
    "$py" -m tiny_voca.train --config "$config" --train "$dir/annotations/train.jsonl" \
      --validation "$dir/annotations/validation.jsonl" --output "$dir/mps" --steps 2 \
      --device mps --max-batch 2 --batch-frames 800 --validation-size 2 --warmup 10 --log-every 1
    ;;
  corpus)
    dir="$run/corpus"
    prepare_data "$dir"
    "$py" -m tiny_voca.dutch_corpus audit --inputs "$dir/source" --asr-model "$asr" --output "$dir/reading-check.json"
    annotate "$dir" --audit "$dir/reading-check.json"
    ;;
  scaled-corpus)
    dir="$run/scaled/corpus"
    mkdir -p "$dir/text"
    src=https://raw.githubusercontent.com/common-voice/common-voice/9f55dafbb562ee6b4a1987432052d219d43890c4/server/data
    curl -fsSL "$src/nl/sentence-collector.txt" -o "$dir/text/sentence-collector.txt"
    curl -fsSL "$src/LICENSE" -o "$dir/text/LICENSE"
    extra=(--extra-train 6500)
    if [[ "${2:-}" == "--extended" ]]; then
      for file in danielsjf.txt pcmill.txt; do
        curl -fsSL "$src/nl/$file" -o "$dir/text/$file"
      done
      extra=(--extra-train 15000 --extra-text "$dir/text/danielsjf.txt" "$dir/text/pcmill.txt")
    fi
    generate_data "$dir" --common-voice "$dir/text/sentence-collector.txt" "${extra[@]}"
    bash "$scripts/scale.sh" prepare
    ;;
  train)
    shift
    ann="$run/corpus/annotations"
    exec "$py" -u -m tiny_voca.train --config "$config" --train "$ann/train.jsonl" --validation "$ann/validation.jsonl" \
      --output "$run/student" --device mps --batch-frames 1400 --max-batch 6 --learning-rate 0.00045 \
      --warmup 60 --lr-half-life 3000 --teacher-hold 150 --teacher-ramp 650 --teacher-final 0 \
      --gan-start 1400 --gan-ramp 500 --gan-ratio 0.35 --crop-samples 3072 --crops 1 \
      --validation-size 12 --log-every 25 --save-every 250 --keep-every 1000 --seed 20261113 \
      --max-seconds 86400 --steps 5000 "$@"
    ;;
  evaluate)
    shift
    export HF_HUB_OFFLINE=1
    if [[ -f "$run/scaled/corpus/annotations/annotations.json" ]]; then
      checkpoint="$run/scaled/student/inference.pt"
      annotations="$run/scaled/corpus/annotations"
    else
      checkpoint="$run/student/inference.pt"
      annotations="$run/corpus/annotations"
    fi
    exec "$py" -u -m tiny_voca.evaluate --checkpoint "${NL_CHECKPOINT:-$checkpoint}" \
      --annotations "${NL_ANNOTATIONS:-$annotations}" --asr-model "$asr" --device cpu \
      --output "$run/held-out" --tag alex-nl --splits test --per-voice 100 --novel 8 "$@"
    ;;
  tests)
    PYTHONPATH=src:tests "$py" -m unittest discover -s tests
    ;;
  *)
    echo "Usage: $0 {pilot|corpus|scaled-corpus|train|evaluate|tests|reference}" >&2
    exit 2
    ;;
esac
