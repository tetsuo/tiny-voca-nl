# Training tiny-voca-nl

Follow the [installation instructions](README.md#installation) and run the commands from the repository root. The examples use PyTorch MPS on Apple Silicon.

## Train a new model

The [Hugging Face dataset](https://huggingface.co/datasets/xxy998/tiny-voca-nl) contains the audio, text, and phoneme timings. Use it to train from random weights without generating new audio or downloading a checkpoint.

```bash
uv pip install --python .venv/bin/python -e '.[prepare]'

corpus=assets/dataset
.venv/bin/hf download xxy998/tiny-voca-nl --repo-type dataset --local-dir "$corpus"
mkdir -p data/annotations
cp "$corpus"/annotations/{annotations.json,train.jsonl,validation.jsonl,test.jsonl} data/annotations/

for split in train validation test; do
  .venv/bin/python -m tiny_voca.prepare --config config.json --split "$split" --workers 5 \
    --inputs "$corpus"/data/"$split"-*.parquet --output data/features
done
```

Preparation computes the spectrograms and pitch, voicing, and energy features. The trainer uses the downloaded annotations for phoneme durations.

```bash
run=runs/dutch
train_args=(
  --config config.json --device mps --output "$run"
  --train data/annotations/train.jsonl --validation data/annotations/validation.jsonl
  --batch-frames 2400 --max-batch 10 --validation-size 24
  --lr-half-life 60000 --reuse-wave-gradient --max-seconds 86400
)
.venv/bin/python -m tiny_voca.train "${train_args[@]}" --steps 20000
```

The trainer saves `last.pt` for resuming and `inference.pt` for synthesis. Both are written under the chosen run directory.

## Generate a corpus with Piper

To build the training corpus from text instead of using the published audio, install eSpeak NG 1.52.0 and the following dependencies. The scripts look for eSpeak in `.tools/espeak-ng/` or on `PATH`.

```bash
uv pip install --python .venv/bin/python -e '.[prepare,evaluate]' 'piper-tts[alignment]==1.8.0' \
  'onnxruntime==1.32.0' 'faster-whisper==1.2.1' 'ctranslate2==4.9.0'

mkdir -p assets/voice-datasets
curl -fLsS https://github.com/OHF-Voice/voice-datasets/releases/download/v1.0.0/nl_NL-alex.zip \
  -o assets/voice-datasets/nl_NL-alex.zip
.venv/bin/hf download rhasspy/piper-voices \
  --revision e929968c3761f1c281875787685a745863db838d \
  --include 'nl/nl_NL/alex/medium/nl_NL-alex-medium.onnx*' --local-dir assets/piper-voices
.venv/bin/hf download Systran/faster-whisper-small config.json model.bin tokenizer.json vocabulary.txt \
  --revision 536b0662742c02347bc0e980a01041f333bce120 --local-dir assets/faster-whisper-small
```

The recipe combines OHF Voice transcripts, pinned [Dutch Common Voice text](https://github.com/common-voice/common-voice/tree/9f55dafbb562ee6b4a1987432052d219d43890c4/server/data/nl), and [project prompts](prompts.txt). It splits by normalized text before synthesis, converts Piper's phoneme timings to 256-sample frames, and uses Whisper to check selected readings.

```bash
NL_RUN=runs/smoke ./scripts/run.sh pilot  # optional two-step check
NL_RUN=runs/corpus ./scripts/run.sh scaled-corpus --extended
NL_RUN=runs/corpus ./scripts/scale.sh train --steps 20000 --reuse-wave-gradient
```

This writes annotations under `runs/corpus/scaled/corpus/annotations/` and checkpoints under `runs/corpus/scaled/student/`.

## Resume a previous run

Use the run's full training checkpoint (`last.pt` or a numbered `step-*.pt`), the same training settings, and the same dataset and annotation manifests. `inference.pt` and the downloadable model weights do not contain optimizer state and cannot resume a run.

For the first example above, rerun the training command with `--resume` and a higher **total** step target:

```bash
.venv/bin/python -m tiny_voca.train "${train_args[@]}" --resume "$run/last.pt" --steps 25000
```

For a run started with `scale.sh train`, use the same `NL_RUN` and training options, adding `--resume` with that run's `scaled/student/last.pt` and increasing `--steps`.

## Fit the excitation correction

Fit the correction to an inference export, not a training checkpoint. The fit uses the training split and leaves the model weights unchanged. For the first example:

```bash
uv pip install --python .venv/bin/python 'scipy==1.18.1'
.venv/bin/python scripts/calibrate_excitation.py fit \
  --checkpoint "$run/inference.pt" --annotations data/annotations \
  --train-count 512 --fit-device mps --output "$run/calibrated.pt"
```

For a Piper recipe run, substitute its `scaled/student/inference.pt` and `scaled/corpus/annotations/`. Use the resulting checkpoint with the [inference command](README.md#synthesize-speech). The original inference export stays unchanged.
