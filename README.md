# tiny-voca-nl

Dutch text-to-speech based on [tiny-voca](https://github.com/n4ze3m/tiny-voca). It uses a roughly 4.5-million-parameter source-filter model to produce 24 kHz mono audio in a single `nl-NL` voice. eSpeak NG handles pronunciation; the model generates the audio offline on CPU. The [browser demo](https://huggingface.co/spaces/xxy998/tiny-voca-nl) runs the same calibrated model with ONNX Runtime Web and eSpeak NG compiled to WebAssembly.

The model was trained from scratch on speech synthesized by the Alex voice from [Piper Voices](https://huggingface.co/rhasspy/piper-voices/tree/e929968c3761f1c281875787685a745863db838d/nl/nl_NL/alex/medium). A separate fit adjusts the renderer's pulse-to-noise mix to reduce audible buzz without changing the base model weights. Some utterances can still sound mechanical.

## Installation

From the repository root, with `uv` and Python 3.12 available:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . 'torch==2.15.0'
```

## Synthesize speech

Download `tiny-voca-nl.pt` from the [Hugging Face model repo](https://huggingface.co/xxy998/tiny-voca-nl). Install [eSpeak NG 1.52.0](https://github.com/espeak-ng/espeak-ng/tree/4870adfa25b1a32b4361592f1be8a40337c58d6c) with Dutch language data. For the launcher below, place the executable and data at `.tools/espeak-ng/bin/espeak-ng` and `.tools/espeak-ng/share/espeak-ng-data/`, respectively.

```sh
./scripts/speak.sh --checkpoint /path/to/tiny-voca-nl.pt \
  --text 'De trein naar Utrecht vertrekt over vijf minuten.' --output speech.wav
```

If eSpeak is installed elsewhere, put it on `PATH` and invoke `.venv/bin/tiny-voca-nl-speak --device cpu` directly with the same checkpoint, text, and output arguments. Piper and the training corpus are not needed for inference.

## Training

The [Hugging Face dataset](https://huggingface.co/datasets/xxy998/tiny-voca-nl) contains the Dutch audio, text, and phoneme annotations. See [TRAINING.md](TRAINING.md) for the recipe.

## Regression tests

```sh
PATH="$PWD/.tools/espeak-ng/bin:$PATH" ESPEAK_DATA_PATH="$PWD/.tools/espeak-ng/share" \
  OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
```

Tests don't require model weights or training data. To enable the optional preparation and ONNX Runtime checks, install the preparation extra and ONNX dependencies:

```sh
uv pip install --python .venv/bin/python -e '.[prepare]' 'onnx==1.23.2' 'onnxruntime==1.32.0'
```

## Browser demo

See [web/README.md](web/README.md) to build and serve the demo locally.

## License and attribution

- Project code is Apache-2.0; see [LICENSE](LICENSE). It is based on upstream [tiny-voca at `903f940`](https://github.com/n4ze3m/tiny-voca/tree/903f940120a939f3fe8ee92a38096c2aa3ee59b3).
- Training prompts come from the [OHF Voice Alex transcripts](https://github.com/OHF-Voice/voice-datasets/releases/tag/v1.0.0), Dutch [Common Voice text](https://github.com/common-voice/common-voice/tree/9f55dafbb562ee6b4a1987432052d219d43890c4/server/data/nl), and [project-authored sentences](prompts.txt).
- eSpeak NG is GPL-3.0-or-later. See the [browser documentation](web/README.md) for third-party components used by the web build.
