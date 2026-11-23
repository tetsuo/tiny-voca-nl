# Browser demo

[Try tiny-voca-nl on Hugging Face Spaces](https://huggingface.co/spaces/xxy998/tiny-voca-nl). The demo uses eSpeak NG and ONNX Runtime Web to synthesize Dutch speech in the browser. The model and runtime assets total about 35 MB uncompressed.

## Build locally

Follow the [Python setup](../README.md#installation) and download `tiny-voca-nl.pt` from [Hugging Face](https://huggingface.co/xxy998/tiny-voca-nl). The build requires Emscripten, CMake, curl, tar, and eSpeak NG 1.52.0. Set `ESPEAK_ROOT` to the eSpeak installation prefix if it isn't at `.tools/espeak-ng/`.

From the repository root:

```sh
uv pip install --python .venv/bin/python 'onnx==1.23.2'
./web/build.sh /path/to/tiny-voca-nl.pt
python3 -m http.server 8000 --bind 127.0.0.1 --directory web/public
```

Open <http://127.0.0.1:8000>. The build writes generated files to `web/public/assets/`.

## Verify the export

With Node.js installed, compare browser phonemization with desktop eSpeak and ONNX audio with PyTorch:

```sh
uv pip install --python .venv/bin/python 'onnxruntime==1.32.0'
.venv/bin/python web/verify.py --checkpoint /path/to/tiny-voca-nl.pt
```

## Credits

Playback code comes from [tiny-voca](https://github.com/n4ze3m/tiny-voca/tree/1e91b449ab9c418db2e0a2134a22202cd0dffcc5/web) (Apache-2.0). The demo uses [eSpeak NG 1.52.0](https://github.com/espeak-ng/espeak-ng/tree/4870adfa25b1a32b4361592f1be8a40337c58d6c) (GPL-3.0-or-later) and [ONNX Runtime Web 1.20.1](https://www.npmjs.com/package/onnxruntime-web/v/1.20.1) (MIT).
