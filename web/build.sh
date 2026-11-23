#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
python=${PYTHON:-.venv/bin/python}
checkpoint=${1:-checkpoints/calibrated.pt}
espeak=${ESPEAK_ROOT:-$root/.tools/espeak-ng}
revision=4870adfa25b1a32b4361592f1be8a40337c58d6c
ort_version=1.20.1
assets=$root/web/public/assets
work=$(mktemp -d "${TMPDIR:-/tmp}/tiny-voca-web.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
export ESPEAK_DATA_PATH="$espeak/share"
export PATH="$espeak/bin:$PATH" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
test -d "$ESPEAK_DATA_PATH/espeak-ng-data"
"$espeak/bin/espeak-ng" --version | grep -E '(^| )1\.52\.0( |$)'
mkdir -p "$work/stage/ort" "$work/espeak" "$work/ort"

# Match the browser demo's ONNX Runtime WASM distribution, served locally.
curl -fsSL "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-$ort_version.tgz" -o "$work/ort.tar.gz"
curl -fsSL "https://codeload.github.com/espeak-ng/espeak-ng/tar.gz/$revision" -o "$work/espeak-ng-source.tar.gz"
"$python" - "$work" <<'PY'
from base64 import b64encode
from hashlib import file_digest
from pathlib import Path
import sys

root = Path(sys.argv[1])
for name, algorithm, expected in (
    ('ort.tar.gz', 'sha512', 'TePF6XVpLL1rWVMIl5Y9ACBQcyCNFThZON/jgElNd9Txb73CIEGlklhYR3UEr1cp5r0rbGI6nDwwrs79g7WjoA=='),
    ('espeak-ng-source.tar.gz', 'sha256', 'cd83f84c4e495f281ac14e919aecf2834306ec1ea1b498de8ef6000f3a0f90de'),
):
    with (root / name).open('rb') as f:
        digest = file_digest(f, algorithm)
    actual = b64encode(digest.digest()).decode() if algorithm == 'sha512' else digest.hexdigest()
    if actual != expected:
        raise SystemExit(f'Checksum mismatch: {name}')
PY
tar -xzf "$work/ort.tar.gz" -C "$work/ort"
cp "$work/ort/package/dist/"{ort.wasm.min.mjs,ort-wasm-simd-threaded.mjs,ort-wasm-simd-threaded.wasm} "$work/stage/ort/"
for notice in LICENSE ThirdPartyNotices.txt; do
  curl -fsSL "https://cdn.jsdelivr.net/gh/microsoft/onnxruntime@v$ort_version/$notice" -o "$work/stage/ort/$notice"
done
tar -xzf "$work/espeak-ng-source.tar.gz" -C "$work/espeak" --strip-components=1
# Emscripten's wchar.h declares the functions that the upstream shim remaps.
# Include it first so the remapping never reaches the system declarations.
"$python" - "$work/espeak/src/include/compat/wctype.h" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]); text = p.read_text()
anchor = '#include <ucd/ucd.h>'
assert text.count(anchor) == 1
p.write_text(text.replace(anchor, '#ifdef __EMSCRIPTEN__\n#include <wchar.h>\n#endif\n' + anchor))
PY
emcmake cmake -S "$work/espeak" -B "$work/build" -DCMAKE_BUILD_TYPE=Release \
  -DUSE_ASYNC=OFF -DUSE_MBROLA=OFF -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF
cmake --build "$work/build" --target espeak-ng --parallel "${JOBS:-4}"
# Keep Dutch and the English/French voices the Dutch dictionary switches to.
# The common phoneme tables are compiled as a set and must stay together.
pack="$work/browser-data/espeak-ng-data"
mkdir -p "$pack/lang/gmw" "$pack/lang/roa"
(
  cd "$ESPEAK_DATA_PATH/espeak-ng-data"
  cp intonations phondata phonindex phontab nl_dict en_dict fr_dict "$pack/"
  cp lang/gmw/{nl,en,en-US} "$pack/lang/gmw/"
  cp lang/roa/fr "$pack/lang/roa/"
)
emcc -O3 -I"$work/espeak/src/include" -c "$root/web/espeak.c" -o "$work/espeak-adapter.o"
em++ "$work/espeak-adapter.o" \
  "$work/build/src/libespeak-ng/libespeak-ng.a" "$work/build/src/speechPlayer/libspeechPlayer.a" \
  "$work/build/src/ucd-tools/libucd.a" -O3 --no-entry \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH=1 \
  -sEXPORTED_RUNTIME_METHODS=ccall,FS --preload-file "$pack@/espeak-ng-data" \
  -o "$work/stage/espeak.mjs"
cp "$work/espeak/"COPYING* "$work/stage/"
cp "$work/espeak-ng-source.tar.gz" "$work/stage/"

"$python" -m tiny_voca.export --checkpoint "$checkpoint" --output "$work/stage/model.onnx"
# Do not replace an existing local build until all steps have succeeded.
mkdir -p "$assets"
cp -R "$work/stage/." "$assets/"
printf '\nWeb assets built in %s\n' "$assets"
