"""Compare Dutch browser tokens and calibrated ONNX audio with native Python."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
import torch

from tiny_voca import dutch
from tiny_voca.export import INPUTS, export
from tiny_voca.infer import finished, load_model
from tiny_voca.text import frontend_fingerprint

CASES = [
    'De trein naar Utrecht vertrekt over vijf minuten.',
    'Op 29-02-2024 betaalde dr. De Vries € 12,50.',
    'Het is 08:15 uur en buiten is het −3,05 graden.',
    'Gaat de deur open? Ja, de deur gaat open!',
    'Het huis, de trein en de oude boot; alstublieft',
    '“Een naïeve jongen...” ２３ mensen nemen de fiets.',
]
ROOT = Path(__file__).resolve().parent.parent


def frontend():
    expected = []
    for text in CASES:
        parts = dutch.sentences(finished(text))
        phones = [dutch.phonemize(part) for part in parts]
        assert not any(p.unknown for p in phones), text
        expected.append(dict(parts=parts, readings=[dict(tokens=list(p.ids), stress=list(p.stress), symbols=list(p.symbols))
                                                    for p in phones]))
    script = '''
import fs from 'node:fs';
import {loadTrace, createPhonemizer, sentences} from '../dutch.mjs';
const {symbols} = JSON.parse(fs.readFileSync('model.json'));
const phonemize = createPhonemizer(symbols, await loadTrace({}));
const output = JSON.parse(fs.readFileSync(0, 'utf8')).map(text => {
  const parts = sentences(text);
  return {parts, readings: parts.map(phonemize)};
});
console.log(JSON.stringify(output));
'''
    output = subprocess.run(['node', '--input-type=module', '-e', script], input=json.dumps(CASES),
                            cwd=ROOT / 'web/public/assets', text=True, capture_output=True)
    if output.returncode:
        raise RuntimeError(output.stderr)
    for text, actual, wanted in zip(CASES, json.loads(output.stdout), expected, strict=True):
        if actual != wanted:
            raise ValueError(f'Dutch front end mismatch for {text!r}\nBrowser: {actual}\nNative: {wanted}')
    print(f'Browser WASM and desktop phoneme IDs/stress match for {len(CASES)} texts.')


def audio(checkpoint):
    model = load_model(checkpoint)
    c = model.config
    assert c.calibrated_excitation and c.language == 'nl-NL'
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    with TemporaryDirectory() as tmp:
        path = export(model, Path(tmp) / 'parity.onnx', external_noise=True)
        session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        for text, speed, shift, pitch_range in [(CASES[0], 1., 0., 1.), (CASES[4], 1.25, .05, .8), (CASES[2], .9, 0., 1.)]:
            phones = dutch.phonemize(text)
            ids, stress = torch.tensor([phones.ids]), torch.tensor([phones.stress])
            voice, lengths = torch.tensor([0]), torch.tensor([ids.shape[1]])
            reference, size, info = model.synthesize(ids, stress, lengths, voice, speed=speed, pitch_shift=shift, pitch_range=pitch_range)
            noise = torch.randn((1, int(size[0])), generator=torch.Generator().manual_seed(0))
            scalars = tuple(torch.tensor([v], dtype=torch.float32) for v in (speed, shift, pitch_range))
            feed = {name: value.numpy() for name, value in zip((*INPUTS, 'noise'), (ids, stress, voice, *scalars, noise), strict=True)}
            wave, duration = session.run(None, feed)
            assert np.array_equal(duration, info['durations'].numpy()), text
            actual, target = wave.astype('float64'), reference.numpy().astype('float64')
            assert actual.shape == target.shape and np.isfinite(actual).all(), text
            nrmse = np.linalg.norm(actual - target) / np.linalg.norm(target)
            assert nrmse < 0.01, (text, nrmse)
            print(f'ONNX vs PyTorch (fixed noise): {size.item()} samples, relative RMS error {nrmse:.4%}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/calibrated.pt')
    parser.add_argument('--audio-only', action='store_true')
    args = parser.parse_args()
    espeak = Path(os.environ.get('ESPEAK_ROOT', ROOT / '.tools/espeak-ng'))
    os.environ['PATH'] = str(espeak / 'bin') + os.pathsep + os.environ['PATH']
    os.environ['ESPEAK_DATA_PATH'] = str(espeak / 'share')
    torch.set_num_threads(1)
    if not args.audio_only:
        config = json.loads((ROOT / 'web/public/assets/model.json').read_text())
        assert config['frontend_fingerprint'] == frontend_fingerprint('nl-NL')
        frontend()
    audio(args.checkpoint)
