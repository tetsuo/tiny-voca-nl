import { createPhonemizer, loadTrace, sentences } from './dutch.mjs';
import * as ort from './assets/ort/ort.wasm.min.mjs';
let config, phonemize, session;

async function init() {
  const response = await fetch('./assets/model.json');
  if (!response.ok) throw Error('Modelbestanden ontbreken. Voer web/build.sh uit.');
  config = await response.json();
  if (config.language !== 'nl-NL' || !config.calibrated_excitation || config.voices.join() !== 'alex') {
    throw Error('Verwacht een gekalibreerd Nederlands model met stem-ID alex.');
  }
  ort.env.wasm.wasmPaths = new URL('./assets/ort/', import.meta.url).href;
  ort.env.wasm.numThreads = 1;
  const [model, trace] = await Promise.all([
    ort.InferenceSession.create(new URL('./assets/model.onnx', import.meta.url).href, { executionProviders: ['wasm'] }),
    loadTrace(),
  ]);
  session = model;
  phonemize = createPhonemizer(config.symbols, trace);
}

const indices = (values, dims) => new ort.Tensor('int64', BigInt64Array.from(values, BigInt), dims);
const scalar = value => new ort.Tensor('float32', Float32Array.of(value), [1]);

async function* speak(text, speed) {
  const phones = phonemize(text);
  if (!phones) return;
  const {tokens, stress, symbols} = phones;
  if (tokens.length <= Math.min(config.max_tokens, 256)) {
    const {audio, durations} = await session.run({
      tokens: indices(tokens, [1, tokens.length]), stress: indices(stress, [1, stress.length]),
      voice: indices([0], [1]), speed: scalar(speed), pitch_shift: scalar(0), pitch_range: scalar(1),
    });
    const frames = durations.data.reduce((sum, n) => sum + Number(n), 0);
    if (frames <= config.max_frames) {
      const samples = audio.data;
      if (!samples.length || samples.length !== frames * config.hop) throw Error('Ongeldige uitvoerlengte.');
      let peak = 0;
      for (const x of samples) {
        if (!Number.isFinite(x)) throw Error('Ongeldige audio uit het model.');
        peak = Math.max(peak, Math.abs(x));
      }
      if (peak > 0.99) for (let i = 0; i < samples.length; i++) samples[i] *= 0.99 / peak;
      yield {samples, symbols};
      return;
    }
  }
  const clauses = text.split(/(?<=[,;:—])\s+/);
  const words = text.split(/\s+/);
  const split = Math.ceil(words.length / 2);
  const parts = clauses.length > 1 ? clauses : words.length > 1 ? [words.slice(0, split).join(' '), words.slice(split).join(' ')] : [];
  if (!parts.length) throw Error('Deze tekst is te lang om in één keer voor te lezen.');
  for (const part of parts) yield* speak(part, speed);
}

self.onmessage = async ({data}) => {
  if (data.type !== 'load' && data.type !== 'speak') return;
  try {
    if (data.type === 'load') {
      await init();
      postMessage({type: 'ready', step: config.step, sampleRate: config.sample_rate});
      return;
    }
    const {id, text, speed} = data;
    if (!session || !Number.isFinite(speed) || speed < 0.25 || speed > 4 || text.length > 5000) throw Error('Ongeldige aanvraag.');
    let count = 0;
    for (const sentence of sentences(text)) {
      for await (const {samples, symbols} of speak(sentence, speed)) {
        postMessage({type: 'sentence', id, samples, symbols}, [samples.buffer]);
        count++;
      }
    }
    if (!count) throw Error('Geen uitspreekbare Nederlandse tekst.');
    postMessage({type: 'done', id});
  } catch (error) {
    postMessage({type: 'error', id: data.id, message: String(error.message || error)});
  }
};
