"""Fit and check a frame-dependent, three-band excitation on the Dutch student's own training split."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft
from scipy.signal import butter, hilbert, sosfiltfilt
import torch

from tiny_voca import dutch
from tiny_voca.data import AnnotatedDataset, collate
from tiny_voca.infer import load_model
from tiny_voca.journal import atomic_json, file_hash
from tiny_voca.model import SourceFilter
from tiny_voca.train import atomic_save


BANDS = ((700, 2000), (2000, 4000), (4000, 8000))
RATE, HOP = 24000, 256
FILTERS = [butter(4, pair, btype="bandpass", fs=RATE, output="sos") for pair in BANDS]
GRID = SourceFilter.excitation_grid


def compare_frames(item, predicted, phones_only):
    """Time-warp within each phone. Keep active, interior frames and real waveform windows."""
    ref, gen = item["durations"].numpy(), predicted.numpy()
    phone = np.repeat(item["tokens"].numpy(), ref)
    index = np.repeat(np.arange(len(ref)), ref)
    r = np.arange(len(index))
    relative = (r - (ref.cumsum() - ref)[index] + .5) / ref[index]
    s = (gen.cumsum() - gen)[index] + (gen[index] * relative).astype(int)
    is_phone = np.isin(phone, list(dutch.PHONE_IDS))
    word_boundary = phone == dutch.SYMBOL_TO_ID['<sp>']
    timed = is_phone if phones_only else (is_phone | word_boundary)
    active = item["prosody"][:, 2].numpy() * 2 - 4 > np.log(10**(-45/20))
    keep = (timed & active & (gen[index] >= 3) & (np.minimum(relative, 1-relative)*ref[index] >= .95)
            & (r >= 2) & (s >= 2) & (r+2 < len(r)) & (s+2 < gen.sum()))
    vowels = np.isin(phone, list(dutch.VOWEL_IDS)) & (item["pitch_trusted"].numpy() > .5)
    voiceless = np.isin(phone, [dutch.SYMBOL_TO_ID[x] for x in dutch.VOICELESS]) & (item["prosody"][:, 1].numpy() < .5)
    classes = np.select((vowels, voiceless, word_boundary), ('vowel', 'voiceless', 'word_boundary'), default='other')
    return r[keep], s[keep], classes[keep]


def band_features(wave, pitch, frames):
    """Measure envelope crest over two periods, waveform periodicity near F0, and band RMS in dBFS."""
    centers = frames * HOP
    period = RATE / pitch[frames]
    offsets = np.arange(-480, 480)
    samples = centers[:, None] + offsets
    cycle = np.abs(offsets) < period[:, None]
    lags = np.arange(1, 530)
    near = (lags >= np.floor(.93 * period[:, None])) & (lags <= np.ceil(1.07 * period[:, None]))
    result = []
    for filt in FILTERS:
        narrow = sosfiltfilt(filt, wave).astype(np.float32)
        env = np.abs(hilbert(narrow))[samples]
        crest = np.where(cycle, env, 0).max(-1) * cycle.sum(-1) / np.maximum((env * cycle).sum(-1), 1e-9)
        w = narrow[samples]
        w -= w.mean(-1, keepdims=True)
        energy = w*w
        db = 10*np.log10(np.maximum(energy.mean(-1), 1e-12))
        integral = np.pad(np.cumsum(energy, axis=1, dtype='float64'), ((0, 0), (1, 0)))
        norm = np.sqrt(np.maximum(integral[:, 960-lags] * (integral[:, -1:] - integral[:, lags]), 1e-20))
        corr = irfft(abs(rfft(w, n=2048))**2, n=2048)[:, lags] / norm
        result.append(np.stack((crest, np.max(np.where(near, corr, -1), axis=1), db), axis=-1))
    return np.stack(result, axis=1)


def deviations(actual, ref):
    # Relative crest gives the same scale to a change from 2 to 3 as from 4 to 6.
    crest = np.abs(np.log(np.maximum(actual[..., 0], 1)) - np.log(np.maximum(ref[..., 0], 1)))
    periodicity = np.abs(actual[..., 1] - ref[..., 1])
    db = np.abs(actual[..., 2] - ref[..., 2])
    return crest, periodicity, db


def cost(actual, ref):
    crest, periodicity, db = deviations(actual, ref)
    return crest + .75 * periodicity + .02 * np.minimum(db, 12)


def render(model, b):
    """Capture the ordinary inference inputs once. Replay different excitations on the same acoustic states."""
    inputs = {}

    def capture(module, args, kwargs):
        inputs['args'], inputs['kwargs'] = args, kwargs

    handle = model.vocoder.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        wave, _, info = model.synthesize(b['tokens'], b['stress'], b['token_lengths'], b['speakers'], seed=0)
    finally:
        handle.remove()
    return wave[0].numpy(), info['durations'][0], inputs['args'], inputs['kwargs']


def french_render(vocoder, inputs, kwargs, q):
    original = vocoder.config
    vocoder.config = replace(original, aperiodicity=(1000., q), noise_above=None, calibrated_excitation=False)
    try:
        return vocoder(*inputs, **kwargs)[0].numpy()
    finally:
        vocoder.config = original


def training_indices(length, count):
    if count < 1:
        raise ValueError('training sample count must be positive')
    return np.random.default_rng(20261113).permutation(length)[:count]


@torch.inference_mode()
def fit(args):
    model = load_model(args.checkpoint)
    c = model.config
    if c.language != 'nl-NL' or c.aperiodicity or c.noise_above or c.calibrated_excitation:
        raise ValueError('calibration starts with the unmodified Dutch inference checkpoint')
    ds = AnnotatedDataset(args.annotations / 'train.jsonl', c)
    indices = training_indices(len(ds), args.train_count)
    dim = c.channels + 1
    placement = {'device': args.fit_device, 'dtype': torch.float32 if args.fit_device == 'mps' else torch.float64}
    xx = torch.zeros((3, dim, dim), **placement)
    xy = torch.zeros((3, dim, len(GRID)), **placement)
    counts = np.zeros(3, dtype=int)
    for n, idx in enumerate(indices):
        item = ds[int(idx)]
        wave, durations, inputs, kwargs = render(model, collate([item]))
        r, s, _ = compare_frames(item, durations, args.phones_only)
        if not len(r):
            continue
        ref_pitch = model.frequency(item['prosody'][:, 0]).numpy()
        ref = band_features(item['wave'].numpy(), ref_pitch, r)
        pitch = inputs[1][0].numpy()
        scores = []
        for q in GRID:
            candidate = wave if q == 0 else french_render(model.vocoder, inputs, kwargs, q)
            scores.append(cost(band_features(candidate, pitch, s), ref))
        scores = np.stack(scores, axis=-1)
        x = np.column_stack((inputs[0][0, s].numpy(), np.ones(len(s))))
        for band in range(3):
            active = ref[:, band, 2] >= -55
            if not active.any():
                continue
            features = torch.as_tensor(x[active], **placement)
            error = torch.as_tensor(scores[active, band], **placement)
            # Only differences between excitation settings affect the eventual choice.
            xx[band] += features.T @ features
            xy[band] += features.T @ (error - error[:, :1])
            counts[band] += len(features)
        if (n+1) % 64 == 0:
            print(f'fitted observations for {n+1}/{len(indices)} training sentences', flush=True)
    if np.any(counts <= dim):
        raise ValueError('too few audible training windows to calibrate the filter')
    weights = []
    for covariance, cross in zip(xx, xy):
        penalty = torch.eye(dim, **placement) * (.025 * covariance.diagonal()[:-1].mean())
        penalty[-1, -1] = 0
        weights.append(torch.linalg.solve(covariance + penalty, cross).T)
    weights = torch.cat(weights).float().cpu()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state['config'] = replace(c, calibrated_excitation=True).to_dict()
    state['model']['vocoder.excitation_cost.weight'] = weights[:, :-1].contiguous()
    state['model']['vocoder.excitation_cost.bias'] = weights[:, -1].contiguous()
    if args.output.resolve() == args.checkpoint.resolve():
        raise ValueError('give the calibrated checkpoint its own path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(state, args.output)
    atomic_json(args.output.with_suffix('.fit.json'), {'checkpoint_sha256': file_hash(args.checkpoint),
        'annotation': ds.annotation, 'train_ids': [ds.rows[int(i)]['id'] for i in indices],
        'active_frames_per_band': counts.tolist(), 'power_fractions': GRID, 'bands_hz': BANDS, 'phones_only': args.phones_only,
        'fit_device': args.fit_device, 'render_device': 'cpu',
        'new_parameters': weights.numel(), 'calibrated_sha256': file_hash(args.output)})


@torch.inference_mode()
def check(args):
    models = {name: load_model(path) for name, path in args.models}
    if len(models) != len(args.models):
        raise ValueError('each checkpoint needs its own name')
    first = next(iter(models.values()))
    ds = AnnotatedDataset(args.annotations / (args.split+'.jsonl'), first.config)
    all_rows = []
    for i, row in enumerate(ds.rows):
        if args.id and row['id'] != args.id:
            continue
        item = ds[i]
        b = collate([item])
        ref_pitch = first.frequency(item['prosody'][:, 0]).numpy()
        per_row = {}
        for name, model in models.items():
            wave, dur, inputs, _ = render(model, b)
            r, s, classes = compare_frames(item, dur, args.phones_only)
            if not len(r):
                continue
            ref = band_features(item['wave'].numpy(), ref_pitch, r)
            actual = band_features(wave, inputs[1][0].numpy(), s)
            total = cost(actual, ref)
            metrics = np.stack((*deviations(actual, ref), total), axis=-1)
            observed = {}
            for group in np.unique(classes):
                for label, mask in (('whole', np.ones(len(s), dtype=bool)), ('first', 2*s < dur.sum().item()),
                                    ('second', 2*s >= dur.sum().item())):
                    for j, (lo, hi) in enumerate(BANDS):
                        selected = (classes == group) & mask & (ref[:, j, 2] >= -55)
                        if selected.any():
                            observed[f'{group}/{label}/{lo}-{hi}'] = metrics[selected, j, :].mean(0).tolist()
            per_row[name] = observed
        all_rows.append({'id': row['id'], 'comparisons': per_row})
    if not all_rows:
        raise ValueError('no matching utterance in this split')
    means = {}
    for name in models:
        by_key = defaultdict(list)
        for row in all_rows:
            for key, values in row['comparisons'].get(name, {}).items():
                by_key[key].append(values)
        means[name] = {k: np.mean(v, axis=0).tolist() for k, v in by_key.items()}
    score = {name: float(np.mean([v[-1] for k, v in groups.items() if '/whole/' in k])) for name, groups in means.items()}
    atomic_json(args.output, {'split': args.split, 'utterances': len(all_rows), 'phones_only': args.phones_only,
        'models': {k: file_hash(p) for k,p in args.models},
        'measurements': ['absolute_log_crest_error', 'absolute_periodicity_error', 'absolute_band_db_error', 'combined'],
        'score': score, 'mean_sentence_errors': means, 'rows': all_rows})
    print(json.dumps(score), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    fitter = sub.add_parser('fit')
    fitter.add_argument('--checkpoint', type=Path, required=True)
    fitter.add_argument('--train-count', type=int, default=512)
    fitter.add_argument('--fit-device', choices=('mps', 'cpu'), default='mps',
                        help='Device for fitting the ridge regression. Source and reference audio are measured on CPU.')
    checker = sub.add_parser('check')
    checker.add_argument('--models', nargs=2, action='append', required=True, metavar=('LABEL', 'CHECKPOINT'))
    checker.add_argument('--split', choices=('validation', 'test'), default='validation')
    checker.add_argument('--id', help='Check a single known sentence for a listening comparison')
    for cmd in (fitter, checker):
        cmd.add_argument('--phones-only', action='store_true', help='Reproduce the first pilot, which excluded audible word boundaries')
        cmd.add_argument('--annotations', type=Path, required=True)
        cmd.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    (fit if args.action == 'fit' else check)(args)
