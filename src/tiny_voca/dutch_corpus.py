"""Build the Alex synthetic corpus from the staged, pinned Dutch sources."""
import argparse
from collections import Counter
import csv
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from . import dutch
from .config import Config, _digest
from .journal import atomic_json, file_hash
from .text import frontend_fingerprint

SPLITS = ("train", "validation", "test")
ALEX_REVISION = "e929968c3761f1c281875787685a745863db838d"
CV_REVISION = "9f55dafbb562ee6b4a1987432052d219d43890c4"
CV_SHA256 = "92c6c47187a587f9cdb03516beb73b857ebcc65ff77800039e5cc38cfea980de"
CV_FILES = {"sentence-collector.txt": CV_SHA256,
            "danielsjf.txt": "cd171d4479243f161145ab3b3f3c46afcb1cfa5770b70a852ee40d1db8923a50",
            "pcmill.txt": "e86490c369c06817359d3d2e19475b95db63501acd72b9ed414205fdef847f03"}
DATASET = "nl_NL-alex/nl-NL/metadata.csv"
# The staged synthesis and real-recording smoke examples are outside all splits.
SMOKE = ("Goedemorgen, de trein naar Utrecht vertrekt over vijf minuten.",
         "Wij willen hen bedanken voor de aanwezigheid!")
SEED = 1701


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").split("\n") if s.strip()]


def write_rows(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for r in rows:
            file.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def plan(archive, extra, pilot):
    with zipfile.ZipFile(archive) as z:
        corpus = csv.reader(io.StringIO(z.read(DATASET).decode("utf-8-sig")), delimiter="|")
        inputs = [(name, text, "OHF-Voice/voice-datasets v1.0.0") for name, text in corpus]
    inputs.extend((f"contrast-{i:03}", s, "Dutch contrast prompts v1") for i, s in
                  enumerate(extra.read_text(encoding="utf-8").splitlines(), 1) if s.strip())
    seen = {dutch.split_key(dutch.spoken_text(s)) for s in SMOKE}
    by_split = {split: [] for split in SPLITS}
    rejected = []
    for name, text, source in inputs:
        try:
            spoken = dutch.spoken_text(text)
            key = dutch.split_key(spoken)
            if not key or key in seen:
                raise ValueError("duplicate normalized text or reserved smoke text")
            if re.search(r"\d|€", spoken):
                raise ValueError("unresolved written number or currency")
            if len(dutch.sentences(spoken)) != 1:
                raise ValueError("multiple sentences in source row")
        except ValueError as error:
            rejected.append({"source_id": name, "text": text, "reason": str(error)})
            continue
        seen.add(key)
        digest = hashlib.sha256(key.encode()).hexdigest()
        bucket = int(digest[:8], 16) % 100
        split = "test" if bucket < 5 else "validation" if bucket < 10 else "train"
        by_split[split].append({"id": "nl-" + digest[:16], "split": split, "text": spoken,
                                "original_text": text, "normalized_key": key, "text_id": digest,
                                "source_id": name, "origin": source, "voice_id": "alex"})
    for split, rows in by_split.items():
        rows.sort(key=lambda r: r["id"])
        if pilot:
            by_split[split] = rows[:12 if split == "train" else 3]
    return by_split, rejected


def additional_train_text(path, limit, existing):
    """Choose plain Dutch sentences in the training hash buckets before synthesis."""
    from .evaluate import DUTCH_NOVEL
    if file_hash(path) != CV_FILES.get(path.name) or (limit is not None and limit < 1):
        raise ValueError("expected a pinned Dutch Common Voice file and a positive count")
    seen = {r["normalized_key"] for rows in existing.values() for r in rows}
    seen.update(dutch.split_key(dutch.spoken_text(s)) for s in (*SMOKE, *DUTCH_NOVEL))
    candidates = []
    url = f"https://github.com/common-voice/common-voice/blob/{CV_REVISION}/server/data/nl/{path.name}"
    for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if (not 6 <= len(text.split()) <= 18 or not re.fullmatch(r"[\w'’ ,\-]+[.!?]", text)
                or "_" in text or needs_reading_check(text)):
            continue
        spoken = dutch.spoken_text(text)
        key = dutch.split_key(spoken)
        if key in seen:
            continue
        seen.add(key)
        digest = hashlib.sha256(key.encode()).hexdigest()
        if int(digest[:8], 16) % 100 < 10:
            continue
        candidates.append({"id": "nl-" + digest[:16], "split": "train", "text": spoken,
                           "original_text": text, "normalized_key": key, "text_id": digest,
                           "source_id": f"nl/{path.name}:{line}", "source_url": f"{url}#L{line}",
                           "source_license": "CC0-1.0", "origin": "Mozilla Common Voice " + CV_REVISION,
                           "voice_id": "alex"})
    candidates.sort(key=lambda r: r["id"])
    if limit is not None and len(candidates) < limit:
        raise ValueError(f"only {len(candidates)} supplemental training sentences available")
    return candidates[:limit]


def mapped_durations(phones, raw, rate, native_samples, samples, config):
    """Assign frame centre i*hop to its half-open source interval.

    Count centres strictly before each cumulative boundary using integer ceil.
    Resampling by scipy.signal.resample_poly emits ceil(N * target/native).
    """
    units = dutch.teacher_units(phones)
    if len(units) != len(raw) or [u for u, _ in units] != [u for u, _ in raw]:
        raise ValueError("teacher/student IPA mismatch")
    if any(type(n) is not int or n < 0 for _, n in raw) or sum(n for _, n in raw) != native_samples:
        raise ValueError("native timing coverage mismatch")
    if samples != (native_samples * config.sample_rate + rate - 1) // rate:
        raise ValueError("resampled length mismatch")
    owned = [0] * len(phones)
    for (_, owner), (_, n) in zip(units, raw):
        owned[owner] += n
    frames = (samples + config.hop - 1) // config.hop
    denom = rate * config.hop
    ends = [(int(n) * config.sample_rate + denom - 1) // denom for n in np.cumsum(owned)]
    durations = [b - a for a, b in zip([0] + ends, ends)]
    if sum(durations) != frames or any(d < (t in dutch.PHONE_IDS) for t, d in zip(phones.ids, durations)):
        raise ValueError("uncovered phone or frame")
    return durations


def load_teacher(model):
    import onnx
    import onnxruntime as ort
    from piper import PiperVoice
    from piper.config import PiperConfig
    from piper.patch_voice_with_alignment import add_alignment_output
    ort.set_seed(SEED)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.use_deterministic_compute = True
    graph = onnx.load(str(model))
    add_alignment_output(graph)
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    config = PiperConfig.from_dict(json.loads(Path(str(model) + ".json").read_text()))
    if (config.espeak_voice, config.sample_rate, config.num_speakers) != ("nl", 22050, 1):
        raise ValueError("expected the staged single-speaker Alex voice")
    return PiperVoice(session, config)


def generate(args, c):
    import soundfile as sf
    from piper.config import SynthesisConfig
    from scipy.signal import resample_poly
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    espeak = dutch.require_runtime()
    identity = {"model": args.model.name, "model_sha256": file_hash(args.model), "model_revision": ALEX_REVISION,
                "config_sha256": file_hash(str(args.model) + ".json"), "corpus_sha256": file_hash(args.archive),
                "prompts_sha256": file_hash(args.prompts), "piper": version("piper-tts"), "onnxruntime": version("onnxruntime"),
                "espeak": espeak, "seed": SEED, "volume": 0.85, "pilot": args.pilot,
                "frontend": frontend_fingerprint(c.language), "audio_source": "Piper Alex synthetic"}
    expected = {"model_sha256": "758fa64ec461545972fc3e509511b48849c6f3f5c821ad55c71d9741d4352bdc",
                "config_sha256": "01eb08f80975aa4643c267abed81e6289463bad77be7b5a96d1773385586519b",
                "corpus_sha256": "5b134e4d82829176cd530d71e7a6e804351e18584f18d540a35f8fd90380b5ff"}
    if any(identity[key] != sha for key, sha in expected.items()):
        raise ValueError("input checksums differ from the pinned Alex corpus, teacher or sidecar")
    by_split, excluded = plan(args.archive, args.prompts, args.pilot)
    jobs = list(by_split.items())
    planned = [r for _, rows in jobs for r in rows]
    if args.common_voice is not None:
        if args.pilot:
            raise ValueError("the pilot uses the original split")
        if args.common_voice.name != "sentence-collector.txt":
            raise ValueError("the main text input must be the pinned Dutch sentence collector")
        initial_count = min(6500, args.extra_train) if args.extra_text else args.extra_train
        extra = additional_train_text(args.common_voice, initial_count, by_split)
        # Retain the first scale stage verbatim, then include the other writing domains.
        for path in args.extra_text:
            available = additional_train_text(path, None, {**by_split, "extra": extra})
            extra.extend(available[:args.extra_train - len(extra)])
        if len(extra) < args.extra_train:
            extra.extend(additional_train_text(args.common_voice, args.extra_train - len(extra), {**by_split, "extra": extra}))
        # Keep all original training and held-out teacher draws in their original order.
        jobs.append(("train", extra))
        planned.extend(extra)
        identity.update(common_voice_revision=CV_REVISION, common_voice_sha256=CV_SHA256, extra_train=len(extra))
        if args.extra_text:
            identity["common_voice_extra_files"] = {p.name: CV_FILES[p.name] for p in args.extra_text}
    write_rows(root / "plan.jsonl", planned)
    write_rows(root / "plan.rejected.jsonl", excluded)
    atomic_json(root / "teacher.json", {"complete": False, "identity": identity})
    teacher = load_teacher(args.model)
    settings = SynthesisConfig(volume=identity["volume"])
    (root / "wav").mkdir(exist_ok=True)
    accepted_rows = {split: [] for split in SPLITS}
    rejected_rows = {split: [] for split in SPLITS}
    for split, rows in jobs:
        accepted, rejected = accepted_rows[split], rejected_rows[split]
        for row in rows:
            try:
                phones = dutch.phonemize(row["text"])
                if len(phones) > c.max_tokens:
                    raise ValueError("text exceeds max_tokens")
                units = dutch.teacher_units(phones)
                raw_ipa = "".join(u for u, _ in units[1:-1])
                missing = sorted(set(raw_ipa) - set(teacher.config.phoneme_id_map))
                if missing:
                    raise ValueError(f"codes missing from teacher inventory: {missing}")
                # Directly condition the teacher on the student's segmented pronunciation.
                chunk, = teacher.synthesize("[[" + raw_ipa + "]]", settings, include_alignments=True)
                if chunk.phoneme_alignments is None:
                    raise ValueError("teacher did not return timings")
                raw = [(a.phoneme, int(a.num_samples)) for a in chunk.phoneme_alignments]
                native = chunk.audio_float_array
                wave = resample_poly(native, 160, 147).astype(np.float32)
                wave *= min(1., 0.99 / max(float(np.max(np.abs(wave))), 1e-8))
                mapped_durations(phones, raw, chunk.sample_rate, len(native), len(wave), c)
                audio = "wav/" + row["id"] + ".wav"
                sf.write(root / audio, wave, c.sample_rate, subtype="FLOAT")
                pad = teacher.config.phoneme_id_map["_"][0]
                internal_pad_samples = sum(int(n) for t, n in zip(chunk.phoneme_ids, chunk.phoneme_id_samples) if t == pad)
                accepted.append({**row, "audio": audio, "teacher_model": args.model.name + "@" + ALEX_REVISION,
                                 "native_rate": chunk.sample_rate, "native_samples": len(native), "samples": len(wave),
                                 "piper_alignments": raw, "internal_pad_samples": internal_pad_samples})
            except ValueError as error:
                rejected.append({**row, "reason": str(error)})
            if (len(accepted) + len(rejected)) % 100 == 0:
                print(f"{split} {len(accepted)} accepted, {len(rejected)} rejected", flush=True)
        write_rows(root / (split + ".jsonl"), accepted)
        write_rows(root / (split + ".rejected.jsonl"), rejected)
        print(f"{split} {len(accepted)} accepted, {len(rejected)} rejected so far", flush=True)
    atomic_json(root / "teacher.json", {"complete": True, "identity": identity})


def needs_reading_check(text):
    """Written numbers, number words, initials and mid-sentence capitals."""
    return (bool(re.search(r"\d|€|\b(?:[A-Z]\s+){2}|(?<!^)\b[A-Z][a-zA-ZÀ-ÿ]+", text))
            or bool(re.search(r"\b(?:twee|drie|vier|vijf|zes|zeven|acht|negen|tien|elf|twaalf|twintig|dertig|"
                              r"veertig|vijftig|zestig|zeventig|tachtig|negentig|honderd|duizend|miljoen)\w*\b",
                              text, re.I)))


def audit(root, model, output):
    """A conservative ASR screen for ambiguous readings before accepting labels."""
    from faster_whisper import WhisperModel
    from .evaluate import normalizer, transcribe
    score_text = normalizer("nl-NL")
    asr = WhisperModel(str(model), device="cpu", compute_type="int8", cpu_threads=5, local_files_only=True)
    by_id = {}
    for split in SPLITS:
        for row in lines(root / (split + ".jsonl")):
            if not needs_reading_check(row["original_text"]):
                continue
            transcript = transcribe(asr, root / row["audio"], "nl")
            by_id[row["id"]] = {"split": split, "original": row["original_text"], "spoken": row["text"],
                                "transcript": transcript, "match": score_text(row["text"]) == score_text(transcript)}
    atomic_json(output, {"model_sha256": file_hash(model / "model.bin"), "teacher_sha256": file_hash(root / "teacher.json"),
                         "rows": by_id})


def quantiles(values):
    if not values:
        return []
    return [round(float(x), 4) for x in np.quantile(values, [0, .1, .5, .9, 1])]


def annotate(root, features, output, c, audit_path=None):
    from .data import AnnotatedDataset
    import soundfile as sf
    import torch
    teacher = json.loads((root / "teacher.json").read_text())
    if not teacher["complete"]:
        raise ValueError("teacher generation incomplete")
    identity = {"version": 1, "method": "Piper raw IPA, grouped following pad and stress, half-open frame centres",
                "frontend": frontend_fingerprint(c.language), "acoustic": c.acoustic_fingerprint, "teacher": teacher["identity"],
                "timing_manifests": {s: file_hash(root / (s + ".jsonl")) for s in SPLITS}}
    review = None
    if audit_path is not None:
        review = json.loads(audit_path.read_text())
        if review["teacher_sha256"] != file_hash(root / "teacher.json"):
            raise ValueError("reading check is for a different teacher run")
        identity["reading_check_sha256"] = file_hash(audit_path)
    if teacher["identity"]["frontend"] != identity["frontend"]:
        raise ValueError("teacher and student front ends differ")
    fingerprint = _digest(identity)
    metadata = {"identity": identity, "fingerprint": fingerprint, "acoustic_root": str(features.resolve())}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "annotations.json", {**metadata, "complete": False})
    summary = {"splits": {}, "exclusions_before_split": len(lines(root / "plan.rejected.jsonl"))}
    seen = set()
    for split in SPLITS:
        teacher_rows = {r["id"]: r for r in lines(root / (split + ".jsonl"))}
        prep = json.loads((features / (split + ".preparation.json")).read_text())
        if not prep["complete"] or prep["identity"]["sources"][0]["sha256"] != identity["timing_manifests"][split]:
            raise ValueError("incomplete preparation or different source manifest")
        rejected = lines(root / (split + ".rejected.jsonl")) + lines(features / (split + ".rejected.jsonl"))
        accepted, lengths, counts = [], [], Counter()
        for r in lines(features / (split + ".jsonl")):
            source = teacher_rows[r["id"]]
            if source["normalized_key"] in seen:
                raise ValueError("normalized text leakage")
            seen.add(source["normalized_key"])
            try:
                if review is not None and needs_reading_check(source["original_text"]):
                    if not review["rows"][r["id"]]["match"]:
                        raise ValueError("reading review: Dutch ASR disagrees, pronunciation unconfirmed")
                phones = dutch.phonemize(source["original_text"])
                ds = mapped_durations(phones, source["piper_alignments"], source["native_rate"], source["native_samples"], source["samples"], c)
                if sum(ds) != r["frames"] or r["text"] != source["text"]:
                    raise ValueError("feature/teacher mismatch")
                frame_symbols = np.repeat(phones.symbols, ds)
                item = torch.load(features / r["path"], map_location="cpu", weights_only=True)
                wave, rate = sf.read(root / source["audio"], dtype="float32")
                stored = item["wave"].numpy()
                if (rate != c.sample_rate or len(wave) != source["samples"] or not np.array_equal(wave, stored[:len(wave)])
                        or len(stored) != r["frames"] * c.hop or np.any(stored[len(wave):])):
                    raise ValueError("prepared waveform differs from the aligned teacher or its zero padding")
                voicing = item["prosody"][:, 1].numpy()
                voiced = np.isin(frame_symbols, dutch.VOWELS)
                unvoiced = np.isin(frame_symbols, list(dutch.VOICELESS))
                if len(voicing) != len(frame_symbols):
                    raise ValueError("feature frame count mismatch")
                phone_lengths = [d for t, d in zip(phones.ids, ds) if t in dutch.PHONE_IDS]
                qc = {"min_phone_frames": min(phone_lengths), "max_phone_frames": max(phone_lengths),
                      "vowel_voiced": float(voicing[voiced].mean()) if voiced.any() else None,
                      "voiceless_voiced": float(voicing[unvoiced].mean()) if unvoiced.any() else None,
                      "internal_pad_samples": source["internal_pad_samples"],
                      "stress_samples": sum(n for s, n in source["piper_alignments"] if s in ("ˈ", "ˌ")),
                      "resampled_samples": len(wave), "wave_pcm_sha256": hashlib.sha256(wave.tobytes()).hexdigest()}
                accepted.append({**r, "split": split, "original_text": source["original_text"], "tokens": list(phones.ids),
                                 "stress": list(phones.stress), "symbols": " ".join(phones.symbols), "durations": ds,
                                 "normalized_key": source["normalized_key"], "qc": qc, "unknown": [],
                                 "frontend": identity["frontend"], "annotation": fingerprint})
                lengths.extend(phone_lengths)
                counts.update(s for s in phones.symbols if s in dutch.PHONE_SET)
            except ValueError as error:
                rejected.append({"id": r["id"], "text": r["text"], "reason": str(error)})
        write_rows(output / (split + ".jsonl"), accepted)
        write_rows(output / (split + ".rejected.jsonl"), rejected)
        summary["splits"][split] = {"accepted": len(accepted), "rejected": len(rejected),
            "reasons": dict(Counter(r["reason"] for r in rejected)), "seconds": sum(r["frames"] for r in accepted) * c.hop / c.sample_rate,
            "unpadded_seconds": sum(r["qc"]["resampled_samples"] for r in accepted) / c.sample_rate,
            "phone_frames_min_p10_p50_p90_max": quantiles(lengths), "phone_counts": dict(counts),
            "vowel_voiced_quantiles": quantiles([r["qc"]["vowel_voiced"] for r in accepted if r["qc"]["vowel_voiced"] is not None]),
            "voiceless_voiced_quantiles": quantiles([r["qc"]["voiceless_voiced"] for r in accepted if r["qc"]["voiceless_voiced"] is not None]),
            "zero_frame_phones": sum(d == 0 for d in lengths)}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "annotations.json", {**metadata, "complete": True})
    try:
        for split in SPLITS:
            AnnotatedDataset(output / (split + ".jsonl"), c, cache=False)
    except Exception:
        atomic_json(output / "annotations.json", {**metadata, "complete": False})
        raise
    print(json.dumps({s: summary["splits"][s]["accepted"] for s in SPLITS}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    for name in ("archive", "model", "prompts", "output"):
        gen.add_argument("--" + name, required=True, type=Path)
    gen.add_argument("--pilot", action="store_true", help="First 12/3/3 rows of the final text split")
    gen.add_argument("--common-voice", type=Path, help="Add training texts from the pinned Dutch sentence collector")
    gen.add_argument("--extra-train", type=int, default=6500, help="Additional normalized texts with --common-voice")
    gen.add_argument("--extra-text", nargs="*", type=Path, default=[], help="Other pinned Dutch Common Voice text files")
    ann = sub.add_parser("annotate")
    for name in ("inputs", "features", "output"):
        ann.add_argument("--" + name, required=True, type=Path)
    ann.add_argument("--audit", type=Path, help="Optional Dutch source reading check")
    check = sub.add_parser("audit")
    for name in ("inputs", "asr-model", "output"):
        check.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    c = Config(language="nl-NL")
    if args.command == "generate":
        generate(args, c)
    elif args.command == "audit":
        audit(args.inputs, args.asr_model, args.output)
    else:
        annotate(args.inputs, args.features, args.output, c, args.audit)


if __name__ == "__main__":
    main()
