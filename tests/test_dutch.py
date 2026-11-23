import json
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

from tiny_voca import Config, TinyVoca, dutch, text
from tiny_voca.data import AnnotatedDataset
from tiny_voca.dutch_corpus import DATASET, mapped_durations, plan
from tiny_voca.evaluate import normalizer
from tiny_voca.infer import load_model
from tiny_voca.model import SourceFilter
from tiny_voca.train import check_leakage, export_model


class DutchFrontendTests(unittest.TestCase):
    def test_mixed_excitation_is_checkpoint_scoped_and_respects_padding(self):
        c = Config(language="nl-NL", channels=32)
        plain = SourceFilter(c)
        rng = torch.Generator().manual_seed(123)
        lengths = torch.tensor([20, 12])
        hidden = torch.randn(2, 20, c.channels, generator=rng)
        pitch = torch.rand(2, 20, generator=rng) * 80 + 100
        noise = torch.randn(2, 20*c.hop, generator=rng)
        breath = torch.randn(2, 20*c.hop, generator=rng)
        voiced = torch.rand(2, 20, generator=rng) > .5
        baseline = plain(hidden, pitch, lengths, noise=noise)
        self.assertNotIn("aperiodicity", c.to_dict())
        self.assertNotIn("noise_above", Config().to_dict())
        for setting in ({"aperiodicity": (1000., .5)}, {"noise_above": (2., 33.)}):
            with self.subTest(setting=setting):
                cfg = replace(c, **setting)
                self.assertEqual(Config.from_dict(json.loads(json.dumps(cfg.to_dict()))), cfg)
                self.assertEqual(cfg.acoustic_fingerprint, c.acoustic_fingerprint)
                vocoder = SourceFilter(cfg)
                vocoder.load_state_dict(plain.state_dict(), strict=True)
                torch.testing.assert_close(vocoder(hidden, pitch, lengths, noise=noise), baseline, atol=0, rtol=0)
                actual = vocoder(hidden, pitch, lengths, noise=noise, voiced=voiced, breath=breath)
                self.assertGreater((actual-baseline).abs().max().item(), .01)
                end = int(lengths[1])
                count = end*c.hop
                short = vocoder(hidden[1:, :end], pitch[1:, :end], lengths[1:], noise=noise[1:, :count],
                                voiced=voiced[1:, :end], breath=breath[1:, :count])
                torch.testing.assert_close(short[0], actual[1, :count])
                self.assertTrue(torch.all(actual[1, count:] == 0))
        with self.assertRaises(ValueError):
            replace(c, noise_above=(2., 33.), aperiodicity=(1000., .5))

    def test_calibrated_three_band_render_replays_uniform_fractions_and_padding(self):
        c = Config(language="nl-NL", channels=32, calibrated_excitation=True)
        self.assertEqual(Config.from_dict(c.to_dict()), c)
        self.assertNotIn("calibrated_excitation", Config().to_dict())
        with self.assertRaises(ValueError):
            replace(c, aperiodicity=(1000., .4))
        fitted = SourceFilter(c)
        fixed = SourceFilter(replace(c, calibrated_excitation=False, aperiodicity=(1000., .4)))
        with torch.no_grad():
            fixed.harmonic.load_state_dict(fitted.harmonic.state_dict())
            fixed.noise.load_state_dict(fitted.noise.state_dict())
            fitted.excitation_cost.weight.zero_()
            bias = fitted.excitation_cost.bias.view(3, -1)
            bias.fill_(4)
            bias[:, fitted.excitation_grid.index(.4)] = 0
        rng = torch.Generator().manual_seed(218)
        lengths = torch.tensor([22, 14])
        h = torch.randn(2, 22, c.channels, generator=rng)
        f0 = 95 + 30 * torch.rand(2, 22, generator=rng)
        noise = torch.randn(2, 22*c.hop, generator=rng)
        kwargs = {"noise": noise, "voiced": f0 > 100}
        expected = fixed(h, f0, lengths, **kwargs)
        actual = fitted(h, f0, lengths, **kwargs)
        torch.testing.assert_close(actual, expected)
        with torch.no_grad():
            fitted.excitation_cost.weight.normal_(generator=rng)
        vary = fitted.excitation_fractions(h)
        self.assertGreater(vary.std(1).min().item(), .01)
        self.assertTrue(((vary >= 0) & (vary <= 1)).all())
        full = fitted(h, f0, lengths, **kwargs)
        n = int(lengths[1])
        short = fitted(h[1:, :n], f0[1:, :n], lengths[1:], noise=noise[1:, :n*c.hop], voiced=f0[1:, :n] > 100)
        torch.testing.assert_close(short[0], full[1, :n*c.hop])
        self.assertTrue(torch.all(full[1, n*c.hop:] == 0))

    def test_output_response_applies_to_both_filters_and_round_trips(self):
        c = Config(language="nl-NL", channels=32)
        self.assertNotIn("output_filter", c.to_dict())
        gain_db = -6.020599913
        filtered = replace(c, output_filter=((0., gain_db), (12000., gain_db)))
        self.assertEqual(Config.from_dict(json.loads(json.dumps(filtered.to_dict()))), filtered)
        self.assertEqual(c.acoustic_fingerprint, filtered.acoustic_fingerprint)
        original, candidate = SourceFilter(c), SourceFilter(filtered)
        candidate.load_state_dict(original.state_dict(), strict=True)
        rng = torch.Generator().manual_seed(351)
        hidden = torch.randn(2, 15, c.channels, generator=rng)
        lengths = torch.tensor([15, 9])
        f0 = torch.full((2, 15), 127.)
        source = torch.randn(2, 15*c.hop, generator=rng)
        expected = original(hidden, f0, lengths, noise=source)
        actual = candidate(hidden, f0, lengths, noise=source)
        torch.testing.assert_close(actual, .5 * expected)
        self.assertTrue(torch.all(actual[1, 9*c.hop:] == 0))
        with self.assertRaises(ValueError):
            replace(c, output_filter=((0., 0.), (11000., -9.), (9000., -12.), (12000., -12.)))

    def test_inventory_is_ordered_versioned_and_selected_by_config(self):
        en, nl = Config(), Config(language="nl-NL")
        self.assertEqual((nl.speakers, nl.vocab_size), (("alex",), 80))
        self.assertEqual((en.speakers, en.vocab_size), (("F2", "M2"), 76))
        self.assertEqual(len(set(dutch.SYMBOLS)), nl.vocab_size)
        self.assertEqual(dutch.SYMBOL_TO_ID["<pad>"], 0)
        self.assertEqual(text.frontend_fingerprint("nl-NL"), "d85fba8e2c83feb45bb9dea9ce28ae4e299684d42b126c53e824a42aa5d5ef20")
        self.assertNotEqual(text.frontend_fingerprint("nl-NL"), text.frontend_fingerprint())
        with patch.object(dutch, "VERSION", 2):
            self.assertNotEqual(text.frontend_fingerprint("nl-NL"), "d85fba8e2c83feb45bb9dea9ce28ae4e299684d42b126c53e824a42aa5d5ef20")
        with self.assertRaises(ValueError):
            Config(language="nl-NL", vocab_size=en.vocab_size)
        self.assertEqual(Config.from_dict(en.to_dict()), en)
        self.assertEqual(Config.from_dict(nl.to_dict()), nl)
        self.assertEqual(nl.acoustic_fingerprint, en.acoustic_fingerprint)
        self.assertNotIn("language", en.to_dict())

    def test_spoken_normalization(self):
        cases = {
            "De prijs is € 12,05.": "De prijs is twaalf euro vijf cent.",
            "Dat is €2,50, toch?": "Dat is twee euro vijftig cent, toch?",
            "Het kost 1.234,50 euro.": "Het kost duizendtweehonderdvierendertig euro vijftig cent.",
            "Het was −3,04 graden.": "Het was min drie komma nul vier graden.",
            "1.000 bezoekers en 22% korting.": "duizend bezoekers en tweeëntwintig procent korting.",
            "Op 31-03-2026 om 9.00 uur.": "Op eenendertig maart tweeduizendzesentwintig om negen uur.",
            "Op 1998-12-02.": "Op twee december negentienhonderdachtennegentig.",
            "Mevr. De Vries woont in ’s-Hertogenbosch.": "mevrouw De Vries woont in 's-Hertogenbosch.",
            "Dr. Van Dijk, bijv. om 13:05.": "dokter Van Dijk, bijvoorbeeld om dertien uur vijf.",
            "Willem-Alexander loopt naar Nijmegen.": "Willem-Alexander loopt naar Nijmegen.",
        }
        for raw, spoken in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(dutch.spoken_text(raw), spoken)
                self.assertEqual(dutch.spoken_text(spoken), spoken)
                self.assertEqual(normalizer("nl-NL")(raw), dutch.split_key(spoken))
        with self.assertRaises(ValueError):
            dutch.spoken_text("Op 31-02-2026.")
        self.assertEqual(dutch.sentences("Mevr. Jansen komt. wat kost €3,50?"),
                         ["mevrouw Jansen komt.", "wat kost drie euro vijftig cent?"])

    def test_espeak_stdin_for_dutch_is_newline_terminated(self):
        with patch.object(text, "espeak_executable", return_value="espeak-ng"), patch.object(text.subprocess, "run") as run:
            run.return_value.stdout = "k_ˈɑ_t\n"
            self.assertEqual(text.espeak_words("kat", voice="nl")[0][-1], ("t", 0))
            self.assertEqual(run.call_args.kwargs["input"], "kat\n")
            self.assertIn("nl", run.call_args.args[0])

    @unittest.skipUnless(shutil.which("espeak-ng"), "requires eSpeak NG 1.52.0")
    def test_dutch_contrasts_compound_stress_and_final_consonant(self):
        phones = dutch.phonemize("De trein rijdt langs het huis bij het hout. De fietsbel klinkt zacht.")
        for s in ("ɛɪ", "œy", "ʌʊ", "x"):
            self.assertIn(s, phones.symbols)
        self.assertEqual(phones.unknown, ())
        self.assertEqual(phones.symbols[-3:], ("t", ".", text.EOS))
        levels = dutch.phonemize("scheepvaartmaatschappij").stress
        self.assertIn(1, levels)
        self.assertIn(2, levels)
        self.assertEqual(dutch.phonemize("De kat").symbols[-2:], ("t", text.EOS))
        self.assertNotEqual(dutch.phonemize("maan").ids, dutch.phonemize("man").ids)

    def test_normalized_key_is_deduplicated_before_assignment(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with zipfile.ZipFile(root / "corpus.zip", "w") as bundle:
                bundle.writestr(DATASET, "a.wav|Het kost € 3,50.\nb.wav|HET KOST DRIE EURO VIJFTIG CENT!\nc.wav|Kom hier.\n")
            (root / "extra").write_text("Kom hier?\n")
            splits, rejected = plan(root / "corpus.zip", root / "extra", False)
            self.assertEqual(sum(map(len, splits.values())), 2)
            self.assertEqual(len(rejected), 2)
            row = next(r for rows in splits.values() for r in rows if r["source_id"] == "a.wav")
            self.assertEqual(row["original_text"], "Het kost € 3,50.")
            self.assertEqual(row["text"], "Het kost drie euro vijftig cent.")
            train_row = {**row, "text_hash": "a"}
            val_row = {**row, "id": "other", "text_hash": "b", "text": row["text"].upper()}
            with self.assertRaisesRegex(ValueError, "leakage"):
                check_leakage(SimpleNamespace(rows=[train_row]), SimpleNamespace(rows=[val_row]))



def example(symbols=(text.BOS, "ɛɪ", "x", text.EOS), stress=(0, 1, 0, 0)):
    return text.Phonemized(symbols, tuple(dutch.SYMBOL_TO_ID[s] for s in symbols), stress, ())


class DutchAlignmentTests(unittest.TestCase):
    def test_stress_diphthong_and_following_pads_stay_with_the_token(self):
        # Each raw count already includes the pad immediately following that Piper symbol.
        phones = example()
        raw = [("^", 31), ("ˈ", 100), ("ɛ", 311), ("ɪ", 127), ("x", 299), ("$", 0)]
        self.assertEqual(dutch.teacher_units(phones), [("^", 0), ("ˈ", 1), ("ɛ", 1), ("ɪ", 1), ("x", 2), ("$", 3)])
        self.assertEqual(mapped_durations(phones, raw, 22050, 868, 945, Config(language="nl-NL")), [1, 2, 1, 0])
        with self.assertRaisesRegex(ValueError, "native timing"):
            mapped_durations(phones, raw, 22050, 869, 945, Config(language="nl-NL"))
        with self.assertRaisesRegex(ValueError, "resampled"):
            mapped_durations(phones, raw, 22050, 868, 944, Config(language="nl-NL"))
        with self.assertRaisesRegex(ValueError, "IPA"):
            mapped_durations(phones, [("^", 31), ("ˌ", 100), *raw[2:]], 22050, 868, 945, Config(language="nl-NL"))
        raw[4:] = [("x", 0), ("$", 299)]
        with self.assertRaisesRegex(ValueError, "uncovered phone"):
            mapped_durations(phones, raw, 22050, 868, 945, Config(language="nl-NL"))

    def test_clause_space_and_word_space_are_distinct_owners(self):
        phones = example((text.BOS, "aː", ",", "p", text.SP, "y", text.EOS), (0, 2, 0, 0, 0, 0, 0))
        self.assertEqual(dutch.teacher_units(phones), [("^", 0), ("ˌ", 1), ("a", 1), ("ː", 1), (",", 2),
                                                          (" ", 2), ("p", 3), (" ", 4), ("y", 5), ("$", 6)])

    def test_dataset_and_checkpoint_use_dutch_inventory(self):
        c = Config(language="nl-NL", channels=32, feedforward=64, predictor_channels=16,
                   encoder_layers=1, prosody_layers=1, acoustic_layers=1)
        m = TinyVoca(c)
        for symbol in dutch.SYMBOLS:
            i = dutch.SYMBOL_TO_ID[symbol]
            self.assertEqual(bool(m.phone_tokens[i]), i in dutch.PHONE_IDS)
            self.assertEqual(bool(m.vowel_tokens[i]), symbol in dutch.VOWELS)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            metadata = {"complete": True, "fingerprint": "timing1", "acoustic_root": name}
            (root / "annotations.json").write_text(json.dumps(metadata))
            row = {"id": "one", "speaker": "alex", "tokens": [1, dutch.SYMBOL_TO_ID["ʊ"], 2], "stress": [0, 2, 0],
                   "durations": [0, 1, 0], "frames": 1, "path": "wave.pt", "annotation": "timing1",
                   "frontend": text.frontend_fingerprint(c.language), "acoustic": c.acoustic_fingerprint}
            torch.save({"mel": torch.zeros(1, c.n_mels), "prosody": torch.zeros(1, 3), "wave": torch.zeros(c.hop)}, root / "wave.pt")
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(row))
            self.assertEqual(AnnotatedDataset(manifest, c, voiced_vowels=True)[0]["prosody"][0, 1].item(), 1)
            for change, error in (({"durations": [0, 0, 1]}, "phoneme without frames"),
                                  ({"stress": [0, 3, 0]}, "stress"), ({"frontend": text.frontend_fingerprint()}, "front end"),
                                  ({"annotation": "timing2"}, "annotation run")):
                manifest.write_text(json.dumps({**row, **change}))
                with self.assertRaisesRegex(ValueError, error):
                    AnnotatedDataset(manifest, c)
            path = root / "student.pt"
            export_model(m, path, 1)
            self.assertEqual(load_model(path).config, c)
            state = torch.load(path, weights_only=True)
            state["frontend"]["fingerprint"] = text.frontend_fingerprint()
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "different phoneme"):
                load_model(path, check_frontend=False)


if __name__ == "__main__":
    unittest.main()
