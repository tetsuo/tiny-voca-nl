"""nl-NL text, phone inventory and the exact pronunciation passed to Alex.

eSpeak's separator distinguishes diphthongs from adjacent vowels. The teacher
receives these same phones as raw IPA. Piper times each code point, stress mark
and following pad. The ownership map folds all of them back into student tokens.
"""
from datetime import date
from functools import lru_cache
import re
import unicodedata

from .text import (BOS, EOS, PAD, SP, UNK, NothingToPronounce, Phonemized,
                   espeak_version, espeak_words, normalize_text)

VOICE = "nl"
VERSION = 1
PUNCTUATION = (",", ".", "?", "!", ";", ":")
CONSONANTS = ("p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ", "x", "ɣ", "h",
              "m", "n", "ŋ", "ɲ", "l", "r", "j", "ʋ", "w", "tʲ", "tʃ", "dʒ", "ɾ", "ʔ", "θ", "ð", "ɹ")
VOWELS = ("ɑ", "aː", "ɛ", "eː", "ɪ", "i", "ɔ", "oː", "ɵ", "y", "u", "ə", "øː", "ɛɪ", "œy",
          "ʌʊ", "ɪː", "ɔː", "ɛː", "iː", "uː", "yː", "yʊ", "eʊ", "aɪ", "a", "ɒ", "o", "ɑ̃",
          "aʊ", "ɔɪ", "eɪ", "oʊ", "æ", "ʌ", "ɜː", "ʊ")
SYMBOLS = (PAD, BOS, EOS, SP, UNK) + PUNCTUATION + CONSONANTS + VOWELS
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
PHONE_SET = frozenset(CONSONANTS + VOWELS)
PHONE_IDS = frozenset(SYMBOL_TO_ID[s] for s in PHONE_SET | {UNK})
VOWEL_IDS = frozenset(SYMBOL_TO_ID[s] for s in VOWELS)
VOICELESS = frozenset(("p", "t", "k", "f", "s", "ʃ", "x", "h", "tʲ", "tʃ", "ʔ", "θ"))

SMALL = ("nul", "een", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen",
         "tien", "elf", "twaalf", "dertien", "veertien", "vijftien", "zestien", "zeventien", "achttien", "negentien")
TENS = ("", "", "twintig", "dertig", "veertig", "vijftig", "zestig", "zeventig", "tachtig", "negentig")
MONTHS = ("januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december")
ABBREVIATIONS = {"d.w.z.": "dat wil zeggen", "bijv": "bijvoorbeeld",
                 "m.a.w.": "met andere woorden", "a.u.b.": "alstublieft", "bijv.": "bijvoorbeeld",
                 "dhr.": "de heer", "mevr.": "mevrouw", "dr.": "dokter", "prof.": "professor", "nr.": "nummer"}
_ABBR = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, sorted(ABBREVIATIONS, key=len, reverse=True))) + r")(?!\w)", re.I)
_NUMBER = r"(?:(?:\d{1,3}(?:\.\d{3})+)|\d+)"


def cardinal(n):
    if n < 20:
        return SMALL[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return (SMALL[unit] + ("ën" if unit in (2, 3) else "en") if unit else "") + TENS[tens]
    for scale, word in ((10**9, " miljard "), (10**6, " miljoen "), (1000, "duizend"), (100, "honderd")):
        if n >= scale:
            high, low = divmod(n, scale)
            first = "" if high == 1 and scale <= 1000 else cardinal(high)
            return (first + word + (cardinal(low) if low else "")).strip()


def year(n):
    if 1100 <= n < 2000:
        high, low = divmod(n, 100)
        return SMALL[high] + "honderd" + (cardinal(low) if low else "")
    return cardinal(n)


def spoken_text(text):
    """Expand written Dutch forms before splitting, phonemization or hashing."""
    text = normalize_text(text)
    text = re.sub(r"(?<!\w)[−-](?=\s*[\d€])", "min ", text)
    text = _ABBR.sub(lambda m: ABBREVIATIONS[m[0].lower()], text)

    def numeric_date(m):
        a, b, c = map(int, m.groups())
        y, month, day = (a, b, c) if len(m[1]) == 4 else (c, b, a)
        date(y, month, day)  # reject impossible dates
        return f"{cardinal(day)} {MONTHS[month - 1]} {year(y)}"

    text = re.sub(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", numeric_date, text)
    text = re.sub(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", numeric_date, text)

    def amount(m):
        euros, cents = int(m[1].replace(".", "")), int((m[2] or "0").ljust(2, "0"))
        return cardinal(euros) + " euro" + (f" {cardinal(cents)} cent" if cents else "")

    text = re.sub(r"€\s*(" + _NUMBER + r")(?:,(\d{1,2}))?(?!\d|,\d)", amount, text)
    text = re.sub(r"\b(" + _NUMBER + r"),(\d{1,2})\s+euro\b", amount, text, flags=re.I)

    def clock(m):
        h, minute = int(m[1]), int(m[2])
        if h > 23 or minute > 59:
            raise ValueError(f"ongeldige tijd {m[0]}")
        return cardinal(h) + " uur" + (" " + cardinal(minute) if minute else "")

    text = re.sub(r"\b(\d{1,2})[:.](\d{2})\s+uur\b", clock, text, flags=re.I)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", clock, text)
    text = re.sub(r"\b(" + _NUMBER + r"),(\d+)\b",
                  lambda m: cardinal(int(m[1].replace(".", ""))) + " komma " + " ".join(SMALL[int(d)] for d in m[2]), text)
    text = re.sub(r"(?<=\d)\s*%", " procent", text)
    text = re.sub(r"\b(1[1-9]\d\d)\b", lambda m: year(int(m[0])), text)
    text = re.sub(_NUMBER, lambda m: cardinal(int(m[0].replace(".", ""))), text)
    # Quotes have no acoustic token. Keep apostrophes in Dutch words and names.
    text = text.replace('"', "").replace("…", ".").replace("—", ",")
    return " ".join(text.split())


def split_key(spoken):
    return " ".join(re.findall(r"[^\W_]+", spoken.casefold(), flags=re.UNICODE))


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", spoken_text(text)) if split_key(s)]


@lru_cache(maxsize=1)
def require_runtime():
    release = espeak_version()
    if not re.search(r"(?<![\d.])1\.52\.0(?![\d.])", release):
        raise ValueError("the Alex front end requires eSpeak NG 1.52.0")
    return release


def phonemize(text):
    require_runtime()
    symbols, stress, unknown = [BOS], [0], []
    for piece in re.split(r"([,.?!;:])", spoken_text(text)):
        if piece in PUNCTUATION:
            if symbols[-1] != piece:
                symbols.append(piece)
                stress.append(0)
        elif piece.strip():
            for word in espeak_words(piece.strip(), voice=VOICE):
                if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
                    symbols.append(SP)
                    stress.append(0)
                for phone, level in word:
                    phone = unicodedata.normalize("NFC", phone)
                    if phone not in PHONE_SET:
                        unknown.append(phone)
                        phone = UNK
                    symbols.append(phone)
                    stress.append(level)
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise NothingToPronounce("geen uitspreekbare Nederlandse tekst")
    symbols.append(EOS)
    stress.append(0)
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(stress), tuple(unknown))


def teacher_units(phones):
    """Return (Piper symbol, student token index), including both edges.

    Piper includes the following internal pad in each symbol's sample count.
    A stress or length mark belongs to its phone. The space after an internal
    clause delimiter belongs to that delimiter. Word spaces have their own token.
    """
    if phones.unknown:
        raise ValueError(f"unknown Dutch phones: {phones.unknown}")
    units = []
    for i, (s, level) in enumerate(zip(phones.symbols, phones.stress)):
        if s in (BOS, EOS, SP):
            units.append(({BOS: "^", EOS: "$", SP: " "}[s], i))
        elif s in PUNCTUATION:
            units.append((s, i))
            if s in (",", ";", ":") and phones.symbols[i + 1] != EOS:
                units.append((" ", i))
        else:
            raw = ("", "ˈ", "ˌ")[level] + unicodedata.normalize("NFD", s)
            units.extend((char, i) for char in raw)
    return units


def identity():
    return {"language": "nl-NL", "phonemizer": "espeak-ng", "voice": VOICE, "espeak_release": "1.52.0",
            "separator": "_", "normalization_and_mapping_version": VERSION, "symbols": list(SYMBOLS)}
