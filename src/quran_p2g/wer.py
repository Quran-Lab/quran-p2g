"""Word error rate against the mushaf, with the Quranic ASR board's normalisation.

Undiacritized (diacritics=False): both sides go through the board's `strip` (every combining mark removed, the
Quranic annotation signs U+06D6..U+06ED removed, alef / yeh / teh marbuta / hamza seats folded), reimplemented here
from Quran-Lab/quranic-asr-benchmark score.py. The reference is the KFGQPC imla'i text (script="imlaei", the default)
or the Tanzil Uthmani text (script="uthmani"). A vowel-only slip does not count, as for a text model that writes no
harakat.

Diacritized (diacritics=True): the reference is the Tanzil Uthmani text; both sides keep their harakat, with only
the pause and annotation signs (U+06D6..U+06DE, U+06E9) and tatweel removed and NFC applied. It is meaningful for text
in the Uthmani script (what quran-p2g writes); another script's spelling conventions would count as errors.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .quran import parse_ref, quran

_TR = {"ٱ": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي", "ـ": "", "ٰ": ""}
_ANNOT = "".join(chr(x) for x in range(0x06D6, 0x06EE))
_PAUSE = set(chr(x) for x in range(0x06D6, 0x06DF)) | {"۩", "ـ"}


def strip(s: str) -> str:
    """The board's undiacritized normalisation (score.py `strip`)."""
    s = "".join(c for c in s if not unicodedata.combining(c))
    for c in _ANNOT:
        s = s.replace(c, "")
    return "".join(_TR.get(c, c) for c in s)


def strip_alef(s: str) -> str:
    """The board's WER(alef) normalisation: strip, then every alef removed."""
    return strip(s).replace("ا", "")


def normalize_diacritized(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return "".join(c for c in s if c not in _PAUSE)


@dataclass(frozen=True)
class WerResult:
    errors: int
    ref_words: int
    hyp_words: int

    @property
    def wer(self) -> float:
        return self.errors / self.ref_words if self.ref_words else 0.0

    def __add__(self, o):
        return WerResult(self.errors + o.errors, self.ref_words + o.ref_words, self.hyp_words + o.hyp_words)


def word_edit_distance(r: list, h: list) -> int:
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return dp[-1]


def reference_text(ref_ayah, script: str = "imlaei") -> str:
    s, a0, a1 = parse_ref(ref_ayah)
    q = quran()
    parts = []
    for a in range(a0, a1 + 1):
        ay = q.ayah(s, a)
        parts.append(ay.text_imlaei if script == "imlaei" else ay.text)
    return " ".join(parts)


def wer(hyp, ref_ayah, diacritics: bool = False, script: str | None = None, alef: bool = False) -> WerResult:
    """hyp: a text, or a P2GResult (its text_imlaei is used undiacritized, its Uthmani text diacritized).
    ref_ayah: "s:a" or "s:a-b". alef=True: the board's WER(alef) variant (undiacritized only)."""
    if script is None:
        script = "uthmani" if diacritics else "imlaei"
    if script not in ("imlaei", "uthmani"):
        raise ValueError("script must be 'imlaei' or 'uthmani'")
    if diacritics and script != "uthmani":
        raise ValueError("diacritized WER is defined against the Uthmani text only")
    if not isinstance(hyp, str):
        if diacritics:
            hyp = hyp.text
        elif script == "imlaei":
            hyp = hyp.text_imlaei
        else:
            hyp = hyp.text_uthmani_plain
    ref = reference_text(ref_ayah, script)
    if diacritics:
        R, H = normalize_diacritized(ref).split(), normalize_diacritized(hyp).split()
    else:
        n = strip_alef if alef else strip
        R, H = n(ref).split(), n(hyp).split()
    return WerResult(word_edit_distance(R, H), len(R), len(H))


def corpus_wer(results) -> WerResult:
    tot = WerResult(0, 0, 0)
    for r in results:
        tot = tot + r
    return tot
