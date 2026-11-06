"""The packaged Quran lattice (data/quran_lattice.json.gz, built by tools/build_artifact.py from quran-g2p).

Per ayah: the Tanzil Uthmani words, the KFGQPC imla'i words and their grouping, and per lattice word (a muqatta'at
text word holds one lattice word per letter name) every legal form with its boundary states:

    L  "i" ibtida (a stop before the word) | "w" wasl
    R  "q" waqf (a stop after the word)    | "w" wasl
    mode  "p" plain (R wasl) | "s" sukun | "r" rawm | "m" ishmam
    wajh  None, or the tag of the alternate wajh at a khilaf site

A first word's wasl forms continue the previous ayah (or the basmala before ayah 1); a last word's wasl forms run
into the next ayah. Within an ayah, a word whose right state is waqf is followed by a word in ibtida.
"""
from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "quran_lattice.json.gz")


@dataclass(frozen=True)
class Form:
    left: str            # "ibtida" | "wasl"
    right: str           # "waqf" | "wasl"
    mode: str            # "plain" | "sukun" | "rawm" | "ishmam"
    wajh: str | None
    tj3g: str
    units: str


@dataclass
class Word:
    surah: int
    ayah: int
    index: int           # lattice word index within the ayah
    text_index: int      # Uthmani (whitespace) word index within the ayah
    stop_after: bool     # a stop may follow this word
    forms: list
    basmala: bool = False


@dataclass
class Ayah:
    surah: int
    ayah: int
    uthmani: list
    imlaei: list
    groups: list         # [u0, u1, e0, e1] half-open groups of Uthmani / imla'i words
    words: list          # Word
    prev: str | None     # "prev" | "basmala" | None: where the first word's wasl forms come from
    has_next: bool
    sites: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.surah}:{self.ayah}"

    @property
    def text(self) -> str:
        return " ".join(self.uthmani)

    @property
    def text_imlaei(self) -> str:
        return " ".join(self.imlaei)


_MODES = {"p": "plain", "s": "sukun", "r": "rawm", "m": "ishmam"}


class Quran:
    """The 6,236 ayat. Use quran() for the shared instance."""

    def __init__(self, path: str = DATA):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        self.meta = {k: v for k, v in raw.items() if k != "ayat"}
        self._raw = {(r["surah"], r["ayah"]): r for r in raw["ayat"]}
        self._cache = {}
        self.n_ayat_in = {}
        for s, a in self._raw:
            self.n_ayat_in[s] = max(self.n_ayat_in.get(s, 0), a)

    def __len__(self):
        return len(self._raw)

    def keys(self):
        return list(self._raw)

    def ayah(self, surah: int, ayah: int) -> Ayah:
        k = (surah, ayah)
        if k not in self._cache:
            if k not in self._raw:
                raise KeyError(f"no ayah {surah}:{ayah}")
            r = self._raw[k]
            words = []
            for i, w in enumerate(r["words"]):
                forms = [Form("ibtida" if f[0] == "i" else "wasl", "waqf" if f[1] == "q" else "wasl", _MODES[f[2]],
                              f[3], f[4], f[5]) for f in w["f"]]
                words.append(Word(surah, ayah, i, w["t"], bool(w["s"]), forms))
            self._cache[k] = Ayah(surah, ayah, r["uth"], r["iml"], r["groups"], words, r["prev"], r["next"],
                                  r.get("sites", {}))
        return self._cache[k]

    def span(self, surah: int, first: int, last: int) -> list:
        return [self.ayah(surah, a) for a in range(first, last + 1)]

    def basmala_words(self) -> list:
        """The basmala's words (1:1) as an optional prefix before ayah 1 of a surah other than 1 and 9."""
        out = []
        for w in self.ayah(1, 1).words:
            out.append(Word(w.surah, w.ayah, w.index, w.text_index, w.stop_after, w.forms, basmala=True))
        return out

    def canonical(self, surah: int, ayah: int, fmt_key: str = "tj3g") -> list:
        """The canonical path of one ayah (ibtida at the start, waqf with sukun at the end, wasl inside, base wajh),
        tokens of the stored kind (tj3g or units)."""
        a = self.ayah(surah, ayah)
        n = len(a.words)
        out = []
        for k, w in enumerate(a.words):
            L = "ibtida" if k == 0 else "wasl"
            R = "waqf" if k == n - 1 else "wasl"
            mode = "sukun" if R == "waqf" else "plain"
            f = next(f for f in w.forms if f.left == L and f.right == R and f.mode == mode and f.wajh is None)
            out.extend(getattr(f, fmt_key).split())
        return out


@lru_cache(maxsize=1)
def quran() -> Quran:
    return Quran()


_REF = re.compile(r"^\s*(\d{1,3})\s*[:.]\s*(\d{1,3})(?:\s*-\s*(?:(\d{1,3})\s*[:.]\s*)?(\d{1,3}))?\s*$")


def parse_ref(ref) -> tuple:
    """'2:255' | '2:255-257' | '2:255-2:257' | (2, 255) | (2, 255, 257) -> (surah, first, last)."""
    if isinstance(ref, (tuple, list)):
        if len(ref) == 2:
            s, a = ref
            return int(s), int(a), int(a)
        s, a, b = ref
        return int(s), int(a), int(b)
    m = _REF.match(str(ref))
    if not m:
        raise ValueError(f"bad ayah reference {ref!r} (expected 's:a' or 's:a-b')")
    s, a = int(m.group(1)), int(m.group(2))
    if m.group(3) and int(m.group(3)) != s:
        raise ValueError(f"a range must stay inside one surah: {ref!r}")
    b = int(m.group(4)) if m.group(4) else a
    if b < a:
        raise ValueError(f"empty range {ref!r}")
    q = quran()
    if s not in q.n_ayat_in or a < 1 or b > q.n_ayat_in[s]:
        raise ValueError(f"no such ayah or range {ref!r}")
    return s, a, b


def ref_str(s: int, a: int, b: int) -> str:
    return f"{s}:{a}" if a == b else f"{s}:{a}-{b}"
