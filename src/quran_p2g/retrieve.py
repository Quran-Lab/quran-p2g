"""Unconstrained mode: which ayah (or run of ayat) was recited?

Every ayah's canonical path is folded (long vowels to short, the ikhfa nasal to noon, held consonants to plain) so a
madd or ghunna slip does not move the search, and indexed by token 4-grams. A hypothesis gathers votes from its
4-grams; the best-voted ayat are rescored by edit distance against the hypothesis, growing a candidate into a range
of neighbouring ayat while the hypothesis is longer than the candidate. Score = 1 - distance / max(len); margin = the
gap to the best candidate whose folded text differs (ayat with identical text, e.g. 55:13 and 55:16, are reported as
`identical`, not as competitors).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .formats import IKHFA, TO_SHORT, check_format, degeminate, to_core, tokenize
from .quran import quran, ref_str

N = 4
TOP = 12
MAX_GROW = 6


@dataclass
class Candidate:
    surah: int
    first: int
    last: int
    score: float
    margin: float = 0.0
    identical: list = field(default_factory=list)

    @property
    def ref(self) -> str:
        return ref_str(self.surah, self.first, self.last)


def fold(core) -> list:
    out = []
    for t in core:
        t = degeminate(t)
        t = TO_SHORT.get(t, t)
        if t == IKHFA:
            t = "ن"
        out.append(t)
    return out


class Index:
    def __init__(self):
        q = quran()
        self.keys = sorted(q.keys())
        self.pos = {k: i for i, k in enumerate(self.keys)}
        self.seqs = [fold(q.canonical(s, a)) for s, a in self.keys]
        sym = {}
        self.codes = []
        for sq in self.seqs:
            self.codes.append(np.array([sym.setdefault(t, len(sym) + 1) for t in sq], dtype=np.int64))
        self.sym = sym
        post = {}
        for i, c in enumerate(self.codes):
            for g in set(self._grams(c)):
                post.setdefault(g, []).append(i)
        self.post = {g: np.array(v, dtype=np.int64) for g, v in post.items()}
        self.text_of = {}
        for i, c in enumerate(self.codes):
            self.text_of.setdefault(c.tobytes(), []).append(i)

    @staticmethod
    def _grams(c):
        if len(c) < N:
            return [tuple(c.tolist())] if len(c) else []
        return [tuple(c[k:k + N].tolist()) for k in range(len(c) - N + 1)]

    def encode(self, folded):
        return np.array([self.sym.get(t, 0) for t in folded], dtype=np.int64)


@lru_cache(maxsize=1)
def index() -> Index:
    return Index()


def _lev(a, b):
    """Unit-cost edit distance of two int arrays (numpy rows)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return max(n, m)
    ar = np.arange(m + 1, dtype=np.int64)
    row = ar.copy()
    for i in range(1, n + 1):
        t = np.empty(m + 1, dtype=np.int64)
        t[0] = i
        t[1:] = np.minimum(row[1:] + 1, row[:-1] + (b != a[i - 1]))
        row = np.minimum.accumulate(t - ar) + ar
    return int(row[-1])


def identify(tokens, fmt: str = "tj3", k: int = 5, tokenized: bool = False) -> list:
    """Hypothesis -> up to k Candidates, best first."""
    check_format(fmt)
    hyp = tokens if tokenized else tokenize(tokens, fmt)
    ix = index()
    h = ix.encode(fold(to_core(list(hyp), fmt)))
    if len(h) == 0:
        return []
    votes = np.zeros(len(ix.keys), dtype=np.float64)
    grams = set(ix._grams(h))
    for g in grams:
        p = ix.post.get(g)
        if p is not None:
            votes[p] += 1.0 / np.log2(2 + len(p))        # rare grams weigh more
    if not votes.any():
        lens = np.array([abs(len(c) - len(h)) for c in ix.codes])
        cand = list(np.argsort(lens)[:TOP])
    else:
        cand = [int(i) for i in np.argsort(-votes)[:TOP] if votes[i] > 0]
    scored = {}
    for i in cand:
        s, a = ix.keys[i]
        lo = hi = i
        best = (_score(h, ix, lo, hi), lo, hi)
        grown = 0
        while grown < MAX_GROW and sum(len(ix.codes[x]) for x in range(lo, hi + 1)) < len(h):
            opts = []
            if hi + 1 < len(ix.keys) and ix.keys[hi + 1][0] == s:
                opts.append((lo, hi + 1))
            if lo - 1 >= 0 and ix.keys[lo - 1][0] == s:
                opts.append((lo - 1, hi))
            if not opts:
                break
            sc = [(_score(h, ix, x, y), x, y) for x, y in opts]
            sc.sort(key=lambda t: (-t[0], t[1]))
            if sc[0][0] < best[0]:
                break
            best = sc[0]
            lo, hi = sc[0][1], sc[0][2]
            grown += 1
        key = (best[1], best[2])
        scored[key] = max(scored.get(key, -1e9), best[0])
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0][1] - kv[0][0], kv[0][0]))
    out = []
    for (lo, hi), sc in ranked:
        s = ix.keys[lo][0]
        c = Candidate(s, ix.keys[lo][1], ix.keys[hi][1], sc)
        if lo == hi:
            c.identical = [f"{x[0]}:{x[1]}" for x in (ix.keys[j] for j in ix.text_of[ix.codes[lo].tobytes()])
                           if x != ix.keys[lo]]
        out.append(c)
    if out:
        top = out[0]
        txt = _concat(ix, *_bounds(ix, top))
        rival = [c.score for c in out[1:] if not np.array_equal(_concat(ix, *_bounds(ix, c)), txt)]
        top.margin = top.score - (max(rival) if rival else 0.0)
    return out[:k]


def _bounds(ix, c):
    return ix.pos[(c.surah, c.first)], ix.pos[(c.surah, c.last)]


def _concat(ix, lo, hi):
    return np.concatenate([ix.codes[x] for x in range(lo, hi + 1)])


def _score(h, ix, lo, hi):
    ref = _concat(ix, lo, hi)
    return 1.0 - _lev(ref, h) / max(len(ref), len(h))
