"""Constrained alignment of a hypothesis against a span's legal-reading lattice.

One form per word, boundary states agreeing (a word in waqf is followed by a word in ibtida), weighted edit
distance (insertion / deletion 4, substitution 2-6, see formats.sub_cost), solved left to right with one numpy row per
form token. Tie-breaks and word-level operations, all in the same units:

  stop        a stop inside the span costs STOP (1); among paths of equal cost the one with fewer stops wins (a
              second, lexicographic criterion), so a stop never decides between two equally good explanations
  skip        a word not said at all costs 2 * len + 1 (len = its shortest legal form): cheaper than deleting its
              tokens one by one (4 each), and cheaper than a stop plus the word said as a one-token fragment
              borrowed from a neighbour's wasl ending, so a skipped word is reported deleted; the same for every
              form, so the skip does not pick a boundary state
  repeat      a run of hypothesis tokens equal to a legal form of a nearby word (the previous REPEAT_BACK words or
              the next word) costs 2 * len + 1, cheaper than absorbing it into a neighbour (4 per token, or a stop
              that lets the neighbour take part of the run in its other form)

An optional prefix (the basmala before ayah 1) may be read or not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .formats import sub_cost

INS = DEL = 4
STOP = 1
REPEAT_BACK = 8
# Internally every cost is K * (cost in the units above) + (number of stops): the stop count is a pure tie-break
# that never outweighs one unit of cost. Reported costs are in the units above.
K = 1 << 12
_INS = _DEL = INS * K
_STOP = STOP * K + 1
INF = 1 << 60


def skip_cost(n_tokens: int) -> int:
    """A word not said at all (n_tokens: its shortest legal form)."""
    return 2 * n_tokens + 1


def repeat_cost(n_tokens: int) -> int:
    """An inserted repeat of a nearby word's legal form of n_tokens tokens."""
    return 2 * n_tokens + 1
S_STOP, WASL = "stop", "wasl"


@dataclass
class WordAlign:
    word: int                     # index in the aligned word list
    form: int                     # index into that word's forms
    j0: int
    j1: int
    ops: list                     # (op, form token | None, hyp token | None), op in eq / sub / del / ins
    cost: int
    skipped: bool = False


@dataclass
class Insertion:
    boundary: int                 # inserted before word `boundary` (len(words) = after the last word)
    j0: int
    j1: int
    kind: str                     # "repeat" | "edge"
    of_word: int | None = None    # the word a repeat repeats


@dataclass
class Alignment:
    cost: int
    words: list                   # WordAlign, one per word (prefix words only when the prefix was read)
    insertions: list = field(default_factory=list)
    prefix_read: bool = False
    stops: int = 0                # stops inside the span on the chosen path


class _Costs:
    def __init__(self, hyp, fmt):
        self.fmt = fmt
        self.hyp = list(hyp)
        self.m = len(self.hyp)
        self.ar = _INS * np.arange(self.m + 1, dtype=np.int64)
        self._rows = {}

    def row(self, tok):
        r = self._rows.get(tok)
        if r is None:
            r = np.array([K * sub_cost(tok, h, self.fmt) for h in self.hyp], dtype=np.int64)
            self._rows[tok] = r
        return r

    def chain(self, v):
        return np.minimum.accumulate(v - self.ar) + self.ar

    def lev(self, row0, toks):
        mat = np.empty((len(toks) + 1, self.m + 1), dtype=np.int64)
        mat[0] = self.chain(row0)
        for i, x in enumerate(toks, 1):
            prev = mat[i - 1]
            t = np.empty(self.m + 1, dtype=np.int64)
            t[0] = prev[0] + _DEL
            if self.m:
                t[1:] = np.minimum(prev[1:] + _DEL, prev[:-1] + self.row(x))
            mat[i] = self.chain(t)
        return np.minimum(mat, INF)

    def back(self, mat, row0, toks, j):
        """Backtrace one word from (len(toks), j) -> (j0, ops)."""
        i, ops = len(toks), []
        hyp = self.hyp
        while i > 0 or (j > 0 and mat[0][j] != row0[j]):
            if i > 0 and j > 0:
                s = K * sub_cost(toks[i - 1], hyp[j - 1], self.fmt)
                if mat[i][j] == mat[i - 1][j - 1] + s:
                    ops.append(("eq" if s == 0 else "sub", toks[i - 1], hyp[j - 1]))
                    i -= 1
                    j -= 1
                    continue
            if i > 0 and mat[i][j] == mat[i - 1][j] + _DEL:
                ops.append(("del", toks[i - 1], None))
                i -= 1
                continue
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
        ops.reverse()
        return j, ops


def _repeat_matches(hyp, words_toks):
    """start i -> [(length, {word indices whose legal forms equal hyp[i:i+length]})]."""
    by = {}
    for w, forms in enumerate(words_toks):
        for t in forms:
            if t:
                by.setdefault(t, set()).add(w)
    lens = sorted({len(t) for t in by})
    out = [[] for _ in range(len(hyp) + 1)]
    for i in range(len(hyp)):
        for L in lens:
            if i + L > len(hyp):
                break
            ws = by.get(tuple(hyp[i:i + L]))
            if ws:
                out[i].append((L, ws))
    return out


def _closure(vec, matches, lo, hi):
    """Repeat insertions at one boundary: words lo..hi may be repeated. Returns (vec, pred_i, pred_w)."""
    vec = vec.copy()
    n = len(vec)
    pred_i = np.full(n, -1, dtype=np.int64)
    pred_w = np.full(n, -1, dtype=np.int64)
    for i in range(n - 1):
        if vec[i] >= INF or not matches[i]:
            continue
        for L, ws in matches[i]:
            cand = [w for w in ws if lo <= w <= hi]
            if not cand:
                continue
            c = vec[i] + K * repeat_cost(L)
            if c <= vec[i + L]:                 # ties: the later copy is the repeat
                vec[i + L] = c
                pred_i[i + L] = i
                pred_w[i + L] = max(cand)
    return vec, pred_i, pred_w


def align(hyp, words_toks, words_states, fmt, *, n_prefix=0, open_left=False, open_right=False,
          free_edges=False, repeats=True) -> Alignment:
    """hyp: tokens. words_toks[w]: the token tuple of each form of word w; words_states[w]: (left, right) per form.
    The first n_prefix words are an optional prefix (read in full or skipped at no cost)."""
    C = _Costs(hyp, fmt)
    m = C.m
    n = len(words_toks)
    matches = _repeat_matches(hyp, words_toks) if repeats else [[] for _ in range(m + 1)]
    start = np.zeros(m + 1, dtype=np.int64) if free_edges else np.full(m + 1, INF, dtype=np.int64)
    start[0] = 0
    init = {S_STOP: start}
    if open_left and n_prefix == 0:
        init[WASL] = start.copy()
    bounds = []          # per boundary b: {state: (post_vec, pred_i, pred_w)}
    choices = []         # per word w: {dst: form index array}
    bypass = None

    def close(vecs, b):
        out = {}
        for s, v in vecs.items():
            if repeats:
                out[s] = _closure(v, matches, max(0, b - REPEAT_BACK), min(n - 1, b))
            else:
                out[s] = (v, np.full(m + 1, -1), np.full(m + 1, -1))
        return out

    bounds.append(close(init, 0))
    for w in range(n):
        new, choice = {}, {}
        skip = K * skip_cost(min(len(t) for t in words_toks[w]))
        for fi, (toks, (L, R)) in enumerate(zip(words_toks[w], words_states[w])):
            src = S_STOP if L == "ibtida" else WASL
            if src not in bounds[w]:
                continue
            row0 = bounds[w][src][0]
            pen = _STOP if (R == "waqf" and w < n - 1) else 0
            last = C.lev(row0, toks)[-1] + pen
            skip_v = row0 + skip + pen
            dst = S_STOP if R == "waqf" else WASL
            for cand, code in ((last, fi), (skip_v, -(fi + 1))):
                if dst not in new:
                    new[dst] = cand.copy()
                    choice[dst] = np.full(m + 1, code, dtype=np.int64)
                else:
                    better = cand < new[dst]
                    new[dst] = np.where(better, cand, new[dst])
                    choice[dst] = np.where(better, code, choice[dst])
        choices.append(choice)
        if w + 1 == n_prefix:
            # the prefix may be skipped: the ayah's first word then starts from the initial state
            stop = new.get(S_STOP, np.full(m + 1, INF, dtype=np.int64))
            bypass = start < stop
            new[S_STOP] = np.where(bypass, start, stop)
        bounds.append(close(new, w + 1))
    ends = {s: v[0] for s, v in bounds[n].items() if s == S_STOP or (s == WASL and open_right)}
    if not ends:
        raise ValueError("no legal path through the lattice")
    best_s, best_j, best = None, None, INF + 1
    for s, v in ends.items():
        if free_edges:
            j = int(m - np.argmin(v[::-1]))            # the last minimum: fewest free trailing tokens
            j = j if v[j] == v.min() else int(np.argmin(v))
        else:
            j = m
        if v[j] < best:
            best_s, best_j, best = s, j, int(v[j])
    res_words, ins = [], []
    if free_edges and best_j < m:
        ins.append(Insertion(n, best_j, m, "edge"))
    s, j, b = best_s, best_j, n
    prefix_read = n_prefix > 0
    while True:
        post, pred_i, pred_w = bounds[b][s]
        while pred_i[j] >= 0:
            ins.append(Insertion(b, int(pred_i[j]), j, "repeat", int(pred_w[j])))
            j = int(pred_i[j])
        if b == n_prefix and n_prefix and bypass is not None and s == S_STOP and bypass[j]:
            prefix_read = False
            b = 0
            break
        if b == 0:
            break
        w = b - 1
        code = int(choices[w][s][j])
        fi = code if code >= 0 else -code - 1
        toks = words_toks[w][fi]
        L = words_states[w][fi][0]
        src = S_STOP if L == "ibtida" else WASL
        row0 = bounds[w][src][0]
        if code < 0:
            res_words.append(WordAlign(w, fi, j, j, [("del", t, None) for t in toks],
                                       skip_cost(min(len(t) for t in words_toks[w])), skipped=True))
        else:
            mat = C.lev(row0, toks)
            j_end = j
            j0, ops = C.back(mat, row0, toks, j)
            res_words.append(WordAlign(w, fi, j0, j_end, ops, int(mat[-1][j_end] - row0[j0]) // K))
            j = j0
        s, b = src, w
    if j > 0:
        ins.append(Insertion(0, 0, j, "edge"))
    res_words.reverse()
    if not prefix_read:
        res_words = [x for x in res_words if x.word >= n_prefix]
    ins.sort(key=lambda x: (x.j0, x.j1))
    return Alignment(best // K, res_words, ins, prefix_read, stops=best % K)
