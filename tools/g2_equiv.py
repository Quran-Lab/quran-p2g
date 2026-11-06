"""G2 equivalence: when a planted edit can be attributed in more than one equally valid way.

A planted case counts as exact (strict) when quran-p2g reports exactly the planted statuses. It counts as exact up to
an equivalent attribution when the output is the planted statuses of another attribution that is proven equally
valid, mechanically:

  same_string  another edit of the same type on another word (another word skipped, another word repeated, the same
               consonant + vowel inserted at any position of another word) produces the identical phone string, so no
               phone string can tell the two attributions apart (e.g. which of two identical adjacent words was
               skipped, or which side of a word boundary an inserted syllable belongs to when it equals its
               neighbour)
  equal_cost   the planted attribution and the output cost the same under the aligner's cost (both are optimal
               explanations of the phones)

Nothing else is loosened: an output that is neither the planted statuses nor those of a proven equivalent stays a
miss. Read-back residuals are classified by named context rules in gate_g2.py, not here.
"""
from __future__ import annotations

SHORT = ("a", "i", "u")
LONG = ("aa", "ii", "uu")


def statuses_match(r, typ, ti, key):
    """r: a P2GResult. Does it report exactly the statuses of `typ` planted on Uthmani word ti?"""
    words = r.words
    flagged = {w.index for w in words if w.status != "match"}
    ins = r.insertions
    if typ == "word_repeat":
        return not flagged and len(ins) == 1 and ins[0].kind == "repeat" and ins[0].of == f"{key}:{ti}"
    want = "deleted" if typ == "word_skip" else "deviant"
    return flagged == {ti} and not ins and words[ti].status == want


def _concat(cw, k, new):
    return [t for j, w in enumerate(cw) for t in (new if j == k else w[0])]


def _inserted_pair(old, new):
    """new = old with two tokens inserted contiguously -> (position, the two tokens) or None."""
    if len(new) != len(old) + 2:
        return None
    for p in range(len(old) + 1):
        if new[:p] + new[p + 2:] == old:
            return p, new[p:p + 2]
    return None


def same_string_alternatives(cw, k, typ, new):
    """Uthmani word indices j != cw[k]'s word such that the same edit type on word j gives the identical phones.

    cw: canonical words [(tokens, legal forms, Uthmani index)], k: the edited lattice word, new: its planted tokens.
    """
    hyp = _concat(cw, k, new)
    out = set()
    ti = cw[k][2]
    for j, (toks, _, tj) in enumerate(cw):
        if tj == ti:
            continue
        if typ == "word_skip":
            cands = [[]]
        elif typ == "word_repeat":
            cands = [list(toks) + list(toks)]
        elif typ == "inserted_syllable":
            ins = _inserted_pair(list(cw[k][0]), list(new))
            if ins is None:
                continue
            pair = list(ins[1])
            cands = [list(toks[:p]) + pair + list(toks[p:]) for p in range(len(toks) + 1)]
        else:
            cands = []                       # a substitution changes one token, which belongs to one word only
        if any(_concat(cw, j, c) == hyp for c in cands):
            out.add(tj)
    return sorted(out)


def weighted_lev(a, b, fmt):
    """The aligner's within-word edit distance (insertion / deletion 4, formats.sub_cost)."""
    from quran_p2g.formats import sub_cost
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = 4 * i
    for j in range(m + 1):
        D[0][j] = 4 * j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 4, D[i][j - 1] + 4, D[i - 1][j - 1] + sub_cost(a[i - 1], b[j - 1], fmt))
    return D[n][m]


def planted_cost(word_forms, word_states, k, n_words, typ, canon, new, fmt):
    """Cost of the planted attribution under the aligner's cost: every other word on its canonical form (cost 0,
    no stop inside the ayah), word k skipped, repeated, or said as `new` against its best form with the canonical
    boundary states."""
    from quran_p2g.align import repeat_cost, skip_cost
    if typ == "word_skip":
        return skip_cost(min(len(t) for t in word_forms))
    if typ == "word_repeat":
        return repeat_cost(len(canon))
    L = "ibtida" if k == 0 else "wasl"
    R = "waqf" if k == n_words - 1 else "wasl"
    return min(weighted_lev(t, new, fmt) for t, st in zip(word_forms, word_states) if st == (L, R))
