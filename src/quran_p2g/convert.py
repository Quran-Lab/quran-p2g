"""Constrained conversion: phones + an ayah (or range) -> per-word statuses and text.

Statuses (per Uthmani word of the reference):
  match     the word's phones are one of its legal forms on the chosen reading path
  deviant   phones were said for the word but they are none of its legal forms
  deleted   no phones were said for the word
and inserted items between words: a repeat of a nearby word, the basmala before ayah 1, or (free_edges) material
before the first / after the last word.

A deviant word gets a level, the most visible kind of difference from the legal form it was aligned to:
  tajweed   only how a sakin noon / tanwin / meem was realised (ikhfa, iqlab, idgham against izhar): no text shows it
  harakat   short vowels and shadda: a diacritized text shows it, an undiacritized one does not
  letters   anything else (a consonant, a long vowel, an added or dropped syllable)
`text` (Uthmani, diacritized) keeps the mushaf word for a tajweed-level deviation and spells the rest; `text_imlaei`
(undiacritized) keeps the imla'i word up to the harakat level. This is what a text transcriber would have written.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import align as AL
from .formats import IKHFA, LONG, SHORT, check_format, form_tokens, sub_cost, to_core, tokenize
from .quran import parse_ref, quran, ref_str
from .spell import Spelling, _plain, spell_core

LEVELS = ("tajweed", "harakat", "letters")
_NASALS = {"ن", IKHFA, "م"}


@dataclass
class WordResult:
    ayah: str
    index: int                 # Uthmani word index within the ayah
    uthmani: str
    imlaei: str
    status: str                # match | deviant | deleted
    phones: list               # hypothesis tokens aligned to the word (matching space)
    expected: list             # the legal form it was aligned to
    form: dict                 # {"left", "right", "mode", "wajh"} of that form
    level: str | None = None
    spelling: Spelling | None = None
    edits: list = field(default_factory=list)
    text: str = ""
    text_imlaei: str = ""

    def to_dict(self):
        d = {"kind": "word", "ayah": self.ayah, "index": self.index, "status": self.status,
             "uthmani": self.uthmani, "imlaei": self.imlaei, "phones": " ".join(self.phones),
             "expected": " ".join(self.expected), "form": self.form}
        if self.status == "deviant":
            d["level"] = self.level
            d["spelling"] = {"diacritized": self.spelling.diacritized, "plain": self.spelling.plain}
            d["edits"] = self.edits
        return d


@dataclass
class Inserted:
    kind: str                  # repeat | basmala | edge
    phones: list
    after: str | None          # "s:a:w" of the word it follows (None at the start)
    of: str | None = None      # repeat: "s:a:w" of the word repeated
    status: str = "inserted"
    text: str = ""
    text_imlaei: str = ""
    match: bool | None = None  # basmala: read exactly

    def to_dict(self):
        d = {"kind": self.kind, "status": "inserted", "phones": " ".join(self.phones), "after": self.after,
             "text": self.text}
        if self.of:
            d["of"] = self.of
        if self.match is not None:
            d["match"] = self.match
        return d


@dataclass
class P2GResult:
    ref: str
    fmt: str
    items: list                # WordResult and Inserted, in reading order
    cost: int
    retrieval: dict | None = None

    @property
    def words(self):
        return [x for x in self.items if isinstance(x, WordResult)]

    @property
    def insertions(self):
        return [x for x in self.items if isinstance(x, Inserted)]

    @property
    def text(self) -> str:
        return " ".join(x.text for x in self.items if x.text and x.kind_in_text())

    @property
    def text_imlaei(self) -> str:
        return " ".join(x.text_imlaei for x in self.items if x.text_imlaei and x.kind_in_text())

    @property
    def text_uthmani_plain(self) -> str:
        """Uthmani words, keeping the mushaf word up to the harakat level (for undiacritized Uthmani WER)."""
        out = []
        for x in self.items:
            if not x.kind_in_text():
                continue
            if isinstance(x, WordResult) and x.status == "deviant":
                out.append(x.uthmani if x.level in ("tajweed", "harakat") else x.spelling.plain)
            elif x.text:
                out.append(x.text)
        return " ".join(out)

    @property
    def counts(self) -> dict:
        c = {"match": 0, "deviant": 0, "deleted": 0, "inserted": 0}
        for x in self.items:
            if isinstance(x, WordResult):
                c[x.status] += 1
            elif x.kind != "basmala":
                c["inserted"] += 1
        return c

    def to_dict(self, words=True):
        d = {"ayah": self.ref, "format": self.fmt, "text": self.text, "text_imlaei": self.text_imlaei,
             "counts": self.counts}
        if self.retrieval is not None:
            d["retrieval"] = self.retrieval
        if words:
            d["items"] = [x.to_dict() for x in self.items]
        return d


def _in_text(self):
    return not (isinstance(self, Inserted) and self.kind == "basmala")


WordResult.kind_in_text = _in_text
Inserted.kind_in_text = _in_text


# ------------------------------------------------------------------------------------------------ deviation level

def _wlev_ops(a, b):
    """Weighted alignment ops of core token lists a (expected) -> b (said)."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = 4 * i
    for j in range(1, m + 1):
        D[0][j] = 4 * j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 4, D[i][j - 1] + 4, D[i - 1][j - 1] + sub_cost(a[i - 1], b[j - 1], "tj3g"))
    ops, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + sub_cost(a[i - 1], b[j - 1], "tj3g"):
            ops.append(("eq" if a[i - 1] == b[j - 1] else "sub", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and D[i][j] == D[i - 1][j] + 4:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def _base(t):
    return t[0] if t not in SHORT and t not in LONG else t


def _unvowelled(seq, k):
    return k + 1 >= len(seq) or (seq[k + 1] not in SHORT and seq[k + 1] not in LONG)


def deviation_level(expected_core, said_core):
    """-> (level, edits): the most visible class of the differences, and the edits as [op, expected, said]."""
    ops = _wlev_ops(expected_core, said_core)
    lv, edits = -1, []
    for op, i, j in ops:
        if op == "eq":
            continue
        e = expected_core[i] if i is not None else None
        s = said_core[j] if j is not None else None
        edits.append([op, e, s])
        if op == "sub":
            if e in SHORT and s in SHORT:
                k = 1
            elif _base(e) == _base(s) and e not in SHORT + LONG:
                k = 1                                            # held vs plain: shadda
            elif e[0] in _NASALS and s[0] in _NASALS and IKHFA in (e, s):
                k = 0
            elif {_base(e), _base(s)} == {"ن", "م"} and _unvowelled(expected_core, i) \
                    and _unvowelled(said_core, j):
                k = 0                                            # iqlab against izhar
            else:
                k = 2
        else:
            t, seq, pos = (e, expected_core, i) if op == "del" else (s, said_core, j)
            if t in SHORT:
                k = 1
            elif t[0] in ("ن", IKHFA) and _unvowelled(seq, pos):
                k = 0                                            # tanwin / sakin noon assimilated or not
            elif t not in LONG and ((pos > 0 and _base(seq[pos - 1]) == _base(t)) or
                                    (pos + 1 < len(seq) and _base(seq[pos + 1]) == _base(t))):
                k = 1                                            # a doubled consonant: shadda
            else:
                k = 2
        lv = max(lv, k)
    return (LEVELS[lv] if lv >= 0 else "tajweed"), edits


# ------------------------------------------------------------------------------------------------ conversion

_PREP = {}


def _prep(word, fmt):
    key = (word.surah, word.ayah, word.index, word.basmala, fmt)
    p = _PREP.get(key)
    if p is None:
        toks, states, idx, seen = [], [], [], set()
        for fi, f in enumerate(word.forms):
            t = form_tokens(f.tj3g, f.units, fmt)
            k = (t, f.left, f.right)
            if k in seen:
                continue
            seen.add(k)
            toks.append(t)
            states.append((f.left, f.right))
            idx.append(fi)
        p = (toks, states, idx)
        _PREP[key] = p
    return p


def convert(tokens, ayah=None, fmt: str = "tj3", *, open_edges: bool = True, basmala: bool = True,
            free_edges: bool = False, repeats: bool = True) -> P2GResult:
    """Phones -> text and per-word statuses.

    tokens: a string (whitespace-separated tokens; the muaalem script as is) or a token list.
    ayah:   "s:a", "s:a-b" (a range inside one surah), (s, a), or None to identify the ayah first.
    open_edges: the first word may continue the previous ayah in wasl and the last may run into the next.
    basmala: before ayah 1 of a surah other than 1 and 9 the basmala may be read (reported, not in the text).
    free_edges: material before the first / after the last word is reported as inserted at no cost (clips that
                hold part of a neighbouring ayah).
    """
    check_format(fmt)
    hyp = tokenize(tokens, fmt)
    retrieval = None
    if ayah is None:
        from .retrieve import identify
        cands = identify(hyp, fmt=fmt, tokenized=True)
        if not cands:
            raise ValueError("could not identify an ayah for an empty hypothesis")
        best = cands[0]
        ayah = (best.surah, best.first, best.last)
        retrieval = {"ayah": best.ref, "score": round(best.score, 4), "margin": round(best.margin, 4),
                     "identical": best.identical, "candidates": [c.ref for c in cands[:5]]}
    s, a0, a1 = parse_ref(ayah)
    q = quran()
    ayat = q.span(s, a0, a1)
    words = []
    pre = q.basmala_words() if (basmala and a0 == 1 and s not in (1, 9)) else []
    words.extend(pre)
    for ay in ayat:
        words.extend(ay.words)
    preps = [_prep(w, fmt) for w in words]
    al = AL.align(hyp, [p[0] for p in preps], [p[1] for p in preps], fmt, n_prefix=len(pre),
                  open_left=open_edges and ayat[0].prev == "prev", open_right=open_edges and ayat[-1].has_next,
                  free_edges=free_edges, repeats=repeats)
    return _build(hyp, fmt, words, preps, al, ayat, len(pre), ref_str(s, a0, a1), retrieval)


def _wkey(w):
    return f"{w.surah}:{w.ayah}:{w.text_index}"


def _build(hyp, fmt, words, preps, al, ayat, n_pre, ref, retrieval):
    by_word = {x.word: x for x in al.words}
    ins_at = {}
    for ins in al.insertions:
        ins_at.setdefault(ins.boundary, []).append(ins)
    items = []

    def emit_insertions(b, after):
        for ins in ins_at.get(b, []):
            ph = hyp[ins.j0:ins.j1]
            if ins.kind == "repeat":
                ow = words[ins.of_word]
                it = Inserted("repeat", ph, after, of=_wkey(ow))
                if ow.basmala:
                    it.text = it.text_imlaei = ""
                    it.kind = "basmala"
                else:
                    ay = next(x for x in ayat if (x.surah, x.ayah) == (ow.surah, ow.ayah))
                    single = sum(1 for w in ay.words if w.text_index == ow.text_index) == 1
                    if single:
                        it.text = ay.uthmani[ow.text_index]
                        it.text_imlaei = _imlaei_of(ay, ow.text_index)
                    else:
                        sp = spell_core(to_core(ph, fmt))
                        it.text, it.text_imlaei = sp.diacritized, sp.plain
            else:
                sp = spell_core(to_core(ph, fmt))
                it = Inserted("edge", ph, after, text=sp.diacritized, text_imlaei=sp.plain)
            items.append(it)

    # the basmala prefix
    if n_pre and al.prefix_read:
        ph, ok = [], True
        for w in range(n_pre):
            x = by_word[w]
            ph.extend(hyp[x.j0:x.j1])
            ok = ok and tuple(hyp[x.j0:x.j1]) == preps[w][0][x.form]
        items.append(Inserted("basmala", ph, None, match=ok))
    for b in range(0, n_pre + 1):
        emit_insertions(b, None)
    # ayah words grouped by Uthmani word
    w = n_pre
    for ay in ayat:
        groups = {}
        for lw in ay.words:
            groups.setdefault(lw.text_index, []).append(lw)
        results = []
        for ti in sorted(groups):
            lws = groups[ti]
            ph, exp, core_e, core_s, statuses, form = [], [], [], [], [], None
            for lw in lws:
                x = by_word[w]
                toks, states, idx = preps[w]
                f = lw.forms[idx[x.form]]
                said = hyp[x.j0:x.j1]
                ph.extend(said)
                exp.extend(toks[x.form])
                core_e.extend(to_core(list(toks[x.form]), fmt))
                core_s.extend(to_core(list(said), fmt))
                statuses.append("deleted" if not said else ("match" if tuple(said) == toks[x.form] else "deviant"))
                form = {"left": f.left, "right": f.right, "mode": f.mode, "wajh": f.wajh} if form is None else \
                    dict(form, right=f.right, mode=f.mode)
                w += 1
                if lw is not lws[-1]:
                    emit_insertions(w, _wkey(lw))
            if all(st == "match" for st in statuses):
                status = "match"
            elif all(st == "deleted" for st in statuses):
                status = "deleted"
            else:
                status = "deviant"
            r = WordResult(ay.key, ti, ay.uthmani[ti], _imlaei_of(ay, ti), status, ph, exp, form)
            if status == "deviant":
                r.level, r.edits = deviation_level(core_e, core_s)
                r.spelling = spell_core(core_s, ay.uthmani[ti], geminates_known=fmt in ("tj3g", "units"),
                                        waqf=form["right"] == "waqf")
            r.text = "" if status == "deleted" else (
                r.uthmani if status == "match" or r.level == "tajweed" else r.spelling.diacritized)
            results.append(r)
            items.append(r)
            emit_insertions(w, _wkey(lws[-1]))
        _imlaei_texts(ay, results)
    return P2GResult(ref, fmt, items, al.cost, retrieval)


def _imlaei_of(ay, ti):
    for u0, u1, e0, e1 in ay.groups:
        if u0 <= ti < u1:
            return " ".join(ay.imlaei[e0:e1])
    return ""


def _imlaei_texts(ay, results):
    """Undiacritized imla'i text per Uthmani word, by imla'i group (1:1, 1:n, n:1)."""
    by = {r.index: r for r in results}
    for u0, u1, e0, e1 in ay.groups:
        rs = [by[i] for i in range(u0, u1)]
        ok = all(r.status == "match" or (r.status == "deviant" and r.level in ("tajweed", "harakat")) for r in rs)
        if ok:
            rs[0].text_imlaei = " ".join(ay.imlaei[e0:e1])
            for r in rs[1:]:
                r.text_imlaei = ""
            continue
        for r in rs:
            if r.status == "deleted":
                r.text_imlaei = ""
            elif r.status == "deviant" and r.level == "letters":
                r.text_imlaei = r.spelling.plain
            elif u1 - u0 == 1:
                r.text_imlaei = " ".join(ay.imlaei[e0:e1])
            else:
                r.text_imlaei = _plain(r.uthmani)
