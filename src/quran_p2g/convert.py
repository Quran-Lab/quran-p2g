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


def _before_ba(seq, k, next_core):
    """iqlab context: the sound after position k is ba (in the word, or the next word's first sound). 0.1.2: an
    unvowelled noon / meem swap anywhere else is a letter mistake (e.g. kaannahum read kaannahun before lu'lu')."""
    nxt = seq[k + 1] if k + 1 < len(seq) else (next_core[0] if next_core else None)
    return nxt is not None and _base(nxt) == "ب"


def _unvowelled(seq, k):
    return k + 1 >= len(seq) or (seq[k + 1] not in SHORT and seq[k + 1] not in LONG)


_IDGHAM = set("يرملون")
_IKHFA_LETTERS = set("تثجدذزسشصضطظفقك")
_TASHKEEL = "ًٌٍَُِّْٰٕٓٔۖۗۘۙۚ" \
            "ۛۜ۝۞ۣ۟۠ۡۢۤۥۦۧۨ۩۪ۭ۫۬"


def _next_sound(seq, k, next_core):
    nxt = seq[k + 1] if k + 1 < len(seq) else (next_core[0] if next_core else None)
    return _base(nxt) if nxt is not None else None


def _written_iqlab_site(written):
    """the written word holds a noon or tanween that iqlab can turn into a meem sound: a noon followed by ba inside the
    word, or a final noon / tanween (iqlab across the boundary). 0.1.2: a meem that is WRITTEN as meem (lakum) read as
    noon is a letter mistake (lakun), not iqlab."""
    if written is None:
        return False
    core = "".join(c for c in written if c not in _TASHKEEL or c in "ًٌٍ")
    return "نب" in core or core.endswith(("ن", "ً", "ٌ", "ٍ", "ًا", "ًى"))


def _tanween_end(written):
    """the word ends in tanween (its noon sound is not a written letter); unknown spelling counts as tanween, as before"""
    if written is None:
        return True
    core = "".join(c for c in written if c not in _TASHKEEL or c in "ًٌٍ")
    return core.endswith(("ً", "ٌ", "ٍ", "ًا", "ًى"))


_VOWEL_MARKS = set("ًٌٍَُِّ")     # fatha damma kasra shadda tanween


def _sakin_noon_before(written, letter):
    """the written word has a noon with no vowel or shadda on it, followed by `letter` (None: the noon ends the word)"""
    for i, c in enumerate(written):
        if c != "ن":
            continue
        j, marks = i + 1, set()
        while j < len(written) and written[j] in _TASHKEEL:
            marks.add(written[j])
            j += 1
        if marks & _VOWEL_MARKS:
            continue
        nxt = written[j] if j < len(written) else None
        if nxt == letter:
            return True
    return False


def _ikhfa_context(e, s, seq, i, next_core, written):
    """ikhfa nasal against a full noon / meem: tajweed only where ikhfa can happen (0.1.2). Before, any noon / meem /
    ikhfa swap passed, which hid a dropped letter ('bisalaa-in' for 'bisalaamin' aligned as meem -> ikhfa nasal)."""
    other = s if e == IKHFA else e
    if other == IKHFA:
        return True
    nxt = _next_sound(seq, i, next_core)
    if nxt is None:
        return written is None and next_core is None and i + 1 >= len(seq)   # unknown context (direct calls)
    if _base(other) == "م":
        return nxt == "ب"                       # ikhfa shafawi: meem sakin before ba
    return nxt in _IKHFA_LETTERS                # ikhfa haqiqi: noon sakin / tanween before an ikhfa letter


def _noon_may_assimilate(seq, pos, next_core, written, waqf):
    """a noon sound said or not said at seq[pos] is a tajweed difference only where a noon sakin or tanween is WRITTEN
    there and idgham or ikhfa can apply (0.1.2). Before, any unvowelled noon dropped or added before such a letter
    passed, which hid real mistakes ('al-muttaqi' for 'al-muttaqin' before a stop, 'dayinf' for 'dayf')."""
    end = pos + 1 >= len(seq)
    if end and next_core is None:
        return _tanween_end(written)          # at the end of the clip: only a tanween's noon (not a written letter)
    nxt = _next_sound(seq, pos, next_core)
    if nxt not in _IDGHAM | _IKHFA_LETTERS:
        return False
    if written is None:
        return True                           # spelling unknown (direct calls): as before
    if end:                                   # across the boundary: idgham or ikhfa, only in wasl
        return not waqf and (_tanween_end(written) or _sakin_noon_before(written, None))
    # inside a word only ikhfa applies (noon before ya / waw inside a word is izhar: ad-dunya, qinwan, bunyan)
    return nxt in _IKHFA_LETTERS and _sakin_noon_before(written, nxt)


def deviation_level(expected_core, said_core, next_core=None, written=None, waqf=False):
    """-> (level, edits): the most visible class of the differences, and the edits as [op, expected, said].
    next_core: the first core token of the following word (iqlab, idgham and ikhfa can cross the word boundary).
    written: the word's Uthmani text (tells a written noon from a written meem)."""
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
            elif e[0] in _NASALS and s[0] in _NASALS and IKHFA in (e, s) and _ikhfa_context(
                    e, s, expected_core, i, next_core, written):
                k = 0          # ikhfa against izhar (0.1.2: only where ikhfa can happen)
            elif {_base(e), _base(s)} == {"ن", "م"} and _unvowelled(expected_core, i) \
                    and _unvowelled(said_core, j) and _before_ba(expected_core, i, next_core) \
                    and (_base(e) == "ن" or _written_iqlab_site(written)):
                k = 0                                            # iqlab against izhar (0.1.2: before ba, written noon)
            else:
                k = 2
        else:
            t, seq, pos = (e, expected_core, i) if op == "del" else (s, said_core, j)
            if t in SHORT:
                k = 1
            elif t[0] in ("ن", IKHFA) and _unvowelled(seq, pos) and _noon_may_assimilate(
                    seq, pos, next_core, written, waqf):
                k = 0          # tanwin / sakin noon assimilated or not (0.1.2: only where idgham or ikhfa can happen)
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
    _rebalance(hyp, preps, al, fmt)
    _split_extra(hyp, preps, al, fmt)
    return _build(hyp, fmt, words, preps, al, ayat, len(pre), ref_str(s, a0, a1), retrieval)


# ------------------------------------------------------------------------------------------------ word re-split
# 0.1.2: the phone alignment can tie "word A said wrongly" with "word A deleted, its phones absorbed into word B" (the
# insertion and deletion costs make both paths cost the same), and the tie-break sometimes takes the second. One wrong
# word then becomes two wrong words (A deleted + B deviant, e.g. "...la yukhlifu l-mi'ad" with yukhlifu read as "qad"
# gave "qadlmi'ad"), which charges a phoneme model twice for one mistake that a text model is charged once for. The
# re-split tries every boundary between two neighbouring words and keeps the one with the FEWEST words wrong at the
# letter level (deleted counts as wrong); it changes the alignment only when that number strictly drops, so it can never
# hide a mistake. Tried: changing INS / DEL / skip costs (moved other tie-breaks and broke G1 cases); rejected.

def _lev_ops(toks, said, fmt):
    """token Levenshtein (sub_cost, insertion / deletion AL.INS) -> (cost, ops as in AL.WordAlign)"""
    n, m = len(toks), len(said)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * AL.DEL
    for j in range(1, m + 1):
        D[0][j] = j * AL.INS
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + AL.DEL, D[i][j - 1] + AL.INS,
                          D[i - 1][j - 1] + sub_cost(toks[i - 1], said[j - 1], fmt))
    ops, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + sub_cost(toks[i - 1], said[j - 1], fmt):
            ops.append(("eq" if toks[i - 1] == said[j - 1] else "sub", toks[i - 1], said[j - 1])); i -= 1; j -= 1
        elif i and D[i][j] == D[i - 1][j] + AL.DEL:
            ops.append(("del", toks[i - 1], None)); i -= 1
        else:
            ops.append(("ins", None, said[j - 1])); j -= 1
    ops.reverse()
    return D[n][m], ops


def _word_fit(forms, said, fmt, allowed=None):
    """-> (wrong 0/1, edit cost, form index, ops): wrong = deleted or deviant at the letter level; allowed = the form
    indices to consider (boundary states)"""
    best = None
    for fi, t in enumerate(forms):
        if allowed is not None and fi not in allowed:
            continue
        c, ops = _lev_ops(t, said, fmt)
        if best is None or c < best[1]:
            best = (fi, c, ops)
    fi, c, ops = best
    if not said:
        wrong = 1
    elif tuple(said) == tuple(forms[fi]):
        wrong = 0
    else:
        wrong = int(deviation_level(to_core(list(forms[fi]), fmt), to_core(list(said), fmt))[0] == "letters")
    return wrong, c, fi, ops


def _lstate(L):
    return AL.S_STOP if L == "ibtida" else AL.WASL


def _rstate(R):
    return AL.S_STOP if R == "waqf" else AL.WASL


# ------------------------------------------------------------------------------------------------ extra words
# 0.1.3: speech that is not part of the ayah (an extra word, isti'adha, a word said between two right ones) was absorbed
# into a neighbouring word, which then became one deviant word ("rasulan" + "allahi" -> "rasulanllahi"). Alone that
# costs one WER error either way, but next to another mistake it merged two errors into one ("rabbi" repeated before a
# wrong "rabbuk" -> one word "rabbirabbuk"), so a phoneme model paid less than a text model for the same mistakes (3
# mistakes next to each other: 30.65 % WER through the phones against 33.63 % as text). When a deviant word's phones are
# one of its own legal forms (or a neighbour's: a repeat) plus a word-sized remainder (2 or more consonants) at one
# end, the remainder becomes an inserted item of its own, written as a separate word. A one-consonant syllable stays
# inside the word (the G2 inserted_syllable case).

_VOWELS = {"a", "i", "u", "aa", "ii", "uu"}


def _consonants(toks):
    return sum(1 for t in toks if t not in _VOWELS)


_LEX = {}


def _lexicon(fmt):
    """every legal form of every Quran word, in a format's matching tokens (for cutting extra speech into words)"""
    lex = _LEX.get(fmt)
    if lex is None:
        lex = set()
        q = quran()
        for key in q.keys():
            for w in q.ayah(*key).words:
                for f in w.forms:
                    lex.add(tuple(form_tokens(f.tj3g, f.units, fmt)))
        _LEX[fmt] = lex
    return lex


def _known_cover(toks, fmt):
    """the fewest Quran word forms, each of two or more consonants (a word's worth: no one-letter pieces), that exactly
    cover toks [(i, j)], or None"""
    lex = _lexicon(fmt)
    n = len(toks)
    best = [(0, None)] + [(10 ** 9, None)] * n
    for j in range(1, n + 1):
        for i in range(max(0, j - 24), j):
            if best[i][0] + 1 < best[j][0] and _consonants(toks[i:j]) >= 2 and tuple(toks[i:j]) in lex:
                best[j] = (best[i][0] + 1, i)
    if best[n][1] is None:
        return None
    spans, j = [], n
    while j > 0:
        i = best[j][1]
        spans.append((i, j))
        j = i
    return spans[::-1]


def _cut_words(toks, fmt):
    """extra speech -> word-sized spans: the Quran words it is made of when it is made only of Quran words (as many
    words as a text model would write), else one piece (unknown speech is never chopped into made-up words)"""
    return _known_cover(toks, fmt) or [(0, len(toks))]


def _split_extra(hyp, preps, al, fmt):
    ws = sorted(al.words, key=lambda x: x.word)
    by = {x.word: x for x in ws}
    new_ins = []
    for x in ws:
        said = tuple(hyp[x.j0:x.j1])
        forms = [tuple(f) for f in preps[x.word][0]]
        if not said or said in forms:
            continue
        best = None
        for f in forms:
            n = len(f)
            if n == 0 or n >= len(said):
                continue
            for side in ("pre", "suf"):                  # extra before the word / after it
                if (side == "suf" and said[:n] == f) or (side == "pre" and said[-n:] == f):
                    extra = said[n:] if side == "suf" else said[:len(said) - n]
                    if _consonants(extra) >= 2 and (best is None or n > best[0]):
                        best = (n, side, forms.index(f), None)
        if best is None:                                  # a neighbour's legal form at one end: a glued repeat
            for nb, side in ((x.word - 1, "pre"), (x.word + 1, "suf")):
                if nb not in by:
                    continue
                for f in (tuple(t) for t in preps[nb][0]):
                    n = len(f)
                    if n == 0 or n >= len(said) or _consonants(f) < 2:
                        continue
                    if (side == "pre" and said[:n] == f) or (side == "suf" and said[-n:] == f):
                        if best is None or n > best[0]:
                            best = (n, side, None, nb)
        if best is None:
            continue
        n, side, fi, nb = best
        if nb is None:                                    # the word itself matched; the rest is extra
            if side == "suf":
                e0, e1, x.j1 = x.j0 + n, x.j1, x.j0 + n
                bnd = x.word + 1
            else:
                e0, e1, x.j0 = x.j0, x.j1 - n, x.j1 - n
                bnd = x.word
            x.form, x.skipped = fi, False
            for i, j in _cut_words(hyp[e0:e1], fmt):
                new_ins.append(AL.Insertion(bnd, e0 + i, e0 + j, "extra"))
        else:                                             # the neighbour repeated, glued to this word
            if side == "pre":
                r0, r1, x.j0 = x.j0, x.j0 + n, x.j0 + n
                bnd = x.word
            else:
                r0, r1, x.j1 = x.j1 - n, x.j1, x.j1 - n
                bnd = x.word + 1
            new_ins.append(AL.Insertion(bnd, r0, r1, "repeat", of_word=nb))
    # a deviant word whose phones are made ENTIRELY of two or more Quran words, none of which alone covers them (a wrong
    # word glued to an extra word): two words were said, as a text model would write them. The piece closest to the
    # expected word stays the word; the others become inserted words. The phones must also hold at least two consonants
    # more than the word's longest legal form (a changed vowel or letter adds none). A single substituted Quran word is one known word
    # (cover of one piece) and garbled phones are not made of known words, so neither is split. Tried: splitting
    # whenever a known-word end made the rest fit better (split single wrong words in two: 18.36 % against 14.57 %).
    done = {i.boundary for i in new_ins}
    for x in ws:
        said = list(hyp[x.j0:x.j1])
        forms = [list(f) for f in preps[x.word][0]]
        if not said or tuple(said) in [tuple(f) for f in forms] or x.word in done or x.word + 1 in done:
            continue
        if _consonants(said) < max(_consonants(f) for f in forms) + 2:
            continue                                  # no extra word-sized material: one word read wrongly
        cov = _known_cover(said, fmt)
        if cov is None or len(cov) < 2:
            continue
        fits = [min((_lev_ops(f, said[i:j], fmt)[0], fi) for fi, f in enumerate(forms)) for i, j in cov]
        kw = min(range(len(cov)), key=lambda k: fits[k][0])
        j0 = x.j0
        for k, (i, j) in enumerate(cov):
            if k != kw:
                new_ins.append(AL.Insertion(x.word if k < kw else x.word + 1, j0 + i, j0 + j, "extra"))
        x.j0, x.j1 = j0 + cov[kw][0], j0 + cov[kw][1]
        x.form, x.skipped = fits[kw][1], False
    al.insertions.extend(new_ins)
    al.insertions.sort(key=lambda i: (i.boundary, i.j0))


def _rebalance(hyp, preps, al, fmt):
    ws = sorted(al.words, key=lambda x: x.word)
    blocked = {ins.boundary for ins in al.insertions}
    for _ in range(3):
        changed = False
        for k in range(len(ws) - 1):
            a, b = ws[k], ws[k + 1]
            if b.word != a.word + 1 or a.j1 != b.j0 or b.word in blocked:
                continue
            fa, fb = preps[a.word][0], preps[b.word][0]
            sa, sb = preps[a.word][1], preps[b.word][1]
            xa = _word_fit(fa, hyp[a.j0:a.j1], fmt, {a.form})
            xb = _word_fit(fb, hyp[b.j0:b.j1], fmt, {b.form})
            cur = xa[0] + xb[0]
            if cur == 0:
                continue
            # the outer boundaries keep their states (a's left, b's right); the shared boundary must agree, as in
            # the aligner: a word in waqf is followed by a word in ibtida
            ca = [i for i in range(len(fa)) if _lstate(sa[i][0]) == _lstate(sa[a.form][0])]
            cb = [i for i in range(len(fb)) if _rstate(sb[i][1]) == _rstate(sb[b.form][1])]
            pairs = [(i, j) for i in ca for j in cb if _rstate(sa[i][1]) == _lstate(sb[j][0])]
            best = None
            for s in range(a.j0, b.j1 + 1):
                for i, j in pairs:
                    ya, yb = _word_fit(fa, hyp[a.j0:s], fmt, {i}), _word_fit(fb, hyp[s:b.j1], fmt, {j})
                    key = (ya[0] + yb[0], ya[1] + yb[1])
                    if best is None or key < best[0]:
                        best = (key, s, ya, yb)
            if best is None:
                continue
            (wrong, _), s, ya, yb = best
            if wrong < cur:
                a.j1, a.form, a.ops, a.cost, a.skipped = s, ya[2], ya[3], ya[1], s == a.j0
                b.j0, b.form, b.ops, b.cost, b.skipped = s, yb[2], yb[3], yb[1], s == b.j1
                changed = True
        if not changed:
            break


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
                it = Inserted("extra" if ins.kind == "extra" else "edge", ph, after, text=sp.diacritized,
                              text_imlaei=sp.plain)
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
                nx = None                                        # the next word's first sound (iqlab context)
                if w in by_word:
                    nt = preps[w][0][by_word[w].form]
                    nx = to_core(list(nt[:1]), fmt) if nt else None
                r.level, r.edits = deviation_level(core_e, core_s, nx, ay.uthmani[ti], form["right"] == "waqf")
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
