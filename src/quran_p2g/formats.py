"""Input formats: tokenizers, the matching space of each format, and the core token stream used for spelling.

Formats
  tj3      quran-g2p's tj3 phone vocabulary: 28 consonants, the ikhfa nasal ں, a i u aa ii uu, <foreign>.
           A geminate is one token. '|' (word boundary) is accepted and ignored.
  tj3g     tj3 plus a held token per consonant (the consonant followed by U+0651 SHADDA), quran-g2p tokenizer_tj3g.
  units    quran-g2p unit tokens (tokenlayer.phones_to_tokens, the tj2 grammar): a consonant with its shadda,
           tafkheem '^', ghunna '~', haraka, rawm '=', qalqalah / sakt residuals, or a madd letter with ':LEN'.
           Lengths are ignored when matching (a timing question, not a token one).
  muaalem  the phonetic script of quran_transcript / obadx muaalem models (Arabic letters and harakat; shadda as a
           doubled letter, madd as a run of the madd letter). Mapped to tj3; matched in tj3.

The core stream (what the speller reads) is tj3g: consonants with an optional shadda, ں, the six vowels.
"""
from __future__ import annotations

SHADDA = "ّ"
WORD = "|"
FOREIGN = "<foreign>"
IKHFA = "ں"
CONSONANTS = ["ء", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر",
              "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف",
              "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"]
CONS_SET = frozenset(CONSONANTS)
SHORT = ("a", "i", "u")
LONG = ("aa", "ii", "uu")
VOWELS = frozenset(SHORT + LONG)
TO_LONG = {"a": "aa", "i": "ii", "u": "uu"}
TO_SHORT = {"aa": "a", "ii": "i", "uu": "u"}
TJ3 = [CONSONANTS, [IKHFA], list(SHORT + LONG), [FOREIGN]]
TJ3_TOKENS = frozenset(t for g in TJ3 for t in g)
TJ3G_TOKENS = TJ3_TOKENS | frozenset(c + SHADDA for c in CONSONANTS)
FORMATS = ("tj3", "tj3g", "units", "muaalem")


class FormatError(ValueError):
    pass


def check_format(fmt: str) -> str:
    if fmt not in FORMATS:
        raise FormatError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")
    return fmt


def degeminate(tok: str) -> str:
    return tok[:-1] if tok.endswith(SHADDA) and len(tok) == 2 else tok


# ------------------------------------------------------------------------------------------------ tokenizers

def _split(x) -> list:
    if isinstance(x, str):
        return [t for t in x.split() if t != WORD]
    return [t for t in x if t != WORD]


def tokenize(x, fmt: str) -> list:
    """Hypothesis (a string or a token list) -> tokens in the format's matching space."""
    check_format(fmt)
    if fmt == "muaalem":
        if not isinstance(x, str):
            x = "".join(x)
        return [degeminate(t) for t in muaalem_to_core(x)]
    toks = _split(x)
    if fmt == "tj3":
        bad = [t for t in toks if t not in TJ3_TOKENS]
        if bad:
            raise FormatError(f"not tj3 tokens: {bad[:5]}")
        return toks
    if fmt == "tj3g":
        bad = [t for t in toks if t not in TJ3G_TOKENS]
        if bad:
            raise FormatError(f"not tj3g tokens: {bad[:5]}")
        return toks
    out = []
    for t in toks:
        unit_bases(t)                       # validates
        out.append(strip_len(t))
    return out


def strip_len(unit: str) -> str:
    return unit.split(":", 1)[0]


# ------------------------------------------------------------------------------------------------ units (tj2 grammar)

_U_CONS = {c: c for c in CONSONANTS}
_U_CONS.update({IKHFA: IKHFA, "۾": "م", "ٲ": "ء"})       # meem mukhfah, hamza musahhala
_U_MADD = {"ا": "aa", "ۥ": "uu", "ۦ": "ii", "ـ": "aa"}   # alef, small waw, small yeh, tatweel
_U_VOWEL = {"َ": "a", "ُ": "u", "ِ": "i", "۪": "a"}      # fatha, damma, kasra, imala fatha
_U_IKHTILAS = "ؙ"                                                         # small damma (12:11 ikhtilas)
_U_MARKS = {"^", "~", "=", "ڇ", "ۜ"}                                 # tafkheem, ghunna, rawm, qalqalah, sakt


def unit_bases(unit: str) -> list:
    """One unit token -> its core tokens before madd merging: [consonant(+shadda)] [+ vowel] or [long vowel]."""
    body = strip_len(unit)
    if not body:
        raise FormatError(f"empty unit token {unit!r}")
    if body[0] in _U_MADD:
        if len(body) != 1:
            raise FormatError(f"madd unit with marks {unit!r}")
        return [_U_MADD[body[0]]]
    if body[0] == _U_IKHTILAS and len(body) == 1:
        return ["u"]
    if body[0] not in _U_CONS:
        raise FormatError(f"unknown unit base in {unit!r}")
    cons = _U_CONS[body[0]]
    out = [cons]
    for ch in body[1:]:
        if ch == SHADDA:
            if cons != IKHFA:
                out[0] = cons + SHADDA
        elif ch in _U_VOWEL:
            out.append(_U_VOWEL[ch])
        elif ch not in _U_MARKS:
            raise FormatError(f"unknown mark {ch!r} in unit {unit!r}")
    return out


def _push(out: list, tok: str):
    """Append a core token; a long vowel after its short vowel merges into it."""
    if tok in LONG:
        if out and out[-1] == TO_SHORT[tok]:
            out[-1] = tok
            return
        if out and out[-1] == tok:          # a run (muaalem madd counts): still one long vowel
            return
    out.append(tok)


def units_to_core(units) -> list:
    out = []
    for u in _split(units):
        for b in unit_bases(u):
            _push(out, b)
    return out


# ------------------------------------------------------------------------------------------------ muaalem script

_M_CONS = {c: c for c in CONSONANTS}
_M_CONS.update({IKHFA: IKHFA, "۾": "م", "ٲ": "ء"})
_M_VOWEL = {"َ": "a", "ُ": "u", "ِ": "i", "۪": "a", "ؙ": "u"}
_M_MADD = {"ا": "aa", "ـ": "aa", "ۥ": "uu", "ۦ": "ii"}
_M_SILENT = {"ڇ", "ۜ", " ", "\t", "\n"}        # qalqalah mark, sakt mark, spaces


def muaalem_to_core(s: str) -> list:
    """muaalem / quran_transcript phonetic script -> core tokens. A run of one consonant is one token (held when the
    run is 2 or longer, except the ikhfa nasal whose run is its ghunna); a run of a madd letter is one long vowel."""
    chars = [c for c in s if c not in _M_SILENT]
    out, i, n = [], 0, len(chars)
    while i < n:
        ch = chars[i]
        if ch in _M_MADD:
            j = i
            while j < n and chars[j] == ch:
                j += 1
            _push(out, _M_MADD[ch])
            i = j
            continue
        if ch in _M_VOWEL:
            out.append(_M_VOWEL[ch])
            i += 1
            continue
        if ch in _M_CONS:
            c = _M_CONS[ch]
            j = i
            while j < n and _M_CONS.get(chars[j]) == c:
                j += 1
            out.append(c + SHADDA if (j - i >= 2 and c != IKHFA) else c)
            i = j
            continue
        raise FormatError(f"unknown muaalem symbol {ch!r} (U+{ord(ch):04X})")
    return out


# ------------------------------------------------------------------------------------------------ to core

def to_core(tokens, fmt: str) -> list:
    """Matching-space tokens of a format -> core (tj3g-alphabet) tokens."""
    if fmt in ("tj3", "tj3g", "muaalem"):
        return list(tokens)
    return units_to_core(tokens)


def form_tokens(tj3g: str, units: str, fmt: str) -> tuple:
    """A lattice form (its stored tj3g and unit strings) in a format's matching space."""
    if fmt == "tj3g":
        return tuple(tj3g.split())
    if fmt in ("tj3", "muaalem"):
        return tuple(degeminate(t) for t in tj3g.split())
    return tuple(strip_len(u) for u in units.split())


def core_to_tj3(core) -> list:
    return [degeminate(t) for t in core]


# ------------------------------------------------------------------------------------------------ substitution costs

def token_class(tok: str, fmt: str):
    """(kind, key) used by the substitution cost: kind in {C, V, N, X}; key groups near-identical tokens."""
    if fmt == "units":
        b = tok[0]
        if b in _U_MADD:
            return "V", _U_MADD[b][0]
        return "C", _U_CONS.get(b, b)
    if tok in VOWELS:
        return "V", tok[0]
    if tok == IKHFA:
        return "N", "ن"
    if tok == FOREIGN:
        return "X", tok
    return "C", tok[0]


def sub_cost(a: str, b: str, fmt: str) -> int:
    """Weighted substitution cost (insertion and deletion cost 4)."""
    if a == b:
        return 0
    ka, xa = token_class(a, fmt)
    kb, xb = token_class(b, fmt)
    if xa == xb:
        return 2                      # long / short of one vowel, held / plain, same consonant with other marks
    if {ka, kb} <= {"C", "N"}:
        return 4
    if ka == kb == "V":
        return 4
    return 6
