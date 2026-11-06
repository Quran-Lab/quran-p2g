"""Spell-by-sound: core phone tokens -> Arabic, diacritized and undiacritized. Rule-based and deterministic.

The rules invert quran-g2p's reading of the script, in Uthmani dabt conventions (so a spelling can be fed back to
quran-g2p and read the way it was spelled):

  consonant + short vowel        letter + fatha / kasra / damma
  consonant + long vowel         letter + haraka + madd letter (alef / yeh / waw, bare)
  consonant without a vowel      letter + sukun (U+0652); the ikhfa nasal is a bare noon (the Uthmani mark of ikhfa)
  held consonant (tj3g, or the   letter + shadda (+ haraka)
    same consonant twice)
  hamza                          seated by its vowel and the vowel before it (i > u > a): word-initial alef with
                                 hamza above / below, medial and final seats alef / waw / yeh, the bare hamza after a
                                 long vowel or a consonant at the end of a word; hamza + long a as ءَا (the mushaf's
                                 spelling, alef madda in the undiacritized form)
  word-initial consonant         hamzat al-wasl (U+0671) before a word that starts with a consonant cluster
    without a vowel
  a vowel with no consonant      carried on a bare alef (the word was read without its consonant)

Optional hints from the reference word (constrained mode only, never required): the definite article (ٱل before a sun
letter, and the article or hamzat al-wasl read in ibtida), tanwin, teh marbuta and alef maqsura, applied only where
the phones agree with them.
"""
from __future__ import annotations

from dataclasses import dataclass

from .formats import CONS_SET, IKHFA, LONG, SHADDA, SHORT, TO_SHORT, to_core, tokenize

FATHA, DAMMA, KASRA, SUKUN = "َ", "ُ", "ِ", "ْ"
FATHATAN, DAMMATAN, KASRATAN = "ً", "ٌ", "ٍ"
HARAKA = {"a": FATHA, "i": KASRA, "u": DAMMA}
TANWIN = {"a": FATHATAN, "u": DAMMATAN, "i": KASRATAN}
MADD_LETTER = {"aa": "ا", "ii": "ي", "uu": "و"}
ALEF, ALEF_WASLA, ALEF_MAQSURA, TEH_MARBUTA = "ا", "ٱ", "ى", "ة"
LAM = "ل"
HAMZA = "ء"
SEAT = {"a": "أ", "u": "ؤ", "i": "ئ"}          # alef / waw / yeh carrying hamza
ALEF_HAMZA_BELOW = "إ"
SUN = set("تثدذرزسشصضطظلن")
MARKS = set(chr(c) for c in range(0x064B, 0x0653)) | {"ٰ"} | set(chr(c) for c in range(0x06D6, 0x06EE))
_STRENGTH = {None: 0, "a": 1, "u": 2, "i": 3}


@dataclass(frozen=True)
class Spelling:
    diacritized: str
    plain: str

    def __str__(self):
        return self.diacritized


@dataclass
class _Syl:
    letter: str           # base letter ("" for a vowel without consonant)
    gem: bool
    vowel: str | None     # a i u aa ii uu or None
    nasal: bool = False   # ikhfa nasal


def _syllables(core) -> list:
    out, i, n = [], 0, len(core)
    while i < n:
        t = core[i]
        if t in SHORT or t in LONG:
            out.append(_Syl("", False, t))
            i += 1
            continue
        if t == "<foreign>":
            i += 1
            continue
        nasal = t == IKHFA
        gem = t.endswith(SHADDA) and len(t) == 2
        letter = "ن" if nasal else t[0]
        if not nasal and letter not in CONS_SET:
            raise ValueError(f"cannot spell token {t!r}")
        i += 1
        # the same consonant twice with no vowel between: one held consonant (tj3 has no held token)
        if not gem and not nasal and i < n and core[i] == letter:
            gem = True
            i += 1
        v = None
        if i < n and (core[i] in SHORT or core[i] in LONG):
            v = core[i]
            i += 1
        out.append(_Syl(letter, gem, v, nasal))
    return out


def _quality(v):
    return TO_SHORT.get(v, v) if v else None


def _hamza(sy, k, syls):
    """Seat of the hamza of syllable k."""
    own = _quality(sy[k].vowel)
    prev_v = sy[k - 1].vowel if k > 0 else None
    if k == 0:
        return ALEF_HAMZA_BELOW if own == "i" else SEAT["a"]
    last = k == len(sy) - 1
    prev_long = prev_v in LONG
    prev_cons_only = prev_v is None and sy[k - 1].letter != ""
    if last and own is None:                       # final, sakin hamza: seated by the vowel before it
        if prev_long or prev_cons_only:
            return HAMZA
        return SEAT[_quality(prev_v)]
    if last and (prev_long or prev_cons_only) and own is not None and sy[k].vowel in SHORT:
        return HAMZA
    if prev_v == "aa" and own == "a":
        return HAMZA
    if prev_v == "uu" and own == "a":
        return HAMZA
    if prev_v == "ii" or (sy[k - 1].letter == "ي" and prev_v is None):
        return SEAT["i"]
    best = max((own, _quality(prev_v)), key=lambda q: _STRENGTH[q])
    if best is None:
        return SEAT["a"]
    return SEAT[best]


def _ref_info(ref):
    """What the reference word's spelling says about its edges."""
    if not ref:
        return {}
    base = [c for c in ref if c not in MARKS]
    info = {}
    if ref.startswith(ALEF_WASLA + "ل"):
        info["article"] = True
        rest = ref[2:]
        k = 0
        while k < len(rest) and rest[k] in MARKS:
            k += 1
        if k < len(rest):
            nxt = rest[k]
            marks = []
            j = k + 1
            while j < len(rest) and rest[j] in MARKS:
                marks.append(rest[j])
                j += 1
            if nxt in SUN and SHADDA in marks:
                info["sun"] = nxt
    elif ref.startswith(ALEF_WASLA):
        info["wasl"] = True
    tail = ref.rstrip()
    for q, mark in TANWIN.items():
        if mark in tail[-4:]:
            info["tanwin"] = q
            at = tail.rfind(mark)
            after = tail[at + 1:]
            seat = [c for c in after if c not in MARKS]
            if seat:
                info["tanwin_seat"] = seat[0]
            carrier = [c for c in tail[:at] if c not in MARKS]
            if carrier:
                info["tanwin_on"] = carrier[-1]
    if base and base[-1] == TEH_MARBUTA:
        info["marbuta"] = True
    if base and base[-1] == ALEF_MAQSURA:
        info["maqsura"] = True
    return info


def _sun_start(core, k, sun, geminates_known):
    """core[k] is the article's sun letter, held (or, without a held token, followed by its vowel)."""
    if k >= len(core) or not sun or core[k][0] != sun:
        return False
    if geminates_known:
        return core[k] == sun + SHADDA
    return k + 1 < len(core) and (core[k + 1] in SHORT or core[k + 1] in LONG)


_CARRIES = {TEH_MARBUTA: ("ت", "ه"), "أ": (HAMZA,), "إ": (HAMZA,), "ؤ": (HAMZA,), "ئ": (HAMZA,)}


def _tanwin_letter(info, letter):
    """The phones' letter is the one the reference's tanwin sits on (so a final nasal can be its tanwin)."""
    on = info.get("tanwin_on")
    return on is None or letter == on or letter in _CARRIES.get(on, ())


def _qamari_start(core, k):
    return k + 1 < len(core) and core[k] == LAM and core[k + 1] not in SHORT and core[k + 1] not in LONG


def spell_core(core, ref_word: str | None = None, geminates_known: bool = False, waqf: bool = False) -> Spelling:
    """Core (tj3g-alphabet) tokens of one word -> Spelling. geminates_known: the tokens mark held consonants (tj3g,
    units), so an article's sun letter must be held to be read as the article. waqf: the word was read before a stop,
    where a tanwin is never said as a nasal (a final nasal is then the word's own letter)."""
    core = [t for t in core if t != "<foreign>"]
    info = _ref_info(ref_word)
    sun = info.get("sun")
    prefix = ""
    starts_wasl = len(core) >= 3 and core[0] == HAMZA and core[1] in SHORT
    article_a = starts_wasl and core[1] == "a"                        # the article's hamza is always read with fatha
    if info.get("article") and article_a and _qamari_start(core, 2) and core[3][0] not in SUN:
        core, prefix = core[3:], ALEF_WASLA + LAM + SUKUN             # ibtida: al- + a qamari letter
    elif info.get("article") and article_a and _sun_start(core, 2, sun, geminates_known):
        core, prefix = core[2:], ALEF_WASLA + LAM                     # ibtida: a- + the held sun letter
    elif info.get("wasl") and starts_wasl and core[2] not in SHORT and core[2] not in LONG \
            and len(core) > 3 and core[3] not in SHORT and core[3] not in LONG \
            and core[1] == ("u" if len(core) > 4 and core[4] in ("u", "uu") else "i"):
        core, prefix = core[2:], ALEF_WASLA                           # ibtida of a hamzat-al-wasl word
    elif info.get("article") and _qamari_start(core, 0):
        core, prefix = core[1:], ALEF_WASLA + LAM + SUKUN             # wasl: l- + a qamari letter
    elif info.get("article") and _sun_start(core, 0, sun, geminates_known):
        prefix = ALEF_WASLA + LAM                                     # wasl: the held sun letter
    if prefix.endswith(LAM) and core and not core[0].endswith(SHADDA):
        core[0] = core[0] + SHADDA
    syls = _syllables(core)
    if not syls:
        return Spelling(prefix, _plain(prefix))
    # tanwin: the reference ends in tanwin and the phones end in vowel + nasal (wasl) or the waqf long alef
    tanwin = None
    if info.get("tanwin") and len(syls) >= 2:
        last, before = syls[-1], syls[-2]
        if not waqf and last.vowel is None and last.letter in ("ن", "م") and not last.gem \
                and before.vowel == info["tanwin"] and _tanwin_letter(info, before.letter):
            syls = syls[:-1]
            tanwin = info["tanwin"]
        elif info["tanwin"] == "a" and last.vowel == "aa" and info.get("tanwin_seat") in (ALEF, ALEF_MAQSURA) \
                and _tanwin_letter(info, last.letter):
            last.vowel = "a"
            tanwin = "a"
    if not prefix and syls[0].letter and syls[0].vowel is None:
        prefix = ALEF_WASLA                          # a word that starts with a cluster (or ں): hamzat al-wasl
    out = []
    for k, sy in enumerate(syls):
        final = k == len(syls) - 1
        if sy.letter == "":
            out.append(ALEF)
        elif sy.letter == HAMZA:
            seat = _hamza(syls, k, syls)
            # hamza + long a: the mushaf writes the bare hamza before the alef (ءَا), as the Tanzil text does
            out.append(HAMZA if (sy.vowel == "aa" and seat == SEAT["a"]) else seat)
        elif final and info.get("marbuta") and sy.letter in ("ت", "ه"):
            out.append(TEH_MARBUTA)
        else:
            out.append(sy.letter)
        if sy.gem:
            out.append(SHADDA)
        v = sy.vowel
        if final and tanwin and v == tanwin:
            out.append(TANWIN[v])
            seat = info.get("tanwin_seat")
            if v == "a" and seat in (ALEF, ALEF_MAQSURA) and out[-3:-2] != [TEH_MARBUTA] \
                    and sy.letter != HAMZA:
                out.append(seat)
            continue
        if v is None:
            if not sy.nasal:
                out.append(SUKUN)
        elif v in SHORT:
            out.append(HARAKA[v])
        else:
            out.append(HARAKA[v[0]])
            if final and v == "aa" and info.get("maqsura"):
                out.append(ALEF_MAQSURA)
            else:
                out.append(MADD_LETTER[v])
    text = prefix + "".join(out)
    return Spelling(text, _plain(text))


def _plain(s: str) -> str:
    """Undiacritized: marks removed, alef wasla -> alef, the mushaf's hamza + alef -> alef madda (imla'i)."""
    return "".join(c for c in s if c not in MARKS).replace(ALEF_WASLA, ALEF).replace(HAMZA + ALEF, "آ")


def spell(tokens, fmt: str = "tj3", ref_word: str | None = None) -> Spelling:
    """Spell one word (or any run of phones) said as `tokens` in format `fmt`."""
    return spell_core(to_core(tokenize(tokens, fmt), fmt), ref_word, geminates_known=fmt in ("tj3g", "units"))


def spell_words(tokens, fmt: str = "tj3") -> list:
    """Spell a '|'-separated token string word by word (no reference hints)."""
    if isinstance(tokens, str):
        chunks = tokens.split() if fmt == "muaalem" else tokens.split("|")
    else:
        chunks, cur = [], []
        for t in tokens:
            if t == "|":
                chunks.append(cur)
                cur = []
            else:
                cur.append(t)
        chunks.append(cur)
    out = []
    for c in chunks:
        if (c.strip() if isinstance(c, str) else c):
            out.append(spell(c, fmt))
    return out
