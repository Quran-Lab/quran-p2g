"""Regression tests for the bugs fixed in 0.1.1 (each found by a G2 or G3 miss) and for the G2 / G3 equivalence logic."""
import os
import sys

import pytest
from conftest import degem, nfc

import quran_p2g as p2g
from quran_p2g.align import align
from quran_p2g.spell import spell_core

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import g2_equiv as EQ  # noqa: E402

IB, WA, Q = "ibtida", "wasl", "waqf"
SH = "ّ"


def canonical_words(key, fmt):
    """[(tokens, legal forms, Uthmani index)] of the canonical path, as tools/gate_g2.py plants on."""
    ay = p2g.quran().ayah(*key)
    n, out = len(ay.words), []
    for k, w in enumerate(ay.words):
        L = IB if k == 0 else WA
        R = Q if k == n - 1 else WA
        f = next(f for f in w.forms if f.left == L and f.right == R and f.mode == ("sukun" if R == Q else "plain")
                 and f.wajh is None)
        toks = f.tj3g.split()
        out.append((degem(toks) if fmt == "tj3" else toks, None, w.text_index))
    return out


def planted(key, fmt, k, new):
    cw = canonical_words(key, fmt)
    return [t for j, w in enumerate(cw) for t in (new if j == k else w[0])], cw


# ---------------------------------------------------------------------------------------------- aligner calibration

def test_fewer_stops_win_a_tie():
    # "ك a ب": word 1 skipped after word 0 in wasl (skip 2 * 2 + 1 = 5), or word 0 in waqf, a stop (1) and word 1
    # in ibtida said as its first vowel only (one deletion, 4): both cost 5, the path without the stop wins
    T = [[("ك", "a"), ("ك",)], [("ت", "u"), ("a", "ت")], [("ب",)]]
    S = [[(IB, WA), (IB, Q)], [(WA, WA), (IB, WA)], [(WA, Q)]]
    al = align("ك a ب".split(), T, S, "tj3")
    assert al.cost == 5 and al.stops == 0 and al.words[1].skipped


@pytest.mark.parametrize("key,fmt,k", [
    ((2, 102), "tj3g", 25),     # مِنْ: was "a stop after يُعَلِّمَانِ, then مِنْ said as 'i'" (tie at 9)
    ((41, 22), "tj3g", 3),      # a two-token word: the stop explanation was cheaper (5 < 6)
    ((53, 48), "tj3g", 1),      # هُوَ: two stops and a deletion tied the skip
    ((3, 54), "tj3g", 3),       # وَٱللَّهُ after ٱللَّهُ: skip of the wrong copy plus a deletion was cheaper
])
def test_skipped_word_is_deleted(key, fmt, k):
    hyp, cw = planted(key, fmt, k, [])
    r = p2g.convert(hyp, key, fmt)
    ti = cw[k][2]
    assert [w.index for w in r.words if w.status != "match"] == [ti]
    assert r.words[ti].status == "deleted" and not r.insertions


@pytest.mark.parametrize("key,fmt,k", [
    ((2, 61), "tj3", 34),       # مَا after لَكُم: "لَكُمَا, stop, مَا" was cheaper than the repeat in tj3
    ((19, 78), "tj3g", 2),      # أَمِ before ٱتَّخَذَ: a stop let the ibtida form absorb half of the repeat
    ((91, 12), "tj3", 0),
])
def test_repeated_word_is_an_insertion(key, fmt, k):
    cw = canonical_words(key, fmt)
    hyp, _ = planted(key, fmt, k, cw[k][0] + cw[k][0])
    r = p2g.convert(hyp, key, fmt)
    assert all(w.status == "match" for w in r.words)
    assert [(x.kind, x.of) for x in r.insertions] == [("repeat", f"{key[0]}:{key[1]}:{cw[k][2]}")]


# ---------------------------------------------------------------------------------------------- speller

def test_tanwin_hint_needs_the_tanwin_letter():
    # أَثِيمٍ said "ء a ث i م" in waqf: the final meem is the word's own letter, not a tanwin nasal on the tha
    assert nfc(spell_core("ء a ث i م".split(), "أَثِيمٍ").diacritized) == nfc("أَثِمْ")
    assert nfc(spell_core("ء a ث ii م i ن".split(), "أَثِيمٍ").diacritized) == nfc("أَثِيمٍ")


def test_no_tanwin_nasal_in_waqf():
    # مَّكْنُونٌ said with a short u, before a stop: "maknun", not makn + tanwin
    s = spell_core("م a ك ن u ن".split(), "مَّكْنُونٌ", waqf=True).diacritized
    assert nfc(s) == nfc("مَكْنُنْ")
    assert nfc(spell_core("م a ك ن u ن".split(), "مَّكْنُونٌ").diacritized) == nfc("مَكْنٌ")   # wasl: the tanwin


def test_word_starting_with_the_ikhfa_nasal_gets_hamzat_al_wasl():
    s = spell_core(["ں", "ف", "a", "د" + SH, "uu"], "ٱنفَضُّوٓا۟", geminates_known=True).diacritized
    assert s.startswith("ٱن")


def test_article_lam_said_before_a_sun_letter():
    # the article's lam said (not assimilated) before ط, in ibtida: a plain hamza, which reads back with fatha
    s = spell_core("ء a ل ط a ق aa ر i ع a ه".split(), "ٱلْقَارِعَةُ", waqf=True).diacritized
    assert s.startswith("أَلْط")


def test_converted_waqf_word_is_spelled_without_tanwin():
    hyp, cw = planted((37, 49), "tj3", 2, "م a ك ن u ن".split())
    r = p2g.convert(hyp, "37:49", "tj3")
    assert r.words[2].status == "deviant" and r.words[2].form["right"] == "waqf"
    assert nfc(r.words[2].spelling.diacritized) == nfc("مَكْنُنْ")


# ---------------------------------------------------------------------------------------------- G2 equivalence logic

def test_same_string_alternatives_identical_words():
    # 23:36 هَيْهَاتَ هَيْهَاتَ: skipping either copy gives the same phones
    cw = canonical_words((23, 36), "tj3")
    assert EQ.same_string_alternatives(cw, 1, "word_skip", []) == [0]
    assert EQ.same_string_alternatives(cw, 0, "word_repeat", cw[0][0] + cw[0][0]) == [1]


def test_same_string_alternatives_inserted_syllable_at_a_boundary():
    # 4:49 أَلَمْ تَرَ: "ت a" inserted after the ta of تَرَ is the same string as "ت a" at the end of أَلَمْ
    cw = canonical_words((4, 49), "tj3g")
    assert cw[1][0] == "ت a ر a".split()
    assert EQ.same_string_alternatives(cw, 1, "inserted_syllable", "ت a ت a ر a".split()) == [0]
    # an insertion that does not repeat its neighbour has no alternative
    assert EQ.same_string_alternatives(cw, 1, "inserted_syllable", "ت a ب a ر a".split()) == []


def test_substitutions_have_no_alternative():
    cw = canonical_words((1, 2), "tj3")
    new = list(cw[1][0])
    new[1] = "a" if new[1] != "a" else "u"
    assert EQ.same_string_alternatives(cw, 1, "vowel_swap", new) == []


def test_equivalent_attribution_is_accepted_and_nothing_else():
    hyp, cw = planted((23, 36), "tj3", 1, [])
    r = p2g.convert(hyp, "23:36", "tj3")
    flagged = [w.index for w in r.words if w.status != "match"]
    assert len(flagged) == 1
    alts = EQ.same_string_alternatives(cw, 1, "word_skip", [])
    ok = EQ.statuses_match(r, "word_skip", 1, "23:36") or \
        any(EQ.statuses_match(r, "word_skip", j, "23:36") for j in alts)
    assert ok
    # a different edit type is never accepted through the equivalence
    assert not EQ.statuses_match(r, "word_repeat", 1, "23:36")


def test_planted_cost_matches_the_aligner():
    hyp, cw = planted((1, 2), "tj3", 1, [])
    r = p2g.convert(hyp, "1:2", "tj3")
    from quran_p2g.convert import _prep
    toks, states, _ = _prep(p2g.quran().ayah(1, 2).words[1], "tj3")
    assert EQ.planted_cost(toks, states, 1, len(cw), "word_skip", cw[1][0], [], "tj3") == r.cost


# ---------------------------------------------------------------------------------------------- G3 conventions

def test_path_legality_vs_per_word_legality():
    import gate_g3
    ay = p2g.quran().ayah(55, 37)
    cw = canonical_words((55, 37), "tj3")
    w0 = list(cw[0][0])
    assert w0[-1] == "a"
    w0[-1] = "aa"                         # فَإِذَا with its waqf long vowel, then ٱنشَقَّتِ in wasl
    hyp = w0 + [t for w in cw[1:] for t in w[0]]
    _, _, path_flags, _ = gate_g3.bench_convention(ay, hyp, path_legality=True)
    _, _, word_flags, _ = gate_g3.bench_convention(ay, hyp, path_legality=False)
    assert path_flags[0] and not word_flags[0]      # a legal form, but not in a state its neighbour allows
    assert not any(path_flags[1:]) and not any(word_flags[1:])
