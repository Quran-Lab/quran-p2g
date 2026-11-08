"""Regression tests for 0.1.2: one wrong word is one word error (the word re-split), and an unvowelled noon / meem
swap counts as tajweed only in the iqlab context (before ba)."""
import os
import sys

import quran_p2g as p2g

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_regressions_v011 import canonical_words, planted  # noqa: E402


def test_substituted_word_is_one_error_not_two():
    # 3:9 "... la yukhlifu l-mi'ad": yukhlifu read as "qad". 0.1.1 marked yukhlifu deleted and wrote "qadlmi'ad"
    # (two word errors); a text model writing the wrong word is charged one.
    key = (3, 9)
    cw = canonical_words(key, "tj3")
    k = len(cw) - 2
    toks, _ = planted(key, "tj3", k, ["ق", "a", "د"])
    r = p2g.convert(toks, "3:9", "tj3")
    w = p2g.wer(r, "3:9")
    assert w.errors == 1, (w.errors, r.text_imlaei)
    words = [it for it in r.items if hasattr(it, "status")]
    assert words[-1].status == "match" and words[-2].status == "deviant"


def test_noon_meem_swap_outside_iqlab_is_a_letter_mistake():
    # 52:24 "ka'annahum lu'lu'un": meem read as noon before lam is a different word (kaannahun), not tajweed
    key = (52, 24)
    cw = canonical_words(key, "tj3")
    k = next(j for j, w in enumerate(cw) if w[0][-1] == "م" and w[0][:2] == ["ك", "a"])
    toks, _ = planted(key, "tj3", k, cw[k][0][:-1] + ["ن"])
    r = p2g.convert(toks, "52:24", "tj3")
    dev = [it for it in r.items if getattr(it, "status", None) == "deviant"]
    assert dev and dev[0].level == "letters", [(d.uthmani, d.level) for d in dev]
    assert p2g.wer(r, "52:24").errors == 1


def test_iqlab_before_ba_stays_tajweed():
    # noon sakin / tanween before ba read as meem (iqlab) against noon (izhar): a tajweed difference, same letters
    from quran_p2g.convert import deviation_level
    assert deviation_level(["م", "ن", "ب"], ["م", "م", "ب"])[0] == "tajweed"
    assert deviation_level(["م", "u", "ن"], ["م", "u", "م"], ["ب"])[0] == "tajweed"
    assert deviation_level(["م", "u", "ن"], ["م", "u", "م"], ["ل"])[0] == "letters"
