"""Regression tests for 0.1.3: extra speech is written as words of its own, so mistakes side by side are counted one
by one, as a text model's WER would count them."""
import os
import sys

import quran_p2g as p2g

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_regressions_v011 import canonical_words  # noqa: E402


def _flat(words):
    return " ".join(t for w in words for t in w)


def test_extra_word_between_two_right_words_is_its_own_insertion():
    # 3:51 "inna llaha rabbi wa-rabbukum": an extra Quran word said before "llaha" (0.1.2 glued it onto llaha)
    key = (3, 51)
    cw = [w[0] for w in canonical_words(key, "tj3")]
    extra = [w[0] for w in canonical_words((3, 53), "tj3")][1]          # a word of another ayah
    said = cw[:1] + [extra] + cw[1:]
    r = p2g.convert(_flat(said), "3:51", "tj3")
    assert p2g.wer(r, "3:51").errors == 1, r.text_imlaei
    assert r.counts["inserted"] == 1
    # a neighbour may change its connecting vowel next to the new word (a harakat-level difference), never its letters
    assert all(w.level != "letters" for w in r.words if w.status == "deviant")


def test_repeat_glued_to_a_wrong_word_counts_twice():
    # a word repeated, then the next word replaced by another Quran word: two mistakes, as for a text model
    key = (3, 51)
    cw = [w[0] for w in canonical_words(key, "tj3")]
    other = [w[0] for w in canonical_words((2, 255), "tj3")][3]
    said = cw[:3] + [cw[2]] + [other] + cw[4:]
    r = p2g.convert(_flat(said), "3:51", "tj3")
    assert p2g.wer(r, "3:51").errors == 2, r.text_imlaei


def test_one_changed_vowel_is_never_split_into_words():
    # a short vowel changed inside a word adds no material: one deviant word, no insertion
    key = (2, 255)
    cw = [w[0] for w in canonical_words(key, "tj3")]
    k = next(j for j, w in enumerate(cw) if "a" in w and len(w) >= 5)
    w = list(cw[k]); w[w.index("a")] = "u"
    r = p2g.convert(_flat(cw[:k] + [w] + cw[k + 1:]), "2:255", "tj3")
    assert r.counts["inserted"] == 0
