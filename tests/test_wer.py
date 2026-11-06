import pytest

import quran_p2g as p2g
from quran_p2g.wer import WerResult, normalize_diacritized, reference_text, word_edit_distance


def test_board_strip():
    assert p2g.strip("ٱلرَّحْمَٰنِ") == "الرحمن"
    assert p2g.strip("إِلَىٰ") == "الي"
    assert p2g.strip("رَحْمَةٌ") == "رحمه"
    assert p2g.strip_alef("ٱلْعَٰلَمِينَ") == "لعلمين"


def test_reference_scripts():
    assert reference_text("1:1") == "بسم الله الرحمن الرحيم"
    assert reference_text("1:1", "uthmani").startswith("بِسْمِ")
    assert reference_text("1:1-2").endswith("رب العالمين")


def test_wer_text_both_scripts():
    assert p2g.wer("بسم الله الرحمن الرحيم", "1:1").errors == 0
    assert p2g.wer("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ", "1:1").errors == 0
    assert p2g.wer("بسم الله الرحيم", "1:1").errors == 1
    r = p2g.wer("بسم الله", "1:1")
    assert (r.errors, r.ref_words, r.hyp_words) == (2, 4, 2) and r.wer == 0.5


def test_wer_diacritized():
    assert p2g.wer("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ", "1:1", diacritics=True).errors == 0
    assert p2g.wer("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنُ ٱلرَّحِيمِ", "1:1", diacritics=True).errors == 1
    with pytest.raises(ValueError):
        p2g.wer("x", "1:1", diacritics=True, script="imlaei")


def test_wer_alef_variant():
    assert p2g.wer("بسم الله الرحمن الرحيم", "1:1", alef=True).errors == 0


def test_wer_of_result_uthmani_plain():
    r = p2g.convert("ب i س م i ل aa ه i ر a ح م aa ن u ر a ح ii م", "1:1")
    assert p2g.wer(r, "1:1", script="uthmani").errors == 0
    assert p2g.wer(r, "1:1", diacritics=True).errors == 1


def test_corpus_and_helpers():
    t = p2g.corpus_wer([WerResult(1, 4, 4), WerResult(0, 6, 6)])
    assert (t.errors, t.ref_words) == (1, 10) and t.wer == 0.1
    assert WerResult(0, 0, 0).wer == 0.0
    assert word_edit_distance(["a", "b"], ["b"]) == 1
    assert normalize_diacritized("بِسْمِۖ") == "بِسْمِ"
    with pytest.raises(ValueError):
        p2g.wer("x", "1:1", script="other")
