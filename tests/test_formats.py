import pytest

from quran_p2g.formats import (FORMATS, FormatError, form_tokens, muaalem_to_core, sub_cost, to_core, tokenize,
                               units_to_core)

SH = "ّ"


def test_formats_listed():
    assert FORMATS == ("tj3", "tj3g", "units", "muaalem")


def test_tj3_tokenize_drops_word_marks():
    assert tokenize("ب i | س م i", "tj3") == ["ب", "i", "س", "م", "i"]
    assert tokenize(["ب", "|", "i"], "tj3") == ["ب", "i"]


def test_tj3_rejects_held_token():
    with pytest.raises(FormatError):
        tokenize("ر" + SH + " a", "tj3")


def test_tj3g_accepts_held_token():
    assert tokenize("ر" + SH + " a", "tj3g") == ["ر" + SH, "a"]


def test_unknown_token():
    with pytest.raises(FormatError):
        tokenize("x a", "tj3")
    with pytest.raises(FormatError):
        tokenize("a", "nope")


def test_units_strip_length_and_core():
    toks = tokenize("ءَ لّ^َ ا:4 ه", "units")
    assert toks == ["ءَ", "لّ^َ", "ا", "ه"]
    assert units_to_core(toks) == ["ء", "a", "ل" + SH, "aa", "ه"]


def test_units_madd_letters_and_marks():
    assert units_to_core("رّ^َ حِ ۦ:4 م") == ["ر" + SH, "a", "ح", "ii", "م"]
    assert units_to_core("دُ ۥ:2 نَ") == ["د", "uu", "ن", "a"]
    assert units_to_core("بُ=") == ["ب", "u"]                 # rawm: the vowel, reduced
    assert units_to_core("بڇ") == ["ب"]                       # qalqalah residual
    with pytest.raises(FormatError):
        units_to_core("ا^")


def test_muaalem_runs():
    # quran_transcript phonetizer output for 1:1 (madd as runs, shadda as a doubled letter)
    core = muaalem_to_core("بِسمِ للَااهِ ررَحمَاانِ ررَحِۦۦۦۦم")
    assert core == ["ب", "i", "س", "م", "i", "ل" + SH, "aa", "ه", "i", "ر" + SH, "a", "ح", "م", "aa", "ن", "i",
                    "ر" + SH, "a", "ح", "ii", "م"]


def test_muaalem_ghunna_and_marks():
    core = muaalem_to_core("ءَںںںجَااااءَ عَجِۦۦۦۦبڇ")
    assert core == ["ء", "a", "ں", "ج", "aa", "ء", "a", "ع", "a", "ج", "ii", "ب"]
    assert tokenize("هُممممُ", "muaalem") == ["ه", "u", "م", "u"]      # matched in tj3: no held token
    with pytest.raises(FormatError):
        muaalem_to_core("x")


def test_form_tokens_per_format():
    tj3g, units = "ل" + SH + " aa ه i", "لَّ ا:2 هِ"
    assert form_tokens(tj3g, units, "tj3g") == ("ل" + SH, "aa", "ه", "i")
    assert form_tokens(tj3g, units, "tj3") == ("ل", "aa", "ه", "i")
    assert form_tokens(tj3g, units, "muaalem") == ("ل", "aa", "ه", "i")
    assert form_tokens(tj3g, units, "units") == ("لَّ", "ا", "هِ")


def test_to_core_identity_for_phone_formats():
    assert to_core(["ب", "a"], "tj3") == ["ب", "a"]


def test_sub_costs():
    assert sub_cost("a", "a", "tj3") == 0
    assert sub_cost("a", "aa", "tj3") == 2
    assert sub_cost("ر" + SH, "ر", "tj3g") == 2
    assert sub_cost("ں", "ن", "tj3") == 2
    assert sub_cost("ب", "ت", "tj3") == 4
    assert sub_cost("a", "i", "tj3") == 4
    assert sub_cost("ب", "a", "tj3") == 6
    assert sub_cost("بَ", "بِ", "units") == 2
    assert sub_cost("بَ", "ا", "units") == 6
