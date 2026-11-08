import random

import pytest
from conftest import degem, nfc

import quran_p2g as p2g
from quran_p2g.convert import deviation_level

SH = "ّ"
FATIHA2 = "ء a ل ح a م د u ل i ل aa ه i ر a ب i ل ع aa ل a م ii ن"


def path(key, fmt, stop_all=False, mode="sukun"):
    """A legal path from the packaged lattice: canonical, or a stop after every stoppable word."""
    ay = p2g.quran().ayah(*key)
    out, n = [], len(ay.words)
    for k, w in enumerate(ay.words):
        L = "ibtida" if k == 0 or (stop_all and ay.words[k - 1].stop_after) else "wasl"
        R = "waqf" if k == n - 1 or (stop_all and w.stop_after) else "wasl"
        fs = [f for f in w.forms if f.left == L and f.right == R and f.wajh is None]
        f = next((f for f in fs if f.mode == (mode if R == "waqf" else "plain")), None) or \
            next(f for f in fs if f.mode == ("sukun" if R == "waqf" else "plain"))
        toks = f.units.split() if fmt == "units" else f.tj3g.split()
        out.extend(degem(toks) if fmt == "tj3" else toks)
    return out


def test_example_match():
    r = p2g.convert(FATIHA2, "1:2")
    assert nfc(r.text) == nfc("ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ")
    assert r.text_imlaei == "الحمد لله رب العالمين"
    assert r.counts == {"match": 4, "deviant": 0, "deleted": 0, "inserted": 0}


@pytest.mark.parametrize("fmt", ["tj3", "tj3g", "units"])
def test_sampled_round_trip(fmt):
    q = p2g.quran()
    rng = random.Random(7)
    for key in rng.sample(sorted(q.keys()), 60):
        for stop_all, mode in ((False, "sukun"), (True, "sukun"), (True, "rawm")):
            r = p2g.convert(path(key, fmt, stop_all, mode), key, fmt)
            ay = q.ayah(*key)
            assert r.text == ay.text and r.text_imlaei == ay.text_imlaei, (key, fmt, stop_all, mode)
            assert r.counts["match"] == len(ay.uthmani)


def test_vowel_slip_is_harakat_level():
    r = p2g.convert(FATIHA2.replace("د u", "د i"), "1:2")
    w = r.words[0]
    assert (w.status, w.level) == ("deviant", "harakat")
    assert w.edits == [["sub", "u", "i"]]
    assert nfc(r.text.split()[0]) == nfc("ٱلْحَمْدِ")
    assert r.text_imlaei == "الحمد لله رب العالمين"          # undiacritized text keeps the word
    assert p2g.wer(r, "1:2").errors == 0
    assert p2g.wer(r, "1:2", diacritics=True).errors == 1


def test_consonant_slip_is_letters_level():
    r = p2g.convert(FATIHA2.replace("ل ع aa", "ل ء aa"), "1:2")
    w = r.words[3]
    assert (w.status, w.level) == ("deviant", "letters")
    assert r.text_imlaei.split()[3] != "العالمين"
    assert p2g.wer(r, "1:2").errors == 1


def test_ikhfa_said_as_izhar_is_tajweed_level():
    toks = " ".join(degem(p2g.quran().canonical(2, 3)))
    assert "ي u ں ف" in toks
    r = p2g.convert(toks.replace("ي u ں ف", "ي u ن ف"), "2:3")
    w = r.words[-1]
    assert (w.status, w.level) == ("deviant", "tajweed")
    assert r.text == p2g.quran().ayah(2, 3).text
    assert p2g.wer(r, "2:3", diacritics=True).errors == 0


def test_deleted_word():
    r = p2g.convert(FATIHA2.replace("ر a ب i ", ""), "1:2")
    assert [w.status for w in r.words] == ["match", "match", "deleted", "match"]
    assert nfc(r.text) == nfc("ٱلْحَمْدُ لِلَّهِ ٱلْعَٰلَمِينَ")
    assert p2g.wer(r, "1:2").errors == 1


def test_repeated_word():
    r = p2g.convert(FATIHA2.replace("ر a ب i", "ر a ب i ر a ب i"), "1:2")
    assert r.counts == {"match": 4, "deviant": 0, "deleted": 0, "inserted": 1}
    ins = r.insertions[0]
    assert (ins.kind, ins.of, nfc(ins.text)) == ("repeat", "1:2:2", nfc("رَبِّ"))
    assert nfc(r.text) == nfc("ٱلْحَمْدُ لِلَّهِ رَبِّ رَبِّ ٱلْعَٰلَمِينَ")


def test_basmala_prefix_is_reported_not_in_text():
    q = p2g.quran()
    toks = " ".join(degem(q.canonical(1, 1))) + " " + " ".join(degem(q.canonical(112, 1)))
    r = p2g.convert(toks, "112:1")
    assert r.text == q.ayah(112, 1).text
    assert r.items[0].kind == "basmala" and r.items[0].match
    r = p2g.convert(toks, "112:1", basmala=False)
    assert r.counts["match"] < 4


def test_range_in_wasl_across_ayat():
    q = p2g.quran()
    toks = []
    for a in (1, 2):
        ay = q.ayah(1, a)
        n = len(ay.words)
        for k, w in enumerate(ay.words):
            L = "ibtida" if (a, k) == (1, 0) else "wasl"
            R = "waqf" if (a, k) == (2, n - 1) else "wasl"
            f = next(f for f in w.forms if f.left == L and f.right == R and f.mode in ("plain", "sukun"))
            toks.extend(degem(f.tj3g.split()))
    r = p2g.convert(toks, "1:1-2")
    assert r.text == q.ayah(1, 1).text + " " + q.ayah(1, 2).text
    assert r.ref == "1:1-2"


def test_open_edge_wasl_from_previous_ayah():
    # 1:2 read straight on from 1:1: the hamzat al-wasl of al-hamdu is not said
    toks = FATIHA2.split()[2:]
    r = p2g.convert(toks, "1:2")
    assert r.counts["match"] == 4 and r.words[0].form["left"] == "wasl"
    r = p2g.convert(toks, "1:2", open_edges=False)
    assert r.words[0].status == "deviant"


def test_free_edges():
    r = p2g.convert("م a ن ي " + FATIHA2, "1:2", free_edges=True)
    assert r.counts["match"] == 4 and r.insertions[0].kind == "edge"


def test_unconstrained():
    r = p2g.convert(FATIHA2, None)
    assert r.ref == "1:2" and r.retrieval["ayah"] == "1:2" and r.retrieval["score"] == 1.0


def test_to_dict():
    d = p2g.convert(FATIHA2.replace("د u", "د i"), "1:2").to_dict()
    assert d["items"][0]["status"] == "deviant" and d["items"][0]["level"] == "harakat"
    assert d["items"][0]["spelling"]["plain"] == "الحمد"
    assert "items" not in p2g.convert(FATIHA2, "1:2").to_dict(words=False)


def test_imlaei_one_to_two_group():
    q = p2g.quran()
    ay = q.ayah(2, 77)
    assert ay.uthmani[0] == "أَوَلَا" and ay.imlaei[:2] == ["أو", "لا"]
    r = p2g.convert(degem(q.canonical(2, 77)), "2:77")
    assert r.text_imlaei == ay.text_imlaei


def test_deviation_levels_directly():
    assert deviation_level(["ك", "a"], ["ك", "i"])[0] == "harakat"
    assert deviation_level(["ك" + SH, "a"], ["ك", "a"])[0] == "harakat"
    assert deviation_level(["ك", "aa"], ["ك", "a"])[0] == "letters"
    assert deviation_level(["ں", "ف"], ["ن", "ف"])[0] == "tajweed"
    # iqlab: a written noon read as meem before ba is tajweed; a written meem read as noon is another word (0.1.2)
    assert deviation_level(["م", "ب"], ["ن", "ب"], written="أَنۢبِئْهُم")[0] == "tajweed"
    assert deviation_level(["ن", "ب"], ["م", "ب"])[0] == "tajweed"
    assert deviation_level(["ل", "a", "ك", "u", "م"], ["ل", "a", "ك", "u", "ن"], ["ب"], "لَكُم")[0] == "letters"
    assert deviation_level(["ع", "a"], ["ء", "a"])[0] == "letters"
    assert deviation_level(["ع", "a"], ["ع", "a", "ن"])[0] == "tajweed"


def test_bad_inputs():
    with pytest.raises(ValueError):
        p2g.convert("ب i", "2:300")
    with pytest.raises(p2g.FormatError):
        p2g.convert("q", "1:1")
