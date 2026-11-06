from conftest import degem

import quran_p2g as p2g
from quran_p2g.retrieve import fold


def test_fold():
    assert fold(["رّ", "aa", "ں"]) == ["ر", "a", "ن"]


def test_identify_exact():
    c = p2g.identify("ء a ل ح a م د u ل i ل aa ه i ر a ب i ل ع aa ل a م ii ن")
    assert c[0].ref == "1:2" and c[0].score == 1.0
    assert c[0].margin > 0                      # 37:182 adds a waw: close, but not identical


def test_identify_reports_identical_ayat():
    q = p2g.quran()
    top = p2g.identify(degem(q.canonical(55, 13)))[0]
    assert top.score == 1.0
    keys = {top.ref} | set(top.identical)
    assert {"55:13", "55:16", "55:77"} <= keys and len(keys) == 31


def test_identify_grows_a_range():
    q = p2g.quran()
    toks = degem(q.canonical(112, 1)) + degem(q.canonical(112, 2)) + degem(q.canonical(112, 3))
    assert p2g.identify(toks)[0].ref == "112:1-3"


def test_identify_tolerates_errors():
    toks = degem(p2g.quran().canonical(2, 255))
    toks[10] = "ق"
    toks[40] = "a"
    assert p2g.identify(toks)[0].ref == "2:255"


def test_identify_empty_and_formats():
    assert p2g.identify([]) == []
    assert p2g.identify("كَتَبَ", "muaalem")
