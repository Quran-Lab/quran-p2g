import pytest
from conftest import nfc

from quran_p2g.quran import parse_ref, quran, ref_str


def test_counts():
    q = quran()
    assert len(q) == 6236
    assert sum(len(q.ayah(s, a).words) for s, a in q.keys()) == 77481
    assert len(q.n_ayat_in) == 114 and q.n_ayat_in[2] == 286


def test_artifact_pins_upstream():
    m = quran().meta
    assert m["format"] == "quran-p2g.lattice.v1"
    assert m["quran_g2p_commit"].startswith("7eeba12")
    assert set(m["pins"]) == {"tanzil-uthmani.txt", "kfgqpc-hafsData_v18.json"}


def test_texts():
    a = quran().ayah(1, 1)
    assert nfc(a.text) == nfc("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ")
    assert a.text_imlaei == "بسم الله الرحمن الرحيم"


def test_imlaei_groups_cover_every_word():
    q = quran()
    multi = 0
    for s, a in q.keys():
        ay = q.ayah(s, a)
        u = e = 0
        for u0, u1, e0, e1 in ay.groups:
            assert (u0, e0) == (u, e) and u1 > u0 and e1 > e0
            u, e = u1, e1
            multi += (u1 - u0, e1 - e0) != (1, 1)
        assert (u, e) == (len(ay.uthmani), len(ay.imlaei))
        assert max(w.text_index for w in ay.words) == len(ay.uthmani) - 1
    assert multi == 58


def test_muqattaat_word_has_several_lattice_words():
    ay = quran().ayah(2, 1)
    assert ay.uthmani == ["الٓمٓ"] and len(ay.words) == 3
    assert [w.stop_after for w in ay.words] == [False, False, True]


def test_edges():
    q = quran()
    assert q.ayah(1, 1).prev is None and q.ayah(9, 1).prev is None
    assert q.ayah(2, 1).prev == "basmala" and q.ayah(2, 2).prev == "prev"
    assert not q.ayah(1, 7).has_next and q.ayah(1, 6).has_next
    first = q.ayah(2, 2).words[0]
    assert {f.left for f in first.forms} == {"ibtida", "wasl"}
    assert {f.left for f in q.ayah(1, 1).words[0].forms} == {"ibtida"}


def test_canonical():
    q = quran()
    assert q.canonical(1, 1) == "ب i س م i لّ aa ه i رّ a ح م aa ن i رّ a ح ii م".split()
    assert q.canonical(1, 1, "units")[:2] == ["بِ", "س"]


def test_khilaf_sites_present():
    ay = quran().ayah(54, 16)
    wajh = {f.wajh for w in ay.words for f in w.forms}
    assert "nudhur_tafkheem" in wajh


def test_basmala_words():
    ws = quran().basmala_words()
    assert len(ws) == 4 and all(w.basmala for w in ws)


@pytest.mark.parametrize("ref,want", [("2:255", (2, 255, 255)), ("2:255-257", (2, 255, 257)),
                                      ("2:255-2:257", (2, 255, 257)), ((1, 1), (1, 1, 1)), ((2, 3, 4), (2, 3, 4))])
def test_parse_ref(ref, want):
    assert parse_ref(ref) == want


@pytest.mark.parametrize("bad", ["2", "2:300", "1:5-3", "2:1-3:2", "115:1", "x"])
def test_parse_ref_errors(bad):
    with pytest.raises(ValueError):
        parse_ref(bad)


def test_ref_str():
    assert ref_str(2, 5, 5) == "2:5" and ref_str(2, 5, 7) == "2:5-7"
