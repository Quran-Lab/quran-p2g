from quran_p2g.align import INS, align

IB, WA, Q = "ibtida", "wasl", "waqf"


def lat(*words):
    """words: lists of (tokens, L, R)."""
    return [[tuple(t.split()) for t, _, _ in w] for w in words], [[(L, R) for _, L, R in w] for w in words]


def simple(*toks):
    """One form per word, the whole span in wasl."""
    n = len(toks)
    return lat(*[[(t, IB if i == 0 else WA, Q if i == n - 1 else WA)] for i, t in enumerate(toks)])


def test_exact_path_costs_zero():
    T, S = simple("ك a", "ت a ب a")
    al = align("ك a ت a ب a".split(), T, S, "tj3")
    assert al.cost == 0
    assert [(w.j0, w.j1) for w in al.words] == [(0, 2), (2, 6)]
    assert not al.insertions


def test_substitution_is_attributed_to_its_word():
    T, S = simple("ك a", "ت a ب a")
    al = align("ك a ت i ب a".split(), T, S, "tj3")
    assert al.cost == 4
    assert [op for op in al.words[1].ops if op[0] != "eq"] == [("sub", "a", "i")]


def test_boundary_states_must_agree():
    # word 0 has a waqf form only; then word 1 must be in ibtida
    T, S = lat([("ك a", IB, Q)], [("ء a ت", IB, Q), ("ت", WA, Q)])
    al = align("ك a ت".split(), T, S, "tj3")
    assert al.words[1].form == 0 and al.cost > 0
    al = align("ك a ء a ت".split(), T, S, "tj3")
    assert al.cost == 1                                  # one stop inside the span


def test_stop_breaks_ties_toward_wasl():
    T, S = lat([("ك a", IB, WA), ("ك", IB, Q)], [("ت a", WA, Q), ("ت a", IB, Q)])
    al = align("ك a ت a".split(), T, S, "tj3")
    assert al.cost == 0 and al.words[0].form == 0


def test_skipped_word():
    T, S = simple("ك a", "ت a ب a", "د i")
    al = align("ك a د i".split(), T, S, "tj3")
    assert al.words[1].skipped and al.words[1].j0 == al.words[1].j1
    assert al.cost == 3 * 4


def test_repeat_is_an_insertion():
    T, S = simple("ك a ت a", "ب a ب a")
    al = align("ك a ت a ك a ت a ب a ب a".split(), T, S, "tj3")
    assert [(x.kind, x.of_word, x.j0, x.j1) for x in al.insertions] == [("repeat", 0, 4, 8)]
    assert all(w.j1 - w.j0 == 4 for w in al.words)
    assert al.cost == 2 * 4 + 2


def test_repeats_can_be_disabled():
    T, S = simple("ك a ت a", "ب a ب a")
    al = align("ك a ت a ك a ت a ب a ب a".split(), T, S, "tj3", repeats=False)
    assert not al.insertions and al.cost == 4 * INS


def test_prefix_read_or_skipped():
    T, S = simple("ب i", "ك a")
    al = align("ب i ك a".split(), T, S, "tj3", n_prefix=1)
    assert al.prefix_read and al.cost == 0
    T2, S2 = lat([("ب i", IB, WA), ("ب", IB, Q)], [("ك a", IB, Q)])
    al = align("ك a".split(), T2, S2, "tj3", n_prefix=1)
    assert not al.prefix_read and al.cost == 0 and [w.word for w in al.words] == [1]


def test_open_edges():
    T, S = lat([("ل ك a", WA, WA), ("ء a ل ك a", IB, WA)], [("ب a", WA, WA), ("ب", WA, Q)])
    al = align("ل ك a ب a".split(), T, S, "tj3", open_left=True, open_right=True)
    assert al.cost == 0
    al = align("ل ك a ب a".split(), T, S, "tj3")
    assert al.cost > 0


def test_free_edges():
    T, S = simple("ك a", "ت a")
    al = align("x y ك a ت a z".split(), T, S, "tj3", free_edges=True)
    assert al.cost == 0
    assert sorted((x.kind, x.j0, x.j1) for x in al.insertions) == [("edge", 0, 2), ("edge", 6, 7)]


def test_empty_hypothesis_deletes_everything():
    T, S = simple("ك a", "ت a")
    al = align([], T, S, "tj3")
    assert all(w.j0 == w.j1 for w in al.words)
