"""G1 round trip: quran-g2p phones -> quran-p2g -> text, over all 6,236 ayat.

    PYTHONPATH=src;<quran-g2p>/src python -X utf8 tools/gate_g1.py --g2p <quran-g2p checkout> --out gates/g1.json

The phones come straight from quran-g2p (a fresh single-ayah lattice per ayah, and phonemize / phonemize_concat for
the canonical, cross-ayah and basmala paths), never from quran-p2g's packaged artifact. Paths per ayah:

  stops   none (canonical: stop only at the ayah end), all (a stop at every stoppable boundary), even, odd
  modes   sukun at every stop; rawm at every stop that allows it; ishmam at every stop that allows it
  wajh    the base wajh, and the alternate wajh at every khilaf site of the ayah (when it has any)
  range   consecutive ayat read in wasl across the ayah ends (phonemize_concat), in chunks of 4 per surah
  basmala the basmala read before ayah 1 in wasl (surahs other than 1 and 9)

Each path in tj3, tj3g and quran-g2p unit tokens. Pass: every word `match`, no insertion, text == the Tanzil
Uthmani text and text_imlaei == the KFGQPC imla'i text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

SHADDA = "\u0651"
G2P = None


def _setup(g2p):
    global G2P
    G2P = g2p
    sys.path.insert(0, os.path.join(g2p, "src"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def renders(phones):
    from quran_g2p import tokenlayer_tj3 as T3
    from quran_g2p.tokenlayer import phones_to_tokens
    g = []
    for p in phones:
        n0 = len(g)
        T3._emit(g, p.base.value)
        if p.kind == "consonant" and p.geminated and len(g) > n0 and g[-1] in T3.CONSONANTS:
            g[-1] += SHADDA
    tj3 = T3.phones_to_tj3(phones, boundaries=False)
    assert [t[0] if t.endswith(SHADDA) and len(t) == 2 else t for t in g] == tj3
    return {"tj3": tj3, "tj3g": g, "units": [t.text for t in phones_to_tokens(list(phones))]}


def _check(r, want_uth, want_iml):
    c = r.counts
    ok = c["match"] == len(r.words) and c["deviant"] == c["deleted"] == c["inserted"] == 0
    ok = ok and r.text == want_uth and r.text_imlaei == want_iml
    return ok


def run_ayah(key):
    _setup(G2P) if G2P is None else None
    from quran_g2p.lattice import build_lattice
    from quran_g2p.phonemize import phonemize
    from quran_g2p.textbank import AyahRef, TextBank
    import quran_p2g as p2g
    s, a = key
    tb = _tb()
    ref = AyahRef(s, a)
    text = tb.ayah(ref)
    ay = p2g.quran().ayah(s, a)
    lat = build_lattice([(ref, text)])
    n = lat.n_words
    stoppable = [k for k in range(n - 1) if lat.stoppable[k]]
    pats = {"none": [], "all": stoppable, "even": stoppable[0::2], "odd": stoppable[1::2]}
    choices = [("base", {})]
    if lat.sites:
        choices.append(("alt", {sid: v["wajh"] for sid, v in lat.sites.items()}))
    paths = {}
    canon = [p for seg in phonemize(text, edition="tanzil", ref=ref).segments for p in seg.phones]
    paths[("phonemize", "", "")] = canon
    lat_canon_equal = None
    for pname, stops in pats.items():
        for mode in ("sukun", "rawm", "ishmam"):
            ends = set(stops) | {n - 1}
            modes = {}
            for w in ends:
                L = "ibtida" if (w == 0 or (w - 1) in stops) else "wasl"
                fs = lat.word_forms(w, L, "waqf")
                if any(f.mode == mode for f in fs):
                    modes[w] = mode
            for cname, choice in choices:
                try:
                    ph = lat.path(stops, modes, choice)
                except KeyError:
                    continue
                paths[(pname, mode, cname)] = ph
                if pname == "none" and mode == "sukun" and cname == "base":
                    lat_canon_equal = [p.base for p in ph] == [p.base for p in canon]
    out = {"key": f"{s}:{a}", "paths": 0, "fail": [], "lattice_canon_equals_phonemize": lat_canon_equal}
    seen = set()
    for pk, ph in paths.items():
        rd = renders(ph)
        for fmt, toks in rd.items():
            k = (fmt, tuple(toks))
            if k in seen:
                continue
            seen.add(k)
            out["paths"] += 1
            r = p2g.convert(toks, (s, a), fmt)
            if not _check(r, ay.text, ay.text_imlaei):
                out["fail"].append({"path": list(pk), "fmt": fmt, "counts": r.counts, "text": r.text})
    return out


_TB_ = None


def _tb():
    global _TB_
    if _TB_ is None:
        from quran_g2p.textbank import TextBank
        _TB_ = TextBank.load("tanzil")
    return _TB_


def run_range(task):
    """('range', s, a0, a1) or ('basmala', s)."""
    from quran_g2p.concat import phonemize_concat
    from quran_g2p.textbank import AyahRef
    import quran_p2g as p2g
    tb = _tb()
    q = p2g.quran()
    out = {"task": list(task), "paths": 0, "fail": []}
    if task[0] == "range":
        _, s, a0, a1 = task
        items = [(AyahRef(s, a), tb.ayah(AyahRef(s, a))) for a in range(a0, a1 + 1)]
        want_u = " ".join(q.ayah(s, a).text for a in range(a0, a1 + 1))
        want_i = " ".join(q.ayah(s, a).text_imlaei for a in range(a0, a1 + 1))
        ref = (s, a0, a1)
    else:
        _, s = task
        items = [(AyahRef(1, 1), tb.ayah(AyahRef(1, 1))), (AyahRef(s, 1), tb.ayah(AyahRef(s, 1)))]
        want_u, want_i, ref = q.ayah(s, 1).text, q.ayah(s, 1).text_imlaei, (s, 1)
    res = phonemize_concat(items, edition="tanzil")
    ph = [p for seg in res.segments for p in seg.phones]
    for fmt, toks in renders(ph).items():
        out["paths"] += 1
        r = p2g.convert(toks, ref, fmt)
        ok = _check(r, want_u, want_i)
        if task[0] == "basmala":
            b = [x for x in r.items if getattr(x, "kind", "") == "basmala"]
            ok = ok and len(b) == 1 and b[0].match
        if not ok:
            out["fail"].append({"fmt": fmt, "counts": r.counts, "text": r.text})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2p", default="C:/Users/Anon/research/quran-g2p")
    ap.add_argument("--out", default="gates/g1.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    _setup(args.g2p)
    import quran_p2g as p2g
    keys = sorted(p2g.quran().keys())
    if args.limit:
        keys = keys[:args.limit]
    t0 = time.time()
    with Pool(args.workers, initializer=_setup, initargs=(args.g2p,)) as pool:
        ayat = list(pool.imap(run_ayah, keys, chunksize=16))
        print(f"ayat done {time.time() - t0:.0f}s", flush=True)
        tasks = []
        q = p2g.quran()
        for s in sorted(q.n_ayat_in):
            n = q.n_ayat_in[s]
            if args.limit and (s, 1) not in keys:
                continue
            for a0 in range(1, n + 1, 4):
                tasks.append(("range", s, a0, min(n, a0 + 3)))
            if s not in (1, 9):
                tasks.append(("basmala", s))
        rng = list(pool.imap(run_range, tasks, chunksize=4))
    fails = [x for x in ayat if x["fail"]]
    rfails = [x for x in rng if x["fail"]]
    summ = {
        "ayat": len(ayat),
        "ayat_all_paths_pass": len(ayat) - len(fails),
        "single_ayah_paths": sum(x["paths"] for x in ayat),
        "single_ayah_path_failures": sum(len(x["fail"]) for x in ayat),
        "lattice_canonical_equals_phonemize": sum(1 for x in ayat if x["lattice_canon_equals_phonemize"]),
        "range_and_basmala_tasks": len(rng),
        "range_paths": sum(x["paths"] for x in rng if x["task"][0] == "range"),
        "basmala_paths": sum(x["paths"] for x in rng if x["task"][0] == "basmala"),
        "range_and_basmala_failures": sum(len(x["fail"]) for x in rng),
        "seconds": round(time.time() - t0, 1),
        "failures": fails[:50], "range_failures": rfails[:50],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summ, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if "failures" not in k or isinstance(v, int)},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
