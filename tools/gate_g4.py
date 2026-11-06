"""G4 unconstrained retrieval: top-1 ayah identification.

    python -X utf8 tools/gate_g4.py --out gates/g4.json

1. The canonical tj3 of every one of the 6,236 ayat (rendered from the packaged lattice; G1 shows it equals
   quran-g2p's phonemize output) as the query.
2. The AGG (agg18k) V4 benchmark hypotheses of the public tier t1p_panel_correct_test, against the row's ayah.

Correct (strict): the top candidate is exactly the ayah. Correct up to identical text: the top candidate is the ayah or
an ayah with the same folded canonical phones (the Quran repeats some ayat word for word, e.g. 55:13 and 55:16;
no phone string can tell them apart). A top candidate that is a range is counted correct only in the
"contains" column. The last column counts a query correct when the top candidate is the ayah, an ayah with the same
folded phones, or a range that contains the ayah (a clip that runs into the next ayah is identified as the range it
covers).

The exact-ayah ceiling on the canonical set: a phone string shared by k ayat is one query answer for all k of them,
so at most one of the k can be answered exactly; with groups of sizes k the ceiling is 6,236 - sum(k - 1).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def q_canon(key):
    import quran_p2g as p2g
    s, a = key
    toks = [t[0] if len(t) == 2 and t[1] == "\u0651" else t for t in p2g.quran().canonical(s, a)]
    return _judge(f"{s}:{a}", p2g.identify(toks, "tj3", k=5))


def q_hyp(item):
    import quran_p2g as p2g
    ayah, tj3 = item
    return _judge(ayah, p2g.identify(tj3, "tj3", k=5))


def _judge(ayah, cands):
    if not cands:
        return {"ayah": ayah, "top": None, "strict": False, "same_text": False, "contains": False,
                "same_or_range": False}
    top = cands[0]
    s, a = map(int, ayah.split(":"))
    strict = top.ref == ayah
    same = strict or (top.first == top.last and ayah in [f"{top.surah}:{top.first}"] + top.identical) or \
        (top.first == top.last and _same_text(top, s, a))
    contains = top.surah == s and top.first <= a <= top.last
    return {"ayah": ayah, "top": top.ref, "score": round(top.score, 4), "margin": round(top.margin, 4),
            "strict": strict, "same_text": same, "contains": contains, "same_or_range": same or contains}


def _same_text(top, s, a):
    from quran_p2g.retrieve import index
    ix = index()
    return ix.codes[ix.pos[(top.surah, top.first)]].tobytes() == ix.codes[ix.pos[(s, a)]].tobytes()


def summarize(rs):
    n = len(rs)
    return {"queries": n, "top1_strict_pct": round(100 * sum(r["strict"] for r in rs) / n, 2),
            "top1_same_text_pct": round(100 * sum(r["same_text"] for r in rs) / n, 2),
            "top1_contains_pct": round(100 * sum(r["contains"] for r in rs) / n, 2),
            "top1_same_text_or_range_pct": round(100 * sum(r["same_or_range"] for r in rs) / n, 2),
            "top1_strict": sum(r["strict"] for r in rs),
            "misses": [r for r in rs if not r["same_text"]][:60]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "gates", "g4.json"))
    ap.add_argument("--hyps", default="F:/v4_bench_reads/agg18k/hyps.jsonl")
    ap.add_argument("--manifest", default="F:/v4_bench/manifests/t1p_panel_correct_test.jsonl.gz")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    import gzip
    import quran_p2g as p2g
    t0 = time.time()
    keys = sorted(p2g.quran().keys())
    with Pool(args.workers) as pool:
        canon = pool.map(q_canon, keys, chunksize=32)
        out = {"canonical_all_ayat": summarize(canon)}
        print(json.dumps({k: v for k, v in out["canonical_all_ayat"].items() if k != "misses"}), flush=True)
        if os.path.exists(args.hyps) and os.path.exists(args.manifest):
            hyps = {}
            for l in open(args.hyps, encoding="utf-8"):
                d = json.loads(l)
                hyps[d["id"]] = d["tj3"]
            items = []
            for l in gzip.open(args.manifest, "rt", encoding="utf-8"):
                r = json.loads(l)
                h = hyps.get(r["id"], hyps.get(r["id"].split(":", 1)[-1]))
                if h:
                    items.append((r["ayah"], h))
            hr = pool.map(q_hyp, items, chunksize=8)
            out["agg18k_t1p_panel_correct_test"] = summarize(hr)
            print(json.dumps({k: v for k, v in out["agg18k_t1p_panel_correct_test"].items() if k != "misses"}))
    out["seconds"] = round(time.time() - t0, 1)
    sizes = collections.Counter(len(v) for v in __import__("quran_p2g.retrieve", fromlist=["index"]).index()
                                .text_of.values())
    out["identical_text_groups"] = {str(k): v for k, v in sorted(sizes.items())}
    n = sum(k * v for k, v in sizes.items())
    ceiling = n - sum((k - 1) * v for k, v in sizes.items())
    out["canonical_strict_ceiling"] = {"ayat": n, "answerable_exactly": ceiling,
                                       "pct": round(100 * ceiling / n, 2)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
