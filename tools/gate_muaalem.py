"""Muaalem format round trip: quran_transcript's phonetizer (the script obadx muaalem models are trained on) for every
ayah -> quran-p2g (fmt="muaalem") -> text.

    python -X utf8 tools/gate_muaalem.py --out gates/muaalem.json

quran_transcript reads its own Uthmani text (Aya(s, a).get().uthmani) with MoshafAttributes(rewaya="hafs", madd
lengths 4); it is an independent implementation of the Hafs rules, so a difference is either a different default
wajh or a disagreement between the two engines. Reported: ayat whose script maps to exactly quran-g2p's canonical tj3,
ayat that convert to the mushaf text with every word a match (a different legal form still matches), and every
residual with the words it touches.
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
_M = None


def _moshaf():
    global _M
    if _M is None:
        from quran_transcript import MoshafAttributes
        _M = MoshafAttributes(rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4, madd_mottasel_waqf=4,
                              madd_aared_len=4)
    return _M


def run(key):
    import quran_p2g as p2g
    from quran_p2g.formats import degeminate, muaalem_to_core
    from quran_transcript import Aya, quran_phonetizer
    s, a = key
    out = {"ayah": f"{s}:{a}"}
    try:
        ph = quran_phonetizer(Aya(s, a).get().uthmani, _moshaf()).phonemes
    except Exception as e:
        out["error"] = f"phonetizer: {type(e).__name__}"
        return out
    q = p2g.quran()
    canon = [degeminate(t) for t in q.canonical(s, a)]
    mapped = [degeminate(t) for t in muaalem_to_core(ph)]
    out["tokens_equal"] = mapped == canon
    r = p2g.convert(ph, (s, a), "muaalem")
    ay = q.ayah(s, a)
    out["text_equal"] = r.text == ay.text and r.text_imlaei == ay.text_imlaei
    out["all_match"] = r.counts["match"] == len(r.words) and not r.insertions
    if not out["all_match"]:
        out["words"] = [[w.index, w.uthmani, w.status, " ".join(w.phones), " ".join(w.expected)]
                        for w in r.words if w.status != "match"] + \
                       [["ins", x.kind, " ".join(x.phones)] for x in r.insertions]
        out["script"] = ph
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "gates", "muaalem.json"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    import quran_p2g as p2g
    t0 = time.time()
    keys = sorted(p2g.quran().keys())
    with Pool(args.workers) as pool:
        rs = pool.map(run, keys, chunksize=32)
    bad = [r for r in rs if not r.get("all_match")]
    pairs = collections.Counter()
    for r in bad:
        for w in r.get("words", []):
            if w[0] != "ins":
                pairs[(w[3], w[4])] += 1
    out = {"ayat": len(rs), "phonetizer_errors": sum(1 for r in rs if "error" in r),
           "tokens_equal_canonical": sum(1 for r in rs if r.get("tokens_equal")),
           "text_round_trip": sum(1 for r in rs if r.get("text_equal")),
           "all_words_match": sum(1 for r in rs if r.get("all_match")),
           "residual_ayat": len(bad),
           "residual_word_pairs_said_vs_expected": [[a, b, n] for (a, b), n in pairs.most_common(60)],
           "residuals": bad[:400], "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("residuals",)}, ensure_ascii=False, indent=1)[:5000])


if __name__ == "__main__":
    main()
