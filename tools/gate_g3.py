"""G3 real model output: V4 benchmark tj3 hypotheses -> text, WER, and word statuses against the benchmark scorer.

    python -X utf8 tools/gate_g3.py --out gates/g3.json --text-dir F:/v4_bench/_work/p2g

For every model (agg18k, v31) and every manifest tier with hypotheses, each hypothesis is converted in constrained
mode with the row's ayah (quran-p2g defaults: open edges, basmala allowed, strict edges). Writes one JSONL per model
and tier to --text-dir (id, ayah, text, text_imlaei, counts, word statuses) for the benchmark to consume.

Reported (README) only for the public correct tiers t1p_panel_correct_dev / t1p_panel_correct_test:
  - corpus WER, undiacritized against the imla'i text (board normalisation) and diacritized against the Uthmani text
  - per-word agreement with score_v4_bench.tj3_word_flags (imported read-only) on the same hypotheses, the lattice
    words of a muqatta'at word folded to its Uthmani word; every disagreement is put in a class
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
PUBLIC = ("t1p_panel_correct_dev", "t1p_panel_correct_test")
MODELS = {"agg18k": "F:/v4_bench_reads/agg18k/hyps.jsonl", "v31": "F:/v4_bench_reads/v31/hyps.jsonl"}
BENCH = "F:/v4_bench"
S = None


def _scorer():
    global S
    if S is None:
        sys.path.insert(0, BENCH)
        import score_v4_bench as mod          # read-only import, never modified
        S = mod
    return S


def job(args):
    row, hyp_tj3, compare = args
    import quran_p2g as p2g
    sc = _scorer()
    ayah = sc.row_ayah(row)
    out = {"id": row["id"], "ayah": ayah}
    if not ayah:
        out["error"] = "no ayah"
        return out, None
    try:
        r = p2g.convert(hyp_tj3, ayah, "tj3")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out, None
    out.update({"text": r.text, "text_imlaei": r.text_imlaei, "counts": r.counts,
                "words": [[w.index, w.status] + ([w.level] if w.status == "deviant" else []) for w in r.words],
                "insertions": [[x.kind, x.after, " ".join(x.phones)] for x in r.insertions]})
    wu = p2g.wer(r, ayah)
    wd = p2g.wer(r, ayah, diacritics=True)
    stats = {"wer_u": [wu.errors, wu.ref_words], "wer_d": [wd.errors, wd.ref_words]}
    if compare:
        lat = row.get("lattice")
        q = p2g.quran()
        s, a = map(int, ayah.split(":"))
        ay = q.ayah(s, a)
        toks = sc.toks(hyp_tj3)
        theirs = sc.tj3_word_flags(lat, toks) if lat and len(lat["words"]) == len(ay.words) else None
        if theirs is None:
            stats["cmp"] = None
        else:
            stats["cmp"] = compare_flags(r, ay, lat, toks, theirs[0], hyp_tj3)
    return out, stats


def _fold(ay, lattice_flags):
    """lattice-word flags -> Uthmani-word flags (a muqatta'at word is flagged when any of its letter names is)."""
    tw = collections.defaultdict(bool)
    for lw, f in zip(ay.words, lattice_flags):
        tw[lw.text_index] = tw[lw.text_index] or bool(f)
    return tw


def bench_convention(ay, hyp, path_legality=False, open_edges=False, repeats=False):
    """quran-p2g run under the benchmark scorer's conventions: closed edges (ibtida at the start, waqf at the end), no
    basmala item, no repeat items (repeated material stays inside the word it is aligned to), and a word flagged when
    its phones are none of its legal forms in any boundary state (per-word legality). Each convention can be switched
    back to quran-p2g's own to see which one a disagreement comes from.
    -> (alignment, per lattice word segments, per lattice word flags, the forms of each lattice word)"""
    from quran_p2g import align as AL
    from quran_p2g.convert import _prep
    preps = [_prep(w, "tj3") for w in ay.words]
    al = AL.align(hyp, [p[0] for p in preps], [p[1] for p in preps], "tj3",
                  open_left=open_edges and ay.prev == "prev", open_right=open_edges and ay.has_next, repeats=repeats)
    n = len(ay.words)
    segs = [()] * n
    flags = [True] * n
    for x in al.words:
        segs[x.word] = tuple(hyp[x.j0:x.j1])
        if path_legality:
            flags[x.word] = segs[x.word] != preps[x.word][0][x.form]
    if not path_legality:
        for w in range(n):
            legal = {t for t, (L, R) in zip(*preps[w][:2])
                     if not (w == 0 and L == "wasl") and not (w == n - 1 and R == "wasl")}
            flags[w] = segs[w] not in legal
    return al, segs, flags, preps


def their_cost(lat, groups):
    """The scorer's own cost (unit Levenshtein over its lattice, boundary states agreeing, no stop cost) of the
    hypothesis cut into fixed groups [(lattice word indices, tokens)]; inside a group the cut between lattice words is
    free. One group over the whole hypothesis gives the scorer's optimum."""
    import numpy as np
    sc = _scorer()
    cost = {"stop": 0}
    for wids, X in groups:
        X = list(X)
        m = len(X)
        rows = {e: c + np.arange(m + 1, dtype=np.int64) for e, c in cost.items()}
        for w in wids:
            new = {}
            for f in lat["words"][w]["forms"]:
                src = "stop" if f["L"] == "ibtida" else "wasl"
                if src not in rows:
                    continue
                last = sc._lev_from(rows[src], f["tj3"], X)[-1]
                dst = "stop" if f["R"] == "waqf" else "wasl"
                new[dst] = last if dst not in new else np.minimum(new[dst], last)
            rows = new
        cost = {e: int(v[m]) for e, v in rows.items()}
    return cost.get("stop")


def our_cost(preps, segs):
    """quran-p2g's cost (closed edges, no repeats) of the hypothesis cut into fixed per lattice word segments."""
    import g2_equiv as EQ
    from quran_p2g.align import STOP, skip_cost
    n = len(preps)
    best = {"stop": 0}
    for w in range(n):
        toks, states, _ = preps[w]
        skip = skip_cost(min(len(t) for t in toks))
        new = {}
        for t, (L, R) in zip(toks, states):
            src = "stop" if L == "ibtida" else "wasl"
            if src not in best or (w == 0 and L == "wasl") or (w == n - 1 and R == "wasl"):
                continue
            c = EQ.weighted_lev(t, list(segs[w]), "tj3")
            if not segs[w]:
                c = min(c, skip)
            c += best[src] + (STOP if (R == "waqf" and w < n - 1) else 0)
            dst = "stop" if R == "waqf" else "wasl"
            new[dst] = min(new.get(dst, c), c)
        best = new
    return best.get("stop")


def compare_flags(r, ay, lat, toks, their_lat_flags, hyp_tj3):
    """Per Uthmani word: quran-p2g's flag (not a match) vs the scorer's, raw and under the scorer's conventions.

    Every raw disagreement gets the first convention that removes it (b): open edges (wasl into the next / from the
    previous ayah), the basmala item, repeat items, path legality (boundary states must agree along the reading).
    A disagreement left under all of the scorer's conventions is an attribution tie (a) when quran-p2g's segmentation
    costs the scorer's own optimum under the scorer's cost (or the scorer's segmentation costs quran-p2g's optimum
    under quran-p2g's cost); anything else is reported as a difference."""
    from quran_p2g.formats import tokenize
    hyp = tokenize(hyp_tj3, "tj3")
    tw = _fold(ay, their_lat_flags)
    ours = {w.index: w.status != "match" for w in r.words}
    raw_dis = sorted(i for i in ours if ours[i] != tw[i])
    # the conventions, switched to the scorer's one at a time
    steps = [("wasl_edge", dict(path_legality=True, open_edges=False, repeats=True)),
             ("repeat_item", dict(path_legality=True, open_edges=False, repeats=False)),
             ("path_legality", dict(path_legality=False, open_edges=False, repeats=False))]
    flags_at = {}
    for name, kw in steps:
        al, segs, fl, preps = bench_convention(ay, hyp, **kw)
        flags_at[name] = _fold(ay, fl)
    conv_al, conv_segs, conv_fl, preps = al, segs, fl, preps
    conv = flags_at["path_legality"]
    classes = {}
    basmala = any(x.kind == "basmala" for x in r.insertions)
    for i in raw_dis:
        cls = None
        if basmala and i == 0:
            cls = "basmala_item"
        for name, _ in steps:
            if cls is None and flags_at[name][i] == tw[i]:
                cls = name
        classes[i] = cls
    conv_dis = sorted(i for i in ours if conv[i] != tw[i])
    ties = {}
    if conv_dis:
        n = len(ay.words)
        opt_theirs = their_cost(lat, [(list(range(n)), hyp)])
        ours_in_theirs = their_cost(lat, [([w], conv_segs[w]) for w in range(n)])
        their_segs = [tuple(seg) for seg, _ in _scorer().lattice_word_spans(lat, hyp)]
        theirs_in_ours = our_cost(preps, their_segs)
        for i in conv_dis:
            if ours_in_theirs == opt_theirs:
                ties[i] = "tie_in_scorer_cost"
            elif theirs_in_ours == conv_al.cost:
                ties[i] = "tie_in_p2g_cost"
            else:
                ties[i] = "differs"
    dis = []
    for i in sorted(set(raw_dis) | set(conv_dis)):
        w = r.words[i]
        dis.append({"word": i, "ours": ours[i], "theirs": tw[i], "raw": i in raw_dis,
                    "convention": classes.get(i), "under_convention": ties.get(i),
                    "phones": " ".join(w.phones), "expected": " ".join(w.expected)})
    n_words = len(ours)
    return {"words": n_words, "agree": n_words - len(raw_dis),
            "agree_convention": n_words - len(conv_dis),
            "agree_convention_ties": n_words - sum(1 for i in conv_dis if ties[i] == "differs"),
            "dis": dis, "ours_flagged": sum(ours.values()), "theirs_flagged": sum(tw[i] for i in ours)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "gates", "g3.json"))
    ap.add_argument("--text-dir", default="F:/v4_bench/_work/p2g")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tiers", default="")
    args = ap.parse_args()
    sc = _scorer()
    rows = {}
    for p in sorted(glob.glob(os.path.join(BENCH, "manifests", "*.jsonl.gz"))):
        tier = os.path.basename(p)[:-len(".jsonl.gz")]
        if args.tiers and tier not in args.tiers.split(","):
            continue
        for r in sc.read_jsonl(p):
            rows.setdefault(tier, []).append(r)
    summary = {"models": {}, "note": "README numbers: public correct tiers only"}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for model, path in MODELS.items():
            hyps = {}
            for l in open(path, encoding="utf-8"):
                d = json.loads(l)
                hyps[d["id"]] = d["tj3"]
            msum = {}
            for tier, rs in rows.items():
                jobs = []
                for r in rs:
                    h = hyps.get(r["id"], hyps.get(r["id"].split(":", 1)[-1]))
                    if h is not None:
                        jobs.append((r, h, tier in PUBLIC))
                if not jobs:
                    continue
                res = pool.map(job, jobs, chunksize=8)
                os.makedirs(os.path.join(args.text_dir, model), exist_ok=True)
                with open(os.path.join(args.text_dir, model, tier + ".jsonl"), "w", encoding="utf-8") as f:
                    for o, _ in res:
                        f.write(json.dumps(o, ensure_ascii=False) + "\n")
                if tier not in PUBLIC:
                    msum[tier] = {"rows": len(res), "errors": sum(1 for o, _ in res if "error" in o)}
                    continue
                st = [s for _, s in res if s]
                eu = sum(s["wer_u"][0] for s in st)
                nu = sum(s["wer_u"][1] for s in st)
                ed = sum(s["wer_d"][0] for s in st)
                nd = sum(s["wer_d"][1] for s in st)
                cmp = [s["cmp"] for s in st if s.get("cmp")]
                words = sum(c["words"] for c in cmp)
                agree = sum(c["agree"] for c in cmp)
                agree_c = sum(c["agree_convention"] for c in cmp)
                agree_ct = sum(c["agree_convention_ties"] for c in cmp)
                # raw disagreements by the convention that removes them (b), or by the tie class left under all of
                # the scorer's conventions (a), or "differs"; plus words that disagree only under the conventions
                dcls = collections.Counter()
                ex = []
                for (o, s) in res:
                    if s and s.get("cmp"):
                        for d in s["cmp"]["dis"]:
                            k = ("raw:" if d["raw"] else "convention_only:") + (d["convention"] or d["under_convention"])
                            dcls[k] += 1
                            ex.append(dict(d, id=o["id"], ayah=o["ayah"], cls=k))
                counts = collections.Counter()
                for o, _ in res:
                    for k, v in (o.get("counts") or {}).items():
                        counts[k] += v
                msum[tier] = {"rows": len(res), "converted": len(st),
                              "errors": sum(1 for o, _ in res if "error" in o),
                              "wer_undiacritized_imlaei": round(100 * eu / max(1, nu), 2),
                              "wer_diacritized_uthmani": round(100 * ed / max(1, nd), 2),
                              "ref_words": nu, "word_counts": dict(counts),
                              "agreement": {"rows_compared": len(cmp), "words": words, "agree": agree,
                                            "agree_pct": round(100 * agree / max(1, words), 2),
                                            "agree_convention": agree_c,
                                            "agree_convention_pct": round(100 * agree_c / max(1, words), 2),
                                            "agree_convention_ties": agree_ct,
                                            "agree_convention_ties_pct": round(100 * agree_ct / max(1, words), 2),
                                            "ours_flagged": sum(c["ours_flagged"] for c in cmp),
                                            "theirs_flagged": sum(c["theirs_flagged"] for c in cmp),
                                            "disagreement_classes": dict(dcls), "examples": ex}}
                print(model, tier, json.dumps({k: v for k, v in msum[tier].items() if k != "agreement"},
                                              ensure_ascii=False), flush=True)
            summary["models"][model] = msum
    summary["seconds"] = round(time.time() - t0, 1)
    # the full report (with per-clip examples) stays next to the text hyps; the repo gets public aggregates only
    json.dump(summary, open(os.path.join(args.text_dir, "g3_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    public = {"models": {m: {t: {k: v for k, v in ms[t].items() if k != "agreement"} |
                             {"agreement": {k: v for k, v in ms[t]["agreement"].items() if k != "examples"}}
                             for t in PUBLIC if t in ms} for m, ms in summary["models"].items()},
              "note": "public correct tiers only; per-clip outputs are not published"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(public, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for m, ms in summary["models"].items():
        for t in PUBLIC:
            if t in ms:
                a = ms[t]["agreement"]
                print(m, t, a["agree_pct"], a["agree_convention_pct"], a["agree_convention_ties_pct"], a["disagreement_classes"])


if __name__ == "__main__":
    main()
