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
            fl = theirs[0]
            tw = collections.defaultdict(bool)
            for lw, f in zip(ay.words, fl):
                tw[lw.text_index] = tw[lw.text_index] or bool(f)
            ours = {w.index: w.status != "match" for w in r.words}
            dis = []
            for i in sorted(ours):
                if ours[i] != tw[i]:
                    dis.append(_classify(i, ours, tw, r, ay))
            stats["cmp"] = {"words": len(ours), "agree": len(ours) - len(dis), "dis": dis,
                            "ours_flagged": sum(ours.values()), "theirs_flagged": sum(tw[i] for i in ours)}
    return out, stats


def _classify(i, ours, theirs, r, ay):
    """Why the two word flags differ at Uthmani word i."""
    ours_flag = ours[i]
    w = r.words[i]
    for j in (i - 1, i + 1):
        if j in ours and ours[j] != theirs[j] and ours[j] != ours_flag:
            return ["one_error_attributed_to_neighbouring_word", i, " ".join(w.phones), " ".join(w.expected)]
    n = len(ay.uthmani)
    ins_after = {x.after for x in r.insertions if x.kind == "repeat"}
    key = f"{ay.surah}:{ay.ayah}:{i}"
    prev_key = f"{ay.surah}:{ay.ayah}:{i - 1}"
    if not ours_flag:
        if i == 0 and w.form.get("left") == "wasl":
            return ["ours_match_wasl_from_previous_ayah", i]
        if i == n - 1 and w.form.get("right") == "wasl":
            return ["ours_match_wasl_into_next_ayah", i]
        if key in ins_after or prev_key in ins_after or any(x.kind == "repeat" for x in r.insertions):
            return ["ours_repeat_insertion_theirs_absorbed", i]
        if any(x.kind == "basmala" for x in r.insertions) and i == 0:
            return ["ours_basmala_prefix", i]
        return ["ours_match_other", i, " ".join(w.phones)]
    if w.status == "deleted":
        return ["ours_deleted_theirs_match", i]
    return ["ours_deviant_theirs_match", i, " ".join(w.phones), " ".join(w.expected)]


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
                dcls = collections.Counter(d[0] for c in cmp for d in c["dis"])
                ex = collections.defaultdict(list)
                for (o, s) in res:
                    if s and s.get("cmp"):
                        for d in s["cmp"]["dis"]:
                            if len(ex[d[0]]) < 6:
                                ex[d[0]].append([o["id"], o["ayah"]] + d[1:])
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
                print(m, t, a["agree_pct"], a["disagreement_classes"])


if __name__ == "__main__":
    main()
