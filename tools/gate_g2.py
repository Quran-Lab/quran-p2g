"""G2 planted single edits: does quran-p2g flag exactly the edited word, and does its spelling read back?

    PYTHONPATH=src python -X utf8 tools/gate_g2.py --fmt tj3g --per-ayah 4 --out gates/g2_tj3g.json

For every ayah, `per-ayah` cases are planted on its canonical path (ibtida at the start, waqf with sukun at the end,
wasl inside), cycling through the edit types and choosing the word and the position at random (seed fixed):

  vowel_swap         a short vowel replaced by another short vowel
  consonant_swap     a consonant replaced by a confusable one (emphatic pairs, throat pairs, interdental pairs)
  long_short         a long vowel shortened or a short vowel lengthened
  shadda_drop        a held consonant said plain (tj3g only: tj3 has no held token)
  word_skip          the word not said
  word_repeat        the word said twice
  inserted_syllable  a consonant + short vowel inserted inside the word, before one of its consonants

An edit that turns the word into another of its legal forms is redrawn. Only Uthmani words with one lattice word are
edited (a muqatta'at word holds several). Expected: the edited word deviant (skip: deleted; repeat: one inserted
repeat of that word and every word a match), every other word a match, nothing else inserted.

Spelling read-back (substitution-type edits): the spelled word replaces the word in the ayah text, quran-g2p reads the
ayah (canonical stops), and that word's tokens must equal the planted tokens.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from multiprocessing import Pool

SHADDA = "\u0651"
HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = [("س", "ص"), ("ت", "ط"), ("د", "ض"), ("ذ", "ظ"), ("ز", "ظ"), ("ض", "ظ"), ("د", "ط"), ("ك", "ق"),
         ("ز", "ص"), ("ث", "س"), ("ذ", "ز"), ("ذ", "د"), ("ث", "ت"), ("ء", "ع"), ("ه", "ح"), ("ح", "ع"),
         ("خ", "غ"), ("ح", "خ")]
CONF = collections.defaultdict(list)
for x, y in PAIRS:
    CONF[x].append(y)
    CONF[y].append(x)
TYPES_G = ["vowel_swap", "consonant_swap", "long_short", "shadda_drop", "word_skip", "word_repeat", "inserted_syllable"]
CONS = list("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي")
G2P = None


def _setup(g2p):
    global G2P
    G2P = g2p
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    if g2p:
        sys.path.insert(0, os.path.join(g2p, "src"))


def degem(t):
    return t[0] if t.endswith(SHADDA) and len(t) == 2 else t


def canonical_words(q, s, a, fmt):
    ay = q.ayah(s, a)
    n = len(ay.words)
    out = []
    for k, w in enumerate(ay.words):
        L = "ibtida" if k == 0 else "wasl"
        R = "waqf" if k == n - 1 else "wasl"
        mode = "sukun" if R == "waqf" else "plain"
        f = next(f for f in w.forms if f.left == L and f.right == R and f.mode == mode and f.wajh is None)
        toks = f.tj3g.split()
        if fmt == "tj3":
            toks = [degem(t) for t in toks]
        legal = set()
        for g in w.forms:
            t = g.tj3g.split()
            legal.add(tuple(degem(x) for x in t) if fmt == "tj3" else tuple(t))
        out.append((toks, legal, w.text_index))
    return out


def plant(rng, typ, toks):
    """-> edited tokens or None when the edit does not apply to this word."""
    t = list(toks)
    if typ == "vowel_swap":
        idx = [i for i, x in enumerate(t) if x in ("a", "i", "u")]
        if not idx:
            return None
        i = rng.choice(idx)
        t[i] = rng.choice([v for v in ("a", "i", "u") if v != t[i]])
    elif typ == "consonant_swap":
        idx = [i for i, x in enumerate(t) if degem(x) in CONF]
        if not idx:
            return None
        i = rng.choice(idx)
        g = t[i].endswith(SHADDA) and len(t[i]) == 2
        t[i] = rng.choice(CONF[degem(t[i])]) + (SHADDA if g else "")
    elif typ == "long_short":
        idx = [i for i, x in enumerate(t) if x in ("a", "i", "u", "aa", "ii", "uu")]
        if not idx:
            return None
        i = rng.choice(idx)
        t[i] = {"a": "aa", "i": "ii", "u": "uu", "aa": "a", "ii": "i", "uu": "u"}[t[i]]
    elif typ == "shadda_drop":
        idx = [i for i, x in enumerate(t) if x.endswith(SHADDA) and len(x) == 2]
        if not idx:
            return None
        i = rng.choice(idx)
        t[i] = t[i][0]
    elif typ == "inserted_syllable":
        idx = [p for p in range(1, len(t)) if t[p] not in ("a", "i", "u", "aa", "ii", "uu")]
        if not idx:
            return None
        p = rng.choice(idx)
        c = rng.choice([c for c in CONS if c not in (degem(t[p - 1]), degem(t[p]))])
        t[p:p] = [c, rng.choice(("a", "i", "u"))]
    return t


def run_ayah(job):
    key, fmt, n_cases, seed, readback = job
    import quran_p2g as p2g
    s, a = key
    q = p2g.quran()
    ay = q.ayah(s, a)
    cw = canonical_words(q, s, a, fmt)
    counts = collections.Counter(w[2] for w in cw)
    single = [k for k, w in enumerate(cw) if counts[w[2]] == 1]
    types = [t for t in TYPES_G if not (fmt == "tj3" and t == "shadda_drop")]
    rng = random.Random(f"{seed}:{s}:{a}:{fmt}")
    out = []
    for c in range(n_cases):
        typ = types[(s * 1000 + a + c) % len(types)]
        case = None
        for _ in range(30):
            if not single:
                break
            k = rng.choice(single)
            toks, legal, ti = cw[k]
            if typ == "word_skip":
                new = []
            elif typ == "word_repeat":
                new = toks + toks
            else:
                new = plant(rng, typ, toks)
                if new is None or tuple(new) in legal:
                    continue
            case = (k, ti, new)
            break
        if case is None:
            out.append({"ayah": f"{s}:{a}", "type": typ, "skipped": True})
            continue
        k, ti, new = case
        hyp = [t for j, w in enumerate(cw) for t in (new if j == k else w[0])]
        r = p2g.convert(hyp, (s, a), fmt)
        words = r.words
        flagged = {w.index for w in words if w.status != "match"}
        ins = r.insertions
        if typ == "word_repeat":
            exact = not flagged and len(ins) == 1 and ins[0].kind == "repeat" and ins[0].of == f"{s}:{a}:{ti}"
            truth_units, pred_units = {"ins"}, set(flagged) | ({"ins"} if ins else set())
        else:
            want = "deleted" if typ == "word_skip" else "deviant"
            exact = flagged == {ti} and not ins and words[ti].status == want
            truth_units, pred_units = {ti}, set(flagged) | ({"ins"} if ins else set())
        rec = {"ayah": f"{s}:{a}", "type": typ, "word": ti, "exact": exact,
               "tp": len(truth_units & pred_units), "fp": len(pred_units - truth_units),
               "fn": len(truth_units - pred_units)}
        if not exact:
            rec["got"] = [[w.index, w.status, " ".join(w.phones)] for w in words if w.status != "match"] + \
                         [["ins", x.kind, " ".join(x.phones)] for x in ins]
            rec["planted"] = " ".join(new)
        if readback and typ not in ("word_skip", "word_repeat") and words[ti].status == "deviant":
            rec["readback"] = _readback(s, a, ay, ti, words[ti].spelling.diacritized, new, fmt)
            rec["spelled"] = words[ti].spelling.diacritized
        out.append(rec)
    return out


_TB = None


def _readback(s, a, ay, ti, spelled, planted, fmt):
    """quran-g2p reads the ayah with the spelled word in place; that word's tokens vs the planted tokens."""
    global _TB
    try:
        from quran_g2p import tokenlayer_tj3 as T3
        from quran_g2p.phonemize import phonemize
        from quran_g2p.textbank import AyahRef
        words = list(ay.uthmani)
        words[ti] = spelled
        res = phonemize(" ".join(words), edition="tanzil", ref=AyahRef(s, a))
        ph = [p for seg in res.segments for p in seg.phones if p.word_index == ti]
        g = []
        for p in ph:
            n0 = len(g)
            T3._emit(g, p.base.value)
            if p.kind == "consonant" and p.geminated and len(g) > n0 and g[-1] in T3.CONSONANTS:
                g[-1] += SHADDA
        if fmt == "tj3":
            g = [degem(x) for x in g]
        return "equal" if g == list(planted) else "differs:" + " ".join(g)
    except Exception as e:                       # the engine refuses the spelling (fail-closed decode)
        return "engine_refused:" + type(e).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", default="tj3g", choices=["tj3g", "tj3"])
    ap.add_argument("--per-ayah", type=int, default=4)
    ap.add_argument("--seed", default="g2-v1")
    ap.add_argument("--g2p", default="C:/Users/Anon/research/quran-g2p")
    ap.add_argument("--no-readback", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="gates/g2.json")
    args = ap.parse_args()
    _setup(args.g2p)
    import quran_p2g as p2g
    keys = sorted(p2g.quran().keys())
    if args.limit:
        keys = keys[:args.limit]
    jobs = [(k, args.fmt, args.per_ayah, args.seed, not args.no_readback) for k in keys]
    t0 = time.time()
    with Pool(args.workers, initializer=_setup, initargs=(args.g2p,)) as pool:
        recs = [r for rs in pool.imap(run_ayah, jobs, chunksize=16) for r in rs]
    done = [r for r in recs if not r.get("skipped")]
    by = collections.defaultdict(collections.Counter)
    for r in done:
        for t in (r["type"], "ALL"):
            c = by[t]
            c["cases"] += 1
            c["exact"] += r["exact"]
            c["tp"] += r["tp"]
            c["fp"] += r["fp"]
            c["fn"] += r["fn"]
            if "readback" in r:
                c["readback_cases"] += 1
                c["readback_equal"] += r["readback"] == "equal"
                c["readback_refused"] += r["readback"].startswith("engine_refused")
    table = {}
    for t, c in by.items():
        table[t] = {"cases": c["cases"], "exact_pct": round(100 * c["exact"] / c["cases"], 2),
                    "precision": round(100 * c["tp"] / max(1, c["tp"] + c["fp"]), 2),
                    "recall": round(100 * c["tp"] / max(1, c["tp"] + c["fn"]), 2)}
        if c["readback_cases"]:
            table[t].update({"readback_cases": c["readback_cases"],
                             "readback_equal_pct": round(100 * c["readback_equal"] / c["readback_cases"], 2),
                             "readback_engine_refused": c["readback_refused"]})
    summ = {"fmt": args.fmt, "planted_cases": len(done), "skipped_no_site": len(recs) - len(done),
            "seconds": round(time.time() - t0, 1), "by_type": table,
            "misses": [r for r in done if not r["exact"]][:300],
            "readback_differs": [r for r in done if r.get("readback", "equal") != "equal"][:300]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summ, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("misses", "readback_differs")}, ensure_ascii=False,
                     indent=1))


if __name__ == "__main__":
    main()
