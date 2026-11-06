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

Exact (strict): the output is exactly the planted statuses. Exact up to an equivalent attribution: the output is the
planted statuses of another attribution proven equally valid (tools/g2_equiv.py: the same edit on another word gives the
identical phone string, or the planted attribution and the output cost the same). Both numbers are reported.

Spelling read-back (substitution-type edits): the spelled word replaces the word in the ayah text, quran-g2p reads the
ayah (canonical stops: ibtida, wasl inside, waqf at the end, so the same neighbours and the same boundary states as the
planted reading), and that word's tokens must equal the planted tokens. A difference that a named reading rule of
quran-g2p forces whatever is written is classified with the rule (and, where a stop removes the context, verified by
reading the word again with a stop after it):

  long_vowel_before_cluster  a word-final long vowel is shortened before a word that starts with a cluster (wasl);
                             with a stop after the word, the spelling reads back long
  allah_madd                 the madd of the name of Allah is read long whatever is written
  ikhfa_before_non_ikhfa     the ikhfa nasal exists only before the 15 ikhfa letters; written as a bare noon before any
                             other letter it is read by that letter's noon rule
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
IKHFA = "\u06ba"
IKHFA_LETTERS = set("\u0635\u0630\u062b\u0643\u062c\u0634\u0642\u0633\u062f\u0637\u0632\u0641\u062a\u0636\u0638")
TYPES_G = ["vowel_swap", "consonant_swap", "long_short", "shadda_drop", "word_skip", "word_repeat", "inserted_syllable"]
CONS = list("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي")
G2P = None


def _setup(g2p):
    global G2P
    G2P = g2p
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    sys.path.insert(0, HERE)
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
    from quran_p2g.convert import _prep
    import g2_equiv as EQ
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
        exact = EQ.statuses_match(r, typ, ti, f"{s}:{a}")
        truth_units = {"ins"} if typ == "word_repeat" else {ti}
        pred_units = set(flagged) | ({"ins"} if ins else set())
        rec = {"ayah": f"{s}:{a}", "type": typ, "word": ti, "exact": exact, "equiv": None,
               "tp": len(truth_units & pred_units), "fp": len(pred_units - truth_units),
               "fn": len(truth_units - pred_units)}
        if not exact:
            alts = EQ.same_string_alternatives(cw, k, typ, new)
            hit = [j for j in alts if EQ.statuses_match(r, typ, j, f"{s}:{a}")]
            toks_k, states_k, _ = _prep(ay.words[k], fmt)
            c_planted = EQ.planted_cost(toks_k, states_k, k, len(cw), typ, cw[k][0], new, fmt)
            if hit:
                rec["equiv"] = "same_string"
                truth_units = {"ins"} if typ == "word_repeat" else {hit[0]}
            elif c_planted == r.cost:
                rec["equiv"] = "equal_cost"
                truth_units = pred_units
            rec["got"] = [[w.index, w.status, " ".join(w.phones)] for w in words if w.status != "match"] + \
                         [["ins", x.kind, " ".join(x.phones), x.of] for x in ins]
            rec["planted"] = " ".join(new)
            rec["same_string_words"] = alts
            rec["cost_planted"], rec["cost_output"] = c_planted, r.cost
        # precision / recall up to the equivalence: against the units of the attribution the output is equal to
        rec["tp_eq"] = len(truth_units & pred_units)
        rec["fp_eq"] = len(pred_units - truth_units)
        rec["fn_eq"] = len(truth_units - pred_units)
        if readback and typ not in ("word_skip", "word_repeat") and words[ti].status == "deviant":
            rb, got_rb = _readback(q, s, a, ay, ti, words[ti].spelling.diacritized, new, fmt)
            rec["readback"] = rb
            rec["spelled"] = words[ti].spelling.diacritized
            if rb != "equal":
                rec["readback_got"] = " ".join(got_rb)
                rec["planted"] = " ".join(new)
        out.append(rec)
    return out


def _read(text, s, a, ref=True, stops=None):
    """quran-g2p reads `text` (the ayah s:a, or any text without the reference) -> {lattice word: tj3g tokens}."""
    from quran_g2p import tokenlayer_tj3 as T3
    from quran_g2p.phonemize import phonemize
    from quran_g2p.textbank import AyahRef
    from quran_g2p.waqf import WaqfSpec
    kw = {"waqf": WaqfSpec(tuple(stops))} if stops else {}
    res = phonemize(text, edition="tanzil", ref=AyahRef(s, a) if ref else None, **kw)
    out = collections.defaultdict(list)
    for seg in res.segments:
        for p in seg.phones:
            g = out[p.word_index]
            n0 = len(g)
            T3._emit(g, p.base.value)
            if p.kind == "consonant" and p.geminated and len(g) > n0 and g[-1] in T3.CONSONANTS:
                g[-1] += SHADDA
    return out


def _fmt(toks, fmt):
    return [degem(x) for x in toks] if fmt == "tj3" else list(toks)


def _canon_form(lw, k, n):
    L = "ibtida" if k == 0 else "wasl"
    R = "waqf" if k == n - 1 else "wasl"
    mode = "sukun" if R == "waqf" else "plain"
    return next(f for f in lw.forms if f.left == L and f.right == R and f.mode == mode and f.wajh is None).tj3g.split()


def _readback(q, s, a, ay, ti, spelled, planted, fmt):
    """quran-g2p reads the ayah with the spelled word in place; that word's tokens vs the planted tokens.

    -> (status, tokens read back). status: "equal"; "rule:<name>" (a named reading rule of quran-g2p forces the
    difference whatever is written; verified by a second reading without the context where one exists);
    "equal_without_reference" (quran-g2p refuses the spelling at a word its ayah reference pins, a muqatta'at letter
    name or the tasheel of 41:44, and reads it as planted without the reference); "differs"; "engine_refused:<error>".
    """
    planted = list(planted)
    words = list(ay.uthmani)
    words[ti] = spelled
    text = " ".join(words)
    lat = sorted(lw.index for lw in ay.words if lw.text_index == ti)
    n = len(ay.words)
    pick = lambda m: [t for i in lat for t in m.get(i, [])]
    try:
        full = _read(text, s, a)
    except Exception as e:                       # the engine refuses the spelling (fail-closed decode)
        try:
            g = _fmt(pick(_read(text, s, a, ref=False)), fmt)
        except Exception:
            return "engine_refused:" + type(e).__name__, []
        return ("equal_without_reference" if g == planted else "engine_refused:" + type(e).__name__), g
    G = pick(full)
    g = _fmt(G, fmt)
    if g == planted:
        return "equal", g
    # long_vowel_before_cluster: only the final long vowel is shortened, and with a stop after the word it is long
    if planted[-1] in ("aa", "ii", "uu") and g == planted[:-1] + [planted[-1][0]] and lat[-1] < n - 1:
        if _fmt(pick(_read(text, s, a, stops=[lat[-1]])), fmt) == planted:
            return "rule:long_vowel_before_cluster", g
    # waqf_final_short_vowel: the last word of the ayah (read before a stop) ends in a short vowel in the planted
    # reading; read in wasl into the next ayah of the mushaf, the spelling gives the planted tokens
    after = (s, a + 1) if (s, a + 1) in q._raw else (s + 1, 1)
    if lat[-1] == n - 1 and planted[-1] in ("a", "i", "u") and after in q._raw:
        nxt = " ".join(q.ayah(*after).uthmani)
        try:
            gw = _fmt(pick(_read(text + " " + nxt, s, a, ref=False)), fmt)
        except Exception:
            gw = None
        if gw == planted:
            return "rule:waqf_final_short_vowel", g
    # idgham_mithlayn: a sakin consonant before the same consonant merges with it into one held consonant (inside the
    # word, or across the boundary, where the held consonant belongs to the second word)
    merged = lambda x: x if fmt == "tj3" else degem(x) + SHADDA
    for i in range(len(planted) - 1):
        x, y = planted[i], planted[i + 1]
        if x not in ("a", "i", "u", "aa", "ii", "uu", IKHFA) and degem(x) == degem(y) and \
                g == planted[:i] + [merged(x)] + planted[i + 2:]:
            return "rule:idgham_mithlayn", g
    if planted[-1] not in ("a", "i", "u", "aa", "ii", "uu") and g == planted[:-1] and lat[-1] + 1 in full and \
            full[lat[-1] + 1] and full[lat[-1] + 1][0] == degem(planted[-1]) + SHADDA:
        return "rule:idgham_mithlayn", g
    if lat[0] > 0 and G and G[0] == degem(planted[0]) + SHADDA and g[1:] == planted[1:]:
        prev = _canon_form(ay.words[lat[0] - 1], lat[0] - 1, n)
        if degem(prev[-1]) == degem(planted[0]) and full.get(lat[0] - 1) == prev[:-1]:
            return "rule:idgham_mithlayn", g
    if len(g) == len(planted):
        diff = [i for i in range(len(g)) if g[i] != planted[i]]
        plain = "".join(c for c in spelled if "ء" <= c <= "ي")
        if len(diff) == 1:
            i = diff[0]
            # allah_madd: the madd of the name of Allah is read long whatever is written
            if planted[i] == "a" and g[i] == "aa" and 0 < i < len(g) - 1 and degem(g[i - 1]) == "ل" \
                    and g[i + 1] == "ه" and "لله" in plain:
                return "rule:allah_madd", g
            # ikhfa_before_non_ikhfa: the ikhfa nasal exists only before the 15 ikhfa letters
            if planted[i] == IKHFA and (i + 1 >= len(planted) or degem(planted[i + 1]) not in IKHFA_LETTERS):
                return "rule:ikhfa_before_non_ikhfa", g
    return "differs", g


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
            c["exact_eq"] += r["exact"] or bool(r["equiv"])
            if r["equiv"]:
                c["equiv_" + r["equiv"]] += 1
            for x in ("tp", "fp", "fn", "tp_eq", "fp_eq", "fn_eq"):
                c[x] += r[x]
            if "readback" in r:
                c["readback_cases"] += 1
                c["readback_equal"] += r["readback"] == "equal"
                c["readback_refused"] += r["readback"].startswith("engine_refused")
                c["readback_noref"] += r["readback"] == "equal_without_reference"
                if r["readback"].startswith("rule:"):
                    c["readback_" + r["readback"]] += 1
    table = {}
    for t, c in by.items():
        pct = lambda a, b: round(100 * a / max(1, b), 2)
        table[t] = {"cases": c["cases"],
                    "exact_pct": pct(c["exact"], c["cases"]), "exact": c["exact"],
                    "exact_equiv_pct": pct(c["exact_eq"], c["cases"]), "exact_equiv": c["exact_eq"],
                    "equiv_same_string": c["equiv_same_string"], "equiv_equal_cost": c["equiv_equal_cost"],
                    "precision": pct(c["tp"], c["tp"] + c["fp"]), "recall": pct(c["tp"], c["tp"] + c["fn"]),
                    "precision_equiv": pct(c["tp_eq"], c["tp_eq"] + c["fp_eq"]),
                    "recall_equiv": pct(c["tp_eq"], c["tp_eq"] + c["fn_eq"])}
        if c["readback_cases"]:
            rules = {k[len("readback_rule:"):]: v for k, v in c.items() if k.startswith("readback_rule:")}
            n_ok = c["readback_equal"] + c["readback_noref"] + sum(rules.values())
            table[t].update({"readback_cases": c["readback_cases"], "readback_equal": c["readback_equal"],
                             "readback_equal_pct": pct(c["readback_equal"], c["readback_cases"]),
                             "readback_equal_without_reference": c["readback_noref"],
                             "readback_rules": rules,
                             "readback_equal_or_rule": n_ok,
                             "readback_equal_or_rule_pct": pct(n_ok, c["readback_cases"]),
                             "readback_engine_refused": c["readback_refused"]})
    summ = {"fmt": args.fmt, "planted_cases": len(done), "skipped_no_site": len(recs) - len(done),
            "seconds": round(time.time() - t0, 1), "by_type": table,
            "misses": [r for r in done if not r["exact"]],
            "readback_not_equal": [r for r in done if r.get("readback", "equal") != "equal"]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summ, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("misses", "readback_not_equal")}, ensure_ascii=False,
                     indent=1))


if __name__ == "__main__":
    main()
