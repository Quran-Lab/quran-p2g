"""G5 no snapping: does quran-p2g ever write the expected word for phones that differ from it in a way the text can
show? And does a phoneme model pay the same WER as a text model for the same mistake?

    PYTHONPATH=src python -X utf8 tools/gate_g5.py --out gates/g5.json

Every ayah's canonical reading (ibtida at the start, waqf with sukun at the end, wasl inside, tj3 tokens) gets single
edits, seed fixed, and is converted WITH NO AYAH GIVEN (as a leaderboard runs it); WER is the board's undiacritized
WER against the mushaf (p2g.wer).

Part A, one sound changed (three cases per ayah):
  letter_swap    a consonant replaced by any other consonant
  letter_drop    a consonant not said
  letter_add     a consonant said before one of the word's consonants
A case is "counted" when the WER charges at least one error. Every case that is not counted is classified by why the
written text cannot show it (no eyeballing):
  tanween        the sound is a tanween's noon (or its iqlab / ikhfa realisation): tanween is a mark, not a letter
  shadda         the added consonant doubles its neighbour: a held consonant is a shadda, not a letter
  ta_marbuta     ha / ta as the last consonant of a word written with ta marbuta (h at a stop, t in wasl; written ة)
  seat_fold      ya / waw / hamza next to a hamza seat, or ya after alef maqsura: the board's normalisation folds the
                 seats (alef, ya, waw) and alef maqsura into ya
  written_noon   an added noon or meem where a noon sakin or tanween is written, in the word or ending the word before
                 it (the noon said openly: same letters)
  OTHER          anything else: a real snap. The gate passes only if OTHER is 0.

Part B, one word changed (four cases per ayah), the same edit made to the text a text model would write and to the
phones a phoneme model would write; both go through the same WER. The gap is the bias of the phones-to-text step:
  word_swap      a word replaced by a word of another ayah
  word_insert    a word of another ayah inserted inside the ayah
  word_drop      a word not said
  word_repeat    a word said twice
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
SHADDA = "ّ"
IKHFA = "ں"
CONS = list("بتثجحخدذرزسشصضطظعغفقكلمنهوي")
TANWEEN = ("ً", "ٌ", "ٍ")
MARKS = set(chr(c) for c in range(0x064B, 0x0653)) | {"ٰ"} | set(chr(c) for c in range(0x06D6, 0x06EE))
P = None


def _init():
    global P
    import quran_p2g
    P = quran_p2g


def degem(t):
    return t[0] if t.endswith(SHADDA) and len(t) == 2 else t


def words_of(s, a):
    """canonical tj3 tokens per Uthmani word, the Uthmani word and its imla'i text"""
    q = P.quran()
    ay = q.ayah(s, a)
    n = len(ay.words)
    by = collections.OrderedDict()
    for k, w in enumerate(ay.words):
        L = "ibtida" if k == 0 else "wasl"
        R = "waqf" if k == n - 1 else "wasl"
        mode = "sukun" if R == "waqf" else "plain"
        f = next(f for f in w.forms if f.left == L and f.right == R and f.mode == mode and f.wajh is None)
        by.setdefault(w.text_index, []).extend(degem(t) for t in f.tj3g.split())
    canon = P.convert(" ".join(t for v in by.values() for t in v), f"{s}:{a}", "tj3")
    iml = {it.index: it.text_imlaei for it in canon.items if hasattr(it, "index")}
    return [(toks, ay.uthmani[ti], iml.get(ti, "")) for ti, toks in by.items()]


def plain(w):
    return "".join(c for c in w if c not in MARKS)


def has_tanween(w):
    return any(c in w for c in TANWEEN)


def classify(kind, word, toks, k, new, prev_tok, next_tok, prev_word=""):
    tok = toks[k]
    last = k == len(toks) - 1
    last_cons = max([j for j, t in enumerate(toks) if t not in ("a", "i", "u", "aa", "ii", "uu")], default=-1)
    if kind in ("letter_drop", "letter_swap") and has_tanween(word) and tok in ("ن", "م", IKHFA) and \
            (last or (k == len(toks) - 2 and toks[-1] in ("a", "i", "u"))):
        if kind == "letter_drop" or new in ("ن", "م", IKHFA):
            return "tanween"
    if kind == "letter_add" and new in (degem(prev_tok or ""), degem(tok)):
        return "shadda"
    if kind == "letter_swap" and k == last_cons and {tok, new} == {"ه", "ت"} and plain(word).endswith("ة"):
        return "ta_marbuta"
    if (tok in ("ي", "و", "ء") or new in ("ي", "و", "ء")) and any(c in word for c in "ئؤءأإآٔ"):
        return "seat_fold"
    if kind == "letter_add" and new == "ي" and k == 0 and plain(prev_word).endswith("ى"):
        return "seat_fold"
    if kind == "letter_add" and new == "ن" and (has_tanween(word) or "ن" in plain(word)):
        return "written_noon"
    if kind == "letter_add" and new in ("ن", "م") and k == 0 and             (has_tanween(prev_word) or plain(prev_word).rstrip("اى").endswith("ن")):
        return "written_noon"
    return "OTHER"


def run_ayah(job):
    (s, a), seed, pool_words = job
    rng = random.Random(f"{seed}:{s}:{a}")
    ws = words_of(s, a)
    key = f"{s}:{a}"
    out = {"A": [], "B": []}
    flat = lambda W: " ".join(t for w in W for t in w)
    base = [list(w[0]) for w in ws]
    # Part A
    for kind in ("letter_swap", "letter_drop", "letter_add"):
        cand = [(wi, k) for wi, w in enumerate(base) for k, t in enumerate(w) if t in CONS]
        if not cand:
            continue
        wi, k = rng.choice(cand)
        W = [list(w) for w in base]
        new = None
        if kind == "letter_swap":
            new = rng.choice([c for c in CONS if c != W[wi][k]])
            W[wi][k] = new
        elif kind == "letter_drop":
            del W[wi][k]
        else:
            new = rng.choice(CONS)
            W[wi].insert(k, new)
        try:
            e = P.wer(P.convert(flat(W), None, "tj3"), key).errors
        except Exception:
            e = 1
        rec = {"kind": kind, "counted": e > 0}
        if e == 0:
            toks = base[wi]
            prev_tok = toks[k - 1] if k > 0 else (base[wi - 1][-1] if wi > 0 and base[wi - 1] else None)
            next_tok = toks[k + 1] if k + 1 < len(toks) else None
            rec["why"] = classify(kind, ws[wi][1], toks, k, new, prev_tok, next_tok, ws[wi - 1][1] if wi > 0 else "")
            if rec["why"] == "OTHER":
                rec["case"] = [key, ws[wi][1], " ".join(toks), kind, new]
        out["A"].append(rec)
    # Part B
    n = len(ws)
    for kind in ("word_swap", "word_insert", "word_drop", "word_repeat"):
        if n < 3:
            continue
        i = rng.randrange(1, n - 1)
        fp, ft = rng.choice(pool_words)
        W = [list(w) for w in base]
        T = [w[2] for w in ws]
        if kind == "word_swap":
            W[i], T[i] = list(fp), ft
        elif kind == "word_insert":
            W.insert(i, list(fp)); T.insert(i, ft)
        elif kind == "word_drop":
            del W[i]; del T[i]
        else:
            W.insert(i + 1, list(W[i])); T.insert(i + 1, T[i])
        ref_words = P.wer(" ".join(x for x in T if x), key).ref_words
        et = P.wer(" ".join(x for x in T if x), key).errors
        try:
            ep = P.wer(P.convert(flat(W), None, "tj3"), key).errors
        except Exception:
            ep = ref_words
        out["B"].append({"kind": kind, "text": et, "phones": ep, "ref": ref_words})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="g5-v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="gates/g5.json")
    args = ap.parse_args()
    _init()
    keys = sorted(P.quran().keys())
    if args.limit:
        keys = keys[:args.limit]
    rng = random.Random(args.seed)
    pool_words = []
    for s, a in rng.sample(keys, min(400, len(keys))):           # foreign words: wasl phones + imla'i text
        for toks, _, iml in words_of(s, a)[1:-1]:
            if iml and len(toks) >= 3:
                pool_words.append((tuple(toks), iml))
    t0 = time.time()
    with Pool(args.workers, initializer=_init) as pool:
        recs = list(pool.imap(run_ayah, [(k, args.seed, pool_words) for k in keys], chunksize=16))
    A = collections.defaultdict(collections.Counter)
    other = []
    for r in recs:
        for x in r["A"]:
            c = A[x["kind"]]
            c["cases"] += 1
            c["counted"] += x["counted"]
            if not x["counted"]:
                c["not_counted:" + x["why"]] += 1
                if x["why"] == "OTHER":
                    other.append(x["case"])
    B = collections.defaultdict(collections.Counter)
    for r in recs:
        for x in r["B"]:
            c = B[x["kind"]]
            c["cases"] += 1
            c["text_err"] += x["text"]
            c["phone_err"] += x["phones"]
            c["ref"] += x["ref"]
    res = {"ayat": len(keys), "seconds": round(time.time() - t0, 1),
           "A_one_sound": {k: dict(v) for k, v in A.items()},
           "A_other_cases": other,
           "B_one_word": {k: {"cases": v["cases"], "text_wer": round(100 * v["text_err"] / v["ref"], 2),
                              "phones_wer": round(100 * v["phone_err"] / v["ref"], 2)} for k, v in B.items()},
           "pass": len(other) == 0}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "A_other_cases"}, ensure_ascii=False, indent=1))
    print("OTHER", len(other), other[:15])


if __name__ == "__main__":
    main()
