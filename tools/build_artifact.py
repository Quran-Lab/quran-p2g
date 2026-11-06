"""Build src/quran_p2g/data/quran_lattice.json.gz from quran-g2p.

    PYTHONPATH=<quran-g2p>/src python -X utf8 tools/build_artifact.py --g2p <quran-g2p checkout>

For every ayah N the word lattice is built over the window [N-1, N, N+1] (the basmala stands in for N-1 before
ayah 1 of every surah except 1 and 9) with only N's words labelled, so N's first word also gets its forms in wasl
from the previous ayah (or the basmala) and N's last word its forms in wasl into the next ayah. A word's phones
depend only on its two boundary states (quran-g2p lattice.py, stop locality), so the window is enough.

Per form we keep the boundary states, the waqf mode, the wajh tag at a khilaf site, the tj3g tokens (tj3 plus the
held consonant, from Phone.geminated) and the quran-g2p unit tokens (tokenlayer.phones_to_tokens).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import unicodedata
from multiprocessing import Pool

SHADDA = "\u0651"
MODE = {"plain": "p", "sukun": "s", "rawm": "r", "ishmam": "m"}


def tj3g_of(phones):
    from quran_g2p import tokenlayer_tj3 as T3
    out = []
    for p in phones:
        n0 = len(out)
        T3._emit(out, p.base.value)
        if p.kind == "consonant" and p.geminated and len(out) > n0 and out[-1] in T3.CONSONANTS:
            out[-1] = out[-1] + SHADDA
    return out


def units_of(phones):
    from quran_g2p.tokenlayer import phones_to_tokens
    return [t.text for t in phones_to_tokens(list(phones))]


def form_row(f):
    return ["i" if f.left == "ibtida" else "w", "q" if f.right == "waqf" else "w", MODE[f.mode], f.wajh,
            " ".join(tj3g_of(f.phones)), " ".join(units_of(f.phones))]


_TB = None


def _tb():
    global _TB
    if _TB is None:
        from quran_g2p.textbank import TextBank
        _TB = TextBank.load("tanzil")
    return _TB


def build_ayah(key):
    from quran_g2p.lattice import _word_maps, build_lattice
    from quran_g2p.config import HafsConfig
    from quran_g2p.textbank import AyahRef
    s, a = key
    tb = _tb()
    n_ayat = sum(1 for r in tb.refs() if r.surah == s)
    items, prev_src = [], None
    if a > 1:
        items.append((AyahRef(s, a - 1), tb.ayah(AyahRef(s, a - 1))))
        prev_src = "prev"
    elif s not in (1, 9):
        items.append((AyahRef(1, 1), tb.ayah(AyahRef(1, 1))))
        prev_src = "basmala"
    me = len(items)
    items.append((AyahRef(s, a), tb.ayah(AyahRef(s, a))))
    has_next = a < n_ayat
    if has_next:
        items.append((AyahRef(s, a + 1), tb.ayah(AyahRef(s, a + 1))))
    word_item, text_word = _word_maps(items, "tanzil", HafsConfig())
    ws = [w for w, ii in enumerate(word_item) if ii == me]
    lo, hi = ws[0], ws[-1]
    lat = build_lattice(items, word_range=(lo, hi))
    words = []
    for w in range(lo, hi + 1):
        forms = []
        for (ww, L, R), fs in sorted(lat.forms.items()):
            if ww == w:
                forms.extend(form_row(f) for f in fs)
        for (site, ww, L, R), fs in sorted(lat.site_forms.items()):
            if ww == w:
                forms.extend(form_row(f) for f in fs)
        seen, uniq = set(), []
        for r in forms:
            k = tuple(r[:2]) + (r[2], r[3], r[4], r[5])
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        stop_after = bool(lat.stoppable[w]) if w < lat.n_words - 1 else True
        words.append({"t": text_word[w][1], "s": int(stop_after), "f": uniq})
    sites = {sid: {"knob": v["knob"], "wajh": v["wajh"], "words": [x - lo for x in v["words"]]}
             for sid, v in lat.sites.items()}
    return {"surah": s, "ayah": a, "prev": prev_src, "next": has_next, "words": words, "sites": sites}


# ------------------------------------------------------------------------------------------------ imla'i alignment

def _strip(s):
    s = "".join(c for c in s if not unicodedata.combining(c))
    for c in "".join(chr(x) for x in range(0x06D6, 0x06EE)):
        s = s.replace(c, "")
    tr = {"ٱ": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي", "ـ": "", "ٰ": ""}
    return "".join(tr.get(c, c) for c in s)


def _skel(w):
    return _strip(w).replace("ا", "").replace("ء", "")


def _ed(a, b):
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[-1]


def align_imlaei(uth, iml):
    """Monotone groups [u0, u1, e0, e1] (half-open) of Uthmani / imla'i words, 1:1, 1:2, 2:1, 1:3, 3:1, minimising the
    skeleton edit distance (+1 per 1:2 / 2:1 / 1:3 / 3:1 group, +2 per 2:2)."""
    n, m = len(uth), len(iml)
    INF = 10 ** 9
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0
    shapes = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]
    for i in range(n + 1):
        for j in range(m + 1):
            if D[i][j] >= INF:
                continue
            for di, dj in shapes:
                if i + di > n or j + dj > m:
                    continue
                c = _ed(_skel("".join(uth[i:i + di])), _skel("".join(iml[j:j + dj]))) + {(1, 1): 0, (2, 2): 2}.get((di, dj), 1)
                if D[i][j] + c < D[i + di][j + dj]:
                    D[i + di][j + dj] = D[i][j] + c
                    B[i + di][j + dj] = (i, j)
    groups, i, j = [], n, m
    while i or j:
        pi, pj = B[i][j]
        groups.append([pi, i, pj, j])
        i, j = pi, pj
    return groups[::-1], D[n][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2p", default="C:/Users/Anon/research/quran-g2p")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "src", "quran_p2g", "data"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.g2p, "src"))
    from quran_g2p.textbank import AyahRef
    tb = _tb()
    keys = [(r.surah, r.ayah) for r in tb.refs()]
    if args.limit:
        keys = keys[:args.limit]
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = []
        for k, r in enumerate(pool.imap(build_ayah, keys, chunksize=8)):
            res.append(r)
            if k % 500 == 0:
                print(f"{k}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
    kf = json.loads(open(os.path.join(args.g2p, "data", "kfgqpc-hafsData_v18.json"), encoding="utf-8-sig").read())
    eml = {(int(r["sora"]), int(r["aya_no"])): r["aya_text_emlaey"].split() for r in kf}
    bad = []
    for r in res:
        key = (r["surah"], r["ayah"])
        uth = tb.ayah(AyahRef(*key)).split()
        iml = eml[key]
        groups, cost = align_imlaei(uth, iml)
        r["uth"], r["iml"], r["groups"] = uth, iml, groups
        if any((g[1] - g[0], g[3] - g[2]) != (1, 1) for g in groups):
            bad.append((key, cost, [[" ".join(uth[g[0]:g[1]]), " ".join(iml[g[2]:g[3]])] for g in groups
                                    if (g[1] - g[0], g[3] - g[2]) != (1, 1)]))
        nt = max(w["t"] for w in r["words"]) + 1
        assert nt == len(uth), (key, nt, len(uth))
    commit = subprocess.run(["git", "-C", args.g2p, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    pins = {}
    for fn in ("tanzil-uthmani.txt", "kfgqpc-hafsData_v18.json"):
        pins[fn] = hashlib.sha256(open(os.path.join(args.g2p, "data", fn), "rb").read()).hexdigest()
    basm = next(r for r in res if (r["surah"], r["ayah"]) == (1, 1))
    art = {"format": "quran-p2g.lattice.v1", "quran_g2p_commit": commit, "edition": "tanzil",
           "imlaei_source": "KFGQPC Hafs v18 aya_text_emlaey", "pins": pins,
           "basmala": {"surah": 1, "ayah": 1}, "ayat": res}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "quran_lattice.json.gz")
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(art, f, ensure_ascii=False, separators=(",", ":"))
    n_forms = sum(len(w["f"]) for r in res for w in r["words"])
    n_words = sum(len(w) for w in (r["words"] for r in res))
    report = {"ayat": len(res), "lattice_words": n_words, "forms": n_forms, "seconds": round(time.time() - t0, 1),
              "bytes": os.path.getsize(path), "quran_g2p_commit": commit, "non_1to1_imlaei_ayat": len(bad),
              "imlaei_groups_with_cost": [b for b in bad if b[1] > 2][:50], "imlaei_examples": bad[:12]}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    _ = basm


if __name__ == "__main__":
    main()
