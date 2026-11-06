"""quran-p2g command line.

  quran-p2g convert hyps.jsonl --format tj3 --out text.jsonl [--field tj3] [--ayah-field ayah] [--compact]
  quran-p2g wer text.jsonl [--diacritics] [--field text_imlaei]      corpus WER of converted (or any text) rows
  quran-p2g wer --text "..." --ayah 1:1                              WER of one text
  quran-p2g spell "<tokens>" [--format tj3] [--ref WORD]
  quran-p2g identify "<tokens>" [--format tj3] [-k 5]
  quran-p2g text "<tokens>" --ayah 1:1 [--format tj3]                one conversion, printed

Input rows (JSONL): {"id": ..., "<field>": tokens, "ayah": "s:a" or "s:a-b" (optional: identified when missing)}.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys

from . import FORMATS, convert, corpus_wer, identify, spell, wer
from .wer import WerResult


def _open(path, mode="rt"):
    if path == "-":
        return sys.stdin if "r" in mode else sys.stdout
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode.replace("t", ""), encoding="utf-8")


def _rows(path):
    with _open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def cmd_convert(a):
    out = _open(a.out, "wt") if a.out else sys.stdout
    n = bad = 0
    for row in _rows(a.input):
        field = a.field or next((k for k in (a.format, "tokens", "phones", "hyp") if k in row), None)
        rec = {"id": row.get("id")}
        try:
            if field is None or field not in row:
                raise ValueError(f"no token field (tried {a.field or a.format}, tokens, phones, hyp)")
            r = convert(row[field], row.get(a.ayah_field) or None, a.format, free_edges=a.free_edges)
            rec.update(r.to_dict(words=not a.compact))
        except Exception as e:                                # one bad row must not stop a corpus
            rec["error"] = f"{type(e).__name__}: {e}"
            bad += 1
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    if a.out:
        out.close()
    print(f"converted {n - bad} / {n} rows" + (f", {bad} errors" if bad else ""), file=sys.stderr)
    return 0 if not bad else 1


def cmd_wer(a):
    if a.text is not None:
        if not a.ayah:
            print("--ayah is required with --text", file=sys.stderr)
            return 2
        r = wer(a.text, a.ayah, diacritics=a.diacritics)
        print(json.dumps({"errors": r.errors, "ref_words": r.ref_words, "wer": round(100 * r.wer, 2)}))
        return 0
    tot, n, skipped = WerResult(0, 0, 0), 0, 0
    field = a.field or ("text" if a.diacritics else "text_imlaei")
    for row in _rows(a.input):
        ay = row.get(a.ayah_field)
        if not ay or field not in row:
            skipped += 1
            continue
        tot = corpus_wer([tot, wer(row[field], ay, diacritics=a.diacritics)])
        n += 1
    print(json.dumps({"rows": n, "skipped": skipped, "field": field, "diacritics": a.diacritics,
                      "errors": tot.errors, "ref_words": tot.ref_words, "wer": round(100 * tot.wer, 2)}))
    return 0


def cmd_spell(a):
    s = spell(a.tokens, a.format, a.ref)
    print(json.dumps({"diacritized": s.diacritized, "plain": s.plain}, ensure_ascii=False))
    return 0


def cmd_identify(a):
    for c in identify(a.tokens, a.format, k=a.k):
        print(json.dumps({"ayah": c.ref, "score": round(c.score, 4), "margin": round(c.margin, 4),
                          "identical": c.identical}, ensure_ascii=False))
    return 0


def cmd_text(a):
    r = convert(a.tokens, a.ayah, a.format)
    if a.json:
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=1))
    else:
        print(r.text)
        print(r.text_imlaei)
        for w in r.items:
            d = w.to_dict()
            if d["status"] != "match":
                print(json.dumps(d, ensure_ascii=False))
    return 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="quran-p2g", description="Phonemes in, Quranic text out.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("convert", help="convert a JSONL of phone hypotheses to text")
    p.add_argument("input")
    p.add_argument("--format", default="tj3", choices=FORMATS)
    p.add_argument("--out")
    p.add_argument("--field", help="token field (default: the format name, tokens, phones or hyp)")
    p.add_argument("--ayah-field", default="ayah")
    p.add_argument("--free-edges", action="store_true", help="material before / after the ayah is not an error")
    p.add_argument("--compact", action="store_true", help="omit the per-word items")
    p.set_defaults(fn=cmd_convert)
    p = sub.add_parser("wer", help="WER against the mushaf (board normalisation)")
    p.add_argument("input", nargs="?")
    p.add_argument("--text")
    p.add_argument("--ayah")
    p.add_argument("--diacritics", action="store_true")
    p.add_argument("--field")
    p.add_argument("--ayah-field", default="ayah")
    p.set_defaults(fn=cmd_wer)
    p = sub.add_parser("spell", help="spell phones by sound")
    p.add_argument("tokens")
    p.add_argument("--format", default="tj3", choices=FORMATS)
    p.add_argument("--ref", help="the reference word (orthographic hints only)")
    p.set_defaults(fn=cmd_spell)
    p = sub.add_parser("identify", help="which ayah was recited")
    p.add_argument("tokens")
    p.add_argument("--format", default="tj3", choices=FORMATS)
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(fn=cmd_identify)
    p = sub.add_parser("text", help="convert one hypothesis and print the text")
    p.add_argument("tokens")
    p.add_argument("--ayah")
    p.add_argument("--format", default="tj3", choices=FORMATS)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_text)
    a = ap.parse_args(argv)
    if a.cmd == "wer" and a.text is None and not a.input:
        ap.error("wer needs an input file or --text")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
