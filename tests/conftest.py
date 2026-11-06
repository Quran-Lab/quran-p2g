import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SHADDA = "ّ"


def degem(toks):
    return [t[0] if len(t) == 2 and t[1] == SHADDA else t for t in toks]


def nfc(s):
    import unicodedata
    return unicodedata.normalize("NFC", s)
