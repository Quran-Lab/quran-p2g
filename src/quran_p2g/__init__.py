"""quran-p2g: phonemes in, Quranic text out. The inverse companion of quran-g2p.

    import quran_p2g as p2g
    r = p2g.convert("ب i س م i ل aa ه i ...", ayah="1:1", fmt="tj3")
    r.text, r.text_imlaei, r.words
    p2g.spell("ك a t a b a")
    p2g.identify(tokens)
    p2g.wer(r, "1:1")
"""
from .convert import Inserted, P2GResult, WordResult, convert, deviation_level
from .formats import FORMATS, FormatError, tokenize
from .quran import parse_ref, quran
from .retrieve import Candidate, identify
from .spell import Spelling, spell, spell_words
from .wer import WerResult, corpus_wer, strip, strip_alef, wer

__version__ = "0.1.1"
__all__ = ["convert", "spell", "spell_words", "identify", "wer", "corpus_wer", "strip", "strip_alef", "tokenize",
           "quran", "parse_ref", "P2GResult", "WordResult", "Inserted", "Spelling", "Candidate", "WerResult",
           "FORMATS", "FormatError", "deviation_level", "__version__"]
