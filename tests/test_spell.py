import pytest
from conftest import nfc

from quran_p2g.spell import spell, spell_core, spell_words

SH = "ّ"


@pytest.mark.parametrize("toks,want", [
    ("ك a ت a ب a", "كَتَبَ"),
    ("ب i س م i", "بِسْمِ"),
    ("ق aa ل a", "قَالَ"),
    ("ف ii ه i", "فِيهِ"),
    ("ي a ق uu ل u", "يَقُولُ"),
    ("ء i ذ aa", "إِذَا"),
    ("ء u م م u", "أُمُّ"),                      # the same consonant twice: held
    ("ي a س ء a ل uu ن a", "يَسْأَلُونَ"),
    ("م u ء م i ن uu ن", "مُؤْمِنُونْ"),
    ("ب i ء ر u ں", "بِئْرُن"),                  # ikhfa nasal: bare noon
    ("س a م aa ء", "سَمَاءْ"),
    ("ء aa م a ن uu", "ءَامَنُو"),
    ("ل ك i ت aa ب", "ٱلْكِتَابْ"),             # starts with a cluster: hamzat al-wasl
    ("a ل aa", "اَلَا"),                        # a vowel with no consonant rides a bare alef
])
def test_spell_by_sound(toks, want):
    assert nfc(spell(toks).diacritized) == nfc(want)


def test_plain_form():
    s = spell("ء aa م a ن uu")
    assert s.plain == "آمنو"
    assert spell("ل ك i ت aa ب").plain == "الكتاب"


def test_held_tokens_tj3g():
    assert nfc(spell("ر" + SH + " a ب" + SH + " i", "tj3g").diacritized) == nfc("رَّبِّ")


@pytest.mark.parametrize("toks,ref,want", [
    ("ر a ح م aa ن i", "ٱلرَّحْمَٰنِ", "ٱلرَّحْمَانِ"),          # wasl: the sun letter carries the article
    ("ء a ر a ح م aa ن i", "ٱلرَّحْمَٰنِ", "ٱلرَّحْمَانِ"),       # ibtida
    ("ء a ل ح a م د u", "ٱلْحَمْدُ", "ٱلْحَمْدُ"),             # ibtida, qamari
    ("ل ح a م د i", "ٱلْحَمْدُ", "ٱلْحَمْدِ"),                 # wasl, qamari
    ("ء i ه د i ن aa", "ٱهْدِنَا", "ٱهْدِنَا"),                 # ibtida of a hamzat-al-wasl verb
    ("ع a ل ii م a ن", "عَلِيمًا", "عَلِيمًا"),                  # tanwin read in wasl
    ("ع a ل ii م aa", "عَلِيمًا", "عَلِيمًا"),                   # fathatan at waqf
    ("ع a ل aa", "عَلَىٰ", "عَلَى"),                           # alef maqsura
    ("ر a ح م a ت u ں", "رَحْمَةٌ", "رَحْمَةٌ"),                # teh marbuta + tanwin
])
def test_reference_hints(toks, ref, want):
    assert nfc(spell(toks, ref_word=ref).diacritized) == nfc(want)


def test_hint_needs_agreeing_phones():
    # the article's hamza is read with fatha; another vowel is what was said, spelled with its hamza
    assert spell("ء i ل ح a م د u", ref_word="ٱلْحَمْدُ").diacritized.startswith("إِ")
    # tj3g: a sun letter said plain is not the article
    assert spell("ر a ح ii م", "tj3g", ref_word="ٱلرَّحِيمِ").diacritized == "رَحِيمْ"


def test_spell_words_and_empty():
    out = spell_words("ك a ت a ب a | ب i")
    assert [s.diacritized for s in out] == ["كَتَبَ", "بِ"]
    assert spell_core([]).diacritized == ""


def test_spell_units_and_muaalem():
    assert spell("كَ تَ بَ", "units").diacritized == "كَتَبَ"
    assert spell("كَتَبَ", "muaalem").diacritized == "كَتَبَ"
