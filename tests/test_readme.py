"""The README example must show what the code really prints."""
import os

import quran_p2g as p2g

README = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "README.md")


def test_readme_example_outputs_are_real():
    text = open(README, encoding="utf-8").read()
    said = "ء a ل ح a م د i ل i ل aa ه i ر a ب i ل ء aa ل a م ii ن"
    assert f'said = "{said}"' in text
    r = p2g.convert(said, ayah="1:2", fmt="tj3")
    assert f"r.text          # {r.text!r}" in text
    assert f"r.text_imlaei   # {r.text_imlaei!r}" in text
    assert f"# {[(w.status, w.level) for w in r.words]!r}" in text
    w = (p2g.wer(r, "1:2").wer, p2g.wer(r, "1:2", diacritics=True).wer)
    assert f"# {w!r}" in text


def test_readme_cli_spell_example():
    text = open(README, encoding="utf-8").read()
    s = p2g.spell("ي a ق uu ل uu ن a")
    assert f'{{"diacritized": "{s.diacritized}", "plain": "{s.plain}"}}' in text
