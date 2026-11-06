import json

from quran_p2g.cli import main

FATIHA2 = "ء a ل ح a م د u ل i ل aa ه i ر a ب i ل ع aa ل a م ii ن"


def test_convert_and_wer(tmp_path, capsys):
    src = tmp_path / "hyps.jsonl"
    rows = [{"id": "a", "tj3": FATIHA2, "ayah": "1:2"},
            {"id": "b", "tj3": FATIHA2.replace("د u", "د i")},
            {"id": "c", "tj3": "q q", "ayah": "1:2"}]
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "text.jsonl"
    assert main(["convert", str(src), "--format", "tj3", "--out", str(out)]) == 1      # one bad row
    got = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert got[0]["text_imlaei"] == "الحمد لله رب العالمين"
    assert got[1]["ayah"] == "1:2" and got[1]["retrieval"]["ayah"] == "1:2"
    assert "error" in got[2]
    capsys.readouterr()
    assert main(["wer", str(out)]) == 0
    d = json.loads(capsys.readouterr().out)
    assert (d["rows"], d["errors"], d["ref_words"]) == (2, 0, 8)
    assert main(["wer", str(out), "--diacritics"]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == 1


def test_wer_single_text(capsys):
    assert main(["wer", "--text", "بسم الله", "--ayah", "1:1"]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == 2


def test_spell_identify_text(capsys):
    main(["spell", "ك a ت a ب a"])
    assert json.loads(capsys.readouterr().out)["diacritized"] == "كَتَبَ"
    main(["identify", FATIHA2, "-k", "1"])
    assert json.loads(capsys.readouterr().out)["ayah"] == "1:2"
    main(["text", FATIHA2, "--ayah", "1:2"])
    assert capsys.readouterr().out.splitlines()[1] == "الحمد لله رب العالمين"
    main(["text", FATIHA2, "--ayah", "1:2", "--json"])
    assert json.loads(capsys.readouterr().out)["counts"]["match"] == 4
