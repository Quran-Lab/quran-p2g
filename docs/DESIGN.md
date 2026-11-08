# Design notes

Decisions taken while building quran-p2g, with the reason for each. The numbers that back them are in
[GATES.md](GATES.md).

## 1. What the converter is allowed to know

A recitation is constrained: the reciter is reading a known ayah, or a known run of ayat. Everything that
can legally be heard for that span is enumerable, and quran-g2p enumerates it: its word lattice
(`quran_g2p.lattice.build_lattice`) holds, for every word, every form under every boundary state

| state | meaning |
|---|---|
| left `ibtida` / `wasl` | a stop before the word, or connected to the previous word |
| right `waqf` / `wasl` | a stop after the word (with its waqf mode: sukun, rawm, ishmam), or connected |
| wajh | the base reading, or the second transmitted wajh at a khilaf site |

A word's phones depend only on its two boundary states (quran-g2p proves this stop locality in its own
tests), so the lattice is per word and any legal reading is one form per word with agreeing states. quran-p2g
aligns phones to that lattice. A word whose phones are one of its legal forms is the mushaf word, exactly;
anything else was said differently and is spelled from its sound.

## 2. The packaged lattice, and why quran-g2p is not a runtime dependency

quran-g2p loads its pinned texts relative to its source tree, so a plain `pip install` of it from git cannot
find its data. quran-p2g therefore ships the lattice as data
(`src/quran_p2g/data/quran_lattice.json.gz`, 3.1 MB) built by `tools/build_artifact.py` from quran-g2p commit
`7eeba12`. The runtime needs numpy and nothing else. quran-g2p is needed only to rebuild the artifact and to
run gates G1 and G2, which generate their inputs with it.

How the artifact is built: for every ayah N the lattice is built over the window (N-1, N, N+1) with only N's
words labelled. The previous ayah (or the basmala before ayah 1 of every surah except 1 and 9) gives N's first
word its forms in wasl, and the next ayah gives N's last word its forms in wasl. A per-surah build would give
the same forms but is quadratic in quran-g2p; the window is enough because of stop locality, and gate G1
checks the result against quran-g2p's own `phonemize` and `phonemize_concat`.

| | |
|---|---|
| ayat | 6,236 |
| lattice words | 77,481 (a muqatta'at word such as الٓمٓ holds one per letter name) |
| forms | 367,715 (tj3g tokens and quran-g2p unit tokens for each) |

## 3. Texts

- **Uthmani**: the Tanzil Uthmani text exactly as quran-g2p loads it (its pinned file, basmala split off the
  first ayah of each surah). This is the text the lattice's word index refers to, so a matched word is
  written exactly as in that text.
- **Imla'i**: KFGQPC Hafs v18 `aya_text_emlaey`, the plain-spelling text that text ASR models are trained to
  write. Its words are aligned to the Tanzil words by a dynamic program over letter skeletons (1:1, 1:2, 2:1,
  1:3, 3:1 and 2:2 groups). 58 ayat have a group that is not 1:1, all of the form أَوَلَمْ -> أو لم or
  هَٰٓأَنتُمْ -> ها أنتم. Both pinned files carry their SHA-256 in the artifact.

## 4. Input formats

| format | what it is | matched as |
|---|---|---|
| `tj3` | quran-g2p's 38-class phone vocabulary: 28 consonants, the ikhfa nasal ں, a i u aa ii uu | tj3 |
| `tj3g` | tj3 plus a held (geminate) token per consonant, quran-g2p `tokenizer_tj3g` | tj3g |
| `units` | quran-g2p unit tokens (`tokenlayer.phones_to_tokens`, consonant + marks + haraka, or a madd letter with its length) | units, with `:LEN` removed |
| `muaalem` | the phonetic script of `quran_transcript` and the obadx muaalem models | mapped to tj3 |

`|` word marks are accepted and ignored: the lattice decides word boundaries, because a geminate across a word
boundary (هُم مُّنذِرٌ) is one tj3 token that belongs to the second word.

Lengths are not judged. Madd counts, ghunna length and gemination in tj3 are timing questions; the unit format's
`:LEN` is dropped before matching.

The muaalem mapping: a run of one consonant is one token (held when the run is 2 or longer, except the ikhfa
nasal, whose run is its ghunna); a run of a madd letter after its short vowel is one long vowel; the qalqalah
and sakt marks carry no token; ٲ (tasheel) maps to hamza and ۾ (meem ikhfa) to meem. Matching is done in tj3
because the script doubles a letter both for shadda and for the ghunna of an idgham with ghunna, which tj3g
keeps apart.

## 5. Alignment

Weighted edit distance from the hypothesis to the best legal path, solved word by word with one numpy row per
form token. Costs, in one unit:

| operation | cost |
|---|---|
| insertion, deletion of a token | 4 |
| substitution: long vs short of one vowel, held vs plain, ں vs ن, same unit base | 2 |
| substitution: consonant for consonant, vowel for vowel | 4 |
| substitution: consonant for vowel | 6 |
| a stop inside the span | 1; and among paths of equal cost, the one with fewer stops wins |
| a word not said at all | 2 x the length of its shortest form + 1 |
| a repeat: a run equal to a legal form of one of the previous 8 words or the next word | 2 x length + 1 |

Why these three. A legal stop must stay almost free (a reader may stop wherever the rules allow), but a stop also
switches the word before it to its waqf form, which can drop a connecting vowel or a final consonant. In 0.1.0 (skip
3 x length, repeat 2 x length + 2) that let a stop plus a word fragment explain a skipped or repeated word as cheaply
or more cheaply than the skip or the repeat itself: يُعَلِّمَانِ مِنْ أَحَدٍ with مِنْ skipped came out as "a stop
after يُعَلِّمَانْ, then مِنْ said as its kasra", and لَكُم مَّا مَا (tj3) as "لَكُمَا, a stop, مَا". G2 found 63
such cases (21 in tj3g, 42 in tj3). With a skip and a repeat at 2 x length + 1, the single event costs at most what
the stop-and-fragment reading costs, and the second criterion (fewer stops) decides the tie. The criterion is lexicographic, not a cost: internally
every cost is 4,096 x its value plus the number of stops, so it never outweighs one unit of cost, and a reading that
stops where the rules allow is still matched exactly (G1).

Options: `open_edges` (default on) lets the first word continue the previous ayah in wasl and the last word run
into the next, because a clip cut at an ayah boundary often is. `basmala` (default on) lets the basmala be read
before ayah 1 of a surah other than 1 and 9; it is reported as an item and left out of the text. `free_edges`
(default off) charges nothing for material before the first and after the last word, for clips that hold part
of a neighbouring ayah.

A khilaf site's alternate wajh is offered per word. The lattice ties a site's words to one choice; the aligner
does not enforce that tie. Only one site spans more than one word (30:54, the three words of ضعف read with
fath or damm of the dad), so a reading that mixes the two wajh inside 30:54 passes as legal there and nowhere
else.

## 6. Word statuses and deviation levels

Per Uthmani word: `match`, `deviant` (phones said, none of them a legal form), `deleted` (no phones). Between
words: `inserted` items (a repeat of a nearby word, the basmala, or edge material with `free_edges`).

A deviant word gets a level, the most visible class of its differences from the legal form it was aligned to:

| level | differences | shown by |
|---|---|---|
| `tajweed` | how a sakin noon, tanwin or meem was realised: ikhfa or iqlab against izhar, a tanwin noon assimilated or not | no text |
| `harakat` | short vowels; a held consonant said plain or the reverse (shadda) | diacritized text only |
| `letters` | everything else: a consonant, a long vowel lengthened or shortened, a syllable added or dropped | any text |

A long vowel counts as `letters` because standard spelling writes it with a letter (alef, waw, yeh), even
where the Uthmani rasm writes it with a small sign.

The two texts follow the levels. `text` (Uthmani, diacritized) keeps the mushaf word for a `tajweed`
deviation and writes the spelling otherwise. `text_imlaei` (undiacritized) keeps the imla'i word up to
`harakat`. That is what a transcriber who writes no harakat would have written for the same audio, so
the undiacritized WER of a phoneme model means the same as the WER of a text model. The statuses themselves
stay phonetic: every deviation is reported, whatever its level.

## 7. Spelling by sound

`quran_p2g.spell` inverts the way quran-g2p reads the script, in Uthmani dabt conventions, so that a spelled
word can be given back to quran-g2p (gate G2 does exactly that):

- consonant + short vowel: the letter and its haraka; consonant + long vowel: haraka and a bare madd letter;
- consonant with no vowel: sukun; the ikhfa nasal: a bare noon (the Uthmani mark of ikhfa);
- a held consonant (tj3g, units, or the same consonant twice in tj3): shadda;
- hamza: seated by its own vowel and the vowel before it (kasra over damma over fatha), bare after a long
  vowel or a sakin letter at the end of a word, and ءَا for hamza + long a as the mushaf writes it;
- a word that starts with a consonant cluster gets hamzat al-wasl (ٱ);
- a vowel with no consonant before it rides a bare alef (the consonant was not said).

In constrained mode the reference word may supply orthographic hints, applied only where the phones agree:
the definite article (ٱل before a held sun letter or a sakin lam, and the article or hamzat al-wasl read in
ibtida, which is dropped back to ٱ only when its vowel is the one the word takes), tanwin, teh marbuta and
alef maqsura. The undiacritized form removes every mark, writes alef for alef wasla and alef madda for ءا.

Three hint rules were tightened in 0.1.1 after the G2 read-back showed spellings that did not read back:

- a final nasal is taken as the tanwin only when the letter before it is the one the reference's tanwin sits on
  (أَثِيمٍ said `ء a ث i م` was spelled أَثٍ, losing the meem), and never when the word was read before a stop,
  where a tanwin is not said as a nasal (مَّكْنُونٌ said with a short u at the end of 37:49 was spelled مَكْنٌ);
- a word that starts with the ikhfa nasal starts with a cluster too and gets hamzat al-wasl (ٱنفَضُّوا۟ said
  with a changed consonant was spelled with a bare noon, which the previous word's tanwin then assimilated);
- the article's lam said before a sun letter in ibtida (not assimilated) is written with a plain hamza, أَلْ:
  quran-g2p reads ٱلْ before a sun letter as hamzat al-wasl with kasra, not as the article.

Some phone strings cannot be written so that a reader of the script says them, because a reading rule applies
whatever is written. The G2 read-back lists them by rule (section 12): a long vowel before a word that starts with a
cluster, a short vowel at the end of a word read before a stop, the madd in the name of Allah, two identical
consonants with the first sakin (idgham mithlayn), and the ikhfa nasal before a letter that takes no ikhfa.

## 8. Retrieval (no ayah given)

Every ayah's canonical path is folded (long to short vowels, ں to ن, held to plain) and indexed by token
4-grams. A hypothesis votes with its 4-grams (rarer grams weigh more), the 12 best-voted ayat are rescored by
edit distance, and a candidate grows into a run of neighbouring ayat while the hypothesis is longer than it.
Score is 1 - distance / max(length). The Quran repeats some ayat word for word (the largest group is the 31
occurrences of فَبِأَىِّ ءَالَآءِ رَبِّكُمَا تُكَذِّبَانِ); such ayat are returned as `identical`, never as competitors, and
the margin is measured to the best candidate whose text differs.

## 9. WER

`quran_p2g.wer` reimplements the Quranic ASR board's normalisation (`strip` in
Quran-Lab/quranic-asr-benchmark score.py: every combining mark and the annotation signs U+06D6..U+06ED removed,
hamza seats, alef forms, alef maqsura and teh marbuta folded). Undiacritized WER is computed against the
imla'i text by default (or the Uthmani text with `script="uthmani"`); diacritized WER against the Uthmani text
with only pause signs and tatweel removed. Given a conversion result, `wer` picks the matching text itself.

The board's own references are in a diacritized simple script, which this package does not ship. After
`strip` the two differ only where KFGQPC's imla'i spelling and the board's simple spelling differ.

## 10. What is out of scope

- Riwayat other than Hafs ʿan ʿAsim (tariq al-Shatibiyyah), which is the scope of quran-g2p.
- Isti'adha, takbir and anything else said outside the mushaf text: reported as insertions or absorbed into
  the edge word.
- A range across two surahs. Ranges stay inside one surah.
- A learned P2G. A learned broad-Arabic phoneme-to-grapheme model exists elsewhere in Quran Lab's work;
  this package does not use or need one: inside the mushaf every word is known, and outside it a rule-based
  spelling is deterministic and auditable.

## 11. Conventions, and how the V4 benchmark scorer differs

Some differences between quran-p2g's word statuses and another tool's are not errors on either side but different
answers to a question of convention. quran-p2g's answers, and the V4 benchmark scorer's (`tj3_word_flags` in
`score_v4_bench.py`):

| question | quran-p2g (the convention of this package) | V4 benchmark scorer |
|---|---|---|
| ayah edges | open: the first word may continue the previous ayah and the last may run into the next, in wasl (a clip cut at an ayah boundary often does) | ibtida at the start, waqf at the end |
| repeated material | an inserted repeat item; the words around it stay a match, so WER counts one insertion | absorbed into a neighbouring word, which is flagged |
| legality | the reading must be one legal path: boundary states agree from word to word | each word against all its forms, whatever the state of its neighbours |
| basmala before ayah 1 | an optional item, left out of the text | not in the lattice |

The legality row is the one where quran-p2g is stricter for a reason: فَإِذَا said with its waqf long vowel and then
ٱنشَقَّتِ in wasl is no legal reading (the long vowel is shortened before the cluster), and quran-p2g flags it. The
scorer passes both words, because each is a legal form on its own.

G3 measures agreement three ways: raw (quran-p2g's defaults), under the scorer's conventions (quran-p2g run with
`open_edges=False`, `basmala=False`, `repeats=False`, a word passing when its phones are any of its legal forms in any
boundary state; `bench_convention` in `tools/gate_g3.py`), and up to ties: a word that still disagrees under the
scorer's conventions is an attribution tie when quran-p2g's segmentation costs exactly the scorer's own optimum under
the scorer's own cost (unit edit distance over its lattice). Every raw disagreement is put in the first convention
that removes it, or in the tie class.

## 12. What G2 counts as exact

A planted edit sometimes has more than one attribution that no phone string can tell apart. Skipping either of the
two words of هَيْهَاتَ هَيْهَاتَ gives the same phones; a consonant and vowel inserted right after the same consonant
and vowel at the start of a word (تَرَ said تَتَرَ after أَلَمْ) is the same string as the syllable added to the end of
the word before. G2 reports two numbers. Strict: quran-p2g reports exactly the planted statuses. Equivalent: it
reports the planted statuses of another attribution, and the same edit type on that other word produces the identical
phone string (`tools/g2_equiv.py` enumerates every such word: every other word skipped or repeated, the inserted pair
at every position of every other word). Nothing else is accepted, and both numbers are published.

The read-back check gives the spelled word back to quran-g2p inside its ayah, with the same neighbours and the same
stops as the planted reading. Where the tokens differ, the difference is accepted only when a named reading rule
forces it whatever is written, checked mechanically:

| rule | the planted tokens | check |
|---|---|---|
| `long_vowel_before_cluster` | a word-final long vowel before a word that starts with a cluster, in wasl | read back short; read again with a stop after the word, long |
| `waqf_final_short_vowel` | a short vowel at the end of the last word of the ayah, read before a stop | read back without it; read again in wasl into the next ayah, with it |
| `allah_madd` | the name of Allah with a short a | read back long, the only difference |
| `idgham_mithlayn` | a sakin consonant before the same consonant | read back as one held consonant (inside the word, or across the boundary) |
| `ikhfa_before_non_ikhfa` | the ikhfa nasal before a letter that takes no ikhfa | the only difference is that nasal |

quran-g2p refuses a non-mushaf spelling at a word its ayah reference pins (the letter name قٓ of 50:1, the tasheel
of ءَا۬عْجَمِىٌّ in 41:44); such a word is read again without the reference and counted separately.

## 0.1.2: one wrong word is one word error

A symmetric test (the same word-level error put into a text model's output and into a phoneme model's phones, both
scored by the same WER) showed that a wrongly read word cost a phoneme model more than a text model: 18.11 % against
14.57 % WER when one word per clip was replaced. The phone alignment could tie "word A read wrongly" with "word A
deleted and its phones absorbed by word B", and the tie-break sometimes took the second, so one wrong word became two
(for example "qadlmi'ad" for "yukhlifu l-mi'ad" with yukhlifu read as "qad"). After the alignment, a re-split now tries
every boundary between two neighbouring words, keeps the boundary states legal (a word in waqf is followed by a word in
ibtida, the outer boundaries unchanged), and takes the split with the fewest words wrong at the letter level. It changes
the alignment only when that number strictly drops, so it cannot hide a mistake. The same test after the fix: 14.75 %
against 14.57 %; inserted, deleted and repeated words within 0.25 points of the text path; one letter swapped in
15.5 % of words reads 15.42 %.

The unvowelled noon / meem swap is tajweed only in the iqlab context (the next sound, in the word or at the start of
the next word, is ba). Anywhere else it is a different word ("ka'annahun" for "ka'annahum") and counts as a letter
mistake.
