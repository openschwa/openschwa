# M2 ground-truth audit scripts (Aix-MARSEC)

These scripts produced the label-quality findings behind the
`eval/reports-intonation/aix-intonation-2026-08-20.md` exam report.
All of them run offline against `eval/data/aix/4` (the corpus is not
committed); `mark_acoustics.py` additionally needs the engine venv and the
block wavs (it was run on the exam machine).

- `tsm_census.py` / `full_census.py` - count every non-word character in
  the TextGrid "Text" tier. The full census shows the SEC mark inventory:
  \` grave 8,203, * 6,618, ~ 5,065, \\ backslash 3,108, / 2,783, _ 2,682,
  , 2,570 (stress, not a tone), > 692, ^ 305, < 254 - plus UI boundaries.
- `char_context.py` - where #, , and ' appear: # is a standalone boundary
  interval; , is the SEC stress mark (the Aix `get_str` script confirms
  '\'' and ',' mean "stressed syllable").
- `tsm_intsint_crosscheck.py` - per-mark INTSINT movement; shows no mark is
  cleanly directional at the word level (declination dominates).
- `momel_direction2.py` - Momel-target slope per mark, word-to-unit-end
  window; same conclusion as the engine-based check.
- `shape_analysis.py` - two-window glide shape (word vs tail) per mark on
  unit-final marked words, from Momel targets.
- `mark_overlap.py` - symbol-level confusion between annotators B and G on
  the 98 double-transcribed passages (diagonals only 16-35%).
- `annotator_ceiling4.py` - THE ceiling: on units with identical boundaries,
  B and G agree on the nuclear tone 36.2% overall (fall 62%, fall-rise 14%,
  rise 0/3); restricted to fall/rise/fall-rise: 50.0% (n=96).
- `mark_acoustics.py` - the engine's own F0 shape at all 10,200 unit-final
  marked words; "high fall" and "high rise" marks have identical acoustic
  distributions (33/24 vs 34/24) - the marks are decoupled from the glide.
- `consensus3.py` - corrected B-vs-G agreement with fall-rise counted as a
  disagreement: binary fall/rise agreement **90.3%** (223/247), fall
  reproducibility 81.6% - the binary labels are adequate for the 90% bar.
- `slopes.py` / `windows.py` / `smoothglide.py` / `gapruns.py` - why the
  engine fails despite adequate labels: terminal slopes median -4.3 st/s for
  falls, +0.1 for rises; every window definition gives falls and rises
  identical slope distributions; the short-slice F0 tracks are octave-chaotic
  (median in-run slope -89 st/s for both classes) - the fix is at the
  pitch-extraction level, not the threshold level.

Legend (SEC manual, FIG2, via ICAME's static copy): 14 prosodic characters -
5 tones x high/low {fall, rise, level, fall-rise, rise-fall} + stress circle +
up/down arrows + boundaries. ASCII equivalents in Aix: \\ high fall,
\` low fall, / high rise, \`/ low fall-rise; * ~ _ could not be assigned a
tone identity from the acoustics and are excluded from the exam classes.
