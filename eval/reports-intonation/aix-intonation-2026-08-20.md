# M2 intonation exam - Aix-MARSEC (2026-08-20)

Engine nuclear-tone verdicts vs the SEC tonetic-stress-mark labels of
Aix-MARSEC (Hirst, Auran & Bouzon 2002-2004, v2009), block-disjoint split.
Companion data: `aix-intonation-2026-08-20.json` (+ `.jsonl` per-unit records).

## Result

| metric | value | bar |
| --- | --- | --- |
| units run | 7,021 (all 53 blocks, B 3,146 / G 3,875) | - |
| test units | 974 (speaker-disjoint) | - |
| fall-vs-rise accuracy (test) | **0.3778** | >= 0.90 **NOT met** |
| fall accuracy (test) | 0.3881 (n=706) | - |
| rise accuracy (test) | 0.2093 (n=43) | - |
| fall-rise accuracy (test) | 0.1778 (n=225) | - |
| mean octave-error rate (test) | **0.0004** (0.04% of voiced frames) | < 0.02 **met** |
| alignment | ok 5,058 / failed 1,761 / skipped 149 / low-confidence 53 | - |

Full-corpus confusion (all 6,387 units with a detection; expected -> detected):

| expected \ detected | fall | rise | fall-rise | level |
| --- | --- | --- | --- | --- |
| fall (4,533) | 1,667 (37%) | 888 (20%) | 582 (13%) | 1,396 (31%) |
| rise (282) | 79 (28%) | 80 (28%) | 30 (11%) | 93 (33%) |
| fall-rise (1,541) | 519 (34%) | 332 (22%) | 242 (16%) | 448 (29%) |

Predict-fall baseline on fall+rise: **94.1%** (the corpus is 17:1 fall-heavy;
raw accuracy on this set is trivially gamed - the honest read is per-class).
High-confidence (>= 0.9) detections: 2,641 units at 50.9% agreement with the labels.

## The ground truth is the story (corrected)

Before the run, the labels themselves were audited:

- **Multi-class ceiling.** 98 passages were transcribed twice (BJW/GOK).
  On units with *identical boundaries*, the two experts agree on the nuclear
  tone **36.2%** overall - 62% for falls, 14% fall-rise, 0/3 rise; restricted
  to fall/rise/fall-rise: **50.0%** (n=96). Fall-vs-fall-rise is the mushy
  boundary (the SEC manual itself flags the "shallow fall" ambiguity).
- **Binary ceiling (the bar's own question).** On time-matched units the
  annotators agree on fall-vs-rise **90.3%** (223/247 when both commit to
  fall or rise); fall reproducibility is 81.6% (222/272, leaking mostly to
  fall-rise). **The binary labels are adequate for the 90% bar.**
- **Imbalance.** 5,077 fall vs 303 rise units - news reading barely rises,
  so a predict-fall dummy scores 94%; only balanced/per-class numbers are
  meaningful.

The engine agrees with BJW 37.1% and with GOK 35.7% (no better annotator
exists - dropping either changes nothing; each passage already carries a
single annotator's labels). That is **far below** the ~82-90% binary label
ceiling: the engine, not the labels, is the failure. On the 198 cleanest
agreed-fall units the engine scores only 27.8%.

## Where the engine fails (follow-up diagnostics)

- Terminal-window slopes: falls median **-4.3 st/s** (threshold -8), rises
  median **+0.1** - the glide is not in the terminal window at all.
- No window definition recovers it: whole-slice, first-0.3s, max-|slope|
  0.2s window, median-smoothed variants all give fall and rise *identical*
  distributions (test accuracy 56-68%, below the 94% predict-fall baseline).
- Root cause: the short-slice F0 tracks are octave-chaotic. Voicing gaps
  jump 30-100 st (e.g. 21.4 -> 29.6 -> None -> 0.0), and even *within*
  contiguous voiced runs the median local slope is **-89 st/s for both
  classes** - the track oscillates octaves constantly on creaky broadcast
  voices in 0.55 s of context. No window, threshold or smoothing on top of
  this track can reach the bar; the fix is at the pitch-extraction level
  (Praat octave handling / longer context), which is a research task, not a
  threshold tweak. The octave-error bar stays met (0.04%) - it measures
  adjacent-frame rescues, not gap-crossing jumps.

## What changed for this run

- Harness slice now ends at the nuclear glide (nucleus + 0.45 s), not the
  unit tail - mid-unit nuclei were previously measured as level tails
  (fall recall went 29% -> 50% in the smoke).
- SEC low fall (grave) joins the fall class (high/low variants collapse;
  the SEC manual FIG2 documents both).

## Verdict

M2 bar **not met on Aix-MARSEC**, and the engine is the failure: the binary
labels are ~82-90% reproducible, but the engine's short-slice pitch tracks
carry no glide signal at all (identical slope statistics for falls and
rises). Two honest paths remain: (1) pitch-extraction research (Praat
octave/gap handling on short, creaky audio), retested on Aix as a research
benchmark; (2) the fall-vs-rise bar moves to controlled recordings -
scripted tone contrasts, recorded and verified by listening - which is the
app's real use case and the material human testing needs anyway. Aix remains
the label-free octave-error evidence (bar met). The label conventions, the
ceiling analysis and the slope diagnostics live in `eval/scratch/`.
