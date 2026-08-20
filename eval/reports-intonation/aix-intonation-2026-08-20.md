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

## The ground truth is the story

Before the run, the labels themselves were audited:

- **Inter-annotator ceiling.** 98 passages were transcribed twice (BJW/GOK).
  On units with *identical boundaries*, the two experts agree on the nuclear
  tone **36.2%** overall - 62% for falls, 14% fall-rise, 0/3 rise; restricted
  to fall/rise/fall-rise: **50.0%** (n=96).
- **Marks vs acoustics.** The engine's own F0 shape at 10,200 unit-final
  marked words shows the "high fall" (\`) and "high rise" (/) marks have
  *identical* acoustic distributions (33% falling / 24% rising vs 34% / 24%).
  The marks are largely decoupled from the measurable glide.
- **Imbalance.** 5,077 fall vs 303 rise units - news reading barely rises.

The engine agrees with BJW 38.0% and with GOK 37.6% - i.e. it performs **at
the human inter-annotator ceiling** on this corpus (36-50%). No classifier
can hit the roadmap's 90% fall-vs-rise bar against labels that reproduce
themselves at ~36-50%; the 37.8% number measures the labels' noise floor,
not the DSP. The octave-error bar is label-free and is met with room to spare.

## What changed for this run

- Harness slice now ends at the nuclear glide (nucleus + 0.45 s), not the
  unit tail - mid-unit nuclei were previously measured as level tails
  (fall recall went 29% -> 50% in the smoke).
- SEC low fall (grave) joins the fall class (high/low variants collapse;
  the SEC manual FIG2 documents both).

## Verdict

M2 bar **not met on Aix-MARSEC**, and not met *by construction*: the corpus
labels cannot certify a 90% tone-class bar. Aix remains the label-free
octave-error evidence (bar met) and a descriptive reality check. The
fall-vs-rise bar needs controlled recordings: scripted tone contrasts
recorded and verified by listening, the app's real use case. The Aix-MARSEC
label conventions, the ceiling analysis scripts and the mark census live in
`eval/scratch/`.
