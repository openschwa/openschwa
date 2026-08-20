# Decision record: the mirror pivot (2026-08-20)

Status: DECIDED by the project owner · Scope: replaces the M1 judge line's
product role; the judge line is parked as research, not deleted.

## The decision

M1's shipped feedback stops being a **judge** — "your /ð/ sounded more like
/z/" — and becomes a **mirror**: the engine reports what it *heard* at the
focus slot of a known drill sentence, and the learner is the judge. Three
states, all confidence-gated:

1. heard-as-intended: "I heard /ð/ — right on target."
2. heard-other: "I heard /z/ where you were going for /ð/."
3. couldn't-tell: "I couldn't tell what that /ð/ sounded like — try once
   more, a little slower." (a first-class refusal, never a failure)

The owner's answers that shaped this record:

1. **Scope:** drills-only mirror first, as the test of the direction, until
   it "works fine enough". Free-form transcription stays a later ambition.
2. **Display:** the whole sentence, in phones. The focus phone is never
   guessed from speech — the drill text declares it and the trusted
   alignment pins its interval; the ear only labels that interval, and the
   UI composes the whole-sentence view from the drill's expected phones with
   the focus slot filled by the ear.
3. **Judge fate:** parked as research inside the project (code + eval
   reports stay; they ship nothing). If the mirror approach holds, the judge
   work is carved out into a research archive later.

## Why this direction

The post-M1 research report (models-and-data-post-m1.md) closed the judge
line's scaling question honestly: error-annotated L2 audio does not exist at
open scale — minutes, not hours; connected speech has none at all. The judge
needed error labels. The mirror needs only **transcripts** — "what phone was
spoken here" — and that data exists in bulk (Common Voice EN ≈ 3,000 h,
CC0; GLOBE CC-BY). Connected speech flips from being the judge's scaling
wall into being the mirror's training data. Same ambition, different target;
"no feedback beats wrong feedback" survives unchanged as "no *claim* beats an
honest 'couldn't tell'."

The product shape is a proven one (ELSA/SpeechAce report what they hear;
human teachers do the same: "I heard 'zis' — say it again"), and it fits
OpenSchwa's promise better than an error verdict: the mirror makes a softer,
more honest claim that a model can actually back with evidence.

## What the exam now measures

The eval harness gained a mirror exam (eval/README.md): for every held-out
token whose *realized* phone the corpora record, the ear's report is scored
against it.

- **accuracy**: among confident reports, P(heard == realized)
- **coverage**: share of tokens the mirror answers (the rest are refusals)
- realized×heard confusion table — every systematic mishearing is visible
- per-L1 audit of the single accent-blind line (informational, never a gate)

The bar: **accuracy ≥ 0.90 at coverage ≥ 0.4** on held-out speakers — the
judge bar's numbers re-read as mirror honesty. The calibration file gains a
separate `hearing_platt` / `hearing_threshold` block (P(heard == realized));
the judge's substitution fit is never reused for hearing and ships only if
its own exam passed.

## Where the ear comes from

Phase 0 (this milestone chunk): the best ears we already hold — the
XLS-R-300M fusion classifier (tinyschwa-v1), the M1 CTC judge (dh-contrast-v1),
and the charsiu aligner — examined as-is, no training spend. The exam report
is the go/no-go evidence for the concept.

Phase 1 (next): the real ear — XLS-R-300M frozen + a CTC phone head
fine-tuned on transcript-only mass (Common Voice CC0 + GLOBE CC-BY + our
held corpora), on the RTX 4060 laptop. Full phone vocabulary, so the mirror
serves every drill phone; connected speech scales because the labels are
transcripts. NC corpora stay eval-only; shippable weights are trained on
CC0/CC-BY data only (the report's §7 Q4 default stands until the owner says
otherwise).

## Open questions still standing (report §7)

1. Scope after the /ð/ pilot: /θ/ next?
2. Russian-accented audio: Datatang AESRC2020 RU / self-record / defer?
3. Paid corpora: any budget, or free + registration-only?
4. Weight distribution: CC0/CC-BY-trained only (current default)?
5. Hardware ceiling: Mac + 4060 laptop, or is cloud GPU conceivable?

None of these gate Phase 0 — they are needed before Phase 1 training spend.
