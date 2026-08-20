# Evaluation harness

Offline measurement of feedback quality. **A feedback type ships only when this
harness proves it meets the bar.** This is project policy, not a target, and
this file is where the bar is defined — everywhere else in the repo cites here:

- precision **≥ 0.90** at recall **≥ 0.4** on held-out data (the numeric
  targets are `PRECISION_TARGET` / `RECALL_TARGET` in
  `src/openschwa_eval/harness.py`; they change only by an explicit project
  decision — decided 2026-08-20: the 0.4 recall floor is firm);
- the **per-L1 breakdown is a fairness audit** — informational, reported per
  run so accent-specific failure stays visible, but the gate is the pooled
  number;
- a human spot-check of ~30 flagged items finds no absurd flags;
- the operating threshold lives in a committed calibration file traceable to a
  committed report under `reports*/`.

The threshold sweep is precision-first: among operating points holding the
precision target, the highest recall wins. A feedback type that cannot
clear the bar does not ship — the engine says "retry" instead, which is a
first-class outcome rather than a failure.

Imports the engine as a library - no HTTP in the loop.

## Running it (M1)

```bash
just models                              # acoustic model(s) into the cache
cd eval && uv sync --extra ml            # harness env, with the engine's ml extra
uv run python run_eval.py \
    --contrast "ð:z,d,v" \
    --l2arctic data/l2arctic --speechocean762 data/speechocean762 \
    --model wav2vec2-espeak-cv-ft --model charsiu-en-w2v2-ctc \
    --out reports/
```

Per model the harness: runs every labeled target token through the full
pipeline (include_ungated=true, no HTTP), fits Platt calibration on the
train pool, sweeps the operating threshold precision-first, measures the
held-out pool through the *shipped* code path (one threshold for every
learner - the judge is blind to who is speaking; the per-L1 breakdown in the
report is a fairness audit, not a gate), and
writes <run>.md + <run>.json under reports/. The winner's calibration is
committed to the engine's scoring/calibration.yaml **only when `--commit` is
passed AND the bar is met** (candidate runs never write it; only the winner
re-run may); a run that misses the bar commits nothing and says so in the
report.
Candidates whose manifest role is 'contrast' (e.g. the Option 3 fine-tuned
judge dh-contrast-v1) are wired through the engine's dedicated
contrast_model_id path: the default aligner keeps aligning, the candidate
only scores the focus segment. The Option 3 exam lives in reports-v4/ (a
separate directory because its run tag collided with the earlier bake-off —
the laptop clock had not rolled over); verdict there: negative, bar not met.

Checkpointed: token-level results stream to reports/checkpoints/<run>.jsonl,
so an interrupted run resumes instead of restarting. --limit N runs the
first N utterances for a smoke pass. The flagged-items list in the report is
the human spot-check (~30 items, with audio paths and labels).

## Datasets (not committed; adapters expect local paths)

- **speechocean762** - open MDD corpus, phone-level accuracy scores from five
  experts; L1-Mandarin speakers (adults and children). Ground truth is
  resource/scores-detail.json: each expert spells realized phones with
  braces around errors; >=3/5 experts bracketing a phone marks it substituted.
  https://www.openslr.org/101/
- **L2-ARCTIC** - six L1 backgrounds (Hindi, Korean, Mandarin, Spanish,
  Arabic, Vietnamese); annotation TextGrids carry mispronunciation tags
  ("target,realized,code" triples: s/d/a) in the phones tier. Only annotated
  utterances have ground truth and are evaluated. License: CC BY-NC 4.0 -
  fine for evaluation, never committed.
  https://psi.engr.tamu.edu/l2-arctic-corpus/

Both skew to specific L1s and read speech - eval numbers are a floor of
evidence, not proof for every learner. Report metrics **per contrast and per
L1 group**; accent-specific failures must not average away.

## Procedure per feedback type

1. Synthesize an exercise spec from each utterance's transcript (the harness
   knows the text — same information the real app has).
2. Run the full pipeline with `include_ungated=true`.
3. Score flags against corpus labels: precision / recall / F1.
4. Sweep the confidence threshold → PR curve → pick the operating point for
   the precision target; write it to `engine/.../scoring/calibration.yaml`
   (strict train/held-out split hygiene).
5. Emit a versioned report under `reports/` (markdown + JSON, committed) so
   every shipped threshold is traceable to evidence.

CI runs only a tiny bundled-fixture smoke subset; full runs are manual/nightly.

## Integrity note (2026-08-20)

A hand-written `calibration.yaml` was found staged next to the engine's
scoring code (rejected by rename, never committed). It carried round-number
Platt constants, `threshold: 0.5` — which collapses `decide()`'s bands so the
`uncertain` verdict becomes unreachable — cited as provenance a report whose
own status was `SHIPPING BAR NOT MET`, included a /z/-target contrast no exam
ever produced, and loosened the alignment gates below the engine defaults.
It validated cleanly against the schema: nothing in code stopped it.

The defense is now in code:
`engine/tests/test_calibration.py::test_committed_calibration_traceable_to_passing_report`
fails whenever a committed calibration does not trace, value-for-value, to a
committed report with `status: ok`. The only sanctioned writer of
`calibration.yaml` remains this harness, via `run_eval.py --commit` on a full
run that meets the bar.
