# Evaluation harness

Offline measurement of feedback quality. **A feedback type ships only when this
harness proves it meets the bar.** This is project policy, not a target, and
this file is where the bar is defined — everywhere else in the repo cites here:

- precision **≥ 0.90** at recall **≥ ~0.3–0.5** on held-out data;
- **no L1 group below ~0.8 precision** — accent-specific failure must not
  average away;
- a human spot-check of ~30 flagged items finds no absurd flags;
- the operating threshold lives in a committed calibration file traceable to a
  committed report under `reports/`.

If the bar cannot be met at useful recall, the threshold moves until precision
holds. Recall is negotiable; precision is not. A feedback type that cannot
clear the bar does not ship — the engine says "retry" instead, which is a
first-class outcome rather than a failure.

Imports the engine as a library — no HTTP in the loop. Lands in M1.

## Datasets (not committed; adapters expect local paths)

- **speechocean762** — open MDD corpus, phone-level accuracy scores from five
  experts; L1-Mandarin speakers (adults and children).
  https://www.openslr.org/101/
- **L2-ARCTIC** — six L1 backgrounds (Hindi, Korean, Mandarin, Spanish,
  Arabic, Vietnamese); TextGrids tag substitutions/deletions (e.g. ð → d).
  https://psi.engr.tamu.edu/l2-arctic-corpus/

Both skew to specific L1s and read speech — eval numbers are a floor of
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
