# m1-2026-08-21-ear-xlsr-v1-θ-vs-s-t-f

- model: ear-xlsr-v1
- contrast: /θ/ vs ['s', 't', 'f']
- tokens: 0 train / 302 cal / 569 held-out
- **status: SHIPPING BAR NOT MET** (the mirror; judge: SHIPPING BAR NOT MET)

## Mirror - what the ear heard (shipped line)

- Platt: p = sigmoid(0.482 * hearing_score + -1.118)  [P(heard == realized)]
- threshold: 0.995
- cal: accuracy 0.0 / coverage 0.0 (cal status SHIPPING BAR NOT MET)
- **held-out: accuracy 0.0 / coverage 0.0** / answer-rate 0.0
- raw top-1 hearing accuracy (no gating): 0.1335
- confident reports 0 of 569 tokens (scored 567; unscorable 0)

### realized x heard (confident reports)

| realized \ heard |  |
|---|

### Mirror per L1 (held-out)

Fairness audit of the single shipped line: every group is heard at
the same global operating point 0.995. Informational
only - it never gates the bar.

| l1 | tokens | confident | accuracy | coverage | fair |
|---|---|---|---|---|---|
| arabic | 34 | 0 | 0.0 | 0.0 | True |
| hindi | 43 | 0 | 0.0 | 0.0 | True |
| korean | 30 | 0 | 0.0 | 0.0 | True |
| mandarin | 391 | 0 | 0.0 | 0.0 | True |
| spanish | 34 | 0 | 0.0 | 0.0 | True |
| vietnamese | 37 | 0 | 0.0 | 0.0 | True |

### Mirror spot-check (confident reports, mishearings first)


## Judge variants (research archive - parked by the mirror pivot)

| variant | threshold | status | cal P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.4604 |
| spike | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.6335 |
| vote | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5124 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5329 |

## Judge operating point (calibration split, research archive)

- Platt: p = sigmoid(0.141 * score + -1.443)
- threshold: 0.995
- cal: precision 0.0, recall 0.0
- cal->test precision gap: 0.0

## Judge held-out (research archive)

- precision 0.0 / recall 0.0 / f1 0.0
- AUC 0.6335
- verdicts 567, refused 2 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.995. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 34 | 5 | 0.0 | 0.0 | 0.0 | False |
| hindi | 43 | 31 | 0.0 | 0.0 | 0.0 | False |
| korean | 30 | 9 | 0.0 | 0.0 | 0.0 | False |
| mandarin | 391 | 8 | 0.0 | 0.0 | 0.0 | False |
| spanish | 34 | 30 | 0.0 | 0.0 | 0.0 | False |
| vietnamese | 37 | 28 | 0.0 | 0.0 | 0.0 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 | AUC |
|---|---|---|---|---|---|---|---|
| l2arctic | 221 | 107 | 0.0 | 0.0 | 0.0 | 0.6108 |
| so762 | 348 | 4 | 0.0 | 0.0 | 0.0 | 0.3275 |

## Alignment sanity

- statuses: {'ok': 868, 'low_confidence': 3}
- mean alignment confidence (ok): 0.9027

## Latency

- cold 5659.4 ms, median warm 31.8 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

