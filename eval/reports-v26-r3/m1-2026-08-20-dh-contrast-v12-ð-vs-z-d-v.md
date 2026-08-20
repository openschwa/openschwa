# m1-2026-08-20-dh-contrast-v12-ð-vs-z-d-v

- model: dh-contrast-v12
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 0 train / 1312 cal / 2410 held-out
- **status: pooled-bar-not-met**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | cal P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.925 | pooled-bar-not-met | 1.0/0.0036 | 0.0/0.0 | 0.0 | 0.6915 |
| spike | 0.925 | pooled-bar-not-met | 1.0/0.0036 | 0.0/0.0 | 0.0 | 0.6915 |
| vote | 0.63 | SHIPPING BAR NOT MET | 0.6336/0.3013 | 0.2791/0.1609 | 0.2041 | 0.5229 |
| gop | 0.59 | pooled-bar-not-met | 1.0/0.0018 | 0.0/0.0 | 0.0 | 0.4362 |

## Operating point (calibration split)

- Platt: p = sigmoid(0.346 * score + 0.105)
- threshold: 0.925
- cal: precision 1.0, recall 0.0036
- cal->test precision gap: 1.0

## Held-out

- precision 0.0 / recall 0.0 / f1 0.0
- AUC 0.6915
- verdicts 2407, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.925. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 150 | 20 | 0.0 | 0.0 | 0.0 | False |
| hindi | 158 | 85 | 0.0 | 0.0 | 0.0 | False |
| korean | 161 | 102 | 0.0 | 0.0 | 0.0 | False |
| mandarin | 1640 | 111 | 0.0 | 0.0 | 0.0 | False |
| spanish | 152 | 122 | 0.0 | 0.0 | 0.0 | False |
| vietnamese | 149 | 82 | 0.0 | 0.0 | 0.0 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 | AUC |
|---|---|---|---|---|---|---|---|
| l2arctic | 915 | 521 | 0.0 | 0.0 | 0.0 | 0.5964 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 | 0.9779 |

## Alignment sanity

- statuses: {'ok': 3718, 'low_confidence': 4}
- mean alignment confidence (ok): 0.9001

## Latency

- cold 4731.9 ms, median warm 29.2 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

