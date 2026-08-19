# m1-2026-08-19-dh-contrast-v1-ð-vs-z-d-v

- model: dh-contrast-v1
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: ok-no-l1-floor**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.625 | ok-no-l1-floor | 1.0/0.0007 | 0.0/0.0 | 0.0 | 0.5522 |
| spike | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5172 |
| vote | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5484 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.200 * score + -0.677)
- threshold: 0.625
- train: precision 1.0, recall 0.0007

## Held-out

- precision 0.0 / recall 0.0 / f1 0.0
- AUC 0.5522
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 (held-out)

| l1 | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.0 | 0.0 | 0.0 |
| hindi | 205 | 93 | 0.0 | 0.0 | 0.0 |
| korean | 205 | 125 | 0.0 | 0.0 | 0.0 |
| mandarin | 1646 | 109 | 0.0 | 0.0 | 0.0 |
| spanish | 140 | 88 | 0.0 | 0.0 | 0.0 |
| vietnamese | 196 | 139 | 0.0 | 0.0 | 0.0 |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.0 | 0.0 | 0.0 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 1540.7 ms, median warm 159.3 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

