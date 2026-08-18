# m1-2026-08-18-wav2vec2-espeak-cv-ft-ð-vs-z-d-v

- model: wav2vec2-espeak-cv-ft
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: SHIPPING BAR NOT MET**
- **shipping variant: gop**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.549 |
| spike | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5492 |
| vote | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.581 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5941 |

## Operating point (train split)

- Platt: p = sigmoid(0.151 * score + -0.514)
- threshold: 0.995
- train: precision 0.0, recall 0.0

## Held-out

- precision 0.0 / recall 0.0 / f1 0.0
- AUC 0.5941
- verdicts 2608, refused 0 (audio-problem 0)

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

- statuses: {'ok': 6565, 'low_confidence': 113}
- mean alignment confidence (ok): 0.8176

## Latency

- cold 0.0 ms, median warm 356.5 ms
- download size 1.26 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

