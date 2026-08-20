# m1-2026-08-20-dh-contrast-v10-ð-vs-z-d-v

- model: dh-contrast-v10
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: pooled-bar-not-met**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.995 | pooled-bar-not-met | 1.0/0.0094 | 1.0/0.0067 | 0.0134 | 0.833 |
| spike | 0.995 | pooled-bar-not-met | 1.0/0.0087 | 1.0/0.0067 | 0.0134 | 0.833 |
| vote | 0.855 | SHIPPING BAR NOT MET | 0.8553/0.3329 | 0.6877/0.3114 | 0.4287 | 0.6348 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.837 * score + 1.342)
- threshold: 0.995
- train: precision 1.0, recall 0.0094

## Held-out

- precision 1.0 / recall 0.0067 / f1 0.0134
- AUC 0.833
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.995. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.0 | 0.0 | 0.0 | False |
| hindi | 205 | 93 | 0.0 | 0.0 | 0.0 | False |
| korean | 205 | 125 | 0.0 | 0.0 | 0.0 | False |
| mandarin | 1646 | 109 | 1.0 | 0.0183 | 0.036 | False |
| spanish | 140 | 88 | 0.0 | 0.0 | 0.0 | False |
| vietnamese | 196 | 139 | 1.0 | 0.0144 | 0.0284 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 1.0 | 0.0067 | 0.0134 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 3090.2 ms, median warm 29.0 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [TP] l2arctic-TXHC-arctic_a0037 /ð/ -> /z/ (p=0.996) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0037.wav
- [TP] l2arctic-TLV-arctic_a0095 /ð/ -> /v/ (p=0.996) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0095.wav
- [TP] l2arctic-TXHC-arctic_a0041 /ð/ -> /z/ (p=0.9955) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0041.wav
- [TP] l2arctic-HQTV-arctic_b0358 /ð/ -> /z/ (p=0.9952) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0358.wav
