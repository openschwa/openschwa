# m1-2026-08-19-dh-contrast-v2-ð-vs-z-d-v

- model: dh-contrast-v2
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: pooled-bar-not-met**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.97 | pooled-bar-not-met | 1.0/0.0007 | 0.5/0.0017 | 0.0034 | 0.8433 |
| spike | 0.97 | pooled-bar-not-met | 1.0/0.0007 | 0.5/0.0017 | 0.0034 | 0.8432 |
| vote | 0.73 | SHIPPING BAR NOT MET | 0.7313/0.3892 | 0.6566/0.4411 | 0.5277 | 0.6865 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.536 * score + 0.424)
- threshold: 0.97
- train: precision 1.0, recall 0.0007

## Held-out

- precision 0.5 / recall 0.0017 / f1 0.0034
- AUC 0.8433
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.97. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.0 | 0.0 | 0.0 | False |
| hindi | 205 | 93 | 0.0 | 0.0 | 0.0 | False |
| korean | 205 | 125 | 0.0 | 0.0 | 0.0 | False |
| mandarin | 1646 | 109 | 0.0 | 0.0 | 0.0 | False |
| spanish | 140 | 88 | 0.0 | 0.0 | 0.0 | False |
| vietnamese | 196 | 139 | 1.0 | 0.0072 | 0.0143 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 1.0 | 0.0017 | 0.0034 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 2248.6 ms, median warm 25.5 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] so762-036310080 /ð/ -> /z/ (p=0.9726) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER3631/036310080.WAV
- [TP] l2arctic-HQTV-arctic_b0358 /ð/ -> /z/ (p=0.9705) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0358.wav
