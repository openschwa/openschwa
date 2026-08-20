# m1-2026-08-19-dh-contrast-v6-ð-vs-z-d-v

- model: dh-contrast-v6
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: pooled-bar-not-met**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.975 | pooled-bar-not-met | 0.9048/0.0137 | 0.6364/0.0118 | 0.0231 | 0.8912 |
| spike | 0.975 | pooled-bar-not-met | 0.9048/0.0137 | 0.6364/0.0118 | 0.0231 | 0.8908 |
| vote | 0.85 | SHIPPING BAR NOT MET | 0.8545/0.509 | 0.7285/0.4562 | 0.5611 | 0.703 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.650 * score + 1.113)
- threshold: 0.975
- train: precision 0.9048, recall 0.0137

## Held-out

- precision 0.6364 / recall 0.0118 / f1 0.0231
- AUC 0.8912
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.975. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 1.0 | 0.025 | 0.0488 | False |
| hindi | 205 | 93 | 1.0 | 0.0108 | 0.0213 | False |
| korean | 205 | 125 | 0.0 | 0.0 | 0.0 | False |
| mandarin | 1646 | 109 | 0.3333 | 0.0183 | 0.0348 | False |
| spanish | 140 | 88 | 1.0 | 0.0114 | 0.0225 | False |
| vietnamese | 196 | 139 | 1.0 | 0.0144 | 0.0284 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.875 | 0.0118 | 0.0233 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 4390.1 ms, median warm 28.7 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-NCC-arctic_a0041 /ð/ -> /z/ (p=0.9754) label=correct l1=mandarin data/l2arctic/NCC/wav/arctic_a0041.wav
- [FP] so762-011090334 /ð/ -> /z/ (p=0.9754) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER1109/011090334.WAV
- [FP] so762-028970146 /ð/ -> /z/ (p=0.9756) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER2897/028970146.WAV
- [FP] so762-096470008 /ð/ -> /z/ (p=0.9814) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER9647/096470008.WAV
- [TP] l2arctic-HQTV-arctic_b0358 /ð/ -> /z/ (p=0.979) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0358.wav
- [TP] l2arctic-SKA-arctic_a0117 /ð/ -> /z/ (p=0.9762) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0117.wav
- [TP] l2arctic-TLV-arctic_a0095 /ð/ -> /v/ (p=0.9799) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0095.wav
- [TP] l2arctic-LXC-arctic_a0081 /ð/ -> /v/ (p=0.9802) label=deleted l1=mandarin data/l2arctic/LXC/wav/arctic_a0081.wav
- [TP] l2arctic-TXHC-arctic_a0037 /ð/ -> /z/ (p=0.9755) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0037.wav
- [TP] l2arctic-ERMS-arctic_a0029 /ð/ -> /v/ (p=0.9758) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0029.wav
- [TP] l2arctic-TNI-arctic_a0246 /ð/ -> /z/ (p=0.977) label=substituted l1=hindi data/l2arctic/TNI/wav/arctic_a0246.wav
