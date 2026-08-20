# m1-2026-08-20-dh-contrast-v10-ð-vs-z-d-v

- model: dh-contrast-v10
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: pooled-bar-not-met**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.86 | pooled-bar-not-met | 0.9003/0.4108 | 0.7124/0.367 | 0.4844 | 0.8361 |
| spike | 0.87 | pooled-bar-not-met | 0.9007/0.3668 | 0.7153/0.33 | 0.4516 | 0.8361 |
| vote | 0.815 | SHIPPING BAR NOT MET | 0.8192/0.7097 | 0.6083/0.6431 | 0.6252 | 0.7604 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.535 * score + 0.112)
- threshold: 0.86
- train: precision 0.9003, recall 0.4108

## Held-out

- precision 0.7124 / recall 0.367 / f1 0.4844
- AUC 0.8361
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.86. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.5152 | 0.425 | 0.4658 | False |
| hindi | 205 | 93 | 0.6744 | 0.3118 | 0.4265 | False |
| korean | 205 | 125 | 0.7917 | 0.304 | 0.4393 | False |
| mandarin | 1646 | 109 | 0.5 | 0.3486 | 0.4108 | False |
| spanish | 140 | 88 | 0.8966 | 0.2955 | 0.4444 | False |
| vietnamese | 196 | 139 | 0.9091 | 0.5036 | 0.6481 | True |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.7927 | 0.3676 | 0.5023 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 3027.4 ms, median warm 29.1 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0097 /ð/ -> /v/ (p=0.9454) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0097.wav
- [FP] l2arctic-ABA-arctic_a0097 /ð/ -> /v/ (p=0.8881) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0097.wav
- [FP] l2arctic-ASI-arctic_a0082 /ð/ -> /d/ (p=0.8783) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0082.wav
- [FP] l2arctic-ASI-arctic_a0097 /ð/ -> /d/ (p=0.8608) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0097.wav
- [FP] l2arctic-ASI-arctic_a0137 /ð/ -> /d/ (p=0.8613) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0137.wav
- [FP] l2arctic-BWC-arctic_a0031 /ð/ -> /d/ (p=0.89) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0031.wav
- [FP] l2arctic-BWC-arctic_a0047 /ð/ -> /d/ (p=0.8664) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0047.wav
- [FP] l2arctic-EBVS-arctic_a0148 /ð/ -> /d/ (p=0.8862) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0148.wav
- [FP] l2arctic-HJK-arctic_a0029 /ð/ -> /d/ (p=0.8696) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0029.wav
- [FP] l2arctic-HJK-arctic_a0047 /ð/ -> /v/ (p=0.9107) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0047.wav
- [FP] l2arctic-HJK-arctic_b0019 /ð/ -> /d/ (p=0.8714) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0019.wav
- [FP] l2arctic-HJK-arctic_b0106 /ð/ -> /d/ (p=0.9045) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0106.wav
- [FP] l2arctic-HJK-arctic_b0201 /ð/ -> /d/ (p=0.9275) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0201.wav
- [FP] l2arctic-HJK-arctic_b0315 /ð/ -> /d/ (p=0.8769) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0315.wav
- [FP] l2arctic-HQTV-arctic_a0081 /ð/ -> /d/ (p=0.9155) label=correct l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0081.wav
- [TP] l2arctic-HQTV-arctic_b0251 /ð/ -> /d/ (p=0.8954) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0251.wav
- [TP] l2arctic-TNI-arctic_a0116 /ð/ -> /d/ (p=0.8965) label=substituted l1=hindi data/l2arctic/TNI/wav/arctic_a0116.wav
- [TP] l2arctic-ERMS-arctic_a0121 /ð/ -> /d/ (p=0.8815) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0121.wav
- [TP] l2arctic-SKA-arctic_a0122 /ð/ -> /z/ (p=0.9481) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0122.wav
- [TP] l2arctic-TNI-arctic_a0365 /ð/ -> /d/ (p=0.8833) label=substituted l1=hindi data/l2arctic/TNI/wav/arctic_a0365.wav
- [TP] l2arctic-HQTV-arctic_b0201 /ð/ -> /d/ (p=0.9156) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0201.wav
- [TP] l2arctic-HJK-arctic_a0112 /ð/ -> /d/ (p=0.8927) label=substituted l1=korean data/l2arctic/HJK/wav/arctic_a0112.wav
- [TP] l2arctic-EBVS-arctic_a0113 /ð/ -> /d/ (p=0.8891) label=substituted l1=spanish data/l2arctic/EBVS/wav/arctic_a0113.wav
- [TP] l2arctic-SKA-arctic_b0534 /ð/ -> /z/ (p=0.9594) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_b0534.wav
- [TP] l2arctic-BWC-arctic_b0400 /ð/ -> /d/ (p=0.923) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_b0400.wav
- [TP] l2arctic-NCC-arctic_a0019 /ð/ -> /d/ (p=0.8658) label=substituted l1=mandarin data/l2arctic/NCC/wav/arctic_a0019.wav
- [TP] l2arctic-HQTV-arctic_a0108 /ð/ -> /d/ (p=0.8678) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0108.wav
- [TP] l2arctic-HKK-arctic_a0139 /ð/ -> /v/ (p=0.912) label=substituted l1=korean data/l2arctic/HKK/wav/arctic_a0139.wav
- [TP] l2arctic-SKA-arctic_a0095 /ð/ -> /z/ (p=0.9482) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0095.wav
- [TP] l2arctic-HQTV-arctic_b0425 /ð/ -> /d/ (p=0.9097) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0425.wav
