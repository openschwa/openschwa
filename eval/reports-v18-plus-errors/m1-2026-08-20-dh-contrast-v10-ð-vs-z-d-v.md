# m1-2026-08-20-dh-contrast-v10-ð-vs-z-d-v

- model: dh-contrast-v10
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: SHIPPING BAR NOT MET**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.72 | SHIPPING BAR NOT MET | 0.8144/0.3675 | 0.5928/0.3333 | 0.4267 | 0.8521 |
| spike | 0.72 | SHIPPING BAR NOT MET | 0.8144/0.3675 | 0.5928/0.3333 | 0.4267 | 0.8521 |
| vote | 0.805 | SHIPPING BAR NOT MET | 0.8065/0.3971 | 0.593/0.3704 | 0.456 | 0.6476 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.688 * score + 0.758)
- threshold: 0.72
- train: precision 0.8144, recall 0.3675

## Held-out

- precision 0.5928 / recall 0.3333 / f1 0.4267
- AUC 0.8521
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.72. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.4762 | 0.25 | 0.3279 | False |
| hindi | 205 | 93 | 0.6818 | 0.1613 | 0.2609 | False |
| korean | 205 | 125 | 0.7115 | 0.296 | 0.4181 | False |
| mandarin | 1646 | 109 | 0.2977 | 0.3578 | 0.325 | False |
| spanish | 140 | 88 | 0.8182 | 0.3068 | 0.4463 | False |
| vietnamese | 196 | 139 | 0.9333 | 0.5036 | 0.6542 | True |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.7725 | 0.3322 | 0.4646 |
| so762 | 1495 | 1 | 0.0127 | 1.0 | 0.025 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 3121.9 ms, median warm 29.0 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0060 /ð/ -> /v/ (p=0.8665) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0060.wav
- [FP] l2arctic-ABA-arctic_a0097 /ð/ -> /v/ (p=0.9047) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0097.wav
- [FP] l2arctic-ABA-arctic_a0122 /ð/ -> /v/ (p=0.9601) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0122.wav
- [FP] l2arctic-ASI-arctic_a0082 /ð/ -> /d/ (p=0.7982) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0082.wav
- [FP] l2arctic-BWC-arctic_a0029 /ð/ -> /d/ (p=0.9436) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0029.wav
- [FP] l2arctic-BWC-arctic_a0031 /ð/ -> /d/ (p=0.9094) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0031.wav
- [FP] l2arctic-BWC-arctic_a0042 /ð/ -> /d/ (p=0.81) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0042.wav
- [FP] l2arctic-BWC-arctic_a0047 /ð/ -> /d/ (p=0.9383) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0047.wav
- [FP] l2arctic-BWC-arctic_b0288 /ð/ -> /d/ (p=0.9473) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_b0288.wav
- [FP] l2arctic-EBVS-arctic_a0081 /ð/ -> /z/ (p=0.9675) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0081.wav
- [FP] l2arctic-EBVS-arctic_a0109 /ð/ -> /d/ (p=0.795) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0109.wav
- [FP] l2arctic-EBVS-arctic_a0148 /ð/ -> /d/ (p=0.8897) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0148.wav
- [FP] l2arctic-EBVS-arctic_a0299 /ð/ -> /d/ (p=0.8198) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0299.wav
- [FP] l2arctic-HJK-arctic_a0029 /ð/ -> /d/ (p=0.9676) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0029.wav
- [FP] l2arctic-HJK-arctic_a0047 /ð/ -> /v/ (p=0.9203) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0047.wav
- [TP] l2arctic-PNV-arctic_a0128 /ð/ -> /d/ (p=0.9137) label=substituted l1=vietnamese data/l2arctic/PNV/wav/arctic_a0128.wav
- [TP] l2arctic-HJK-arctic_a0416 /ð/ -> /d/ (p=0.9061) label=substituted l1=korean data/l2arctic/HJK/wav/arctic_a0416.wav
- [TP] l2arctic-ABA-arctic_a0500 /ð/ -> /z/ (p=0.986) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0500.wav
- [TP] l2arctic-TNI-arctic_a0144 /ð/ -> /d/ (p=0.9489) label=substituted l1=hindi data/l2arctic/TNI/wav/arctic_a0144.wav
- [TP] l2arctic-TLV-arctic_a0003 /ð/ -> /d/ (p=0.8461) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0003.wav
- [TP] l2arctic-BWC-arctic_a0089 /ð/ -> /d/ (p=0.9597) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_a0089.wav
- [TP] l2arctic-HQTV-arctic_b0518 /ð/ -> /d/ (p=0.7956) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0518.wav
- [TP] l2arctic-BWC-arctic_a0101 /ð/ -> /d/ (p=0.8976) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_a0101.wav
- [TP] l2arctic-TLV-arctic_a0003 /ð/ -> /d/ (p=0.9585) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0003.wav
- [TP] l2arctic-HQTV-arctic_b0106 /ð/ -> /d/ (p=0.9059) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0106.wav
- [TP] l2arctic-ABA-arctic_a0501 /ð/ -> /z/ (p=0.9848) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0501.wav
- [TP] l2arctic-YDCK-arctic_b0201 /ð/ -> /d/ (p=0.839) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_b0201.wav
- [TP] l2arctic-EBVS-arctic_a0170 /ð/ -> /d/ (p=0.7213) label=substituted l1=spanish data/l2arctic/EBVS/wav/arctic_a0170.wav
- [TP] l2arctic-TXHC-arctic_a0136 /ð/ -> /z/ (p=0.9822) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0136.wav
- [TP] l2arctic-YDCK-arctic_a0029 /ð/ -> /d/ (p=0.8356) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_a0029.wav
