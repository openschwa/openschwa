# m1-2026-08-20-dh-contrast-v10-ð-vs-z-d-v

- model: dh-contrast-v10
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 0 train / 1312 cal / 2410 held-out
- **status: pooled-bar-not-met**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | cal P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.88 | pooled-bar-not-met | 0.919/0.3503 | 0.7343/0.2011 | 0.3158 | 0.891 |
| spike | 0.88 | pooled-bar-not-met | 0.919/0.3503 | 0.7343/0.2011 | 0.3158 | 0.891 |
| vote | 0.835 | SHIPPING BAR NOT MET | 0.8364/0.7332 | 0.695/0.6724 | 0.6835 | 0.7954 |
| gop | 0.59 | pooled-bar-not-met | 1.0/0.0018 | 0.0/0.0 | 0.0 | 0.4362 |

## Operating point (calibration split)

- Platt: p = sigmoid(0.495 * score + 0.227)
- threshold: 0.88
- cal: precision 0.919, recall 0.3503
- cal->test precision gap: 0.1847

## Held-out

- precision 0.7343 / recall 0.2011 / f1 0.3158
- AUC 0.891
- verdicts 2407, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.88. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 150 | 20 | 0.7647 | 0.65 | 0.7027 | False |
| hindi | 158 | 85 | 0.9286 | 0.1529 | 0.2626 | False |
| korean | 161 | 102 | 0.88 | 0.2157 | 0.3465 | False |
| mandarin | 1640 | 111 | 0.5763 | 0.3063 | 0.4 | False |
| spanish | 152 | 122 | 0.875 | 0.1148 | 0.2029 | False |
| vietnamese | 149 | 82 | 0.75 | 0.1098 | 0.1915 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 | AUC |
|---|---|---|---|---|---|---|---|
| l2arctic | 915 | 521 | 0.875 | 0.2015 | 0.3276 | 0.8177 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 | 0.7411 |

## Alignment sanity

- statuses: {'ok': 3718, 'low_confidence': 4}
- mean alignment confidence (ok): 0.9001

## Latency

- cold 4643.0 ms, median warm 29.2 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0003 /ð/ -> /v/ (p=0.8816) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0003.wav
- [FP] l2arctic-ABA-arctic_a0046 /ð/ -> /d/ (p=0.9029) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0046.wav
- [FP] l2arctic-ABA-arctic_a0097 /ð/ -> /v/ (p=0.9405) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0097.wav
- [FP] l2arctic-ABA-arctic_a0097 /ð/ -> /v/ (p=0.8848) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0097.wav
- [FP] l2arctic-ASI-arctic_b0081 /ð/ -> /d/ (p=0.8937) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_b0081.wav
- [FP] l2arctic-BWC-arctic_a0031 /ð/ -> /d/ (p=0.8865) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0031.wav
- [FP] l2arctic-BWC-arctic_b0430 /ð/ -> /d/ (p=0.9013) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_b0430.wav
- [FP] l2arctic-ERMS-arctic_a0131 /ð/ -> /z/ (p=0.9163) label=correct l1=spanish data/l2arctic/ERMS/wav/arctic_a0131.wav
- [FP] l2arctic-ERMS-arctic_a0227 /ð/ -> /d/ (p=0.9059) label=correct l1=spanish data/l2arctic/ERMS/wav/arctic_a0227.wav
- [FP] l2arctic-THV-arctic_a0121 /ð/ -> /z/ (p=0.9324) label=correct l1=vietnamese data/l2arctic/THV/wav/arctic_a0121.wav
- [FP] l2arctic-THV-arctic_a0519 /ð/ -> /v/ (p=0.8907) label=correct l1=vietnamese data/l2arctic/THV/wav/arctic_a0519.wav
- [FP] l2arctic-THV-arctic_b0481 /ð/ -> /d/ (p=0.8918) label=correct l1=vietnamese data/l2arctic/THV/wav/arctic_b0481.wav
- [FP] l2arctic-YKWK-arctic_a0024 /ð/ -> /d/ (p=0.8878) label=correct l1=korean data/l2arctic/YKWK/wav/arctic_a0024.wav
- [FP] l2arctic-YKWK-arctic_a0108 /ð/ -> /d/ (p=0.8845) label=correct l1=korean data/l2arctic/YKWK/wav/arctic_a0108.wav
- [FP] l2arctic-YKWK-arctic_b0143 /ð/ -> /d/ (p=0.8883) label=correct l1=korean data/l2arctic/YKWK/wav/arctic_b0143.wav
- [TP] l2arctic-ERMS-arctic_a0065 /ð/ -> /d/ (p=0.8936) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0065.wav
- [TP] l2arctic-BWC-arctic_a0134 /ð/ -> /d/ (p=0.8879) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_a0134.wav
- [TP] l2arctic-ASI-arctic_a0029 /ð/ -> /d/ (p=0.9025) label=substituted l1=hindi data/l2arctic/ASI/wav/arctic_a0029.wav
- [TP] l2arctic-ABA-arctic_b0324 /ð/ -> /z/ (p=0.9313) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_b0324.wav
- [TP] l2arctic-YKWK-arctic_b0106 /ð/ -> /d/ (p=0.8848) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_b0106.wav
- [TP] l2arctic-BWC-arctic_b0476 /ð/ -> /d/ (p=0.8862) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_b0476.wav
- [TP] l2arctic-ABA-arctic_a0037 /ð/ -> /z/ (p=0.9521) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0037.wav
- [TP] l2arctic-ERMS-arctic_a0269 /ð/ -> /d/ (p=0.8824) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0269.wav
- [TP] l2arctic-THV-arctic_a0062 /ð/ -> /d/ (p=0.9009) label=substituted l1=vietnamese data/l2arctic/THV/wav/arctic_a0062.wav
- [TP] l2arctic-THV-arctic_b0527 /ð/ -> /d/ (p=0.9149) label=substituted l1=vietnamese data/l2arctic/THV/wav/arctic_b0527.wav
- [TP] l2arctic-THV-arctic_b0355 /ð/ -> /d/ (p=0.8803) label=substituted l1=vietnamese data/l2arctic/THV/wav/arctic_b0355.wav
- [TP] l2arctic-YKWK-arctic_a0006 /ð/ -> /d/ (p=0.9187) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_a0006.wav
- [TP] l2arctic-ABA-arctic_b0438 /ð/ -> /z/ (p=0.9055) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_b0438.wav
- [TP] l2arctic-BWC-arctic_a0120 /ð/ -> /d/ (p=0.9096) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_a0120.wav
- [TP] l2arctic-ERMS-arctic_a0086 /ð/ -> /d/ (p=0.9113) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0086.wav
