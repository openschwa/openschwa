# m1-2026-08-20-dh-contrast-v11-ð-vs-z-d-other

- model: dh-contrast-v11
- contrast: /ð/ vs ['z', 'd', 'other']
- tokens: 0 train / 1312 cal / 2410 held-out
- **status: SHIPPING BAR NOT MET**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | cal P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.515 | SHIPPING BAR NOT MET | 0.5198/0.1906 | 0.2235/0.1494 | 0.1791 | 0.606 |
| spike | 0.515 | SHIPPING BAR NOT MET | 0.5198/0.1906 | 0.2235/0.1494 | 0.1791 | 0.606 |
| vote | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5449 |
| gop | 0.59 | pooled-bar-not-met | 1.0/0.0018 | 0.0/0.0 | 0.0 | 0.4362 |

## Operating point (calibration split)

- Platt: p = sigmoid(0.612 * score + -1.118)
- threshold: 0.515
- cal: precision 0.5198, recall 0.1906
- cal->test precision gap: 0.2963

## Held-out

- precision 0.2235 / recall 0.1494 / f1 0.1791
- AUC 0.606
- verdicts 2407, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.515. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 150 | 20 | 0.5 | 0.35 | 0.4118 | False |
| hindi | 158 | 85 | 0.6452 | 0.2353 | 0.3448 | False |
| korean | 161 | 102 | 0.5556 | 0.098 | 0.1667 | False |
| mandarin | 1640 | 111 | 0.1102 | 0.2523 | 0.1534 | False |
| spanish | 152 | 122 | 0.5556 | 0.041 | 0.0763 | False |
| vietnamese | 149 | 82 | 0.3478 | 0.0976 | 0.1524 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 | AUC |
|---|---|---|---|---|---|---|---|
| l2arctic | 915 | 521 | 0.6016 | 0.1478 | 0.2373 | 0.5914 |
| so762 | 1495 | 1 | 0.0045 | 1.0 | 0.009 | 0.8974 |

## Alignment sanity

- statuses: {'ok': 3718, 'low_confidence': 4}
- mean alignment confidence (ok): 0.9001

## Latency

- cold 3524.3 ms, median warm 29.9 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0041 /ð/ -> /z/ (p=0.52) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0041.wav
- [FP] l2arctic-ABA-arctic_a0062 /ð/ -> /d/ (p=0.5179) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0062.wav
- [FP] l2arctic-ABA-arctic_a0076 /ð/ -> /d/ (p=0.5471) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0076.wav
- [FP] l2arctic-ABA-arctic_a0117 /ð/ -> /z/ (p=0.7918) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0117.wav
- [FP] l2arctic-ABA-arctic_b0106 /ð/ -> /other/ (p=0.5489) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_b0106.wav
- [FP] l2arctic-ABA-arctic_b0433 /ð/ -> /d/ (p=0.5356) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_b0433.wav
- [FP] l2arctic-ABA-arctic_b0518 /ð/ -> /d/ (p=0.5173) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_b0518.wav
- [FP] l2arctic-ASI-arctic_a0077 /ð/ -> /z/ (p=0.6302) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0077.wav
- [FP] l2arctic-ASI-arctic_a0089 /ð/ -> /d/ (p=0.5888) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0089.wav
- [FP] l2arctic-ASI-arctic_a0091 /ð/ -> /d/ (p=0.5345) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0091.wav
- [FP] l2arctic-ASI-arctic_a0113 /ð/ -> /d/ (p=0.6198) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0113.wav
- [FP] l2arctic-ASI-arctic_a0191 /ð/ -> /d/ (p=0.5773) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0191.wav
- [FP] l2arctic-ASI-arctic_b0073 /ð/ -> /d/ (p=0.5449) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_b0073.wav
- [FP] l2arctic-ASI-arctic_b0158 /ð/ -> /d/ (p=0.5553) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_b0158.wav
- [FP] l2arctic-ASI-arctic_b0215 /ð/ -> /d/ (p=0.6051) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_b0215.wav
- [TP] l2arctic-BWC-arctic_b0189 /ð/ -> /d/ (p=0.5569) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_b0189.wav
- [TP] l2arctic-BWC-arctic_b0358 /ð/ -> /d/ (p=0.5615) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_b0358.wav
- [TP] l2arctic-BWC-arctic_b0481 /ð/ -> /d/ (p=0.6169) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_b0481.wav
- [TP] l2arctic-ASI-arctic_a0003 /ð/ -> /d/ (p=0.5386) label=substituted l1=hindi data/l2arctic/ASI/wav/arctic_a0003.wav
- [TP] l2arctic-BWC-arctic_b0476 /ð/ -> /other/ (p=0.5377) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_b0476.wav
- [TP] l2arctic-YKWK-arctic_a0046 /ð/ -> /d/ (p=0.6369) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_a0046.wav
- [TP] l2arctic-BWC-arctic_a0088 /ð/ -> /d/ (p=0.5817) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_a0088.wav
- [TP] l2arctic-ASI-arctic_a0062 /ð/ -> /d/ (p=0.5206) label=substituted l1=hindi data/l2arctic/ASI/wav/arctic_a0062.wav
- [TP] l2arctic-THV-arctic_a0089 /ð/ -> /d/ (p=0.5245) label=substituted l1=vietnamese data/l2arctic/THV/wav/arctic_a0089.wav
- [TP] l2arctic-BWC-arctic_b0527 /ð/ -> /d/ (p=0.5755) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_b0527.wav
- [TP] l2arctic-YKWK-arctic_a0026 /ð/ -> /d/ (p=0.6118) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_a0026.wav
- [TP] l2arctic-BWC-arctic_b0476 /ð/ -> /d/ (p=0.5253) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_b0476.wav
- [TP] l2arctic-BWC-arctic_a0139 /ð/ -> /z/ (p=0.6889) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_a0139.wav
- [TP] l2arctic-YKWK-arctic_b0197 /ð/ -> /other/ (p=0.5161) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_b0197.wav
- [TP] l2arctic-ERMS-arctic_a0103 /ð/ -> /other/ (p=0.5416) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0103.wav
