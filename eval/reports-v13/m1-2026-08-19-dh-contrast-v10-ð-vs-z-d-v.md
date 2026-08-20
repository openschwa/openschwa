# m1-2026-08-19-dh-contrast-v10-ð-vs-z-d-v

- model: dh-contrast-v10
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: SHIPPING BAR NOT MET**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.72 | SHIPPING BAR NOT MET | 0.8745/0.322 | 0.5814/0.2946 | 0.3911 | 0.8501 |
| spike | 0.735 | SHIPPING BAR NOT MET | 0.8743/0.3213 | 0.5853/0.2946 | 0.3919 | 0.8501 |
| vote | 0.86 | SHIPPING BAR NOT MET | 0.8635/0.3928 | 0.5802/0.3653 | 0.4483 | 0.6436 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.683 * score + 0.569)
- threshold: 0.72
- train: precision 0.8745, recall 0.322

## Held-out

- precision 0.5814 / recall 0.2946 / f1 0.3911
- AUC 0.8501
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.72. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.4615 | 0.3 | 0.3636 | False |
| hindi | 205 | 93 | 0.8095 | 0.1828 | 0.2982 | False |
| korean | 205 | 125 | 0.6415 | 0.272 | 0.382 | False |
| mandarin | 1646 | 109 | 0.2617 | 0.2569 | 0.2593 | False |
| spanish | 140 | 88 | 0.7391 | 0.1932 | 0.3063 | False |
| vietnamese | 196 | 139 | 0.9437 | 0.482 | 0.6381 | True |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.7598 | 0.2934 | 0.4234 |
| so762 | 1495 | 1 | 0.0139 | 1.0 | 0.0274 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 4374.2 ms, median warm 29.1 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0030 /ð/ -> /v/ (p=0.9685) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0030.wav
- [FP] l2arctic-ABA-arctic_a0060 /ð/ -> /v/ (p=0.8156) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0060.wav
- [FP] l2arctic-ABA-arctic_a0094 /ð/ -> /v/ (p=0.9022) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0094.wav
- [FP] l2arctic-ABA-arctic_a0097 /ð/ -> /v/ (p=0.8711) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0097.wav
- [FP] l2arctic-ABA-arctic_a0426 /ð/ -> /d/ (p=0.7785) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0426.wav
- [FP] l2arctic-ASI-arctic_a0082 /ð/ -> /d/ (p=0.9681) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0082.wav
- [FP] l2arctic-ASI-arctic_a0139 /ð/ -> /d/ (p=0.7368) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0139.wav
- [FP] l2arctic-BWC-arctic_a0029 /ð/ -> /d/ (p=0.7217) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0029.wav
- [FP] l2arctic-BWC-arctic_a0029 /ð/ -> /d/ (p=0.9596) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0029.wav
- [FP] l2arctic-BWC-arctic_a0047 /ð/ -> /d/ (p=0.8257) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0047.wav
- [FP] l2arctic-BWC-arctic_b0288 /ð/ -> /d/ (p=0.9896) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_b0288.wav
- [FP] l2arctic-EBVS-arctic_a0023 /ð/ -> /d/ (p=0.7384) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0023.wav
- [FP] l2arctic-EBVS-arctic_a0148 /ð/ -> /d/ (p=0.7567) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0148.wav
- [FP] l2arctic-EBVS-arctic_a0299 /ð/ -> /d/ (p=0.7968) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0299.wav
- [FP] l2arctic-HJK-arctic_a0029 /ð/ -> /d/ (p=0.9909) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0029.wav
- [TP] l2arctic-BWC-arctic_a0134 /ð/ -> /d/ (p=0.932) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_a0134.wav
- [TP] l2arctic-HQTV-arctic_a0078 /ð/ -> /d/ (p=0.9382) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0078.wav
- [TP] l2arctic-YDCK-arctic_a0003 /ð/ -> /d/ (p=0.8508) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_a0003.wav
- [TP] l2arctic-TNI-arctic_a0029 /ð/ -> /d/ (p=0.8853) label=substituted l1=hindi data/l2arctic/TNI/wav/arctic_a0029.wav
- [TP] l2arctic-HQTV-arctic_a0007 /ð/ -> /d/ (p=0.9612) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0007.wav
- [TP] l2arctic-YDCK-arctic_a0542 /ð/ -> /d/ (p=0.7405) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_a0542.wav
- [TP] l2arctic-HQTV-arctic_b0358 /ð/ -> /d/ (p=0.9759) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0358.wav
- [TP] l2arctic-NJS-arctic_a0041 /ð/ -> /z/ (p=1.0) label=substituted l1=spanish data/l2arctic/NJS/wav/arctic_a0041.wav
- [TP] l2arctic-HQTV-arctic_a0225 /ð/ -> /d/ (p=0.9126) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0225.wav
- [TP] l2arctic-HQTV-arctic_a0051 /ð/ -> /d/ (p=0.9205) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0051.wav
- [TP] l2arctic-YKWK-arctic_a0006 /ð/ -> /d/ (p=0.7393) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_a0006.wav
- [TP] l2arctic-SKA-arctic_a0095 /ð/ -> /z/ (p=0.9925) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0095.wav
- [TP] l2arctic-TXHC-arctic_a0536 /ð/ -> /z/ (p=0.9856) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0536.wav
- [TP] l2arctic-PNV-arctic_b0315 /ð/ -> /d/ (p=0.8697) label=substituted l1=vietnamese data/l2arctic/PNV/wav/arctic_b0315.wav
- [TP] l2arctic-ABA-arctic_a0159 /ð/ -> /z/ (p=0.9978) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0159.wav
