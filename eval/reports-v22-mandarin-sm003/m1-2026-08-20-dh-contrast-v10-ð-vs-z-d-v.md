# m1-2026-08-20-dh-contrast-v10-ð-vs-z-d-v

- model: dh-contrast-v10
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: pooled-bar-not-met**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.89 | pooled-bar-not-met | 0.9006/0.1047 | 0.8072/0.1128 | 0.1979 | 0.9018 |
| spike | 0.89 | pooled-bar-not-met | 0.9018/0.1061 | 0.8072/0.1128 | 0.1979 | 0.9018 |
| vote | 0.66 | SHIPPING BAR NOT MET | 0.6648/0.7805 | 0.5985/0.7828 | 0.6783 | 0.8138 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.578 * score + -0.695)
- threshold: 0.89
- train: precision 0.9006, recall 0.1047

## Held-out

- precision 0.8072 / recall 0.1128 / f1 0.1979
- AUC 0.9018
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.89. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.5556 | 0.125 | 0.2041 | False |
| hindi | 205 | 93 | 0.7778 | 0.0753 | 0.1373 | False |
| korean | 205 | 125 | 0.8333 | 0.04 | 0.0763 | False |
| mandarin | 1646 | 109 | 0.5385 | 0.0642 | 0.1148 | False |
| spanish | 140 | 88 | 0.875 | 0.0795 | 0.1458 | False |
| vietnamese | 196 | 139 | 0.9474 | 0.259 | 0.4068 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.8481 | 0.113 | 0.1994 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 3060.4 ms, median warm 28.9 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-BWC-arctic_a0029 /ð/ -> /d/ (p=0.8952) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0029.wav
- [FP] l2arctic-LXC-arctic_b0430 /ð/ -> /d/ (p=0.9256) label=correct l1=mandarin data/l2arctic/LXC/wav/arctic_b0430.wav
- [FP] l2arctic-MBMPS-arctic_a0136 /ð/ -> /d/ (p=0.9018) label=correct l1=spanish data/l2arctic/MBMPS/wav/arctic_a0136.wav
- [FP] l2arctic-RRBI-arctic_a0029 /ð/ -> /d/ (p=0.9183) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_a0029.wav
- [FP] l2arctic-RRBI-arctic_b0396 /ð/ -> /d/ (p=0.897) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_b0396.wav
- [FP] l2arctic-SKA-arctic_a0075 /ð/ -> /z/ (p=0.8942) label=correct l1=arabic data/l2arctic/SKA/wav/arctic_a0075.wav
- [FP] l2arctic-TLV-arctic_a0459 /ð/ -> /d/ (p=0.897) label=correct l1=vietnamese data/l2arctic/TLV/wav/arctic_a0459.wav
- [FP] l2arctic-TLV-arctic_a0542 /ð/ -> /d/ (p=0.9067) label=correct l1=vietnamese data/l2arctic/TLV/wav/arctic_a0542.wav
- [FP] l2arctic-YBAA-arctic_b0432 /ð/ -> /d/ (p=0.9056) label=correct l1=arabic data/l2arctic/YBAA/wav/arctic_b0432.wav
- [FP] l2arctic-YDCK-arctic_a0041 /ð/ -> /d/ (p=0.9034) label=correct l1=korean data/l2arctic/YDCK/wav/arctic_a0041.wav
- [FP] l2arctic-ZHAA-arctic_a0006 /ð/ -> /v/ (p=0.9102) label=correct l1=arabic data/l2arctic/ZHAA/wav/arctic_a0006.wav
- [FP] l2arctic-ZHAA-arctic_b0438 /ð/ -> /d/ (p=0.9203) label=correct l1=arabic data/l2arctic/ZHAA/wav/arctic_b0438.wav
- [FP] so762-011350290 /ð/ -> /z/ (p=0.8978) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER1135/011350290.WAV
- [FP] so762-028970153 /ð/ -> /z/ (p=0.9227) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER2897/028970153.WAV
- [FP] so762-091180035 /ð/ -> /z/ (p=0.8931) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER9118/091180035.WAV
- [TP] l2arctic-HQTV-arctic_a0029 /ð/ -> /d/ (p=0.8967) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0029.wav
- [TP] l2arctic-TLV-arctic_a0073 /ð/ -> /d/ (p=0.9157) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0073.wav
- [TP] l2arctic-HQTV-arctic_b0358 /ð/ -> /d/ (p=0.895) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0358.wav
- [TP] l2arctic-HQTV-arctic_b0251 /ð/ -> /d/ (p=0.9343) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0251.wav
- [TP] l2arctic-ERMS-arctic_a0013 /ð/ -> /d/ (p=0.8975) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0013.wav
- [TP] l2arctic-TNI-arctic_a0144 /ð/ -> /d/ (p=0.8959) label=substituted l1=hindi data/l2arctic/TNI/wav/arctic_a0144.wav
- [TP] l2arctic-TLV-arctic_a0003 /ð/ -> /d/ (p=0.9107) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0003.wav
- [TP] l2arctic-YBAA-arctic_a0139 /ð/ -> /z/ (p=0.8995) label=substituted l1=arabic data/l2arctic/YBAA/wav/arctic_a0139.wav
- [TP] l2arctic-HQTV-arctic_a0078 /ð/ -> /d/ (p=0.9077) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0078.wav
- [TP] l2arctic-TLV-arctic_a0089 /ð/ -> /d/ (p=0.8917) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0089.wav
- [TP] l2arctic-ERMS-arctic_a0041 /ð/ -> /d/ (p=0.9041) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0041.wav
- [TP] l2arctic-LXC-arctic_b0026 /ð/ -> /z/ (p=0.9268) label=substituted l1=mandarin data/l2arctic/LXC/wav/arctic_b0026.wav
- [TP] l2arctic-HQTV-arctic_b0358 /ð/ -> /z/ (p=0.9485) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0358.wav
- [TP] l2arctic-HQTV-arctic_a0225 /ð/ -> /d/ (p=0.9222) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0225.wav
- [TP] l2arctic-HQTV-arctic_a0131 /ð/ -> /d/ (p=0.9043) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0131.wav
