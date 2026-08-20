# m1-2026-08-20-dh-contrast-v10-ð-vs-z-d-v

- model: dh-contrast-v10
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: pooled-bar-not-met**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.92 | pooled-bar-not-met | 0.9/0.1365 | 0.7434/0.1414 | 0.2376 | 0.8818 |
| spike | 0.92 | pooled-bar-not-met | 0.9/0.1365 | 0.7434/0.1414 | 0.2376 | 0.8818 |
| vote | 0.695 | SHIPPING BAR NOT MET | 0.6999/0.6686 | 0.6062/0.6633 | 0.6334 | 0.768 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.628 * score + -0.304)
- threshold: 0.92
- train: precision 0.9, recall 0.1365

## Held-out

- precision 0.7434 / recall 0.1414 / f1 0.2376
- AUC 0.8818
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.92. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.6154 | 0.2 | 0.3019 | False |
| hindi | 205 | 93 | 0.625 | 0.0538 | 0.099 | False |
| korean | 205 | 125 | 0.6842 | 0.104 | 0.1806 | False |
| mandarin | 1646 | 109 | 0.5556 | 0.0917 | 0.1575 | False |
| spanish | 140 | 88 | 0.6 | 0.0682 | 0.1224 | False |
| vietnamese | 196 | 139 | 0.9333 | 0.3022 | 0.4565 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.785 | 0.1417 | 0.24 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 4817.7 ms, median warm 29.1 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-BWC-arctic_a0029 /ð/ -> /d/ (p=0.9394) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0029.wav
- [FP] l2arctic-BWC-arctic_a0047 /ð/ -> /d/ (p=0.9432) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0047.wav
- [FP] l2arctic-EBVS-arctic_a0018 /ð/ -> /d/ (p=0.923) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0018.wav
- [FP] l2arctic-HJK-arctic_a0029 /ð/ -> /d/ (p=0.9337) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0029.wav
- [FP] l2arctic-HJK-arctic_b0019 /ð/ -> /d/ (p=0.9561) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0019.wav
- [FP] l2arctic-HJK-arctic_b0106 /ð/ -> /d/ (p=0.9355) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0106.wav
- [FP] l2arctic-HJK-arctic_b0143 /ð/ -> /d/ (p=0.9377) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0143.wav
- [FP] l2arctic-HQTV-arctic_a0081 /ð/ -> /d/ (p=0.9303) label=correct l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0081.wav
- [FP] l2arctic-MBMPS-arctic_a0041 /ð/ -> /v/ (p=0.9624) label=correct l1=spanish data/l2arctic/MBMPS/wav/arctic_a0041.wav
- [FP] l2arctic-MBMPS-arctic_a0136 /ð/ -> /d/ (p=0.9361) label=correct l1=spanish data/l2arctic/MBMPS/wav/arctic_a0136.wav
- [FP] l2arctic-NJS-arctic_a0303 /ð/ -> /d/ (p=0.9203) label=correct l1=spanish data/l2arctic/NJS/wav/arctic_a0303.wav
- [FP] l2arctic-PNV-arctic_a0019 /ð/ -> /d/ (p=0.942) label=correct l1=vietnamese data/l2arctic/PNV/wav/arctic_a0019.wav
- [FP] l2arctic-RRBI-arctic_b0396 /ð/ -> /d/ (p=0.9392) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_b0396.wav
- [FP] l2arctic-SKA-arctic_a0075 /ð/ -> /z/ (p=0.9252) label=correct l1=arabic data/l2arctic/SKA/wav/arctic_a0075.wav
- [FP] l2arctic-SVBI-arctic_b0038 /ð/ -> /d/ (p=0.9337) label=correct l1=hindi data/l2arctic/SVBI/wav/arctic_b0038.wav
- [TP] l2arctic-TLV-arctic_a0134 /ð/ -> /d/ (p=0.9644) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0134.wav
- [TP] l2arctic-TLV-arctic_a0095 /ð/ -> /v/ (p=0.9472) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0095.wav
- [TP] l2arctic-NJS-arctic_a0260 /ð/ -> /d/ (p=0.9371) label=substituted l1=spanish data/l2arctic/NJS/wav/arctic_a0260.wav
- [TP] l2arctic-HQTV-arctic_b0358 /ð/ -> /z/ (p=0.9685) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0358.wav
- [TP] l2arctic-TLV-arctic_a0089 /ð/ -> /d/ (p=0.9523) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0089.wav
- [TP] l2arctic-TLV-arctic_b0251 /ð/ -> /d/ (p=0.9211) label=deleted l1=vietnamese data/l2arctic/TLV/wav/arctic_b0251.wav
- [TP] l2arctic-YDCK-arctic_b0143 /ð/ -> /d/ (p=0.9458) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_b0143.wav
- [TP] l2arctic-HQTV-arctic_a0108 /ð/ -> /d/ (p=0.9327) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0108.wav
- [TP] l2arctic-TLV-arctic_a0459 /ð/ -> /z/ (p=0.9795) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0459.wav
- [TP] l2arctic-HQTV-arctic_b0251 /ð/ -> /d/ (p=0.9741) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0251.wav
- [TP] l2arctic-TNI-arctic_b0197 /ð/ -> /d/ (p=0.9538) label=substituted l1=hindi data/l2arctic/TNI/wav/arctic_b0197.wav
- [TP] l2arctic-TXHC-arctic_a0041 /ð/ -> /z/ (p=0.9767) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0041.wav
- [TP] l2arctic-HQTV-arctic_a0112 /ð/ -> /d/ (p=0.9464) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0112.wav
- [TP] l2arctic-TLV-arctic_a0003 /ð/ -> /d/ (p=0.9598) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0003.wav
- [TP] l2arctic-HQTV-arctic_a0018 /ð/ -> /d/ (p=0.9282) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0018.wav
