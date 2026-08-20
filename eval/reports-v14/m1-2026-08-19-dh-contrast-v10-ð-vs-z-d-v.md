# m1-2026-08-19-dh-contrast-v10-ð-vs-z-d-v

- model: dh-contrast-v10
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: pooled-bar-not-met**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.945 | pooled-bar-not-met | 0.9074/0.1769 | 0.7171/0.1835 | 0.2922 | 0.8578 |
| spike | 0.945 | pooled-bar-not-met | 0.9077/0.1776 | 0.7171/0.1835 | 0.2922 | 0.8578 |
| vote | 0.83 | SHIPPING BAR NOT MET | 0.8313/0.3949 | 0.633/0.4007 | 0.4907 | 0.666 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.749 * score + 0.790)
- threshold: 0.945
- train: precision 0.9074, recall 0.1769

## Held-out

- precision 0.7171 / recall 0.1835 / f1 0.2922
- AUC 0.8578
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.945. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.4545 | 0.125 | 0.1961 | False |
| hindi | 205 | 93 | 0.7647 | 0.1398 | 0.2364 | False |
| korean | 205 | 125 | 0.5625 | 0.072 | 0.1277 | False |
| mandarin | 1646 | 109 | 0.4 | 0.1468 | 0.2148 | False |
| spanish | 140 | 88 | 1.0 | 0.1477 | 0.2574 | False |
| vietnamese | 196 | 139 | 0.9636 | 0.3813 | 0.5464 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.8321 | 0.1838 | 0.3011 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 4464.4 ms, median warm 29.1 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-BWC-arctic_a0029 /ð/ -> /d/ (p=0.9829) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0029.wav
- [FP] l2arctic-BWC-arctic_a0047 /ð/ -> /d/ (p=0.97) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0047.wav
- [FP] l2arctic-HJK-arctic_a0029 /ð/ -> /d/ (p=0.9834) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0029.wav
- [FP] l2arctic-HJK-arctic_b0019 /ð/ -> /d/ (p=0.9533) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0019.wav
- [FP] l2arctic-HJK-arctic_b0106 /ð/ -> /d/ (p=0.9475) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0106.wav
- [FP] l2arctic-HKK-arctic_a0037 /ð/ -> /d/ (p=0.9605) label=correct l1=korean data/l2arctic/HKK/wav/arctic_a0037.wav
- [FP] l2arctic-HKK-arctic_a0112 /ð/ -> /d/ (p=0.9802) label=correct l1=korean data/l2arctic/HKK/wav/arctic_a0112.wav
- [FP] l2arctic-HQTV-arctic_a0081 /ð/ -> /d/ (p=0.9881) label=correct l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0081.wav
- [FP] l2arctic-LXC-arctic_b0430 /ð/ -> /d/ (p=0.9887) label=correct l1=mandarin data/l2arctic/LXC/wav/arctic_b0430.wav
- [FP] l2arctic-RRBI-arctic_a0029 /ð/ -> /d/ (p=0.9738) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_a0029.wav
- [FP] l2arctic-RRBI-arctic_a0095 /ð/ -> /z/ (p=0.9785) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_a0095.wav
- [FP] l2arctic-RRBI-arctic_a0303 /ð/ -> /d/ (p=0.9853) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_a0303.wav
- [FP] l2arctic-RRBI-arctic_b0057 /ð/ -> /d/ (p=0.9742) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_b0057.wav
- [FP] l2arctic-SKA-arctic_a0053 /ð/ -> /z/ (p=0.9678) label=correct l1=arabic data/l2arctic/SKA/wav/arctic_a0053.wav
- [FP] l2arctic-SKA-arctic_a0461 /ð/ -> /z/ (p=0.9686) label=correct l1=arabic data/l2arctic/SKA/wav/arctic_a0461.wav
- [TP] l2arctic-PNV-arctic_b0355 /ð/ -> /d/ (p=0.9695) label=substituted l1=vietnamese data/l2arctic/PNV/wav/arctic_b0355.wav
- [TP] l2arctic-BWC-arctic_a0575 /ð/ -> /z/ (p=0.958) label=deleted l1=mandarin data/l2arctic/BWC/wav/arctic_a0575.wav
- [TP] l2arctic-ABA-arctic_a0500 /ð/ -> /z/ (p=0.9867) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0500.wav
- [TP] l2arctic-NCC-arctic_b0400 /ð/ -> /d/ (p=0.9759) label=substituted l1=mandarin data/l2arctic/NCC/wav/arctic_b0400.wav
- [TP] l2arctic-YDCK-arctic_b0143 /ð/ -> /d/ (p=0.9703) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_b0143.wav
- [TP] l2arctic-ERMS-arctic_a0029 /ð/ -> /v/ (p=0.9917) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0029.wav
- [TP] l2arctic-NJS-arctic_b0538 /ð/ -> /d/ (p=0.9567) label=substituted l1=spanish data/l2arctic/NJS/wav/arctic_b0538.wav
- [TP] l2arctic-SKA-arctic_a0122 /ð/ -> /z/ (p=0.9908) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0122.wav
- [TP] l2arctic-TLV-arctic_a0071 /ð/ -> /d/ (p=0.9778) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0071.wav
- [TP] l2arctic-HQTV-arctic_a0051 /ð/ -> /d/ (p=0.9625) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0051.wav
- [TP] l2arctic-TLV-arctic_a0542 /ð/ -> /d/ (p=0.945) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0542.wav
- [TP] l2arctic-THV-arctic_a0097 /ð/ -> /d/ (p=0.9523) label=substituted l1=vietnamese data/l2arctic/THV/wav/arctic_a0097.wav
- [TP] l2arctic-HQTV-arctic_a0112 /ð/ -> /d/ (p=0.9845) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0112.wav
- [TP] l2arctic-HQTV-arctic_a0137 /ð/ -> /d/ (p=0.9891) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0137.wav
- [TP] l2arctic-RRBI-arctic_a0116 /ð/ -> /d/ (p=0.9474) label=substituted l1=hindi data/l2arctic/RRBI/wav/arctic_a0116.wav
