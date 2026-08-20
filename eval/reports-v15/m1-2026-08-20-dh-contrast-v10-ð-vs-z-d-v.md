# m1-2026-08-20-dh-contrast-v10-ð-vs-z-d-v

- model: dh-contrast-v10
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: pooled-bar-not-met**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.905 | pooled-bar-not-met | 0.9011/0.2895 | 0.6678/0.3182 | 0.431 | 0.8546 |
| spike | 0.905 | pooled-bar-not-met | 0.9011/0.2895 | 0.6678/0.3182 | 0.431 | 0.8546 |
| vote | 0.865 | SHIPPING BAR NOT MET | 0.8699/0.3812 | 0.6554/0.4226 | 0.5138 | 0.6785 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.661 * score + 1.103)
- threshold: 0.905
- train: precision 0.9011, recall 0.2895

## Held-out

- precision 0.6678 / recall 0.3182 / f1 0.431
- AUC 0.8546
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.905. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.4848 | 0.4 | 0.4384 | False |
| hindi | 205 | 93 | 0.7692 | 0.2151 | 0.3361 | False |
| korean | 205 | 125 | 0.7556 | 0.272 | 0.4 | False |
| mandarin | 1646 | 109 | 0.378 | 0.2844 | 0.3246 | False |
| spanish | 140 | 88 | 0.8 | 0.1818 | 0.2963 | False |
| vietnamese | 196 | 139 | 0.9351 | 0.518 | 0.6667 | True |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.7683 | 0.3187 | 0.4505 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 4766.6 ms, median warm 29.7 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0030 /ð/ -> /v/ (p=0.9769) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0030.wav
- [FP] l2arctic-ABA-arctic_a0094 /ð/ -> /v/ (p=0.9717) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0094.wav
- [FP] l2arctic-ABA-arctic_a0097 /ð/ -> /v/ (p=0.9934) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0097.wav
- [FP] l2arctic-ABA-arctic_b0518 /ð/ -> /v/ (p=0.9957) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_b0518.wav
- [FP] l2arctic-ASI-arctic_a0082 /ð/ -> /d/ (p=0.9442) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0082.wav
- [FP] l2arctic-BWC-arctic_a0029 /ð/ -> /d/ (p=0.9931) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0029.wav
- [FP] l2arctic-BWC-arctic_a0047 /ð/ -> /d/ (p=0.9571) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0047.wav
- [FP] l2arctic-BWC-arctic_b0288 /ð/ -> /d/ (p=0.9828) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_b0288.wav
- [FP] l2arctic-EBVS-arctic_a0081 /ð/ -> /z/ (p=0.944) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0081.wav
- [FP] l2arctic-EBVS-arctic_a0148 /ð/ -> /d/ (p=0.9739) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0148.wav
- [FP] l2arctic-HJK-arctic_a0029 /ð/ -> /d/ (p=0.9909) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0029.wav
- [FP] l2arctic-HJK-arctic_a0047 /ð/ -> /v/ (p=0.9663) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0047.wav
- [FP] l2arctic-HJK-arctic_b0106 /ð/ -> /d/ (p=0.9309) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0106.wav
- [FP] l2arctic-HKK-arctic_a0037 /ð/ -> /d/ (p=0.9547) label=correct l1=korean data/l2arctic/HKK/wav/arctic_a0037.wav
- [FP] l2arctic-HKK-arctic_a0128 /ð/ -> /d/ (p=0.9723) label=correct l1=korean data/l2arctic/HKK/wav/arctic_a0128.wav
- [TP] l2arctic-ABA-arctic_a0159 /ð/ -> /z/ (p=0.9937) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0159.wav
- [TP] l2arctic-HQTV-arctic_b0201 /ð/ -> /d/ (p=0.9928) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0201.wav
- [TP] l2arctic-ASI-arctic_b0017 /ð/ -> /d/ (p=0.9901) label=substituted l1=hindi data/l2arctic/ASI/wav/arctic_b0017.wav
- [TP] l2arctic-YKWK-arctic_a0384 /ð/ -> /d/ (p=0.9821) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_a0384.wav
- [TP] l2arctic-NCC-arctic_b0430 /ð/ -> /d/ (p=0.9896) label=substituted l1=mandarin data/l2arctic/NCC/wav/arctic_b0430.wav
- [TP] l2arctic-HKK-arctic_a0081 /ð/ -> /d/ (p=0.9135) label=substituted l1=korean data/l2arctic/HKK/wav/arctic_a0081.wav
- [TP] l2arctic-YKWK-arctic_a0025 /ð/ -> /d/ (p=0.9953) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_a0025.wav
- [TP] l2arctic-YKWK-arctic_b0197 /ð/ -> /d/ (p=0.9703) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_b0197.wav
- [TP] l2arctic-HQTV-arctic_b0518 /ð/ -> /d/ (p=0.974) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0518.wav
- [TP] l2arctic-ASI-arctic_b0371 /ð/ -> /d/ (p=0.958) label=substituted l1=hindi data/l2arctic/ASI/wav/arctic_b0371.wav
- [TP] l2arctic-YBAA-arctic_a0139 /ð/ -> /z/ (p=0.9416) label=substituted l1=arabic data/l2arctic/YBAA/wav/arctic_a0139.wav
- [TP] l2arctic-BWC-arctic_a0101 /ð/ -> /d/ (p=0.9479) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_a0101.wav
- [TP] l2arctic-YDCK-arctic_b0047 /ð/ -> /d/ (p=0.9458) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_b0047.wav
- [TP] l2arctic-HQTV-arctic_a0112 /ð/ -> /d/ (p=0.9894) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0112.wav
- [TP] l2arctic-YDCK-arctic_b0068 /ð/ -> /d/ (p=0.9698) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_b0068.wav
