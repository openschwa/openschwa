# m1-2026-08-19-dh-contrast-v5-ð-vs-z-d-v

- model: dh-contrast-v5
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: SHIPPING BAR NOT MET**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.895 | SHIPPING BAR NOT MET | 0.8985/0.3069 | 0.6951/0.2879 | 0.4071 | 0.8807 |
| spike | 0.895 | SHIPPING BAR NOT MET | 0.8985/0.3069 | 0.6951/0.2879 | 0.4071 | 0.8804 |
| vote | 0.855 | SHIPPING BAR NOT MET | 0.8589/0.5495 | 0.7153/0.5202 | 0.6023 | 0.7295 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.721 * score + 1.056)
- threshold: 0.895
- train: precision 0.8985, recall 0.3069

## Held-out

- precision 0.6951 / recall 0.2879 / f1 0.4071
- AUC 0.8807
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.895. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.4783 | 0.275 | 0.3492 | False |
| hindi | 205 | 93 | 0.7742 | 0.2581 | 0.3871 | False |
| korean | 205 | 125 | 0.619 | 0.208 | 0.3114 | False |
| mandarin | 1646 | 109 | 0.5254 | 0.2844 | 0.369 | False |
| spanish | 140 | 88 | 0.7333 | 0.25 | 0.3729 | False |
| vietnamese | 196 | 139 | 0.9344 | 0.4101 | 0.57 | True |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.76 | 0.2884 | 0.4181 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 4582.2 ms, median warm 28.6 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0094 /ð/ -> /v/ (p=0.9235) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0094.wav
- [FP] l2arctic-ASI-arctic_a0082 /ð/ -> /d/ (p=0.9218) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0082.wav
- [FP] l2arctic-BWC-arctic_a0029 /ð/ -> /d/ (p=0.9491) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0029.wav
- [FP] l2arctic-BWC-arctic_a0031 /ð/ -> /d/ (p=0.9271) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0031.wav
- [FP] l2arctic-BWC-arctic_a0047 /ð/ -> /d/ (p=0.9133) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0047.wav
- [FP] l2arctic-BWC-arctic_b0288 /ð/ -> /d/ (p=0.9371) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_b0288.wav
- [FP] l2arctic-EBVS-arctic_a0071 /ð/ -> /d/ (p=0.9029) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0071.wav
- [FP] l2arctic-EBVS-arctic_a0081 /ð/ -> /z/ (p=0.9627) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0081.wav
- [FP] l2arctic-EBVS-arctic_a0148 /ð/ -> /d/ (p=0.9011) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0148.wav
- [FP] l2arctic-EBVS-arctic_a0299 /ð/ -> /d/ (p=0.9082) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0299.wav
- [FP] l2arctic-HJK-arctic_a0029 /ð/ -> /d/ (p=0.9562) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0029.wav
- [FP] l2arctic-HJK-arctic_a0047 /ð/ -> /v/ (p=0.9703) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0047.wav
- [FP] l2arctic-HJK-arctic_a0054 /ð/ -> /d/ (p=0.9071) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0054.wav
- [FP] l2arctic-HJK-arctic_b0019 /ð/ -> /d/ (p=0.9001) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0019.wav
- [FP] l2arctic-HJK-arctic_b0106 /ð/ -> /d/ (p=0.9037) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0106.wav
- [TP] l2arctic-HQTV-arctic_a0030 /ð/ -> /d/ (p=0.9248) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0030.wav
- [TP] l2arctic-TXHC-arctic_a0053 /ð/ -> /z/ (p=0.9765) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0053.wav
- [TP] l2arctic-BWC-arctic_a0575 /ð/ -> /z/ (p=0.9651) label=deleted l1=mandarin data/l2arctic/BWC/wav/arctic_a0575.wav
- [TP] l2arctic-TXHC-arctic_b0251 /ð/ -> /d/ (p=0.9457) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_b0251.wav
- [TP] l2arctic-YDCK-arctic_a0091 /ð/ -> /d/ (p=0.9084) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_a0091.wav
- [TP] l2arctic-BWC-arctic_b0358 /ð/ -> /d/ (p=0.8977) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_b0358.wav
- [TP] l2arctic-YKWK-arctic_a0006 /ð/ -> /d/ (p=0.9341) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_a0006.wav
- [TP] l2arctic-HQTV-arctic_a0013 /ð/ -> /d/ (p=0.9336) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0013.wav
- [TP] l2arctic-YDCK-arctic_a0116 /ð/ -> /d/ (p=0.9274) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_a0116.wav
- [TP] l2arctic-NCC-arctic_a0019 /ð/ -> /z/ (p=0.9735) label=substituted l1=mandarin data/l2arctic/NCC/wav/arctic_a0019.wav
- [TP] l2arctic-TXHC-arctic_a0037 /ð/ -> /z/ (p=0.9817) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0037.wav
- [TP] l2arctic-PNV-arctic_a0128 /ð/ -> /d/ (p=0.9067) label=substituted l1=vietnamese data/l2arctic/PNV/wav/arctic_a0128.wav
- [TP] l2arctic-ABA-arctic_a0500 /ð/ -> /z/ (p=0.9808) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0500.wav
- [TP] l2arctic-RRBI-arctic_b0017 /ð/ -> /d/ (p=0.9351) label=substituted l1=hindi data/l2arctic/RRBI/wav/arctic_b0017.wav
- [TP] l2arctic-TNI-arctic_a0246 /ð/ -> /z/ (p=0.9746) label=substituted l1=hindi data/l2arctic/TNI/wav/arctic_a0246.wav
