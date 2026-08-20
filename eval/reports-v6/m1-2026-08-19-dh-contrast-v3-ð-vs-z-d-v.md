# m1-2026-08-19-dh-contrast-v3-ð-vs-z-d-v

- model: dh-contrast-v3
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: SHIPPING BAR NOT MET**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.92 | SHIPPING BAR NOT MET | 0.8966/0.2253 | 0.7754/0.2441 | 0.3713 | 0.8797 |
| spike | 0.92 | SHIPPING BAR NOT MET | 0.8966/0.2253 | 0.7754/0.2441 | 0.3713 | 0.8797 |
| vote | 0.86 | SHIPPING BAR NOT MET | 0.8638/0.3617 | 0.7333/0.3519 | 0.4755 | 0.657 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.846 * score + 1.331)
- threshold: 0.92
- train: precision 0.8966, recall 0.2253

## Held-out

- precision 0.7754 / recall 0.2441 / f1 0.3713
- AUC 0.8797
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.92. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.5625 | 0.225 | 0.3214 | False |
| hindi | 205 | 93 | 0.7143 | 0.1075 | 0.1869 | False |
| korean | 205 | 125 | 0.7812 | 0.2 | 0.3185 | False |
| mandarin | 1646 | 109 | 0.5581 | 0.2202 | 0.3158 | False |
| spanish | 140 | 88 | 0.9286 | 0.1477 | 0.2549 | False |
| vietnamese | 196 | 139 | 0.9412 | 0.4604 | 0.6184 | True |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.8382 | 0.2445 | 0.3786 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 4665.8 ms, median warm 26.1 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ASI-arctic_a0082 /ð/ -> /d/ (p=0.9524) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0082.wav
- [FP] l2arctic-BWC-arctic_a0029 /ð/ -> /d/ (p=0.9818) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0029.wav
- [FP] l2arctic-BWC-arctic_a0031 /ð/ -> /d/ (p=0.9356) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0031.wav
- [FP] l2arctic-BWC-arctic_a0047 /ð/ -> /d/ (p=0.9781) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0047.wav
- [FP] l2arctic-EBVS-arctic_a0081 /ð/ -> /z/ (p=0.9239) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0081.wav
- [FP] l2arctic-HJK-arctic_a0029 /ð/ -> /d/ (p=0.9879) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0029.wav
- [FP] l2arctic-HJK-arctic_a0047 /ð/ -> /v/ (p=0.988) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0047.wav
- [FP] l2arctic-HJK-arctic_b0019 /ð/ -> /v/ (p=0.9677) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0019.wav
- [FP] l2arctic-HJK-arctic_b0106 /ð/ -> /d/ (p=0.9679) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0106.wav
- [FP] l2arctic-HKK-arctic_a0037 /ð/ -> /d/ (p=0.9415) label=correct l1=korean data/l2arctic/HKK/wav/arctic_a0037.wav
- [FP] l2arctic-HQTV-arctic_a0081 /ð/ -> /d/ (p=0.9918) label=correct l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0081.wav
- [FP] l2arctic-LXC-arctic_a0121 /ð/ -> /d/ (p=0.9678) label=correct l1=mandarin data/l2arctic/LXC/wav/arctic_a0121.wav
- [FP] l2arctic-NCC-arctic_a0041 /ð/ -> /z/ (p=0.9905) label=correct l1=mandarin data/l2arctic/NCC/wav/arctic_a0041.wav
- [FP] l2arctic-PNV-arctic_a0019 /ð/ -> /d/ (p=0.9534) label=correct l1=vietnamese data/l2arctic/PNV/wav/arctic_a0019.wav
- [FP] l2arctic-RRBI-arctic_a0095 /ð/ -> /z/ (p=0.9767) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_a0095.wav
- [TP] l2arctic-YDCK-arctic_b0201 /ð/ -> /d/ (p=0.9481) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_b0201.wav
- [TP] l2arctic-EBVS-arctic_a0254 /ð/ -> /d/ (p=0.9532) label=substituted l1=spanish data/l2arctic/EBVS/wav/arctic_a0254.wav
- [TP] l2arctic-RRBI-arctic_b0017 /ð/ -> /d/ (p=0.9844) label=substituted l1=hindi data/l2arctic/RRBI/wav/arctic_b0017.wav
- [TP] l2arctic-HQTV-arctic_b0251 /ð/ -> /d/ (p=0.9614) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0251.wav
- [TP] l2arctic-TLV-arctic_a0003 /ð/ -> /d/ (p=0.9365) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0003.wav
- [TP] l2arctic-TLV-arctic_b0469 /ð/ -> /d/ (p=0.9377) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_b0469.wav
- [TP] l2arctic-YDCK-arctic_a0131 /ð/ -> /d/ (p=0.9659) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_a0131.wav
- [TP] l2arctic-HQTV-arctic_a0029 /ð/ -> /d/ (p=0.97) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0029.wav
- [TP] l2arctic-YDCK-arctic_a0369 /ð/ -> /d/ (p=0.9628) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_a0369.wav
- [TP] l2arctic-TNI-arctic_b0158 /ð/ -> /d/ (p=0.9337) label=substituted l1=hindi data/l2arctic/TNI/wav/arctic_b0158.wav
- [TP] l2arctic-NJS-arctic_a0041 /ð/ -> /z/ (p=0.9932) label=substituted l1=spanish data/l2arctic/NJS/wav/arctic_a0041.wav
- [TP] l2arctic-TLV-arctic_b0355 /ð/ -> /d/ (p=0.9391) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_b0355.wav
- [TP] l2arctic-YKWK-arctic_a0037 /ð/ -> /d/ (p=0.9472) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_a0037.wav
- [TP] l2arctic-HQTV-arctic_a0088 /ð/ -> /d/ (p=0.9602) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0088.wav
- [TP] l2arctic-SKA-arctic_a0095 /ð/ -> /z/ (p=0.9898) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0095.wav
