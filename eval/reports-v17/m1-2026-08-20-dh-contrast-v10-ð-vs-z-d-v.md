# m1-2026-08-20-dh-contrast-v10-ð-vs-z-d-v

- model: dh-contrast-v10
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: SHIPPING BAR NOT MET**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.87 | SHIPPING BAR NOT MET | 0.8855/0.2289 | 0.6984/0.2222 | 0.3372 | 0.8648 |
| spike | 0.87 | SHIPPING BAR NOT MET | 0.8855/0.2289 | 0.6984/0.2222 | 0.3372 | 0.8648 |
| vote | 0.785 | SHIPPING BAR NOT MET | 0.7874/0.7061 | 0.6404/0.6566 | 0.6484 | 0.7738 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.637 * score + 0.209)
- threshold: 0.87
- train: precision 0.8855, recall 0.2289

## Held-out

- precision 0.6984 / recall 0.2222 / f1 0.3372
- AUC 0.8648
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.87. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.6316 | 0.3 | 0.4068 | False |
| hindi | 205 | 93 | 0.8333 | 0.2151 | 0.3419 | False |
| korean | 205 | 125 | 0.619 | 0.104 | 0.1781 | False |
| mandarin | 1646 | 109 | 0.4259 | 0.211 | 0.2822 | False |
| spanish | 140 | 88 | 0.8 | 0.1364 | 0.233 | False |
| vietnamese | 196 | 139 | 0.9286 | 0.3741 | 0.5333 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.8 | 0.2226 | 0.3483 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 4904.8 ms, median warm 29.0 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0060 /ð/ -> /v/ (p=0.8936) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0060.wav
- [FP] l2arctic-BWC-arctic_a0029 /ð/ -> /d/ (p=0.912) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0029.wav
- [FP] l2arctic-BWC-arctic_a0031 /ð/ -> /d/ (p=0.884) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0031.wav
- [FP] l2arctic-BWC-arctic_a0047 /ð/ -> /d/ (p=0.8885) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0047.wav
- [FP] l2arctic-BWC-arctic_a0047 /ð/ -> /d/ (p=0.8846) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0047.wav
- [FP] l2arctic-BWC-arctic_b0288 /ð/ -> /d/ (p=0.8995) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_b0288.wav
- [FP] l2arctic-EBVS-arctic_a0081 /ð/ -> /z/ (p=0.9414) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0081.wav
- [FP] l2arctic-HJK-arctic_a0029 /ð/ -> /d/ (p=0.9129) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0029.wav
- [FP] l2arctic-HJK-arctic_a0047 /ð/ -> /v/ (p=0.9493) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0047.wav
- [FP] l2arctic-HJK-arctic_a0091 /ð/ -> /d/ (p=0.8717) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0091.wav
- [FP] l2arctic-HJK-arctic_b0315 /ð/ -> /d/ (p=0.8713) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0315.wav
- [FP] l2arctic-HKK-arctic_a0112 /ð/ -> /d/ (p=0.8869) label=correct l1=korean data/l2arctic/HKK/wav/arctic_a0112.wav
- [FP] l2arctic-HQTV-arctic_a0081 /ð/ -> /d/ (p=0.9239) label=correct l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0081.wav
- [FP] l2arctic-MBMPS-arctic_a0041 /ð/ -> /v/ (p=0.8757) label=correct l1=spanish data/l2arctic/MBMPS/wav/arctic_a0041.wav
- [FP] l2arctic-NCC-arctic_a0041 /ð/ -> /z/ (p=0.9231) label=correct l1=mandarin data/l2arctic/NCC/wav/arctic_a0041.wav
- [TP] l2arctic-ERMS-arctic_a0159 /ð/ -> /d/ (p=0.8741) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0159.wav
- [TP] l2arctic-HQTV-arctic_b0201 /ð/ -> /d/ (p=0.8961) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0201.wav
- [TP] l2arctic-TLV-arctic_a0134 /ð/ -> /d/ (p=0.8773) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0134.wav
- [TP] l2arctic-HQTV-arctic_a0459 /ð/ -> /v/ (p=0.9337) label=deleted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0459.wav
- [TP] l2arctic-TNI-arctic_a0144 /ð/ -> /d/ (p=0.8896) label=substituted l1=hindi data/l2arctic/TNI/wav/arctic_a0144.wav
- [TP] l2arctic-BWC-arctic_b0400 /ð/ -> /d/ (p=0.8922) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_b0400.wav
- [TP] l2arctic-ABA-arctic_a0500 /ð/ -> /z/ (p=0.9314) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0500.wav
- [TP] l2arctic-TLV-arctic_b0469 /ð/ -> /d/ (p=0.8812) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_b0469.wav
- [TP] l2arctic-HQTV-arctic_a0013 /ð/ -> /d/ (p=0.9104) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0013.wav
- [TP] l2arctic-PNV-arctic_b0140 /ð/ -> /d/ (p=0.894) label=substituted l1=vietnamese data/l2arctic/PNV/wav/arctic_b0140.wav
- [TP] l2arctic-BWC-arctic_a0081 /ð/ -> /d/ (p=0.9152) label=deleted l1=mandarin data/l2arctic/BWC/wav/arctic_a0081.wav
- [TP] l2arctic-BWC-arctic_a0134 /ð/ -> /d/ (p=0.9201) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_a0134.wav
- [TP] l2arctic-YDCK-arctic_a0029 /ð/ -> /d/ (p=0.9) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_a0029.wav
- [TP] l2arctic-HQTV-arctic_a0078 /ð/ -> /d/ (p=0.8931) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0078.wav
- [TP] l2arctic-THV-arctic_b0189 /ð/ -> /d/ (p=0.8726) label=substituted l1=vietnamese data/l2arctic/THV/wav/arctic_b0189.wav
