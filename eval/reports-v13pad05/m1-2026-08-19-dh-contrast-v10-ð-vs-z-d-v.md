# m1-2026-08-19-dh-contrast-v10-ð-vs-z-d-v

- model: dh-contrast-v10
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: SHIPPING BAR NOT MET**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.815 | SHIPPING BAR NOT MET | 0.8298/0.1971 | 0.615/0.1936 | 0.2945 | 0.8609 |
| spike | 0.99 | pooled-bar-not-met | 1.0/0.013 | 0.75/0.0152 | 0.0297 | 0.8609 |
| vote | 0.665 | SHIPPING BAR NOT MET | 0.6676/0.6931 | 0.571/0.6768 | 0.6194 | 0.7633 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.448 * score + -0.525)
- threshold: 0.815
- train: precision 0.8298, recall 0.1971

## Held-out

- precision 0.615 / recall 0.1936 / f1 0.2945
- AUC 0.8609
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.815. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.72 | 0.45 | 0.5538 | False |
| hindi | 205 | 93 | 0.75 | 0.0645 | 0.1188 | False |
| korean | 205 | 125 | 0.6486 | 0.192 | 0.2963 | False |
| mandarin | 1646 | 109 | 0.2364 | 0.1193 | 0.1585 | False |
| spanish | 140 | 88 | 0.75 | 0.1364 | 0.2308 | False |
| vietnamese | 196 | 139 | 0.913 | 0.3022 | 0.4541 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.7703 | 0.1922 | 0.3077 |
| so762 | 1495 | 1 | 0.0256 | 1.0 | 0.05 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 4547.9 ms, median warm 27.6 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0032 /ð/ -> /v/ (p=0.9181) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0032.wav
- [FP] l2arctic-BWC-arctic_a0029 /ð/ -> /d/ (p=0.8899) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0029.wav
- [FP] l2arctic-BWC-arctic_a0047 /ð/ -> /d/ (p=0.8335) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0047.wav
- [FP] l2arctic-BWC-arctic_b0288 /ð/ -> /d/ (p=0.8899) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_b0288.wav
- [FP] l2arctic-EBVS-arctic_a0023 /ð/ -> /d/ (p=0.8321) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0023.wav
- [FP] l2arctic-EBVS-arctic_a0299 /ð/ -> /d/ (p=0.8389) label=correct l1=spanish data/l2arctic/EBVS/wav/arctic_a0299.wav
- [FP] l2arctic-HJK-arctic_a0024 /ð/ -> /d/ (p=0.8405) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0024.wav
- [FP] l2arctic-HJK-arctic_a0029 /ð/ -> /d/ (p=0.849) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0029.wav
- [FP] l2arctic-HJK-arctic_a0047 /ð/ -> /v/ (p=1.0) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0047.wav
- [FP] l2arctic-HJK-arctic_a0092 /ð/ -> /d/ (p=0.9052) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0092.wav
- [FP] l2arctic-HJK-arctic_a0260 /ð/ -> /d/ (p=0.8535) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0260.wav
- [FP] l2arctic-HJK-arctic_a0542 /ð/ -> /d/ (p=0.9205) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0542.wav
- [FP] l2arctic-HJK-arctic_b0143 /ð/ -> /d/ (p=0.8472) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0143.wav
- [FP] l2arctic-HJK-arctic_b0201 /ð/ -> /d/ (p=0.8438) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0201.wav
- [FP] l2arctic-HJK-arctic_b0315 /ð/ -> /d/ (p=0.8604) label=correct l1=korean data/l2arctic/HJK/wav/arctic_b0315.wav
- [TP] l2arctic-SKA-arctic_b0492 /ð/ -> /z/ (p=0.9734) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_b0492.wav
- [TP] l2arctic-YKWK-arctic_a0101 /ð/ -> /d/ (p=0.8446) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_a0101.wav
- [TP] so762-030070019 /ð/ -> /z/ (p=0.8761) label=substituted l1=mandarin data/speechocean762/WAVE/SPEAKER3007/030070019.WAV
- [TP] l2arctic-PNV-arctic_a0047 /ð/ -> /d/ (p=0.8481) label=substituted l1=vietnamese data/l2arctic/PNV/wav/arctic_a0047.wav
- [TP] l2arctic-SKA-arctic_b0433 /ð/ -> /z/ (p=0.918) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_b0433.wav
- [TP] l2arctic-HQTV-arctic_a0013 /ð/ -> /d/ (p=0.8321) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0013.wav
- [TP] l2arctic-SKA-arctic_a0501 /ð/ -> /z/ (p=1.0) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0501.wav
- [TP] l2arctic-TLV-arctic_a0073 /ð/ -> /d/ (p=0.8962) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0073.wav
- [TP] l2arctic-TLV-arctic_a0089 /ð/ -> /d/ (p=0.8841) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0089.wav
- [TP] l2arctic-SKA-arctic_a0117 /ð/ -> /z/ (p=1.0) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0117.wav
- [TP] l2arctic-HQTV-arctic_b0201 /ð/ -> /d/ (p=0.9015) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0201.wav
- [TP] l2arctic-TLV-arctic_a0019 /ð/ -> /v/ (p=0.8294) label=deleted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0019.wav
- [TP] l2arctic-EBVS-arctic_a0089 /ð/ -> /d/ (p=0.8217) label=substituted l1=spanish data/l2arctic/EBVS/wav/arctic_a0089.wav
- [TP] l2arctic-TXHC-arctic_a0041 /ð/ -> /z/ (p=1.0) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0041.wav
- [TP] l2arctic-NJS-arctic_a0090 /ð/ -> /d/ (p=0.8381) label=substituted l1=spanish data/l2arctic/NJS/wav/arctic_a0090.wav
