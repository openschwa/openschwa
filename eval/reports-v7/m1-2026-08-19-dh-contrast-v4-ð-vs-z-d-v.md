# m1-2026-08-19-dh-contrast-v4-ð-vs-z-d-v

- model: dh-contrast-v4
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: SHIPPING BAR NOT MET**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.97 | SHIPPING BAR NOT MET | 0.8919/0.0238 | 0.6667/0.0269 | 0.0518 | 0.901 |
| spike | 0.97 | SHIPPING BAR NOT MET | 0.8919/0.0238 | 0.6667/0.0269 | 0.0518 | 0.901 |
| vote | 0.845 | SHIPPING BAR NOT MET | 0.8479/0.6238 | 0.7082/0.5926 | 0.6453 | 0.7602 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.738 * score + 0.472)
- threshold: 0.97
- train: precision 0.8919, recall 0.0238

## Held-out

- precision 0.6667 / recall 0.0269 / f1 0.0518
- AUC 0.901
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.97. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.6667 | 0.1 | 0.1739 | False |
| hindi | 205 | 93 | 0.6667 | 0.0215 | 0.0417 | False |
| korean | 205 | 125 | 0.0 | 0.0 | 0.0 | False |
| mandarin | 1646 | 109 | 0.5833 | 0.0642 | 0.1157 | False |
| spanish | 140 | 88 | 0.0 | 0.0 | 0.0 | False |
| vietnamese | 196 | 139 | 1.0 | 0.0216 | 0.0423 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.8421 | 0.027 | 0.0523 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 4625.7 ms, median warm 28.7 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-RRBI-arctic_a0095 /ð/ -> /z/ (p=0.9708) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_a0095.wav
- [FP] l2arctic-SKA-arctic_a0025 /ð/ -> /z/ (p=0.9752) label=correct l1=arabic data/l2arctic/SKA/wav/arctic_a0025.wav
- [FP] l2arctic-SKA-arctic_a0461 /ð/ -> /z/ (p=0.9705) label=correct l1=arabic data/l2arctic/SKA/wav/arctic_a0461.wav
- [FP] so762-005630169 /ð/ -> /z/ (p=0.9799) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER0563/005630169.WAV
- [FP] so762-011350290 /ð/ -> /z/ (p=0.9755) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER1135/011350290.WAV
- [FP] so762-028970146 /ð/ -> /z/ (p=0.9771) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER2897/028970146.WAV
- [FP] so762-028970153 /ð/ -> /z/ (p=0.9795) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER2897/028970153.WAV
- [FP] so762-096470008 /ð/ -> /z/ (p=0.9788) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER9647/096470008.WAV
- [TP] l2arctic-SKA-arctic_b0534 /ð/ -> /z/ (p=0.9711) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_b0534.wav
- [TP] l2arctic-TLV-arctic_a0459 /ð/ -> /z/ (p=0.9801) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0459.wav
- [TP] l2arctic-RRBI-arctic_a0071 /ð/ -> /z/ (p=0.9715) label=substituted l1=hindi data/l2arctic/RRBI/wav/arctic_a0071.wav
- [TP] l2arctic-SKA-arctic_a0117 /ð/ -> /z/ (p=0.9775) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0117.wav
- [TP] l2arctic-TXHC-arctic_a0136 /ð/ -> /z/ (p=0.9735) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0136.wav
- [TP] l2arctic-TNI-arctic_a0246 /ð/ -> /z/ (p=0.97) label=substituted l1=hindi data/l2arctic/TNI/wav/arctic_a0246.wav
- [TP] l2arctic-TXHC-arctic_a0041 /ð/ -> /z/ (p=0.9769) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0041.wav
- [TP] l2arctic-TLV-arctic_a0095 /ð/ -> /v/ (p=0.9752) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0095.wav
- [TP] l2arctic-ABA-arctic_a0500 /ð/ -> /z/ (p=0.9776) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0500.wav
- [TP] l2arctic-HQTV-arctic_b0358 /ð/ -> /z/ (p=0.981) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0358.wav
- [TP] l2arctic-TXHC-arctic_a0053 /ð/ -> /z/ (p=0.9717) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0053.wav
- [TP] l2arctic-TXHC-arctic_a0536 /ð/ -> /z/ (p=0.9768) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0536.wav
- [TP] l2arctic-LXC-arctic_b0026 /ð/ -> /z/ (p=0.9788) label=substituted l1=mandarin data/l2arctic/LXC/wav/arctic_b0026.wav
- [TP] l2arctic-TXHC-arctic_a0037 /ð/ -> /z/ (p=0.9739) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0037.wav
- [TP] l2arctic-ABA-arctic_a0159 /ð/ -> /z/ (p=0.9817) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0159.wav
- [TP] l2arctic-LXC-arctic_a0081 /ð/ -> /v/ (p=0.9764) label=deleted l1=mandarin data/l2arctic/LXC/wav/arctic_a0081.wav
