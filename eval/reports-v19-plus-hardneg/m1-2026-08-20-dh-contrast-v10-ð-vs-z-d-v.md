# m1-2026-08-20-dh-contrast-v10-ð-vs-z-d-v

- model: dh-contrast-v10
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: pooled-bar-not-met**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.985 | pooled-bar-not-met | 0.9211/0.0253 | 0.6667/0.0269 | 0.0518 | 0.8642 |
| spike | 0.985 | pooled-bar-not-met | 0.9211/0.0253 | 0.6667/0.0269 | 0.0518 | 0.8642 |
| vote | 0.83 | SHIPPING BAR NOT MET | 0.831/0.5466 | 0.6491/0.5387 | 0.5888 | 0.7263 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.848 * score + 0.465)
- threshold: 0.985
- train: precision 0.9211, recall 0.0253

## Held-out

- precision 0.6667 / recall 0.0269 / f1 0.0518
- AUC 0.8642
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.985. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.7778 | 0.175 | 0.2857 | False |
| hindi | 205 | 93 | 1.0 | 0.0108 | 0.0213 | False |
| korean | 205 | 125 | 0.0 | 0.0 | 0.0 | False |
| mandarin | 1646 | 109 | 0.4545 | 0.0459 | 0.0833 | False |
| spanish | 140 | 88 | 0.0 | 0.0 | 0.0 | False |
| vietnamese | 196 | 139 | 1.0 | 0.0216 | 0.0423 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.8889 | 0.027 | 0.0524 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 3284.5 ms, median warm 29.1 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-SKA-arctic_a0025 /ð/ -> /z/ (p=0.9907) label=correct l1=arabic data/l2arctic/SKA/wav/arctic_a0025.wav
- [FP] l2arctic-SKA-arctic_a0461 /ð/ -> /z/ (p=0.9899) label=correct l1=arabic data/l2arctic/SKA/wav/arctic_a0461.wav
- [FP] so762-011090273 /ð/ -> /z/ (p=0.9858) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER1109/011090273.WAV
- [FP] so762-011350345 /ð/ -> /z/ (p=0.9891) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER1135/011350345.WAV
- [FP] so762-011560068 /ð/ -> /z/ (p=0.988) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER1156/011560068.WAV
- [FP] so762-028970146 /ð/ -> /z/ (p=0.9923) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER2897/028970146.WAV
- [FP] so762-028970153 /ð/ -> /z/ (p=0.9904) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER2897/028970153.WAV
- [FP] so762-096470008 /ð/ -> /z/ (p=0.9856) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER9647/096470008.WAV
- [TP] l2arctic-SKA-arctic_a0117 /ð/ -> /z/ (p=0.9918) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0117.wav
- [TP] l2arctic-TLV-arctic_a0095 /ð/ -> /v/ (p=0.9904) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0095.wav
- [TP] l2arctic-RRBI-arctic_a0071 /ð/ -> /z/ (p=0.99) label=substituted l1=hindi data/l2arctic/RRBI/wav/arctic_a0071.wav
- [TP] l2arctic-SKA-arctic_a0095 /ð/ -> /z/ (p=0.9894) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0095.wav
- [TP] l2arctic-TXHC-arctic_a0136 /ð/ -> /z/ (p=0.9893) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0136.wav
- [TP] l2arctic-TLV-arctic_a0459 /ð/ -> /z/ (p=0.9912) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0459.wav
- [TP] l2arctic-TXHC-arctic_a0041 /ð/ -> /z/ (p=0.9887) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0041.wav
- [TP] l2arctic-SKA-arctic_a0122 /ð/ -> /z/ (p=0.9909) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0122.wav
- [TP] l2arctic-ABA-arctic_a0500 /ð/ -> /z/ (p=0.9925) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0500.wav
- [TP] l2arctic-ABA-arctic_a0501 /ð/ -> /z/ (p=0.9874) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0501.wav
- [TP] l2arctic-TXHC-arctic_a0053 /ð/ -> /z/ (p=0.9898) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0053.wav
- [TP] l2arctic-YBAA-arctic_a0139 /ð/ -> /z/ (p=0.9882) label=substituted l1=arabic data/l2arctic/YBAA/wav/arctic_a0139.wav
- [TP] l2arctic-LXC-arctic_b0026 /ð/ -> /z/ (p=0.9937) label=substituted l1=mandarin data/l2arctic/LXC/wav/arctic_b0026.wav
- [TP] l2arctic-TXHC-arctic_a0037 /ð/ -> /z/ (p=0.9907) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0037.wav
- [TP] l2arctic-ABA-arctic_a0159 /ð/ -> /z/ (p=0.9886) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0159.wav
- [TP] l2arctic-HQTV-arctic_b0358 /ð/ -> /z/ (p=0.9936) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0358.wav
