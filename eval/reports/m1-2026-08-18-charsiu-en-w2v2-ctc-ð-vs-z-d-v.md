# m1-2026-08-18-charsiu-en-w2v2-ctc-ð-vs-z-d-v

- model: charsiu-en-w2v2-ctc
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: SHIPPING BAR NOT MET**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.545 | SHIPPING BAR NOT MET | 0.2453/0.0094 | 0.1892/0.0118 | 0.0222 | 0.5829 |
| spike | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.581 |
| vote | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.533 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(-0.039 * score + -0.881)
- threshold: 0.545
- train: precision 0.2453, recall 0.0094

## Held-out

- precision 0.1892 / recall 0.0118 / f1 0.0222
- AUC 0.5829
- verdicts 2605, refused 3 (audio-problem 0)

## Per L1 (held-out)

| l1 | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.0 | 0.0 | 0.0 |
| hindi | 205 | 93 | 0.2222 | 0.0215 | 0.0392 |
| korean | 205 | 125 | 1.0 | 0.032 | 0.062 |
| mandarin | 1646 | 109 | 0.0769 | 0.0092 | 0.0164 |
| spanish | 140 | 88 | 0.0 | 0.0 | 0.0 |
| vietnamese | 196 | 139 | 0.0 | 0.0 | 0.0 |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.28 | 0.0118 | 0.0227 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6667, 'low_confidence': 11}
- mean alignment confidence (ok): 0.905

## Latency

- cold 0.0 ms, median warm 135.4 ms
- download size 0.38 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0058 /ð/ -> /z/ (p=0.546) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0058.wav
- [FP] l2arctic-ABA-arctic_a0060 /ð/ -> /z/ (p=0.546) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0060.wav
- [FP] l2arctic-ABA-arctic_a0461 /ð/ -> /z/ (p=0.546) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0461.wav
- [FP] l2arctic-NJS-arctic_a0029 /ð/ -> /z/ (p=0.546) label=correct l1=spanish data/l2arctic/NJS/wav/arctic_a0029.wav
- [FP] l2arctic-NJS-arctic_a0060 /ð/ -> /z/ (p=0.546) label=correct l1=spanish data/l2arctic/NJS/wav/arctic_a0060.wav
- [FP] l2arctic-PNV-arctic_b0074 /ð/ -> /z/ (p=0.546) label=correct l1=vietnamese data/l2arctic/PNV/wav/arctic_b0074.wav
- [FP] l2arctic-RRBI-arctic_a0058 /ð/ -> /z/ (p=0.546) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_a0058.wav
- [FP] l2arctic-RRBI-arctic_b0197 /ð/ -> /z/ (p=0.546) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_b0197.wav
- [FP] l2arctic-RRBI-arctic_b0371 /ð/ -> /z/ (p=0.546) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_b0371.wav
- [FP] l2arctic-SKA-arctic_a0461 /ð/ -> /z/ (p=0.546) label=correct l1=arabic data/l2arctic/SKA/wav/arctic_a0461.wav
- [FP] l2arctic-SKA-arctic_a0461 /ð/ -> /z/ (p=0.546) label=correct l1=arabic data/l2arctic/SKA/wav/arctic_a0461.wav
- [FP] l2arctic-SVBI-arctic_a0292 /ð/ -> /z/ (p=0.546) label=correct l1=hindi data/l2arctic/SVBI/wav/arctic_a0292.wav
- [FP] l2arctic-SVBI-arctic_a0292 /ð/ -> /z/ (p=0.546) label=correct l1=hindi data/l2arctic/SVBI/wav/arctic_a0292.wav
- [FP] l2arctic-SVBI-arctic_b0073 /ð/ -> /z/ (p=0.546) label=correct l1=hindi data/l2arctic/SVBI/wav/arctic_b0073.wav
- [FP] l2arctic-SVBI-arctic_b0197 /ð/ -> /z/ (p=0.546) label=correct l1=hindi data/l2arctic/SVBI/wav/arctic_b0197.wav
- [TP] l2arctic-HJK-arctic_a0088 /ð/ -> /z/ (p=0.546) label=substituted l1=korean data/l2arctic/HJK/wav/arctic_a0088.wav
- [TP] l2arctic-LXC-arctic_b0075 /ð/ -> /z/ (p=0.546) label=substituted l1=mandarin data/l2arctic/LXC/wav/arctic_b0075.wav
- [TP] l2arctic-SVBI-arctic_a0058 /ð/ -> /z/ (p=0.546) label=substituted l1=hindi data/l2arctic/SVBI/wav/arctic_a0058.wav
- [TP] l2arctic-HJK-arctic_a0542 /ð/ -> /z/ (p=0.546) label=substituted l1=korean data/l2arctic/HJK/wav/arctic_a0542.wav
- [TP] l2arctic-YDCK-arctic_b0143 /ð/ -> /z/ (p=0.546) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_b0143.wav
- [TP] l2arctic-ASI-arctic_a0075 /ð/ -> /z/ (p=0.546) label=substituted l1=hindi data/l2arctic/ASI/wav/arctic_a0075.wav
- [TP] l2arctic-YDCK-arctic_a0091 /ð/ -> /z/ (p=0.546) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_a0091.wav
