# m1-2026-08-19-tinyschwa-v1-ð-vs-z-d-v

- model: tinyschwa-v1
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 4070 train / 2608 held-out
- **status: SHIPPING BAR NOT MET**
- **shipping variant: mean**

## Score variants (same frames, three aggregations)

| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.8 | SHIPPING BAR NOT MET | 0.7893/0.1379 | 0.648/0.1364 | 0.2253 | 0.8657 |
| spike | 0.775 | SHIPPING BAR NOT MET | 0.79/0.1819 | 0.6627/0.1886 | 0.2936 | 0.8657 |
| vote | 0.61 | SHIPPING BAR NOT MET | 0.6142/0.7668 | 0.5376/0.7694 | 0.633 | 0.7871 |
| gop | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5357 |

## Operating point (train split)

- Platt: p = sigmoid(0.580 * score + -0.584)
- threshold: 0.8
- train: precision 0.7893, recall 0.1379

## Held-out

- precision 0.648 / recall 0.1364 / f1 0.2253
- AUC 0.8657
- verdicts 2607, refused 1 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.8. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 216 | 40 | 0.8 | 0.2 | 0.32 | False |
| hindi | 205 | 93 | 0.7143 | 0.0538 | 0.1 | False |
| korean | 205 | 125 | 0.9286 | 0.104 | 0.1871 | False |
| mandarin | 1646 | 109 | 0.2391 | 0.1009 | 0.1419 | False |
| spanish | 140 | 88 | 0.9091 | 0.1136 | 0.202 | False |
| vietnamese | 196 | 139 | 0.9189 | 0.2446 | 0.3864 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 |
|---|---|---|---|---|---|
| l2arctic | 1113 | 593 | 0.871 | 0.1366 | 0.2362 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 |

## Alignment sanity

- statuses: {'ok': 6672, 'low_confidence': 6}
- mean alignment confidence (ok): 0.9047

## Latency

- cold 8040.1 ms, median warm 34.4 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-BWC-arctic_b0288 /ð/ -> /d/ (p=0.8549) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_b0288.wav
- [FP] l2arctic-HJK-arctic_a0047 /ð/ -> /v/ (p=0.9204) label=correct l1=korean data/l2arctic/HJK/wav/arctic_a0047.wav
- [FP] l2arctic-HQTV-arctic_a0081 /ð/ -> /d/ (p=0.8517) label=correct l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0081.wav
- [FP] l2arctic-MBMPS-arctic_a0041 /ð/ -> /v/ (p=0.8847) label=correct l1=spanish data/l2arctic/MBMPS/wav/arctic_a0041.wav
- [FP] l2arctic-NCC-arctic_a0041 /ð/ -> /z/ (p=0.851) label=correct l1=mandarin data/l2arctic/NCC/wav/arctic_a0041.wav
- [FP] l2arctic-PNV-arctic_a0019 /ð/ -> /d/ (p=0.8049) label=correct l1=vietnamese data/l2arctic/PNV/wav/arctic_a0019.wav
- [FP] l2arctic-RRBI-arctic_a0095 /ð/ -> /z/ (p=0.8851) label=correct l1=hindi data/l2arctic/RRBI/wav/arctic_a0095.wav
- [FP] l2arctic-SKA-arctic_a0047 /ð/ -> /z/ (p=0.8395) label=correct l1=arabic data/l2arctic/SKA/wav/arctic_a0047.wav
- [FP] l2arctic-SKA-arctic_a0064 /ð/ -> /z/ (p=0.8733) label=correct l1=arabic data/l2arctic/SKA/wav/arctic_a0064.wav
- [FP] l2arctic-THV-arctic_a0519 /ð/ -> /v/ (p=0.9149) label=correct l1=vietnamese data/l2arctic/THV/wav/arctic_a0519.wav
- [FP] l2arctic-TNI-arctic_a0139 /ð/ -> /d/ (p=0.8054) label=correct l1=hindi data/l2arctic/TNI/wav/arctic_a0139.wav
- [FP] l2arctic-TXHC-arctic_a0094 /ð/ -> /z/ (p=0.9011) label=correct l1=mandarin data/l2arctic/TXHC/wav/arctic_a0094.wav
- [FP] so762-001220013 /ð/ -> /v/ (p=0.8984) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER0122/001220013.WAV
- [FP] so762-005630122 /ð/ -> /z/ (p=0.8661) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER0563/005630122.WAV
- [FP] so762-005670289 /ð/ -> /d/ (p=0.8104) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER0567/005670289.WAV
- [TP] l2arctic-TLV-arctic_a0136 /ð/ -> /d/ (p=0.8334) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0136.wav
- [TP] l2arctic-HQTV-arctic_a0036 /ð/ -> /d/ (p=0.8363) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0036.wav
- [TP] l2arctic-SKA-arctic_a0501 /ð/ -> /z/ (p=0.8414) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_a0501.wav
- [TP] l2arctic-TLV-arctic_a0019 /ð/ -> /v/ (p=0.8889) label=deleted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0019.wav
- [TP] l2arctic-LXC-arctic_b0026 /ð/ -> /z/ (p=0.9588) label=substituted l1=mandarin data/l2arctic/LXC/wav/arctic_b0026.wav
- [TP] l2arctic-HQTV-arctic_a0013 /ð/ -> /d/ (p=0.8311) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_a0013.wav
- [TP] l2arctic-TLV-arctic_a0073 /ð/ -> /d/ (p=0.8128) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_a0073.wav
- [TP] l2arctic-ERMS-arctic_a0029 /ð/ -> /v/ (p=0.9135) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0029.wav
- [TP] l2arctic-TLV-arctic_b0355 /ð/ -> /d/ (p=0.8279) label=substituted l1=vietnamese data/l2arctic/TLV/wav/arctic_b0355.wav
- [TP] l2arctic-LXC-arctic_a0081 /ð/ -> /v/ (p=0.9244) label=deleted l1=mandarin data/l2arctic/LXC/wav/arctic_a0081.wav
- [TP] l2arctic-YDCK-arctic_b0225 /ð/ -> /d/ (p=0.8017) label=substituted l1=korean data/l2arctic/YDCK/wav/arctic_b0225.wav
- [TP] l2arctic-SKA-arctic_b0534 /ð/ -> /z/ (p=0.8843) label=substituted l1=arabic data/l2arctic/SKA/wav/arctic_b0534.wav
- [TP] l2arctic-NJS-arctic_a0039 /ð/ -> /d/ (p=0.8517) label=substituted l1=spanish data/l2arctic/NJS/wav/arctic_a0039.wav
- [TP] l2arctic-TXHC-arctic_a0053 /ð/ -> /z/ (p=0.8279) label=substituted l1=mandarin data/l2arctic/TXHC/wav/arctic_a0053.wav
- [TP] l2arctic-HQTV-arctic_b0106 /ð/ -> /d/ (p=0.8463) label=substituted l1=vietnamese data/l2arctic/HQTV/wav/arctic_b0106.wav
