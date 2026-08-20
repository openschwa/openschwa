# m1-2026-08-20-tinyschwa-v1-ð-vs-z-d-v

- model: tinyschwa-v1
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 0 train / 1312 cal / 2410 held-out
- **status: ok** (the mirror; judge: SHIPPING BAR NOT MET)

## Mirror - what the ear heard (shipped line)

- Platt: p = sigmoid(0.630 * hearing_score + -0.143)  [P(heard == realized)]
- threshold: 0.825
- cal: accuracy 0.9056 / coverage 0.2188 (cal status ok)
- **held-out: accuracy 0.9528 / coverage 0.4228** / answer-rate 0.4233
- raw top-1 hearing accuracy (no gating): 0.7875
- confident reports 1019 of 2410 tokens (scored 2407; unscorable 1)

### realized x heard (confident reports)

| realized \ heard | /d/ | /v/ | /z/ | /ð/ |
|---|---|---|---|---|
| /ð/ | 2 | 10 | 29 | 964 |
| /d/ | 1 | 2 | 0 | 2 |
| /s/ | 0 | 0 | 1 | 0 |
| /z/ | 0 | 0 | 5 | 0 |
| /θ/ | 0 | 0 | 1 | 0 |
| deleted | 1 | 0 | 0 | 0 |

### Mirror per L1 (held-out)

Fairness audit of the single shipped line: every group is heard at
the same global operating point 0.825. Informational
only - it never gates the bar.

| l1 | tokens | confident | accuracy | coverage | fair |
|---|---|---|---|---|---|
| arabic | 150 | 76 | 0.9605 | 0.5067 | True |
| hindi | 158 | 3 | 0.6667 | 0.019 | False |
| korean | 161 | 3 | 0.6667 | 0.0186 | False |
| mandarin | 1640 | 924 | 0.9588 | 0.5634 | True |
| spanish | 152 | 5 | 0.2 | 0.0329 | False |
| vietnamese | 149 | 8 | 0.875 | 0.0537 | False |

### Mirror spot-check (confident reports, mishearings first)

- [WRONG] l2arctic-ABA-arctic_a0003 heard /v/, realized /ð/ (p=0.8426) l1=arabic data/l2arctic/ABA/wav/arctic_a0003.wav
- [WRONG] l2arctic-ABA-arctic_a0117 heard /z/, realized /ð/ (p=0.8676) l1=arabic data/l2arctic/ABA/wav/arctic_a0117.wav
- [WRONG] l2arctic-ABA-arctic_a0159 heard /z/, realized /s/ (p=0.8995) l1=arabic data/l2arctic/ABA/wav/arctic_a0159.wav
- [WRONG] l2arctic-ASI-arctic_a0047 heard /v/, realized /d/ (p=0.9028) l1=hindi data/l2arctic/ASI/wav/arctic_a0047.wav
- [WRONG] l2arctic-BWC-arctic_a0051 heard /z/, realized /θ/ (p=0.8535) l1=mandarin data/l2arctic/BWC/wav/arctic_a0051.wav
- [WRONG] l2arctic-BWC-arctic_a0053 heard /d/, realized /ð/ (p=0.8498) l1=mandarin data/l2arctic/BWC/wav/arctic_a0053.wav
- [WRONG] l2arctic-BWC-arctic_b0288 heard /d/, realized /ð/ (p=0.8333) l1=mandarin data/l2arctic/BWC/wav/arctic_b0288.wav
- [WRONG] l2arctic-ERMS-arctic_a0029 heard /v/, realized /d/ (p=0.9221) l1=spanish data/l2arctic/ERMS/wav/arctic_a0029.wav
- [WRONG] l2arctic-ERMS-arctic_a0037 heard /z/, realized /ð/ (p=0.8368) l1=spanish data/l2arctic/ERMS/wav/arctic_a0037.wav
- [WRONG] l2arctic-ERMS-arctic_a0176 heard /ð/, realized /d/ (p=0.8555) l1=spanish data/l2arctic/ERMS/wav/arctic_a0176.wav
- [WRONG] l2arctic-ERMS-arctic_b0150 heard /ð/, realized /d/ (p=0.8334) l1=spanish data/l2arctic/ERMS/wav/arctic_b0150.wav
- [WRONG] l2arctic-THV-arctic_a0519 heard /v/, realized /ð/ (p=0.9157) l1=vietnamese data/l2arctic/THV/wav/arctic_a0519.wav
- [WRONG] l2arctic-YKWK-arctic_a0369 heard /d/, realized /∅/ (p=0.8253) l1=korean data/l2arctic/YKWK/wav/arctic_a0369.wav
- [WRONG] so762-001220013 heard /v/, realized /ð/ (p=0.8857) l1=mandarin data/speechocean762/WAVE/SPEAKER0122/001220013.WAV
- [WRONG] so762-005630122 heard /z/, realized /ð/ (p=0.8701) l1=mandarin data/speechocean762/WAVE/SPEAKER0563/005630122.WAV
- [right] so762-014650011 heard /ð/, realized /ð/ (p=0.9051) l1=mandarin data/speechocean762/WAVE/SPEAKER1465/014650011.WAV
- [right] l2arctic-ABA-arctic_a0461 heard /ð/, realized /ð/ (p=0.9036) l1=arabic data/l2arctic/ABA/wav/arctic_a0461.wav
- [right] so762-030140092 heard /ð/, realized /ð/ (p=0.8281) l1=mandarin data/speechocean762/WAVE/SPEAKER3014/030140092.WAV
- [right] so762-005670282 heard /ð/, realized /ð/ (p=0.9186) l1=mandarin data/speechocean762/WAVE/SPEAKER0567/005670282.WAV
- [right] so762-010990276 heard /ð/, realized /ð/ (p=0.8955) l1=mandarin data/speechocean762/WAVE/SPEAKER1099/010990276.WAV
- [right] so762-069020059 heard /ð/, realized /ð/ (p=0.9141) l1=mandarin data/speechocean762/WAVE/SPEAKER6902/069020059.WAV
- [right] so762-001330097 heard /ð/, realized /ð/ (p=0.9104) l1=mandarin data/speechocean762/WAVE/SPEAKER0133/001330097.WAV
- [right] so762-020070132 heard /ð/, realized /ð/ (p=0.887) l1=mandarin data/speechocean762/WAVE/SPEAKER2007/020070132.WAV
- [right] so762-022520067 heard /ð/, realized /ð/ (p=0.8351) l1=mandarin data/speechocean762/WAVE/SPEAKER2252/022520067.WAV
- [right] so762-011810222 heard /ð/, realized /ð/ (p=0.9208) l1=mandarin data/speechocean762/WAVE/SPEAKER1181/011810222.WAV
- [right] l2arctic-ABA-arctic_a0136 heard /ð/, realized /ð/ (p=0.8332) l1=arabic data/l2arctic/ABA/wav/arctic_a0136.wav
- [right] so762-023550167 heard /ð/, realized /ð/ (p=0.8808) l1=mandarin data/speechocean762/WAVE/SPEAKER2355/023550167.WAV
- [right] so762-050720003 heard /ð/, realized /ð/ (p=0.8694) l1=mandarin data/speechocean762/WAVE/SPEAKER5072/050720003.WAV
- [right] so762-060990121 heard /ð/, realized /ð/ (p=0.901) l1=mandarin data/speechocean762/WAVE/SPEAKER6099/060990121.WAV
- [right] so762-007360299 heard /ð/, realized /ð/ (p=0.8838) l1=mandarin data/speechocean762/WAVE/SPEAKER0736/007360299.WAV

## Judge variants (research archive - parked by the mirror pivot)

| variant | threshold | status | cal P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.85 | SHIPPING BAR NOT MET | 0.8615/0.1016 | 0.3673/0.0345 | 0.063 | 0.866 |
| spike | 0.85 | SHIPPING BAR NOT MET | 0.8615/0.1016 | 0.3673/0.0345 | 0.063 | 0.866 |
| vote | 0.66 | SHIPPING BAR NOT MET | 0.6637/0.8058 | 0.537/0.7222 | 0.616 | 0.7749 |
| gop | 0.59 | pooled-bar-not-met | 1.0/0.0018 | 0.0/0.0 | 0.0 | 0.4362 |

## Judge operating point (calibration split, research archive)

- Platt: p = sigmoid(0.593 * score + -0.458)
- threshold: 0.85
- cal: precision 0.8615, recall 0.1016
- cal->test precision gap: 0.4942

## Judge held-out (research archive)

- precision 0.3673 / recall 0.0345 / f1 0.063
- AUC 0.866
- verdicts 2407, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.85. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 150 | 20 | 0.8333 | 0.25 | 0.3846 | False |
| hindi | 158 | 85 | 1.0 | 0.0118 | 0.0233 | False |
| korean | 161 | 102 | 1.0 | 0.0196 | 0.0385 | False |
| mandarin | 1640 | 111 | 0.1471 | 0.045 | 0.069 | False |
| spanish | 152 | 122 | 1.0 | 0.0164 | 0.0323 | False |
| vietnamese | 149 | 82 | 0.75 | 0.0366 | 0.0698 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 | AUC |
|---|---|---|---|---|---|---|---|
| l2arctic | 915 | 521 | 0.8182 | 0.0345 | 0.0663 | 0.7477 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 | 0.8699 |

## Alignment sanity

- statuses: {'ok': 3718, 'low_confidence': 4}
- mean alignment confidence (ok): 0.9001

## Latency

- cold 10734.4 ms, median warm 330.8 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0117 /ð/ -> /z/ (p=0.8667) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0117.wav
- [FP] l2arctic-BWC-arctic_a0053 /ð/ -> /d/ (p=0.8606) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_a0053.wav
- [FP] l2arctic-BWC-arctic_b0288 /ð/ -> /d/ (p=0.8756) label=correct l1=mandarin data/l2arctic/BWC/wav/arctic_b0288.wav
- [FP] l2arctic-THV-arctic_a0519 /ð/ -> /v/ (p=0.9286) label=correct l1=vietnamese data/l2arctic/THV/wav/arctic_a0519.wav
- [FP] so762-001220013 /ð/ -> /v/ (p=0.9141) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER0122/001220013.WAV
- [FP] so762-005630122 /ð/ -> /z/ (p=0.8856) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER0563/005630122.WAV
- [FP] so762-007360374 /ð/ -> /z/ (p=0.8995) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER0736/007360374.WAV
- [FP] so762-011350257 /ð/ -> /v/ (p=0.9385) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER1135/011350257.WAV
- [FP] so762-014200339 /ð/ -> /z/ (p=0.9448) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER1420/014200339.WAV
- [FP] so762-023270321 /ð/ -> /z/ (p=0.9222) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER2327/023270321.WAV
- [FP] so762-024480188 /ð/ -> /z/ (p=0.8636) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER2448/024480188.WAV
- [FP] so762-024480292 /ð/ -> /z/ (p=0.9357) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER2448/024480292.WAV
- [FP] so762-027400171 /ð/ -> /z/ (p=0.8636) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER2740/027400171.WAV
- [FP] so762-028970146 /ð/ -> /z/ (p=0.927) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER2897/028970146.WAV
- [FP] so762-034120111 /ð/ -> /z/ (p=0.8855) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER3412/034120111.WAV
- [TP] l2arctic-THV-arctic_a0030 /ð/ -> /d/ (p=0.8587) label=substituted l1=vietnamese data/l2arctic/THV/wav/arctic_a0030.wav
- [TP] l2arctic-ERMS-arctic_a0071 /ð/ -> /z/ (p=0.9094) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0071.wav
- [TP] l2arctic-ABA-arctic_b0492 /ð/ -> /z/ (p=0.9077) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_b0492.wav
- [TP] l2arctic-BWC-arctic_a0081 /ð/ -> /v/ (p=0.8835) label=deleted l1=mandarin data/l2arctic/BWC/wav/arctic_a0081.wav
- [TP] l2arctic-ASI-arctic_a0047 /ð/ -> /v/ (p=0.8962) label=substituted l1=hindi data/l2arctic/ASI/wav/arctic_a0047.wav
- [TP] l2arctic-BWC-arctic_a0051 /ð/ -> /z/ (p=0.9021) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_a0051.wav
- [TP] l2arctic-YKWK-arctic_a0037 /ð/ -> /z/ (p=0.9154) label=substituted l1=korean data/l2arctic/YKWK/wav/arctic_a0037.wav
- [TP] l2arctic-BWC-arctic_a0041 /ð/ -> /z/ (p=0.8694) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_a0041.wav
- [TP] l2arctic-BWC-arctic_a0108 /ð/ -> /d/ (p=0.8563) label=substituted l1=mandarin data/l2arctic/BWC/wav/arctic_a0108.wav
- [TP] l2arctic-THV-arctic_a0091 /ð/ -> /d/ (p=0.8805) label=substituted l1=vietnamese data/l2arctic/THV/wav/arctic_a0091.wav
- [TP] l2arctic-ABA-arctic_a0159 /ð/ -> /z/ (p=0.9381) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0159.wav
- [TP] l2arctic-ERMS-arctic_a0029 /ð/ -> /v/ (p=0.9274) label=substituted l1=spanish data/l2arctic/ERMS/wav/arctic_a0029.wav
- [TP] l2arctic-ABA-arctic_a0500 /ð/ -> /z/ (p=0.9238) label=substituted l1=arabic data/l2arctic/ABA/wav/arctic_a0500.wav
- [TP] l2arctic-THV-arctic_a0046 /ð/ -> /z/ (p=0.8877) label=substituted l1=vietnamese data/l2arctic/THV/wav/arctic_a0046.wav
- [TP] l2arctic-YKWK-arctic_a0369 /ð/ -> /d/ (p=0.8567) label=deleted l1=korean data/l2arctic/YKWK/wav/arctic_a0369.wav
