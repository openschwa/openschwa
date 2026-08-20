# m1-2026-08-20-charsiu-en-w2v2-ctc-ð-vs-z-d-v

- model: charsiu-en-w2v2-ctc
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 0 train / 1312 cal / 2410 held-out
- **status: SHIPPING BAR NOT MET** (the mirror; judge: SHIPPING BAR NOT MET)

## Mirror - what the ear heard (shipped line)

- Platt: p = sigmoid(0.037 * hearing_score + -0.052)  [P(heard == realized)]
- threshold: 0.56
- cal: accuracy 0.5789 / coverage 0.029 (cal status SHIPPING BAR NOT MET)
- **held-out: accuracy 0.6604 / coverage 0.022** / answer-rate 0.022
- raw top-1 hearing accuracy (no gating): 0.669
- confident reports 53 of 2410 tokens (scored 2407; unscorable 0)

### realized x heard (confident reports)

| realized \ heard | /h/ | /ð/ |
|---|---|---|
| /ð/ | 0 | 34 |
| /d/ | 0 | 18 |
| /h/ | 1 | 0 |

### Mirror per L1 (held-out)

Fairness audit of the single shipped line: every group is heard at
the same global operating point 0.56. Informational
only - it never gates the bar.

| l1 | tokens | confident | accuracy | coverage | fair |
|---|---|---|---|---|---|
| arabic | 150 | 6 | 1.0 | 0.04 | False |
| hindi | 158 | 22 | 0.4545 | 0.1392 | False |
| korean | 161 | 6 | 0.6667 | 0.0373 | False |
| mandarin | 1640 | 14 | 1.0 | 0.0085 | False |
| spanish | 152 | 2 | 0.5 | 0.0132 | False |
| vietnamese | 149 | 3 | 0.0 | 0.0201 | False |

### Mirror spot-check (confident reports, mishearings first)

- [WRONG] l2arctic-ASI-arctic_a0023 heard /ð/, realized /d/ (p=0.5609) l1=hindi data/l2arctic/ASI/wav/arctic_a0023.wav
- [WRONG] l2arctic-ASI-arctic_a0029 heard /ð/, realized /d/ (p=0.5616) l1=hindi data/l2arctic/ASI/wav/arctic_a0029.wav
- [WRONG] l2arctic-ASI-arctic_a0036 heard /ð/, realized /d/ (p=0.5604) l1=hindi data/l2arctic/ASI/wav/arctic_a0036.wav
- [WRONG] l2arctic-ASI-arctic_a0037 heard /ð/, realized /d/ (p=0.5611) l1=hindi data/l2arctic/ASI/wav/arctic_a0037.wav
- [WRONG] l2arctic-ASI-arctic_a0039 heard /ð/, realized /d/ (p=0.5601) l1=hindi data/l2arctic/ASI/wav/arctic_a0039.wav
- [WRONG] l2arctic-ASI-arctic_a0058 heard /ð/, realized /d/ (p=0.5622) l1=hindi data/l2arctic/ASI/wav/arctic_a0058.wav
- [WRONG] l2arctic-ASI-arctic_a0111 heard /ð/, realized /d/ (p=0.5627) l1=hindi data/l2arctic/ASI/wav/arctic_a0111.wav
- [WRONG] l2arctic-ASI-arctic_a0208 heard /ð/, realized /d/ (p=0.5603) l1=hindi data/l2arctic/ASI/wav/arctic_a0208.wav
- [WRONG] l2arctic-ASI-arctic_a0292 heard /ð/, realized /d/ (p=0.5634) l1=hindi data/l2arctic/ASI/wav/arctic_a0292.wav
- [WRONG] l2arctic-ASI-arctic_a0384 heard /ð/, realized /d/ (p=0.56) l1=hindi data/l2arctic/ASI/wav/arctic_a0384.wav
- [WRONG] l2arctic-ASI-arctic_b0047 heard /ð/, realized /d/ (p=0.5609) l1=hindi data/l2arctic/ASI/wav/arctic_b0047.wav
- [WRONG] l2arctic-ASI-arctic_b0315 heard /ð/, realized /d/ (p=0.5606) l1=hindi data/l2arctic/ASI/wav/arctic_b0315.wav
- [WRONG] l2arctic-ERMS-arctic_a0088 heard /ð/, realized /d/ (p=0.5617) l1=spanish data/l2arctic/ERMS/wav/arctic_a0088.wav
- [WRONG] l2arctic-THV-arctic_a0019 heard /ð/, realized /d/ (p=0.5606) l1=vietnamese data/l2arctic/THV/wav/arctic_a0019.wav
- [WRONG] l2arctic-THV-arctic_a0037 heard /ð/, realized /d/ (p=0.5606) l1=vietnamese data/l2arctic/THV/wav/arctic_a0037.wav
- [right] l2arctic-ASI-arctic_a0108 heard /ð/, realized /ð/ (p=0.5605) l1=hindi data/l2arctic/ASI/wav/arctic_a0108.wav
- [right] l2arctic-ASI-arctic_b0073 heard /ð/, realized /ð/ (p=0.5608) l1=hindi data/l2arctic/ASI/wav/arctic_b0073.wav
- [right] l2arctic-ABA-arctic_b0461 heard /ð/, realized /ð/ (p=0.5616) l1=arabic data/l2arctic/ABA/wav/arctic_b0461.wav
- [right] l2arctic-YKWK-arctic_a0137 heard /ð/, realized /ð/ (p=0.5645) l1=korean data/l2arctic/YKWK/wav/arctic_a0137.wav
- [right] so762-024510247 heard /ð/, realized /ð/ (p=0.56) l1=mandarin data/speechocean762/WAVE/SPEAKER2451/024510247.WAV
- [right] so762-000240073 heard /ð/, realized /ð/ (p=0.5639) l1=mandarin data/speechocean762/WAVE/SPEAKER0024/000240073.WAV
- [right] so762-020020108 heard /ð/, realized /ð/ (p=0.5621) l1=mandarin data/speechocean762/WAVE/SPEAKER2002/020020108.WAV
- [right] l2arctic-BWC-arctic_a0037 heard /ð/, realized /ð/ (p=0.5617) l1=mandarin data/l2arctic/BWC/wav/arctic_a0037.wav
- [right] so762-001570030 heard /ð/, realized /ð/ (p=0.5604) l1=mandarin data/speechocean762/WAVE/SPEAKER0157/001570030.WAV
- [right] so762-001200210 heard /ð/, realized /ð/ (p=0.5637) l1=mandarin data/speechocean762/WAVE/SPEAKER0120/001200210.WAV
- [right] l2arctic-ASI-arctic_b0317 heard /ð/, realized /ð/ (p=0.5625) l1=hindi data/l2arctic/ASI/wav/arctic_b0317.wav
- [right] l2arctic-ASI-arctic_a0120 heard /ð/, realized /ð/ (p=0.5641) l1=hindi data/l2arctic/ASI/wav/arctic_a0120.wav
- [right] l2arctic-ASI-arctic_a0191 heard /ð/, realized /ð/ (p=0.5606) l1=hindi data/l2arctic/ASI/wav/arctic_a0191.wav
- [right] so762-010990048 heard /ð/, realized /ð/ (p=0.5645) l1=mandarin data/speechocean762/WAVE/SPEAKER1099/010990048.WAV
- [right] l2arctic-ASI-arctic_a0096 heard /ð/, realized /ð/ (p=0.5642) l1=hindi data/l2arctic/ASI/wav/arctic_a0096.wav

## Judge variants (research archive - parked by the mirror pivot)

| variant | threshold | status | cal P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.605 | SHIPPING BAR NOT MET | 0.4545/0.0091 | 0.1739/0.0077 | 0.0147 | 0.6381 |
| spike | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.6358 |
| vote | 0.995 | SHIPPING BAR NOT MET | 0.0/0.0 | 0.0/0.0 | 0.0 | 0.5397 |
| gop | 0.59 | pooled-bar-not-met | 1.0/0.0018 | 0.0/0.0 | 0.0 | 0.4362 |

## Judge operating point (calibration split, research archive)

- Platt: p = sigmoid(-0.034 * score + -0.515)
- threshold: 0.605
- cal: precision 0.4545, recall 0.0091
- cal->test precision gap: 0.2806

## Judge held-out (research archive)

- precision 0.1739 / recall 0.0077 / f1 0.0147
- AUC 0.6381
- verdicts 2407, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.605. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 150 | 20 | 0.0 | 0.0 | 0.0 | False |
| hindi | 158 | 85 | 0.5714 | 0.0471 | 0.087 | False |
| korean | 161 | 102 | 0.0 | 0.0 | 0.0 | False |
| mandarin | 1640 | 111 | 0.0 | 0.0 | 0.0 | False |
| spanish | 152 | 122 | 0.0 | 0.0 | 0.0 | False |
| vietnamese | 149 | 82 | 0.0 | 0.0 | 0.0 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 | AUC |
|---|---|---|---|---|---|---|---|
| l2arctic | 915 | 521 | 0.3333 | 0.0077 | 0.015 | 0.4129 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 | 0.5459 |

## Alignment sanity

- statuses: {'ok': 3718, 'low_confidence': 4}
- mean alignment confidence (ok): 0.9001

## Latency

- cold 24.8 ms, median warm 20.5 ms
- download size 0.38 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0058 /ð/ -> /z/ (p=0.6055) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0058.wav
- [FP] l2arctic-ABA-arctic_a0060 /ð/ -> /z/ (p=0.6055) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0060.wav
- [FP] l2arctic-ABA-arctic_a0461 /ð/ -> /z/ (p=0.6055) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0461.wav
- [FP] l2arctic-ABA-arctic_b0461 /ð/ -> /z/ (p=0.6055) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_b0461.wav
- [FP] l2arctic-ASI-arctic_a0096 /ð/ -> /z/ (p=0.6055) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0096.wav
- [FP] l2arctic-ASI-arctic_a0120 /ð/ -> /z/ (p=0.6055) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_a0120.wav
- [FP] l2arctic-ASI-arctic_b0197 /ð/ -> /z/ (p=0.6055) label=correct l1=hindi data/l2arctic/ASI/wav/arctic_b0197.wav
- [FP] l2arctic-THV-arctic_a0542 /ð/ -> /z/ (p=0.6055) label=correct l1=vietnamese data/l2arctic/THV/wav/arctic_a0542.wav
- [FP] so762-000240073 /ð/ -> /z/ (p=0.6055) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER0024/000240073.WAV
- [FP] so762-000240320 /ð/ -> /z/ (p=0.6055) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER0024/000240320.WAV
- [FP] so762-001570030 /ð/ -> /z/ (p=0.6055) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER0157/001570030.WAV
- [FP] so762-001570325 /ð/ -> /z/ (p=0.6055) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER0157/001570325.WAV
- [FP] so762-010990048 /ð/ -> /z/ (p=0.6055) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER1099/010990048.WAV
- [FP] so762-012920085 /ð/ -> /z/ (p=0.6055) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER1292/012920085.WAV
- [FP] so762-020020108 /ð/ -> /z/ (p=0.6055) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER2002/020020108.WAV
- [TP] l2arctic-ASI-arctic_a0075 /ð/ -> /z/ (p=0.6055) label=substituted l1=hindi data/l2arctic/ASI/wav/arctic_a0075.wav
- [TP] l2arctic-ASI-arctic_a0058 /ð/ -> /z/ (p=0.6055) label=substituted l1=hindi data/l2arctic/ASI/wav/arctic_a0058.wav
- [TP] l2arctic-ASI-arctic_a0111 /ð/ -> /z/ (p=0.6055) label=substituted l1=hindi data/l2arctic/ASI/wav/arctic_a0111.wav
- [TP] l2arctic-ASI-arctic_a0032 /ð/ -> /z/ (p=0.6055) label=substituted l1=hindi data/l2arctic/ASI/wav/arctic_a0032.wav
