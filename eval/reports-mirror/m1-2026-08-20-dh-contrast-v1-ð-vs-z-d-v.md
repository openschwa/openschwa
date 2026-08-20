# m1-2026-08-20-dh-contrast-v1-ð-vs-z-d-v

- model: dh-contrast-v1
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 0 train / 1312 cal / 2410 held-out
- **status: SHIPPING BAR NOT MET** (the mirror; judge: pooled-bar-not-met)

## Mirror - what the ear heard (shipped line)

- Platt: p = sigmoid(0.275 * hearing_score + -0.052)  [P(heard == realized)]
- threshold: 0.6
- cal: accuracy 0.7035 / coverage 0.2431 (cal status SHIPPING BAR NOT MET)
- **held-out: accuracy 0.7802 / coverage 0.3407** / answer-rate 0.3411
- raw top-1 hearing accuracy (no gating): 0.6233
- confident reports 821 of 2410 tokens (scored 2407; unscorable 2)

### realized x heard (confident reports)

| realized \ heard | /d/ | /v/ | /z/ | /ð/ |
|---|---|---|---|---|
| /ð/ | 23 | 30 | 92 | 620 |
| /d/ | 11 | 5 | 1 | 21 |
| /s/ | 0 | 0 | 1 | 0 |
| /t/ | 1 | 0 | 0 | 0 |
| /z/ | 0 | 0 | 8 | 0 |
| /θ/ | 1 | 1 | 0 | 0 |
| deleted | 1 | 2 | 1 | 0 |

### Mirror per L1 (held-out)

Fairness audit of the single shipped line: every group is heard at
the same global operating point 0.6. Informational
only - it never gates the bar.

| l1 | tokens | confident | accuracy | coverage | fair |
|---|---|---|---|---|---|
| arabic | 150 | 59 | 0.8814 | 0.3933 | False |
| hindi | 158 | 13 | 0.3846 | 0.0823 | False |
| korean | 161 | 13 | 0.3846 | 0.0807 | False |
| mandarin | 1640 | 684 | 0.8138 | 0.4171 | False |
| spanish | 152 | 26 | 0.1923 | 0.1711 | False |
| vietnamese | 149 | 26 | 0.6538 | 0.1745 | False |

### Mirror spot-check (confident reports, mishearings first)

- [WRONG] l2arctic-ABA-arctic_a0003 heard /v/, realized /ð/ (p=0.6051) l1=arabic data/l2arctic/ABA/wav/arctic_a0003.wav
- [WRONG] l2arctic-ABA-arctic_a0036 heard /v/, realized /ð/ (p=0.6217) l1=arabic data/l2arctic/ABA/wav/arctic_a0036.wav
- [WRONG] l2arctic-ABA-arctic_a0078 heard /d/, realized /θ/ (p=0.608) l1=arabic data/l2arctic/ABA/wav/arctic_a0078.wav
- [WRONG] l2arctic-ABA-arctic_a0117 heard /z/, realized /ð/ (p=0.7735) l1=arabic data/l2arctic/ABA/wav/arctic_a0117.wav
- [WRONG] l2arctic-ABA-arctic_a0137 heard /d/, realized /ð/ (p=0.6425) l1=arabic data/l2arctic/ABA/wav/arctic_a0137.wav
- [WRONG] l2arctic-ABA-arctic_a0159 heard /z/, realized /s/ (p=0.6074) l1=arabic data/l2arctic/ABA/wav/arctic_a0159.wav
- [WRONG] l2arctic-ABA-arctic_a0461 heard /v/, realized /ð/ (p=0.6273) l1=arabic data/l2arctic/ABA/wav/arctic_a0461.wav
- [WRONG] l2arctic-ASI-arctic_a0031 heard /ð/, realized /d/ (p=0.6066) l1=hindi data/l2arctic/ASI/wav/arctic_a0031.wav
- [WRONG] l2arctic-ASI-arctic_a0047 heard /v/, realized /d/ (p=0.6496) l1=hindi data/l2arctic/ASI/wav/arctic_a0047.wav
- [WRONG] l2arctic-ASI-arctic_a0071 heard /z/, realized /d/ (p=0.6347) l1=hindi data/l2arctic/ASI/wav/arctic_a0071.wav
- [WRONG] l2arctic-ASI-arctic_a0077 heard /z/, realized /ð/ (p=0.629) l1=hindi data/l2arctic/ASI/wav/arctic_a0077.wav
- [WRONG] l2arctic-ASI-arctic_a0095 heard /z/, realized /ð/ (p=0.639) l1=hindi data/l2arctic/ASI/wav/arctic_a0095.wav
- [WRONG] l2arctic-ASI-arctic_a0108 heard /v/, realized /ð/ (p=0.6909) l1=hindi data/l2arctic/ASI/wav/arctic_a0108.wav
- [WRONG] l2arctic-ASI-arctic_b0136 heard /d/, realized /ð/ (p=0.6108) l1=hindi data/l2arctic/ASI/wav/arctic_b0136.wav
- [WRONG] l2arctic-ASI-arctic_b0215 heard /d/, realized /ð/ (p=0.6127) l1=hindi data/l2arctic/ASI/wav/arctic_b0215.wav
- [right] so762-007360299 heard /ð/, realized /ð/ (p=0.6413) l1=mandarin data/speechocean762/WAVE/SPEAKER0736/007360299.WAV
- [right] so762-060670049 heard /ð/, realized /ð/ (p=0.6071) l1=mandarin data/speechocean762/WAVE/SPEAKER6067/060670049.WAV
- [right] so762-030140144 heard /ð/, realized /ð/ (p=0.654) l1=mandarin data/speechocean762/WAVE/SPEAKER3014/030140144.WAV
- [right] so762-020140059 heard /ð/, realized /ð/ (p=0.6131) l1=mandarin data/speechocean762/WAVE/SPEAKER2014/020140059.WAV
- [right] so762-010500163 heard /ð/, realized /ð/ (p=0.6122) l1=mandarin data/speechocean762/WAVE/SPEAKER1050/010500163.WAV
- [right] so762-096230007 heard /ð/, realized /ð/ (p=0.6299) l1=mandarin data/speechocean762/WAVE/SPEAKER9623/096230007.WAV
- [right] so762-060990155 heard /ð/, realized /ð/ (p=0.6724) l1=mandarin data/speechocean762/WAVE/SPEAKER6099/060990155.WAV
- [right] so762-004570283 heard /ð/, realized /ð/ (p=0.6161) l1=mandarin data/speechocean762/WAVE/SPEAKER0457/004570283.WAV
- [right] so762-007650337 heard /ð/, realized /ð/ (p=0.7038) l1=mandarin data/speechocean762/WAVE/SPEAKER0765/007650337.WAV
- [right] so762-003060229 heard /ð/, realized /ð/ (p=0.627) l1=mandarin data/speechocean762/WAVE/SPEAKER0306/003060229.WAV
- [right] so762-096330020 heard /ð/, realized /ð/ (p=0.6795) l1=mandarin data/speechocean762/WAVE/SPEAKER9633/096330020.WAV
- [right] so762-030270026 heard /ð/, realized /ð/ (p=0.7035) l1=mandarin data/speechocean762/WAVE/SPEAKER3027/030270026.WAV
- [right] so762-055470104 heard /ð/, realized /ð/ (p=0.6497) l1=mandarin data/speechocean762/WAVE/SPEAKER5547/055470104.WAV
- [right] so762-096260011 heard /ð/, realized /ð/ (p=0.7) l1=mandarin data/speechocean762/WAVE/SPEAKER9626/096260011.WAV
- [right] so762-024380204 heard /ð/, realized /ð/ (p=0.6334) l1=mandarin data/speechocean762/WAVE/SPEAKER2438/024380204.WAV

## Judge variants (research archive - parked by the mirror pivot)

| variant | threshold | status | cal P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.9 | pooled-bar-not-met | 1.0/0.0018 | 0.0/0.0 | 0.0 | 0.6861 |
| spike | 0.575 | SHIPPING BAR NOT MET | 0.6/0.0109 | 0.5/0.0038 | 0.0076 | 0.5881 |
| vote | 0.505 | SHIPPING BAR NOT MET | 0.5464/0.5554 | 0.3564/0.4828 | 0.4101 | 0.6139 |
| gop | 0.59 | pooled-bar-not-met | 1.0/0.0018 | 0.0/0.0 | 0.0 | 0.4362 |

## Judge operating point (calibration split, research archive)

- Platt: p = sigmoid(0.352 * score + -0.327)
- threshold: 0.9
- cal: precision 1.0, recall 0.0018
- cal->test precision gap: 1.0

## Judge held-out (research archive)

- precision 0.0 / recall 0.0 / f1 0.0
- AUC 0.6861
- verdicts 2407, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.9. Informational
only - it never gates the bar; it shows whether the accent-blind line
treats every language group alike, and names the group when it does
not.

| l1 | tokens | positives | precision | recall | f1 | fair |
|---|---|---|---|---|---|---|
| arabic | 150 | 20 | 0.0 | 0.0 | 0.0 | False |
| hindi | 158 | 85 | 0.0 | 0.0 | 0.0 | False |
| korean | 161 | 102 | 0.0 | 0.0 | 0.0 | False |
| mandarin | 1640 | 111 | 0.0 | 0.0 | 0.0 | False |
| spanish | 152 | 122 | 0.0 | 0.0 | 0.0 | False |
| vietnamese | 149 | 82 | 0.0 | 0.0 | 0.0 | False |

## Per corpus (held-out)

| corpus | tokens | positives | precision | recall | f1 | AUC |
|---|---|---|---|---|---|---|---|
| l2arctic | 915 | 521 | 0.0 | 0.0 | 0.0 | 0.6053 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 | 0.9812 |

## Alignment sanity

- statuses: {'ok': 3718, 'low_confidence': 4}
- mean alignment confidence (ok): 0.9001

## Latency

- cold 3003.5 ms, median warm 25.8 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] so762-052200168 /ð/ -> /z/ (p=0.9032) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER5220/052200168.WAV
