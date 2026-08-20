# m1-2026-08-20-dh-contrast-v1-ð-vs-z-d-v

- model: dh-contrast-v1
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 0 train / 1312 cal / 2410 held-out
- **status: SHIPPING BAR NOT MET** (the mirror; judge: pooled-bar-not-met)

## Mirror - what the ear heard (shipped line)

- Platt: p = sigmoid(0.275 * hearing_score + -0.052)  [P(heard == realized)]
- threshold: 0.62
- cal: accuracy 0.7037 / coverage 0.1867 (cal status SHIPPING BAR NOT MET)
- **held-out: accuracy 0.7904 / coverage 0.2697** / answer-rate 0.27
- raw top-1 hearing accuracy (no gating): 0.6233
- confident reports 650 of 2410 tokens (scored 2407; unscorable 1)

### realized x heard (confident reports)

| realized \ heard | /d/ | /v/ | /z/ | /ð/ |
|---|---|---|---|---|
| /ð/ | 12 | 22 | 79 | 498 |
| /d/ | 8 | 5 | 1 | 11 |
| /t/ | 1 | 0 | 0 | 0 |
| /z/ | 0 | 0 | 7 | 0 |
| /θ/ | 0 | 1 | 0 | 0 |
| deleted | 1 | 2 | 1 | 0 |

### Mirror per L1 (held-out)

Fairness audit of the single shipped line: every group is heard at
the same global operating point 0.62. Informational
only - it never gates the bar.

| l1 | tokens | confident | accuracy | coverage | fair |
|---|---|---|---|---|---|
| arabic | 150 | 43 | 0.907 | 0.2867 | False |
| hindi | 158 | 8 | 0.375 | 0.0506 | False |
| korean | 161 | 10 | 0.4 | 0.0621 | False |
| mandarin | 1640 | 559 | 0.819 | 0.3409 | False |
| spanish | 152 | 15 | 0.2 | 0.0987 | False |
| vietnamese | 149 | 15 | 0.4667 | 0.1007 | False |

### Mirror spot-check (confident reports, mishearings first)

- [WRONG] l2arctic-ABA-arctic_a0036 heard /v/, realized /ð/ (p=0.6217) l1=arabic data/l2arctic/ABA/wav/arctic_a0036.wav
- [WRONG] l2arctic-ABA-arctic_a0117 heard /z/, realized /ð/ (p=0.7736) l1=arabic data/l2arctic/ABA/wav/arctic_a0117.wav
- [WRONG] l2arctic-ABA-arctic_a0137 heard /d/, realized /ð/ (p=0.6425) l1=arabic data/l2arctic/ABA/wav/arctic_a0137.wav
- [WRONG] l2arctic-ABA-arctic_a0461 heard /v/, realized /ð/ (p=0.6273) l1=arabic data/l2arctic/ABA/wav/arctic_a0461.wav
- [WRONG] l2arctic-ASI-arctic_a0047 heard /v/, realized /d/ (p=0.6497) l1=hindi data/l2arctic/ASI/wav/arctic_a0047.wav
- [WRONG] l2arctic-ASI-arctic_a0071 heard /z/, realized /d/ (p=0.6347) l1=hindi data/l2arctic/ASI/wav/arctic_a0071.wav
- [WRONG] l2arctic-ASI-arctic_a0077 heard /z/, realized /ð/ (p=0.629) l1=hindi data/l2arctic/ASI/wav/arctic_a0077.wav
- [WRONG] l2arctic-ASI-arctic_a0095 heard /z/, realized /ð/ (p=0.639) l1=hindi data/l2arctic/ASI/wav/arctic_a0095.wav
- [WRONG] l2arctic-ASI-arctic_a0108 heard /v/, realized /ð/ (p=0.6909) l1=hindi data/l2arctic/ASI/wav/arctic_a0108.wav
- [WRONG] l2arctic-BWC-arctic_a0081 heard /v/, realized /∅/ (p=0.6241) l1=mandarin data/l2arctic/BWC/wav/arctic_a0081.wav
- [WRONG] l2arctic-BWC-arctic_a0139 heard /v/, realized /θ/ (p=0.6883) l1=mandarin data/l2arctic/BWC/wav/arctic_a0139.wav
- [WRONG] l2arctic-BWC-arctic_a0575 heard /z/, realized /∅/ (p=0.7282) l1=mandarin data/l2arctic/BWC/wav/arctic_a0575.wav
- [WRONG] l2arctic-BWC-arctic_b0400 heard /v/, realized /∅/ (p=0.6837) l1=mandarin data/l2arctic/BWC/wav/arctic_b0400.wav
- [WRONG] l2arctic-ERMS-arctic_a0029 heard /v/, realized /d/ (p=0.6945) l1=spanish data/l2arctic/ERMS/wav/arctic_a0029.wav
- [WRONG] l2arctic-ERMS-arctic_a0032 heard /ð/, realized /d/ (p=0.6319) l1=spanish data/l2arctic/ERMS/wav/arctic_a0032.wav
- [right] so762-020300097 heard /ð/, realized /ð/ (p=0.6968) l1=mandarin data/speechocean762/WAVE/SPEAKER2030/020300097.WAV
- [right] so762-020140004 heard /ð/, realized /ð/ (p=0.6772) l1=mandarin data/speechocean762/WAVE/SPEAKER2014/020140004.WAV
- [right] so762-054280062 heard /ð/, realized /ð/ (p=0.6509) l1=mandarin data/speechocean762/WAVE/SPEAKER5428/054280062.WAV
- [right] so762-007650192 heard /ð/, realized /ð/ (p=0.7226) l1=mandarin data/speechocean762/WAVE/SPEAKER0765/007650192.WAV
- [right] so762-096330023 heard /ð/, realized /ð/ (p=0.6224) l1=mandarin data/speechocean762/WAVE/SPEAKER9633/096330023.WAV
- [right] so762-003060107 heard /ð/, realized /ð/ (p=0.6699) l1=mandarin data/speechocean762/WAVE/SPEAKER0306/003060107.WAV
- [right] so762-096230006 heard /ð/, realized /ð/ (p=0.6436) l1=mandarin data/speechocean762/WAVE/SPEAKER9623/096230006.WAV
- [right] so762-014200076 heard /ð/, realized /ð/ (p=0.6225) l1=mandarin data/speechocean762/WAVE/SPEAKER1420/014200076.WAV
- [right] so762-015020113 heard /ð/, realized /ð/ (p=0.6346) l1=mandarin data/speechocean762/WAVE/SPEAKER1502/015020113.WAV
- [right] so762-024410322 heard /ð/, realized /ð/ (p=0.6447) l1=mandarin data/speechocean762/WAVE/SPEAKER2441/024410322.WAV
- [right] so762-030750170 heard /ð/, realized /ð/ (p=0.6513) l1=mandarin data/speechocean762/WAVE/SPEAKER3075/030750170.WAV
- [right] so762-096180008 heard /ð/, realized /ð/ (p=0.6774) l1=mandarin data/speechocean762/WAVE/SPEAKER9618/096180008.WAV
- [right] so762-003060107 heard /ð/, realized /ð/ (p=0.6704) l1=mandarin data/speechocean762/WAVE/SPEAKER0306/003060107.WAV
- [right] so762-010750011 heard /ð/, realized /ð/ (p=0.6547) l1=mandarin data/speechocean762/WAVE/SPEAKER1075/010750011.WAV
- [right] so762-020300121 heard /ð/, realized /ð/ (p=0.6931) l1=mandarin data/speechocean762/WAVE/SPEAKER2030/020300121.WAV

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

- cold 4126.7 ms, median warm 26.2 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] so762-052200168 /ð/ -> /z/ (p=0.9032) label=correct l1=mandarin data/speechocean762/WAVE/SPEAKER5220/052200168.WAV
