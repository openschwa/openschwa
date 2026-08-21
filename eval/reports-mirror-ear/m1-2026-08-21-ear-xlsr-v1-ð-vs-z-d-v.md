# m1-2026-08-21-ear-xlsr-v1-ð-vs-z-d-v

- model: ear-xlsr-v1
- contrast: /ð/ vs ['z', 'd', 'v']
- tokens: 0 train / 1312 cal / 2410 held-out
- **status: SHIPPING BAR NOT MET** (the mirror; judge: pooled-bar-not-met)

## Mirror - what the ear heard (shipped line)

- Platt: p = sigmoid(0.473 * hearing_score + -1.037)  [P(heard == realized)]
- threshold: 0.995
- cal: accuracy 0.0 / coverage 0.0 (cal status SHIPPING BAR NOT MET)
- **held-out: accuracy 0.0 / coverage 0.0** / answer-rate 0.0
- raw top-1 hearing accuracy (no gating): 0.2108
- confident reports 0 of 2410 tokens (scored 2407; unscorable 0)

### realized x heard (confident reports)

| realized \ heard |  |
|---|

### Mirror per L1 (held-out)

Fairness audit of the single shipped line: every group is heard at
the same global operating point 0.995. Informational
only - it never gates the bar.

| l1 | tokens | confident | accuracy | coverage | fair |
|---|---|---|---|---|---|
| arabic | 150 | 0 | 0.0 | 0.0 | True |
| hindi | 158 | 0 | 0.0 | 0.0 | True |
| korean | 161 | 0 | 0.0 | 0.0 | True |
| mandarin | 1640 | 0 | 0.0 | 0.0 | True |
| spanish | 152 | 0 | 0.0 | 0.0 | True |
| vietnamese | 149 | 0 | 0.0 | 0.0 | True |

### Mirror spot-check (confident reports, mishearings first)


## Judge variants (research archive - parked by the mirror pivot)

| variant | threshold | status | cal P/R | held-out P/R | f1 | AUC |
|---|---|---|---|---|---|---|
| mean | 0.52 | pooled-bar-not-met | 1.0/0.0036 | 0.1111/0.0019 | 0.0038 | 0.6246 |
| spike | 0.63 | pooled-bar-not-met | 1.0/0.0181 | 0.5294/0.0172 | 0.0334 | 0.5996 |
| vote | 0.63 | pooled-bar-not-met | 1.0/0.0036 | 0.0/0.0 | 0.0 | 0.6367 |
| gop | 0.59 | pooled-bar-not-met | 1.0/0.0018 | 0.0/0.0 | 0.0 | 0.4362 |

## Judge operating point (calibration split, research archive)

- Platt: p = sigmoid(-2.189 * score + 1.534)
- threshold: 0.63
- cal: precision 1.0, recall 0.0036
- cal->test precision gap: 1.0

## Judge held-out (research archive)

- precision 0.0 / recall 0.0 / f1 0.0
- AUC 0.6367
- verdicts 2407, refused 3 (audio-problem 0)

## Per L1 audit (held-out)

Fairness check of the single shipped line: every group is scored at
the same global operating point 0.63. Informational
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
| l2arctic | 915 | 521 | 0.0 | 0.0 | 0.0 | 0.4966 |
| so762 | 1495 | 1 | 0.0 | 0.0 | 0.0 | 0.937 |

## Alignment sanity

- statuses: {'ok': 3718, 'low_confidence': 4}
- mean alignment confidence (ok): 0.9001

## Latency

- cold 7823.6 ms, median warm 31.2 ms
- download size 0.00 GB

## Flagged items for human spot-check

(paths relative to the corpus roots; review per eval/README.md)

- [FP] l2arctic-ABA-arctic_a0051 /ð/ -> /d/ (p=0.659) label=correct l1=arabic data/l2arctic/ABA/wav/arctic_a0051.wav
