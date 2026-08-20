# M1 mirror bake-off - all three ears (2026-08-20) - /ð/ vs ['z', 'd', 'v']

| model | mirror accuracy | coverage | raw top-1 | judge AUC | status | median warm ms | size GB |
|---|---|---|---|---|---|---|---|
| charsiu-en-w2v2-ctc | 0.6604 | 0.022 | 0.669 | 0.6381 | SHIPPING BAR NOT MET | 20.5 | 0.38 |
| dh-contrast-v1 | 0.7904 | 0.2697 | 0.6233 | 0.6861 | SHIPPING BAR NOT MET | 26.2 | 0.00 |
| tinyschwa-v1 | 0.9528 | 0.4228 | 0.7875 | 0.866 | ok | 330.8 | 0.00 |

## Alignment sanity (held-out token runs)

| model | statuses | mean ok confidence |
|---|---|---|
| charsiu-en-w2v2-ctc | {'ok': 3718, 'low_confidence': 4} | 0.9001 |
| dh-contrast-v1 | {'ok': 3718, 'low_confidence': 4} | 0.9001 |
| tinyschwa-v1 | {'ok': 3718, 'low_confidence': 4} | 0.9001 |

**Winner: tinyschwa-v1**
