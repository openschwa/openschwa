# M1 mirror bake-off - /ð/ vs ['z', 'd', 'v']

| model | mirror accuracy | coverage | raw top-1 | judge AUC | status | median warm ms | size GB |
|---|---|---|---|---|---|---|---|
| dh-contrast-v1 | 0.7802 | 0.3407 | 0.6233 | 0.6861 | SHIPPING BAR NOT MET | 25.8 | 0.00 |
| charsiu-en-w2v2-ctc | 0.5 | 0.0008 | 0.7044 | 0.6381 | mirror-bar-not-met | 20.3 | 0.38 |

## Alignment sanity (held-out token runs)

| model | statuses | mean ok confidence |
|---|---|---|
| dh-contrast-v1 | {'ok': 3718, 'low_confidence': 4} | 0.9001 |
| charsiu-en-w2v2-ctc | {'ok': 3718, 'low_confidence': 4} | 0.9001 |

**Winner: none (mirror bar not met)**
