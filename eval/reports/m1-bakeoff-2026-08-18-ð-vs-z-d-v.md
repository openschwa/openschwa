# M1 bake-off - /ð/ vs ['z', 'd', 'v']

| model | held-out P | held-out R | f1 | AUC | status | median warm ms | size GB |
|---|---|---|---|---|---|---|---|
| wav2vec2-espeak-cv-ft | 0.0 | 0.0 | 0.0 | 0.5941 | SHIPPING BAR NOT MET | 356.5 | 1.26 |
| charsiu-en-w2v2-ctc | 0.1892 | 0.0118 | 0.0222 | 0.5829 | SHIPPING BAR NOT MET | 135.4 | 0.38 |

## Alignment sanity (held-out token runs)

| model | statuses | mean ok confidence |
|---|---|---|
| wav2vec2-espeak-cv-ft | {'ok': 6565, 'low_confidence': 113} | 0.8176 |
| charsiu-en-w2v2-ctc | {'ok': 6667, 'low_confidence': 11} | 0.905 |

**Winner: none (bar not met)**
