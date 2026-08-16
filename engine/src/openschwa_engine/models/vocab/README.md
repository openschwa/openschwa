# Committed model vocabularies

Each file is the verbatim `vocab.json` of a pinned upstream model, committed so
that phone-set mapping is testable without downloading gigabytes of weights, and
so that an upstream vocabulary change is caught rather than silently reindexing
every phone.

`models/registry.py` compares the downloaded vocabulary against the snapshot at
load and refuses to run on a mismatch. `docs/architecture.md` names phone-set
mapping bugs as a top risk precisely because they corrupt verdicts quietly.

| File | Upstream | Revision | License |
|---|---|---|---|
| `wav2vec2-espeak-cv-ft.json` | [`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) | see `registry.MANIFEST` | Apache-2.0 |

These files are third-party data under their own licenses, not AGPL-licensed
OpenSchwa source.
