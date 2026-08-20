import json
from pathlib import Path
from openschwa_engine.audio import decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.prosody import track, nuclear_tone
from openschwa_engine.prosody.compare import _voiced_series, _slope

settings = Settings(warm_model_on_start=False)
manifest = json.load(open("manifests/intonation-recordings.json"))
items = {i["id"]: i for i in manifest["items"]}
excl = set(json.load(open("data/recordings/exclusions.json")))
rows = []
for p in sorted(Path("data/recordings").glob("*.wav")):
    stem = p.stem
    if stem in excl:
        continue
    iid, rep = stem.rsplit("_", 1)
    item = items[iid]
    if item["tone"] not in ("fall", "rise"):
        continue
    decoded = decode_wav(p.read_bytes())
    prepared = prepare(decoded.samples, decoded.sample_rate, vad_backend=settings.vad_backend)
    f0 = track(prepared.samples_16k, 16000)
    if f0 is None:
        continue
    dur = f0.start_s + (len(f0.semitones) - 1) * f0.hop_s
    se = prepared.speech_interval_s[1] if prepared.speech_interval_s else dur
    end = min(dur, se)
    last_voiced = None
    for i, v in enumerate(f0.semitones):
        if v is None:
            continue
        t = f0.start_s + i * f0.hop_s
        if t <= end + 1e-6:
            last_voiced = t
        else:
            break
    if last_voiced is not None:
        end = min(end, last_voiced)
    t, v = _voiced_series(f0, max(0.0, end - 0.35), end)
    slope = _slope(t, v) if v.size >= 3 else None
    mid = float(t[0]) + (float(t[-1]) - float(t[0])) / 2.0 if v.size >= 3 else 0.0
    first = _slope(t[t <= mid], v[t <= mid]) if v.size >= 3 and v[t <= mid].size >= 3 else None
    second = _slope(t[t > mid], v[t > mid]) if v.size >= 3 and v[t > mid].size >= 3 else None
    tone, conf = nuclear_tone(f0, end_s=se)
    rows.append((stem, item["tone"], tone, slope, first, second, len(v)))
for r in rows:
    if r[1] != r[2]:
        print(r)
