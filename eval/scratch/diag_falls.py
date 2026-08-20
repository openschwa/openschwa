import json
from pathlib import Path
from openschwa_engine.audio import decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.prosody import track, nuclear_tone
from openschwa_engine.prosody.compare import _voiced_series, _slope

settings = Settings(warm_model_on_start=False)
manifest = json.load(open("manifests/intonation-recordings.json"))
items = {i["id"]: i for i in manifest["items"]}
rows = []
for p in sorted(Path("data/recordings").glob("*.wav")):
    iid, rep = p.stem.rsplit("_", 1)
    item = items[iid]
    if item["tone"] != "fall":
        continue
    decoded = decode_wav(p.read_bytes())
    prepared = prepare(decoded.samples, decoded.sample_rate, vad_backend=settings.vad_backend)
    f0 = track(prepared.samples_16k, 16000)
    if f0 is None:
        continue
    dur = f0.start_s + (len(f0.semitones) - 1) * f0.hop_s
    se = prepared.speech_interval_s[1] if prepared.speech_interval_s else dur
    end = min(dur, se)
    vt = [
        f0.start_s + i * f0.hop_s
        for i, v in enumerate(f0.semitones)
        if v is not None and f0.start_s + i * f0.hop_s <= end + 1e-6
    ]
    lv = vt[-1] if vt else end
    t1, v1 = _voiced_series(f0, max(0.0, end - 0.35), end)
    s_engine = _slope(t1, v1) if v1.size >= 3 else None
    t2, v2 = _voiced_series(f0, max(0.0, lv - 0.35), lv)
    s_lv = _slope(t2, v2) if v2.size >= 3 else None
    t3, v3 = _voiced_series(f0, 0.0, end)
    s_max = None
    if v3.size >= 3:
        best = None
        for i in range(len(t3)):
            for j in range(i + 3, len(t3)):
                if t3[j] - t3[i] > 0.25:
                    break
                s = _slope(t3[i : j + 1], v3[i : j + 1])
                if s is not None and (best is None or abs(s) > abs(best)):
                    best = s
        s_max = best
    tone, conf = nuclear_tone(f0, end_s=se)
    rows.append(
        (p.stem, tone, round(s_engine or 0.0, 1), round(s_lv or 0.0, 1), round(s_max or 0.0, 1))
    )
print("id | engine | engine-win slope | last-voiced slope | max-slope")
for r in rows:
    print(" ", r)
