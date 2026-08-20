import json
from pathlib import Path
from openschwa_engine.audio import decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.prosody import track
from openschwa_engine.prosody.compare import _voiced_series, _theil_sen

settings = Settings(warm_model_on_start=False)
manifest = json.load(open("manifests/intonation-recordings.json"))
items = {i["id"]: i for i in manifest["items"]}
excl = set(json.load(open("data/recordings/exclusions.json")))

data = []
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
    slope = _theil_sen(t, v) if v.size >= 3 else None
    mid = float(t[0]) + (float(t[-1]) - float(t[0])) / 2.0 if v.size >= 3 else 0.0
    first = _theil_sen(t[t <= mid], v[t <= mid]) if v.size >= 3 and v[t <= mid].size >= 3 else None
    second = _theil_sen(t[t > mid], v[t > mid]) if v.size >= 3 and v[t > mid].size >= 3 else None
    data.append((stem, item["tone"], slope, first, second))

print("verified core items:", len(data))

def score(fall_t, rise_t, fr_t):
    correct = 0
    for stem, tone, slope, first, second in data:
        if slope is None:
            continue
        if first is not None and second is not None and first <= -fr_t and second >= fr_t:
            det = "fall_rise"
        elif slope >= rise_t:
            det = "rise"
        elif slope <= -fall_t:
            det = "fall"
        else:
            det = "level"
        if det == tone:
            correct += 1
    return correct / len(data)

results = []
for fall_t in (4, 5, 6, 7, 8):
    for rise_t in (5, 6, 7, 8, 9):
        for fr_t in (10, 12, 14, 16):
            results.append((score(fall_t, rise_t, fr_t), fall_t, rise_t, fr_t))
results.sort(reverse=True)
for acc, fall_t, rise_t, fr_t in results[:12]:
    print("acc=%.4f fall_t=%d rise_t=%d fr_t=%d" % (acc, fall_t, rise_t, fr_t))
print()
print("per-class at the top combo:")
best = results[0]
for stem, tone, slope, first, second in data:
    det = None
    if slope is None:
        det = "none"
    elif first is not None and second is not None and first <= -best[3] and second >= best[3]:
        det = "fall_rise"
    elif slope >= best[2]:
        det = "rise"
    elif slope <= -best[1]:
        det = "fall"
    else:
        det = "level"
    if det != tone:
        print("  miss %s %s -> %s (slope=%.1f first=%.1f second=%.1f)" % (stem, tone, det, slope or 0.0, first or 0.0, second or 0.0))
