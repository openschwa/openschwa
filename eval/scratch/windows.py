import sys, statistics, random
from pathlib import Path
from collections import Counter
sys.path.insert(0, "src")
from openschwa_eval.datasets.aixmarsec import AixMarsec
from openschwa_engine.audio import decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.prosody import track

settings = Settings(warm_model_on_start=False)
corpus = AixMarsec(Path("data/aix/4"))
cache = {}

def lin_slope(pts):
    if len(pts) < 3: return None
    xs = [x for x, _ in pts]; ys = [y for _, y in pts]
    xm = sum(xs)/len(xs); ym = sum(ys)/len(ys)
    den = sum((x-xm)**2 for x in xs)
    if den <= 0: return None
    return sum((x-xm)*(y-ym) for x, y in zip(xs, ys)) / den

def windows(t, t0, slice_start, slice_end):
    out = {}
    sem = t.semitones
    pts = [(t0 + i*t.hop_s, v) for i, v in enumerate(sem) if v is not None]
    if len(pts) < 3: return out
    out["whole"] = lin_slope(pts)
    dur = t0 + (len(sem)-1)*t.hop_s
    # terminal 0.35 (engine)
    tp = [(x, v) for x, v in pts if x >= dur - 0.35]
    out["terminal"] = lin_slope(tp)
    # first 0.3s of the slice (the nucleus glide should start right after pre-roll)
    fp = [(x, v) for x, v in pts if x <= t0 + 0.3]
    out["first03"] = lin_slope(fp)
    # max-slope 0.2s subwindow (glide detector)
    best = None
    for i in range(len(pts)):
        for j in range(i+3, len(pts)):
            if pts[j][0] - pts[i][0] > 0.22: break
            s = lin_slope(pts[i:j+1])
            if s is None: continue
            if best is None or abs(s) > abs(best): best = s
    out["maxwin"] = best
    # trim check: how much did prepare/VAD cut
    out["slice_dur"] = slice_end - slice_start
    out["track_dur"] = dur
    return out

rows = []
for unit in corpus.units():
    decoded = cache.get(unit.audio_path)
    if decoded is None:
        decoded = decode_wav(unit.audio_path.read_bytes())
        cache[unit.audio_path] = decoded
    rate = decoded.sample_rate
    start = max(0, int(unit.nucleus_s * rate) - int(0.1 * rate))
    end = min(len(decoded.samples), int(unit.end_s * rate), int(unit.nucleus_s * rate) + int(0.45 * rate))
    if end - start < int(0.3 * rate):
        continue
    prepared = prepare(decoded.samples[start:end], rate, vad_backend=settings.vad_backend)
    t = track(prepared.samples_16k, 16000)
    if t is None or t.semitones is None:
        continue
    w = windows(t, t.start_s, start/rate, end/rate)
    if not w: continue
    rows.append((unit.block_id, unit.expected_tone, w))

for cls in ("fall", "rise"):
    sub = [r for r in rows if r[1] == cls]
    print("%s n=%d" % (cls, len(sub)))
    for key in ("whole", "terminal", "first03", "maxwin"):
        vals = [r[2][key] for r in sub if r[2].get(key) is not None]
        if not vals: continue
        print("  %-9s median=%7.1f  p25=%7.1f  p75=%7.1f  neg=%.2f pos=%.2f" % (
            key, statistics.median(vals),
            sorted(vals)[len(vals)//4], sorted(vals)[3*len(vals)//4],
            sum(1 for v in vals if v <= -8)/len(vals),
            sum(1 for v in vals if v >= 8)/len(vals)))
    trims = [r[2]["slice_dur"] - r[2]["track_dur"] for r in sub]
    print("  slice_vs_track: median trim=%.2fs (negative=track longer)" % statistics.median(trims))

# block-disjoint accuracy for each window key, threshold -8/+8, fall vs rise
blocks = sorted({r[0] for r in rows})
rng = random.Random("42:intonation-blocks"); rng.shuffle(blocks)
n_test = max(1, len(blocks)//5)
test_blocks = set(blocks[:n_test])
for key in ("whole", "terminal", "first03", "maxwin"):
    for lo, hi in ((8, 8), (6, 6), (4, 8)):
        tp = fn = tn = fp = 0
        for (b, e, w) in rows:
            if b not in test_blocks or e not in ("fall", "rise"): continue
            s = w.get(key)
            if s is None: continue
            pred = "fall" if s <= -lo else ("rise" if s >= hi else None)
            if pred is None: continue
            if pred == e:
                if e == "fall": tp += 1
                else: tn += 1
            else:
                if e == "fall": fn += 1
                else: fp += 1
        n = tp+fn+tn+fp
        print("%s lo=%d hi=%d: test n=%d acc=%.3f (fall_recall=%.2f rise_recall=%.2f)" % (
            key, lo, hi, n, (tp+tn)/n if n else 0, tp/(tp+fn) if (tp+fn) else 0, tn/(tn+fp) if (tn+fp) else 0))
