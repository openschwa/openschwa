import sys, statistics, random
from pathlib import Path
sys.path.insert(0, "src")
from openschwa_eval.datasets.aixmarsec import AixMarsec
from openschwa_engine.audio import decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.prosody import track

settings = Settings(warm_model_on_start=False)
corpus = AixMarsec(Path("data/aix/4"))
cache = {}

def med_smooth(vals, k=5):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - k//2); hi = min(len(vals), i + k//2 + 1)
        out.append(statistics.median(vals[lo:hi]))
    return out

def slope_of(xs, ys):
    if len(xs) < 3: return None
    xm = sum(xs)/len(xs); ym = sum(ys)/len(ys)
    den = sum((x-xm)**2 for x in xs)
    if den <= 0: return None
    return sum((x-xm)*(y-ym) for x, y in zip(xs, ys)) / den

def glide_feats(sem, t0, hop, win_s=0.25, k=5):
    pts = [(t0 + i*hop, v) for i, v in enumerate(sem) if v is not None]
    if len(pts) < 6: return None
    sm = med_smooth([v for _, v in pts], k)
    xs = [x for x, _ in pts]
    s_min = None; s_max = None; t_min = None; t_max = None
    for i in range(len(pts)):
        for j in range(i+3, len(pts)):
            if xs[j] - xs[i] > win_s + 1e-6: break
            s = slope_of(xs[i:j+1], sm[i:j+1])
            if s is None: continue
            if s_min is None or s < s_min: s_min, t_min = s, xs[i]
            if s_max is None or s > s_max: s_max, t_max = s, xs[i]
    return s_min, s_max, t_min, t_max

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
    if t is None or t.semitones is None: continue
    f = glide_feats(t.semitones, t.start_s, t.hop_s)
    if f is None: continue
    rows.append((unit.block_id, unit.expected_tone, f))

for cls in ("fall", "rise"):
    sub = [r for r in rows if r[1] == cls]
    mn = [r[2][0] for r in sub if r[2][0] is not None]
    mx = [r[2][1] for r in sub if r[2][1] is not None]
    print("%s n=%d  min-slope median=%.1f p25=%.1f  |  max-slope median=%.1f p75=%.1f" % (
        cls, len(sub), statistics.median(mn), sorted(mn)[len(mn)//4],
        statistics.median(mx), sorted(mx)[3*len(mx)//4]))

blocks = sorted({r[0] for r in rows})
rng = random.Random("42:intonation-blocks"); rng.shuffle(blocks)
n_test = max(1, len(blocks)//5)
test_blocks = set(blocks[:n_test])
cal = [r for r in rows if r[0] not in test_blocks]
test = [r for r in rows if r[0] in test_blocks]
def acc(subset, th):
    tp = fn = tn = fp = 0
    for (b, e, f) in subset:
        if e not in ("fall", "rise"): continue
        s_min, s_max, t_min, t_max = f
        if s_min is None and s_max is None: continue
        if s_min is not None and s_min <= -th and (s_max is None or abs(s_min) >= abs(s_max)):
            pred = "fall"
        elif s_max is not None and s_max >= th:
            pred = "rise"
        else:
            continue
        if pred == e:
            if e == "fall": tp += 1
            else: tn += 1
        else:
            if e == "fall": fn += 1
            else: fp += 1
    n = tp+fn+tn+fp
    return (tp+tn)/n if n else 0, tp/(tp+fn) if (tp+fn) else 0, tn/(tn+fp) if (tn+fp) else 0, n
best = None
for th in range(2, 31):
    a, fr, rr, n = acc(cal, th)
    if best is None or a > best[0]: best = (a, th, fr, rr, n)
print("cal best:", best)
for th in (best[1], best[1]+4, best[1]-4 if best[1] > 6 else 6):
    a, fr, rr, n = acc(test, th)
    print("test th=%d: acc=%.3f fall_recall=%.2f rise_recall=%.2f n=%d" % (th, a, fr, rr, n))
