from pathlib import Path
from collections import Counter, defaultdict
import math, sys
sys.path.insert(0, "src")
from openschwa_engine.audio import decode_wav, prepare
from openschwa_engine.prosody import track

root = Path("data/aix/4/VERSION 1 (2009)/Aix-Marsec/TextGrid")
sounds = Path("data/aix/4/Sounds")
mark_chars = {60:"<",62:">",47:"/",126:"~",42:"*",95:"_",96:chr(96),94:"^",92:chr(92)}

def parse_tiers(path):
    i = 0
    lines = path.read_text(errors="replace").splitlines()
    tiers = {}
    while i < len(lines):
        s = lines[i].strip()
        if s == '"IntervalTier"':
            name = lines[i+1].strip().strip('"')
            n = int(lines[i+4].strip()); i += 5
            ints = []
            for _ in range(n):
                x1 = float(lines[i].strip()); x2 = float(lines[i+1].strip()); lab = lines[i+2].strip().strip('"')
                ints.append((x1,x2,lab)); i += 3
            tiers[name] = ints
        elif s == '"TextTier"':
            name = lines[i+1].strip().strip('"')
            n = int(lines[i+4].strip()); i += 5
            pts = []
            for _ in range(n):
                t = float(lines[i].strip()); lab = lines[i+1].strip().strip('"')
                pts.append((t,lab)); i += 2
            tiers[name] = pts
        else:
            i += 1
    return tiers

def tone_mark(lab):
    if not lab: return ""
    return "".join(ch for ch in lab if ord(ch) in mark_chars and ch not in ",|#")

def slope_st(sem, t0, hop, a, b, min_pts=3):
    pts = [(t0 + i*hop, v) for i, v in enumerate(sem) if v is not None and a <= t0 + i*hop <= b]
    if len(pts) < min_pts: return None
    xs = [x for x, _ in pts]; ys = [y for _, y in pts]
    xm = sum(xs)/len(xs); ym = sum(ys)/len(ys)
    den = sum((x-xm)**2 for x in xs)
    if den <= 0: return None
    return sum((x-xm)*(y-ym) for x, y in zip(xs, ys)) / den

per_mark = defaultdict(lambda: Counter())
files = sorted(root.rglob("*.TextGrid"))
cache = {}
n_units = 0
for p in files:
    try: tiers = parse_tiers(p)
    except Exception: continue
    if "Text" not in tiers or "UI" not in tiers: continue
    words = [(x1, x2, lab) for (x1, x2, lab) in tiers["Text"] if lab and lab != "_"]
    ui = tiers["UI"]
    units = defaultdict(list)
    for w in words:
        for (a, b, lab) in ui:
            if a - 0.01 <= w[0] <= b + 0.01:
                units[(a, b)].append(w); break
    # block wav for this passage
    block = p.stem[0]
    wav = sounds / block / (p.stem + ".wav")
    if not wav.exists():
        # maybe block wav contains multiple passages: try all wavs in block dir
        candidates = sorted(sounds.glob(block + "/*.wav"))
        wav = None
        for c in candidates:
            wav = c; break
    if wav is None or not wav.exists(): continue
    decoded = cache.get(str(wav))
    if decoded is None:
        decoded = decode_wav(wav.read_bytes()); cache[str(wav)] = decoded
    rate = decoded.sample_rate
    for (a, b), ws in units.items():
        marked = [(w, tone_mark(w[2])) for w in ws if tone_mark(w[2])]
        if not marked: continue
        (x1, x2, lab), mark = marked[-1]
        n_units += 1
        s = int(x1 * rate) - int(0.15 * rate)
        e = int(b * rate) + int(0.1 * rate)
        s = max(0, s); e = min(len(decoded.samples), e)
        if e - s < int(0.3 * rate): continue
        prepared = prepare(decoded.samples[s:e], rate)
        t = track(prepared.samples_16k, 16000)
        if t is None or t.semitones is None: continue
        t0 = t.start_s
        g_word = slope_st(t.semitones, t0, t.hop_s, x1 - s/rate - 0.02, x2 - s/rate + 0.05)
        g_tail = slope_st(t.semitones, t0, t.hop_s, x1 - s/rate, b - s/rate)
        d_word = "fall" if (g_word or 0) <= -6 else ("rise" if (g_word or 0) >= 6 else "flat")
        d_tail = "fall" if (g_tail or 0) <= -4 else ("rise" if (g_tail or 0) >= 4 else "flat")
        per_mark[mark][d_word + "/" + d_tail] += 1

print("units analyzed:", n_units)
print()
for m, c in sorted(per_mark.items(), key=lambda kv: -sum(kv[1].values())):
    total = sum(c.values())
    print(repr(m), "n=", total)
    for k in ["fall/flat", "fall/rise", "fall/fall", "rise/flat", "rise/rise", "rise/fall",
              "flat/flat", "flat/rise", "flat/fall"]:
        v = c.get(k, 0)
        if v >= total * 0.05:
            print(f"    {k}: {v} ({v/total:.2f})")
