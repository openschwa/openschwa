from pathlib import Path
from collections import Counter, defaultdict
import math

root = Path("eval/data/aix/4/VERSION 1 (2009)/Aix-Marsec/TextGrid")
mark_chars = {60:"<",62:">",47:"/",126:"~",42:"*",95:"_",96:chr(96),94:"^",92:chr(92),44:","}

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

def unit_end_for(ui, x):
    best = None
    for (a, b, lab) in ui:
        if a - 0.01 <= x <= b + 0.01:
            best = b
            if lab.strip().endswith("|"):
                return b
    return best

def glide(mpts, a, b, min_pts=2, min_st=3.0):
    win = sorted([(t, v) for (t, v) in mpts if a <= t <= b])
    if len(win) < min_pts: return None
    t0, v0 = win[0]; t1, v1 = win[-1]
    if v0 <= 0 or v1 <= 0: return None
    exc = 12 * math.log2(v1 / v0)
    if exc <= -min_st: return "fall"
    if exc >= min_st: return "rise"
    return "flat"

per_mark = defaultdict(lambda: Counter())
files = list(root.rglob("*.TextGrid"))
n_last = 0
for p in files:
    try: tiers = parse_tiers(p)
    except Exception: continue
    if "Text" not in tiers: continue
    momel = tiers.get("Valeurs de F0") or tiers.get("F0")
    if not momel: continue
    mpts = []
    for (t, lab) in momel:
        try: mpts.append((t, float(lab)))
        except ValueError: continue
    if not mpts: continue
    ui = tiers.get("UI", [])
    words = [(x1, x2, lab) for (x1, x2, lab) in tiers["Text"] if lab and lab != "_"]
    # group words into units via UI
    units = defaultdict(list)
    for w in words:
        for (a, b, lab) in ui:
            if a - 0.01 <= w[0] <= b + 0.01:
                units[id((a, b))].append(w)
                break
    for key, ws in units.items():
        marked = [w for w in ws if any(ord(ch) in mark_chars for ch in w[2])]
        if not marked: continue
        last = marked[-1]
        x1, x2, lab = last
        marks = [ch for ch in lab if ord(ch) in mark_chars]
        # handle two-char TSM: treat the full mark string
        uend = max(w[1] for w in ws)
        g_word = glide(mpts, x1 - 0.02, x2 + 0.05)
        g_tail = glide(mpts, x2, uend + 0.05)
        if g_word is None and g_tail is None: continue
        n_last += 1
        key2 = "".join(marks)
        per_mark[key2][(g_word or "-") + "/" + (g_tail or "-")] += 1

print("last-marked words analyzed:", n_last)
print()
for m, c in sorted(per_mark.items(), key=lambda kv: -sum(kv[1].values())):
    total = sum(c.values())
    print(repr(m), "n=", total)
    for k in ["fall/flat", "fall/rise", "fall/fall", "rise/flat", "rise/fall", "rise/rise",
              "flat/flat", "flat/rise", "flat/fall", "fall/-", "rise/-", "flat/-", "-/rise", "-/fall", "-/flat", "-/-"]:
        v = c.get(k, 0)
        if v: print(f"    {k}: {v} ({v/total:.2f})")
