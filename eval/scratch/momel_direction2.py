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
    # UI tier: intervals with label ending in | or ||
    best = None
    for (a, b, lab) in ui:
        if a <= x <= b:
            best = b
            if lab.strip().endswith("|"):
                return b
    return best

per_mark = defaultdict(lambda: Counter())
files = list(root.rglob("*.TextGrid"))
for p in files:
    try:
        tiers = parse_tiers(p)
    except Exception:
        continue
    if "Text" not in tiers: continue
    momel = tiers.get("Valeurs de F0") or tiers.get("F0")
    if not momel: continue
    mpts = []
    for (t, lab) in momel:
        try: mpts.append((t, float(lab)))
        except ValueError: continue
    if not mpts: continue
    ui = tiers.get("UI", [])
    for (x1, x2, lab) in tiers["Text"]:
        if not lab or lab == "_": continue
        marks = [ch for ch in lab if ord(ch) in mark_chars]
        if not marks: continue
        uend = unit_end_for(ui, x1)
        if uend is None: uend = x2 + 0.4
        win = [(t, v) for (t, v) in mpts if x1 - 0.05 <= t <= max(uend, x2) + 0.1]
        if len(win) < 2: continue
        win.sort()
        t0, v0 = win[0]; t1, v1 = win[-1]
        if v0 <= 0 or v1 <= 0: continue
        exc = 12 * math.log2(v1 / v0)
        dur = max(t1 - t0, 0.001)
        slope = exc / dur
        if slope <= -4: d = "FALL"
        elif slope >= 4: d = "RISE"
        else: d = "FLAT"
        for m in marks:
            per_mark[m][d] += 1

for m, c in sorted(per_mark.items(), key=lambda kv: -sum(kv[1].values())):
    total = sum(c.values())
    print(repr(m), "n=", total,
          "FALL=%.2f" % (c["FALL"]/total), "RISE=%.2f" % (c["RISE"]/total), "FLAT=%.2f" % (c["FLAT"]/total))
