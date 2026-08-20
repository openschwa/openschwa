from pathlib import Path
from collections import Counter, defaultdict

root = Path("eval/data/aix/4/VERSION 1 (2009)/Aix-Marsec/TextGrid")
mark_chars = {60:"<",62:">",47:"/",126:"~",42:"*",95:"_",96:chr(96),94:"^",92:chr(92)}

def parse_tiers(path):
    i = 0
    lines = path.read_text(errors="replace").splitlines()
    tiers = {}
    while i < len(lines):
        s = lines[i].strip()
        if s == '"IntervalTier"':
            name = lines[i+1].strip().strip('"')
            i += 5
            ints = []
            for _ in range(int(lines[i-1].strip())):
                x1 = float(lines[i].strip()); x2 = float(lines[i+1].strip()); lab = lines[i+2].strip().strip('"')
                ints.append((x1,x2,lab)); i += 3
            tiers[name] = ints
        elif s == '"TextTier"':
            name = lines[i+1].strip().strip('"')
            i += 5
            pts = []
            for _ in range(int(lines[i-1].strip())):
                t = float(lines[i].strip()); lab = lines[i+1].strip().strip('"')
                pts.append((t,lab)); i += 2
            tiers[name] = pts
        else:
            i += 1
    return tiers

def mark_of(lab):
    if not lab or lab == "_": return None
    ms = [ch for ch in lab if ord(ch) in mark_chars]
    return "".join(ms) if ms else None

names = [p.name[:-len('.TextGrid')] for p in root.rglob('*.TextGrid')]
codes = sorted({n[:-1] for n in names if n[-1] in 'BG' and (n[:-1]+'B' in names) and (n[:-1]+'G' in names)})
print("overlap codes:", len(codes))
pairs = defaultdict(lambda: Counter())
for code in codes:
    tb = parse_tiers(root / code[0] / (code+'B.TextGrid'))
    tg = parse_tiers(root / code[0] / (code+'G.TextGrid'))
    wb = [(x1, lab) for (x1,x2,lab) in tb["Text"] if lab and lab != "_"]
    wg = [(x1, lab) for (x1,x2,lab) in tg["Text"] if lab and lab != "_"]
    for (x1, lab) in wb:
        best = None
        for (y1, lab2) in wg:
            if abs(y1 - x1) < 0.35:
                best = lab2; break
        if best is not None:
            mb, mg = mark_of(lab), mark_of(best)
            if mb and mg:
                pairs[mb][mg] += 1

for mb, c in sorted(pairs.items(), key=lambda kv: -sum(kv[1].values())):
    print(repr(mb), dict(c.most_common(6)))
print("paired marked words:", sum(sum(c.values()) for c in pairs.values()))
