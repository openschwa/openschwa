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
def tone_class(mark):
    if mark in (chr(92), chr(96)): return "fall"
    if mark == "/": return "rise"
    if mark in (chr(96)+"/", "/"+chr(96)): return "fall_rise"
    return "other:" + mark
def nuclear_words(tiers):
    words = [(x1, x2, lab) for (x1, x2, lab) in tiers["Text"] if lab and lab != "_"]
    ui = tiers.get("UI", [])
    units = defaultdict(list)
    for w in words:
        for (a, b, lab) in ui:
            if a - 0.01 <= w[0] <= b + 0.01:
                units[(a, b)].append(w); break
        else:
            units[(None, w[0])].append(w)
    out = []
    for key, ws in units.items():
        marked = [(w, tone_mark(w[2])) for w in ws if tone_mark(w[2])]
        if not marked: continue
        (x1, x2, lab), mark = marked[-1]
        out.append((key, tone_class(mark), x1))
    return out

names = [p.name[:-len('.TextGrid')] for p in root.rglob('*.TextGrid')]
codes = sorted({n[:-1] for n in names if n[-1] in 'BG' and (n[:-1]+'B' in names) and (n[:-1]+'G' in names)})
agree = Counter(); b_total = Counter(); n_matched = 0
for code in codes:
    tb = parse_tiers(root / code[0] / (code+'B.TextGrid'))
    tg = parse_tiers(root / code[0] / (code+'G.TextGrid'))
    nb = nuclear_words(tb); ng = nuclear_words(tg)
    # same-segmentation match: unit edges within 0.2s
    gb = {(a, b): (cls, x1) for (a, b), cls, x1 in ng}
    for (a, b), cls, x1 in nb:
        best = None
        for (ga, gb_), (gcls, gy) in gb.items():
            if abs(ga - a) < 0.2 and abs(gb_ - b) < 0.2:
                best = gcls; break
        if best is None: continue
        n_matched += 1
        b_total[cls] += 1
        if cls == best: agree[cls] += 1
print("matched same-segmentation nuclei:", n_matched)
print("distribution (B):", dict(b_total))
for cls in sorted(b_total):
    a = agree.get(cls, 0); t = b_total[cls]
    print(f"  {cls}: {a}/{t} = {a/t:.2f}")
total = sum(b_total.values()); ta = sum(agree.values())
print(f"OVERALL: {ta}/{total} = {ta/total:.3f}")
sub = {c: b_total.get(c, 0) for c in ("fall", "rise", "fall_rise")}
sub_a = sum(agree.get(c, 0) for c in sub)
print(f"RESTRICTED fall/rise/fall_rise: {sub_a}/{sum(sub.values())} = {sub_a/sum(sub.values()):.3f}")
