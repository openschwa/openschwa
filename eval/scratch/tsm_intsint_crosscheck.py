from pathlib import Path
from collections import Counter, defaultdict

root = Path("eval/data/aix/4/VERSION 1 (2009)/Aix-Marsec/TextGrid")
files = list(root.rglob("*.TextGrid"))
print("textgrids:", len(files))

mark_chars = {60: "<", 62: ">", 47: "/", 126: "~", 42: "*", 95: "_", 96: chr(96), 94: "^", 92: chr(92)}

def parse_short(path):
    lines = path.read_text(errors="replace").splitlines()
    tiers = {}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == '"IntervalTier"':
            name = lines[i+1].strip().strip('"')
            xmin = float(lines[i+2].strip())
            xmax = float(lines[i+3].strip())
            n = int(lines[i+4].strip())
            i += 5
            ints = []
            for _ in range(n):
                x1 = float(lines[i].strip()); x2 = float(lines[i+1].strip()); lab = lines[i+2].strip().strip('"')
                ints.append((x1, x2, lab)); i += 3
            tiers[name] = ("interval", ints)
        elif s == '"TextTier"':
            name = lines[i+1].strip().strip('"')
            xmin = float(lines[i+2].strip())
            xmax = float(lines[i+3].strip())
            n = int(lines[i+4].strip())
            i += 5
            pts = []
            for _ in range(n):
                t = float(lines[i].strip()); lab = lines[i+1].strip().strip('"')
                pts.append((t, lab)); i += 2
            tiers[name] = ("points", pts)
        else:
            i += 1
    return tiers

rank = {"B": 0, "S": 1, "D": 1, "M": 2, "U": 3, "T": 4, "H": 5, "L": -1}
per_mark = defaultdict(lambda: Counter())

for p in files:
    tiers = parse_short(p)
    if "Text" not in tiers or "Intsint" not in tiers:
        continue
    _, words = tiers["Text"]
    _, pts = tiers["Intsint"]
    for (x1, x2, lab) in words:
        if not lab or lab == "_":
            continue
        marks = [ch for ch in lab if ord(ch) in mark_chars]
        if not marks:
            continue
        win = [l for (t, l) in pts if x1 - 0.1 <= t <= x2 + 0.3]
        if len(win) < 2:
            continue
        seq = [l for l in win if l in rank]
        if len(seq) < 2:
            continue
        a, b = rank[seq[0]], rank[seq[-1]]
        if b - a >= 1: move = "UP"
        elif a - b >= 1: move = "DOWN"
        else: move = "FLAT"
        for m in marks:
            per_mark[m][move] += 1

for m, c in sorted(per_mark.items(), key=lambda kv: -sum(kv[1].values())):
    total = sum(c.values())
    print(repr(m), "n=", total, " ".join(k + "=" + str(v) for k, v in c.most_common()))
