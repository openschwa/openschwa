from pathlib import Path
from collections import Counter

root = Path("eval/data/aix/4/VERSION 1 (2009)/Aix-Marsec/TextGrid")
hash_ctx = Counter(); comma_ctx = Counter(); apo_ctx = Counter()
for p in list(root.rglob("*.TextGrid"))[:60]:
    i = 0
    lines = p.read_text(errors="replace").splitlines()
    in_text = False
    prev = None
    for line in lines:
        s = line.strip()
        if s == '"IntervalTier"':
            prev = "tier"; in_text = False; continue
        if prev == "tier":
            in_text = (s == '"Text"')
            prev = None
            continue
        if in_text and s.startswith('"'):
            body = s.strip('"')
            if '#' in body: hash_ctx[body] += 1
            if ',' in body: comma_ctx[body] += 1
            if "'" in body: apo_ctx[body] += 1
print("== # contexts ==")
for k, v in hash_ctx.most_common(8): print(repr(k), v)
print("== , contexts ==")
for k, v in comma_ctx.most_common(12): print(repr(k), v)
print("== ' contexts ==")
for k, v in apo_ctx.most_common(8): print(repr(k), v)
