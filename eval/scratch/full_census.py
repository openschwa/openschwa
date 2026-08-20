from pathlib import Path
from collections import Counter

root = Path("eval/data/aix/4/VERSION 1 (2009)/Aix-Marsec/TextGrid")
files = list(root.rglob("*.TextGrid"))
allch = Counter()
# words carrying any non-alphanumeric lead
per_tier = Counter()
import re
for p in files:
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
            for ch in body:
                allch[ch] += 1
nonword = {ch: c for ch, c in allch.items() if not ch.isalnum() and ch not in ' .'}
for ch, c in sorted(nonword.items(), key=lambda kv: -kv[1]):
    print(repr(ch), c)
