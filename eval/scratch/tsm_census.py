from pathlib import Path
from collections import Counter

marks = Counter()
names = {60: "<", 62: ">", 47: "/", 126: "~", 42: "*", 95: "_", 96: chr(96), 94: "^", 92: chr(92)}
files = list(Path("eval/data/aix/4/VERSION 1 (2009)/Aix-Marsec/TextGrid").rglob("*.TextGrid"))
print("textgrids:", len(files))
in_text_tier = False
for p in files:
    in_text_tier = False
    prev = None
    for line in p.read_text(errors="replace").splitlines():
        s = line.strip()
        if s == '"IntervalTier"':
            in_text_tier = False
            prev = "tier"
            continue
        if prev == "tier":
            in_text_tier = s == '"Text"'
            prev = None
            continue
        if in_text_tier and s.startswith('"'):
            body = s.strip('"')
            if body and body != "_":
                for ch in body:
                    o = ord(ch)
                    if o in names:
                        marks[names[o]] += 1
print(marks)
