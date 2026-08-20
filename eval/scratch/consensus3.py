import sys
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, "src")
from openschwa_eval.datasets.aixmarsec import AixMarsec

corpus = AixMarsec(Path("data/aix/4"))
units_by_pass = defaultdict(list)
for u in corpus.units():
    units_by_pass[u.passage_id[:5] + u.annotator].append(u)

# match B units to G units by unit start (0.3s), class set = fall/rise/fall_rise
agree = Counter()
disagree = Counter()
n_unmatched = 0
for key, us in units_by_pass.items():
    if key[-1] != "B":
        continue
    gs = units_by_pass.get(key[:5] + "G")
    if not gs:
        continue
    for ub in us:
        ug = None
        for g in gs:
            if abs(g.start_s - ub.start_s) < 0.3:
                ug = g
                break
        if ug is None:
            n_unmatched += 1
            continue
        if ug.expected_tone == ub.expected_tone:
            agree[ub.expected_tone] += 1
        else:
            disagree[(ub.expected_tone, ug.expected_tone)] += 1
total = sum(agree.values()) + sum(disagree.values())
print("time-matched pairs (fall/rise/fall_rise):", total, "unmatched:", n_unmatched)
print("agree by class:", dict(agree))
print("disagree pairs:", dict(disagree))
print("overall 3-class agreement: %.4f" % (sum(agree.values()) / total))
# binary question: fall vs rise among units labeled fall or rise by BOTH
bin_agree = agree["fall"] + agree["rise"]
bin_dis = sum(v for (a, b), v in disagree.items() if a in ("fall", "rise") and b in ("fall", "rise"))
print("binary fall-vs-rise (both annotators used fall/rise): %d/%d = %.4f" % (bin_agree, bin_agree + bin_dis, bin_agree / (bin_agree + bin_dis)))
# fall vs everything-else: B says fall; does G also say fall?
b_fall = agree["fall"] + sum(v for (a, b), v in disagree.items() if a == "fall")
g_fall = agree["fall"]
print("fall reproducibility: B=fall & G=fall %d / B=fall %d = %.4f" % (g_fall, b_fall, g_fall / b_fall))
