import sys, statistics
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "src")
from openschwa_eval.datasets.aixmarsec import AixMarsec
from openschwa_engine.audio import decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.prosody import track
from openschwa_engine.prosody.compare import _voiced_series, _slope

settings = Settings(warm_model_on_start=False)
corpus = AixMarsec(Path("data/aix/4"))
cache = {}
rows = []
for unit in corpus.units():
    decoded = cache.get(unit.audio_path)
    if decoded is None:
        decoded = decode_wav(unit.audio_path.read_bytes())
        cache[unit.audio_path] = decoded
    rate = decoded.sample_rate
    start = max(0, int(unit.nucleus_s * rate) - int(0.1 * rate))
    end = min(len(decoded.samples), int(unit.end_s * rate), int(unit.nucleus_s * rate) + int(0.45 * rate))
    if end - start < int(0.3 * rate):
        continue
    prepared = prepare(decoded.samples[start:end], rate, vad_backend=settings.vad_backend)
    t = track(prepared.samples_16k, 16000)
    if t is None or t.semitones is None:
        continue
    duration = t.start_s + (len(t.semitones) - 1) * t.hop_s
    wstart = max(0.0, duration - 0.35)
    times, values = _voiced_series(t, wstart)
    slope = _slope(times, values) if values.size >= 3 else None
    rows.append((unit.block_id, unit.expected_tone, slope, len(values)))

# slope distribution per class
for cls in ("fall", "rise", "fall_rise"):
    ss = [s for (b, e, s, n) in rows if e == cls and s is not None]
    voiced = [n for (b, e, s, n) in rows if e == cls]
    if not ss:
        print(cls, "no slopes"); continue
    print("%s: n=%d slope median=%.1f p10=%.1f p90=%.1f | window-voiced-frames median=%d (min=%d)" % (
        cls, len(ss), statistics.median(ss),
        sorted(ss)[len(ss)//10], sorted(ss)[9*len(ss)//10],
        statistics.median(voiced), min(voiced)))
    for thr in (4, 6, 8, 10, 12):
        falls = sum(1 for s in ss if s <= -thr)
        rises = sum(1 for s in ss if s >= thr)
        print("    thr=%d: fall-side %.2f  rise-side %.2f" % (thr, falls/len(ss), rises/len(ss)))

# block-disjoint threshold sweep for fall vs rise
from collections import Counter
import random
blocks = sorted({b for (b, e, s, n) in rows})
rng = random.Random("42:intonation-blocks")
rng.shuffle(blocks)
n_test = max(1, len(blocks) // 5)
test_blocks = set(blocks[:n_test])
def sweep(subset, best_only=False):
    best = (0, None, None, None)
    for lo in range(2, 21):
        for hi in range(2, 21):
            tp = fp = tn = fn = 0
            for (b, e, s, n) in subset:
                if e not in ("fall", "rise") or s is None:
                    continue
                pred = "fall" if s <= -lo else ("rise" if s >= hi else "abstain")
                if pred == "abstain":
                    continue
                if pred == e:
                    if e == "fall": tp += 1
                    else: tn += 1
                else:
                    if e == "fall": fn += 1
                    else: fp += 1
            if tp + fp + tn + fn == 0:
                continue
            acc = (tp + tn) / (tp + fp + tn + fn)
            cov = (tp + fp + tn + fn) / sum(1 for (b, e, s, n) in subset if e in ("fall", "rise") and s is not None)
            if best[0] is None or acc > best[0]:
                best = (acc, lo, hi, cov)
    return best
cal = [(b, e, s, n) for (b, e, s, n) in rows if b not in test_blocks]
test = [(b, e, s, n) for (b, e, s, n) in rows if b in test_blocks]
print("cal sweep (best acc/cov):", sweep(cal))
print("test at cal-best + all thresholds:", sweep(test))
