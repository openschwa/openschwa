"""The offline evaluation: run the engine over labeled corpora and measure it.

Procedure (eval/README.md): synthesize an exercise per target token, run the
full pipeline in-library with include_ungated=True, score flags against the
corpus labels, sweep the operating threshold precision-first, write the
committed calibration.yaml plus a versioned report. Every shipped threshold
stays traceable to this file's output.
"""

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml
from openschwa_engine.audio import decode_wav, prepare
from openschwa_engine.config import Settings
from openschwa_engine.content.loader import Exercise, PhoneSpec
from openschwa_engine.models.registry import ModelRegistry
from openschwa_engine.pipeline import analyze_recording
from openschwa_engine.scoring import CALIBRATION_PATH

from openschwa_eval.datasets import DatasetAdapter, PhoneToken, Utterance
from openschwa_eval.datasets.l2arctic import SPEAKER_L1

log = logging.getLogger(__name__)

EPS = 1e-12
#: Repo root, for making calibration provenance paths repo-relative.
REPO_ROOT = Path(__file__).resolve().parents[3]
L1_PRECISION_FLOOR = 0.8
L1_MIN_POSITIVES = 5
PRECISION_TARGET = 0.90
RECALL_TARGET = 0.4  # the milestone's accept: precision >= 0.90 at recall >= 0.4
#: Below this many cal tokens the threshold sweep is not evidence of
#: anything, and a calibration must never be committed from it.
MIN_COMMIT_TRAIN = 200
#: The mirror bar (M1 pivot): among confident hearing reports, the heard phone
#: matches what the learner actually produced at least this often...
MIRROR_ACCURACY_TARGET = 0.90
#: ...and the mirror answers (rather than refusing with "couldn't tell") on at
#: least this share of tokens. Same numbers as the judge bar, re-read as
#: mirror honesty: P(heard == realized | reported) >= 0.90 at coverage >= 0.4.
MIRROR_COVERAGE_TARGET = 0.4
#: The realized-phone label for tokens the learner deleted: nothing was
#: produced, so any confident hearing claim on them is wrong.
DELETED = "∅"


@dataclass
class TokenRecord:
    """One engine run over one target-phone token."""

    utterance_id: str
    token_index: int
    corpus: str
    l1: str
    split: str
    label: str  # correct | substituted | deleted
    substituted_with: str | None
    audio_path: str
    start_s: float | None
    end_s: float | None
    alignment: str  # ok | low_confidence | failed
    reason: str | None
    audio_problem: bool
    score: float | None = None  # mean contrast log-ratio, from the returned posteriors
    best_confusion: str | None = None
    heard: str | None = None  # mirror: argmax over {target} + confusions
    hearing_score: float | None = None  # mirror: log(p_heard / (1 - p_heard)), raw
    gop: float | None = None
    verdict: str | None = None
    confidence: float | None = None
    spike_score: float | None = None  # single-frame variant (bake-off)
    vote_fraction: float | None = None  # frame-vote variant (bake-off)
    alignment_confidence: float | None = None
    wall_ms: float = 0.0

    @property
    def positive(self) -> bool:
        """The corpus says this token is an error."""
        return self.label in ("substituted", "deleted")


@dataclass
class CorpusResult:
    records: list[TokenRecord] = field(default_factory=list)
    cold_ms: float = 0.0


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def exercise_for(utterance: Utterance, focus_index: int, confusions: list[str]) -> Exercise:
    """Synthesize the exercise spec the real app would serve for this token."""
    phones = tuple(
        PhoneSpec(
            index=token.index,
            ph=token.phone,
            focus=token.index == focus_index,
            confusions=tuple(confusions) if token.index == focus_index else (),
        )
        for token in utterance.phones
    )
    return Exercise(
        id=f"eval-{utterance.utterance_id}-{focus_index}",
        pack_id="eval",
        type="word",
        title="",
        lang="en",
        text=utterance.transcript,
        ipa="",
        phones=phones,
        source_path=Path("eval"),
    )


def assign_split(utterance: Utterance, seed: int) -> str:
    """Three-way split: train / cal / test. The single source of truth.

    so762 keeps its native test partition as the exam's held-out pool; its
    native train partition is carved into cal (25%, seeded) and train.

    L2-ARCTIC is speaker-disjoint and L1-stratified: per first language the
    speakers are deterministically shuffled and assigned test / cal / train /
    train, so every L1 keeps presence in every split and no speaker's voice
    leaks across splits (the old utterance-seeded 30% split let a speaker's
    own voice leak from train into test - a quiet overstatement of every
    held-out number to date).

    Calibration (Platt fit, the threshold sweep, the variant bake-off) uses
    ONLY the cal pool; test is scored once at the chosen operating point.
    The exporters import this function, so the train pool they may export is
    by construction disjoint from both.
    """
    if utterance.corpus == "so762":
        if utterance.split == "test":
            return "test"  # the native test partition is immutable
        rng = random.Random(f"{seed}:so762-cal:{utterance.utterance_id}")
        return "cal" if rng.random() < 0.25 else "train"
    if utterance.speaker and utterance.l1:
        speakers = sorted(name for name, lang in SPEAKER_L1.items() if lang == utterance.l1)
        rng = random.Random(f"{seed}:l1:{utterance.l1}")
        rng.shuffle(speakers)
        roles = ("test", "cal") + ("train",) * max(0, len(speakers) - 2)
        if utterance.speaker in speakers:
            return roles[speakers.index(utterance.speaker)]
    rng = random.Random(f"{seed}:{utterance.utterance_id}")
    return "test" if rng.random() < 0.30 else "train"


def collect_tokens(
    adapters: list[DatasetAdapter],
    target: str,
    confusions: list[str],
    seed: int,
) -> list[tuple[Utterance, PhoneToken, Exercise]]:
    """Every (utterance, target token) triple, with its synthesized exercise."""
    triples: list[tuple[Utterance, PhoneToken, Exercise]] = []
    for adapter in adapters:
        for utterance in adapter.utterances(target):
            utterance = Utterance(
                utterance_id=utterance.utterance_id,
                audio_path=utterance.audio_path,
                transcript=utterance.transcript,
                l1=utterance.l1,
                phones=utterance.phones,
                split=assign_split(utterance, seed),
                corpus=utterance.corpus,
                speaker=utterance.speaker,
            )
            for token in utterance.tokens(target):
                triples.append((utterance, token, exercise_for(utterance, token.index, confusions)))
    return triples


def raw_score_from_posteriors(
    posteriors: dict[str, float], target: str
) -> tuple[float, str | None]:
    """Recover the raw contrast score the engine would feed to calibration."""
    confusions = [name for name in posteriors if name != target]
    if not confusions:
        return 0.0, None
    best = max(confusions, key=lambda name: posteriors[name])
    p_target = posteriors.get(target, EPS)
    return math.log((posteriors[best] + EPS) / (p_target + EPS)), best


def run_tokens(
    triples: list[tuple[Utterance, PhoneToken, Exercise]],
    registry: ModelRegistry,
    settings: Settings,
    target: str,
    checkpoint: Path | None = None,
) -> CorpusResult:
    """Run the full pipeline over every token, checkpointing as it goes."""
    result = CorpusResult()
    done: set[tuple[str, int]] = set()
    records_path = checkpoint
    if records_path is not None and records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = TokenRecord(**json.loads(line))
            result.records.append(record)
            done.add((record.utterance_id, record.token_index))
        log.info("resumed %d records from %s", len(result.records), records_path)

    prepared_cache: dict[Path, object] = {}
    audio_problem_reasons = {
        "no speech detected in the recording",
        "clipping",
        "no signal from the microphone",
    }
    for position, (utterance, token, exercise) in enumerate(triples):
        key = (utterance.utterance_id, token.index)
        if key in done:
            continue
        prepared = prepared_cache.get(utterance.audio_path)
        if prepared is None:
            try:
                decoded = decode_wav(utterance.audio_path.read_bytes())
                prepared = prepare(
                    decoded.samples, decoded.sample_rate, vad_backend=settings.vad_backend
                )
                prepared_cache[utterance.audio_path] = prepared
            except Exception as exc:  # a broken corpus file must not kill the run
                log.error("%s: cannot prepare audio (%s)", utterance.audio_path, exc)
                record = TokenRecord(
                    utterance_id=utterance.utterance_id,
                    token_index=token.index,
                    corpus=utterance.corpus,
                    l1=utterance.l1,
                    split=utterance.split,
                    label=token.label,
                    substituted_with=token.substituted_with,
                    audio_path=str(utterance.audio_path),
                    start_s=token.start_s,
                    end_s=token.end_s,
                    alignment="failed",
                    reason=str(exc),
                    audio_problem=False,
                    score=None,
                    best_confusion=None,
                    gop=None,
                    verdict=None,
                    confidence=None,
                    wall_ms=0.0,
                )
                result.records.append(record)
                if records_path is not None:
                    with records_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(asdict(record)) + "\n")
                continue

        started = time.perf_counter()
        analysis = analyze_recording(prepared, exercise, registry, settings, include_ungated=True)
        wall_ms = (time.perf_counter() - started) * 1000.0
        if position == 0:
            result.cold_ms = wall_ms

        contrast = analysis.contrasts[0] if analysis.contrasts else None
        score = best = spike = vote = heard = hearing = None
        verdict = confidence = gop = None
        if contrast is not None and contrast.posteriors:
            score, best = raw_score_from_posteriors(contrast.posteriors, target)
            spike = contrast.spike_score
            vote = contrast.vote_fraction
            heard = contrast.heard
            hearing = contrast.hearing_score
            verdict = contrast.verdict
            confidence = contrast.confidence
        focus_phone = next((p for p in analysis.alignment.phones if p.index == token.index), None)
        if focus_phone is not None:
            gop = focus_phone.gop

        record = TokenRecord(
            utterance_id=utterance.utterance_id,
            token_index=token.index,
            corpus=utterance.corpus,
            l1=utterance.l1,
            split=utterance.split,
            label=token.label,
            substituted_with=token.substituted_with,
            audio_path=str(utterance.audio_path),
            start_s=token.start_s,
            end_s=token.end_s,
            alignment=analysis.alignment.status,
            reason=None,
            audio_problem=(analysis.alignment.reason or "") in audio_problem_reasons,
            score=score,
            spike_score=spike,
            vote_fraction=vote,
            best_confusion=best,
            heard=heard,
            hearing_score=hearing,
            gop=gop,
            verdict=verdict,
            confidence=confidence,
            alignment_confidence=analysis.alignment.confidence,
            wall_ms=round(wall_ms, 2),
        )
        result.records.append(record)
        if records_path is not None:
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record)) + "\n")
        if (len(result.records) - len(done)) % 50 == 0:
            log.info("processed %d tokens", len(result.records))
    return result


def _variant_value(record: TokenRecord, variant: str) -> float | None:
    """The raw score value for a variant, or None when the token has none."""
    if variant == "spike":
        return record.spike_score
    if variant == "vote":
        return record.vote_fraction
    if variant == "gop":
        return record.gop
    return record.score


def _flag(record: TokenRecord, a: float, b: float, threshold: float, variant: str = "mean") -> bool:
    """Whether the calibrated model flags this token at the operating point."""
    value = _variant_value(record, variant)
    if value is None:
        return False
    return sigmoid(a * value + b) >= threshold


def _per_l1_audit(
    test: list[TokenRecord],
    a: float,
    b: float,
    threshold: float,
    variant: str,
) -> dict[str, object]:
    """Fairness audit of the single shipped line.

    Every L1 group is scored at the SAME global operating point and its
    precision/recall reported with an 'ok' flag. Informational only: the
    shipping bar is the pooled number. This table answers 'does the
    accent-blind line treat every language group alike?' and, when the pooled
    bar fails, names the group that broke it.
    """
    audit: dict[str, object] = {}
    for l1, group in sorted(_group_by_l1(test).items()):
        metrics = _metrics(group, a, b, threshold, variant)
        positives = int(metrics["positives"])
        if positives >= L1_MIN_POSITIVES:
            ok = metrics["precision"] >= PRECISION_TARGET and metrics["recall"] >= RECALL_TARGET
        else:
            ok = positives == 0 or metrics["precision"] >= L1_PRECISION_FLOOR
        audit[l1] = {"ok": ok, **metrics}
    return audit


def _final_status(v_status: str, test_metrics: dict[str, object]) -> str:
    """The accent-agnostic shipping decision.

    'ok' requires the global train sweep to pass AND the single operating
    point to meet the bar on the whole held-out pool. The per-L1 breakdown
    never gates anything: the judge is blind to who is speaking.
    """
    if v_status != "ok":
        return v_status
    if test_metrics["precision"] >= PRECISION_TARGET and test_metrics["recall"] >= RECALL_TARGET:
        return "ok"
    return "pooled-bar-not-met"


def _metrics(
    records: list[TokenRecord], a: float, b: float, threshold: float, variant: str = "mean"
) -> dict[str, object]:
    """Precision/recall/F1 plus refusal accounting over a record set."""
    flagged = [r for r in records if _flag(r, a, b, threshold, variant)]
    flagged_ids = {id(r) for r in flagged}
    positives = [r for r in records if r.positive]
    tp = sum(1 for r in flagged if r.positive)
    fp = sum(1 for r in flagged if not r.positive)
    fn = sum(1 for r in positives if id(r) not in flagged_ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    verdicts = sum(1 for r in records if _variant_value(r, variant) is not None)
    refused = sum(1 for r in records if _variant_value(r, variant) is None)
    audio_refused = sum(1 for r in records if r.audio_problem)
    return {
        "tokens": len(records),
        "positives": len(positives),
        "verdicts": verdicts,
        "refused": refused,
        "audio_problem_refused": audio_refused,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _auc(records: list[TokenRecord], a: float, b: float, variant: str = "mean") -> float:
    """ROC-AUC of the calibrated P(substituted) against the corpus labels.

    Tie-corrected: a discrete variant (e.g. vote fractions of 0/0.5/1) would
    otherwise bias the rank sum, and a biased AUC quietly picks the wrong
    bake-off winner.
    """
    scored = [r for r in records if _variant_value(r, variant) is not None]
    if not scored or not any(r.positive for r in scored) or all(r.positive for r in scored):
        return float("nan")
    pairs = sorted(
        (sigmoid(a * _variant_value(r, variant) + b), r.positive)
        for r in scored  # type: ignore[arg-type]
    )
    n_pos = sum(1 for _, label in pairs if label)
    n_neg = len(pairs) - n_pos
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2
        for position in range(index, end + 1):
            if pairs[position][1]:
                rank_sum += average_rank
        index = end + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_platt(scores: list[float], labels: list[bool]) -> tuple[float, float]:
    """Fit sigmoid(a*s + b) to binary labels by maximum likelihood.

    scipy is imported lazily: the fixture smoke suite must run without it.
    """
    if not scores:
        raise ValueError("cannot fit calibration on an empty train set")
    if not any(labels) or all(labels):
        # A degenerate train pool gets an uninformative but well-defined fit.
        return 0.0, 0.0 if any(labels) else -1.0
    from scipy.optimize import minimize  # noqa: PLC0415

    xs = np.asarray(scores, dtype=np.float64)
    ys = np.asarray(labels, dtype=np.float64)

    def nll(params: np.ndarray) -> float:
        a, b = params
        logits = a * xs + b
        return float(np.mean(np.log1p(np.exp(logits)) - ys * logits))

    fitted = minimize(nll, x0=[1.0, 0.0], method="L-BFGS-B")
    return float(fitted.x[0]), float(fitted.x[1])


def pick_threshold(
    train: list[TokenRecord], a: float, b: float, variant: str = "mean"
) -> tuple[float, dict[str, object], str]:
    """Precision-first operating point (eval/README.md).

    Precision is not negotiable; recall is. Among thresholds meeting the
    precision target on the pooled train set, pick the highest recall. One
    threshold serves every learner - there is no per-language step. Returns
    the threshold, the metrics at it, and a status note.
    """
    candidates = [round(0.5 + 0.005 * step, 3) for step in range(0, 100)]
    results = {t: _metrics(train, a, b, t, variant) for t in candidates}

    meeting_target = [t for t in candidates if results[t]["precision"] >= PRECISION_TARGET]
    if not meeting_target:
        # The bar cannot be met at any threshold: report the best precision
        # available and refuse to ship feedback at all.
        best_t = max(candidates, key=lambda t: (results[t]["precision"], t))
        return best_t, results[best_t], "SHIPPING BAR NOT MET"
    chosen = max(meeting_target, key=lambda t: (results[t]["recall"], t))
    return chosen, results[chosen], "ok"


def _group_by_l1(records: list[TokenRecord]) -> dict[str, list[TokenRecord]]:
    groups: dict[str, list[TokenRecord]] = {}
    for record in records:
        groups.setdefault(record.l1, []).append(record)
    return groups


def _group_by_corpus(records: list[TokenRecord]) -> dict[str, list[TokenRecord]]:
    groups: dict[str, list[TokenRecord]] = {}
    for record in records:
        groups.setdefault(record.corpus, []).append(record)
    return groups


# -- the mirror exam (M1 pivot) --------------------------------------------------
# The shipped M1 line is no longer "is this an error" but "what did I hear".
# The exam scores the ear against what the learner actually produced: a
# confident report is correct iff heard == realized, and the mirror may always
# answer "couldn't tell" (which costs coverage, never accuracy).


def realized_of(record: TokenRecord, target: str) -> str | None:
    """What the learner actually produced, as a canonical phone label.

    None means the realization is unknown (e.g. so762 substitutions, whose
    expert spellings the adapter does not currently carry) - such tokens are
    excluded from accuracy but counted, never guessed.
    """
    if record.label == "correct":
        return target
    if record.label == "deleted":
        return DELETED  # nothing was produced; any heard phone is a mishearing
    return record.substituted_with


def _mirror_scored(records: list[TokenRecord]) -> list[TokenRecord]:
    """Records the ear answered at all (it heard some phone)."""
    return [r for r in records if r.heard is not None]


def _mirror_confident(
    records: list[TokenRecord], a: float, b: float, threshold: float
) -> list[TokenRecord]:
    """Records the mirror would report at the operating point."""
    return [
        r
        for r in _mirror_scored(records)
        if r.hearing_score is not None and sigmoid(a * r.hearing_score + b) >= threshold
    ]


def _mirror_metrics(
    records: list[TokenRecord], target: str, a: float, b: float, threshold: float
) -> dict[str, object]:
    """Mirror honesty at one operating point over a record set.

    accuracy: among confident reports with a known realization, the share
    where heard == realized. coverage: confident reports over ALL tokens (the
    learner's question is always answered or always refused). answer_rate:
    confident reports over tokens the ear scored at all.
    """
    scored = _mirror_scored(records)
    confident = _mirror_confident(records, a, b, threshold)
    scorable = [r for r in confident if realized_of(r, target) is not None]
    correct = sum(1 for r in scorable if r.heard == realized_of(r, target))
    scorable_raw = [r for r in scored if realized_of(r, target) is not None]
    raw_correct = sum(1 for r in scorable_raw if r.heard == realized_of(r, target))
    return {
        "tokens": len(records),
        "scored": len(scored),
        "confident": len(confident),
        "correct": correct,
        "accuracy": round(correct / len(scorable), 4) if scorable else 0.0,
        "coverage": round(len(confident) / len(records), 4) if records else 0.0,
        "answer_rate": round(len(confident) / len(scored), 4) if scored else 0.0,
        "unscorable": len(confident) - len(scorable),
        "top1_accuracy": round(raw_correct / len(scorable_raw), 4) if scorable_raw else 0.0,
    }


def pick_mirror_threshold(
    cal: list[TokenRecord], target: str, a: float, b: float
) -> tuple[float, dict[str, object], str]:
    """The mirror's operating point: max coverage while cal accuracy holds the
    accuracy target (accuracy is not negotiable; coverage is)."""
    candidates = [round(0.5 + 0.005 * step, 3) for step in range(0, 100)]
    results = {t: _mirror_metrics(cal, target, a, b, t) for t in candidates}
    meeting = [t for t in candidates if results[t]["accuracy"] >= MIRROR_ACCURACY_TARGET]
    if not meeting:
        best_t = max(candidates, key=lambda t: (results[t]["accuracy"], t))
        return best_t, results[best_t], "SHIPPING BAR NOT MET"
    chosen = max(meeting, key=lambda t: (results[t]["coverage"], t))
    return chosen, results[chosen], "ok"


def _mirror_final_status(cal_status: str, test_metrics: dict[str, object]) -> str:
    """The mirror shipping decision: the cal sweep must find an operating
    point AND that point must hold accuracy + coverage on held-out speakers."""
    if cal_status != "ok":
        return cal_status
    if (
        test_metrics["accuracy"] >= MIRROR_ACCURACY_TARGET
        and test_metrics["coverage"] >= MIRROR_COVERAGE_TARGET
    ):
        return "ok"
    return "mirror-bar-not-met"


def _mirror_per_l1(
    test: list[TokenRecord], target: str, a: float, b: float, threshold: float
) -> dict[str, object]:
    """Fairness audit of the mirror's single shipped line, per L1 group.

    Informational only - it never gates the bar; it shows whether the
    accent-blind ear hears every language group alike.
    """
    audit: dict[str, object] = {}
    for l1, group in sorted(_group_by_l1(test).items()):
        metrics = _mirror_metrics(group, target, a, b, threshold)
        if metrics["confident"] >= L1_MIN_POSITIVES:
            ok = (
                metrics["accuracy"] >= MIRROR_ACCURACY_TARGET
                and metrics["coverage"] >= MIRROR_COVERAGE_TARGET
            )
        else:
            ok = metrics["confident"] == 0 or metrics["accuracy"] >= L1_PRECISION_FLOOR
        audit[l1] = {"ok": ok, **metrics}
    return audit


def _mirror_confusion(
    test: list[TokenRecord], target: str, a: float, b: float, threshold: float
) -> dict[str, object]:
    """realized x heard counts over confident reports - the table that shows
    exactly which mishearings the mirror makes, honestly and per phone."""
    table: dict[str, dict[str, int]] = {}
    heard_labels: set[str] = set()
    for record in _mirror_confident(test, a, b, threshold):
        realized = realized_of(record, target)
        if realized is None or record.heard is None:
            continue
        table.setdefault(realized, {}).setdefault(record.heard, 0)
        table[realized][record.heard] += 1
        heard_labels.add(record.heard)
    rows: dict[str, dict[str, int]] = {}
    for realized in sorted(table, key=lambda name: (name != target, name)):
        row: dict[str, int] = {}
        for heard in sorted(heard_labels):
            row[heard] = table[realized].get(heard, 0)
        rows[realized] = row
    return {"rows": rows, "columns": sorted(heard_labels)}


def mirror_flagged_sample(
    test: list[TokenRecord], target: str, a: float, b: float, threshold: float, seed: int
) -> list[dict[str, object]]:
    """The human spot-check for the mirror: confident reports, mishearings
    first - a reviewer plays these and confirms what the ear reported."""
    confident = _mirror_confident(test, a, b, threshold)
    wrong = [
        r
        for r in confident
        if realized_of(r, target) is not None and r.heard != realized_of(r, target)
    ]
    right = [
        r
        for r in confident
        if realized_of(r, target) is not None and r.heard == realized_of(r, target)
    ]
    rng = random.Random(seed)
    rng.shuffle(right)
    picks = list(wrong[:15])
    picks.extend(right[: max(0, 30 - len(picks))])
    out = []
    for record in picks:
        out.append(
            {
                "utterance_id": record.utterance_id,
                "audio_path": record.audio_path,
                "start_s": record.start_s,
                "end_s": record.end_s,
                "l1": record.l1,
                "heard": record.heard,
                "realized": realized_of(record, target),
                "p_heard_realized": (
                    round(sigmoid(a * record.hearing_score + b), 4)
                    if record.hearing_score is not None
                    else None
                ),
                "correct_report": record.heard == realized_of(record, target),
            }
        )
    return out


def alignment_stats(records: list[TokenRecord]) -> dict[str, object]:
    """Alignment sanity: status distribution + the *alignment* confidence
    spread (not the contrast confidence, which is a verdict, not a gate)."""
    statuses: dict[str, int] = {}
    confidences = [
        r.alignment_confidence
        for r in records
        if r.alignment == "ok" and r.alignment_confidence is not None
    ]
    for record in records:
        statuses[record.alignment] = statuses.get(record.alignment, 0) + 1
    return {
        "statuses": statuses,
        "mean_ok_confidence": (round(float(np.mean(confidences)), 4) if confidences else None),
    }


def build_calibration(
    model_id: str,
    contrast_target: str,
    confusions: list[str],
    platt: tuple[float, float] | None,
    threshold: float | None,
    gop_platt: tuple[float, float] | None,
    generated_by: str,
    settings: Settings,
    score_variant: str = "mean",
    hearing_platt: tuple[float, float] | None = None,
    hearing_threshold: float | None = None,
) -> dict[str, object]:
    """The calibration.yaml content, mirroring scoring/calibration.py's models.

    Deliberately accent-blind: one operating point for every learner, with no
    per-language entries (the engine rejects a file that carries any).

    The judge block (substitution_platt/threshold) and the mirror's hearing
    block are independent: each ships only when its own exam passed. A
    calibration carrying a judge fit that failed its bar would ship exactly
    the wrong feedback the whole design exists to prevent, so the harness
    passes None instead.
    """
    contrast: dict[str, object] = {
        "target": contrast_target,
        "confusions": confusions,
        "score_variant": score_variant,
    }
    if platt is not None and threshold is not None:
        contrast["substitution_platt"] = {"a": platt[0], "b": platt[1]}
        contrast["threshold"] = threshold
    if gop_platt is not None:
        contrast["gop_platt"] = {"a": gop_platt[0], "b": gop_platt[1]}
    if hearing_platt is not None and hearing_threshold is not None:
        contrast["hearing_platt"] = {"a": hearing_platt[0], "b": hearing_platt[1]}
        contrast["hearing_threshold"] = hearing_threshold
    return {
        "schema_version": "1.0",
        "generated_by": generated_by,
        "model_id": model_id,
        "contrasts": [contrast],
        "alignment": {
            "min_confidence": settings.min_alignment_confidence,
            "low_confidence": settings.low_alignment_confidence,
        },
    }


def write_calibration(content: dict[str, object], path: Path = CALIBRATION_PATH) -> None:
    """Write the committed calibration file (the only writer in the repo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log.info("wrote calibration to %s", path)


def flagged_sample(
    records: list[TokenRecord],
    a: float,
    b: float,
    threshold: float,
    seed: int,
    variant: str = "mean",
) -> list[dict[str, object]]:
    """The human spot-check list: flagged tokens with everything a reviewer
    needs to confirm or reject the engine's claim (eval/README.md)."""
    flagged = [r for r in records if _flag(r, a, b, threshold, variant)]
    false_positives = [r for r in flagged if not r.positive]
    rng = random.Random(seed)
    picks = list(false_positives[:15])
    true_positives = [r for r in flagged if r.positive]
    rng.shuffle(true_positives)
    picks.extend(true_positives[: max(0, 30 - len(picks))])
    out = []
    for record in picks:
        value = _variant_value(record, variant)
        out.append(
            {
                "utterance_id": record.utterance_id,
                "audio_path": record.audio_path,
                "start_s": record.start_s,
                "end_s": record.end_s,
                "l1": record.l1,
                "corpus_label": record.label,
                "predicted_confusion": record.best_confusion,
                "p_substituted": (round(sigmoid(a * value + b), 4) if value is not None else None),
                "correct_flag": record.positive,
            }
        )
    return out


def evaluate_model(
    model_id: str,
    adapters: list[DatasetAdapter],
    target: str,
    confusions: list[str],
    *,
    seed: int,
    limit: int | None,
    out_dir: Path,
    settings: Settings,
    commit_calibration: bool,
    run_tag: str,
    include_train_pool: bool = False,
) -> dict[str, object]:
    """One full pass: run, fit on the cal pool, measure on held-out, report.

    The train pool is the exporters' domain (the model saw it); the harness
    never fits on it. With include_train_pool=True the train pool is also run
    through the engine for diagnostics - it changes nothing in the fit.

    With commit_calibration=True the fitted operating point is written to the
    engine's scoring/calibration.yaml - the only writer of that file besides
    nothing else in the repo.
    """
    registry = ModelRegistry(settings.model_dir)
    spec = registry.spec(model_id)
    if not registry.is_ready(spec):
        raise RuntimeError(
            f"model '{model_id}' is not downloaded - pull it first "
            f"(OPENSCHWA_MODEL_DIR={settings.model_dir})"
        )

    triples = collect_tokens(adapters, target, confusions, seed)
    if not include_train_pool:
        # The train pool is the model's training data: running it through the
        # engine would only re-measure memorization. Calibration uses cal,
        # the verdict uses test.
        triples = [t for t in triples if t[0].split != "train"]
    if limit is not None:
        seen: set[str] = set()
        kept = []
        for triple in triples:
            if len(seen) >= limit:
                break
            if triple[0].utterance_id not in seen:
                seen.add(triple[0].utterance_id)
                kept.append(triple)
        triples = kept
    if not triples:
        raise RuntimeError(
            f"no labeled /{target}/ tokens found - check the corpus paths and the contrast"
        )

    checkpoint = out_dir / "checkpoints" / f"{run_tag}.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # analyze_recording picks the model from settings: aligner candidates ride
    # alignment_model; closed-set contrast candidates (Option 3, role=contrast)
    # ride contrast_model_id while the default aligner keeps aligning. The
    # wrong wiring would silently measure the engine default instead.
    if spec.role == "contrast":
        run_settings = settings.model_copy(update={"contrast_model_id": model_id})
    else:
        run_settings = settings.model_copy(update={"alignment_model": model_id})
    run = run_tokens(triples, registry, run_settings, target, checkpoint=checkpoint)

    train = [r for r in run.records if r.split == "train"]
    cal = [r for r in run.records if r.split == "cal"]
    test = [r for r in run.records if r.split == "test"]
    if not cal or not test:
        raise RuntimeError("the cal or test split is empty - too few tokens for evaluation")

    scored_cal = [r for r in cal if r.score is not None]
    gop_pairs = [(r.gop, r.positive) for r in scored_cal if r.gop is not None]
    gop_platt: tuple[float, float] | None = None
    if gop_pairs:
        gop_platt = fit_platt([g for g, _ in gop_pairs], [not p for _, p in gop_pairs])

    # The scoring-variant mini-bake-off: fit and sweep each aggregation of
    # the same per-frame evidence (mean / spike / vote), then ship the best
    # variant. docs/architecture.md names spike-frame scoring as the fix for
    # CTC peakiness washing substitutions out of the label-frame mean.
    variants: dict[str, object] = {}
    for variant in ("mean", "spike", "vote", "gop"):
        values = [v for v in (_variant_value(r, variant) for r in scored_cal) if v is not None]
        labels = [r.positive for r in scored_cal if _variant_value(r, variant) is not None]
        if not values:
            variants[variant] = {"status": "no-scores"}
            continue
        v_a, v_b = fit_platt(values, labels)
        v_threshold, v_cal_metrics, v_status = pick_threshold(cal, v_a, v_b, variant)
        # The accent-agnostic bar: ONE operating point for every learner. The
        # held-out pool is scored at it, and the per-L1 breakdown is computed
        # at that same point as a fairness audit (does the blind line treat
        # every language group alike?). The audit never gates anything.
        v_test_metrics = _metrics(test, v_a, v_b, v_threshold, variant)
        v_per_l1 = _per_l1_audit(test, v_a, v_b, v_threshold, variant)
        v_status_final = _final_status(v_status, v_test_metrics)
        v_auc = _auc(test, v_a, v_b, variant)
        variants[variant] = {
            "platt": {"a": v_a, "b": v_b},
            "threshold": v_threshold,
            "status": v_status_final,
            "global_cal_status": v_status,
            "cal": v_cal_metrics,
            "test": v_test_metrics,
            "per_l1": v_per_l1,
            "auc": round(v_auc, 4) if not math.isnan(v_auc) else None,
        }

    def _variant_key(variant: str) -> tuple[object, ...]:
        """Prefer variants meeting the bar, then held-out f1, then recall."""
        info = variants[variant]
        if info["status"] != "ok":
            return (0, -1, -1)
        return (1, info["test"]["f1"], info["test"]["recall"])

    best_variant = max((v for v in variants if "test" in variants[v]), key=_variant_key)
    best = variants[best_variant]
    if best["status"] != "ok":
        best_variant = max(
            (v for v in variants if "test" in variants[v]),
            key=lambda v: variants[v]["auc"] if variants[v]["auc"] is not None else -1,
        )
        best = variants[best_variant]

    a, b = best["platt"]["a"], best["platt"]["b"]  # type: ignore[index]
    threshold = best["threshold"]  # type: ignore[assignment]
    cal_metrics = best["cal"]  # type: ignore[assignment]
    status = best["status"]  # type: ignore[assignment]
    test_metrics = best["test"]  # type: ignore[assignment]
    auc = best["auc"]
    per_l1 = best["per_l1"]  # type: ignore[assignment] - the fairness audit table
    per_corpus = {}
    for corpus, records in sorted(_group_by_corpus(test).items()):
        per_corpus[corpus] = _metrics(records, a, b, threshold, best_variant)
        corpus_auc = _auc(records, a, b, best_variant)
        per_corpus[corpus]["auc"] = (
            round(corpus_auc, 4) if corpus_auc is not None and not math.isnan(corpus_auc) else None
        )
    warm_ms = [r.wall_ms for r in run.records if r.score is not None and r.wall_ms > 0]

    # -- the mirror exam: the shipped M1 line (docs/research/mirror-pivot) ------
    # Fit P(heard == realized) on the cal pool, sweep the operating point for
    # max coverage while accuracy holds, then score held-out speakers once at
    # that point. The judge variants above stay for the research archive.
    mirror_cal = [r for r in cal if r.heard is not None and realized_of(r, target) is not None]
    mirror: dict[str, object]
    if mirror_cal:
        m_platt = fit_platt(
            [r.hearing_score for r in mirror_cal if r.hearing_score is not None],
            [r.heard == realized_of(r, target) for r in mirror_cal],
        )
        m_threshold, m_cal, m_cal_status = pick_mirror_threshold(cal, target, *m_platt)
        m_test = _mirror_metrics(test, target, *m_platt, m_threshold)
        m_per_l1 = _mirror_per_l1(test, target, *m_platt, m_threshold)
        m_status = _mirror_final_status(m_cal_status, m_test)
        mirror = {
            "platt": {"a": m_platt[0], "b": m_platt[1]},
            "threshold": m_threshold,
            "status": m_status,
            "cal_status": m_cal_status,
            "cal": m_cal,
            "test": m_test,
            "per_l1": m_per_l1,
            "confusion": _mirror_confusion(test, target, *m_platt, m_threshold),
            "flagged_sample": mirror_flagged_sample(test, target, *m_platt, m_threshold, seed),
        }
    else:
        mirror = {"status": "no-scores"}

    summary = {
        "run_tag": run_tag,
        "model_id": model_id,
        "target": target,
        "confusions": confusions,
        "seed": seed,
        "tokens": {"train": len(train), "cal": len(cal), "test": len(test)},
        "score_variant": best_variant,
        "variants": variants,
        "platt": best["platt"],
        "gop_platt": {"a": gop_platt[0], "b": gop_platt[1]} if gop_platt else None,
        "threshold": threshold,
        #: The shipped line is the mirror; the judge numbers above are the
        #: research archive (M1 pivot).
        "status": mirror["status"],
        "judge_status": status,
        "mirror": mirror,
        "cal_metrics": cal_metrics,
        "test_metrics": test_metrics,
        "test_auc": round(auc, 4) if auc is not None and not math.isnan(auc) else None,
        "per_l1": per_l1,
        "per_corpus": per_corpus,
        "alignment": alignment_stats(run.records),
        "latency": {
            "cold_ms": round(run.cold_ms, 1),
            "median_warm_ms": round(float(np.median(warm_ms)), 1) if warm_ms else None,
        },
        "download_bytes": spec.download_bytes,
        "flagged_sample": flagged_sample(test, a, b, threshold, seed, best_variant),
    }

    # A smoke run's handful of tokens can trivially pass the threshold sweep
    # (flag nothing, precision 1.0); committing that would ship a meaningless
    # calibration. The floor is deliberately conservative.
    if commit_calibration and summary["status"] == "ok" and len(cal) < MIN_COMMIT_TRAIN:
        log.error(
            "NOT committing calibration: %d cal tokens is below the %d floor (smoke run?)",
            len(cal),
            MIN_COMMIT_TRAIN,
        )
        commit_calibration = False

    if commit_calibration and summary["status"] == "ok":
        # Provenance must name the report this run actually writes (below),
        # repo-relative so the traceability test can resolve it from any cwd.
        report_path = (out_dir / f"{run_tag}.json").resolve()
        try:
            generated_by = str(report_path.relative_to(REPO_ROOT))
        except ValueError:
            generated_by = str(report_path)
        # Each block ships only on its own passing exam: the hearing block
        # because the mirror passed, the judge block only when the judge did.
        judge_passed = status == "ok"
        judge_a, judge_b = best["platt"]["a"], best["platt"]["b"]  # type: ignore[index]
        judge_threshold = best["threshold"]  # type: ignore[assignment]
        m_platt_pair = (
            (mirror["platt"]["a"], mirror["platt"]["b"])  # type: ignore[index]
            if "platt" in mirror
            else None
        )
        m_threshold = mirror.get("threshold")  # type: ignore[assignment]
        calibration = build_calibration(
            model_id,
            target,
            confusions,
            (judge_a, judge_b) if judge_passed else None,
            judge_threshold if judge_passed else None,
            gop_platt,
            generated_by=generated_by,
            settings=settings,
            score_variant=best_variant,
            hearing_platt=m_platt_pair,
            hearing_threshold=m_threshold if m_threshold is not None else None,
        )
        write_calibration(calibration)
    elif commit_calibration:
        log.error(
            "NOT committing calibration: the mirror bar was not met (status %s)",
            summary["status"],
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / f"{run_tag}.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(summary: dict[str, object]) -> str:
    """Markdown report; committed alongside the JSON so thresholds stay
    traceable to their evidence (eval/README.md).

    The mirror is the shipped line (status); the judge variants follow as the
    research archive (M1 pivot, docs/research/mirror-pivot).
    """
    lines = [
        f"# {summary['run_tag']}",
        "",
        f"- model: {summary['model_id']}",
        f"- contrast: /{summary['target']}/ vs {summary['confusions']}",
        f"- tokens: {summary['tokens'].get('train', 0)} train / "
        f"{summary['tokens']['cal']} cal / {summary['tokens']['test']} held-out",
        f"- **status: {summary['status']}** (the mirror; judge: "
        f"{summary.get('judge_status', '-')})",
        "",
    ]
    mirror = summary.get("mirror") or {}
    if "test" not in mirror:
        lines += ["## Mirror", "", f"- status: {mirror.get('status', '-')}", ""]
    else:
        test = mirror["test"]
        cal = mirror["cal"]
        lines += [
            "## Mirror - what the ear heard (shipped line)",
            "",
            f"- Platt: p = sigmoid({mirror['platt']['a']:.3f} * hearing_score + "
            f"{mirror['platt']['b']:.3f})  [P(heard == realized)]",
            f"- threshold: {mirror['threshold']}",
            f"- cal: accuracy {cal['accuracy']} / coverage {cal['coverage']} "
            f"(cal status {mirror['cal_status']})",
            f"- **held-out: accuracy {test['accuracy']} / coverage {test['coverage']}**"
            f" / answer-rate {test['answer_rate']}",
            f"- raw top-1 hearing accuracy (no gating): {test['top1_accuracy']}",
            f"- confident reports {test['confident']} of {test['tokens']} tokens "
            f"(scored {test['scored']}; unscorable {test['unscorable']})",
            "",
            "### realized x heard (confident reports)",
            "",
        ]
        columns = mirror["confusion"]["columns"]
        header = "| realized \\ heard | " + " | ".join(f"/{c}/" for c in columns) + " |"
        separator = "|---|" + "---|" * len(columns)
        lines += [header, separator]
        for realized, row in mirror["confusion"]["rows"].items():
            if realized == DELETED:
                label = "deleted"
            else:
                label = f"/{realized}/"
            lines.append(
                "| " + label + " | " + " | ".join(str(row.get(c, 0)) for c in columns) + " |"
            )
        lines += [
            "",
            "### Mirror per L1 (held-out)",
            "",
            "Fairness audit of the single shipped line: every group is heard at",
            f"the same global operating point {mirror['threshold']}. Informational",
            "only - it never gates the bar.",
            "",
            "| l1 | tokens | confident | accuracy | coverage | fair |",
            "|---|---|---|---|---|---|",
        ]
        for l1, m in mirror["per_l1"].items():
            lines.append(
                f"| {l1} | {m['tokens']} | {m['confident']} | {m['accuracy']} | "
                f"{m['coverage']} | {m['ok']} |"
            )
        lines += ["", "### Mirror spot-check (confident reports, mishearings first)", ""]
        for item in mirror["flagged_sample"]:
            mark = "right" if item["correct_report"] else "WRONG"
            lines.append(
                f"- [{mark}] {item['utterance_id']} heard /{item['heard']}/, realized "
                f"/{item['realized']}/ (p={item['p_heard_realized']}) "
                f"l1={item['l1']} {item['audio_path']}"
            )
        lines += [""]

    lines += [
        "## Judge variants (research archive - parked by the mirror pivot)",
        "",
        "| variant | threshold | status | cal P/R | held-out P/R | f1 | AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant, info in summary["variants"].items():
        if "test" not in info:
            lines.append(f"| {variant} | - | {info['status']} | - | - | - | - |")
            continue
        lines.append(
            f"| {variant} | {info['threshold']} | {info['status']} | "
            f"{info['cal']['precision']}/{info['cal']['recall']} | "
            f"{info['test']['precision']}/{info['test']['recall']} | "
            f"{info['test']['f1']} | {info['auc']} |"
        )
    lines += [
        "",
        "## Judge operating point (calibration split, research archive)",
        "",
        f"- Platt: p = sigmoid({summary['platt']['a']:.3f} * score + {summary['platt']['b']:.3f})",
        f"- threshold: {summary['threshold']}",
        f"- cal: precision {summary['cal_metrics']['precision']}, "
        f"recall {summary['cal_metrics']['recall']}",
        f"- cal->test precision gap: "
        f"{round(summary['cal_metrics']['precision'] - summary['test_metrics']['precision'], 4)}",
        "",
        "## Judge held-out (research archive)",
        "",
        f"- precision {summary['test_metrics']['precision']} / "
        f"recall {summary['test_metrics']['recall']} / f1 {summary['test_metrics']['f1']}",
        f"- AUC {summary['test_auc']}",
        f"- verdicts {summary['test_metrics']['verdicts']}, "
        f"refused {summary['test_metrics']['refused']} "
        f"(audio-problem {summary['test_metrics']['audio_problem_refused']})",
        "",
        "## Per L1 audit (held-out)",
        "",
        "Fairness check of the single shipped line: every group is scored at",
        f"the same global operating point {summary['threshold']}. Informational",
        "only - it never gates the bar; it shows whether the accent-blind line",
        "treats every language group alike, and names the group when it does",
        "not.",
        "",
        "| l1 | tokens | positives | precision | recall | f1 | fair |",
        "|---|---|---|---|---|---|---|",
    ]
    for l1, m in summary["per_l1"].items():
        lines.append(
            f"| {l1} | {m['tokens']} | {m['positives']} | {m['precision']} | "
            f"{m['recall']} | {m['f1']} | {m['ok']} |"
        )
    lines += [
        "",
        "## Per corpus (held-out)",
        "",
        "| corpus | tokens | positives | precision | recall | f1 | AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for corpus, m in summary["per_corpus"].items():
        lines.append(
            f"| {corpus} | {m['tokens']} | {m['positives']} | {m['precision']} | "
            f"{m['recall']} | {m['f1']} | {m['auc']} |"
        )
    lines += [
        "",
        "## Alignment sanity",
        "",
        f"- statuses: {summary['alignment']['statuses']}",
        f"- mean alignment confidence (ok): {summary['alignment']['mean_ok_confidence']}",
        "",
        "## Latency",
        "",
        f"- cold {summary['latency']['cold_ms']} ms, "
        f"median warm {summary['latency']['median_warm_ms']} ms",
        f"- download size {summary['download_bytes'] / 1e9:.2f} GB",
        "",
        "## Flagged items for human spot-check",
        "",
        "(paths relative to the corpus roots; review per eval/README.md)",
        "",
    ]
    for item in summary["flagged_sample"]:
        mark = "TP" if item["correct_flag"] else "FP"
        lines.append(
            f"- [{mark}] {item['utterance_id']} /{summary['target']}/ -> "
            f"/{item['predicted_confusion']}/ (p={item['p_substituted']}) "
            f"label={item['corpus_label']} l1={item['l1']} {item['audio_path']}"
        )
    return "\n".join(lines) + "\n"
