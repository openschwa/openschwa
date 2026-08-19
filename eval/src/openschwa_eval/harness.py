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

log = logging.getLogger(__name__)

EPS = 1e-12
L1_PRECISION_FLOOR = 0.8
L1_MIN_POSITIVES = 5
PRECISION_TARGET = 0.90
RECALL_TARGET = 0.4  # the milestone's accept: precision >= 0.90 at recall >= 0.4
#: Below this many train tokens the threshold sweep is not evidence of
#: anything, and a calibration must never be committed from it.
MIN_COMMIT_TRAIN = 200


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
    """Corpus-native split when the adapter has one, else a stable seeded one."""
    if utterance.split:
        return utterance.split
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
        score = best = spike = vote = None
        verdict = confidence = gop = None
        if contrast is not None and contrast.posteriors:
            score, best = raw_score_from_posteriors(contrast.posteriors, target)
            spike = contrast.spike_score
            vote = contrast.vote_fraction
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


def _flag_per_l1(
    record: TokenRecord,
    a: float,
    b: float,
    variant: str,
    l1_thresholds: dict[str, float],
    default_threshold: float,
) -> bool:
    """Flags using the token's own L1 operating point, falling back to the
    global one. The v3 exam proved one global cut cannot serve two
    acoustically different populations at once.
    """
    threshold = l1_thresholds.get(record.l1, default_threshold)
    return _flag(record, a, b, threshold, variant)


def _metrics_per_l1(
    records: list[TokenRecord],
    a: float,
    b: float,
    variant: str,
    l1_thresholds: dict[str, float],
    default_threshold: float,
) -> dict[str, object]:
    """Precision/recall/F1 with per-token L1 operating points."""
    flagged = [
        r for r in records if _flag_per_l1(r, a, b, variant, l1_thresholds, default_threshold)
    ]
    flagged_ids = {id(r) for r in flagged}
    positives = [r for r in records if r.positive]
    tp = sum(1 for r in flagged if r.positive)
    fp = sum(1 for r in flagged if not r.positive)
    fn = sum(1 for r in positives if id(r) not in flagged_ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tokens": len(records),
        "positives": len(positives),
        "verdicts": sum(1 for r in records if _variant_value(r, variant) is not None),
        "refused": sum(1 for r in records if _variant_value(r, variant) is None),
        "audio_problem_refused": sum(1 for r in records if r.audio_problem),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def fit_l1_thresholds(
    train: list[TokenRecord], a: float, b: float, variant: str
) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    """Per-L1 operating points: each group gets its own precision-first sweep.

    Groups with too few positives reuse the global threshold: a per-group
    cut fitted on five tokens is noise, and a wrong per-group cut is worse
    than a shared one.
    """
    thresholds: dict[str, float] = {}
    details: dict[str, dict[str, object]] = {}
    for l1, group in sorted(_group_by_l1(train).items()):
        positives = sum(1 for r in group if r.positive)
        if positives < L1_MIN_POSITIVES:
            details[l1] = {"fitted": False, "positives": positives}
            continue
        threshold, metrics, status = pick_threshold(group, a, b, variant)
        thresholds[l1] = threshold
        details[l1] = {
            "fitted": True,
            "threshold": threshold,
            "status": status,
            "train": metrics,
        }
    return thresholds, details


def _bar_status(
    v_status: str,
    test: list[TokenRecord],
    a: float,
    b: float,
    variant: str,
    l1_thresholds: dict[str, float],
    default_threshold: float,
) -> tuple[str, dict[str, object]]:
    """The per-L1 shipping decision.

    'ok' requires every L1 group with enough held-out positives to meet the
    accept criterion at its own operating point (P >= 0.90, R >= 0.4),
    every smaller group to stay above the precision floor, AND the global
    train sweep to have passed (for learners whose L1 we do not know).
    """
    per_group: dict[str, object] = {}
    group_ok = True
    for l1, group in sorted(_group_by_l1(test).items()):
        metrics = _metrics_per_l1(group, a, b, variant, l1_thresholds, default_threshold)
        positives = int(metrics["positives"])
        if positives >= L1_MIN_POSITIVES:
            ok = metrics["precision"] >= PRECISION_TARGET and metrics["recall"] >= RECALL_TARGET
        else:
            ok = positives == 0 or metrics["precision"] >= L1_PRECISION_FLOOR
        per_group[l1] = {"ok": ok, **metrics}
        group_ok = group_ok and ok
    if v_status != "ok":
        return v_status, per_group
    return ("ok" if group_ok else "ok-no-l1-floor"), per_group

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
    """Precision-first operating point, with the per-L1 floor.

    Policy (eval/README.md): precision is not negotiable; recall is. Among
    thresholds meeting the precision target we prefer those keeping every L1
    group above its floor, then the highest recall. Returns the threshold, the
    metrics at it, and a status note.
    """
    candidates = [round(0.5 + 0.005 * step, 3) for step in range(0, 100)]
    results = {t: _metrics(train, a, b, t, variant) for t in candidates}

    meeting_target = [t for t in candidates if results[t]["precision"] >= PRECISION_TARGET]
    if not meeting_target:
        # The bar cannot be met at any threshold: report the best precision
        # available and refuse to ship feedback at all.
        best_t = max(candidates, key=lambda t: (results[t]["precision"], t))
        return best_t, results[best_t], "SHIPPING BAR NOT MET"

    def l1_ok(t: float) -> bool:
        for _l1, records in _group_by_l1(train).items():
            m = _metrics(records, a, b, t, variant)
            positives = int(m["positives"])
            if positives >= L1_MIN_POSITIVES and m["precision"] < L1_PRECISION_FLOOR:
                return False
        return True

    with_floor = [t for t in meeting_target if l1_ok(t)]
    pool = with_floor if with_floor else meeting_target
    status = "ok" if with_floor else "ok-no-l1-floor"
    chosen = max(pool, key=lambda t: (results[t]["recall"], t))
    return chosen, results[chosen], status


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
    platt: tuple[float, float],
    threshold: float,
    gop_platt: tuple[float, float] | None,
    generated_by: str,
    settings: Settings,
    score_variant: str = "mean",
    l1_thresholds: "dict[str, float] | None" = None,
) -> dict[str, object]:
    """The calibration.yaml content, mirroring scoring/calibration.py's models."""
    contrast: dict[str, object] = {
        "target": contrast_target,
        "confusions": confusions,
        "score_variant": score_variant,
        "substitution_platt": {"a": platt[0], "b": platt[1]},
        "threshold": threshold,
    }
    if l1_thresholds:
        contrast["l1_thresholds"] = l1_thresholds
    if gop_platt is not None:
        contrast["gop_platt"] = {"a": gop_platt[0], "b": gop_platt[1]}
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
) -> dict[str, object]:
    """One full pass: run, fit on train, measure on held-out, report.

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
    test = [r for r in run.records if r.split == "test"]
    if not train or not test:
        raise RuntimeError("a split is empty - too few tokens for evaluation")

    scored_train = [r for r in train if r.score is not None]
    gop_pairs = [(r.gop, r.positive) for r in scored_train if r.gop is not None]
    gop_platt: tuple[float, float] | None = None
    if gop_pairs:
        gop_platt = fit_platt([g for g, _ in gop_pairs], [not p for _, p in gop_pairs])

    # The scoring-variant mini-bake-off: fit and sweep each aggregation of
    # the same per-frame evidence (mean / spike / vote), then ship the best
    # variant. docs/architecture.md names spike-frame scoring as the fix for
    # CTC peakiness washing substitutions out of the label-frame mean.
    variants: dict[str, object] = {}
    for variant in ("mean", "spike", "vote", "gop"):
        values = [v for v in (_variant_value(r, variant) for r in scored_train) if v is not None]
        labels = [r.positive for r in scored_train if _variant_value(r, variant) is not None]
        if not values:
            variants[variant] = {"status": "no-scores"}
            continue
        v_a, v_b = fit_platt(values, labels)
        v_threshold, v_train_metrics, v_status = pick_threshold(train, v_a, v_b, variant)
        # Per-L1 operating points: each first-language group sweeps its own
        # precision-first threshold (fit_l1_thresholds), the held-out pool is
        # scored with each token's own group threshold (_metrics_per_l1), and
        # the shipping status requires EVERY group to meet the bar at its own
        # point (_bar_status). This is the per-L1 design: one judge, one
        # measurement, a fair flag-line per language.
        v_l1_thresholds, v_l1_details = fit_l1_thresholds(train, v_a, v_b, variant)
        v_test_metrics = _metrics_per_l1(test, v_a, v_b, variant, v_l1_thresholds, v_threshold)
        v_status_final, v_per_group = _bar_status(
            v_status, test, v_a, v_b, variant, v_l1_thresholds, v_threshold
        )
        v_auc = _auc(test, v_a, v_b, variant)
        variants[variant] = {
            "platt": {"a": v_a, "b": v_b},
            "threshold": v_threshold,
            "l1_thresholds": v_l1_thresholds,
            "l1_fit": v_l1_details,
            "status": v_status_final,
            "global_train_status": v_status,
            "train": v_train_metrics,
            "test": v_test_metrics,
            "per_l1": v_per_group,
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
    l1_thresholds = best["l1_thresholds"]  # type: ignore[assignment]
    train_metrics = best["train"]  # type: ignore[assignment]
    status = best["status"]  # type: ignore[assignment]
    test_metrics = best["test"]  # type: ignore[assignment]
    auc = best["auc"]
    per_l1 = best["per_l1"]  # type: ignore[assignment] - the per-group bar table
    per_corpus = {
        corpus: _metrics_per_l1(records, a, b, best_variant, l1_thresholds, threshold)
        for corpus, records in sorted(_group_by_corpus(test).items())
    }
    warm_ms = [r.wall_ms for r in run.records if r.score is not None and r.wall_ms > 0]

    summary = {
        "run_tag": run_tag,
        "model_id": model_id,
        "target": target,
        "confusions": confusions,
        "seed": seed,
        "tokens": {"train": len(train), "test": len(test)},
        "score_variant": best_variant,
        "variants": variants,
        "platt": best["platt"],
        "gop_platt": {"a": gop_platt[0], "b": gop_platt[1]} if gop_platt else None,
        "threshold": threshold,
        "l1_thresholds": l1_thresholds,
        "status": status,
        "train_metrics": train_metrics,
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
    if commit_calibration and status == "ok" and len(train) < MIN_COMMIT_TRAIN:
        log.error(
            "NOT committing calibration: %d train tokens is below the %d floor (smoke run?)",
            len(train),
            MIN_COMMIT_TRAIN,
        )
        commit_calibration = False

    if commit_calibration and status == "ok":
        calibration = build_calibration(
            model_id,
            target,
            confusions,
            (a, b),
            threshold,
            gop_platt,
            generated_by=f"eval/reports/{run_tag}.json",
            settings=settings,
            score_variant=best_variant,
            l1_thresholds=l1_thresholds,
        )
        write_calibration(calibration)
    elif commit_calibration:
        log.error("NOT committing calibration: the bar was not met (status %s)", status)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / f"{run_tag}.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(summary: dict[str, object]) -> str:
    """Markdown report; committed alongside the JSON so thresholds stay
    traceable to their evidence (eval/README.md)."""
    lines = [
        f"# {summary['run_tag']}",
        "",
        f"- model: {summary['model_id']}",
        f"- contrast: /{summary['target']}/ vs {summary['confusions']}",
        f"- tokens: {summary['tokens']['train']} train / {summary['tokens']['test']} held-out",
        f"- **status: {summary['status']}**",
        f"- **shipping variant: {summary['score_variant']}**",
        "",
        "## Score variants (same frames, three aggregations)",
        "",
        "| variant | threshold | status | train P/R | held-out P/R | f1 | AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant, info in summary["variants"].items():
        if "test" not in info:
            lines.append(f"| {variant} | - | {info['status']} | - | - | - | - |")
            continue
        lines.append(
            f"| {variant} | {info['threshold']} | {info['status']} | "
            f"{info['train']['precision']}/{info['train']['recall']} | "
            f"{info['test']['precision']}/{info['test']['recall']} | "
            f"{info['test']['f1']} | {info['auc']} |"
        )
    lines += [
        "",
        "## Operating point (train split)",
        "",
        f"- Platt: p = sigmoid({summary['platt']['a']:.3f} * score + {summary['platt']['b']:.3f})",
        f"- threshold: {summary['threshold']}",
        f"- train: precision {summary['train_metrics']['precision']}, "
        f"recall {summary['train_metrics']['recall']}",
        "",
        "## Held-out",
        "",
        f"- precision {summary['test_metrics']['precision']} / "
        f"recall {summary['test_metrics']['recall']} / f1 {summary['test_metrics']['f1']}",
        f"- AUC {summary['test_auc']}",
        f"- verdicts {summary['test_metrics']['verdicts']}, "
        f"refused {summary['test_metrics']['refused']} "
        f"(audio-problem {summary['test_metrics']['audio_problem_refused']})",
        "",
        "## Per L1 (held-out)",
        "",
        "Every group with enough train positives gets its own operating point",
        "(fitted); everyone else - including learners whose L1 we do not know -",
        f"uses the global threshold {summary['threshold']}.",
        "",
        "| l1 | tokens | positives | threshold | precision | recall | f1 | ok |",
        "|---|---|---|---|---|---|---|---|",
    ]
    l1_thresholds = summary["l1_thresholds"]
    for l1, m in summary["per_l1"].items():
        threshold = l1_thresholds.get(l1, summary["threshold"])
        source = "fitted" if l1 in l1_thresholds else "global"
        lines.append(
            f"| {l1} | {m['tokens']} | {m['positives']} | {threshold} ({source}) | "
            f"{m['precision']} | {m['recall']} | {m['f1']} | {m['ok']} |"
        )
    lines += [
        "",
        "## Per corpus (held-out)",
        "",
        "| corpus | tokens | positives | precision | recall | f1 |",
        "|---|---|---|---|---|---|",
    ]
    for corpus, m in summary["per_corpus"].items():
        lines.append(
            f"| {corpus} | {m['tokens']} | {m['positives']} | {m['precision']} | "
            f"{m['recall']} | {m['f1']} |"
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
