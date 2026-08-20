"""Run the offline evaluation for one contrast. M1 (mirror line).

Usage:
    uv run python run_eval.py --contrast "ð:z,d,v" \
        --l2arctic /path/to/l2arctic --speechocean762 /path/to/speechocean762 \
        --out reports/

Procedure: see README.md. Output: the mirror exam (hearing accuracy /
coverage, realized-x-heard confusion table, per-L1 breakdown) plus the judge
variants as the research archive, written as a committed markdown + JSON
report; the winning hearing calibration lands in the engine's
scoring/calibration.yaml when (and only when) the mirror bar is met.
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

from openschwa_engine.config import Settings
from openschwa_engine.models.registry import MANIFEST, ModelRegistry

from openschwa_eval.datasets import L2Arctic, SpeechOcean762
from openschwa_eval.harness import evaluate_model

log = logging.getLogger("openschwa-eval")


def bakeoff_report(summaries: "list[dict[str, object]]", winner: "dict | None") -> str:
    """The combined comparison table across candidates. The shipped line is
    the mirror (accuracy/coverage); judge AUC is reported for the research
    archive."""
    lines = [
        f"# M1 mirror bake-off - /{summaries[0]['target']}/ vs {summaries[0]['confusions']}",
        "",
        "| model | mirror accuracy | coverage | raw top-1 | judge AUC | status | median warm ms | size GB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        mirror = s.get("mirror") or {}
        m_test = mirror.get("test") or {}
        lines.append(
            f"| {s['model_id']} | {m_test.get('accuracy', '-')} | "
            f"{m_test.get('coverage', '-')} | {m_test.get('top1_accuracy', '-')} | "
            f"{s['test_auc']} | {s['status']} | {s['latency']['median_warm_ms']} | "
            f"{s['download_bytes'] / 1e9:.2f} |"
        )
    lines += [
        "",
        "## Alignment sanity (held-out token runs)",
        "",
        "| model | statuses | mean ok confidence |",
        "|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['model_id']} | {s['alignment']['statuses']} | "
            f"{s['alignment']['mean_ok_confidence']} |"
        )
    lines += [
        "",
        f"**Winner: {winner['model_id'] if winner else 'none (mirror bar not met)'}**",
        "",
    ]
    return "\n".join(lines)


def parse_contrast(value: str) -> tuple[str, list[str]]:
    target, separator, confusions = value.partition(":")
    if not separator or not confusions:
        raise argparse.ArgumentTypeError(
            f"--contrast '{value}' must look like 'ð:z,d,v' (target:confusions)"
        )
    return target.strip(), [c.strip() for c in confusions.split(",") if c.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contrast",
        default="ð:z,d,v",
        type=parse_contrast,
        help="target:confusion[,confusion...] in canonical IPA",
    )
    parser.add_argument("--l2arctic", help="local L2-ARCTIC corpus root")
    parser.add_argument("--speechocean762", help="local speechocean762 corpus root")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="bake-off candidate model id; repeatable; default: the engine's alignment_model",
    )
    parser.add_argument("--out", default="reports/")
    parser.add_argument(
        "--include-train-pool",
        action="store_true",
        help=(
            "also run the train pool through the engine (diagnostics only - "
            "the harness never fits on it)"
        ),
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help=(
            "write calibration.yaml when the bar is met (the only sanctioned "
            "writer of that file besides the harness itself)"
        ),
    )
    parser.add_argument("--limit", type=int, help="max labeled utterances processed (smoke runs)")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    target, confusions = args.contrast
    adapters = []
    if args.l2arctic:
        adapters.append(L2Arctic(Path(args.l2arctic)))
    if args.speechocean762:
        adapters.append(SpeechOcean762(Path(args.speechocean762)))
    if not adapters:
        parser.error("at least one of --l2arctic / --speechocean762 is required")

    settings = Settings(warm_model_on_start=False)
    registry = ModelRegistry(settings.model_dir)
    model_ids = args.model or [settings.alignment_model]
    for model_id in model_ids:
        if model_id not in MANIFEST:
            parser.error(f"unknown model '{model_id}' (manifest: {sorted(MANIFEST)})")
        if not registry.is_ready(registry.spec(model_id)):
            log.error(
                "model '%s' is not downloaded (OPENSCHWA_MODEL_DIR=%s) - run 'just models'",
                model_id,
                settings.model_dir,
            )
            sys.exit(1)

    stamp = datetime.date.today().isoformat()
    slug = f"{target}-vs-{'-'.join(confusions)}"
    out_dir = Path(args.out)

    summaries = []
    for model_id in model_ids:
        run_tag = f"m1-{stamp}-{model_id}-{slug}"
        log.info("== evaluating %s (%s)", model_id, run_tag)
        summary = evaluate_model(
            model_id,
            adapters,
            target,
            confusions,
            seed=args.split_seed,
            limit=args.limit,
            out_dir=out_dir,
            settings=settings,
            # Candidates never commit: in a bake-off every passing candidate
            # would otherwise write calibration.yaml, last writer wins. Only
            # the winner re-run below may commit, and only with --commit.
            commit_calibration=False,
            run_tag=run_tag,
            include_train_pool=args.include_train_pool,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        winners = [s for s in summaries if s["status"] == "ok"]
        if winners:
            winner = max(
                winners,
                key=lambda s: (s["test_metrics"]["f1"], s["test_metrics"]["precision"]),
            )
            log.info(
                "bake-off winner: %s (held-out f1 %s)",
                winner["model_id"],
                winner["test_metrics"]["f1"],
            )
        else:
            winner = None
            log.error("no candidate met the mirror bar; no calibration committed")
        (out_dir / f"m1-bakeoff-{stamp}-{slug}.md").write_text(
            bakeoff_report(summaries, winner), encoding="utf-8"
        )
    else:
        winner = summaries[0] if summaries[0]["status"] == "ok" else None
        if winner is None:
            log.error(
                "the bar was not met (status %s); no calibration committed",
                summaries[0]["status"],
            )

    if winner is not None and args.commit and args.limit is None:
        # The ONLY sanctioned calibration write: one winner, full run,
        # explicit --commit. Cheap - it resumes from the run's checkpoint.
        # The harness re-checks the bar status and the train-token floor.
        evaluate_model(
            winner["model_id"],
            adapters,
            target,
            confusions,
            seed=args.split_seed,
            limit=args.limit,
            out_dir=out_dir,
            settings=settings,
            commit_calibration=True,
            run_tag=winner["run_tag"],
            include_train_pool=args.include_train_pool,
        )
        log.info("committed calibration for %s", winner["model_id"])
    elif winner is not None:
        log.info(
            "bar met by %s but calibration NOT committed (%s)",
            winner["model_id"],
            "smoke run (--limit) never commits"
            if args.limit is not None
            else "pass --commit to write calibration.yaml",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
