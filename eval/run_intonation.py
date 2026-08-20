"""CLI for the M2 intonation exam over Aix-MARSEC (see intonation.py)."""

import argparse
import json
from datetime import date
from pathlib import Path

from openschwa_engine.config import Settings
from openschwa_engine.models.registry import ModelRegistry

from openschwa_eval.datasets.aixmarsec import AixMarsec
from openschwa_eval.intonation import evaluate_intonation, run_units, split_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aix", required=True, help="Aix-MARSEC root (the '4' folder)")
    parser.add_argument("--out", default="reports-intonation/")
    parser.add_argument("--limit", type=int, default=None, help="max passages to load (smoke)")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    corpus = AixMarsec(Path(args.aix), max_passages=args.limit)
    units = corpus.units()
    print(f"units: {len(units)}", flush=True)
    split = split_blocks(units, args.split_seed)
    test_units = sum(1 for u in units if split[u.block_id] == "test")
    print(f"test units: {test_units} of {len(units)}", flush=True)

    settings = Settings(warm_model_on_start=False)
    registry = ModelRegistry(settings.model_dir)
    stamp = date.today().isoformat()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"aix-intonation-{stamp}.jsonl"
    records = run_units(units, registry, settings, checkpoint=checkpoint)

    summary = evaluate_intonation(records, split)
    print(json.dumps(summary, indent=2))
    (out_dir / f"aix-intonation-{stamp}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
