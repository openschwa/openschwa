"""CLI for the M2 intonation exam over controlled recordings (see recordings.py)."""

import argparse
import json
from datetime import date
from pathlib import Path

from openschwa_eval.recordings import evaluate_recordings, records_jsonl, run_recordings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="the item manifest JSON")
    parser.add_argument("--audio", required=True, help="directory with <id>_<rep>.<ext> files")
    parser.add_argument("--exclude", default=None, help="verification JSON: takes to drop")
    parser.add_argument("--out", default="reports-recordings/")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    audio_dir = Path(args.audio)
    if not audio_dir.is_dir():
        parser.error(f"{audio_dir}: not a directory")
    exclude = (
        set(json.loads(Path(args.exclude).read_text(encoding="utf-8")))
        if args.exclude
        else None
    )
    records = run_recordings(manifest, audio_dir, exclude=exclude)
    summary = evaluate_recordings(records)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    (out_dir / f"recordings-intonation-{stamp}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / f"recordings-intonation-{stamp}.jsonl").write_text(
        records_jsonl(records), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
