"""Human verification pass for the recordings exam.

Plays every flagged take (engine verdict != expected tone) and asks the
human listener to judge the *take*, not the engine: "keep" = the take is
fine, the engine is wrong (counts against the bar); "drop" = the take was
mis-spoken (excluded from scoring). The drops are written as a JSON list
of "<id>_<rep>" keys for run_recordings_exam.py --exclude.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def play(path: Path) -> None:
    subprocess.run(["afplay", str(path)], check=False, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="the exam summary JSON with flags")
    parser.add_argument("--audio", required=True, help="the recordings directory")
    parser.add_argument("--out", default="exclusions.json")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    flags = report["flags_for_verification"]
    audio_dir = Path(args.audio)
    exclusions: list[str] = []
    for flag in flags:
        stem = f"{flag['item_id']}_{flag['rep']}"
        audio = next(audio_dir.glob(f"{stem}.*"), None)
        if audio is None:
            print(f"{stem}: recording missing - re-record, then re-run the exam.")
            continue
        print()
        print(
            f"[{len(exclusions)} dropped so far] {flag['text']!r} "
            f"(expected {flag['expected']}, engine said {flag['detected']})"
        )
        print("  cue: listen and judge the TAKE, not the engine.")
        play(audio)
        verdict = input(
            "  keep (Enter) | d = drop this take | r = replay | q = done > "
        ).strip().lower()
        if verdict == "q":
            break
        if verdict == "r":
            play(audio)
            verdict = input("  keep (Enter) | d = drop this take > ").strip().lower()
        if verdict == "d":
            exclusions.append(stem)
            print(f"  dropped {stem}")
    out = Path(args.out)
    out.write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
    print()
    print(f"dropped {len(exclusions)} takes -> {out}")
    if exclusions:
        print("Re-run the exam with:")
        print(
            "  .venv/bin/python run_recordings_exam.py "
            "--manifest manifests/intonation-recordings.json "
            f"--audio {audio_dir} --exclude {out}"
        )


if __name__ == "__main__":
    main()
