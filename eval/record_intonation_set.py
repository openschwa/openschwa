"""Guided recorder for the controlled intonation set.

Walks the manifest prompt by prompt: shows the line and its cue, counts
down, records ~3.5 s, plays the take back, and asks keep/retake/skip.
Files land in eval/data/recordings/ as <id>_<rep>.wav (16 kHz mono).
Existing files are skipped, so a session can be resumed any time.
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
RECORD_S = 3.5
DEFAULT_AUDIO_DIR = Path(__file__).resolve().parents[1] / "data" / "recordings"


def plan(manifest: dict, audio_dir: Path) -> list[tuple[dict, int, Path, bool]]:
    """The session plan: (item, rep, path, already-recorded)."""
    steps: list[tuple[dict, int, Path, bool]] = []
    for item in manifest["items"]:
        for rep in range(1, manifest.get("reps", 1) + 1):
            path = audio_dir / f"{item['id']}_{rep}.wav"
            steps.append((item, rep, path, path.is_file()))
    return steps


def save_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """int16 mono samples -> WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.astype("<i2").tobytes())


def _countdown(seconds: int = 3, gap: float = 0.7) -> None:
    for n in range(seconds, 0, -1):
        print(n, "...", end=" ", flush=True)
        time.sleep(gap)
    print("recording", flush=True)


def _play(path: Path) -> None:
    import subprocess

    subprocess.run(["afplay", str(path)], check=False, capture_output=True)


def record_session(manifest: dict, audio_dir: Path) -> dict[str, int]:
    """The interactive loop. Returns counts by outcome."""
    steps = plan(manifest, audio_dir)
    done, retaken, skipped, missing = 0, 0, 0, 0
    index = 0
    while index < len(steps):
        item, rep, path, exists = steps[index]
        tone = item["tone"].upper().replace("_", "-")
        if exists:
            index += 1
            continue
        print()
        print(f"[{index + 1}/{len(steps)}] ({tone}) {item['text']!r}  (take {rep})")
        print(f"    cue: {item['cue']}")
        answer = input("    Enter to record | q to quit | s to skip > ").strip().lower()
        if answer == "q":
            break
        if answer == "s":
            skipped += 1
            index += 1
            continue
        _countdown()
        samples = sd.rec(
            int(RECORD_S * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16"
        )
        sd.wait()
        save_wav(path, samples, SAMPLE_RATE)
        _play(path)
        verdict = input("    keep (Enter) | r to retake > ").strip().lower()
        if verdict == "r":
            retaken += 1
            continue  # same step again
        done += 1
        index += 1
    missing = sum(1 for _item, _rep, path, _exists in steps if not path.is_file())
    return {"recorded": done, "retakes": retaken, "skipped": skipped, "still_missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(
            Path(__file__).resolve().parents[1]
            / "manifests"
            / "intonation-recordings.json"
        ),
        help="the item manifest JSON",
    )
    parser.add_argument("--audio", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    audio_dir = Path(args.audio)
    steps = plan(manifest, audio_dir)
    recorded = sum(1 for _i, _r, _p, exists in steps if exists)
    print(
        f"{len(steps)} takes planned; {recorded} already recorded; "
        f"recording {len(steps) - recorded} now."
    )
    print("Speak at normal volume, ~30 cm from the mic. Ctrl-C pauses the session.")
    try:
        counts = record_session(manifest, audio_dir)
    except KeyboardInterrupt:
        steps_after = plan(manifest, audio_dir)
        counts = {
            "recorded": 0,
            "retakes": 0,
            "skipped": 0,
            "interrupted": True,
            "still_missing": sum(1 for _i, _r, _p, exists in steps_after if not exists),
        }
    print()
    print("session summary:", counts)
    if counts.get("still_missing", 0) == 0:
        print("All takes recorded. Run the exam:")
        print(
            f"  .venv/bin/python run_recordings_exam.py "
            f"--manifest {args.manifest} --audio {audio_dir}"
        )
    else:
        print("Re-run this script to resume - existing takes are skipped.")


if __name__ == "__main__":
    main()
