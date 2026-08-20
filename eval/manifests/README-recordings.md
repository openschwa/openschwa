# Recording guide - the intonation exam set

You are recording 68 short prompts, twice each (136 recordings, ~20-25
minutes). Each prompt says exactly what tone to produce - there are no
"wrong" voices here, only wrong *melodies*, and those get caught by the
verification pass, not counted against the engine.

## What you need

- A quiet room (turn off music/TV; close the door).
- Any recorder: this Mac (QuickTime Player -> File -> New Audio Recording,
  or the Voice Memos app), or your phone's voice recorder.
- Speak at normal volume, ~30 cm from the microphone.

## How one recording works

1. Read the prompt and its cue. The cue is the *situation* - it tells you
   the melody. "Please." = a firm request (falls). "Please?" = a surprised
   echo (rises).
2. Say the line ONCE, then stop and stay quiet for a second.
3. Save it as a file named `<item id>_<rep>.<ext>`, e.g. `fr-01a_1.m4a`
   for the first take of "Please.", `fr-01a_2.m4a` for the second.
4. Put all files in `eval/data/recordings/` (create the folder).

The manifest (what to say, in order, with cues) is
`eval/manifests/intonation-recordings.json`. Work through it top to
bottom; the ids are in the file. If a take goes wrong (cough, misread,
wrong melody), just record that take again and overwrite the file.

Tips for the melodies:

- **Fall**: let the pitch slide *down* on the last word, like closing a
  door. Firm, decided.
- **Rise**: let the pitch slide *up* at the end, like asking. The rising
  glide starts late - keep the start level-ish and lift at the very end.
- **Fall-rise**: dip down, then lift a little - the voice of "well,
  maybe...". Takes practice; the cue examples are the guide.
- **Level**: keep the pitch flat and hanging, like the sentence isn't
  finished. Lists work: "one... two... three..." with no final drop.

Don't worry about being theatrical - natural, clear speech is exactly what
we want. This is what the app will hear from learners.

## After recording

Tell me the files are in place (or ask me to convert/split them), and I
run: `python run_recordings_exam.py --manifest manifests/intonation-recordings.json
--audio data/recordings`. The report prints a verification list - we then
listen to the flagged takes together (or you check them) and drop the ones
where the melody wasn't right. Only then does the number count.
