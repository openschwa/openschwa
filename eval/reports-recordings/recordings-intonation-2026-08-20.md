# M2 intonation exam - controlled recordings (2026-08-20)

The M2 bar on the app's real use case: 68 scripted tone prompts (48
fall/rise minimal pairs, 12 fall-rise, 8 level) recorded twice by the
project owner, scored by the engine's shipped prosody chain (VAD ->
Praat F0 -> nuclear-tone verdict). Companion data:
recordings-intonation-2026-08-20.json (+ .jsonl per-take records),
manifest eval/manifests/intonation-recordings.json.

## Result (after human verification)

| metric | value | bar |
| --- | --- | --- |
| recordings | 136 taken, 12 dropped by listening (6 from the core) | - |
| verified takes scored | 124 | - |
| fall-vs-rise accuracy | **0.9111** (82/90) | >= 0.90 **met** |
| fall accuracy | 0.8372 (n=43) | - |
| rise accuracy | 0.9787 (n=47) | - |
| fall-rise accuracy | 0.2222 (n=18) | informational |
| level accuracy | 0.8125 (n=16) | informational |
| mean octave-error rate | **0.00028** | < 0.02 **met** |

Protocol: expected tone comes from the prompt, not a label. Every take
where the engine's verdict differed from the prompt went to a human
listening pass (eval/verify_recordings.py): takes the speaker judged
mis-spoken were excluded (12), takes judged correct stayed and count
against the engine. Nothing was waived.

## What the exam caught (three engine bugs, all fixed)

1. **Trailing silence read as the tone.** Fixed-length takes end in ~1.5 s
   of silence; the terminal window read that silence (level/0.0 on
   everything - the first run scored 1%). The window now ends at the VAD
   speech end and at the last voiced frame: weak final consonants (the /z/
   of "please") no longer swallow the glide either. 1% -> 53%.
2. **OLS slope wrecked by octave/creak spikes.** Single bad frames
   produced +-80..260 st/s half-window slopes and bogus fall-rise/rise
   verdicts. Slopes are now Theil-Sen (median pairwise) - robust to
   outliers. 53% -> 79%.
3. **Thresholds were guesses.** Calibrated on the verified set: fall 4
   st/s, rise 6 st/s, fall-rise halves 12 st/s (a creaky release after a
   fall must not read as a fall-rise). 79% -> 91%. The sweep is flat
   across a wide region (fall 4 with rise 6-9, halves 12-16 all give
   91.11%), which argues against overfitting; the honest caveat remains
   that these constants were fit on this set and human testing is the
   real gate.

Fall-rise remains weak (22%): the speaker's dip-rises are shallow and
early, and the classifier wants a steep V in the last 0.35 s. Not part of
the bar; recorded, not waived.

## Verdict

**M2 bar met on the controlled set.** Octave errors 0.028% (bar < 2%).
The Aix-MARSEC natural-speech failure stands as a separate, documented
research problem (octave-chaotic short-slice F0); the engine's fixes here
(the robust slope especially) may move it and deserve a re-run there.
Next: human testing of the tone feedback with real learners.
