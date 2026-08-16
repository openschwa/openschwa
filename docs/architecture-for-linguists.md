# OpenSchwa architecture — a linguist's guide (no code required)

This is the plain-language twin of [architecture.md](architecture.md). It
explains what the system does, how a recording travels through it, and why the
design refuses to do things it cannot do reliably — no code, no formulas, no
programming background assumed. It assumes you know what a phoneme is;
everything else is explained as it appears.

> The technical document is the reference for contributors. This one is the
> same system, told for the phonetics-literate reader.

## The big picture

OpenSchwa is a pronunciation practice tool that lives in two rooms of the same
house:

- **The front desk** — the interface in your browser. It records you, draws the
  spectrogram, the timeline of sounds, and the pitch curve. It knows nothing
  about phonetics itself; it only displays and collects.
- **The lab** — a Python engine running on your computer. It listens to your
  recording, aligns it to the sounds you were asked to produce, measures pitch,
  and decides what it can honestly say about your pronunciation.

The two rooms exchange letters in a fixed format — a "contract" that both sides
have signed. The front desk asks a question ("analyse this recording of
exercise X"), and the lab answers with a structured report: time intervals,
scores, confidence values. Because the format is fixed and versioned, the two
rooms can be rearranged later — the same lab can serve a desktop app or a
website — without either side changing how it works.

Nothing leaves your machine. The recording, the model, the analysis: all local.
Once the acoustic model is downloaded once, the whole thing works offline.

## The journey of one recording

> **Current status (M0):** today the app *measures*. From M1 on it will also
> *judge* — and only when it can prove it is right (see "The shipping bar"
> below).

Here is what happens between you pressing record and seeing your analysis.

**1. Recording.** The browser captures the rawest possible microphone signal.
It deliberately switches off the browser's usual "help": echo cancellation,
noise suppression, and automatic gain control. Those clean-ups distort exactly
the acoustic details being measured — aspiration, VOT, loudness dynamics. A
side effect: recordings sound quieter than a phone memo. That is normal, and
quiet is not bad — see step 5.

**2. Packing and sending.** The audio is packed into a WAV file and posted to
the lab through a private corridor (localhost — the letter never leaves the
house).

**3. Preparation.** The lab converts the recording to a standard analysis
format: 16 kHz, one channel.

**4. Finding the speech.** A voice activity detector trims away the silence
around your word. It must be careful not to mistake quiet sounds for silence:
the /s/ in *this* sits 20–30 dB below the neighbouring vowel. A naive detector
would cut it off; this one uses hysteresis — it only "gives up" on the sound
after a longer stretch of quiet. (M1 will upgrade this to a smarter,
neural-network-based detector.)

**5. Quality checks.** Is there any sound at all? Is it clipped? "Too quiet"
now means only *dead* — an unplugged or muted microphone. Quiet-but-clean
speech is analysed normally: the acoustic model normalises everything it sees,
so absolute loudness carries no information for it.

**6. Forced alignment — the heart of the system.** The exercise says which
sounds to expect: *this* = /ð ɪ s/. An acoustic model — a neural network
trained on enormous amounts of speech — listens to your recording and stamps
each expected phone onto the timeline: /ð/ from 0.08 to 0.12 s, /ɪ/ from 0.12
to 0.22 s, and so on. Think of a subtitle editor who already knows the script
and only has to mark *when* each line happens.

Notice what this is not: it is **not** transcription. The machine never has to
recognise what you said — it is told what you were trying to say, and asked
only to find where each sound lives in time. That is a much easier, much more
reliable question, and it is the one that matters for feedback.

**7. Scoring each sound.** For every phone, the model asks: at those moments,
how expected was the target sound — and how surprised was it? The gap between
"expected" and "surprised" becomes a score per phone (the standard measure is
called **GOP**, Goodness of Pronunciation). No thresholds are applied here yet;
it is just a measurement.

**8. The contrast question.** For the focus sound of the exercise, the lab
asks a closed question instead of an open one: "was this moment more like /ð/
or more like /z/?" Only those two candidates compete — like grading on a curve
of two. A closed question with known alternatives is dramatically more reliable
than asking "what sound was that?" about accented speech, which even the best
transcription systems get wrong far too often.

**9. Pitch.** A famous phonetics workhorse — **Praat** (wrapped for Python) —
measures your fundamental frequency frame by frame. The lab cleans up octave
jumps, converts pitch to **semitones relative to your own median**, so a deep
male voice and a high female voice are each judged on their own scale. Today it
only draws the curve; M2 will compare it against the teacher's contour
(stretching the two curves in time to match them — a technique called DTW) and
reason about the nuclear tone.

**10. The confidence gate.** Before anything can be shown to you as feedback,
the lab asks one more question: *how sure am I?* Two different things are being
measured, and the design insists on keeping them apart:

- **Confidence** — "is something phone-like even happening here?" → *can this
  recording be analysed at all?*
- **Score** — "was it the sound we asked for?" → *was the pronunciation good?*

Mixing them up is the classic failure: a clear mispronunciation would then be
reported as a processing error, and the student would be told "try again" at
exactly the moment they deserve an explanation. Only confidence feeds the gate.

**11. The answer.** The lab replies with a report: time intervals, scores,
confidence values — and a **feedback list**. At M0 that list can contain only
one thing: *retry*. The app shows measurements honestly and refuses to judge
until each judgment type has passed its exam (the shipping bar below).

## Three promises the design makes

**No feedback beats wrong feedback.** Wrong feedback teaches a nonexistent
mistake and destroys trust; silence costs one retry. Therefore every judgment
is confidence-gated, and a new *kind* of feedback ships only after an offline
exam proves it: the eval harness runs the candidate against real L2 speech
corpora and measures precision and recall. The pass mark for M1's contrast
feedback is **precision ≥ 0.90 at recall ≥ 0.4**. The exam board is
[eval/README.md](../eval/README.md).

**Scripted drills only, never open transcription.** A general speech
recogniser is trained to be robust to accents: it hears *zis* and politely
writes *this*. A machine whose job is to catch that mistake cannot be one that
erases it. The engine always knows the target phone sequence and the expected
confusions, so scoring is always a closed-set question. Conveniently, this is
also how minimal-pair pedagogy works, so the constraint costs the product
nothing.

**Honest refusal.** If the acoustic model is not installed (it is a ~1.3 GB
optional download), the app still runs: it serves exercises, measures audio and
pitch, and answers every analysis with a single "I cannot judge yet" — never
with a guess. Recording problems are diagnosed *before* the model is even
consulted, so those messages stay specific.

## The pieces at a glance

| Piece | What it is | What a linguist should know |
|---|---|---|
| UI (Svelte, in the browser) | The front desk | Draws the spectrogram, phone timeline, and pitch curve; records; knows no phonetics |
| Engine (Python, FastAPI) | The lab | Decodes, aligns, scores, measures pitch, composes feedback |
| Acoustic model | The ear | A wav2vec2 network trained on many languages and fine-tuned with pronunciation dictionaries (espeak); stamps phones onto time |
| Praat (via parselmouth) | The pitch expert | The standard tool of experimental phonetics, called from Python |
| The contract (JSON Schema) | The letter format | Versioned; both sides generate their copies from one source of truth, so they cannot drift apart |
| Content packs (YAML) | The exercise books | Each exercise: text, IPA, phone list, exactly one focus phone + its confusion set, optional reference audio |
| Eval harness | The exam board | Offline; certifies each new feedback type against L2 corpora before it ships |

## If you then read the technical document

The full specification is [architecture.md](architecture.md). A reading map,
translated from this guide's language:

- §1 "Design principles" = the three promises above, stated as invariants.
- §3 "The API contract" = the letter format, field by field.
- §4 "Engine pipeline" = steps 3–11 in technical order, plus the known
  weak spots and why they are owned up front.
- §5 "Content format" = the exercise books.
- §6 "Milestones" = what M0–M4 must each prove to count as done.
- §7 "Evaluation" = the exam board's procedure and the shipping bar.
- §8 "Risks" = everything that can go wrong and the planned counter-move.
- §9 "Carried out of M0" = the deliberate gaps and which milestone closes each.

## Glossary

| Term | Meaning here |
|---|---|
| Phone / phoneme | A speech sound; here, a symbol from the canonical IPA inventory the engine works with |
| Minimal pair | Two words differing in one sound (*this* / *zis*) — the basic unit of the drills |
| Forced alignment | Stamping the *expected* phone sequence onto the recording's timeline |
| Transcription | Recognising what was said — deliberately avoided (see the promises) |
| VAD | Voice activity detection: finding where speech begins and ends |
| F0 / pitch | Fundamental frequency; the acoustic correlate of perceived pitch |
| Semitone | A musical step; pitch expressed in steps rather than hertz, so voices compare fairly |
| DTW | Dynamic time warping: stretching two contours in time to compare their shapes |
| Spectrogram | The picture of sound: time horizontally, frequency vertically, energy as colour |
| VOT | Voice onset time: the gap between a stop's release and the start of voicing |
| GOP | Goodness of Pronunciation: how expected the target phone was, versus what was heard |
| Confidence | How sure the engine is that the recording is analysable at all — the gate's input |
| Calibration | Teaching the scores to mean the same thing across recordings and voices |
| Corpus (pl. corpora) | A collection of recordings with known labels, used to examine the engine |
| Eval harness | The offline exam rig that runs those exams |
| Closed-set discrimination | Asking "which of these known candidates?" instead of "what is this?" |
