"""CTC forced alignment and GOP, in pure numpy.

Deliberately independent of torch: the acoustic model produces a log-posterior
matrix and everything after that is arithmetic this module can do on its own.
That keeps the part most likely to harbour an off-by-one — lattice transitions,
frame-to-second conversion, interval tiling — unit-testable in CI without a
1.2 GB download.

The alignment is Viterbi over the standard CTC lattice: the target phone
sequence is interleaved with blanks, and each frame either stays in its state,
advances one, or skips a blank to reach a new (non-repeating) label.

Caveat carried into the M1 bake-off: CTC posteriors are *peaky*. A well-trained
model spends most frames on blank and spikes on one or two frames per phone, so
interval-averaged GOP is dominated by however the interval was drawn. GOP here
is computed over label frames only, which is the least-bad naive choice; the M1
bake-off compares this against spike-frame scoring and logit temperature.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

LogProbs = npt.NDArray[np.float32]


class AlignmentError(RuntimeError):
    """The target sequence cannot be aligned to this audio. Yields a "retry"."""


@dataclass(frozen=True)
class PhoneSegment:
    """One aligned phone, in frames. Seconds are applied by the caller, which
    knows the VAD offset into the original upload timeline."""

    index: int
    start_frame: int
    end_frame: int  # exclusive
    label_frames: int
    gop: float
    confidence: float


def forced_align(log_probs: LogProbs, targets: list[int], blank: int = 0) -> npt.NDArray[np.int32]:
    """Return the best CTC path as a per-frame extended-state index.

    Extended states alternate blank/label: `z = [blank, y0, blank, y1, ..., blank]`,
    so label `i` lives at state `2i + 1`.
    """
    if not targets:
        raise AlignmentError("empty target phone sequence")
    frames, vocab_size = log_probs.shape
    if max(targets) >= vocab_size or min(targets) < 0:
        raise AlignmentError("target contains a token outside the model vocabulary")

    num_labels = len(targets)
    states = 2 * num_labels + 1
    z = np.full(states, blank, dtype=np.int64)
    z[1::2] = targets

    # A path must emit every label, plus a blank between any repeated pair.
    repeats = sum(1 for a, b in zip(targets, targets[1:], strict=False) if a == b)
    minimum_frames = num_labels + repeats
    if frames < minimum_frames:
        raise AlignmentError(
            f"audio is too short to contain {num_labels} phones "
            f"({frames} frames < {minimum_frames} required)"
        )

    neg_inf = -np.inf
    dp = np.full((frames, states), neg_inf, dtype=np.float64)
    back = np.zeros((frames, states), dtype=np.int8)  # 0: stay, 1: from s-1, 2: from s-2

    dp[0, 0] = log_probs[0, z[0]]
    if states > 1:
        dp[0, 1] = log_probs[0, z[1]]

    # A two-state skip is legal only into a label that differs from the one two
    # states back — otherwise a repeated phone would collapse into a single one.
    skip_allowed = np.zeros(states, dtype=bool)
    skip_allowed[2:] = (z[2:] != blank) & (z[2:] != z[:-2])

    for t in range(1, frames):
        previous = dp[t - 1]
        from_one = np.full(states, neg_inf)
        from_one[1:] = previous[:-1]
        from_two = np.full(states, neg_inf)
        from_two[2:] = previous[:-2]
        from_two[~skip_allowed] = neg_inf

        candidates = np.vstack([previous, from_one, from_two])
        choice = np.argmax(candidates, axis=0)
        dp[t] = candidates[choice, np.arange(states)] + log_probs[t, z]
        back[t] = choice

    last_two = [states - 1, states - 2] if states > 1 else [0]
    end_state = max(last_two, key=lambda s: dp[frames - 1, s])
    if not np.isfinite(dp[frames - 1, end_state]):
        raise AlignmentError("no valid alignment path for this phone sequence")

    path = np.zeros(frames, dtype=np.int32)
    path[frames - 1] = end_state
    for t in range(frames - 1, 0, -1):
        path[t - 1] = path[t] - back[t, path[t]]
    return path


def segment_phones(
    path: npt.NDArray[np.int32],
    log_probs: LogProbs,
    targets: list[int],
) -> list[PhoneSegment]:
    """Turn a CTC path into contiguous, gap-free phone intervals with scores.

    Blank frames between two phones are split at their midpoint so the timeline
    tiles without holes — a UI highlighting "this segment" must not point at a
    gap. Leading and trailing blanks stay outside every phone: that is silence,
    not speech.
    """
    num_labels = len(targets)
    label_frames: list[npt.NDArray[np.int64]] = []
    for i in range(num_labels):
        frames_for_label = np.flatnonzero(path == 2 * i + 1)
        if frames_for_label.size == 0:  # unreachable for a valid path; guard anyway
            raise AlignmentError(f"phone {i} received no frames")
        label_frames.append(frames_for_label)

    boundaries = [int(label_frames[0][0])]
    for i in range(1, num_labels):
        previous_end = int(label_frames[i - 1][-1])
        current_start = int(label_frames[i][0])
        boundaries.append((previous_end + current_start) // 2 + 1)
    boundaries.append(int(label_frames[-1][-1]) + 1)

    best_per_frame = log_probs.max(axis=1)
    segments = []
    for i, token in enumerate(targets):
        frames_for_label = label_frames[i]
        target_logp = log_probs[frames_for_label, token]
        best_logp = best_per_frame[frames_for_label]
        # GOP as the log-posterior ratio against the best-matching phone: 0 is a
        # perfect match, more negative means the model preferred something else.
        gop = float(np.mean(target_logp - best_logp))
        # Confidence deliberately scores the *best* token, not the target one.
        #
        # These answer different questions, and conflating them is a real bug:
        # confidence asks "is the model sure something phone-like is happening
        # here", which is what decides whether the engine can analyse the
        # recording at all; GOP asks "was it the phone we asked for", which is a
        # pronunciation judgement. Scoring confidence on the target would make a
        # clear mispronunciation look like a processing failure, and the learner
        # would be told "try again" when they should be told what they said.
        confidence = float(np.exp(np.mean(best_logp)))
        segments.append(
            PhoneSegment(
                index=i,
                start_frame=boundaries[i],
                end_frame=max(boundaries[i + 1], boundaries[i] + 1),
                label_frames=int(frames_for_label.size),
                gop=gop,
                confidence=min(max(confidence, 0.0), 1.0),
            )
        )
    return segments


def align(
    log_probs: LogProbs, targets: list[int], blank: int = 0
) -> tuple[list[PhoneSegment], float]:
    """Align and score. Returns segments plus overall alignment confidence.

    Overall confidence is the mean of per-phone confidences, i.e. computed over
    label frames only. Averaging across every frame would instead measure how
    confidently the model predicted *blank*, which is high on any recording and
    would defeat the gate.

    It measures whether the audio could be analysed, not whether it was
    pronounced correctly — see `segment_phones` for why those must stay apart.
    """
    path = forced_align(log_probs, targets, blank=blank)
    segments = segment_phones(path, log_probs, targets)
    overall = float(np.mean([s.confidence for s in segments])) if segments else 0.0
    return segments, overall
