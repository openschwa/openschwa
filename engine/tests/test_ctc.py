"""CTC forced alignment, tested on hand-built posteriors.

No model involved: the point is that the lattice, the backtrace, and the
frame-to-interval tiling are right, and those are pure arithmetic. Every phone
boundary the UI draws and every GOP the eval harness reads depends on this file.
"""

import numpy as np
import pytest

from openschwa_engine.alignment.ctc import AlignmentError, align, forced_align, segment_phones

VOCAB = 5
BLANK = 0


def log_probs_for(frame_tokens: list[int], confidence: float = 0.9) -> np.ndarray:
    """Posteriors that put `confidence` on the intended token of each frame."""
    probs = np.full((len(frame_tokens), VOCAB), (1.0 - confidence) / (VOCAB - 1))
    probs[np.arange(len(frame_tokens)), frame_tokens] = confidence
    return np.log(probs).astype(np.float32)


def test_recovers_the_generating_path():
    frames = [BLANK, 1, 1, BLANK, 2, 2, BLANK, 3, 3, BLANK]
    path = forced_align(log_probs_for(frames), [1, 2, 3], blank=BLANK)
    # Extended states alternate blank/label, so label i sits at state 2i+1.
    assert list(path) == [0, 1, 1, 2, 3, 3, 4, 5, 5, 6]


def test_repeated_phones_require_a_separating_blank():
    frames = [1, BLANK, 1]
    path = forced_align(log_probs_for(frames), [1, 1], blank=BLANK)
    assert list(path) == [1, 2, 3]


def test_repeated_phones_reject_audio_with_no_room_for_the_blank():
    with pytest.raises(AlignmentError, match="too short"):
        forced_align(log_probs_for([1, 1]), [1, 1], blank=BLANK)


def test_too_few_frames_is_an_alignment_error_not_a_crash():
    with pytest.raises(AlignmentError, match="too short"):
        forced_align(log_probs_for([1, 2]), [1, 2, 3], blank=BLANK)


def test_empty_target_is_rejected():
    with pytest.raises(AlignmentError):
        forced_align(log_probs_for([1]), [], blank=BLANK)


def test_target_outside_the_vocabulary_is_rejected():
    with pytest.raises(AlignmentError, match="vocabulary"):
        forced_align(log_probs_for([1]), [99], blank=BLANK)


def test_segments_tile_without_gaps_or_overlap():
    frames = [BLANK, 1, 1, BLANK, BLANK, 2, 2, BLANK, 3, BLANK]
    probs = log_probs_for(frames)
    targets = [1, 2, 3]
    segments = segment_phones(forced_align(probs, targets, blank=BLANK), probs, targets)

    assert [s.index for s in segments] == [0, 1, 2]
    for earlier, later in zip(segments, segments[1:], strict=False):
        assert earlier.end_frame == later.start_frame, "a UI highlight must never land in a gap"
    assert all(s.end_frame > s.start_frame for s in segments)


def test_blank_gap_between_phones_is_split_at_its_midpoint():
    # Phone 1 ends at frame 1; phone 2's label starts at frame 6.
    frames = [1, 1, BLANK, BLANK, BLANK, BLANK, 2, 2]
    probs = log_probs_for(frames)
    targets = [1, 2]
    segments = segment_phones(forced_align(probs, targets, blank=BLANK), probs, targets)
    assert segments[0].end_frame == 4
    assert segments[1].start_frame == 4


def test_leading_and_trailing_silence_stays_outside_every_phone():
    frames = [BLANK, BLANK, 1, BLANK, BLANK]
    probs = log_probs_for(frames)
    segments = segment_phones(forced_align(probs, [1], blank=BLANK), probs, [1])
    assert segments[0].start_frame == 2
    assert segments[0].end_frame == 3


def test_gop_is_zero_when_the_target_is_the_best_scoring_token():
    frames = [BLANK, 1, 1, BLANK]
    segments, confidence = align(log_probs_for(frames, confidence=0.95), [1], blank=BLANK)
    assert segments[0].gop == pytest.approx(0.0)
    assert confidence == pytest.approx(0.95, abs=1e-6)


def test_gop_is_negative_when_the_model_prefers_another_phone():
    """The learner said /2/ where the exercise expects /1/: alignment still
    succeeds — that is what makes a substitution detectable — but GOP drops."""
    probs = log_probs_for([BLANK, 2, 2, BLANK])
    segments, _ = align(probs, [1], blank=BLANK)
    assert segments[0].gop < -1.0


def test_a_confident_mispronunciation_stays_confident():
    """The distinction this whole module rests on: confidence says "the audio
    could be analysed", GOP says "it was not the phone we asked for".

    If confidence tracked the target instead, a clear substitution would look
    like a processing failure and the learner would be told to try again rather
    than what they actually said.
    """
    probs = log_probs_for([BLANK, 2, 2, BLANK], confidence=0.95)
    segments, confidence = align(probs, [1], blank=BLANK)
    assert confidence == pytest.approx(0.95, abs=1e-6), "model was sure — just not of /1/"
    assert segments[0].gop < -1.0, "and GOP is what records the mismatch"


def test_confidence_falls_when_the_model_is_sure_of_nothing():
    """Noise or mumbling: no token dominates, so the recording genuinely cannot
    be analysed and the gate should fire."""
    _, confidence = align(log_probs_for([BLANK, 1, 1, BLANK], confidence=0.25), [1], blank=BLANK)
    assert confidence < 0.4


def test_confidence_ignores_blank_frames():
    """Averaging over every frame would mostly measure how sure the model is
    that nothing is happening, which is high on any recording."""
    mostly_blank = [BLANK] * 20 + [1] + [BLANK] * 20
    _, confidence = align(log_probs_for(mostly_blank, confidence=0.99), [1], blank=BLANK)
    assert confidence == pytest.approx(0.99, abs=1e-6)
