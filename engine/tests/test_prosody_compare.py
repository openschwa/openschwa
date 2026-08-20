"""Tone classification and DTW over synthetic contours (no audio needed)."""

from openschwa_engine.prosody import F0Track, contour_match, dtw_distance, nuclear_tone


def _track(values, hop_s=0.05):
    return F0Track(
        hop_s=hop_s,
        start_s=0.0,
        semitones=tuple(values),
        median_hz=150.0,
    )


def test_steep_terminal_fall_is_a_fall():
    # 40 frames, last 0.35 s (7 frames) falls 5 st -> ~ -14 st/s.
    values = [0.0] * 33 + [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    tone, confidence = nuclear_tone(_track(values))
    assert tone == "fall"
    assert confidence > 0.5


def test_steep_terminal_rise_is_a_rise():
    values = [0.0] * 33 + [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert nuclear_tone(_track(values))[0] == "rise"


def test_flat_terminal_is_level():
    values = [0.0] * 40
    assert nuclear_tone(_track(values))[0] == "level"


def test_fall_then_rise_is_fall_rise():
    # Inside the 7-frame terminal window: a steep fall (-40 st/s) then a
    # steep rise (+40 st/s).
    values = [0.0] * 33 + [2.0, 0.0, -2.0, -4.0, -2.0, 0.0, 2.0]
    assert nuclear_tone(_track(values))[0] == "fall_rise"


def test_dtw_is_zero_for_identical_contours():
    track = _track([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    assert dtw_distance(track, track) == 0.0


def test_dtw_measures_contour_difference():
    flat = _track([0.0] * 10)
    rising = _track(list(range(10)))
    assert dtw_distance(flat, rising) > 0.0


def test_contour_match_scores_identity_one_and_divergence_low():
    track = _track([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    assert contour_match(track, track) == 1.0
    rising = _track(list(range(10)))
    assert contour_match(track, rising) < 0.5


def test_unvoiced_contours_are_unmeasurable():
    sparse = _track([None, None, 1.0, None, None])
    tone, confidence = nuclear_tone(sparse)
    assert tone == "level"
    assert confidence == 0.0
    assert dtw_distance(sparse, _track([1.0, 2.0, 3.0])) is None