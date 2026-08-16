"""Acoustic measurements (M1–M3).

Contract: per-phone durations (from alignment); voicing fraction (Praat);
VOT for utterance-initial voiceless stops only (burst detection → voicing
onset), with internal reliability checks that gate out unclear tokens rather
than mis-measure them; formant medians (Burg) over vowel steady-state with a
reliability score — formants are supporting evidence only, never the basis of
a vowel verdict.
"""
