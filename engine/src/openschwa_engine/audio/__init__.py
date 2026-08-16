"""Audio front end: decode → resample 16k mono → VAD trim → quality checks.

The client uploads uncompressed 16-bit PCM WAV, so decoding needs no ffmpeg.
Everything here reports time in seconds, which keeps the API's "all times are
in the original upload timeline" invariant true regardless of resampling.
"""

from openschwa_engine.audio.decode import AudioDecodeError, DecodedAudio, decode_wav
from openschwa_engine.audio.preprocess import (
    MODEL_SAMPLE_RATE,
    PreparedAudio,
    QualityReport,
    assess_quality,
    detect_speech,
    prepare,
    resample_to_model_rate,
)

__all__ = [
    "MODEL_SAMPLE_RATE",
    "AudioDecodeError",
    "DecodedAudio",
    "PreparedAudio",
    "QualityReport",
    "assess_quality",
    "decode_wav",
    "detect_speech",
    "prepare",
    "resample_to_model_rate",
]
