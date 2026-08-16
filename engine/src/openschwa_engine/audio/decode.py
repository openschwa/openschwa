"""WAV decoding.

The client encodes 16-bit PCM WAV itself (docs/architecture.md §3) precisely so
the engine needs no ffmpeg and no lossy-codec round trip. This parser therefore
accepts uncompressed RIFF/WAVE only and says so plainly when handed anything
else — a silent fallback here would mean scoring audio that had been through a
codec that erased the cues we measure.

32-bit float WAV is accepted as a convenience for eval corpora and recording
tools; everything else is rejected.
"""

import struct
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

MAX_CHANNELS = 8


class AudioDecodeError(ValueError):
    """The upload is not WAV the engine can read. Surfaces to the client as 400."""


@dataclass(frozen=True)
class DecodedAudio:
    """Mono float32 in [-1, 1] at the rate it was recorded at.

    The original rate is preserved because every time in the API is expressed in
    the original upload timeline; resampling happens downstream for the model.
    """

    samples: npt.NDArray[np.float32]
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return float(len(self.samples)) / self.sample_rate


def _iter_chunks(data: bytes) -> "list[tuple[bytes, int, int]]":
    """Yield (chunk_id, start, size) for each RIFF chunk body."""
    chunks = []
    pos = 12  # past 'RIFF' + size + 'WAVE'
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        body = pos + 8
        if body + size > len(data):
            # Streaming writers sometimes leave a truncated final chunk length;
            # take what is actually there rather than failing the whole upload.
            size = len(data) - body
        chunks.append((chunk_id, body, size))
        pos = body + size + (size & 1)  # chunks are word-aligned
    return chunks


def decode_wav(data: bytes) -> DecodedAudio:
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioDecodeError("not a RIFF/WAVE file — the client must upload 16-bit PCM WAV")

    chunks = {cid: (start, size) for cid, start, size in _iter_chunks(data)}
    if b"fmt " not in chunks or b"data" not in chunks:
        raise AudioDecodeError("WAV is missing a 'fmt ' or 'data' chunk")

    fmt_start, fmt_size = chunks[b"fmt "]
    audio_format, channels, sample_rate, _byte_rate, _align, bits = struct.unpack_from(
        "<HHIIHH", data, fmt_start
    )
    if audio_format == WAVE_FORMAT_EXTENSIBLE and fmt_size >= 40:
        # The real format sits in the extension's GUID; its first two bytes are
        # the format tag.
        (audio_format,) = struct.unpack_from("<H", data, fmt_start + 24)

    if audio_format not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
        raise AudioDecodeError(
            f"compressed WAV (format 0x{audio_format:04x}) is not supported — "
            "upload uncompressed 16-bit PCM"
        )
    if not 1 <= channels <= MAX_CHANNELS:
        raise AudioDecodeError(f"unsupported channel count: {channels}")
    if sample_rate <= 0:
        raise AudioDecodeError("WAV declares a non-positive sample rate")

    data_start, data_size = chunks[b"data"]
    raw = data[data_start : data_start + data_size]

    if audio_format == WAVE_FORMAT_IEEE_FLOAT and bits == 32:
        frames = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif bits == 16:
        frames = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif bits == 8:
        # 8-bit PCM is unsigned with a 128 offset, unlike every other width.
        frames = (np.frombuffer(raw, dtype="u1").astype(np.float32) - 128.0) / 128.0
    elif bits == 24:
        packed = np.frombuffer(raw[: len(raw) - len(raw) % 3], dtype="u1").reshape(-1, 3)
        as_int = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int8).astype(np.int32) << 16)
        )
        frames = as_int.astype(np.float32) / 8388608.0
    elif bits == 32:
        frames = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioDecodeError(f"unsupported bit depth: {bits}")

    if frames.size == 0:
        raise AudioDecodeError("WAV contains no audio frames")

    usable = frames.size - (frames.size % channels)
    mono = frames[:usable].reshape(-1, channels).mean(axis=1) if channels > 1 else frames

    return DecodedAudio(
        samples=np.ascontiguousarray(np.clip(mono, -1.0, 1.0), dtype=np.float32),
        sample_rate=sample_rate,
    )
