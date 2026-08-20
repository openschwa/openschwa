r"""Aix-MARSEC adapter: intonation units with SEC tonetic-stress-mark labels.

The corpus (Hirst, Auran & Bouzon 2002-2004, version 2009) is freely
distributable with its notice. 53 block wavs (~5.5 h) hold 413 passages
concatenated; each passage has a Praat TextGrid (short format) whose
"Text" tier carries the SEC tonetic stress marks - \ = fall, / = rise -
hand-annotated (BJW/GOK) and corrected. This adapter slices passages out
of their block wavs (energy-refined offsets: the corpus duration table
drifts a few percent from the real audio) and yields one IntonationUnit
per tone-unit-sized stretch, with the nuclear tone from the TSM.
"""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openschwa_eval import textgrid

#: SAMPA -> canonical IPA for the MARSEC phoneme tier. Only the phones the
#: alignment/contrast machinery may see are mapped; anything else is loud.
SAMPA_TO_CANONICAL = {
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "ɡ",
    "tS": "tʃ", "dZ": "dʒ", "f": "f", "v": "v", "T": "θ", "D": "ð",
    "s": "s", "z": "z", "S": "ʃ", "Z": "ʒ", "h": "h", "m": "m", "n": "n",
    "N": "ŋ", "l": "l", "r": "ɹ", "w": "w", "j": "j",
    "i:": "i", "I": "ɪ", "e": "ɛ", "{": "æ", "A:": "ɑ", "Q": "ɔ",
    "O:": "ɔ", "U": "ʊ", "u:": "u", "3:": "ɝ", "@": "ə", "V": "ʌ",
    "eI": "eɪ", "aI": "aɪ", "OI": "ɔɪ", "@U": "oʊ", "aU": "aʊ",
    "I@": "ɪ", "e@": "ɛ", "U@": "ʊ", "i": "i", "u": "u",
}
#: Silence / structural labels on the phoneme tier (never mapped to phones).
STRUCTURAL_LABELS = frozenset({'_', '#', '{', 'END', '', '?', '!'})
#: SEC tonetic stress marks we use as ground truth. The M2 bar covers
#: fall vs rise; the rest are parsed and reported, not gated.
TSM_TONES = {'\\': 'fall', '/': 'rise', '`/': 'fall_rise', '=': 'level'}
TSM_PREFIXES = ('\\', '/', '`/', '=')
BLOCK_PAD_S = 0.05  # silence margin when slicing a passage out of its block


class AixMarsecError(ValueError):
    """The corpus files cannot be parsed into intonation units."""


@dataclass(frozen=True)
class IntonationUnit:
    passage_id: str  # e.g. G0201G
    block_id: str  # e.g. G02
    annotator: str  # B (BJW) or G (GOK)
    audio_path: Path  # the block wav this unit must be sliced from
    start_s: float  # in the block wav's timeline
    end_s: float
    expected_tone: str  # fall | rise | fall_rise | level
    transcript: str


def _parse_short_textgrid(text: str) -> list[textgrid.Tier]:
    """Parse an "ooTextFile short" TextGrid into IntervalTiers + PointTiers."""
    lines = [line.strip() for line in text.splitlines()]
    if not lines or 'ooTextFile short' not in lines[0]:
        raise textgrid.TextGridError('not a short-format TextGrid')
    compact = [line for line in lines if line]
    try:
        header = compact.index('<exists>')
    except ValueError as exc:
        raise textgrid.TextGridError('no <exists> marker') from exc
    n_tiers = int(compact[header + 1])
    index = header + 2
    tiers: list[textgrid.Tier] = []
    point_tiers: list[tuple[str, list[tuple[float, str]]]] = []
    for _ in range(n_tiers):
        kind = compact[index].strip('"')
        index += 1
        name = compact[index].strip('"')
        index += 1
        _xmin = compact[index]
        _xmax = compact[index + 1]
        count = int(compact[index + 2])
        index += 3
        intervals: list[textgrid.Interval] = []
        if kind == 'IntervalTier':
            for _ in range(count):
                a = float(compact[index])
                b = float(compact[index + 1])
                label = compact[index + 2].strip('"')
                index += 3
                intervals.append(textgrid.Interval(a, b, label))
            tiers.append(textgrid.Tier(name, tuple(intervals)))
        elif kind == 'TextTier':
            points = []
            for _ in range(count):
                t = float(compact[index])
                label = compact[index + 1].strip('"')
                index += 2
                points.append((t, label))
            point_tiers.append((name, points))
        else:  # pragma: no cover - unknown tier kinds are loud, not silent
            raise textgrid.TextGridError(f'unknown tier kind {kind!r} in {name!r}')
    return tiers


def _energy_profile(samples: np.ndarray, sample_rate: int, frame_s: float = 0.02) -> np.ndarray:
    """Frame RMS, decimated to frame_s steps. Cheap silence detection."""
    frame = max(1, int(frame_s * sample_rate))
    if samples.size < frame:
        return np.array([0.0])
    trimmed = samples[: samples.size - samples.size % frame]
    frames = trimmed.reshape(-1, frame).astype(np.float64)
    return np.sqrt(np.mean(frames**2, axis=1))


def refine_offsets(
    block_wav: Path,
    passages: list[tuple[str, float]],
) -> dict[str, float]:
    """Passage start offsets inside a block wav, refined by silence valleys.

    The corpus duration table drifts a few percent from the real audio, so
    cumulative-table offsets can be off by a second or two by the block's
    end. Passages sit between silence gaps: each table-prior boundary is
    snapped to the lowest-energy 40 ms frame within +-3 s, which is where
    the true gap is. Passages whose prior lands beyond the wav are dropped
    (truncated blocks, e.g. A12).
    """
    with wave.open(str(block_wav)) as handle:
        assert handle.getframerate() == 16_000, block_wav
        samples = (
            np.frombuffer(handle.readframes(handle.getnframes()), dtype='<i2').astype(
            np.float32
            )
            / 32768.0
        )
    sample_rate = 16_000
    energy = _energy_profile(samples, sample_rate)
    total_s = len(samples) / sample_rate
    offsets: dict[str, float] = {}
    cursor = 0.0
    for passage_id, duration in passages:
        if cursor + duration > total_s + 2.0:
            break  # truncated block: the tail passages have no audio
        offsets[passage_id] = cursor
        prior = cursor + duration
        # Snap the prior into the deepest silence within +-3 s, never before
        # the previous passage (the first silence is not this boundary's).
        frame_s = 0.02
        lo = max(int((cursor + 0.5) / frame_s), int((prior - 3.0) / frame_s))
        hi = min(len(energy) - 1, int((prior + 3.0) / frame_s))
        if hi > lo:
            valley = lo + int(np.argmin(energy[lo : hi + 1]))
            cursor = valley * frame_s
        else:
            cursor = min(prior, total_s)
    return offsets


def _units_from_text_tier(
    text_tier: textgrid.Tier, offsets: dict[str, float], passage_id: str, block_id: str
) -> list[IntonationUnit]:
    """Intonation units: word stretches between SEC boundary markers.

    A unit ends where a word label carries '|' or '||' (or 'END'). The
    nuclear TSM (the unit's last fall/rise mark) is its expected tone;
    units without one are skipped (the M2 bar covers fall vs rise).
    """
    units: list[IntonationUnit] = []
    current: list[textgrid.Interval] = []
    for interval in text_tier.intervals:
        label = interval.text
        if not label:
            continue
        current.append(interval)
        if re.search(r'[|#]', label) or label == 'END':
            units.append(_finish_unit(current, passage_id, block_id, offsets))
            current = []
    if current:
        units.append(_finish_unit(current, passage_id, block_id, offsets))
    return [unit for unit in units if unit is not None]


def _finish_unit(
    intervals: list[textgrid.Interval],
    passage_id: str,
    block_id: str,
    offsets: dict[str, float],
) -> IntonationUnit | None:
    offset = offsets.get(passage_id)
    if offset is None or not intervals:
        return None
    tone: str | None = None
    words: list[str] = []
    for interval in intervals:
        label = interval.text
        # Onset markers (< low, > high) precede the tone mark on the nucleus.
        while label[:1] in ('<', '>'):
            label = label[1:]
        for prefix in TSM_PREFIXES:
            if label.startswith(prefix):
                tone = TSM_TONES[prefix]  # the last mark wins: that is the nucleus
                label = label[len(prefix):]
                break
        label = re.sub(r'[|#]', '', label).strip(', ').strip()
        if label:
            words.append(label)
    if tone not in ('fall', 'rise', 'fall_rise'):
        # '`' is NOT a tone (INTSINT cross-check: mixed movement); units
        # without a trustworthy mark are skipped. The bar gates fall vs rise.
        return None
    start = intervals[0].xmin
    end = intervals[-1].xmax
    if end - start < 0.4:
        return None
    return IntonationUnit(
        passage_id=passage_id,
        block_id=block_id,
        annotator=passage_id[-1],
        audio_path=Path(),  # filled by the corpus (block wav path)
        start_s=round(offset + start, 4),
        end_s=round(offset + end, 4),
        expected_tone=tone,
        transcript=' '.join(words),
    )



class AixMarsec:
    """The Aix-MARSEC corpus, yielding IntonationUnits."""

    def __init__(self, root: Path, *, max_passages: int | None = None):
        self.root = root
        self.max_passages = max_passages
        self._block_wavs: dict[str, Path] = {}
        self._passages: dict[str, list[tuple[str, float]]] = {}
        self._load_durations()
        self._load_block_wavs()

    def _load_durations(self) -> None:
        durations_file = next(self.root.rglob('file-durations.txt'), None)
        if durations_file is None:
            raise AixMarsecError(f'{self.root}: no file-durations.txt found')
        for line in durations_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            name, rest = line.split('\t', 1)
            duration = float(rest.split()[0])
            passage = name[:-4] if name.endswith('.SIG') else name
            # The table has ONE entry per passage (B or G annotator variant);
            # offsets are keyed by the annotator-less passage code (stem[:5]).
            block = passage[:3]
            self._passages.setdefault(block, []).append((passage[:5], duration))

    def _load_block_wavs(self) -> None:
        for wav in self.root.rglob('*.wav'):
            self._block_wavs[wav.stem] = wav

    def units(self) -> list[IntonationUnit]:
        units: list[IntonationUnit] = []
        textgrid_root = next(self.root.rglob('TextGrid'), None)
        if textgrid_root is None or not textgrid_root.is_dir():
            raise AixMarsecError(f'{self.root}: no TextGrid directory')
        offsets_cache: dict[str, dict[str, float]] = {}
        seen = 0
        for tg_path in sorted(textgrid_root.glob('*/*.TextGrid')):
            passage_id = tg_path.stem
            passage_code = passage_id[:5]  # annotator-less: group+block+passage
            block_id = passage_id[:3]
            if block_id not in self._passages:
                continue
            wav = self._block_wavs.get(block_id)
            if wav is None:
                continue
            if block_id not in offsets_cache:
                offsets_cache[block_id] = refine_offsets(wav, self._passages[block_id])
            offsets = offsets_cache[block_id]
            if passage_code not in offsets:
                continue  # truncated block tail
            try:
                tiers = _parse_short_textgrid(tg_path.read_text(encoding='utf-8'))
            except textgrid.TextGridError as exc:
                raise AixMarsecError(f'{tg_path}: {exc}') from exc
            text_tier = next((t for t in tiers if t.name == 'Text'), None)
            if text_tier is None:
                raise AixMarsecError(f'{tg_path}: no Text tier')
            for unit in _units_from_text_tier(text_tier, offsets, passage_code, block_id):
                unit = IntonationUnit(
                    passage_id=passage_id,
                    block_id=unit.block_id,
                    annotator=passage_id[-1],
                    audio_path=wav,
                    start_s=unit.start_s,
                    end_s=unit.end_s,
                    expected_tone=unit.expected_tone,
                    transcript=unit.transcript,
                )
                units.append(unit)
            seen += 1
            if self.max_passages is not None and seen >= self.max_passages:
                return units
        if not units:
            raise AixMarsecError(f'{self.root}: no intonation units extracted')
        return units

