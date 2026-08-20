"""L2-ARCTIC adapter (https://psi.engr.tamu.edu/l2-arctic-corpus/). M1.

Six L1 backgrounds; annotation TextGrids tag substitutions/deletions/additions
on an "IPA" tier as "target,realized,code" triples (code: s/d/a). The phone
tier of the annotation file is the intended canonical sequence; tags correct
the labels. Only utterances with annotation files are usable - they alone
carry ground truth.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

from openschwa_eval import textgrid
from openschwa_eval.arpabet import LabelMappingError, from_l2arctic_label
from openschwa_eval.datasets.base import DatasetAdapter, PhoneToken, Utterance

log = logging.getLogger(__name__)

#: Speaker -> L1, from the original release README (table verbatim).
SPEAKER_L1 = {
    "ABA": "arabic",
    "SKA": "arabic",
    "YBAA": "arabic",
    "ZHAA": "arabic",
    "BWC": "mandarin",
    "LXC": "mandarin",
    "NCC": "mandarin",
    "TXHC": "mandarin",
    "ASI": "hindi",
    "RRBI": "hindi",
    "SVBI": "hindi",
    "TNI": "hindi",
    "HJK": "korean",
    "HKK": "korean",
    "YDCK": "korean",
    "YKWK": "korean",
    "EBVS": "spanish",
    "ERMS": "spanish",
    "MBMPS": "spanish",
    "NJS": "spanish",
    "HQTV": "vietnamese",
    "PNV": "vietnamese",
    "THV": "vietnamese",
    "TLV": "vietnamese",
}


class AdapterError(ValueError):
    """A corpus file cannot be parsed into labeled tokens."""


class L2Arctic(DatasetAdapter):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        #: Labels that could not be mapped and were skipped (warned, not silent).
        self.skipped_labels: set[tuple[str, str]] = set()

    def utterances(self, target_phone: str) -> Iterator[Utterance]:
        for speaker_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            speaker = speaker_dir.name
            l1 = SPEAKER_L1.get(speaker)
            if l1 is None:
                raise AdapterError(f"unknown L2-ARCTIC speaker directory '{speaker}'")
            annotation_dir = speaker_dir / "annotation"
            if not annotation_dir.is_dir():
                continue  # no manual annotations -> no ground truth
            for tg_file in sorted(annotation_dir.glob("*.TextGrid")):
                utterance = self._parse(tg_file, speaker, l1)
                if any(t.phone == target_phone for t in utterance.phones):
                    yield utterance

    def _parse(self, tg_file: Path, speaker: str, l1: str) -> Utterance:
        """The annotation's 'phones' tier is the truth: realized phones for
        correct tokens, and 'target,realized,code' triples for errors. The
        'IPA' tier duplicates the tags in IPA spelling; parsing one tier is
        enough and avoids double-counting."""
        stem = tg_file.stem
        tiers = textgrid.parse(tg_file.read_text(encoding="utf-8"))
        phones_tier = next((tier for tier in tiers if tier.name == "phones"), None)
        if phones_tier is None:
            raise AdapterError(f"{tg_file}: expected a 'phones' tier")

        tokens: list[PhoneToken] = []
        for interval in phones_tier.intervals:
            text = interval.text.strip()
            if not text or text.lower() in ("sil", "sp", "spn", "err"):
                continue
            if text.count(",") == 2:
                target, realized, raw_code = (part.strip() for part in text.split(","))
                code = self._code(raw_code)
                if code is None:
                    # A malformed tag (empty code, stray letter): the token
                    # keeps its 'correct' label and the loss is recorded, not
                    # silent. One bad tag must not kill a whole corpus run.
                    self.skipped_labels.add((str(tg_file), text))
                    log.warning("%s: skipping unparsable annotation tag '%s'", tg_file, text)
                    continue
                if code == "a" or target.lower() in ("sil", "sp"):
                    continue  # an inserted phone: not in the intended sequence
                canonical = self._canonical(target, tg_file)
                if code == "s":
                    label, substituted_with = "substituted", self._realized(realized, tg_file)
                else:
                    label, substituted_with = "deleted", None
            elif "," in text:
                raise AdapterError(f"{tg_file}: unparsable phone annotation '{text}'")
            else:
                try:
                    canonical = self._canonical(text, tg_file)
                except LabelMappingError as exc:
                    # An annotation typo (e.g. 'ER)') must not drop the whole
                    # corpus - but it is logged and counted, never silent.
                    self.skipped_labels.add((str(tg_file), text))
                    log.warning("%s: skipping unmappable label '%s' (%s)", tg_file, text, exc)
                    continue
                label, substituted_with = "correct", None
            tokens.append(
                PhoneToken(
                    index=len(tokens),
                    phone=canonical,
                    start_s=round(interval.xmin, 4),
                    end_s=round(interval.xmax, 4),
                    label=label,
                    substituted_with=substituted_with,
                )
            )

        transcript_file = tg_file.parent.parent / "transcript" / f"{stem}.txt"
        transcript = (
            transcript_file.read_text(encoding="utf-8").strip()
            if transcript_file.is_file()
            else stem
        )
        audio = tg_file.parent.parent / "wav" / f"{stem}.wav"
        if not audio.is_file():
            raise AdapterError(f"{tg_file}: no audio at {audio}")
        return Utterance(
            utterance_id=f"l2arctic-{speaker}-{stem}",
            audio_path=audio,
            transcript=transcript,
            l1=l1,
            phones=tuple(tokens),
            corpus="l2arctic",
            speaker=speaker,
        )

    @staticmethod
    def _code(raw: str) -> str | None:
        """Canonicalize an annotation code; None for unparsable junk.

        The corpus is hand-annotated and codes drift: s/d/a dominate, with
        stray uppercase (S, D), punctuation (d.), and typos (sd, as). The
        known variants map to the three real codes; anything else is refused.
        """
        cleaned = raw.strip().lower()
        if cleaned in ("s", "sd"):
            return "s"
        if cleaned in ("d", "d."):
            return "d"
        if cleaned in ("a", "as"):
            return "a"
        return None

    def _canonical(self, label: str, source: Path) -> str:
        canonical = from_l2arctic_label(label)
        if canonical is None:
            raise AdapterError(f"{source}: tag target '{label}' is not a phone")
        return canonical

    def _realized(self, label: str, source: Path) -> str | None:
        try:
            return from_l2arctic_label(label)
        except LabelMappingError:
            return None  # e.g. 'err' - a substitution, but into what we cannot say
