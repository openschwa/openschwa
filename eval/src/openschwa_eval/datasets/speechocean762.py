"""speechocean762 adapter (https://www.openslr.org/101/). M1.

Phone-level accuracy from five experts; L1-Mandarin speakers. Ground truth:
in resource/scores-detail.json each expert spells the realized phones, with
braces around mispronounced ones - majority vote (3/5) marks a phone as
substituted/deleted. The canonical sequence comes from resource/scores.json.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from openschwa_eval.arpabet import LabelMappingError, from_arpabet
from openschwa_eval.datasets.base import DatasetAdapter, PhoneToken, Utterance

EXPERT_THRESHOLD = 3  # of 5 experts


class AdapterError(ValueError):
    """A corpus file cannot be parsed into labeled tokens."""


class SpeechOcean762(DatasetAdapter):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.scores: dict[str, Any] = self._load_json(root / "resource" / "scores.json")
        self.detail: dict[str, Any] = self._load_json(root / "resource" / "scores-detail.json")
        self.wav_paths: dict[str, Path] = {}
        self.texts: dict[str, str] = {}
        self.splits: dict[str, str] = {}
        for split in ("train", "test"):
            self.wav_paths.update(self._read_scp(root / split / "wav.scp", split))
            self.texts.update(self._read_pairs(root / split / "text"))
            self.splits.update({utt: split for utt in self._read_pairs(root / split / "text")})

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise AdapterError(f"{path}: missing speechocean762 score file")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise AdapterError(f"{path}: expected a JSON object")
        return data

    def _read_scp(self, path: Path, split: str) -> dict[str, Path]:
        if not path.is_file():
            raise AdapterError(f"{path}: missing wav.scp for the {split} split")
        mapping: dict[str, Path] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 2:
                raise AdapterError(f"{path}: unparsable wav.scp line '{line}'")
            mapping[fields[0]] = self.root / fields[1]
        return mapping

    def _read_pairs(self, path: Path) -> dict[str, str]:
        if not path.is_file():
            raise AdapterError(f"{path}: missing text file")
        pairs: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise AdapterError(f"{path}: unparsable text line '{line}'")
            pairs[fields[0]] = fields[1]
        return pairs

    def utterances(self, target_phone: str) -> Iterator[Utterance]:
        for utterance_id, detail in sorted(self.detail.items()):
            canonical_words = self.scores.get(utterance_id, {}).get("words", [])
            tokens, has_target = self._tokens(canonical_words, detail, utterance_id, target_phone)
            if not has_target:
                continue
            audio = self.wav_paths.get(utterance_id)
            if audio is None or not audio.is_file():
                raise AdapterError(f"{utterance_id}: no wav file at {audio}")
            yield Utterance(
                utterance_id=f"so762-{utterance_id}",
                audio_path=audio,
                transcript=self.texts.get(utterance_id, ""),
                l1="mandarin",
                phones=tuple(tokens),
                split=self.splits.get(utterance_id, "train"),
                corpus="so762",
            )

    def _tokens(
        self,
        canonical_words: "list[dict[str, Any]]",
        detail: dict[str, Any],
        utterance_id: str,
        target_phone: str,
    ) -> "tuple[list[PhoneToken], bool]":
        tokens: list[PhoneToken] = []
        has_target = False
        expert_words = detail.get("words", [])
        for word_index, word in enumerate(canonical_words):
            phones = word.get("phones", [])
            expert_word = expert_words[word_index] if word_index < len(expert_words) else None
            expert_lists = expert_word.get("phones", []) if expert_word else []
            for position, label in enumerate(phones):
                try:
                    canonical = from_arpabet(label)
                except LabelMappingError as exc:
                    raise AdapterError(f"{utterance_id}: {exc}") from exc
                error_votes = sum(
                    1
                    for expert in expert_lists
                    if _expert_flags_error(expert, position, len(phones))
                )
                is_error = error_votes >= EXPERT_THRESHOLD
                tokens.append(
                    PhoneToken(
                        index=len(tokens),
                        phone=canonical,
                        label="substituted" if is_error else "correct",
                        substituted_with=None,
                    )
                )
                if canonical == target_phone:
                    has_target = True
        return tokens, has_target


def _expert_flags_error(expert_phones: str, position: int, expected: int) -> bool:
    """Whether this expert bracketed the phone at `position`."""
    tokens = expert_phones.replace("{", " ").replace("}", " ").split()
    if len(tokens) != expected:
        return False  # expert disagrees on structure; vote not counted
    raw = expert_phones.split()
    if position >= len(raw):
        return False
    return raw[position].strip("{}") != raw[position]
