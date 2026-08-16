"""Exercise pack discovery, validation, and lookup.

Packs are read once at app startup and held in memory: an invalid pack is a
deploy-time error, not a per-request surprise. Validation is deliberately
loud — a malformed drill would otherwise reach a learner as a bogus verdict.

Two layers of checking:
1. JSON Schema (`content/schema/exercise.schema.json`) for structure.
2. Rules JSON Schema cannot express — exactly one focus phone on segmental
   types, `confusions` on that phone, a prosody block on intonation types,
   resolvable `pair_with`, and reference-audio paths that stay inside the pack.

Missing reference *audio files* are the one non-fatal case: the first pack
ships before the teacher recordings exist, so a declared-but-absent file is
recorded as `has_reference_audio=False` and logged, not raised.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from openschwa_engine.models.phone_set import PhoneSetError, normalize

log = logging.getLogger(__name__)

SEGMENTAL_TYPES = frozenset({"minimal_pair", "word", "sentence"})


class ContentError(Exception):
    """Raised for any pack that must not be served. Message names the file."""


@dataclass(frozen=True)
class PhoneSpec:
    index: int
    ph: str
    focus: bool = False
    confusions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProsodySpec:
    nuclear_syllable_index: int
    expected_tone: str


@dataclass(frozen=True)
class Exercise:
    id: str
    pack_id: str
    type: str
    title: str
    lang: str
    text: str
    ipa: str
    phones: tuple[PhoneSpec, ...]
    source_path: Path
    reference_audio_path: Path | None = None
    pair_with: str | None = None
    prosody: ProsodySpec | None = None
    learner_notes: str | None = None

    @property
    def phone_labels(self) -> tuple[str, ...]:
        return tuple(p.ph for p in self.phones)

    @property
    def focus_phone(self) -> PhoneSpec | None:
        return next((p for p in self.phones if p.focus), None)

    @property
    def has_reference_audio(self) -> bool:
        return self.reference_audio_path is not None and self.reference_audio_path.is_file()


@dataclass(frozen=True)
class Pack:
    id: str
    title: str
    root: Path
    language: str | None = None
    author: str | None = None
    audio_license: str | None = None


@dataclass(frozen=True)
class ContentLibrary:
    packs: tuple[Pack, ...] = ()
    exercises: dict[str, Exercise] = field(default_factory=dict)

    def get(self, exercise_id: str) -> Exercise | None:
        return self.exercises.get(exercise_id)

    def ordered(self) -> list[Exercise]:
        return sorted(self.exercises.values(), key=lambda e: e.id)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ContentError(f"{path}: invalid YAML — {exc}") from exc
    if not isinstance(data, dict):
        raise ContentError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def _resolve_reference_audio(pack_root: Path, declared: str, source: Path) -> Path:
    """Resolve a pack-relative audio path, refusing anything outside the pack.

    Reference paths come from YAML that a pack author may not have written, and
    the resolved path is served over HTTP — so traversal is rejected at load.
    """
    candidate = (pack_root / declared).resolve()
    if not candidate.is_relative_to(pack_root.resolve()):
        raise ContentError(f"{source}: reference_audio '{declared}' escapes the pack directory")
    return candidate


def _canonical(label: str, source: Path) -> str:
    """Normalise an authored phone, turning a mapping failure into a load error.

    Catching this at startup matters: an unmappable phone discovered mid-request
    would look to a learner like their recording was the problem.
    """
    try:
        return normalize(label)
    except PhoneSetError as exc:
        raise ContentError(f"{source}: {exc}") from exc


def _build_exercise(raw: dict[str, Any], pack: Pack, source: Path) -> Exercise:
    phones = tuple(
        PhoneSpec(
            index=i,
            ph=_canonical(p["ph"], source),
            focus=bool(p.get("focus", False)),
            confusions=tuple(_canonical(c, source) for c in p.get("confusions", ())),
        )
        for i, p in enumerate(raw["phones"])
    )

    ex_type = raw["type"]
    focus_phones = [p for p in phones if p.focus]
    if ex_type in SEGMENTAL_TYPES:
        if len(focus_phones) != 1:
            raise ContentError(
                f"{source}: {ex_type} exercises need exactly one focus phone, found "
                f"{len(focus_phones)}"
            )
        if not focus_phones[0].confusions:
            raise ContentError(
                f"{source}: focus phone /{focus_phones[0].ph}/ needs a non-empty confusion set — "
                "closed-set scoring has nothing to discriminate against without one"
            )
    elif ex_type == "intonation" and raw.get("prosody") is None:
        raise ContentError(f"{source}: intonation exercises need a prosody block")

    prosody_raw = raw.get("prosody")
    prosody = (
        ProsodySpec(
            nuclear_syllable_index=prosody_raw["nuclear_syllable_index"],
            expected_tone=prosody_raw["expected_tone"],
        )
        if prosody_raw
        else None
    )

    return Exercise(
        id=raw["id"],
        pack_id=pack.id,
        type=ex_type,
        title=raw["title"],
        lang=raw["lang"],
        text=raw["text"],
        ipa=raw["ipa"],
        phones=phones,
        source_path=source,
        reference_audio_path=_resolve_reference_audio(pack.root, raw["reference_audio"], source),
        pair_with=raw.get("pair_with"),
        prosody=prosody,
        learner_notes=raw.get("learner_notes"),
    )


def _load_pack(
    pack_dir: Path, validator: jsonschema.Draft202012Validator
) -> tuple[Pack, list[Exercise]]:
    pack_file = pack_dir / "pack.yaml"
    if not pack_file.is_file():
        raise ContentError(f"{pack_dir}: missing pack.yaml")
    raw_pack = _read_yaml(pack_file)
    for required in ("id", "title"):
        if required not in raw_pack:
            raise ContentError(f"{pack_file}: missing required key '{required}'")

    pack = Pack(
        id=raw_pack["id"],
        title=raw_pack["title"],
        root=pack_dir,
        language=raw_pack.get("language"),
        author=raw_pack.get("author"),
        audio_license=raw_pack.get("audio_license"),
    )

    exercises = []
    for source in sorted(pack_dir.glob("exercises/*.yaml")):
        raw = _read_yaml(source)
        errors = sorted(validator.iter_errors(raw), key=lambda e: list(e.absolute_path))
        if errors:
            first = errors[0]
            location = "/".join(str(p) for p in first.absolute_path) or "<root>"
            raise ContentError(f"{source}: schema violation at '{location}' — {first.message}")
        exercises.append(_build_exercise(raw, pack, source))

    if not exercises:
        raise ContentError(f"{pack_dir}: pack contains no exercises")
    return pack, exercises


def load_library(content_dir: Path, schema_path: Path) -> ContentLibrary:
    """Load every pack under `content_dir`. Raises ContentError on any bad pack."""
    if not content_dir.is_dir():
        raise ContentError(f"{content_dir}: content directory does not exist")
    if not schema_path.is_file():
        raise ContentError(f"{schema_path}: exercise schema not found")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    packs: list[Pack] = []
    exercises: dict[str, Exercise] = {}
    for pack_dir in sorted(p for p in content_dir.iterdir() if p.is_dir()):
        pack, pack_exercises = _load_pack(pack_dir, validator)
        packs.append(pack)
        for exercise in pack_exercises:
            if exercise.id in exercises:
                raise ContentError(
                    f"{exercise.source_path}: duplicate exercise id '{exercise.id}' "
                    f"(also in {exercises[exercise.id].source_path})"
                )
            exercises[exercise.id] = exercise

    # Cross-references and recordings can only be checked once every pack is in.
    for exercise in exercises.values():
        if exercise.pair_with and exercise.pair_with not in exercises:
            raise ContentError(
                f"{exercise.source_path}: pair_with '{exercise.pair_with}' matches no exercise"
            )
        if not exercise.has_reference_audio:
            log.warning(
                "%s: reference audio %s not recorded yet — playback disabled for this exercise",
                exercise.id,
                exercise.reference_audio_path,
            )

    return ContentLibrary(packs=tuple(packs), exercises=exercises)
