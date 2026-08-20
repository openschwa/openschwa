"""Pack loading must fail loud and name the file.

A malformed pack that loads anyway becomes a wrong drill in front of a learner,
so every one of these cases is an exception at startup rather than a warning.
"""

from pathlib import Path

import pytest
import yaml

from openschwa_engine.config import Settings
from openschwa_engine.content import ContentError, load_library

SCHEMA_PATH = Settings().content_schema_path

VALID_EXERCISE = {
    "id": "en.seg.test.this",
    "type": "word",
    "title": "test",
    "lang": "en",
    "text": "this",
    "ipa": "ðɪs",
    "phones": [
        {"ph": "ð", "focus": True, "confusions": ["z", "d"]},
        {"ph": "ɪ"},
        {"ph": "s"},
    ],
    "reference_audio": "audio/this-ref.wav",
}


def write_pack(root: Path, *exercises: dict, pack: dict | None = None) -> Path:
    pack_dir = root / "testpack"
    (pack_dir / "exercises").mkdir(parents=True, exist_ok=True)
    (pack_dir / "pack.yaml").write_text(
        yaml.safe_dump(pack or {"id": "testpack", "title": "Test pack"})
    )
    for i, exercise in enumerate(exercises):
        (pack_dir / "exercises" / f"ex{i}.yaml").write_text(yaml.safe_dump(exercise))
    return root


def test_loads_a_valid_pack(tmp_path):
    library = load_library(write_pack(tmp_path, VALID_EXERCISE), SCHEMA_PATH)
    exercise = library.get("en.seg.test.this")
    assert exercise is not None
    assert exercise.pack_id == "testpack"
    assert exercise.phone_labels == ("ð", "ɪ", "s")
    assert exercise.focus_phone is not None and exercise.focus_phone.ph == "ð"


def test_missing_reference_recording_is_not_fatal(tmp_path):
    """The first pack ships before the teacher recordings exist; the API tells
    the UI to hide playback instead of the engine refusing to start."""
    library = load_library(write_pack(tmp_path, VALID_EXERCISE), SCHEMA_PATH)
    assert library.get("en.seg.test.this").has_reference_audio is False


def test_rejects_a_segmental_exercise_without_a_focus_phone(tmp_path):
    broken = {**VALID_EXERCISE, "phones": [{"ph": "ð"}, {"ph": "ɪ"}, {"ph": "s"}]}
    with pytest.raises(ContentError, match="exactly one focus phone"):
        load_library(write_pack(tmp_path, broken), SCHEMA_PATH)


def test_rejects_two_focus_phones(tmp_path):
    broken = {
        **VALID_EXERCISE,
        "phones": [
            {"ph": "ð", "focus": True, "confusions": ["z"]},
            {"ph": "ɪ"},
            {"ph": "s", "focus": True, "confusions": ["z"]},
        ],
    }
    with pytest.raises(ContentError, match="exactly one focus phone"):
        load_library(write_pack(tmp_path, broken), SCHEMA_PATH)


def test_rejects_a_focus_phone_with_no_confusion_set(tmp_path):
    """Closed-set scoring has nothing to discriminate against without one."""
    broken = {**VALID_EXERCISE, "phones": [{"ph": "ð", "focus": True}, {"ph": "ɪ"}, {"ph": "s"}]}
    with pytest.raises(ContentError, match="confusion set"):
        load_library(write_pack(tmp_path, broken), SCHEMA_PATH)


def test_rejects_a_phone_outside_the_canonical_inventory(tmp_path):
    broken = {
        **VALID_EXERCISE,
        "phones": [{"ph": "ʘ", "focus": True, "confusions": ["z"]}, {"ph": "ɪ"}],
    }
    with pytest.raises(ContentError, match="canonical inventory"):
        load_library(write_pack(tmp_path, broken), SCHEMA_PATH)


def test_normalizes_author_spellings(tmp_path):
    """ASCII `g` and `r` are indistinguishable from ɡ/ɹ in most editors."""
    relaxed = {
        **VALID_EXERCISE,
        "text": "grow",
        "ipa": "ɡɹoʊ",
        "phones": [{"ph": "g", "focus": True, "confusions": ["k"]}, {"ph": "r"}, {"ph": "oʊ"}],
    }
    library = load_library(write_pack(tmp_path, relaxed), SCHEMA_PATH)
    assert library.get("en.seg.test.this").phone_labels == ("ɡ", "ɹ", "oʊ")


def test_rejects_an_intonation_exercise_without_a_prosody_block(tmp_path):
    broken = {**VALID_EXERCISE, "type": "intonation"}
    with pytest.raises(ContentError, match="prosody block"):
        load_library(write_pack(tmp_path, broken), SCHEMA_PATH)


def test_rejects_an_unresolvable_pair_with(tmp_path):
    broken = {**VALID_EXERCISE, "pair_with": "en.seg.test.nonexistent"}
    with pytest.raises(ContentError, match="matches no exercise"):
        load_library(write_pack(tmp_path, broken), SCHEMA_PATH)


def test_rejects_duplicate_exercise_ids(tmp_path):
    with pytest.raises(ContentError, match="duplicate exercise id"):
        load_library(write_pack(tmp_path, VALID_EXERCISE, dict(VALID_EXERCISE)), SCHEMA_PATH)


def test_rejects_reference_audio_escaping_the_pack(tmp_path):
    """The resolved path is served over HTTP, so traversal is refused at load."""
    broken = {**VALID_EXERCISE, "reference_audio": "../../../etc/passwd"}
    with pytest.raises(ContentError, match="escapes the pack"):
        load_library(write_pack(tmp_path, broken), SCHEMA_PATH)


def test_reports_the_offending_file_and_path_on_a_schema_violation(tmp_path):
    broken = {**VALID_EXERCISE, "type": "not-a-real-type"}
    with pytest.raises(ContentError, match=r"ex0\.yaml.*schema violation at 'type'"):
        load_library(write_pack(tmp_path, broken), SCHEMA_PATH)


def test_rejects_a_pack_with_no_exercises(tmp_path):
    with pytest.raises(ContentError, match="no exercises"):
        load_library(write_pack(tmp_path), SCHEMA_PATH)


def test_rejects_invalid_yaml(tmp_path):
    write_pack(tmp_path, VALID_EXERCISE)
    (tmp_path / "testpack" / "exercises" / "ex0.yaml").write_text("this: [unclosed")
    with pytest.raises(ContentError, match="invalid YAML"):
        load_library(tmp_path, SCHEMA_PATH)


def test_the_committed_pack_loads(tmp_path):
    """The real packs must satisfy every rule above."""
    settings = Settings()
    library = load_library(settings.content_dir, settings.content_schema_path)
    assert library.get("en.seg.dh-z.this") is not None
    assert library.get("en.intonation.please.fall") is not None
    assert {p.id for p in library.packs} == {"en-dh-z", "en-intonation"}
