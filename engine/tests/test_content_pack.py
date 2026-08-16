"""Every committed exercise pack must validate against the exercise schema.
The exactly-one-focus rule for segmental types is checked here too (it's not
expressible in JSON Schema; the content loader will enforce it at runtime)."""

import json
from pathlib import Path

import jsonschema
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO_ROOT / "content" / "schema" / "exercise.schema.json").read_text())
EXERCISE_FILES = sorted(REPO_ROOT.glob("content/packs/*/exercises/*.yaml"))


def test_exercise_schema_is_valid():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def test_sample_pack_exists():
    assert EXERCISE_FILES, "expected at least one committed exercise pack"


@pytest.mark.parametrize("path", EXERCISE_FILES, ids=lambda p: p.stem)
def test_exercise_validates(path):
    exercise = yaml.safe_load(path.read_text())
    jsonschema.validate(exercise, SCHEMA)
    if exercise["type"] in ("minimal_pair", "word", "sentence"):
        focus_count = sum(1 for p in exercise["phones"] if p.get("focus"))
        assert focus_count == 1, f"{path.name}: segmental exercises need exactly one focus phone"
    if exercise["type"] == "intonation":
        assert exercise.get("prosody"), f"{path.name}: intonation exercises need a prosody block"
