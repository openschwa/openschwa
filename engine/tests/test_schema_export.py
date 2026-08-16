"""Schema-drift gate: every committed JSON Schema must match the pydantic
models exactly. Regenerate with `just schema` after any contract change."""

import json

import pytest

from openschwa_engine.schemas.export import CONTRACTS, build_all


@pytest.mark.parametrize("path", sorted(build_all()), ids=lambda p: p.name)
def test_committed_schema_matches_models(path):
    assert path.exists(), f"missing {path} — run `just schema`"
    committed = json.loads(path.read_text())
    assert committed == build_all()[path], "schema drift — run `just schema` and commit the result"


def test_every_contract_is_versioned_in_its_filename():
    """v1 files are never mutated after release; a breaking change means a new
    module and a new file (docs/architecture.md §3)."""
    assert all(".v1." in name for name in CONTRACTS)
