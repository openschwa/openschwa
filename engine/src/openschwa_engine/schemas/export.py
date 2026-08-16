"""Export the pydantic contract to the committed JSON Schema artifacts.

Run via `just schema` (or `uv run python -m openschwa_engine.schemas.export`).
CI verifies the committed files match (tests/test_schema_export.py), so the
pydantic models can never silently drift from what the UI was generated
against.

Adding a top-level response model means adding it to CONTRACTS and running
`just schema` — the UI's generated types follow automatically.
"""

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from openschwa_engine.schemas.analysis import AnalysisResult
from openschwa_engine.schemas.content import ExerciseCatalog, ExerciseDetail
from openschwa_engine.schemas.system import HealthResponse, ModelCatalog

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = REPO_ROOT / "schemas"

#: filename -> model. Filenames carry the contract version; v1 files are never
#: mutated after release (breaking changes create a v2 module and file).
CONTRACTS: dict[str, type[BaseModel]] = {
    "analysis_result.v1.schema.json": AnalysisResult,
    "exercise_catalog.v1.schema.json": ExerciseCatalog,
    "exercise_detail.v1.schema.json": ExerciseDetail,
    "health.v1.schema.json": HealthResponse,
    "model_catalog.v1.schema.json": ModelCatalog,
}

# Kept for the existing drift test and any external references to the primary
# contract artifact.
DEFAULT_OUT = DEFAULT_OUT_DIR / "analysis_result.v1.schema.json"


def build_schema(model: type[BaseModel], filename: str) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://openschwa.ru/schemas/{filename}",
        **model.model_json_schema(),
    }


def build_all() -> dict[Path, dict[str, Any]]:
    return {
        DEFAULT_OUT_DIR / filename: build_schema(model, filename)
        for filename, model in CONTRACTS.items()
    }


def main(out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, model in CONTRACTS.items():
        path = out_dir / filename
        path.write_text(json.dumps(build_schema(model, filename), indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT_DIR)
