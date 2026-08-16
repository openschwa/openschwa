"""Model manifest, cache state, and the first-run download stream."""

import json
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from openschwa_engine.api.deps import get_registry, get_settings
from openschwa_engine.config import Settings
from openschwa_engine.models.registry import ModelError, ModelRegistry, ModelStatus
from openschwa_engine.schemas.analysis import SCHEMA_VERSION
from openschwa_engine.schemas.system import ModelCatalog, ModelInfo

router = APIRouter()

Registry = Annotated[ModelRegistry, Depends(get_registry)]
Config = Annotated[Settings, Depends(get_settings)]


def model_info(status: ModelStatus) -> ModelInfo:
    return ModelInfo(
        id=status.spec.id,
        state=status.state,  # type: ignore[arg-type]  # registry emits only contract states
        repo_id=status.spec.repo_id,
        revision=status.spec.revision,
        download_bytes=status.spec.download_bytes,
        license=status.spec.license,
        note=status.spec.note,
        runtime_available=status.runtime_available,
    )


@router.get("/models")
def list_models(registry: Registry) -> ModelCatalog:
    return ModelCatalog(
        schema_version=SCHEMA_VERSION,  # type: ignore[arg-type]
        models=[model_info(s) for s in registry.catalog()],
    )


@router.post("/models/pull")
def pull_model(
    registry: Registry, settings: Config, model_id: str | None = None
) -> StreamingResponse:
    """Download weights, streaming NDJSON progress so the first run shows a bar
    rather than an unexplained multi-minute stall. Resumable — a client that
    disconnects can re-POST and continue."""
    try:
        spec = registry.spec(model_id or settings.alignment_model)
    except ModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        events = registry.pull(spec)
    except ModelError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    def stream() -> Iterator[str]:
        for event in events:
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
