"""Liveness plus enough state for the UI to know whether analysis can run.

Also the discovery target for the Tauri sidecar (docs/architecture.md §4).
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from openschwa_engine import __version__
from openschwa_engine.api.deps import get_registry, get_settings
from openschwa_engine.api.models import model_info
from openschwa_engine.config import Settings
from openschwa_engine.models.registry import ModelRegistry
from openschwa_engine.schemas.analysis import SCHEMA_VERSION
from openschwa_engine.schemas.system import HealthResponse

router = APIRouter()


@router.get("/health")
def health(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    statuses = registry.catalog()
    active = next((s for s in statuses if s.spec.id == settings.alignment_model), None)
    return HealthResponse(
        status="ok",
        engine_version=__version__,
        schema_version=SCHEMA_VERSION,  # type: ignore[arg-type]
        alignment_model=settings.alignment_model,
        analysis_available=bool(active and active.state == "ready" and active.runtime_available),
        models=[model_info(s) for s in statuses],
    )
