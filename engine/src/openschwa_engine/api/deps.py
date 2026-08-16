"""Shared request dependencies.

Everything expensive — the content library, the model registry — is built once
in `create_app` and hung on `app.state`. Routers reach it through these
accessors rather than importing module-level singletons, so tests can build an
app over a temporary content directory.
"""

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from openschwa_engine.config import Settings
    from openschwa_engine.content import ContentLibrary
    from openschwa_engine.models.registry import ModelRegistry


def get_settings(request: Request) -> "Settings":
    settings: Settings = request.app.state.settings
    return settings


def get_library(request: Request) -> "ContentLibrary":
    library: ContentLibrary = request.app.state.library
    return library


def get_registry(request: Request) -> "ModelRegistry":
    registry: ModelRegistry = request.app.state.registry
    return registry
