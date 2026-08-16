"""Health and model-state contract (v1).

The UI polls health to know whether the engine is up *and* whether it can
actually analyse anything: a downloaded model with no ML runtime, or a runtime
with no weights, are both "engine is running but cannot judge" states that the
first-run UI has to distinguish.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelState = Literal["ready", "missing", "downloading"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelInfo(_Model):
    id: str
    state: ModelState
    repo_id: str
    revision: str = Field(description="Pinned upstream commit — never a branch name.")
    download_bytes: int
    license: str
    note: str
    runtime_available: bool = Field(
        description=(
            "False when the engine's `ml` extra is not installed; weights alone are not enough."
        )
    )


class ModelCatalog(_Model):
    schema_version: Literal["1.0"]
    models: list[ModelInfo] = []


class HealthResponse(_Model):
    status: Literal["ok"]
    engine_version: str
    schema_version: Literal["1.0"]
    #: The model /v1/analyze will use, from settings.alignment_model.
    alignment_model: str
    #: True when the whole pipeline can run; false means analyses return "retry".
    analysis_available: bool
    models: list[ModelInfo] = []
