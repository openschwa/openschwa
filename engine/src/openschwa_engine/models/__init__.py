"""Model manifest, download/cache, and phone-set mapping.

`phone_set` is the canonical IPA inventory and the per-model mapping tables;
`registry` owns weights on disk. Loading the network itself lives in
`alignment.acoustic` so that torch stays out of every other import path.
"""

from openschwa_engine.models.phone_set import (
    CANONICAL_EN,
    PhoneMap,
    PhoneSetError,
    normalize,
)
from openschwa_engine.models.registry import (
    MANIFEST,
    ModelError,
    ModelRegistry,
    ModelSpec,
    ModelStatus,
    ml_runtime_available,
)

__all__ = [
    "CANONICAL_EN",
    "MANIFEST",
    "ModelError",
    "ModelRegistry",
    "ModelSpec",
    "ModelStatus",
    "PhoneMap",
    "PhoneSetError",
    "ml_runtime_available",
    "normalize",
]
