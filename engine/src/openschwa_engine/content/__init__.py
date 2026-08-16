"""Exercise pack loading + validation.

Packs are discovered under `settings.content_dir` and validated against
`content/schema/exercise.schema.json` at startup — invalid packs fail loud,
naming the file and the offending path. Rules the JSON Schema cannot express
(exactly one focus phone per segmental exercise, a prosody block on intonation
exercises, resolvable `pair_with`) are enforced by the loader.

Reference-audio F0/alignment precompute lands with M2 prosody comparison; M0
only needs to know whether a recording exists.
"""

from openschwa_engine.content.loader import (
    ContentError,
    ContentLibrary,
    Exercise,
    Pack,
    PhoneSpec,
    ProsodySpec,
    load_library,
)

__all__ = [
    "ContentError",
    "ContentLibrary",
    "Exercise",
    "Pack",
    "PhoneSpec",
    "ProsodySpec",
    "load_library",
]
