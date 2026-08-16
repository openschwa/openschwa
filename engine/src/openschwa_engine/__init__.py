"""OpenSchwa analysis engine.

A local FastAPI service that scores scripted pronunciation drills and returns
judgments, annotations, and confidence-gated feedback. It never renders — the
UI owns all audio rendering; the engine returns intervals, scores, and labels.
See docs/architecture.md at the repo root.
"""

__version__ = "0.1.0"
