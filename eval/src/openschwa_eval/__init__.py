"""OpenSchwa offline evaluation harness.

Imports the engine as a library - no HTTP in the loop (eval/README.md).
A feedback type ships only after this harness proves it meets the bar.
"""

from openschwa_eval import arpabet, datasets, harness, textgrid

__all__ = ["arpabet", "datasets", "harness", "textgrid"]
