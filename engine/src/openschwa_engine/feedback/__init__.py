"""Feedback composition: evidence + thresholds -> the gated `feedback[]` list.

The only module allowed to put a judgement in front of a learner, and therefore
the only place the confidence gate has to be enforced.
"""

from openschwa_engine.feedback.composer import compose, retry_item

__all__ = ["compose", "retry_item"]
