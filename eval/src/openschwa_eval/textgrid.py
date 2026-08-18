"""Minimal long-format Praat TextGrid parser (IntervalTiers only).

The engine pulls in parselmouth, but the harness must stay runnable on tiny
fixtures and fail loudly on anything it does not understand, so a 60-line
parser with explicit assumptions beats a black box here. Only the long
("ooTextFile") format is supported; L2-ARCTIC annotation files use it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    xmin: float
    xmax: float
    text: str

    def overlaps(self, other: "Interval") -> float:
        """Fraction of this interval covered by `other`."""
        overlap = min(self.xmax, other.xmax) - max(self.xmin, other.xmin)
        return max(0.0, overlap) / max(1e-9, self.xmax - self.xmin)


@dataclass(frozen=True)
class Tier:
    name: str
    intervals: tuple[Interval, ...]


class TextGridError(ValueError):
    """A TextGrid file cannot be parsed."""


def parse(text: str) -> list[Tier]:
    """Parse a long-format TextGrid, returning its IntervalTiers.

    PointTiers are rejected loudly: the harness never needs them, and
    silently skipping one would be silent loss of evidence.
    """
    lines = [line.strip() for line in text.splitlines()]
    if not lines or "ooTextFile" not in lines[0]:
        raise TextGridError("not a long-format TextGrid")
    if not any("tiers?" in line for line in lines):
        raise TextGridError("not a long-format TextGrid (missing 'tiers?' marker)")

    tiers: list[Tier] = []
    current_name: str | None = None
    current_class: str | None = None
    intervals: list[Interval] = []
    pending_xmin: float | None = None
    pending_xmax: float | None = None
    in_interval = False

    for line in lines:
        if line.startswith("item ["):
            _flush(tiers, current_name, current_class, intervals)
            current_name = current_class = None
            intervals = []
            in_interval = False
            pending_xmin = pending_xmax = None
        elif line.startswith("name ="):
            current_name = _quoted(line)
        elif line.startswith("class ="):
            current_class = _quoted(line)
        elif line.startswith("intervals ["):
            in_interval = True
            pending_xmin = pending_xmax = None
        elif in_interval and line.startswith("xmin ="):
            pending_xmin = float(_quoted(line))
        elif in_interval and line.startswith("xmax ="):
            pending_xmax = float(_quoted(line))
        elif in_interval and line.startswith("text ="):
            if pending_xmin is None or pending_xmax is None:
                raise TextGridError("interval text before its bounds")
            intervals.append(Interval(pending_xmin, pending_xmax, _quoted(line)))
            pending_xmin = pending_xmax = None

    _flush(tiers, current_name, current_class, intervals)
    return tiers


def _flush(tiers, name, tier_class, intervals) -> None:
    if name is None:
        return
    if tier_class != "IntervalTier":
        raise TextGridError(f"tier '{name}' is not an IntervalTier ({tier_class})")
    tiers.append(Tier(name=name, intervals=tuple(intervals)))


def _quoted(line: str) -> str:
    """Extract the quoted string value of a key = "value" line."""
    _, separator, rest = line.partition("=")
    if not separator:
        raise TextGridError(f"expected key = value line, got '{line}'")
    value = rest.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value
